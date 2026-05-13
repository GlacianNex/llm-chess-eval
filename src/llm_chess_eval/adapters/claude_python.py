"""Claude adapter with a Python-execution tool exposed alongside submit_move.

PROTOTYPE — tests whether code-execution access closes the spatial-reasoning gap.

Differences from claude.py:
  - Exposes a second tool, `python_exec`, that runs a Python snippet in a
    subprocess (30s timeout) and returns stdout/stderr/exit code.
  - Tool description names that `python-chess` is importable. We tell the
    model what's installed (same as any real API would), but we deliberately
    do NOT tell it WHEN to use the tool — the experiment is whether it
    chooses to externalize spatial verification on its own.
  - Multi-turn loop: keep calling the model until it emits a `submit_move`
    tool call (or hits MAX_LOOP_ITERATIONS without doing so).
  - tool_choice is "any" (not pinned to submit_move) so the model can pick
    python_exec or submit_move freely. We rely on the system prompt to
    require submit_move as the terminal action.

If this variant scores meaningfully higher than the base claude adapter on
the same legality / CR / PS evals, the gap is "available capability not
invoked" (deficit (b) in our framing) rather than structural.
"""
from __future__ import annotations

import subprocess
import sys
from typing import Any

from anthropic import Anthropic

from ..config import anthropic_api_key
from ._shared import (
    SUBMIT_MOVE_DESCRIPTION,
    SUBMIT_MOVE_PARAMETERS,
    SYSTEM_PROMPT,
    CallOutcome,
    Timer,
    build_user_message,
    parse_tool_input,
)

SUBMIT_MOVE_TOOL = {
    "name": "submit_move",
    "description": SUBMIT_MOVE_DESCRIPTION,
    "input_schema": SUBMIT_MOVE_PARAMETERS,
}

PYTHON_EXEC_TOOL = {
    "name": "python_exec",
    "description": (
        "Execute a snippet of Python 3 code in a sandboxed subprocess. "
        "Returns stdout, stderr, and exit code as a JSON-like dict. "
        "Each call is INDEPENDENT — no variables, imports, or state persist "
        "between calls. Use print() to surface results. "
        "The environment has `python-chess` available (`import chess`). "
        "Each call has a 30s wall-clock timeout."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "Python source code to execute. Use print() for outputs.",
            }
        },
        "required": ["code"],
    },
}

# Cap the inner loop so a runaway model can't burn unbounded calls per move.
MAX_LOOP_ITERATIONS = 10
PYTHON_EXEC_TIMEOUT_S = 30.0
STDOUT_TRUNCATE = 4000
STDERR_TRUNCATE = 1000

# Addendum to the shared SYSTEM_PROMPT for the pytool variant.
PYTOOL_SYSTEM_ADDENDUM = (
    "\n\nYou also have access to a `python_exec` tool. You may use it before "
    "submitting your move if you want to compute or verify anything. It is "
    "optional — use it when you find it helpful. You MUST end the turn with a "
    "`submit_move` call."
)


def _run_python_subprocess(code: str) -> dict[str, Any]:
    """Run `code` in a fresh subprocess. Capture stdout/stderr, enforce timeout."""
    try:
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=PYTHON_EXEC_TIMEOUT_S,
        )
        return {
            "stdout": proc.stdout[-STDOUT_TRUNCATE:],
            "stderr": proc.stderr[-STDERR_TRUNCATE:],
            "exit_code": proc.returncode,
        }
    except subprocess.TimeoutExpired as e:
        return {
            "stdout": (e.stdout or b"").decode(errors="replace")[-STDOUT_TRUNCATE:],
            "stderr": f"[timed out after {PYTHON_EXEC_TIMEOUT_S}s]",
            "exit_code": -1,
        }
    except Exception as e:  # noqa: BLE001
        return {"stdout": "", "stderr": f"sandbox error: {type(e).__name__}: {e}", "exit_code": -1}


class ClaudePythonAdapter:
    def __init__(
        self,
        model: str,
        max_tokens: int = 4096,
        augment_legal_moves: bool = False,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.augment_legal_moves = augment_legal_moves
        self._client = Anthropic(api_key=anthropic_api_key())

    def propose_move(
        self,
        fen: str,
        prior_failed: list[str] | None = None,
        augment_legal_moves: bool | None = None,
        reasoning_effort_override: str | None = None,
    ) -> CallOutcome:
        use_aug = self.augment_legal_moves if augment_legal_moves is None else augment_legal_moves
        user_text = build_user_message(fen, prior_failed=prior_failed, augment_legal_moves=use_aug)

        messages: list[dict[str, Any]] = [{"role": "user", "content": user_text}]

        raw_input: dict | None = None
        error: str | None = None
        in_tok = out_tok = cache_r = cache_c = 0
        loop_iter = 0

        with Timer() as t:
            try:
                while loop_iter < MAX_LOOP_ITERATIONS:
                    loop_iter += 1
                    resp = self._client.messages.create(
                        model=self.model,
                        max_tokens=self.max_tokens,
                        system=[
                            {
                                "type": "text",
                                "text": SYSTEM_PROMPT + PYTOOL_SYSTEM_ADDENDUM,
                                "cache_control": {"type": "ephemeral"},
                            }
                        ],
                        tools=[PYTHON_EXEC_TOOL, SUBMIT_MOVE_TOOL],
                        tool_choice={"type": "any"},
                        messages=messages,
                    )
                    usage = resp.usage
                    in_tok += getattr(usage, "input_tokens", 0) or 0
                    out_tok += getattr(usage, "output_tokens", 0) or 0
                    cache_r += getattr(usage, "cache_read_input_tokens", 0) or 0
                    cache_c += getattr(usage, "cache_creation_input_tokens", 0) or 0

                    tool_blocks = [b for b in resp.content if getattr(b, "type", None) == "tool_use"]
                    submit_block = next((b for b in tool_blocks if b.name == "submit_move"), None)

                    if submit_block is not None:
                        raw_input = dict(submit_block.input)
                        break

                    py_blocks = [b for b in tool_blocks if b.name == "python_exec"]
                    if not py_blocks:
                        # Model neither submitted nor invoked the python tool — give up.
                        error = (
                            f"Model emitted no tool calls on iteration {loop_iter} "
                            f"(stop_reason={resp.stop_reason})"
                        )
                        break

                    # Echo the assistant turn back and feed each python_exec result.
                    messages.append({"role": "assistant", "content": resp.content})
                    tool_results = []
                    for b in py_blocks:
                        code = b.input.get("code", "")
                        result = _run_python_subprocess(code)
                        tool_results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": b.id,
                                "content": (
                                    f"stdout:\n{result['stdout']}\n\n"
                                    f"stderr:\n{result['stderr']}\n\n"
                                    f"exit_code: {result['exit_code']}"
                                ),
                            }
                        )
                    messages.append({"role": "user", "content": tool_results})
                else:
                    error = f"Hit MAX_LOOP_ITERATIONS={MAX_LOOP_ITERATIONS} without submit_move"
            except Exception as e:  # noqa: BLE001
                error = f"{type(e).__name__}: {e}"

        parsed = None
        if raw_input is not None and error is None:
            parsed, parse_err = parse_tool_input(raw_input)
            if parse_err:
                error = parse_err

        return CallOutcome(
            response=parsed,
            raw_tool_input=raw_input,
            latency_ms=t.ms,
            error=error,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cache_read_tokens=cache_r,
            cache_creation_tokens=cache_c,
        )
