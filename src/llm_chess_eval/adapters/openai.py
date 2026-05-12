"""OpenAI adapter — function calling forced via tool_choice.

Requires `openai>=1.0` and OPENAI_API_KEY in env.
Tested against gpt-5, gpt-5-mini. Should also work for gpt-4o variants.
"""
from __future__ import annotations

import json
import os

from ._shared import (
    SUBMIT_MOVE_DESCRIPTION,
    SUBMIT_MOVE_PARAMETERS,
    SYSTEM_PROMPT,
    CallOutcome,
    Timer,
    build_user_message,
    parse_tool_input,
)


def _api_key() -> str:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Export it before running an OpenAI eval."
        )
    return key


SUBMIT_MOVE_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_move",
        "description": SUBMIT_MOVE_DESCRIPTION,
        "parameters": SUBMIT_MOVE_PARAMETERS,
        # Strict mode requires `additionalProperties: false` and all keys in `required`.
        # Our schema satisfies that, but we leave strict=False to avoid breakage if a
        # provider adds new optional fields.
    },
}


class OpenAIAdapter:
    def __init__(
        self,
        model: str,
        # 65536 chosen empirically: GPT-5 at medium reasoning hits `finish_reason="length"`
        # at 16000 on harder chess positions with retry context. Reasoning chains
        # for an LLM stuck on a position can balloon to 30k+ tokens. 65536 gives
        # generous headroom and is well within GPT-5's 128k output limit. Calls
        # that don't need much reasoning consume fewer tokens regardless; this is
        # a ceiling, not a target.
        max_tokens: int = 65536,
        augment_legal_moves: bool = False,
        reasoning_effort: str | None = None,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                "openai package is not installed. Run: pip install openai>=1.0"
            ) from e

        self.model = model
        self.max_tokens = max_tokens
        self.augment_legal_moves = augment_legal_moves
        # We do NOT default reasoning_effort to "low" — that would handicap the model
        # and measure a degraded version of it. Users who want a cheaper/faster eval
        # can pass reasoning_effort="low" explicitly. By default we let the provider's
        # default reasoning level run (currently "medium" for GPT-5).
        self.reasoning_effort = reasoning_effort
        self._client = OpenAI(api_key=_api_key())

    def propose_move(
        self,
        fen: str,
        prior_failed: list[str] | None = None,
        augment_legal_moves: bool | None = None,
        reasoning_effort_override: str | None = None,
    ) -> CallOutcome:
        """Make a single API call.

        reasoning_effort_override: if set (e.g. "medium", "low", "minimal"),
        overrides the adapter-level default for THIS call only. Used by play_game
        to step reasoning effort down after a `finish_reason='length'` failure.
        """
        use_aug = self.augment_legal_moves if augment_legal_moves is None else augment_legal_moves
        user_text = build_user_message(fen, prior_failed=prior_failed, augment_legal_moves=use_aug)

        raw_input = None
        error = None
        in_tok = out_tok = 0

        with Timer() as t:
            try:
                kwargs = dict(
                    model=self.model,
                    max_completion_tokens=self.max_tokens,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_text},
                    ],
                    tools=[SUBMIT_MOVE_TOOL],
                    tool_choice={"type": "function", "function": {"name": "submit_move"}},
                )
                # Per-call override takes precedence over adapter default.
                effort = reasoning_effort_override if reasoning_effort_override is not None else self.reasoning_effort
                if effort is not None:
                    kwargs["reasoning_effort"] = effort
                resp = self._client.chat.completions.create(**kwargs)
                usage = resp.usage
                if usage is not None:
                    in_tok = getattr(usage, "prompt_tokens", 0) or 0
                    out_tok = getattr(usage, "completion_tokens", 0) or 0

                choice = resp.choices[0] if resp.choices else None
                tool_calls = getattr(choice.message, "tool_calls", None) if choice else None
                if not tool_calls:
                    # Capture finish_reason + any content for diagnostics
                    fr = getattr(choice, "finish_reason", None) if choice else None
                    content = getattr(choice.message, "content", None) if choice else None
                    snippet = (content or "")[:120]
                    error = f"Model did not call submit_move (finish_reason={fr!r}, content_preview={snippet!r})"
                else:
                    first = tool_calls[0]
                    args_str = first.function.arguments
                    try:
                        raw_input = json.loads(args_str)
                    except json.JSONDecodeError:
                        try:
                            from json import JSONDecoder
                            raw_input, _ = JSONDecoder().raw_decode(args_str.lstrip())
                        except json.JSONDecodeError as e:
                            error = f"Tool arguments are not valid JSON: {e}"
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
        )
