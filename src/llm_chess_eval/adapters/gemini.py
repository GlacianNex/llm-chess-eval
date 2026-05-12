"""Gemini adapter — function calling via google-genai SDK.

Requires `google-genai>=0.3` and GOOGLE_API_KEY (or GEMINI_API_KEY) in env.
Tested against gemini-3.1-pro, gemini-2.5-pro, gemini-2.5-flash.

Notes on Gemini function calling:
  - The SDK uses `tools=[Tool(function_declarations=[...])]`.
  - We force the call by setting `tool_config.function_calling_config.mode = "ANY"`
    with `allowed_function_names=["submit_move"]`, equivalent to other providers'
    "tool_choice: this specific tool".
  - Gemini's function-declaration schema is OpenAPI 3.0 strict: it does NOT accept
    JSON-Schema-style multi-type (e.g. `type: ["string", "null"]`) nor `null`
    values inside `enum`. We translate the shared schema into Gemini's expected
    shape (single type + nullable: True, enum without null entries) before sending.
"""
from __future__ import annotations

import copy
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
    key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError(
            "Neither GOOGLE_API_KEY nor GEMINI_API_KEY is set. "
            "Export one before running a Gemini eval."
        )
    return key


def _to_gemini_schema(schema: dict) -> dict:
    """Translate JSON Schema into Gemini's OpenAPI-3.0 format.

    Gemini rejects:
      - `type: ["string", "null"]` multi-type → convert to single type + nullable: True
      - `enum: ["a", "b", null]` with null entries → drop null
    """
    out = copy.deepcopy(schema)

    def fix(node: object) -> None:
        if isinstance(node, dict):
            t = node.get("type")
            if isinstance(t, list):
                non_null = [x for x in t if x not in (None, "null")]
                node["type"] = non_null[0] if non_null else "string"
                if "null" in t or None in t:
                    node["nullable"] = True
            if "enum" in node and isinstance(node["enum"], list):
                cleaned = [v for v in node["enum"] if v is not None]
                if len(cleaned) != len(node["enum"]):
                    node["enum"] = cleaned
                    node["nullable"] = True
            for v in node.values():
                fix(v)
        elif isinstance(node, list):
            for item in node:
                fix(item)

    fix(out)
    return out


class GeminiAdapter:
    def __init__(
        self,
        model: str,
        # 65536: Gemini Pro reasoning chains can balloon past 16000 on hard chess
        # positions with retry context. Same fix as the OpenAI adapter for the
        # same root cause (finish_reason hitting length cap, model fails to emit
        # tool call). 65536 leaves generous headroom; calls finish at less when
        # the reasoning is bounded.
        max_tokens: int = 65536,
        augment_legal_moves: bool = False,
    ) -> None:
        try:
            from google import genai
            from google.genai import types
        except ImportError as e:
            raise ImportError(
                "google-genai package is not installed. Run: pip install google-genai"
            ) from e

        self.model = model
        self.max_tokens = max_tokens
        self.augment_legal_moves = augment_legal_moves
        self._client = genai.Client(api_key=_api_key())
        self._types = types

    def propose_move(
        self,
        fen: str,
        prior_failed: list[str] | None = None,
        augment_legal_moves: bool | None = None,
        reasoning_effort_override: str | None = None,  # accepted for protocol parity; Gemini uses thinking_config.thinking_budget instead — not currently wired
    ) -> CallOutcome:
        use_aug = self.augment_legal_moves if augment_legal_moves is None else augment_legal_moves
        user_text = build_user_message(fen, prior_failed=prior_failed, augment_legal_moves=use_aug)
        types = self._types

        gemini_schema = _to_gemini_schema(SUBMIT_MOVE_PARAMETERS)
        function_decl = types.FunctionDeclaration(
            name="submit_move",
            description=SUBMIT_MOVE_DESCRIPTION,
            parameters=gemini_schema,
        )
        tool = types.Tool(function_declarations=[function_decl])
        tool_config = types.ToolConfig(
            function_calling_config=types.FunctionCallingConfig(
                mode="ANY",
                allowed_function_names=["submit_move"],
            )
        )
        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            tools=[tool],
            tool_config=tool_config,
            max_output_tokens=self.max_tokens,
        )

        raw_input = None
        error = None
        in_tok = out_tok = 0

        with Timer() as t:
            try:
                resp = self._client.models.generate_content(
                    model=self.model,
                    contents=user_text,
                    config=config,
                )
                # Token usage
                usage_meta = getattr(resp, "usage_metadata", None)
                if usage_meta is not None:
                    in_tok = getattr(usage_meta, "prompt_token_count", 0) or 0
                    out_tok = getattr(usage_meta, "candidates_token_count", 0) or 0

                # Find the function call in the candidate parts
                fc = None
                for cand in getattr(resp, "candidates", []) or []:
                    parts = getattr(cand.content, "parts", []) or []
                    for part in parts:
                        if getattr(part, "function_call", None) is not None:
                            fc = part.function_call
                            break
                    if fc is not None:
                        break

                if fc is None:
                    error = "Model did not call submit_move"
                else:
                    raw_input = dict(fc.args) if fc.args else {}
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
