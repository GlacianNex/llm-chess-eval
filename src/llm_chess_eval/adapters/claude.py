"""Claude adapter — Anthropic SDK with tool-use forced output and prompt caching."""
from __future__ import annotations

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


class ClaudeAdapter:
    def __init__(
        self,
        model: str,
        max_tokens: int = 2048,
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
    ) -> CallOutcome:
        use_aug = self.augment_legal_moves if augment_legal_moves is None else augment_legal_moves
        user_text = build_user_message(fen, prior_failed=prior_failed, augment_legal_moves=use_aug)

        raw_input = None
        error = None
        in_tok = out_tok = cache_r = cache_c = 0

        with Timer() as t:
            try:
                resp = self._client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    system=[
                        {
                            "type": "text",
                            "text": SYSTEM_PROMPT,
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                    tools=[SUBMIT_MOVE_TOOL],
                    tool_choice={"type": "tool", "name": "submit_move"},
                    messages=[{"role": "user", "content": user_text}],
                )
                usage = resp.usage
                in_tok = getattr(usage, "input_tokens", 0) or 0
                out_tok = getattr(usage, "output_tokens", 0) or 0
                cache_r = getattr(usage, "cache_read_input_tokens", 0) or 0
                cache_c = getattr(usage, "cache_creation_input_tokens", 0) or 0

                tool_block = next(
                    (b for b in resp.content if getattr(b, "type", None) == "tool_use"),
                    None,
                )
                if tool_block is None:
                    error = "Model did not call submit_move"
                else:
                    raw_input = dict(tool_block.input)
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
