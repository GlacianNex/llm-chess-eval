"""Shared prompt, schema, and result-parsing logic across model adapters.

The Claude/OpenAI/Gemini/Llama adapters all use the same JSON schema for the
forced tool call. Each provider wraps the schema differently (input_schema vs
parameters vs function_declarations), but the schema content is identical.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import chess
from pydantic import ValidationError

from ..types import MoveResponse

SYSTEM_PROMPT = """You are a chess analyst. You will be given a chess position in FEN and as an
ASCII board. For each candidate move you consider, you must declare rule-based facts
about that move. These are NOT judgment calls — they are observable from the rules of
chess (what the move does to the board), and you will be graded on whether your claims
are factually correct.

You must:
1. Briefly summarize the position (1-2 sentences).
2. List 2-4 candidate moves you considered. For each candidate fill in:
   - `san`: the move in standard SAN notation (e.g. "Nf3", "exd5", "O-O", "Qxh7+", "e8=Q+")
   - `rationale`: 1-2 sentences on why you considered it
   - `claims`: rule-based facts about what THIS move does (see fields below)
3. Choose exactly one move from your candidates as `chosen_move`.

CRITICAL RULES:
- `chosen_move` MUST be one of the SAN strings you listed in `candidates`.
- Only legal moves are valid.
- Your `claims` must reflect what the move actually does on the board, NOT what you
  wish it did. We will verify every claim deterministically using the rules of chess.

Use the `submit_move` tool to return your answer."""


SUBMIT_MOVE_PARAMETERS = {
    "type": "object",
    "properties": {
        "position_summary": {
            "type": "string",
            "description": "One- or two-sentence summary of the position.",
        },
        "candidates": {
            "type": "array",
            "minItems": 1,
            "maxItems": 6,
            "items": {
                "type": "object",
                "properties": {
                    "san": {
                        "type": "string",
                        "description": "Candidate move in SAN notation.",
                    },
                    "rationale": {
                        "type": "string",
                        "description": "Why this move is being considered.",
                    },
                    "claims": {
                        "type": "object",
                        "description": "Rule-based facts about this move. Every field is required.",
                        "properties": {
                            "is_check": {"type": "boolean"},
                            "is_capture": {"type": "boolean"},
                            "captured_piece": {
                                "type": ["string", "null"],
                                "enum": ["P", "N", "B", "R", "Q", None],
                            },
                            "is_castle": {"type": "boolean"},
                            "is_promotion": {"type": "boolean"},
                            "is_en_passant": {"type": "boolean"},
                            "gives_mate": {"type": "boolean"},
                        },
                        "required": [
                            "is_check", "is_capture", "captured_piece",
                            "is_castle", "is_promotion", "is_en_passant", "gives_mate",
                        ],
                    },
                },
                "required": ["san", "rationale", "claims"],
            },
        },
        "chosen_move": {
            "type": "string",
            "description": "The SAN move you commit to playing. Must appear in candidates.",
        },
    },
    "required": ["position_summary", "candidates", "chosen_move"],
}

SUBMIT_MOVE_DESCRIPTION = "Submit a chess move with reasoning and rule-based claims. Must be called exactly once."


@dataclass
class CallOutcome:
    """Raw outcome of a single adapter call. The eval layer decides scoring."""
    response: MoveResponse | None
    raw_tool_input: dict | None
    latency_ms: int
    error: str | None
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0


def build_user_message(
    fen: str,
    prior_failed: list[str] | None = None,
    augment_legal_moves: bool = False,
) -> str:
    """Construct the user message for a position. Identical across providers."""
    board = chess.Board(fen)
    text = (
        f"Position (FEN): {fen}\n\n"
        f"Board (white at bottom):\n{board}\n\n"
        f"Side to move: {'White' if board.turn else 'Black'}\n\n"
        f"Analyze and submit your move via the `submit_move` tool."
    )
    if augment_legal_moves:
        legal_sans = sorted(board.san(m) for m in board.legal_moves)
        text += (
            f"\n\nFor reference, here are ALL legal moves in this position "
            f"({len(legal_sans)} total):\n{legal_sans}\n"
            "This list is informational. Use it to verify your candidates are legal."
        )
    if prior_failed:
        # Cap to last 3 attempts. Longer history bloats the prompt, which
        # causes reasoning models to spend more tokens reasoning about it,
        # which on retry-heavy moves can exhaust the response token budget.
        # 3 is enough to communicate "you've tried these and they're wrong"
        # without ballooning context as retries accumulate.
        capped = prior_failed[-3:]
        ellipsis = f" (and {len(prior_failed) - 3} earlier)" if len(prior_failed) > 3 else ""
        text += (
            "\n\nIMPORTANT: in earlier attempts at THIS SAME position you proposed "
            f"these moves which were ILLEGAL: {capped}{ellipsis}. "
            "Do NOT propose any of them again. Re-examine the board carefully — "
            "in particular check which of your own pieces are still on the board, "
            "and which squares the opponent attacks. Submit a different, legal move."
        )
    return text


def parse_tool_input(raw_input: Any) -> tuple[MoveResponse | None, str | None]:
    """Validate the raw tool-input dict against the MoveResponse schema."""
    if raw_input is None:
        return None, "Model did not return a tool call"
    try:
        raw_dict = dict(raw_input) if not isinstance(raw_input, dict) else raw_input
    except Exception as e:
        return None, f"Could not coerce tool input to dict: {type(e).__name__}: {e}"
    try:
        return MoveResponse.model_validate(raw_dict), None
    except ValidationError as e:
        return None, f"Tool input failed schema validation: {e}"


class Timer:
    """Context manager for measuring latency in ms."""
    def __init__(self) -> None:
        self.ms = 0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args: Any) -> None:
        self.ms = int((time.perf_counter() - self._start) * 1000)
