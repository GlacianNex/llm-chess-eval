"""Legality eval: given a FEN, does the model produce a legal SAN move?

Score per position: 1.0 if `chosen_move` parses as legal SAN, else 0.0.
Sub-scores per position:
  - tool_called: did the model emit the tool call at all
  - chose_in_candidates: did `chosen_move` appear in its own `candidates` list
  - candidates_legal_rate: fraction of `candidates[].san` that are legal
"""
from __future__ import annotations

import chess

from ..adapters.base import ModelAdapter
from ..types import EvalResultRow, PositionRecord


def _is_legal_san(board: chess.Board, san: str) -> bool:
    try:
        board.parse_san(san)
        return True
    except (chess.IllegalMoveError, chess.InvalidMoveError, chess.AmbiguousMoveError, ValueError):
        return False


def score_position(
    adapter: ModelAdapter,
    pos: PositionRecord,
    run_id: str,
) -> EvalResultRow:
    board = chess.Board(pos.fen)
    outcome = adapter.propose_move(pos.fen)

    if outcome.error or outcome.response is None:
        return EvalResultRow(
            run_id=run_id,
            eval_name="legality",
            model=adapter.model,
            position_id=pos.position_id,
            fen=pos.fen,
            score=0.0,
            sub_scores={
                "tool_called": 0.0 if outcome.response is None else 1.0,
                "chose_in_candidates": 0.0,
                "candidates_legal_rate": 0.0,
            },
            raw_response=outcome.raw_tool_input,
            error=outcome.error,
            latency_ms=outcome.latency_ms,
        )

    resp = outcome.response
    chosen_legal = _is_legal_san(board, resp.chosen_move)
    chose_in_cands = any(c.san == resp.chosen_move for c in resp.candidates)
    legal_cands = sum(_is_legal_san(board, c.san) for c in resp.candidates)
    legal_rate = legal_cands / len(resp.candidates) if resp.candidates else 0.0

    return EvalResultRow(
        run_id=run_id,
        eval_name="legality",
        model=adapter.model,
        position_id=pos.position_id,
        fen=pos.fen,
        score=1.0 if chosen_legal else 0.0,
        sub_scores={
            "tool_called": 1.0,
            "chose_in_candidates": 1.0 if chose_in_cands else 0.0,
            "candidates_legal_rate": legal_rate,
        },
        raw_response=outcome.raw_tool_input,
        error=None,
        latency_ms=outcome.latency_ms,
    )
