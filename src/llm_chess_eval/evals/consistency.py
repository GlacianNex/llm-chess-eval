"""Consistency eval: do the model's RULE-BASED claims match what the move actually does?

For each legal candidate, we compute the ground-truth values for every claim using
python-chess. No engine, no judgment — just the rules of chess. The model is graded
on whether what it says about the move matches what the move objectively does.

Per-candidate consistency = fraction of its 7 claim fields that match ground truth.
Per-position consistency = mean across legal candidates.

Sub-scores: per-field accuracy across all legal candidates in the position.
"""
from __future__ import annotations

import chess

from ..adapters.base import ModelAdapter
from ..types import EvalResultRow, MoveClaims, PositionRecord

CLAIM_FIELDS = (
    "is_check",
    "is_capture",
    "captured_piece",
    "is_castle",
    "is_promotion",
    "is_en_passant",
    "gives_mate",
)


def ground_truth_claims(board: chess.Board, move: chess.Move) -> MoveClaims:
    """Compute what a move ACTUALLY does, per the rules of chess."""
    is_capture = board.is_capture(move)
    is_en_passant = board.is_en_passant(move)

    captured_piece: str | None = None
    if is_capture:
        if is_en_passant:
            captured_piece = "P"
        else:
            target = board.piece_at(move.to_square)
            if target is not None:
                captured_piece = target.symbol().upper()

    # gives_check vs gives_mate: gives_mate is the strict "this is mate" case;
    # is_check is "puts the opponent in check but is NOT mate".
    pushed = board.copy(stack=False)
    pushed.push(move)
    is_mate = pushed.is_checkmate()
    is_check_only = pushed.is_check() and not is_mate

    return MoveClaims(
        is_check=is_check_only,
        is_capture=is_capture,
        captured_piece=captured_piece,
        is_castle=board.is_castling(move),
        is_promotion=move.promotion is not None,
        is_en_passant=is_en_passant,
        gives_mate=is_mate,
    )


def _compare(claim: bool | str | None, truth: bool | str | None, field: str) -> bool:
    """Match rules. captured_piece is None-equivalent when not a capture."""
    if field == "captured_piece":
        # Both None or both same letter (case-insensitive)
        c = (claim or "").upper() if isinstance(claim, str) else claim
        t = (truth or "").upper() if isinstance(truth, str) else truth
        return c == t
    return claim == truth


def score_position(
    adapter: ModelAdapter,
    pos: PositionRecord,
    run_id: str,
) -> EvalResultRow:
    board = chess.Board(pos.fen)
    outcome = adapter.propose_move(pos.fen)

    empty_sub = {f"claim_{k}": 0.0 for k in CLAIM_FIELDS}
    empty_sub["candidates_legal_rate"] = 0.0
    empty_sub["per_candidate_consistency"] = 0.0

    if outcome.error or outcome.response is None:
        return EvalResultRow(
            run_id=run_id,
            eval_name="consistency",
            model=adapter.model,
            position_id=pos.position_id,
            fen=pos.fen,
            score=0.0,
            sub_scores=empty_sub,
            raw_response=outcome.raw_tool_input,
            error=outcome.error,
            latency_ms=outcome.latency_ms,
        )

    resp = outcome.response

    legal_pairs: list[tuple[dict, chess.Move]] = []
    for c in resp.candidates:
        try:
            mv = board.parse_san(c.san)
            legal_pairs.append((c.model_dump(), mv))
        except Exception:
            pass

    n_total = len(resp.candidates) or 1
    n_legal = len(legal_pairs)
    candidates_legal_rate = n_legal / n_total

    # Per-field hit counts, per-candidate consistency, and verbose diff for debugging
    field_hits = {k: 0 for k in CLAIM_FIELDS}
    field_total = 0  # legal-candidate count (same denominator for each field)
    per_cand_consistency: list[float] = []
    diffs: list[dict] = []

    for c_dict, mv in legal_pairs:
        truth = ground_truth_claims(board, mv).model_dump()
        claimed = c_dict["claims"]
        cand_hits = 0
        per_field_match: dict[str, bool] = {}
        for k in CLAIM_FIELDS:
            ok = _compare(claimed.get(k), truth.get(k), k)
            per_field_match[k] = ok
            if ok:
                field_hits[k] += 1
                cand_hits += 1
        field_total += 1
        per_cand_consistency.append(cand_hits / len(CLAIM_FIELDS))
        diffs.append({
            "san": c_dict["san"],
            "claimed": claimed,
            "truth": truth,
            "matches": per_field_match,
            "field_accuracy": cand_hits / len(CLAIM_FIELDS),
        })

    if field_total == 0:
        primary = 0.0
        sub = empty_sub
        sub["candidates_legal_rate"] = candidates_legal_rate
    else:
        primary = sum(per_cand_consistency) / field_total
        sub = {f"claim_{k}": field_hits[k] / field_total for k in CLAIM_FIELDS}
        sub["candidates_legal_rate"] = candidates_legal_rate
        sub["per_candidate_consistency"] = primary

    return EvalResultRow(
        run_id=run_id,
        eval_name="consistency",
        model=adapter.model,
        position_id=pos.position_id,
        fen=pos.fen,
        score=primary,
        sub_scores=sub,
        raw_response={
            "model_response": outcome.raw_tool_input,
            "candidate_diffs": diffs,
        },
        error=None,
        latency_ms=outcome.latency_ms,
    )
