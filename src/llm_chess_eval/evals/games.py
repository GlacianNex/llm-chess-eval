"""Game eval: model plays full games vs Stockfish. Every move instrumented.

Per-move progress logging
=========================
Each API call attempt emits one line to stdout (flushed) and one JSONL line
to `<run_dir>/progress.jsonl`. Format of the stdout line:

    [PROGRESS] game=<game_id> ply=<N> attempt=<K>/<MAX> elapsed=<sec>s ...

This gives an external monitor visibility down to the per-call level. Even
when a single game takes 30+ minutes, the monitor sees an event every API
call (typically every 5-60 seconds depending on the model).

Three modes for handling illegal model moves:
  - "forfeit"    (default): game ends on first illegal, model loses
  - "substitute": Stockfish-best is played in place of the illegal move; game continues.
                  Lets us see whether the model cascades into more errors or stabilizes.
  - "retry":     The model is told its move was illegal and asked to try again,
                  up to `max_retries` times. Each retry costs 0.5x on per-move quality.
                  If all retries fail, the game forfeits on that ply.

Per-game composite score:

    game_score = mean(per_move_quality) * mean(per_move_consistency) * 0.5 ** n_forfeits

per_move_quality blends positional quality and retry-penalty:
    quality_cp     = 1 - clamp(cp_loss, 0, 1000) / 1000
    quality_retry  = 0.5 ** retries_used
    per_move       = quality_cp * quality_retry
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Literal

import chess
import chess.engine

from ..adapters.base import ModelAdapter
from ..config import stockfish_path
from ..engine import analyse_pov
from ..evals.consistency import CLAIM_FIELDS, _compare, ground_truth_claims
from ..types import GameRecord, MoveRecord


def _log_progress(
    progress_path: Path | None,
    game_id: str,
    model: str,
    ply: int,
    attempt: int,
    max_attempts: int,
    elapsed_s: float,
    event: str,
    note: str = "",
) -> None:
    """Emit per-attempt progress to stdout (line-buffered) AND a side JSONL.

    Lets an external monitor see per-call activity without waiting for the
    whole game to complete. Format chosen so a `tail -f`'d progress.jsonl or
    a `grep PROGRESS` over stdout both work.
    """
    line = (
        f"[PROGRESS] game={game_id.split('__')[-1]} model={model} "
        f"ply={ply} attempt={attempt}/{max_attempts} elapsed={elapsed_s:.1f}s "
        f"event={event} {note}".rstrip()
    )
    print(line, flush=True)
    sys.stdout.flush()
    if progress_path is not None:
        try:
            with progress_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "ts": time.time(),
                    "game_id": game_id,
                    "model": model,
                    "ply": ply,
                    "attempt": attempt,
                    "max_attempts": max_attempts,
                    "elapsed_s": elapsed_s,
                    "event": event,
                    "note": note,
                }) + "\n")
        except Exception:
            pass  # never let logging kill the run

GameMode = Literal["forfeit", "substitute", "retry"]
MATE_CLAMP = 100000
CP_LOSS_CAP = 1000

# Gradual reasoning-effort stepdown. When an attempt fails with
# `finish_reason='length'`, the next retry uses a lower reasoning effort
# to give the model a tighter budget to commit. The ladder is gradual:
# default → medium → low → minimal. The model still gets to think; the
# step just shrinks how much.
_REASONING_STEPDOWN_LADDER = [None, "medium", "low", "minimal"]


def _stepdown_for(length_failures: int) -> str | None:
    """Return the reasoning_effort override given how many length-failures
    we've already seen for this move. None = use provider default (full)."""
    idx = min(length_failures, len(_REASONING_STEPDOWN_LADDER) - 1)
    return _REASONING_STEPDOWN_LADDER[idx]


def _live_print_move_quality(cp_loss: int) -> float:
    """Linear quality used only for the live in-harness summary print.
    Canonical scoring lives in chess_reliability.chess_reliability() and
    play_strength.compute_play_strength()."""
    loss = max(0, cp_loss)
    if loss >= CP_LOSS_CAP:
        return 0.0
    return 1.0 - loss / CP_LOSS_CAP


def _live_print_retry_multiplier(retries: int) -> float:
    """Linear-ish retry multiplier for the live in-harness summary print.
    Canonical retry cost is 0.25^n in chess_reliability."""
    return 0.5 ** retries


def _candidate_claim_consistency(board: chess.Board, candidate_dict: dict | None) -> float:
    if candidate_dict is None:
        return 0.0
    try:
        move = board.parse_san(candidate_dict["san"])
    except Exception:
        return 0.0
    truth = ground_truth_claims(board, move).model_dump()
    claimed = candidate_dict.get("claims", {}) or {}
    hits = sum(_compare(claimed.get(k), truth.get(k), k) for k in CLAIM_FIELDS)
    return hits / len(CLAIM_FIELDS)


def _try_parse(board: chess.Board, san: str | None) -> chess.Move | None:
    if san is None:
        return None
    try:
        return board.parse_san(san)
    except Exception:
        return None


def _attempt_move(
    adapter: ModelAdapter,
    board: chess.Board,
    prior_failed: list[str],
    reasoning_effort_override: str | None = None,
) -> tuple[dict | None, chess.Move | None, str | None, float, float, int, str | None, dict | None]:
    """One LLM call. Returns (raw_input, parsed_move_or_None, chosen_san_or_None, candidates_legal_rate, chosen_candidate_claim_consistency, latency_ms, error, chosen_cand_dict)."""
    outcome = adapter.propose_move(
        board.fen(),
        prior_failed=prior_failed or None,
        reasoning_effort_override=reasoning_effort_override,
    )
    resp = outcome.response
    raw = outcome.raw_tool_input
    if resp is None:
        return raw, None, None, 0.0, 0.0, outcome.latency_ms, outcome.error, None

    chosen_san = resp.chosen_move
    n_total = len(resp.candidates)
    n_legal = sum(1 for c in resp.candidates if _try_parse(board, c.san) is not None)
    cand_legal_rate = n_legal / n_total if n_total else 0.0
    chosen_cand_dict = next((c.model_dump() for c in resp.candidates if c.san == chosen_san), None)
    claim_consistency = _candidate_claim_consistency(board, chosen_cand_dict)
    parsed = _try_parse(board, chosen_san)
    return raw, parsed, chosen_san, cand_legal_rate, claim_consistency, outcome.latency_ms, outcome.error, chosen_cand_dict


def play_game(
    adapter: ModelAdapter,
    game_id: str,
    color: str = "white",
    skill: int = 3,
    sf_depth: int = 12,
    max_plies: int = 60,
    starting_fen: str | None = None,
    mode: GameMode = "forfeit",
    max_retries: int = 3,
    progress_path: Path | None = None,
) -> GameRecord:
    if color not in ("white", "black"):
        raise ValueError("color must be 'white' or 'black'")
    if mode not in ("forfeit", "substitute", "retry"):
        raise ValueError(f"unknown mode: {mode}")

    board = chess.Board(starting_fen) if starting_fen else chess.Board()
    model_is_white = color == "white"
    moves: list[MoveRecord] = []
    n_illegal = 0
    result_code = "max_plies"

    sf_path = str(stockfish_path())
    eval_eng = chess.engine.SimpleEngine.popen_uci(sf_path)
    opp_eng = chess.engine.SimpleEngine.popen_uci(sf_path)

    try:
        eval_eng.configure({"Threads": 1})
        opp_eng.configure({"Threads": 1, "Skill Level": skill})

        ply_count = 0
        while ply_count < max_plies:
            if board.is_game_over():
                result_code = board.result()
                break

            if board.turn == (chess.WHITE if model_is_white else chess.BLACK):
                ply_count += 1
                fen_before = board.fen()
                cp_before = analyse_pov(eval_eng, board, sf_depth)
                sf_best = eval_eng.play(board, chess.engine.Limit(depth=sf_depth)).move
                sf_best_san = board.san(sf_best) if sf_best else "?"

                failed_attempts: list[str] = []
                total_latency = 0
                retries_used = 0
                last_raw = None
                last_err = None
                last_cand_legal_rate = 0.0
                last_claim_consistency = 0.0
                last_chosen_san: str | None = None
                chosen_move: chess.Move | None = None
                # Track reasoning-budget exhaustion to gradually step reasoning down.
                # See _stepdown_for() below.
                length_failures = 0

                # Single attempt for forfeit/substitute; multiple for retry.
                max_attempts = (max_retries + 1) if mode == "retry" else 1
                ply_start_ts = time.perf_counter()
                for attempt_idx in range(max_attempts):
                    effort_override = _stepdown_for(length_failures)
                    _log_progress(
                        progress_path, game_id, adapter.model, ply_count,
                        attempt_idx + 1, max_attempts,
                        time.perf_counter() - ply_start_ts,
                        "call_start",
                        (f"prior_failed={failed_attempts}" if failed_attempts else "")
                        + (f" reasoning_effort={effort_override}" if effort_override else ""),
                    )
                    (
                        raw,
                        parsed_move,
                        chosen_san,
                        cand_legal_rate,
                        claim_consistency,
                        latency_ms,
                        err,
                        _chosen_cand,
                    ) = _attempt_move(adapter, board, failed_attempts, reasoning_effort_override=effort_override)
                    # If this attempt failed due to reasoning-budget exhaustion,
                    # step reasoning effort down on the next retry. Gradual:
                    # default → medium → low → minimal. We do NOT skip straight
                    # to minimal — give the model a chance to keep reasoning at
                    # progressively bounded levels.
                    #
                    # Triggers (both are reasoning-budget symptoms):
                    #   - "finish_reason='length'": OpenAI/Gemini hit max_output_tokens
                    #     mid-reasoning, no tool call emitted.
                    #   - "MALFORMED_FUNCTION_CALL": Gemini emits a function call but
                    #     the args don't parse against the schema — typically from
                    #     reasoning that runs long enough to corrupt the tool output.
                    if err and ("finish_reason='length'" in err or "MALFORMED_FUNCTION_CALL" in err):
                        length_failures += 1
                    total_latency += latency_ms
                    _log_progress(
                        progress_path, game_id, adapter.model, ply_count,
                        attempt_idx + 1, max_attempts,
                        time.perf_counter() - ply_start_ts,
                        "call_done" if parsed_move is not None else ("call_illegal" if chosen_san else "call_no_tool"),
                        f"san={chosen_san!r} latency_ms={latency_ms}" + (f" err={err[:80]}" if err else ""),
                    )
                    last_raw = raw
                    last_err = err
                    last_cand_legal_rate = cand_legal_rate
                    last_claim_consistency = claim_consistency
                    last_chosen_san = chosen_san

                    if parsed_move is not None:
                        chosen_move = parsed_move
                        break  # legal — stop retrying

                    # Illegal attempt; record what was tried, increment retry count
                    if chosen_san is not None:
                        failed_attempts.append(chosen_san)
                    if attempt_idx < max_attempts - 1:
                        retries_used += 1  # we will retry

                # Outcome dispatch
                fallback_used = False
                actual_move: chess.Move | None = None

                if chosen_move is not None:
                    actual_move = chosen_move
                else:
                    # All attempts failed (or just the one in forfeit/substitute mode)
                    n_illegal += 1
                    if mode == "substitute":
                        actual_move = sf_best
                        fallback_used = True
                    elif mode == "retry":
                        # exhausted retries → forfeit
                        actual_move = None
                    else:
                        # forfeit
                        actual_move = None

                if actual_move is None:
                    moves.append(MoveRecord(
                        ply=ply_count,
                        fen_before=fen_before,
                        sf_best_san=sf_best_san,
                        cp_before=cp_before,
                        chosen_san=last_chosen_san,
                        chosen_legal=False,
                        chose_in_candidates=False,
                        candidates_legal_rate=last_cand_legal_rate,
                        claim_consistency=last_claim_consistency,
                        actual_played_san=None,
                        chosen_was_top=False,
                        cp_after=None,
                        cp_loss=MATE_CLAMP,
                        latency_ms=total_latency,
                        model_error=last_err,
                        raw_response=last_raw,
                        retries_used=retries_used,
                        failed_attempts=failed_attempts,
                        fallback_used=False,
                    ))
                    result_code = "forfeit_illegal"
                    break

                # We have a legal move to play (LLM's eventual legal pick OR substituted SF best).
                actual_san = board.san(actual_move)
                chosen_was_top = (sf_best is not None) and (actual_move == sf_best)
                board.push(actual_move)
                cp_after_opp = analyse_pov(eval_eng, board, sf_depth)
                cp_after = -cp_after_opp
                cp_loss = max(0, cp_before - cp_after)

                moves.append(MoveRecord(
                    ply=ply_count,
                    fen_before=fen_before,
                    sf_best_san=sf_best_san,
                    cp_before=cp_before,
                    chosen_san=last_chosen_san,
                    chosen_legal=chosen_move is not None,
                    chose_in_candidates=True if chosen_move is not None else False,
                    candidates_legal_rate=last_cand_legal_rate,
                    claim_consistency=last_claim_consistency if chosen_move is not None else 0.0,
                    actual_played_san=actual_san,
                    chosen_was_top=bool(chosen_was_top),
                    cp_after=cp_after,
                    cp_loss=cp_loss,
                    latency_ms=total_latency,
                    model_error=last_err,
                    raw_response=last_raw,
                    retries_used=retries_used,
                    failed_attempts=failed_attempts,
                    fallback_used=fallback_used,
                ))
            else:
                opp_result = opp_eng.play(board, chess.engine.Limit(depth=sf_depth))
                if opp_result.move is None:
                    break
                board.push(opp_result.move)
    finally:
        eval_eng.quit()
        opp_eng.quit()

    if result_code == "max_plies" and board.is_game_over():
        result_code = board.result()

    if result_code == "1-0":
        model_won = model_is_white
    elif result_code == "0-1":
        model_won = not model_is_white
    else:
        model_won = False

    return GameRecord(
        game_id=game_id,
        model=adapter.model,
        color="white" if model_is_white else "black",
        opponent="stockfish",
        skill=skill,
        sf_depth=sf_depth,
        starting_fen=starting_fen or chess.STARTING_FEN,
        moves=moves,
        final_fen=board.fen(),
        result=result_code,
        model_won=model_won,
        n_plies=len(moves),
        n_illegal=n_illegal,
    )


def score_game(record: GameRecord) -> dict:
    """Per-game composite score, including retry penalty in per-move quality."""
    if not record.moves:
        return {
            "game_score": 0.0,
            "per_move_quality": 0.0,
            "per_move_consistency": 0.0,
            "illegal_multiplier": 0.0,
            "mean_cp_loss": 0.0,
            "blunder_rate_300": 0.0,
            "blunder_rate_1000": 0.0,
            "chose_in_candidates_rate": 0.0,
            "candidates_legal_rate_mean": 0.0,
            "chosen_was_top_rate": 0.0,
            "mean_retries": 0.0,
            "retry_success_rate": 0.0,
            "n_fallbacks": 0.0,
        }

    legal_moves = [m for m in record.moves if m.chosen_legal]
    n = len(record.moves)

    # per_move_quality blends cp_loss quality and retry penalty — this is
    # the LIVE-PRINT helper score (linear quality, 0.5^retries). Canonical
    # scoring lives in chess_reliability/play_strength.
    if legal_moves:
        per_move_quality = sum(
            _live_print_move_quality(m.cp_loss) * _live_print_retry_multiplier(m.retries_used)
            for m in legal_moves
        ) / len(legal_moves)
        per_move_consistency = sum(m.claim_consistency for m in legal_moves) / len(legal_moves)
        mean_cp_loss = sum(m.cp_loss for m in legal_moves) / len(legal_moves)
        chosen_top = sum(1 for m in legal_moves if m.chosen_was_top) / len(legal_moves)
        blunder_300 = sum(1 for m in legal_moves if m.cp_loss >= 300) / len(legal_moves)
        blunder_1000 = sum(1 for m in legal_moves if m.cp_loss >= 1000) / len(legal_moves)
    else:
        per_move_quality = 0.0
        per_move_consistency = 0.0
        mean_cp_loss = float(MATE_CLAMP)
        chosen_top = 0.0
        blunder_300 = 1.0
        blunder_1000 = 1.0

    # ANY illegal move counts in the multiplier — substitute mode doesn't get a free pass.
    # Retries that eventually succeed are NOT counted as n_illegal (record.n_illegal only bumps on
    # final failure / substitution / terminal forfeit), so retry-mode games where the model
    # eventually finds a legal move keep multiplier = 1.0 (retries are penalized per-move instead).
    illegal_multiplier = 0.5 ** record.n_illegal

    composite = per_move_quality * per_move_consistency * illegal_multiplier

    return {
        "game_score": composite,
        "per_move_quality": per_move_quality,
        "per_move_consistency": per_move_consistency,
        "illegal_multiplier": illegal_multiplier,
        "mean_cp_loss": mean_cp_loss,
        "blunder_rate_300": blunder_300,
        "blunder_rate_1000": blunder_1000,
        "chose_in_candidates_rate": sum(1 for m in record.moves if m.chose_in_candidates) / n,
        "candidates_legal_rate_mean": sum(m.candidates_legal_rate for m in record.moves) / n,
        "chosen_was_top_rate": chosen_top,
        "mean_retries": sum(m.retries_used for m in record.moves) / n,
        "retry_success_rate": sum(1 for m in record.moves if m.retries_used > 0 and m.chosen_legal) / max(1, sum(1 for m in record.moves if m.retries_used > 0)),
        "n_fallbacks": float(sum(1 for m in record.moves if m.fallback_used)),
    }
