"""PlayStrength (PS) — move-quality metric across honest, completed playthroughs.

Complement to ChessReliability:
  - CR measures rule-following (forfeit mode). Catches illegal moves.
  - PS measures move quality across a full game (retry mode at fixed skill).
    Catches bad legal moves throughout the game, especially in mid/endgame
    where the model's spatial state degrades.

Run as retry mode so games progress to a natural conclusion (the model finds
its own legal moves with feedback rather than Stockfish substituting them in).

Metric (per game, then averaged across games):
    PS_game = (legal_moves_played / max_moves) × max(0, 1 − ACPL_capped / ACPL_CAP)

    where ACPL_capped is the mean over move-level cp_loss values, each clamped to
    ACPL_CAP first (so one mate-blunder doesn't dominate the average), and
    legal_moves_played is the count of moves the model successfully made.

The soft-completion factor (plies/max_plies) gives partial credit for games that
reach mid/endgame even if they eventually forfeit on exhausted retries. A full
40-move game with ACPL 100 → PS ≈ 0.90; a 10-move forfeit at ACPL 50 → PS ≈ 0.24.

This formula is structurally identical to ChessReliability (CR), but PS is
computed over RETRY-mode games at fixed Stockfish skill, so it captures
mid/endgame move quality that CR (forfeit mode) misses.

Reported separately:
  - completion_rate_natural: fraction of games that ended with a real chess
    result (1-0/0-1/draw) rather than running out of retries
  - ACPL by phase (opening / middlegame / endgame)

Phase breakdown reported separately:
  - opening:    moves 1-15
  - middlegame: moves 16-30
  - endgame:    moves 31+
"""
from __future__ import annotations

from ..types import GameRecord

ACPL_CAP = 1000  # 10 pawns of loss per move caps the penalty
OPENING_END = 15
MIDDLE_END = 30


def _phase_for_move(ply: int) -> str:
    if ply <= OPENING_END:
        return "opening"
    if ply <= MIDDLE_END:
        return "middlegame"
    return "endgame"


def compute_play_strength(games: list[GameRecord], max_plies: int = 60) -> dict:
    if not games:
        return {
            "play_strength": 0.0,
            "n_games": 0,
            "completion_rate_natural": 0.0,
            "mean_survival": 0.0,
            "mean_plies_legal": 0.0,
            "quality_component_mean": 0.0,
            "acpl_overall": 0.0,
            "acpl_opening": 0.0,
            "acpl_middlegame": 0.0,
            "acpl_endgame": 0.0,
            "wins": 0, "draws": 0, "losses": 0, "forfeits": 0,
            "per_game_scores": [],
            "per_game_acpl": [],
        }

    completions = 0
    wins = draws = losses = forfeits = 0
    all_cp_capped: list[int] = []
    cp_by_phase: dict[str, list[int]] = {"opening": [], "middlegame": [], "endgame": []}
    per_game_acpl: list[float] = []
    per_game_survival: list[float] = []
    per_game_quality: list[float] = []
    per_game_scores: list[float] = []

    for g in games:
        if g.result == "forfeit_illegal":
            forfeits += 1
        else:
            completions += 1
            if g.model_won:
                wins += 1
            elif g.result == "1/2-1/2":
                draws += 1
            else:
                losses += 1

        legal_moves = [m for m in g.moves if m.chosen_legal]
        per_game = [min(m.cp_loss, ACPL_CAP) for m in legal_moves]
        all_cp_capped.extend(per_game)
        game_acpl = sum(per_game) / len(per_game) if per_game else float(ACPL_CAP)
        per_game_acpl.append(game_acpl)

        for m in legal_moves:
            cp_by_phase[_phase_for_move(m.ply)].append(min(m.cp_loss, ACPL_CAP))

        survival = min(len(legal_moves), max_plies) / max_plies
        quality = max(0.0, 1.0 - game_acpl / ACPL_CAP)
        per_game_survival.append(survival)
        per_game_quality.append(quality)
        per_game_scores.append(survival * quality)

    n = len(games)
    completion_rate_natural = completions / n
    acpl_overall = sum(all_cp_capped) / len(all_cp_capped) if all_cp_capped else 0.0
    ps = sum(per_game_scores) / n

    def _mean(xs: list[int]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    return {
        "play_strength": ps,
        "n_games": n,
        "completion_rate_natural": completion_rate_natural,
        "mean_survival": sum(per_game_survival) / n,
        "mean_plies_legal": sum(s * max_plies for s in per_game_survival) / n,
        "quality_component_mean": sum(per_game_quality) / n,
        "acpl_overall": acpl_overall,
        "acpl_opening": _mean(cp_by_phase["opening"]),
        "acpl_middlegame": _mean(cp_by_phase["middlegame"]),
        "acpl_endgame": _mean(cp_by_phase["endgame"]),
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "forfeits": forfeits,
        "per_game_scores": per_game_scores,
        "per_game_acpl": per_game_acpl,
        "moves_by_phase": {k: len(v) for k, v in cp_by_phase.items()},
    }
