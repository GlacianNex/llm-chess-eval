"""PlayQuality (formerly PlayStrength) — move-strength metric across full games.

Complement to ChessReliability:
  - ChessReliability measures rule-following: can the model produce a legal
    move on first attempt at a low Stockfish skill, with steep per-retry cost.
  - PlayQuality measures move strength once a legal move is found. Stockfish
    skill 5 (harder opponent), retry mode with max_retries=3 (no per-retry
    cost). Captures mid/endgame move quality that ChessReliability's tighter
    retry budget misses.

Metric (per game, then averaged across games):
    per_move_score = move_quality(cp_loss) × game_phase_weight(ply)
    move_quality(cp_loss) = exp(-cp_loss / QUALITY_DECAY_CONSTANT)
    game_phase_weight(ply) = 1 / 1.5 / 2 / 3 by ply bucket (softened from
    earlier 1/2/4/8 so the denominator isn't dominated by late plies)

    game_score = sum_over_legal_moves(per_move_score) / max_possible_weighted_score

PlayQuality does NOT include a retry cost multiplier — that's ChessReliability's
job. Once a move is found legally, PlayQuality scores its strength independent
of how many attempts it took. This is the conceptual split between the two
metrics: Reliability is "can the model play legal chess on first attempt";
PlayQuality is "given it found a legal move, how good was it."

The denominator (max_possible_weighted_score = sum of phase weights for plies
1..max_plies) is constant per max_plies. An early forfeit loses both the
missing per-move scores AND the high-weight late plies that would have
contributed disproportionately. This bakes the memorization-cliff thesis
into the metric: surviving to ply 30 is worth more than playing ply 5 well.

Default config:
    - retry mode with max_retries = 3 (no per-retry cost)
    - QUALITY_DECAY_CONSTANT = 150 (same as ChessReliability)
    - Phase weights 1/1.5/2/3 at ply boundaries 10/20/30 (same as ChessReliability)
    - Stockfish skill 5, max_plies 60

Reported separately:
  - completion_rate_natural: fraction of games that ended with a real chess
    result (1-0/0-1/draw) rather than running out of retries
  - ACPL by phase (opening / middlegame / endgame)
"""
from __future__ import annotations

import math

from ..types import GameRecord

ACPL_CAP = 1000  # 10 pawns of loss per move caps the penalty
QUALITY_DECAY_CONSTANT = 150.0
OPENING_END = 15
MIDDLE_END = 30


def _phase_for_move(ply: int) -> str:
    if ply <= OPENING_END:
        return "opening"
    if ply <= MIDDLE_END:
        return "middlegame"
    return "endgame"


def _game_phase_weight(ply: int) -> float:
    """Softened phase weight; matches CR. Endgame at 3× opening (was 8×)."""
    if ply < 10:
        return 1.0
    if ply < 20:
        return 1.5
    if ply < 30:
        return 2.0
    return 3.0


def _max_possible_weighted_score(max_plies: int) -> float:
    return sum(_game_phase_weight(p) for p in range(1, max_plies + 1))


def _move_quality_from_cp_loss(cp_loss: int) -> float:
    """Exponential decay quality; matches CR."""
    return math.exp(-max(0, cp_loss) / QUALITY_DECAY_CONSTANT)


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

    max_weighted_score = _max_possible_weighted_score(max_plies)
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
        # Diagnostic mean quality (exp decay), useful for the per-cell scorecard.
        quality_mean = (
            sum(_move_quality_from_cp_loss(m.cp_loss) for m in legal_moves) / len(legal_moves)
        ) if legal_moves else 0.0
        # Composite PS: weighted sum / max possible weighted. PS does NOT
        # apply a retry cost multiplier (that's CR's job) — once a move is
        # found, it's scored on its strength alone.
        weighted_sum = sum(
            _move_quality_from_cp_loss(m.cp_loss) * _game_phase_weight(m.ply)
            for m in legal_moves
        )
        game_score = weighted_sum / max_weighted_score if max_weighted_score else 0.0
        per_game_survival.append(survival)
        per_game_quality.append(quality_mean)
        per_game_scores.append(game_score)

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
