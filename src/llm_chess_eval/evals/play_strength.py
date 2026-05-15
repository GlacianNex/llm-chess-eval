"""PlayStrength (PS) — primary composite metric.

PlayStrength is a single 0–1 score that captures how strongly a model plays
chess across full games against an amateur-tier Stockfish opponent. It bakes
three things into one number: move quality, rule-following discipline (cost
for needing retries), and game-phase progression (late game weighted more).

    per_move_score = move_quality(cp_loss) × retry_cost(retries) × game_phase_weight(ply)

    move_quality(cp_loss) = exp(-cp_loss / QUALITY_DECAY_CONSTANT)
        exponential decay so the top of the scale isn't squashed.
        cp_loss=0 → 1.000  (Stockfish's top move)
        cp_loss=50 → 0.717 (grandmaster-level)
        cp_loss=150 → 0.368 (competent club)
        cp_loss=500 → 0.036 (blunder)
        cp_loss=1000 → 0.001 (mate-level blunder)

    retry_cost(retries) = 0.25 ^ retries_used
        steep multiplicative penalty — 1 retry costs 75% of move value,
        2 retries cost 94%. Models that need the retry safety net to find
        a legal move can't recover the score even when they eventually do.

    game_phase_weight(ply) = 1 / 1.5 / 2 / 3 by ply bucket
        Endgame counts 3× opening, encoding the memorization-cliff thesis
        (later plies are progressively more novel from training) without
        making scores collapse for models that break down in middlegame.

Per-game score:
    game_score = sum_over_legal_moves(per_move_score) / max_possible_weighted_score

    where max_possible_weighted_score = sum(game_phase_weight(p) for p in 1..max_plies).
    The denominator is constant per max_plies, so an early forfeit loses
    BOTH the missing per-move scores AND the high-weight plies that would
    have contributed disproportionately to the denominator. Forfeit at
    ply 20 of 40 keeps ~30% of the achievable score; forfeit at ply 5
    keeps ~6%.

PlayStrength is the mean of game_score across N games. Bounded [0, 1];
higher is better.

Default config:
    - retry mode with max_retries = 10 (so games have many chances to complete)
    - QUALITY_DECAY_CONSTANT = 150 (puts grandmaster-level play at ~0.7 and
      club-level at ~0.35; leaves real headroom above current matrix top)
    - Phase weights 1/1.5/2/3 at ply boundaries 10/20/30
    - Stockfish skill 3, max_moves 40

The metric is bounded [0, 1]; Stockfish self-play would score 1.0. Existing
games.jsonl data does not need re-collection when this scoring is updated —
only re-scoring.

A supplemental metric `MoveQuality` (see `move_quality.py`) removes the
retry_cost factor — useful for asking "how good are the moves themselves
once a legal one is found?" — but PlayStrength is the headline composite.
"""
from __future__ import annotations

import math
from pathlib import Path

from ..types import GameRecord, MoveRecord

CP_LOSS_CAP = 1000  # retained for legacy compatibility on capped reporting
QUALITY_DECAY_CONSTANT = 150.0
RETRY_COST_BASE = 0.25
DEFAULT_MAX_RETRIES = 10


def _game_phase_weight(ply: int) -> float:
    """Softened phase weights — late game still counts more, but not geometrically.
    Endgame at 3× opening so scores aren't dominated by whether the model
    reaches ply 30+. Still encodes the memorization-cliff thesis (later
    plies are progressively more novel from training) but with a less
    punishing denominator for models that break down in middlegame.
    """
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
    """Exponential decay quality. cp_loss in centipawns vs Stockfish's best."""
    return math.exp(-max(0, cp_loss) / QUALITY_DECAY_CONSTANT)


def _retry_cost_multiplier(retries_used: int, base: float = RETRY_COST_BASE) -> float:
    """Multiplier applied to a move's score based on how many retries it took."""
    return base ** max(0, retries_used)


def _per_move_score(move: MoveRecord) -> float:
    """Per-move score: move_quality × retry_cost_multiplier × game_phase_weight.

    Returns 0 for illegal moves (retries exhausted and game forfeited).
    """
    if not move.chosen_legal:
        return 0.0
    move_quality = _move_quality_from_cp_loss(move.cp_loss)
    retry_cost = _retry_cost_multiplier(move.retries_used)
    phase_weight = _game_phase_weight(move.ply)
    return move_quality * retry_cost * phase_weight


def play_strength(
    games: list[GameRecord],
    max_plies: int = 40,
) -> dict:
    """Compute PlayStrength and component diagnostics across a set of games."""
    if not games:
        return {
            "play_strength": 0.0,
            "n_games": 0,
            "survival_component_mean": 0.0,
            "quality_component_mean": 0.0,
            "retry_cost_mean": 0.0,
            "mean_plies_legal": 0.0,
            "mean_retries_per_move": 0.0,
            "completion_rate": 0.0,
            "per_game_scores": [],
        }

    per_game = []
    survivals = []
    mean_qualities = []
    mean_retry_costs = []
    total_retries = 0
    total_legal_moves = 0
    completions = 0
    max_weighted_score = _max_possible_weighted_score(max_plies)

    for g in games:
        legal_moves = [m for m in g.moves if m.chosen_legal]
        plies_legal = len(legal_moves)
        survival = min(plies_legal, max_plies) / max_plies

        if legal_moves:
            # Diagnostic means (still useful for the per-cell scorecard).
            mean_quality = sum(_move_quality_from_cp_loss(m.cp_loss) for m in legal_moves) / len(legal_moves)
            mean_retry_cost = sum(_retry_cost_multiplier(m.retries_used) for m in legal_moves) / len(legal_moves)
            total_retries += sum(m.retries_used for m in legal_moves)
            total_legal_moves += len(legal_moves)
            # Composite per-game score: weighted sum of legal-move scores
            # divided by the maximum achievable weighted score for the
            # game's max_plies. Unplayed (post-forfeit) plies contribute 0
            # to numerator but their phase weight is still in denominator,
            # so forfeit penalty scales with how late in the game it
            # happened (later forfeit costs more high-weight plies).
            weighted_sum = sum(_per_move_score(m) for m in legal_moves)
            score = weighted_sum / max_weighted_score if max_weighted_score else 0.0
        else:
            mean_quality = 0.0
            mean_retry_cost = 0.0
            score = 0.0

        per_game.append(score)
        survivals.append(survival)
        mean_qualities.append(mean_quality)
        mean_retry_costs.append(mean_retry_cost)
        if plies_legal >= max_plies or g.result in ("1-0", "0-1", "1/2-1/2"):
            completions += 1

    n = len(games)
    return {
        "play_strength": sum(per_game) / n,
        "n_games": n,
        "survival_component_mean": sum(survivals) / n,
        "quality_component_mean": sum(mean_qualities) / n,
        "retry_cost_mean": sum(mean_retry_costs) / n,
        "mean_plies_legal": sum(s * max_plies for s in survivals) / n,
        "mean_retries_per_move": (total_retries / total_legal_moves) if total_legal_moves else 0.0,
        "completion_rate": completions / n,
        "per_game_scores": per_game,
    }


def load_games(jsonl_path: Path) -> list[GameRecord]:
    out: list[GameRecord] = []
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(GameRecord.model_validate_json(line))
    return out
