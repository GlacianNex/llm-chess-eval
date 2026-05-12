"""ChessReliability (CR) — composite metric for rule-following ability.

CR rewards models for producing legal moves cheaply (few or no retries) and
penalizes those that need many attempts to find a legal move, even if they
eventually do. This makes the metric more sensitive than the original
forfeit-on-first-illegal version: a model that needs 4 retries every move
scores well below a model that needs 0, even though both produce legal play.

Per-move score:
    move_score = quality × retry_penalty
    quality       = 1 − clamp(cp_loss, 0, 1000) / 1000
    retry_penalty = RETRY_PENALTY_BASE ^ retries_used

Per-game score:
    game_score = (legal_moves_played / max_moves) × mean(move_score)

CR is the mean of game_score across N games.

Default config:
    - retry mode with max_retries = 10 (so games have many chances to complete)
    - RETRY_PENALTY_BASE = 0.5  (1 retry → 0.5, 3 retries → 0.125, 10 retries → 0.001)
    - Stockfish skill 3, max_moves 40

The combination of "many tries, steep decay" lets weak models complete games
(so we see their late-game move quality) while ensuring retry-heavy moves
contribute almost nothing to the score. A model needing 5+ retries on every
move scores essentially zero even though the game runs long.

Why retries-with-penalty (replaces forfeit-on-first-illegal):
The original forfeit version of CR scored very weak models (e.g., DeepSeek-chat
forfeiting at move 1 in 4 of 5 games) at essentially zero, losing the resolution
to differentiate them from a model that forfeits at move 0. By allowing retries
with a steep per-retry penalty, the metric preserves resolution at the low end
while still firmly punishing models that produce illegal moves frequently.
Bounded [0, 1]; higher is better.
"""
from __future__ import annotations

from pathlib import Path

from ..types import GameRecord, MoveRecord

CP_LOSS_CAP = 1000
RETRY_PENALTY_BASE = 0.5
DEFAULT_MAX_RETRIES = 10


def _quality_cp(cp_loss: int) -> float:
    loss = max(0, cp_loss)
    if loss >= CP_LOSS_CAP:
        return 0.0
    return 1.0 - loss / CP_LOSS_CAP


def _retry_penalty(retries_used: int, base: float = RETRY_PENALTY_BASE) -> float:
    return base ** max(0, retries_used)


def _move_score(move: MoveRecord) -> float:
    """Per-move score combining cp_loss quality and retry penalty.

    Returns 0 for illegal moves (retries exhausted and game forfeited).
    """
    if not move.chosen_legal:
        return 0.0
    q = _quality_cp(move.cp_loss)
    r = _retry_penalty(move.retries_used)
    return q * r


def chess_reliability(
    games: list[GameRecord],
    max_plies: int = 40,
) -> dict:
    """Compute CR and component diagnostics across a set of games."""
    if not games:
        return {
            "chess_reliability": 0.0,
            "n_games": 0,
            "survival_component_mean": 0.0,
            "quality_component_mean": 0.0,
            "retry_penalty_mean": 0.0,
            "mean_plies_legal": 0.0,
            "mean_retries_per_move": 0.0,
            "completion_rate": 0.0,
            "per_game_scores": [],
        }

    per_game = []
    survivals = []
    qualities = []
    retry_penalties = []
    total_retries = 0
    total_legal_moves = 0
    completions = 0

    for g in games:
        legal_moves = [m for m in g.moves if m.chosen_legal]
        plies_legal = len(legal_moves)
        survival = min(plies_legal, max_plies) / max_plies

        if legal_moves:
            quality = sum(_quality_cp(m.cp_loss) for m in legal_moves) / len(legal_moves)
            penalty = sum(_retry_penalty(m.retries_used) for m in legal_moves) / len(legal_moves)
            move_score_mean = sum(_move_score(m) for m in legal_moves) / len(legal_moves)
            total_retries += sum(m.retries_used for m in legal_moves)
            total_legal_moves += len(legal_moves)
        else:
            quality = 0.0
            penalty = 0.0
            move_score_mean = 0.0

        score = survival * move_score_mean
        per_game.append(score)
        survivals.append(survival)
        qualities.append(quality)
        retry_penalties.append(penalty)
        if plies_legal >= max_plies or g.result in ("1-0", "0-1", "1/2-1/2"):
            completions += 1

    n = len(games)
    return {
        "chess_reliability": sum(per_game) / n,
        "n_games": n,
        "survival_component_mean": sum(survivals) / n,
        "quality_component_mean": sum(qualities) / n,
        "retry_penalty_mean": sum(retry_penalties) / n,
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
