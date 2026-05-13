"""Score the matrix at multiple metric variants for comparison.

Variants computed for both CR and PS:
  - "linear" (current):  quality = 1 − cp_loss/1000,   no phase weight
  - "exp/100":           quality = exp(-cp_loss/100),  + phase weight
  - "exp/150":           quality = exp(-cp_loss/150),  + phase weight
  - "exp/200":           quality = exp(-cp_loss/200),  + phase weight

Phase weight (applied to all exp variants):
  - ply 1-9:    1.0  (opening, in-distribution training data)
  - ply 10-24:  1.5  (mid-game, partially OOD)
  - ply 25+:    2.0  (late mid / endgame, fully OOD per memorization cliff)

Per-game score under new formula:
  num = sum over legal moves of [quality(cp_loss) * 0.25^retries * phase_weight(ply)]
  den = sum over ply in 1..max_plies of phase_weight(ply)   (full-game potential)
  per_game = num / den

This denominator structure penalizes forfeits more sharply because unplayed
late plies have higher phase weight. Under the current "survival × quality"
formula, those losses are just linear; here, losing a ply-30 move costs 2×
what losing a ply-5 move does.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = PROJECT_ROOT / "runs"
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def phase_weight_gentle(ply: int) -> float:
    """1 / 1.5 / 2 — original proposal, modest late-game bonus."""
    if ply < 10:
        return 1.0
    if ply < 25:
        return 1.5
    return 2.0


def phase_weight_geometric(ply: int) -> float:
    """1 / 2 / 4 / 8 — each phase doubles. Aligns with memorization-cliff
    thesis: opening (0% novel) → mid (50%) → endgame-start (75%) → deep
    endgame (100% novel from training data)."""
    if ply < 10:
        return 1.0
    if ply < 20:
        return 2.0
    if ply < 30:
        return 4.0
    return 8.0


def phase_weight_cubic(ply: int) -> float:
    """1 / 3 / 9 — even steeper, late-game dominates score."""
    if ply < 10:
        return 1.0
    if ply < 25:
        return 3.0
    return 9.0


# Active phase weight — switch by import
phase_weight = phase_weight_gentle


def max_possible_weighted(max_plies: int) -> float:
    return sum(phase_weight(p) for p in range(1, max_plies + 1))


def linear_quality(cp_loss: int) -> float:
    loss = max(0, min(cp_loss, 1000))
    return 1.0 - loss / 1000


def exp_quality(cp_loss: int, tau: float) -> float:
    loss = max(0, cp_loss)
    return math.exp(-loss / tau)


def retry_penalty(retries: int) -> float:
    return 0.25 ** max(0, retries)


def score_game_linear(moves: list[dict], max_plies: int) -> float:
    """Current formula: survival × mean(quality × retry_penalty)."""
    legal = [m for m in moves if m.get("chosen_legal")]
    if not legal:
        return 0.0
    plies_legal = len(legal)
    survival = min(plies_legal, max_plies) / max_plies
    move_score_mean = sum(
        linear_quality(m.get("cp_loss", 0) or 0) * retry_penalty(m.get("retries_used", 0) or 0)
        for m in legal
    ) / len(legal)
    return survival * move_score_mean


def score_game_exp_phase(moves: list[dict], max_plies: int, tau: float) -> float:
    """New formula: sum(quality_exp × retry × phase_weight) / max_weighted_possible."""
    num = 0.0
    for m in moves:
        if not m.get("chosen_legal"):
            continue
        ply = m.get("ply") or 0
        cp = m.get("cp_loss", 0) or 0
        r = m.get("retries_used", 0) or 0
        num += exp_quality(cp, tau) * retry_penalty(r) * phase_weight(ply)
    den = max_possible_weighted(max_plies)
    return num / den


def load_games(run_dir: Path) -> list[dict]:
    f = run_dir / "games.jsonl"
    if not f.exists():
        return []
    return [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]


def find_latest_run(model: str, skill: int) -> Path | None:
    pattern = f"*__games_retry__{model}__skill{skill}"
    matches = sorted(RUNS_DIR.glob(pattern))
    return matches[-1] if matches else None


def score_with(games: list[dict], max_plies: int, tau: float, pw_fn) -> float:
    """Helper: temporarily install a phase_weight function and score."""
    global phase_weight
    saved = phase_weight
    phase_weight = pw_fn
    try:
        n = len(games)
        return sum(score_game_exp_phase(g["moves"], max_plies, tau) for g in games) / n
    finally:
        phase_weight = saved


def score_run(games: list[dict], max_plies: int) -> dict:
    if not games:
        return {}
    n = len(games)
    lin = sum(score_game_linear(g["moves"], max_plies) for g in games) / n
    return {
        "linear": lin,
        "gentle_1_1.5_2": score_with(games, max_plies, 150, phase_weight_gentle),
        "geometric_1_2_4_8": score_with(games, max_plies, 150, phase_weight_geometric),
        "cubic_1_3_9": score_with(games, max_plies, 150, phase_weight_cubic),
    }


MODELS = [
    "gemini-3.1-flash-lite",
    "gemini-2.5-pro",
    "gpt-5",
    "deepseek-reasoner",
    "claude-opus-4-7",
    "deepseek-chat",
    "claude-haiku-4-5-20251001",
    "gpt-5-mini",
]


def main() -> None:
    print(f"All exp variants use quality = exp(-cp_loss/150), retry_penalty = 0.25^n")
    print(f"Phase-weight schemes differ:")
    print(f"  gentle:    ply<10=1.0,  ply<25=1.5,  ply>=25=2.0")
    print(f"  geometric: ply<10=1.0,  ply<20=2.0,  ply<30=4.0,  ply>=30=8.0")
    print(f"  cubic:     ply<10=1.0,  ply<25=3.0,  ply>=25=9.0")
    print()
    print(f"{'model':<32} | {'kind':<3} | {'linear':>7} | {'gentle':>7} | {'geom':>7} | {'cubic':>7}")
    print("-" * 80)
    for model in MODELS:
        for kind, skill, max_plies in [("CR", 3, 40), ("PS", 5, 60)]:
            run = find_latest_run(model, skill)
            if not run:
                continue
            games = load_games(run)
            if not games:
                continue
            err_429 = sum(
                1 for g in games for m in g.get("moves", [])
                if "429" in (m.get("model_error") or "")
            )
            total_moves = sum(1 for g in games for _ in g.get("moves", []))
            if err_429 and err_429 == total_moves:
                print(f"{model:<32} | {kind:<3} | quota-corrupt run, skipped")
                continue
            r = score_run(games, max_plies)
            print(f"{model:<32} | {kind:<3} | "
                  f"{r['linear']:>7.3f} | {r['gentle_1_1.5_2']:>7.3f} | "
                  f"{r['geometric_1_2_4_8']:>7.3f} | {r['cubic_1_3_9']:>7.3f}")


if __name__ == "__main__":
    main()
