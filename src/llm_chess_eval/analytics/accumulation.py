"""Error-accumulation analytics over played games.

Answers:
  1. Per-ply error rate: at ply N (over all games), what fraction of moves had
     issue X? Issues we track: illegal chosen, blunder >=300 cp, cands_legal<1.0,
     claim_consistency<1.0, chose_not_top.
  2. Survival: at ply N, what fraction of games are still alive (not forfeited)?
  3. Blunder autocorrelation: P(blunder at N+1 | blunder at N) vs P(blunder at N+1 | clean at N).
     Conditioned on both plies being from the same game and both being legal moves.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from ..types import GameRecord, MoveRecord


def load_games(jsonl_path: Path) -> list[GameRecord]:
    out: list[GameRecord] = []
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        out.append(GameRecord.model_validate_json(line))
    return out


def per_ply_table(games: Iterable[GameRecord]) -> list[dict]:
    """Bucket moves by ply across all games; compute per-ply rates."""
    by_ply: dict[int, list[MoveRecord]] = {}
    for g in games:
        for m in g.moves:
            by_ply.setdefault(m.ply, []).append(m)
    rows = []
    for ply in sorted(by_ply):
        moves = by_ply[ply]
        n = len(moves)
        illegal = sum(1 for m in moves if not m.chosen_legal)
        legal = [m for m in moves if m.chosen_legal]
        blunder_300 = sum(1 for m in legal if m.cp_loss >= 300)
        blunder_1000 = sum(1 for m in legal if m.cp_loss >= 1000)
        mean_cp_loss = sum(m.cp_loss for m in legal) / len(legal) if legal else float("nan")
        cands_under_1 = sum(1 for m in moves if m.candidates_legal_rate < 1.0)
        claim_imperfect = sum(1 for m in legal if m.claim_consistency < 1.0)
        chose_top = sum(1 for m in legal if m.chosen_was_top)
        rows.append({
            "ply": ply,
            "n_moves": n,
            "illegal_rate": illegal / n,
            "blunder_rate_300": (blunder_300 / len(legal)) if legal else float("nan"),
            "blunder_rate_1000": (blunder_1000 / len(legal)) if legal else float("nan"),
            "mean_cp_loss": mean_cp_loss,
            "cands_imperfect_rate": cands_under_1 / n,
            "claim_imperfect_rate": (claim_imperfect / len(legal)) if legal else float("nan"),
            "chose_top_rate": (chose_top / len(legal)) if legal else float("nan"),
        })
    return rows


def survival_table(games: Iterable[GameRecord]) -> list[dict]:
    """For each ply, what fraction of games are still being played?

    "Alive" = game has not terminated by this ply. In substitute mode, an illegal
    chosen move does NOT end the game (Stockfish-best is played), so the game
    is still alive even when chosen_legal=False. We also report `legal_rate`
    separately: among games alive at ply N, fraction where the model's chosen
    move was legal.
    """
    glist = list(games)
    n_games = len(glist)
    # A game is "alive at ply N" iff (its max ply >= N) OR it forfeited exactly at ply N.
    # Both forfeit and natural game-over after ply N stop the model from playing more.
    rows = []
    max_ply = max((g.n_plies for g in glist), default=0)
    for ply in range(1, max_ply + 1):
        alive = 0
        legal_at_ply = 0
        for g in glist:
            move_at_ply = next((m for m in g.moves if m.ply == ply), None)
            if move_at_ply is None:
                continue
            alive += 1
            if move_at_ply.chosen_legal:
                legal_at_ply += 1
        rows.append({
            "ply": ply,
            "alive_games": alive,
            "alive_fraction": alive / n_games,
            "legal_rate_among_alive": (legal_at_ply / alive) if alive else 0.0,
        })
    return rows


def blunder_autocorrelation(games: Iterable[GameRecord], threshold_cp: int = 300) -> dict:
    """For consecutive (legal,legal) ply pairs within the same game:
      P(blunder at N+1 | blunder at N)
      P(blunder at N+1 | clean at N)
    Where blunder = cp_loss >= threshold_cp.
    """
    pairs = {"after_blunder": [], "after_clean": []}
    for g in games:
        legal_in_order = sorted([m for m in g.moves if m.chosen_legal], key=lambda x: x.ply)
        for prev, nxt in zip(legal_in_order, legal_in_order[1:]):
            prev_blunder = prev.cp_loss >= threshold_cp
            nxt_blunder = nxt.cp_loss >= threshold_cp
            if prev_blunder:
                pairs["after_blunder"].append(nxt_blunder)
            else:
                pairs["after_clean"].append(nxt_blunder)

    def rate(lst: list[bool]) -> float:
        return sum(lst) / len(lst) if lst else float("nan")

    return {
        "threshold_cp": threshold_cp,
        "n_after_blunder": len(pairs["after_blunder"]),
        "n_after_clean": len(pairs["after_clean"]),
        "p_blunder_after_blunder": rate(pairs["after_blunder"]),
        "p_blunder_after_clean": rate(pairs["after_clean"]),
    }


def summarize_run(jsonl_path: Path, model_label: str | None = None) -> None:
    """Print a full accumulation summary for one games.jsonl file."""
    games = load_games(jsonl_path)
    if not games:
        print(f"(no games in {jsonl_path})")
        return
    model_label = model_label or games[0].model
    print(f"\n{'='*72}\n{model_label}  ({len(games)} games)\n{'='*72}")

    print(f"\nResults: " + ", ".join(f"{g.result}({g.n_plies})" for g in games))

    print("\nPer-ply rates (across all games):")
    print(f"  {'ply':>3} {'n':>3} {'illegal':>8} {'cp_loss':>8} {'b300':>6} {'cands<1':>8} {'claim<1':>8} {'top':>6}")
    for r in per_ply_table(games):
        cp = f"{r['mean_cp_loss']:.0f}" if r['mean_cp_loss'] == r['mean_cp_loss'] else "-"
        b3 = f"{r['blunder_rate_300']:.2f}" if r['blunder_rate_300'] == r['blunder_rate_300'] else "-"
        ci = f"{r['claim_imperfect_rate']:.2f}" if r['claim_imperfect_rate'] == r['claim_imperfect_rate'] else "-"
        tp = f"{r['chose_top_rate']:.2f}" if r['chose_top_rate'] == r['chose_top_rate'] else "-"
        print(
            f"  {r['ply']:>3} {r['n_moves']:>3} {r['illegal_rate']:>8.2f} "
            f"{cp:>8} {b3:>6} {r['cands_imperfect_rate']:>8.2f} {ci:>8} {tp:>6}"
        )

    print("\nSurvival (alive = game still being played; legal_rate = of alive games, fraction where model proposed legally):")
    for r in survival_table(games):
        bar = "#" * int(r["alive_fraction"] * 20)
        print(f"  ply {r['ply']:>3}: alive={r['alive_fraction']:.2f}  legal_rate={r['legal_rate_among_alive']:.2f}  {bar}")

    for thresh in (200, 300, 500):
        ac = blunder_autocorrelation(games, threshold_cp=thresh)
        print(f"\nBlunder autocorrelation (threshold = {thresh} cp loss):")
        print(f"  pairs after blunder: n={ac['n_after_blunder']}  P(blunder next) = {ac['p_blunder_after_blunder']:.2f}")
        print(f"  pairs after clean:   n={ac['n_after_clean']}  P(blunder next) = {ac['p_blunder_after_clean']:.2f}")


if __name__ == "__main__":
    import sys
    for path in sys.argv[1:]:
        summarize_run(Path(path))
