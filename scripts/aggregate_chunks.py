"""Aggregate chunked games.jsonl files into a single virtual dataset for scoring.

For each model+skill pair, finds all run dirs in a given time window and
concatenates their games.jsonl files (filtering out quota-corrupted games).
Outputs a combined virtual run dir suitable for the matrix script.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = PROJECT_ROOT / "runs"
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from llm_chess_eval.evals.play_strength import play_strength, load_games
from llm_chess_eval.evals.move_quality import move_quality


def is_clean_game(g: dict) -> bool:
    """A game is clean if no move has a quota/credit/429 error."""
    for m in g.get("moves", []):
        e = (m.get("model_error") or "").lower()
        if "429" in e or "quota" in e or "credit balance" in e or "insufficient_quota" in e:
            return False
    return True


def aggregate(model: str, skill: int, since: str = "20260514T020000Z") -> tuple[Path | None, int, int]:
    """Concatenate all clean games from runs in (model, skill) after `since`.
    Returns (output_path, n_clean_games, n_quota_corrupt_games_skipped)."""
    pattern = f"*__games_retry__{model}__skill{skill}"
    candidates = sorted(RUNS_DIR.glob(pattern))
    clean_games_text: list[str] = []
    n_clean = 0
    n_corrupt = 0
    for d in candidates:
        if d.name < since:
            continue
        f = d / "games.jsonl"
        if not f.exists():
            continue
        try:
            with f.open(encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        g = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if is_clean_game(g):
                        clean_games_text.append(line)
                        n_clean += 1
                    else:
                        n_corrupt += 1
        except Exception:
            continue

    if not clean_games_text:
        return None, 0, n_corrupt

    out_dir = RUNS_DIR / f"AGGREGATED__games_retry__{model}__skill{skill}"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "games.jsonl").write_text("\n".join(clean_games_text) + "\n", encoding="utf-8")
    return out_dir, n_clean, n_corrupt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=[
        "claude-opus-4-7", "claude-haiku-4-5-20251001", "claude-sonnet-4-6",
        "gpt-5", "gpt-5-mini",
        "gemini-2.5-pro", "gemini-3.1-flash-lite",
        "deepseek-reasoner", "deepseek-chat",
    ])
    ap.add_argument("--since", default="20260514T020000Z",
                    help="Only include runs after this timestamp (default: 2026-05-14 02:00 UTC = start of 20/10 bump)")
    args = ap.parse_args()

    print(f"{'model':<32} | {'metric':<12} | {'N clean':>8} | {'N corrupt':>10} | {'score':>6}")
    print("-" * 80)
    for model in args.models:
        for label, skill, max_plies in [("PlayStrength", 3, 40), ("MoveQuality", 5, 60)]:
            out_dir, n_clean, n_corrupt = aggregate(model, skill, args.since)
            if not out_dir or n_clean == 0:
                print(f"{model:<32} | {label:<12} | {'-':>8} | {n_corrupt:>10} | {'-':>6}")
                continue
            games = load_games(out_dir / "games.jsonl")
            if label == "PlayStrength":
                r = play_strength(games, max_plies=max_plies)
                score = r["play_strength"]
            else:
                r = move_quality(games, max_plies=max_plies)
                score = r["move_quality"]
            print(f"{model:<32} | {label:<12} | {n_clean:>8} | {n_corrupt:>10} | {score:>6.3f}")


if __name__ == "__main__":
    main()
