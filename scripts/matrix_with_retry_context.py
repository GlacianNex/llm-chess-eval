"""Augment the v2 matrix with retry-context columns so CR / PS numbers are
interpretable without reading the methodology section.

For each (model, mode) cell we compute:
  - CR or PS (the existing headline score)
  - max_retries — the rule of that mode (10 for CR, 3 for PS)
  - first_attempt_legal_rate — fraction of plies where the FIRST adapter
    call returned a legal SAN. This is the "no retry crutch" legality
    number; reads independently of how forgiving the harness is.
  - mean_retries_per_move — total retries / total plies attempted
  - games_n, plies_n — sample-size context

Reads from runs/ directly; identifies the latest run per (model, skill)
unless overridden via --runs.

Usage:
    python scripts/matrix_with_retry_context.py
    python scripts/matrix_with_retry_context.py --runs run1 run2 run3
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = PROJECT_ROOT / "runs"

sys.path.insert(0, str(PROJECT_ROOT / "src"))
from llm_chess_eval.evals.chess_reliability import chess_reliability, load_games  # noqa: E402


def load_games_dict(run_dir: Path) -> list[dict]:
    f = run_dir / "games.jsonl"
    if not f.exists():
        return []
    return [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]


def aggregate_retry_context(games: list[dict]) -> dict:
    total_plies = 0
    first_attempt_legal = 0
    total_retries = 0
    for g in games:
        for m in g.get("moves", []):
            total_plies += 1
            retries = m.get("retries_used", 0) or 0
            total_retries += retries
            # "first attempt legal" = the model's first proposal was legal,
            # i.e. it succeeded without any retry. retries_used==0 AND the
            # final chosen move was legal.
            if retries == 0 and m.get("chosen_legal"):
                first_attempt_legal += 1
    return {
        "n_plies": total_plies,
        "n_games": len(games),
        "first_attempt_legal_rate": (first_attempt_legal / total_plies) if total_plies else 0.0,
        "mean_retries_per_move": (total_retries / total_plies) if total_plies else 0.0,
    }


def find_latest_run(model: str, skill: int) -> Path | None:
    pattern = f"*__games_retry__{model}__skill{skill}"
    matches = sorted(RUNS_DIR.glob(pattern))
    return matches[-1] if matches else None


def score_run(run_dir: Path, max_plies: int, kind: str) -> dict | None:
    f = run_dir / "games.jsonl"
    if not f.exists():
        return None
    scored = load_games(f)  # GameRecord typed objects
    raw = load_games_dict(run_dir)  # raw dicts for retry-context
    if not scored:
        return None
    cr_record = chess_reliability(scored, max_plies=max_plies)
    ctx = aggregate_retry_context(raw)
    return {
        "run": run_dir.name,
        "kind": kind,
        "score": cr_record["chess_reliability"],
        "max_retries": 10 if kind == "CR" else 3,
        **ctx,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=[
        "claude-opus-4-7", "claude-haiku-4-5-20251001",
        "gpt-5", "gpt-5-mini",
        "gemini-2.5-pro", "gemini-3.1-flash-lite", "gemini-3.1-pro-preview",
        "deepseek-reasoner", "deepseek-chat",
    ])
    args = ap.parse_args()

    print(f"{'model':<35} | {'kind':<3} | {'score':>5} | {'max_ret':>7} | "
          f"{'1st-leg':>7} | {'mean_ret':>8} | {'games':>5} | {'plies':>5}")
    print("-" * 110)
    for model in args.models:
        for kind, skill, max_plies in [("CR", 3, 40), ("PS", 5, 60)]:
            run = find_latest_run(model, skill)
            if not run:
                print(f"{model:<35} | {kind:<3} | {'-':>5} | {'-':>7} | {'-':>7} | {'-':>8} | {'-':>5} | {'-':>5}  NO RUN")
                continue
            try:
                r = score_run(run, max_plies, kind)
            except Exception as e:
                print(f"{model:<35} | {kind:<3} | ERR | {type(e).__name__}: {e!s:.60}")
                continue
            if r is None:
                continue
            print(f"{model:<35} | {kind:<3} | {r['score']:.3f} | {r['max_retries']:>7} | "
                  f"{r['first_attempt_legal_rate']:.3f}   | {r['mean_retries_per_move']:>8.2f} | "
                  f"{r['n_games']:>5} | {r['n_plies']:>5}")


if __name__ == "__main__":
    main()
