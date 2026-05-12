"""Compile the final v2 cross-family matrix from all benchmark runs.

Walks runs/ and emits a per-(provider, tier) row with CR (final formula),
PS, and ACPL by phase. Uses the *latest* clean run per model. Marks any
model that still shows 'did not call submit_move' errors as suspect.

Intended to be run after the three pending re-runs (OpenAI, Gemini Pro,
DeepSeek) complete; the script picks the most recent run per model so
older buggy data is automatically superseded.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from llm_chess_eval.config import BENCHMARK_MATRIX
from llm_chess_eval.evals.chess_reliability import chess_reliability, load_games
from llm_chess_eval.evals.play_strength import compute_play_strength

RUNS = Path("C:/Users/igorc/Projects/LLM_Chess_Eval/runs")


def find_latest_run(model: str, mode: str, skill: int) -> Path | None:
    """Return the most-recent games.jsonl for a (model, mode, skill) triple."""
    pattern_parts = f"games_{mode}__{model}__skill{skill}"
    candidates = [p for p in sorted(RUNS.iterdir()) if pattern_parts in p.name]
    if not candidates:
        return None
    latest = candidates[-1]
    jsonl = latest / "games.jsonl"
    return jsonl if jsonl.is_file() else None


def has_token_exhaustion_errors(jsonl: Path) -> int:
    """Count 'did not call submit_move' errors in a games run."""
    if not jsonl or not jsonl.is_file():
        return -1
    return sum(1 for line in jsonl.read_text(encoding="utf-8").splitlines()
               if "did not call submit_move" in line)


def main() -> None:
    print(f"{'provider':<10} {'tier':<10} {'model':<55} {'CR':>6} {'PS':>6} "
          f"{'op':>6} {'mid':>6} {'end':>6} {'bug?':>5}")
    print("-" * 120)

    rows: list[dict[str, Any]] = []

    for provider, tiers in BENCHMARK_MATRIX.items():
        for tier, model in tiers.items():
            cr_jsonl = find_latest_run(model, "retry", 3) or find_latest_run(model, "forfeit", 3)
            ps_jsonl = find_latest_run(model, "retry", 5)

            row: dict[str, Any] = {"provider": provider, "tier": tier, "model": model}

            if cr_jsonl is not None:
                games = load_games(cr_jsonl)
                cr_res = chess_reliability(games, max_plies=40)
                row["CR"] = cr_res["chess_reliability"]
                row["CR_source"] = cr_jsonl.parent.name
                row["CR_errors"] = has_token_exhaustion_errors(cr_jsonl)
            else:
                row["CR"] = None
                row["CR_source"] = "no run found"
                row["CR_errors"] = -1

            if ps_jsonl is not None:
                games = load_games(ps_jsonl)
                ps_res = compute_play_strength(games, max_plies=60)
                row["PS"] = ps_res["play_strength"]
                row["ACPL_op"] = ps_res["acpl_opening"]
                row["ACPL_mid"] = ps_res["acpl_middlegame"]
                row["ACPL_end"] = ps_res["acpl_endgame"]
                row["PS_source"] = ps_jsonl.parent.name
                row["PS_errors"] = has_token_exhaustion_errors(ps_jsonl)
            else:
                row["PS"] = None
                row["ACPL_op"] = row["ACPL_mid"] = row["ACPL_end"] = None
                row["PS_source"] = "no run found"
                row["PS_errors"] = -1

            rows.append(row)

            cr_str = f"{row['CR']:.3f}" if row['CR'] is not None else "  -  "
            ps_str = f"{row['PS']:.3f}" if row['PS'] is not None else "  -  "
            op = f"{row['ACPL_op']:.0f}" if row.get('ACPL_op') is not None else "-"
            mid = f"{row['ACPL_mid']:.0f}" if row.get('ACPL_mid') is not None else "-"
            end = f"{row['ACPL_end']:.0f}" if row.get('ACPL_end') is not None else "-"
            bug = "YES" if (row['CR_errors'] > 0 or row['PS_errors'] > 0) else "no"

            print(f"{provider:<10} {tier:<10} {model:<55} "
                  f"{cr_str:>6} {ps_str:>6} {op:>6} {mid:>6} {end:>6} {bug:>5}")

    print("\nProvenance:")
    for r in rows:
        print(f"  {r['provider']}/{r['tier']} {r['model']}")
        print(f"    CR from: {r['CR_source']} (token-bug errors: {r['CR_errors']})")
        print(f"    PS from: {r['PS_source']} (token-bug errors: {r['PS_errors']})")

    # Emit JSON for downstream automation
    out_path = Path("scripts/v2_matrix_snapshot.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")
    print(f"\nJSON snapshot written to: {out_path}")


if __name__ == "__main__":
    main()
