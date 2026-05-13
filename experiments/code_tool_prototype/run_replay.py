"""Replay a position bank against baseline AND pytool variants of a model.

For each position, we call the model directly via build_adapter (no Stockfish
in the loop) and record:
  - chosen_san, chosen_legal (the move actually legal in this FEN)
  - the raw tool input (so we can see candidates and claims)
  - latency / token counts (pytool with sandbox invocation will be much higher)
  - source provenance (which game/ply baseline failed on)

Compares baseline vs pytool per-position. The key signals:
  1. Does pytool flip a previously-illegal position to legal?
  2. Does pytool invoke the python_exec sandbox (visible as multi-call latency)?
  3. Where pytool fails too — same SAN as baseline (persistent wrong belief) or different?

Output:
  results_<model>_baseline.jsonl
  results_<model>_pytool.jsonl
  summary.md
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

EXPDIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPDIR.parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import chess
from llm_chess_eval.adapters.factory import build_adapter


def run_one(adapter, fen: str) -> dict:
    board = chess.Board(fen)
    t0 = time.perf_counter()
    outcome = adapter.propose_move(fen)
    elapsed_ms = int((time.perf_counter() - t0) * 1000)

    chosen_san = None
    chosen_legal = None
    raw = outcome.raw_tool_input or {}
    if raw and "chosen_move" in raw:
        chosen_san = raw["chosen_move"]
        try:
            move = board.parse_san(chosen_san)
            chosen_legal = move in board.legal_moves
        except Exception:
            chosen_legal = False

    return {
        "chosen_san": chosen_san,
        "chosen_legal": chosen_legal,
        "error": outcome.error,
        "latency_ms": elapsed_ms,
        "input_tokens": outcome.input_tokens,
        "output_tokens": outcome.output_tokens,
        "raw_tool_input": raw,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", type=Path, default=EXPDIR / "replay_positions.jsonl")
    ap.add_argument("--base-model", default="claude-opus-4-7",
                    help="The model id to run BOTH as baseline and as -pytool variant.")
    ap.add_argument("--n", type=int, default=20, help="Limit to first N positions (mid-game biased order).")
    ap.add_argument("--filter-mid", action="store_true", default=True,
                    help="Prefer mid-game positions (ply 10-24). Default on.")
    ap.add_argument("--variant", choices=["baseline", "pytool", "both"], default="both")
    args = ap.parse_args()

    rows = [json.loads(l) for l in args.bank.read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.filter_mid:
        rows = [r for r in rows if 10 <= (r.get("source_ply") or 0) < 25]
    rows = rows[: args.n]
    print(f"Running on {len(rows)} positions")

    variants = ["baseline", "pytool"] if args.variant == "both" else [args.variant]
    for variant in variants:
        model_id = args.base_model + ("-pytool" if variant == "pytool" else "")
        print(f"\n=== variant: {variant} ({model_id}) ===")
        adapter = build_adapter(model_id, max_tokens=4096)
        out_path = EXPDIR / f"results_{args.base_model}_{variant}.jsonl"
        n_legal = 0
        n_illegal = 0
        n_error = 0
        with out_path.open("w", encoding="utf-8") as f:
            for i, r in enumerate(rows, 1):
                res = run_one(adapter, r["fen"])
                rec = {
                    "position_id": r["position_id"],
                    "fen": r["fen"],
                    "baseline_illegal_san": r["baseline_illegal_san"],
                    "source_run": r["source_run"],
                    "variant": variant,
                    **res,
                }
                f.write(json.dumps(rec) + "\n")
                f.flush()
                if res["error"]:
                    n_error += 1
                    flag = "ERR"
                elif res["chosen_legal"]:
                    n_legal += 1
                    flag = "OK "
                else:
                    n_illegal += 1
                    flag = "ILL"
                chosen_str = (res.get("chosen_san") or "-")
                print(f"  {i:2d}/{len(rows)} {flag} {chosen_str:<8} "
                      f"(was illegal: {r['baseline_illegal_san']:<8}) "
                      f"latency={res['latency_ms']/1000:.1f}s tok_out={res['output_tokens']}"
                      + (f"  err={res['error'][:60]}" if res.get("error") else ""))
        print(f"  -> legal: {n_legal}  illegal: {n_illegal}  error: {n_error}")


if __name__ == "__main__":
    main()
