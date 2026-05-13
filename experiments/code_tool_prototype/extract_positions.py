"""Walk CR run logs and extract FENs where the model proposed an illegal move.

Output is a JSONL position bank that can be replayed against any model:
each row has (fen_before, ground_truth_baseline_chosen_san, source_run, ply).
By construction these FENs emerged from real Stockfish play in games against
random openings, so they are provably outside any LLM's training data —
unlike the hand-curated bank in data/positions/legality_v1.jsonl which is
saturated with named openings and textbook endgames.

Usage:
    python scripts/extract_illegal_positions.py [--model claude-opus-4-7] [--out PATH]

Default: scans every CR run dir and emits data/positions/illegal_replay.jsonl
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = PROJECT_ROOT / "runs"
DEFAULT_OUT = Path(__file__).resolve().parent / "replay_positions.jsonl"


def extract_from_run(run_dir: Path, model_filter: str | None) -> list[dict]:
    """Each game in games.jsonl has moves[].  A move is an 'illegal' instance if
    chosen_san is set but chosen_legal is False, OR if failed_attempts has any
    entries.  We record one position per ILLEGAL ATTEMPT (not per move) — i.e.
    the position-before-move plus the SAN that was rejected.
    """
    games_file = run_dir / "games.jsonl"
    if not games_file.exists():
        return []

    out: list[dict] = []
    for line in games_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        g = json.loads(line)
        model = g.get("model", "")
        if model_filter and model != model_filter:
            continue
        for m in g.get("moves", []) or []:
            fen = m.get("fen_before")
            if not fen:
                continue
            # Top-level chosen move (after any retries) — if illegal, it failed even with retries.
            if m.get("chosen_san") and not m.get("chosen_legal"):
                out.append({
                    "position_id": f"{run_dir.name[:18]}__g{g.get('game_id','?')[-4:]}__ply{m.get('ply')}__final",
                    "fen": fen,
                    "baseline_model": model,
                    "baseline_illegal_san": m["chosen_san"],
                    "baseline_legal_in_failed": [str(f) for f in (m.get("failed_attempts") or [])][:5],
                    "source_run": run_dir.name,
                    "source_game": g.get("game_id"),
                    "source_ply": m.get("ply"),
                    "notes": "Position from real Stockfish playthrough — provably out-of-distribution.",
                })
            # Earlier failed attempts (before the model eventually succeeded or forfeited).
            for fa in (m.get("failed_attempts") or []):
                if isinstance(fa, str) and fa:
                    out.append({
                        "position_id": f"{run_dir.name[:18]}__g{g.get('game_id','?')[-4:]}__ply{m.get('ply')}__retry",
                        "fen": fen,
                        "baseline_model": model,
                        "baseline_illegal_san": fa,
                        "source_run": run_dir.name,
                        "source_game": g.get("game_id"),
                        "source_ply": m.get("ply"),
                        "notes": "Position from real Stockfish playthrough — provably out-of-distribution.",
                    })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="Only extract from games played by this baseline model.")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--dedupe-fens", action="store_true",
                    help="Keep only the first illegal entry per unique FEN (sharper test).")
    args = ap.parse_args()

    rows: list[dict] = []
    for run_dir in sorted(RUNS_DIR.glob("*__games_retry__*__skill*")):
        rows.extend(extract_from_run(run_dir, args.model))

    if args.dedupe_fens:
        seen = set()
        dedup = []
        for r in rows:
            if r["fen"] in seen:
                continue
            seen.add(r["fen"])
            dedup.append(r)
        rows = dedup

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    by_model = {}
    for r in rows:
        by_model[r["baseline_model"]] = by_model.get(r["baseline_model"], 0) + 1
    print(f"wrote {len(rows)} positions to {args.out}")
    for m, c in sorted(by_model.items(), key=lambda kv: -kv[1]):
        print(f"  {m}: {c}")


if __name__ == "__main__":
    main()
