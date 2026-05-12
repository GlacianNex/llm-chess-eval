"""Print failed positions and illegal candidate moves for a legality run."""
import json
import sys
from pathlib import Path

import chess


def main(*paths: str) -> None:
    for path_str in paths:
        path = Path(path_str)
        print(f"\n{'='*72}\n{path.parent.name}\n{'='*72}")
        rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        failed = [r for r in rows if r["score"] < 1.0]
        all_illegal_cands = []
        for r in rows:
            board = chess.Board(r["fen"])
            for c in (r.get("raw_response") or {}).get("candidates", []):
                try:
                    board.parse_san(c["san"])
                except Exception:
                    all_illegal_cands.append((r["position_id"], c["san"], r["fen"]))

        if failed:
            print(f"\nFAILED ({len(failed)}/{len(rows)}):")
            for r in failed:
                board = chess.Board(r["fen"])
                resp = r.get("raw_response") or {}
                print(f"  - {r['position_id']:30s} chose: {resp.get('chosen_move','?')}")
                print(f"      fen: {r['fen']}")
                cands = resp.get("candidates", [])
                print(f"      candidates: {[c['san'] for c in cands]}")
        else:
            print("\nAll chosen moves legal.")

        if all_illegal_cands:
            print(f"\nILLEGAL CANDIDATES ({len(all_illegal_cands)} across all positions):")
            for pid, san, fen in all_illegal_cands:
                print(f"  - {pid:30s} {san}   ({fen})")


if __name__ == "__main__":
    main(*sys.argv[1:])
