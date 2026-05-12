"""Validate every FEN in a position bank parses as a legal chess position."""
import sys
from pathlib import Path

import chess

from llm_chess_eval.harness.runner import load_position_bank


def main(bank_path: str) -> int:
    bank = load_position_bank(Path(bank_path))
    print(f"positions: {len(bank)}")
    bad = []
    for p in bank:
        try:
            b = chess.Board(p.fen)
            status = b.status()
            if status != chess.STATUS_VALID:
                bad.append((p.position_id, f"invalid status: {status!r}"))
                continue
            n_legal = b.legal_moves.count()
            side = "white" if b.turn else "black"
            print(f"  {p.position_id:30s} OK  ({n_legal} legal moves, {side} to move)")
        except Exception as e:
            bad.append((p.position_id, f"parse error: {type(e).__name__}: {e}"))

    print()
    if bad:
        print("PROBLEMS:")
        for pid, msg in bad:
            print(f"  {pid}: {msg}")
        return 1
    print("ALL POSITIONS VALID")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
