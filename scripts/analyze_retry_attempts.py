"""For each retry-mode ply with failed attempts, show what was tried.
Reveals whether the model iterates to a different move when told 'illegal' or
re-proposes essentially the same wrong idea."""
import json
import sys
from pathlib import Path

import chess

from llm_chess_eval.analytics.illegal_taxonomy import classify_illegal_san


def main(path: str) -> None:
    p = Path(path)
    n_retries_total = 0
    n_same_piece = 0
    n_same_target = 0
    n_same_move = 0  # exact same SAN twice
    n_different_idea = 0
    repeat_examples = []

    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        g = json.loads(line)
        for m in g["moves"]:
            failed = m.get("failed_attempts") or []
            if not failed:
                continue
            board = chess.Board(m["fen_before"])
            chosen = m.get("chosen_san")
            # full sequence: failed_attempts then chosen (legal pick if successful)
            sequence = list(failed)
            if m["chosen_legal"] and chosen and chosen not in sequence:
                sequence.append(chosen)
            print(f"\nply {m['ply']}  fen={m['fen_before']}")
            print(f"  sequence: {sequence}  -> played: {m['actual_played_san']}")
            for i, san in enumerate(sequence):
                cat, _ = classify_illegal_san(board, san) if san != m["actual_played_san"] else ("LEGAL", {})
                print(f"    attempt {i+1}: {san}  [{cat}]")
            n_retries_total += len(failed)
            # Pairwise compare
            for prev, nxt in zip(sequence, sequence[1:]):
                if prev == nxt:
                    n_same_move += 1
                    repeat_examples.append((g["game_id"], m["ply"], prev, nxt))
                # crude check for same target square (last [a-h][1-8])
                import re
                prev_tgt = re.findall(r"[a-h][1-8]", prev)
                nxt_tgt = re.findall(r"[a-h][1-8]", nxt)
                if prev_tgt and nxt_tgt and prev_tgt[-1] == nxt_tgt[-1]:
                    n_same_target += 1
                # same leading piece letter (or both pawns)
                def piece(s):
                    return s[0] if s[0].isupper() else "P"
                if piece(prev) == piece(nxt):
                    n_same_piece += 1
                else:
                    n_different_idea += 1

    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print(f"total retry attempts: {n_retries_total}")
    print(f"pairs reproposing SAME EXACT MOVE: {n_same_move}")
    print(f"pairs with same target square:     {n_same_target}")
    print(f"pairs with same piece type:        {n_same_piece}")
    print(f"pairs with different piece type:   {n_different_idea}")
    if repeat_examples:
        print("\nExamples of model proposing same move twice in a row:")
        for ex in repeat_examples[:5]:
            print(f"  {ex}")


if __name__ == "__main__":
    main(sys.argv[1])
