"""Show every illegal-move event in a game with: ply, position summary, model's reasoning, classification."""
import json
import sys
from pathlib import Path

import chess

from llm_chess_eval.analytics.illegal_taxonomy import classify_illegal_san


def main(path: str) -> None:
    p = Path(path)
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        g = json.loads(line)
        print(f"\n{'='*80}")
        print(f"{g['game_id']}  color={g['color']}  result={g['result']}  plies={g['n_plies']}  illegals={g['n_illegal']}")
        print(f"{'='*80}")
        for m in g["moves"]:
            if m.get("failed_attempts") or not m["chosen_legal"]:
                board = chess.Board(m["fen_before"])
                attempted = list(m.get("failed_attempts") or [])
                if not m["chosen_legal"] and m.get("chosen_san") and m["chosen_san"] not in attempted:
                    attempted.append(m["chosen_san"])
                print(f"\n  ply {m['ply']}  [played: {m.get('actual_played_san')!r}]   sf_best: {m['sf_best_san']}")
                summary = (m.get("raw_response") or {}).get("position_summary", "")
                if summary:
                    print(f"    summary: {summary[:200]}")
                for san in attempted:
                    cat, det = classify_illegal_san(board, san)
                    rationale = ""
                    cands = (m.get("raw_response") or {}).get("candidates", [])
                    for c in cands:
                        if c["san"] == san:
                            rationale = c.get("rationale", "")[:140]
                            break
                    print(f"    -> {san:8s} [{cat}]  rationale: {rationale}")


if __name__ == "__main__":
    main(sys.argv[1])
