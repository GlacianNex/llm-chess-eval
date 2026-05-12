"""Pretty-print a played game with per-move metrics."""
import json
import sys
from pathlib import Path


def main(path: str) -> None:
    p = Path(path)
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        g = json.loads(line)
        print(f"\n{'='*72}")
        print(f"{g['game_id']}")
        print(f"model={g['model']}  color={g['color']}  skill={g['skill']}  "
              f"result={g['result']}  plies={g['n_plies']}  illegal={g['n_illegal']}")
        print(f"{'='*72}")
        print(f"{'ply':>3}  {'fen':<55}  {'sf_best':<8}  {'chose':<10}  {'legal':<5}  {'cp_b':>5}  {'cp_a':>5}  {'loss':>5}  {'top':<3}")
        for m in g["moves"]:
            print(
                f"{m['ply']:>3}  {m['fen_before']:<55}  "
                f"{m['sf_best_san']:<8}  {str(m['chosen_san']):<10}  "
                f"{str(m['chosen_legal']):<5}  "
                f"{m['cp_before']:>5}  {str(m['cp_after']):>5}  {m['cp_loss']:>5}  "
                f"{'Y' if m['chosen_was_top'] else '.':<3}"
            )
            if not m["chosen_legal"]:
                cands = (m["raw_response"] or {}).get("candidates", [])
                print(f"      candidates: {[c['san'] for c in cands]}")
                print(f"      err: {m.get('model_error')}")


if __name__ == "__main__":
    main(sys.argv[1])
