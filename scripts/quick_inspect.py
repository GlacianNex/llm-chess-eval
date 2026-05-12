"""Quick inspector for a games.jsonl — print per-move details for one game."""
import json
import sys

with open(sys.argv[1], encoding="utf-8") as f:
    for line in f:
        g = json.loads(line)
        print(f"\n=== {g['game_id']}  result={g['result']}  plies={g['n_plies']} ===")
        for m in g["moves"]:
            err = (m.get("model_error") or "")[:120]
            failed = m.get("failed_attempts") or []
            print(f"  ply {m['ply']}: chosen={m['chosen_san']!r:25s} legal={m['chosen_legal']!s:5s} "
                  f"retries={m['retries_used']:2d} fallback={m['fallback_used']!s:5s}")
            if failed:
                print(f"    failed_attempts: {failed}")
            if err:
                print(f"    error: {err}")
