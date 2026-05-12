"""Investigate the persistent 'did not call submit_move' errors in GPT-5 runs."""
import json
from pathlib import Path

for run in [
    "20260512T123033Z__games_retry__gpt-5__skill3",
    "20260512T130536Z__games_retry__gpt-5__skill5",
    "20260512T131511Z__games_retry__gpt-5-mini__skill3",
]:
    p = Path(f"C:/Users/igorc/Projects/LLM_Chess_Eval/runs/{run}/games.jsonl")
    if not p.is_file():
        continue
    print(f"\n{'='*80}")
    print(run)
    print('=' * 80)
    for line in p.read_text(encoding="utf-8").splitlines()[:2]:
        g = json.loads(line)
        print(f"\nGAME: {g['game_id'].split('__')[-1]}  result={g['result']}  plies={g['n_plies']}  illegal={g['n_illegal']}")
        for m in g["moves"]:
            err = (m.get("model_error") or "")[:300]
            print(f"  ply {m['ply']}: chosen={m['chosen_san']!r:15s}  legal={m['chosen_legal']!s:5s}  "
                  f"retries={m['retries_used']:2d}  latency={m['latency_ms']:>6}ms")
            if err:
                print(f"    err: {err}")
