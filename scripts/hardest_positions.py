"""Which positions in the legality bank caused the most failures across all runs?"""
import json
from collections import defaultdict
from pathlib import Path

RUNS = Path("C:/Users/igorc/Projects/LLM_Chess_Eval/runs")


def main():
    by_pos = defaultdict(lambda: {"n_runs": 0, "chose_illegal": 0, "cands_illegal_total": 0, "cands_total": 0, "claim_imperfect": 0, "consistency_runs": 0})
    for d in sorted(RUNS.iterdir()):
        if not d.is_dir():
            continue
        for jsonl_name in ("legality.jsonl", "consistency.jsonl"):
            jsonl = d / jsonl_name
            if not jsonl.is_file():
                continue
            is_legality = jsonl_name == "legality.jsonl"
            for line in jsonl.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                pid = r["position_id"]
                if is_legality:
                    by_pos[pid]["n_runs"] += 1
                    if r["score"] < 1.0:
                        by_pos[pid]["chose_illegal"] += 1
                    cands = (r.get("raw_response") or {}).get("candidates", [])
                    cl = r["sub_scores"].get("candidates_legal_rate", 1.0)
                    by_pos[pid]["cands_total"] += len(cands)
                    by_pos[pid]["cands_illegal_total"] += round((1 - cl) * len(cands))
                else:
                    by_pos[pid]["consistency_runs"] += 1
                    if r["score"] < 1.0:
                        by_pos[pid]["claim_imperfect"] += 1

    rows = []
    for pid, d in by_pos.items():
        chose_illegal_rate = d["chose_illegal"] / max(1, d["n_runs"])
        cands_illegal_rate = d["cands_illegal_total"] / max(1, d["cands_total"])
        claim_imp = d["claim_imperfect"] / max(1, d["consistency_runs"])
        # Combined "trouble score"
        trouble = chose_illegal_rate + cands_illegal_rate + claim_imp
        rows.append((trouble, pid, d, chose_illegal_rate, cands_illegal_rate, claim_imp))

    rows.sort(reverse=True)
    print(f"{'position_id':30s} {'legality_runs':>13} {'chose_ill%':>10} {'cands_ill%':>10} {'claim_imp%':>10} {'trouble':>8}")
    for trouble, pid, d, chose_ill, cands_ill, claim_imp in rows:
        print(f"{pid:30s} {d['n_runs']:>13} {chose_ill*100:>9.0f}% {cands_ill*100:>9.0f}% {claim_imp*100:>9.0f}% {trouble:>8.2f}")


if __name__ == "__main__":
    main()
