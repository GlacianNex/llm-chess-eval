"""For each position in the legality bank, check whether failures in legality
correlate with failures in consistency, on the same model."""
import json
from collections import defaultdict
from pathlib import Path

RUNS = Path("C:/Users/igorc/Projects/LLM_Chess_Eval/runs")


def collect_by_model(eval_name: str) -> dict:
    """{(model, pos_id) -> [score, ...] across runs}"""
    out: dict = defaultdict(list)
    for d in sorted(RUNS.iterdir()):
        if not d.is_dir():
            continue
        jsonl = d / f"{eval_name}.jsonl"
        if not jsonl.is_file():
            continue
        for line in jsonl.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            out[(r["model"], r["position_id"])].append(r["score"])
    return out


def main() -> None:
    leg = collect_by_model("legality")
    con = collect_by_model("consistency")

    print(f"{'position_id':30s} {'model':20s} {'leg_n':>5} {'leg_score':>10} {'con_n':>5} {'con_score':>10}")
    keys = sorted(set(leg.keys()) | set(con.keys()))
    same_fail = 0
    only_leg_fail = 0
    only_con_fail = 0
    both_pass = 0
    for (model, pid) in keys:
        leg_scores = leg.get((model, pid), [])
        con_scores = con.get((model, pid), [])
        leg_mean = sum(leg_scores) / len(leg_scores) if leg_scores else float("nan")
        con_mean = sum(con_scores) / len(con_scores) if con_scores else float("nan")
        leg_fail = leg_mean < 1.0 if leg_scores else False
        con_fail = con_mean < 1.0 if con_scores else False
        if leg_fail and con_fail:
            same_fail += 1
        elif leg_fail:
            only_leg_fail += 1
        elif con_fail:
            only_con_fail += 1
        else:
            both_pass += 1
        leg_s = f"{leg_mean:.2f}" if leg_scores else "-"
        con_s = f"{con_mean:.2f}" if con_scores else "-"
        flag = " <-- both fail" if (leg_fail and con_fail) else (" <-- legality only" if leg_fail and not con_fail else (" <-- consistency only" if con_fail and not leg_fail else ""))
        print(f"{pid:30s} {model:20s} {len(leg_scores):>5} {leg_s:>10} {len(con_scores):>5} {con_s:>10}{flag}")

    print("\n=== Summary ===")
    print(f"  both legality + consistency fail: {same_fail}")
    print(f"  legality only:                     {only_leg_fail}")
    print(f"  consistency only:                  {only_con_fail}")
    print(f"  both pass:                         {both_pass}")


if __name__ == "__main__":
    main()
