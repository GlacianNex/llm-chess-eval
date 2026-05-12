"""Print a 1-line legality summary for one or more JSONL files."""
import json
import sys
from pathlib import Path


def main(*paths: str) -> None:
    print(f"{'run':<70} {'n':>3} {'score':>6} {'cands_legal':>11} {'errors':>6}")
    for p in paths:
        path = Path(p)
        rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        if not rows:
            continue
        n = len(rows)
        score = sum(r["score"] for r in rows) / n
        cands = sum(r["sub_scores"].get("candidates_legal_rate", 0) for r in rows) / n
        errors = sum(1 for r in rows if r.get("error"))
        print(f"{path.parent.name:<70} {n:>3} {score:>6.3f} {cands:>11.3f} {errors:>6}")


if __name__ == "__main__":
    main(*sys.argv[1:])
