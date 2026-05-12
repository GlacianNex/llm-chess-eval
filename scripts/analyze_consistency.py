"""Print per-candidate claim-vs-actual diffs from a consistency run."""
import json
import sys
from pathlib import Path


def main(*paths: str) -> None:
    for path in paths:
        print(f"\n{'='*72}\n{Path(path).parent.name}\n{'='*72}")
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            ss = r["sub_scores"]
            rr = r.get("raw_response") or {}
            print(
                f"{r['position_id']:25s} "
                f"within100={ss['within_100cp']:.2f} "
                f"dir={ss['direction_match']:.2f} "
                f"top={ss['chosen_was_top']:.0f} "
                f"cp_loss={ss['chosen_cp_loss']:.0f}"
            )
            print(f"  sf_best={rr.get('sf_best')!r}  cp_before={rr.get('cp_before')}")
            mr = rr.get("model_response") or {}
            chose = mr.get("chosen_move")
            print(f"  chosen_move={chose!r}")
            for d in rr.get("candidate_diffs", []):
                print(
                    f"    {d['san']:8s} claim={d['claim_cp']:+6d}  "
                    f"actual={d['actual_cp']:+6d}  diff={d['abs_diff_cp']}"
                )
            print()


if __name__ == "__main__":
    main(*sys.argv[1:])
