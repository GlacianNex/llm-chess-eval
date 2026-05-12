"""Analyze whether reasoning_effort affects move quality.

For each move in each game, identify the reasoning_effort level on which
the FINAL ACCEPTED attempt was made (after any length-failure stepdowns).
Then bucket cp_loss / legality / per_move_score by that level.

Three buckets to compare:
  - "default" — first attempt, full reasoning budget, succeeded without stepdown
  - "medium" / "low" / "minimal" — succeeded only after stepping reasoning down
  - "forfeit" — even minimal failed; move counted as illegal forfeit

If quality is preserved as reasoning effort drops, the model isn't using the
extra thinking budget productively for chess. If quality collapses at lower
levels, full reasoning is doing real work and our stepdown is a damage-limiter.

Joins games.jsonl (per-move outcomes) with progress.jsonl (per-attempt
events including the reasoning_effort=... note) by game_id + ply.
"""
from __future__ import annotations
import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = PROJECT_ROOT / "runs"

REASONING_RE = re.compile(r"reasoning_effort=(\w+)")


def parse_progress(progress_path: Path) -> dict[tuple[str, int], list[str]]:
    """Return map (game_id, ply) -> list of reasoning_effort values in attempt order.
    'None' / missing means default (full) reasoning.
    """
    out: dict[tuple[str, int], list[str]] = defaultdict(list)
    if not progress_path.exists():
        return out
    for line in progress_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            ev = json.loads(line)
        except Exception:
            continue
        if ev.get("event") != "call_start":
            continue
        gid = ev.get("game_id", "")
        ply = ev.get("ply")
        note = ev.get("note", "") or ""
        m = REASONING_RE.search(note)
        level = m.group(1) if m else "default"
        out[(gid, ply)].append(level)
    return out


def analyze_run(run_dir: Path) -> dict[str, list[float]]:
    """Walk games.jsonl + progress.jsonl, return cp_losses bucketed by the
    reasoning_effort level on the FINAL accepted attempt."""
    games_path = run_dir / "games.jsonl"
    progress_path = run_dir / "progress.jsonl"
    if not games_path.exists():
        return {}
    progress_map = parse_progress(progress_path)
    by_level: dict[str, list[float]] = defaultdict(list)

    for line in games_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        g = json.loads(line)
        gid_full = g.get("game_id", "")
        for m in g.get("moves", []) or []:
            ply = m.get("ply")
            if not m.get("chosen_legal"):
                # forfeit — count under "forfeit" bucket with cp_loss=1000 (cap)
                by_level["forfeit"].append(1000)
                continue
            retries = m.get("retries_used", 0) or 0
            efforts = progress_map.get((gid_full, ply), [])
            # Final accepted attempt index = retries (0-indexed)
            level = efforts[retries] if retries < len(efforts) else "default"
            cp_loss = m.get("cp_loss", 0) or 0
            # Cap at 1000 to match chess_reliability scoring (mate-in-N gets
            # huge raw cp_loss values that distort the mean).
            cp_loss = min(max(cp_loss, 0), 1000)
            by_level[level].append(cp_loss)
    return by_level


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=None,
                    help="Specific run dir names (basename). If omitted, scans all latest CR+PS per model.")
    args = ap.parse_args()

    if args.runs:
        run_paths = [RUNS_DIR / r for r in args.runs]
    else:
        run_paths = list(RUNS_DIR.glob("*__games_retry__*__skill*"))

    # Aggregate by model across runs (combining CR + PS).
    by_model: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for run in sorted(run_paths):
        # Extract model name from run dir.
        name = run.name
        try:
            model = name.split("__games_retry__")[1].rsplit("__skill", 1)[0]
        except Exception:
            continue
        levels = analyze_run(run)
        for lvl, cps in levels.items():
            by_model[model][lvl].extend(cps)

    LEVELS = ["default", "medium", "low", "minimal", "forfeit"]
    print(f"{'model':<35} | " + " | ".join(f"{l:>9}" for l in LEVELS) + " |")
    print("-" * (35 + (12 + 3) * len(LEVELS)))
    for model, levels in sorted(by_model.items()):
        cells = []
        for lvl in LEVELS:
            cps = levels.get(lvl, [])
            if not cps:
                cells.append(f"{'-':>9}")
            else:
                mean_cp = statistics.mean(cps)
                cells.append(f"{int(mean_cp):>4}cp/{len(cps):>3}")
        print(f"{model:<35} | " + " | ".join(cells) + " |")

    print()
    print("Reading:")
    print("  Cell format: <mean_cp_loss>cp / <n_moves_in_bucket>")
    print("  default = first attempt succeeded at full reasoning")
    print("  medium/low/minimal = succeeded only after stepdown (signal: full-reasoning length-fail)")
    print("  forfeit = even minimal failed (cp counted as 1000 cap)")


if __name__ == "__main__":
    main()
