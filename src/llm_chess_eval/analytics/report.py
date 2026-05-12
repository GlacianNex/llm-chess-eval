"""Aggregate markdown report across all runs in the runs/ directory.

Walks `runs/*/` for legality/consistency/games JSONL files and produces a
single scorecard organized by eval and model. Highlights:
  - per-eval leaderboard across models
  - prompt-variant comparison (baseline vs augmented)
  - game-mode comparison (forfeit / substitute / retry)
  - survival curves and accumulation summaries
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from ..analytics.accumulation import per_ply_table, survival_table
from ..evals.games import score_game
from ..types import EvalResultRow, GameRecord


@dataclass
class RunInfo:
    run_dir: Path
    timestamp: str           # ISO-ish from dir name
    eval_kind: str           # "legality" | "consistency" | "games_forfeit" | "games_substitute" | "games_retry"
    model: str
    extra: dict              # skill, etc.


def _parse_run_dir(d: Path) -> RunInfo | None:
    """Match patterns:
      <ts>__legality__<model>
      <ts>__consistency__<model>
      <ts>__games_<mode>__<model>__skill<N>
      <ts>__games__<model>__skill<N>           (legacy, treat as forfeit)
    """
    parts = d.name.split("__")
    if len(parts) < 3:
        return None
    ts = parts[0]
    kind = parts[1]
    if kind in ("legality", "consistency"):
        return RunInfo(run_dir=d, timestamp=ts, eval_kind=kind, model=parts[2], extra={})
    if kind == "games" or kind.startswith("games_"):
        mode = "forfeit" if kind == "games" else kind.split("_", 1)[1]
        model = parts[2]
        skill = None
        if len(parts) >= 4 and parts[3].startswith("skill"):
            try:
                skill = int(parts[3].replace("skill", ""))
            except ValueError:
                pass
        return RunInfo(
            run_dir=d, timestamp=ts,
            eval_kind=f"games_{mode}", model=model,
            extra={"mode": mode, "skill": skill},
        )
    return None


def discover_runs(runs_root: Path) -> list[RunInfo]:
    if not runs_root.is_dir():
        return []
    out: list[RunInfo] = []
    for d in sorted(runs_root.iterdir()):
        if not d.is_dir():
            continue
        info = _parse_run_dir(d)
        if info is not None:
            out.append(info)
    return out


def _load_eval_rows(jsonl: Path) -> list[EvalResultRow]:
    rows: list[EvalResultRow] = []
    for line in jsonl.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(EvalResultRow.model_validate_json(line))
    return rows


def _load_game_records(jsonl: Path) -> list[GameRecord]:
    rows: list[GameRecord] = []
    for line in jsonl.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(GameRecord.model_validate_json(line))
    return rows


def _legality_summary(rows: list[EvalResultRow]) -> dict:
    n = len(rows)
    if n == 0:
        return {"n": 0}
    mean_score = sum(r.score for r in rows) / n
    sub_keys = list(rows[0].sub_scores.keys())
    sub = {k: sum(r.sub_scores.get(k, 0.0) for r in rows) / n for k in sub_keys}
    errors = sum(1 for r in rows if r.error)
    return {"n": n, "score": mean_score, "errors": errors, **sub}


def _consistency_summary(rows: list[EvalResultRow]) -> dict:
    n = len(rows)
    if n == 0:
        return {"n": 0}
    mean_score = sum(r.score for r in rows) / n
    sub_keys = list(rows[0].sub_scores.keys())
    sub = {k: sum(r.sub_scores.get(k, 0.0) for r in rows) / n for k in sub_keys}
    return {"n": n, "score": mean_score, **sub}


def _games_summary(records: list[GameRecord]) -> dict:
    n = len(records)
    if n == 0:
        return {"n": 0}
    scored = [score_game(r) for r in records]
    mean = lambda k: sum(s[k] for s in scored) / n  # noqa: E731
    wins = sum(1 for r in records if r.model_won)
    draws = sum(1 for r in records if r.result == "1/2-1/2")
    forfeits = sum(1 for r in records if r.result == "forfeit_illegal")
    return {
        "n": n,
        "wins": wins,
        "draws": draws,
        "losses": n - wins - draws,
        "forfeits": forfeits,
        "game_score": mean("game_score"),
        "per_move_quality": mean("per_move_quality"),
        "per_move_consistency": mean("per_move_consistency"),
        "illegal_multiplier": mean("illegal_multiplier"),
        "mean_cp_loss": mean("mean_cp_loss"),
        "blunder_rate_300": mean("blunder_rate_300"),
        "chosen_was_top_rate": mean("chosen_was_top_rate"),
        "candidates_legal_rate_mean": mean("candidates_legal_rate_mean"),
        "mean_retries": mean("mean_retries"),
        "n_fallbacks_mean": mean("n_fallbacks"),
        "avg_plies": sum(r.n_plies for r in records) / n,
        "games_with_illegal": sum(1 for r in records if r.n_illegal > 0),
    }


def _format_table(headers: list[str], rows: list[list[str]]) -> str:
    if not rows:
        return ""
    sep = "| " + " | ".join("---" for _ in headers) + " |"
    lines = ["| " + " | ".join(headers) + " |", sep]
    for r in rows:
        lines.append("| " + " | ".join(r) + " |")
    return "\n".join(lines)


def build_report(runs_root: Path, out_path: Path) -> str:
    runs = discover_runs(runs_root)
    by_kind_model: dict[tuple[str, str], list[RunInfo]] = defaultdict(list)
    for r in runs:
        by_kind_model[(r.eval_kind, r.model)].append(r)

    lines: list[str] = []
    lines.append("# LLM Chess Eval — Aggregate Report\n")
    lines.append(f"Discovered {len(runs)} runs in `{runs_root}`.\n")

    # --- Legality ---
    lines.append("## Legality (chosen move legal y/n)\n")
    leg_rows = []
    for (kind, model), rs in sorted(by_kind_model.items()):
        if kind != "legality":
            continue
        for r in rs:
            jsonl = r.run_dir / "legality.jsonl"
            if not jsonl.is_file():
                continue
            s = _legality_summary(_load_eval_rows(jsonl))
            if s.get("n", 0) == 0:
                continue
            leg_rows.append([
                r.timestamp, model, str(s["n"]),
                f"{s['score']:.3f}",
                f"{s.get('candidates_legal_rate', 0):.3f}",
                f"{s.get('chose_in_candidates', 0):.3f}",
                str(s.get("errors", 0)),
            ])
    if leg_rows:
        lines.append(_format_table(
            ["ts", "model", "n", "chose_legal", "cands_legal", "chose_in_cands", "errors"],
            leg_rows,
        ))
    else:
        lines.append("_(no legality runs found)_")
    lines.append("")

    # --- Consistency ---
    lines.append("## Consistency (rule-based claim accuracy)\n")
    con_rows = []
    for (kind, model), rs in sorted(by_kind_model.items()):
        if kind != "consistency":
            continue
        for r in rs:
            jsonl = r.run_dir / "consistency.jsonl"
            if not jsonl.is_file():
                continue
            s = _consistency_summary(_load_eval_rows(jsonl))
            if s.get("n", 0) == 0:
                continue
            con_rows.append([
                r.timestamp, model, str(s["n"]),
                f"{s['score']:.3f}",
                f"{s.get('claim_is_check', 0):.3f}",
                f"{s.get('claim_gives_mate', 0):.3f}",
                f"{s.get('claim_is_capture', 0):.3f}",
                f"{s.get('candidates_legal_rate', 0):.3f}",
            ])
    if con_rows:
        lines.append(_format_table(
            ["ts", "model", "n", "per_cand_consist", "is_check", "gives_mate", "is_capture", "cands_legal"],
            con_rows,
        ))
    else:
        lines.append("_(no consistency runs found)_")
    lines.append("")

    # --- Games (by mode) ---
    lines.append("## Games (full-game eval)\n")
    game_rows = []
    games_by_run: dict[tuple[str, str, str], list[GameRecord]] = {}  # key = (kind, model, ts)
    for (kind, model), rs in sorted(by_kind_model.items()):
        if not kind.startswith("games_"):
            continue
        for r in rs:
            jsonl = r.run_dir / "games.jsonl"
            if not jsonl.is_file():
                continue
            recs = _load_game_records(jsonl)
            if not recs:
                continue
            games_by_run[(kind, model, r.timestamp)] = recs
            s = _games_summary(recs)
            game_rows.append([
                r.timestamp,
                kind.replace("games_", ""),
                model,
                str(r.extra.get("skill", "?")),
                f"{s['wins']}/{s['draws']}/{s['losses']}",
                str(s["forfeits"]),
                str(s["games_with_illegal"]),
                f"{s['avg_plies']:.1f}",
                f"{s['game_score']:.3f}",
                f"{s['per_move_quality']:.3f}",
                f"{s['per_move_consistency']:.3f}",
                f"{s['mean_cp_loss']:.0f}",
                f"{s['chosen_was_top_rate']:.3f}",
                f"{s['mean_retries']:.2f}",
                f"{s['n_fallbacks_mean']:.1f}",
            ])
    if game_rows:
        lines.append(_format_table(
            ["ts", "mode", "model", "skill", "W/D/L", "forfeits", "any_illegal", "avg_plies",
             "game_score", "qual", "consist", "cp_loss", "top_rate", "ret/mv", "fallbacks"],
            game_rows,
        ))
    else:
        lines.append("_(no game runs found)_")
    lines.append("")

    # --- Survival curves per game run ---
    if games_by_run:
        lines.append("## Survival curves\n")
        lines.append("Fraction of games still alive (legal-move-only) at each LLM ply, by run.\n")
        for (kind, model, ts), recs in sorted(games_by_run.items()):
            lines.append(f"### {model} — {kind.replace('games_', '')} — {ts}\n")
            surv = survival_table(recs)
            rows = [[
                str(r["ply"]),
                f"{r['alive_fraction']:.2f}",
                f"{r['legal_rate_among_alive']:.2f}",
                "#" * int(r["alive_fraction"] * 20),
            ] for r in surv]
            lines.append(_format_table(["ply", "alive_frac", "legal_rate", "bar"], rows))
            lines.append("")
            pp = per_ply_table(recs)
            cum_rows = []
            for r in pp:
                cum_rows.append([
                    str(r["ply"]), str(r["n_moves"]),
                    f"{r['illegal_rate']:.2f}",
                    "-" if r['mean_cp_loss'] != r['mean_cp_loss'] else f"{r['mean_cp_loss']:.0f}",
                    "-" if r['chose_top_rate'] != r['chose_top_rate'] else f"{r['chose_top_rate']:.2f}",
                ])
            lines.append("Per-ply rates:\n")
            lines.append(_format_table(["ply", "n", "illegal", "mean_cp_loss", "top_rate"], cum_rows))
            lines.append("")

    text = "\n".join(lines)
    out_path.write_text(text, encoding="utf-8")
    return text


if __name__ == "__main__":
    import sys
    from ..config import PROJECT_ROOT, RUNS_DIR
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else PROJECT_ROOT / "report.md"
    text = build_report(RUNS_DIR, out)
    print(text)
    print(f"\n[written to {out}]")
