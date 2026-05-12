"""Runner: drives an eval over a position bank, writes JSONL, prints a summary."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)

from ..config import RUNS_DIR
from ..types import EvalResultRow, PositionRecord


def load_position_bank(jsonl_path: Path) -> list[PositionRecord]:
    rows = []
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(PositionRecord.model_validate_json(line))
    return rows


def make_run_dir(eval_name: str, model: str) -> tuple[str, Path]:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{ts}__{eval_name}__{model}"
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_id, run_dir


def run_eval(
    eval_name: str,
    model: str,
    positions: Iterable[PositionRecord],
    score_fn: Callable[[PositionRecord, str], EvalResultRow],
    console: Console | None = None,
) -> tuple[Path, list[EvalResultRow]]:
    console = console or Console()
    positions = list(positions)
    run_id, run_dir = make_run_dir(eval_name, model)
    out_path = run_dir / f"{eval_name}.jsonl"

    rows: list[EvalResultRow] = []
    with out_path.open("w", encoding="utf-8") as f, Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(f"{eval_name} [{model}]", total=len(positions))
        for pos in positions:
            row = score_fn(pos, run_id)
            f.write(row.model_dump_json() + "\n")
            f.flush()
            rows.append(row)
            progress.advance(task)

    _print_summary(console, eval_name, model, rows, out_path)
    return out_path, rows


def _print_summary(console: Console, eval_name: str, model: str, rows: list[EvalResultRow], out_path: Path) -> None:
    if not rows:
        console.print("[yellow]No positions scored.[/yellow]")
        return
    n = len(rows)
    mean = sum(r.score for r in rows) / n
    errors = sum(1 for r in rows if r.error)
    console.rule(f"{eval_name} [{model}]")
    console.print(f"positions: [bold]{n}[/bold]   mean score: [bold]{mean:.3f}[/bold]   errors: {errors}")
    if rows[0].sub_scores:
        for key in rows[0].sub_scores:
            avg = sum(r.sub_scores.get(key, 0.0) for r in rows) / n
            console.print(f"  {key}: {avg:.3f}")
    console.print(f"results: [dim]{out_path}[/dim]")
