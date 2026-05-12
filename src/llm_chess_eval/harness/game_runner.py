"""Run multiple games and persist a per-game JSONL + aggregate summary."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console

from ..adapters.base import ModelAdapter
from ..config import RUNS_DIR
from ..evals.games import play_game, score_game
from ..types import GameRecord


def make_run_dir(model: str, skill: int, mode: str) -> tuple[str, Path]:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{ts}__games_{mode}__{model}__skill{skill}"
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_id, run_dir


def run_games(
    adapter: ModelAdapter,
    n_games: int,
    skill: int,
    sf_depth: int,
    max_plies: int,
    color: str,
    mode: str = "forfeit",
    max_retries: int = 3,
    console: Console | None = None,
) -> tuple[Path, list[GameRecord]]:
    console = console or Console()
    run_id, run_dir = make_run_dir(adapter.model, skill, mode)
    out_path = run_dir / "games.jsonl"

    records: list[GameRecord] = []
    with out_path.open("w", encoding="utf-8") as f:
        for i in range(n_games):
            game_color = color
            if color == "alternating":
                game_color = "white" if i % 2 == 0 else "black"
            game_id = f"{run_id}__g{i+1:03d}_{game_color}"
            console.print(f"[bold cyan]game {i+1}/{n_games}[/bold cyan]  ({game_color} vs sf-skill-{skill}, mode={mode})")
            record = play_game(
                adapter=adapter,
                game_id=game_id,
                color=game_color,
                skill=skill,
                sf_depth=sf_depth,
                max_plies=max_plies,
                mode=mode,
                max_retries=max_retries,
            )
            f.write(record.model_dump_json() + "\n")
            f.flush()
            records.append(record)
            sc = score_game(record)
            console.print(
                f"  result=[bold]{record.result}[/bold]  plies={record.n_plies}  "
                f"illegal={record.n_illegal}  retries_total={int(sc['mean_retries']*record.n_plies)}  "
                f"fallbacks={int(sc['n_fallbacks'])}  "
                f"game_score={sc['game_score']:.3f}  mean_cp_loss={sc['mean_cp_loss']:.0f}"
            )

    _print_summary(console, adapter.model, skill, mode, records)
    return out_path, records


def _print_summary(console: Console, model: str, skill: int, mode: str, records: list[GameRecord]) -> None:
    n = len(records)
    if n == 0:
        return
    scored = [score_game(r) for r in records]
    mean = lambda key: sum(s[key] for s in scored) / n  # noqa: E731

    wins = sum(1 for r in records if r.model_won)
    draws = sum(1 for r in records if r.result == "1/2-1/2")
    losses = n - wins - draws
    forfeits = sum(1 for r in records if r.result == "forfeit_illegal")
    games_with_illegal = sum(1 for r in records if r.n_illegal > 0)

    console.rule(f"games [{model}] vs Stockfish skill {skill}  mode={mode}")
    console.print(
        f"games: [bold]{n}[/bold]   "
        f"W/D/L: {wins}/{draws}/{losses}   "
        f"forfeits-on-illegal: {forfeits}   "
        f"games-with-any-illegal: {games_with_illegal}/{n}"
    )
    console.print(f"  composite game_score (mean): [bold]{mean('game_score'):.3f}[/bold]")
    console.print(f"  per_move_quality:           {mean('per_move_quality'):.3f}")
    console.print(f"  per_move_consistency:       {mean('per_move_consistency'):.3f}")
    console.print(f"  illegal_multiplier (mean):  {mean('illegal_multiplier'):.3f}")
    console.print(f"  mean cp_loss:               {mean('mean_cp_loss'):.0f}")
    console.print(f"  blunder rate (300+ cp):     {mean('blunder_rate_300'):.3f}")
    console.print(f"  blunder rate (1000+ cp):    {mean('blunder_rate_1000'):.3f}")
    console.print(f"  chose_was_sf_top rate:      {mean('chosen_was_top_rate'):.3f}")
    console.print(f"  cands_legal_rate (mean):    {mean('candidates_legal_rate_mean'):.3f}")
    console.print(f"  mean retries per move:      {mean('mean_retries'):.2f}")
    console.print(f"  retry success rate:         {mean('retry_success_rate'):.3f}")
    console.print(f"  fallbacks (substitute mode): {mean('n_fallbacks'):.1f}")
    avg_plies = sum(r.n_plies for r in records) / n
    console.print(f"  avg plies per game:         {avg_plies:.1f}")
