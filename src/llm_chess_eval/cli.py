"""CLI entry point."""
from __future__ import annotations

from pathlib import Path

import click
from rich.console import Console

from .adapters.factory import build_adapter
from .config import BENCHMARK_MATRIX, DATA_DIR, KNOWN_MODELS, benchmark_models
from .evals.consistency import score_position as score_consistency
from .evals.legality import score_position as score_legality
from .harness.game_runner import run_games
from .harness.runner import load_position_bank, run_eval


@click.group()
def main() -> None:
    """LLM Chess Eval — chess-based logical-thinking benchmark."""


@main.command("legality")
@click.option("--model", required=True, type=str, help="Model ID (any from KNOWN_MODELS in config.py, or any name with a recognized prefix).")
@click.option("--n", type=int, default=None, help="Limit to first N positions (default: all).")
@click.option("--bank", type=click.Path(path_type=Path), default=None, help="Override position bank JSONL.")
@click.option("--augment-legal-moves/--no-augment-legal-moves", default=False,
              help="If set, include the full list of legal SAN moves in the prompt (informational).")
def cmd_legality(model: str, n: int | None, bank: Path | None, augment_legal_moves: bool) -> None:
    """Run the legality eval: model must produce a legal SAN move per position."""
    console = Console()
    bank_path = bank or (DATA_DIR / "positions" / "legality_v1.jsonl")
    positions = load_position_bank(bank_path)
    if n is not None:
        positions = positions[:n]

    adapter = build_adapter(model=model, augment_legal_moves=augment_legal_moves)
    run_eval(
        eval_name="legality",
        model=model,
        positions=positions,
        score_fn=lambda pos, run_id: score_legality(adapter, pos, run_id),
        console=console,
    )


@main.command("consistency")
@click.option("--model", required=True, type=str, help="Model ID (any from KNOWN_MODELS in config.py, or any name with a recognized prefix).")
@click.option("--n", type=int, default=None, help="Limit to first N positions (default: all).")
@click.option("--bank", type=click.Path(path_type=Path), default=None, help="Override position bank JSONL.")
@click.option("--augment-legal-moves/--no-augment-legal-moves", default=False,
              help="If set, include the full list of legal SAN moves in the prompt (informational).")
def cmd_consistency(model: str, n: int | None, bank: Path | None, augment_legal_moves: bool) -> None:
    """Run the consistency eval: model's rule-based claims must match python-chess ground truth."""
    console = Console()
    bank_path = bank or (DATA_DIR / "positions" / "legality_v1.jsonl")
    positions = load_position_bank(bank_path)
    if n is not None:
        positions = positions[:n]

    adapter = build_adapter(model=model, augment_legal_moves=augment_legal_moves)
    run_eval(
        eval_name="consistency",
        model=model,
        positions=positions,
        score_fn=lambda pos, run_id: score_consistency(adapter, pos, run_id),
        console=console,
    )


@main.command("games")
@click.option("--model", required=True, type=str, help="Model ID (any from KNOWN_MODELS in config.py, or any name with a recognized prefix).")
@click.option("--games", "n_games", type=int, default=2, help="Number of games to play.")
@click.option("--skill", type=int, default=3, help="Stockfish skill level for the opponent (0-20).")
@click.option("--sf-depth", type=int, default=12, help="Stockfish search depth for ground-truth eval AND opponent moves.")
@click.option("--max-plies", type=int, default=60, help="Cap on LLM moves per game (game ends if reached).")
@click.option("--color", type=click.Choice(["white", "black", "alternating"]), default="alternating")
@click.option(
    "--mode",
    type=click.Choice(["forfeit", "substitute", "retry"]),
    default="forfeit",
    help="How to handle an illegal chosen move. forfeit=end game; substitute=play Stockfish-best; "
         "retry=re-prompt model up to --max-retries times.",
)
@click.option("--max-retries", type=int, default=3, help="Retry attempts in retry mode before forfeit.")
@click.option("--augment-legal-moves/--no-augment-legal-moves", default=False,
              help="If set, include the full list of legal SAN moves in each prompt (informational).")
def cmd_games(
    model: str, n_games: int, skill: int, sf_depth: int, max_plies: int,
    color: str, mode: str, max_retries: int, augment_legal_moves: bool,
) -> None:
    """Play full games vs Stockfish. Every move instrumented; composite score penalizes illegals heavily."""
    console = Console()
    adapter = build_adapter(model=model, augment_legal_moves=augment_legal_moves)
    run_games(
        adapter=adapter,
        n_games=n_games,
        skill=skill,
        sf_depth=sf_depth,
        max_plies=max_plies,
        color=color,
        mode=mode,
        max_retries=max_retries,
        console=console,
    )


@main.command("reliability")
@click.option("--games-jsonl", type=click.Path(path_type=Path), multiple=True, required=False,
              help="Path(s) to games.jsonl files. If omitted, runs a fresh gauntlet.")
@click.option("--model", type=str, default=None,
              help="If running fresh, model ID (any provider known to KNOWN_MODELS).")
@click.option("--games", "n_games", type=int, default=5, help="Number of games (when running fresh).")
@click.option("--skill", type=int, default=3, help="Stockfish skill level for the opponent.")
@click.option("--sf-depth", type=int, default=12, help="Stockfish search depth.")
@click.option("--max-plies", type=int, default=40, help="Cap on LLM plies per game.")
@click.option("--max-retries", type=int, default=10,
              help="Per-move retry budget. Each retry costs 0.5^n on per-move score. Default 10.")
def cmd_reliability(games_jsonl: tuple[Path, ...], model: str | None, n_games: int,
                    skill: int, sf_depth: int, max_plies: int, max_retries: int) -> None:
    """Compute ChessReliability score: a single 0-1 metric per model.

    CR = mean over games of (legal_moves / max_plies) × mean(quality × retry_penalty).
    Retries are allowed but penalized 0.5^n per move; game forfeits if all retries fail.
    """
    from .evals.chess_reliability import chess_reliability, load_games

    if games_jsonl:
        for path in games_jsonl:
            games = load_games(path)
            result = chess_reliability(games, max_plies=max_plies)
            mdl = games[0].model if games else "?"
            click.echo(f"\n{path.parent.name}  [{mdl}]")
            click.echo(f"  ChessReliability:    {result['chess_reliability']:.3f}")
            click.echo(f"  n_games:             {result['n_games']}")
            click.echo(f"  survival_component:  {result['survival_component_mean']:.3f}  "
                       f"(mean {result['mean_plies_legal']:.1f}/{max_plies} plies)")
            click.echo(f"  quality_component:   {result['quality_component_mean']:.3f}")
            click.echo(f"  retry_cost_mean:     {result['retry_cost_mean']:.3f}  "
                       f"(mean {result['mean_retries_per_move']:.2f} retries/move)")
            click.echo(f"  completion_rate:     {result['completion_rate']:.2f}")
            click.echo(f"  per-game: {[f'{s:.3f}' for s in result['per_game_scores']]}")
        return

    if model is None:
        raise click.UsageError("Provide either --games-jsonl or --model.")

    from .harness.game_runner import run_games
    adapter = build_adapter(model=model)
    out_path, records = run_games(
        adapter=adapter,
        n_games=n_games,
        skill=skill,
        sf_depth=sf_depth,
        max_plies=max_plies,
        color="alternating",
        mode="retry",
        max_retries=max_retries,
        console=Console(),
    )
    result = chess_reliability(records, max_plies=max_plies)
    click.echo("")
    click.echo(f"ChessReliability ({model}, skill {skill}, {n_games} games, max_retries={max_retries}): "
               f"[bold]{result['chess_reliability']:.3f}[/bold]")


@main.command("play-strength")
@click.option("--games-jsonl", type=click.Path(path_type=Path), multiple=True, required=False,
              help="Path(s) to games.jsonl from retry-mode runs. If omitted, runs fresh.")
@click.option("--model", type=str, default=None,
              help="If running fresh, model ID (any provider known to KNOWN_MODELS).")
@click.option("--games", "n_games", type=int, default=3, help="Number of games (when running fresh).")
@click.option("--skill", type=int, default=5, help="Stockfish skill level (default 5 ~= moderate amateur).")
@click.option("--sf-depth", type=int, default=12, help="Stockfish search depth.")
@click.option("--max-plies", type=int, default=60, help="Cap on LLM moves per game (longer to reach endgame).")
@click.option("--max-retries", type=int, default=3, help="Retry attempts per move before forfeit.")
def cmd_play_strength(games_jsonl: tuple[Path, ...], model: str | None, n_games: int,
                      skill: int, sf_depth: int, max_plies: int, max_retries: int) -> None:
    """Compute PlayStrength score: move quality across honest full-game playthroughs.

    Complements ChessReliability — PS measures move quality (ACPL), CR measures
    rule-following. Uses retry mode at fixed Stockfish skill so games progress
    to a natural conclusion.
    """
    from .evals.chess_reliability import load_games
    from .evals.play_strength import compute_play_strength

    if games_jsonl:
        for path in games_jsonl:
            games = load_games(path)
            result = compute_play_strength(games, max_plies=max_plies)
            mdl = games[0].model if games else "?"
            click.echo(f"\n{path.parent.name}  [{mdl}]")
            click.echo(f"  PlayStrength:           {result['play_strength']:.3f}")
            click.echo(f"  mean survival:          {result['mean_survival']:.3f}  "
                       f"({result['mean_plies_legal']:.1f}/{max_plies} legal moves)")
            click.echo(f"  mean quality:           {result['quality_component_mean']:.3f}")
            click.echo(f"  natural completion:     {result['completion_rate_natural']:.2f}")
            click.echo(f"  ACPL overall:           {result['acpl_overall']:.0f} cp")
            click.echo(f"  ACPL by phase:          opening={result['acpl_opening']:.0f}  "
                       f"middlegame={result['acpl_middlegame']:.0f}  "
                       f"endgame={result['acpl_endgame']:.0f}")
            click.echo(f"  moves by phase:         {result['moves_by_phase']}")
            click.echo(f"  W/D/L/forfeit:          {result['wins']}/{result['draws']}/{result['losses']}/{result['forfeits']}")
            click.echo(f"  per-game scores:        {[f'{x:.3f}' for x in result['per_game_scores']]}")
            click.echo(f"  per-game ACPL:          {[f'{x:.0f}' for x in result['per_game_acpl']]}")
        return

    if model is None:
        raise click.UsageError("Provide either --games-jsonl or --model.")

    from .harness.game_runner import run_games
    adapter = build_adapter(model=model)
    out_path, records = run_games(
        adapter=adapter,
        n_games=n_games,
        skill=skill,
        sf_depth=sf_depth,
        max_plies=max_plies,
        color="alternating",
        mode="retry",
        max_retries=max_retries,
        console=Console(),
    )
    result = compute_play_strength(records, max_plies=max_plies)
    click.echo("")
    click.echo(f"PlayStrength ({model}, skill {skill}, {n_games} games, retry mode): "
               f"{result['play_strength']:.3f}")
    click.echo(f"  survival={result['mean_survival']:.3f} ({result['mean_plies_legal']:.1f}/{max_plies} moves), "
               f"quality={result['quality_component_mean']:.3f}, "
               f"ACPL={result['acpl_overall']:.0f} cp "
               f"(open={result['acpl_opening']:.0f}, mid={result['acpl_middlegame']:.0f}, "
               f"end={result['acpl_endgame']:.0f})")


@main.command("benchmark")
@click.option("--tier", type=click.Choice(["frontier", "budget", "both"]), default="both",
              help="Which tier(s) of the benchmark matrix to run.")
@click.option("--provider", type=click.Choice(list(BENCHMARK_MATRIX.keys()) + ["all"]), default="all",
              help="Filter to a single provider, or 'all'.")
@click.option("--cr-games", type=int, default=5, help="N games per model for ChessReliability.")
@click.option("--ps-games", type=int, default=3, help="N games per model for PlayStrength.")
@click.option("--cr-skill", type=int, default=3, help="Stockfish skill for CR (forfeit mode).")
@click.option("--ps-skill", type=int, default=5, help="Stockfish skill for PS (retry mode).")
@click.option("--cr-max-plies", type=int, default=40)
@click.option("--ps-max-plies", type=int, default=60)
@click.option("--cr-max-retries", type=int, default=10,
              help="Retry budget for CR. Each retry costs 0.5^n on per-move score.")
@click.option("--ps-max-retries", type=int, default=3,
              help="Retry budget for PS. Retries do not penalize per-move quality.")
@click.option("--dry-run", is_flag=True, help="Print the planned model set and exit.")
def cmd_benchmark(tier: str, provider: str, cr_games: int, ps_games: int,
                  cr_skill: int, ps_skill: int, cr_max_plies: int, ps_max_plies: int,
                  cr_max_retries: int, ps_max_retries: int, dry_run: bool) -> None:
    """Run CR + PS for every model in the benchmark matrix.

    The matrix is (provider × tier): four providers × {frontier, budget}.
    Edit BENCHMARK_MATRIX in config.py to change which models populate each cell.
    """
    from .evals.chess_reliability import chess_reliability
    from .evals.play_strength import compute_play_strength
    from .harness.game_runner import run_games

    # Resolve the model set
    tiers = ["frontier", "budget"] if tier == "both" else [tier]
    providers = [provider] if provider != "all" else list(BENCHMARK_MATRIX.keys())
    selected: list[tuple[str, str, str]] = []  # (provider, tier, model_id)
    for p in providers:
        for t in tiers:
            mid = BENCHMARK_MATRIX[p][t]
            selected.append((p, t, mid))

    console = Console()
    console.rule("Benchmark plan")
    for p, t, m in selected:
        console.print(f"  {p:<10} {t:<8} {m}")
    console.print(f"  total: {len(selected)} models")
    console.print(f"  CR config: {cr_games} games, skill {cr_skill}, max_plies {cr_max_plies}, mode=retry, max_retries {cr_max_retries} (0.5^n per-retry penalty)")
    console.print(f"  PS config: {ps_games} games, skill {ps_skill}, max_plies {ps_max_plies}, mode=retry, max_retries {ps_max_retries} (no per-retry penalty)")

    if dry_run:
        return

    results: list[dict] = []
    for p, t, model in selected:
        console.rule(f"{p} / {t} — {model}")
        try:
            adapter_cr = build_adapter(model=model)
            _, cr_records = run_games(
                adapter=adapter_cr, n_games=cr_games, skill=cr_skill,
                sf_depth=12, max_plies=cr_max_plies,
                color="alternating", mode="retry", max_retries=cr_max_retries, console=console,
            )
            cr = chess_reliability(cr_records, max_plies=cr_max_plies)

            adapter_ps = build_adapter(model=model)
            _, ps_records = run_games(
                adapter=adapter_ps, n_games=ps_games, skill=ps_skill,
                sf_depth=12, max_plies=ps_max_plies,
                color="alternating", mode="retry", max_retries=ps_max_retries, console=console,
            )
            ps = compute_play_strength(ps_records, max_plies=ps_max_plies)

            results.append({
                "provider": p, "tier": t, "model": model,
                "CR": cr["chess_reliability"],
                "PS": ps["play_strength"],
                "ACPL_overall": ps["acpl_overall"],
                "ACPL_opening": ps["acpl_opening"],
                "ACPL_middlegame": ps["acpl_middlegame"],
                "ACPL_endgame": ps["acpl_endgame"],
            })
        except Exception as e:  # noqa: BLE001
            console.print(f"[red]Failed on {model}: {type(e).__name__}: {e}[/red]")
            results.append({
                "provider": p, "tier": t, "model": model, "error": str(e),
            })

    # Print matrix summary
    console.rule("Benchmark matrix results")
    console.print(f"{'provider':<10} {'tier':<10} {'model':<55} {'CR':>5} {'PS':>5} {'ACPL_op':>7} {'ACPL_mid':>8} {'ACPL_end':>8}")
    for r in results:
        if "error" in r:
            console.print(f"{r['provider']:<10} {r['tier']:<10} {r['model']:<55} [red]ERR: {r['error'][:40]}[/red]")
        else:
            console.print(
                f"{r['provider']:<10} {r['tier']:<10} {r['model']:<55} "
                f"{r['CR']:>5.3f} {r['PS']:>5.3f} {r['ACPL_opening']:>7.0f} "
                f"{r['ACPL_middlegame']:>8.0f} {r['ACPL_endgame']:>8.0f}"
            )


@main.command("report")
@click.option("--out", type=click.Path(path_type=Path), default=None, help="Output markdown path (default: report.md in project root).")
def cmd_report(out: Path | None) -> None:
    """Build aggregate markdown scorecard from everything in runs/."""
    from .analytics.report import build_report
    from .config import PROJECT_ROOT, RUNS_DIR
    out_path = out or (PROJECT_ROOT / "report.md")
    text = build_report(RUNS_DIR, out_path)
    click.echo(text)
    click.echo(f"\n[written to {out_path}]")


@main.command("check-env")
def cmd_check_env() -> None:
    """Verify Python, python-chess, Stockfish, and ANTHROPIC_API_KEY."""
    console = Console()
    import chess
    console.print(f"python-chess: [green]{chess.__version__}[/green]")

    from .config import anthropic_api_key, stockfish_path

    try:
        sf = stockfish_path()
        console.print(f"stockfish:    [green]{sf}[/green]")
    except FileNotFoundError as e:
        console.print(f"stockfish:    [red]{e}[/red]")

    try:
        key = anthropic_api_key()
        console.print(f"api key:      [green]set (len={len(key)})[/green]")
    except RuntimeError as e:
        console.print(f"api key:      [red]{e}[/red]")


if __name__ == "__main__":
    main()
