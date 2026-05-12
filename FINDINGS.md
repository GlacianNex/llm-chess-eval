# LLM Chess Eval — Findings (digest)

This is a short results-only digest. The full writeup with methodology, definitions, calculations, design rationale, reproduction recipe, and caveats lives in **[HANDOFF.md](HANDOFF.md)**.

## Headline

LLMs can describe chess rules accurately (99% correct on rule-claim questions) but cannot reliably apply them spatially on out-of-distribution positions. Frontier reasoning-tier models score in the **0.15-0.45** range on the headline ChessReliability metric while playing 95-99% Stockfish-quality moves on the moves they do play. **The gap between move quality and game completion is the spatial-reasoning ceiling.**

## What we measured

Two composite scores per model, both bounded `[0, 1]`:

- **ChessReliability (CR)** — rule-following over full games vs Stockfish skill 3, with retries permitted but penalized geometrically (`0.5^n` per retry, max 10 retries before forfeit).
- **PlayStrength (PS)** — move quality (centipawn loss) over honest playthroughs vs Stockfish skill 5 in retry mode.

Both formulas multiply a survival factor (legal moves played / max moves) by a quality factor. Either factor near zero collapses the score.

## The matrix

Eight cells: frontier and budget tier across four providers. Standardized config (see HANDOFF.md §"How to reproduce"). Cells currently being re-run are marked.

| Provider | Tier | Model | CR | PS | ACPL open / mid / end |
|---|---|---|---|---|---|
| Anthropic | frontier | `claude-opus-4-7` | **0.427** | 0.259 | 84 / 241 / – |
| Anthropic | budget | `claude-haiku-4-5-20251001` | 0.210 | 0.094 | 62 / – / – |
| OpenAI | frontier | `gpt-5` | _running_ | _running_ | _running_ |
| OpenAI | budget | `gpt-5-mini` | _running_ | _running_ | _running_ |
| Google | frontier | `gemini-3.1-pro-preview` | _running_ | _running_ | _running_ |
| Google | budget | `gemini-3.1-flash-lite` | **0.633** | **0.726** | 43 / 71 / 66 |
| DeepSeek | frontier | `deepseek-reasoner` | _running_ | _running_ | _running_ |
| DeepSeek | budget | `deepseek-chat` | _running_ | _running_ | _running_ |

## Three findings from the clean data

**1. Frontier-tier scores are unimpressive in absolute terms.** Anthropic's strongest reasoning model (Opus 4.7) scores 0.427 CR and 0.259 PS — well below a 1500-rated human. Gemini 3.1 Flash Lite (a non-reasoning budget model) outscores Opus on both metrics: 0.633 CR and 0.726 PS. The strongest model on this benchmark so far is a budget tier from a different family.

**2. The memorization cliff is universal.** Every model that reaches mid-game shows the ACPL gradient: 25-130 cp opening → 55-241 cp middlegame → 8-66 cp endgame (where reached). Opening positions are essentially perfect for every model; quality degrades sharply on out-of-distribution positions. Every standard opening position in our 20-position bank shows a 0% failure rate across every model tested. Mid-game and synthetic endgame positions show 33-67% failure rates on the harder examples.

**3. Models form persistent wrong beliefs.** The most striking single qualitative finding: models don't make random illegal moves. They form coherent-but-wrong mental models of a position and commit to them across multiple stateless API calls. Examples from the data:

- Opus proposed the same illegal king-capture `Kxg4` **five times** across 11 plies in one game, with progressive but consistent rationales. Each call is independent, but the same FEN deterministically prompts the same wrong pattern-match.
- Sonnet proposed `d7` three times to advance a pawn that does not exist in the position.
- In retry mode, when feedback tells the model "your last move was illegal," 17/22 retries iterated to a different piece type — but forfeits clustered on plies where ALL retry attempts shared the same structural mistake (e.g., none of 4 retries addressed an existing check).

This is qualitatively different from typical "LLM hallucinations." A standard hallucination is a one-shot plausible-sounding wrong fact. The pattern here is convergence on the same wrong belief across many independent invocations — the wrong belief is encoded in the model's deterministic response to a specific input.

## Failure mode breakdown

Across 224 illegal moves classified, **~95% are spatial-reasoning failures**: line-of-sight checks missed (27%), sliding-piece paths through occupied squares (23%), king walking into attacked squares (15%), per-piece position tracking errors (12%), pawn movement direction/distance (10%), 2D adjacency (5%). The non-spatial remainder is castling state and inventory errors.

## What this implies

The benchmark exposes a structural weakness that survives training improvements and scale. Models can describe chess rules verbally with high accuracy while applying them spatially at much lower rates. Chess is a clean test of this dimension; the same failure profile should appear in any domain requiring sustained reasoning over structured 2D state (UI layouts, robotics simulations, multi-step refactoring on unfamiliar codebases, physical reasoning, map navigation).

**See [HANDOFF.md](HANDOFF.md) for full methodology, calculations, design rationale, reproduction recipe, and the cross-family interpretation.**
