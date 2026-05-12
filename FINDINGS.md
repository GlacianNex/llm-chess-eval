# LLM Chess Eval — Findings (digest)

This is a short results-only digest. The full writeup with methodology, definitions, calculations, design rationale, reproduction recipe, and caveats lives in **[HANDOFF.md](HANDOFF.md)**.

## Headline

**No model achieves ≥90% legal-move rate on first attempt across the eight-cell cross-provider matrix.** The best is a budget non-reasoning model (Gemini 3.1 Flash Lite at 87.4%); the worst is a frontier reasoning model (GPT-5 at 2.4%). Every cell relies on the retry mechanism to complete games at all.

LLMs can describe chess rules accurately (99% correct on rule-claim questions) but cannot reliably apply them spatially on out-of-distribution positions. Frontier reasoning models score in the **0.03-0.70 range** on the headline ChessReliability metric. The 20× spread within the frontier tier — and the fact that the matrix is topped by Google's frontier and bottomed by OpenAI's frontier — means "reasoning tier" is not the relevant axis.

## What we measured

Two composite scores per model, both bounded `[0, 1]`:

- **ChessReliability (CR)** — rule-following over full games vs Stockfish skill 3, with retries permitted but penalized geometrically (`0.25^n` per retry — 1 retry = 25% credit, 2 retries = 6%, 3 retries = 1.5%; max 10 retries before forfeit).
- **PlayStrength (PS)** — move quality (centipawn loss) over honest playthroughs vs Stockfish skill 5 in retry mode (max 3 retries, no per-retry penalty).

Both formulas multiply a survival factor (legal moves played / max moves) by a quality factor. Either factor near zero collapses the score.

Two diagnostic columns alongside the composite scores make the matrix readable: **first-attempt legal rate** (fraction of plies where the very first model proposal was legal) and **mean retries per move**. Together they let you tell whether a CR score reflects "model played legal chess and was graded on quality" or "model needed the retry safety net to recover."

## The matrix

Eight cells: frontier and budget tier across four providers. Standardized config (see HANDOFF.md §"How to reproduce"). Sorted by CR.

| Provider | Tier | Model | CR | PS | 1st-attempt legal (CR) | mean retries/move (CR) |
|---|---|---|---|---|---|---|
| Google | frontier | `gemini-2.5-pro` | **0.705** | 0.285 | 82.8% | 0.32 |
| Google | budget | `gemini-3.1-flash-lite` | 0.619 | **0.664** | **87.4%** | 0.31 |
| DeepSeek | frontier | `deepseek-reasoner` | 0.459 | 0.351 | 78.7% | 0.22 |
| Anthropic | frontier | `claude-opus-4-7` | 0.395 | 0.188 | 68.9% | 0.75 |
| Anthropic | budget | `claude-haiku-4-5-20251001` | 0.187 | 0.067 | 54.3% | 1.29 |
| DeepSeek | budget | `deepseek-chat` | 0.142 | 0.052 | 40.9% | 2.14 |
| OpenAI | budget | `gpt-5-mini` | 0.099 | 0.055 | 18.3% | 2.38 |
| OpenAI | frontier | `gpt-5` | 0.033 | 0.015 | **2.4%** | 3.40 |

`gemini-3.1-pro-preview` is on a 250-req/day cap (preview-model policy applies even on paid tier), substituted with `gemini-2.5-pro` for the frontier-Google cell.

## Four findings from the clean data

**1. No model plays legal chess reliably on first attempt.** The best in the matrix is 87.4% first-attempt-legal — meaning ~13% of moves were illegal on first try and required the harness to feed back errors. Frontier reasoning models from OpenAI, Anthropic, and DeepSeek score 2.4%, 68.9%, and 78.7% on this dimension respectively. The retry-penalty mechanism in CR is what lets these models score above zero; in a real game (no retries permitted) every model in the matrix would forfeit early.

**2. Reasoning-tier supremacy doesn't hold for this task.** Across the four frontier reasoning models, CR ranges from 0.033 to 0.705 — a 20× spread. The top of the matrix is a frontier reasoning model (Gemini 2.5 Pro); the bottom is also a frontier reasoning model (GPT-5). The strongest budget non-reasoning model (Flash Lite) outscores three of the four frontier reasoning models on CR and all of them on PS. Reasoning-tier optimization helps when reasoning fits in the response budget; OpenAI's specifically does not on this benchmark.

**3. The memorization cliff is universal.** Every model that reaches mid-game shows the ACPL gradient: 25-130 cp opening → 55-241 cp middlegame → 8-66 cp endgame (where reached). Opening positions are essentially perfect for every model; quality degrades sharply on out-of-distribution positions. Every standard opening position in our 20-position bank shows a 0% failure rate across every model tested. Mid-game and synthetic endgame positions show 33-67% failure rates on the harder examples.

**4. Models form persistent wrong beliefs.** The most striking single qualitative finding: models don't make random illegal moves. They form coherent-but-wrong mental models of a position and commit to them across multiple stateless API calls. Examples from the data:

- Opus proposed the same illegal king-capture `Kxg4` **five times** across 11 plies in one game, with progressive but consistent rationales. Each call is independent, but the same FEN deterministically prompts the same wrong pattern-match.
- Sonnet proposed `d7` three times to advance a pawn that does not exist in the position.
- In retry mode, when feedback tells the model "your last move was illegal," 17/22 retries iterated to a different piece type — but forfeits clustered on plies where ALL retry attempts shared the same structural mistake (e.g., none of 4 retries addressed an existing check).

This is qualitatively different from typical "LLM hallucinations." A standard hallucination is a one-shot plausible-sounding wrong fact. The pattern here is convergence on the same wrong belief across many independent invocations — the wrong belief is encoded in the model's deterministic response to a specific input.

## Failure mode breakdown

Across 224 illegal moves classified, **~95% are spatial-reasoning failures**: line-of-sight checks missed (27%), sliding-piece paths through occupied squares (23%), king walking into attacked squares (15%), per-piece position tracking errors (12%), pawn movement direction/distance (10%), 2D adjacency (5%). The non-spatial remainder is castling state and inventory errors.

## What this implies

The benchmark exposes a structural weakness that survives training improvements and scale. Models can describe chess rules verbally with high accuracy while applying them spatially at much lower rates. Chess is a clean test of this dimension; the same failure profile should appear in any domain requiring sustained reasoning over structured 2D state (UI layouts, robotics simulations, multi-step refactoring on unfamiliar codebases, physical reasoning, map navigation).

The retry-context columns matter for interpreting the score: a model with high CR achieved by mostly playing legal moves is doing something qualitatively different from a model with the same CR achieved through the retry safety net. The 0.25^n penalty is steep enough that retry-dependent models cluster near zero, surfacing the difference.

**See [HANDOFF.md](HANDOFF.md) for full methodology, calculations, design rationale, reproduction recipe, and the cross-family interpretation.**
