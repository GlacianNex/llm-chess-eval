# LLM Chess Eval — Findings (digest)

This is a short results-only digest. The full writeup with methodology, definitions, calculations, design rationale, reproduction recipe, and caveats lives in **[HANDOFF.md](HANDOFF.md)**.

## Headline

**Across eight model-tier cells from four providers, the strongest model in the matrix achieves a ChessReliability score of 0.64 on a [0, 1] scale, with engine-quality self-play as the 1.0 reference.** The top score reflects a model that picks legal moves on first attempt ~88% of the time and reaches the endgame phase with playable, if mediocre, move quality. No model in the matrix exceeds that. The bottom of the matrix sits at 0.03 — models that forfeit on illegal moves before reaching mid-game.

The ranking does not track "reasoning tier" or "frontier vs budget". It tracks something more specific: whether the model can produce a legal SAN move on first attempt while staying alive long enough to reach the positions outside its training distribution.

## What we measure

Two scores per model, both bounded `[0, 1]`:

- **ChessReliability** — rule-following over full games. Stockfish skill 3 opponent, up to 10 retries per illegal move, with steep per-retry cost. Catches models that can't produce legal play on first attempt.
- **PlayQuality** — move strength once a legal move is found. Stockfish skill 5 opponent (harder), max 3 retries, no per-retry cost. Catches models that play legal-but-bad moves.

Per-move score is the product of three factors:

```
per_move_score  =  move_quality(cp_loss)  ×  retry_cost(retries)  ×  game_phase_weight(ply)

   move_quality(cp_loss)  =  exp(-cp_loss / 150)
   retry_cost(retries)    =  0.25 ^ retries
   game_phase_weight(ply) =  1 / 2 / 4 / 8  (doubles at ply 10, 20, 30)

per_game_score  =  sum(per_move_score for legal moves)  /  max_possible_weighted_score
```

The exponential quality decay separates engine-level play from grandmaster from intermediate. The 0.25^retries cost makes the retry safety net expensive (one retry costs 75% of the move's value). The geometric phase weight encodes the memorization-cliff thesis directly: late-game plies (out of training distribution) are weighted 8× the openings.

ChessReliability uses all three factors. PlayQuality uses move_quality × phase_weight only — once a legal move is found, PlayQuality scores its strength independent of how many retries it took.

## The matrix

Eight cells: frontier and budget tier across four providers. Sorted by ChessReliability.

| Provider | Tier | Model | Reliability | PlayQuality | first-try legal | avg retries/move |
|---|---|---|---|---|---|---|
| Google | budget | `gemini-3.1-flash-lite` | **0.639** | 0.217 | 87.8% | 0.16 |
| Google | frontier | `gemini-2.5-pro` | 0.527 | **0.274** | 93.8% | 0.07 |
| OpenAI | frontier | `gpt-5` | 0.410 | _re-running_ | **97.8%** | 0.15 |
| DeepSeek | frontier | `deepseek-reasoner` | 0.373 | 0.086 | 78.3% | 0.27 |
| Anthropic | frontier | `claude-opus-4-7` | 0.164 | 0.068 | 63.1% | 0.79 |
| DeepSeek | budget | `deepseek-chat` | 0.040 | 0.017 | 38.3% | 2.65 |
| Anthropic | budget | `claude-haiku-4-5-20251001` | 0.032 | 0.014 | 53.7% | 1.74 |
| OpenAI | budget | `gpt-5-mini` | _re-running_ | _re-running_ | _re-running_ | _re-running_ |

`gemini-3.1-pro-preview` is on a 250 req/day cap (Google's preview-track policy applies regardless of paid tier) — too tight for the gauntlet. We use the GA `gemini-2.5-pro` for the published frontier-Google cell.

## Four findings from the data

**1. The top of the matrix is a budget non-reasoning model.** Gemini 3.1 Flash Lite leads ChessReliability at 0.639 by being almost-always-legal on first try (87.8%) and staying alive into mid/endgame. Frontier reasoning models from Anthropic and DeepSeek score below it. Reasoning-tier optimization is neither necessary (Flash Lite has no extended thinking) nor sufficient (frontier OpenAI and Anthropic both score below it).

**2. GPT-5 has the highest first-try legal rate in the matrix at 97.8%** — better than Flash Lite (87.8%) or Gemini 2.5 Pro (93.8%). It's also the only model that nearly never proposes an illegal first move. Its overall Reliability score (0.410) reflects something different: when the rare illegal move does happen, GPT-5's stepdown ladder needs more retries than other models to recover, which compounds the cost. **PlayQuality re-run pending after a token-budget configuration bug fix** — the prior published number was unreliable; new measurement is in flight.

**3. The memorization cliff is universal and built into the score.** Every model that reaches mid-game shows the ACPL gradient: 25-130 cp opening → 55-241 cp middlegame → 50-130 cp endgame. Standard opening positions in our 20-position bank show 0% failure rate across every model tested. Mid-game and synthetic endgame positions show 33-67% failure rates on the hardest examples. The geometric phase weight (1/2/4/8 by ply bucket) means models that fail to reach endgame score very low — surviving long is half the score by construction.

**4. Models form persistent wrong beliefs.** The most striking single qualitative finding: models don't make random illegal moves. They form coherent-but-wrong mental models of a position and commit to them across multiple stateless API calls. Examples from the game logs:

- Claude Opus proposed the same illegal king-capture `Kxg4` **five times** across 11 plies in one game, each time with a different but consistent rationale. Each call is independent, but the same FEN deterministically reproduces the same wrong pattern-match.
- Claude Sonnet proposed `d7` three times to advance a pawn that does not exist on the board.
- In retry mode, when feedback tells the model "your last move was illegal," 17 of 22 retries iterated to a different piece — but forfeits clustered on plies where ALL retry attempts shared the same structural mistake (e.g., none addressed an existing check because none of the retries perceived the check).

This is qualitatively different from typical "LLM hallucinations." A standard hallucination is a one-shot plausible-sounding wrong fact. The pattern here is convergence on the same wrong belief across many independent invocations — the belief is the model's deterministic response to a specific input, not a random error.

## Note on the Anthropic scores

Anthropic models score lowest in the matrix. This is **not** an artifact of the methodology — Anthropic was the only provider unaffected by the response-token-budget configuration issue that bit other providers (their `max_tokens` parameter counts output tokens only, with thinking on a separate budget). The Anthropic numbers reflect genuine model behavior.

What the matrix exposes is that **Anthropic's reasoning models struggle specifically with sustained 2D spatial state-tracking** — the cognitive failure mode this benchmark was designed to isolate. Anthropic excels on benchmarks that reward strong single-shot reasoning over textual or symbolic state (MMLU, GPQA, coding). It underperforms here because the benchmark is biased toward the dimension where Anthropic is comparatively weakest. The same models that fail to reach the endgame in chess routinely solve graduate-level math and write production code — chess-style spatial reasoning is a specific weakness, not a general one.

The matrix is not a ranking of "best AI." It's a ranking on one cognitive dimension. Anthropic is at the bottom of *this* dimension; other dimensions order the providers very differently.

## Failure mode breakdown

Across 224 illegal moves classified, **~95% are spatial-reasoning failures**: line-of-sight checks missed (27%), sliding-piece paths through occupied squares (23%), king walking into attacked squares (15%), per-piece position tracking errors (12%), pawn movement direction/distance (10%), 2D adjacency (5%). The non-spatial remainder is castling state and inventory errors.

## What this implies

The benchmark exposes a structural weakness that survives training improvements and scale. Models can describe chess rules verbally with high accuracy while applying them spatially at much lower rates. The same failure profile should appear in any domain requiring sustained reasoning over structured 2D state — UI layouts, robotics simulations, multi-step refactoring on unfamiliar codebases, physical reasoning, map navigation.

The scoring is designed to be *hill-climbable*: the top of the current matrix (0.64) leaves real room above. A model that reaches engine-level play across all phases would score near 0.95+. Today's gap between "best in matrix" and "what a strong chess engine does" is substantial — there is significant cognitive headroom to grow into.

**See [HANDOFF.md](HANDOFF.md) for full methodology, definitions, calculations, design rationale, reproduction recipe, and deep dives.**
