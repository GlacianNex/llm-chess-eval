# LLM Chess Eval

A reproducible benchmark that measures whether LLMs can maintain coherent internal state and apply rules across many reasoning turns, using chess as a substrate. The cognitive failure it isolates — **state reconstruction and 2D spatial reasoning on out-of-distribution positions** — shows up in many domains. Chess is the cleanest substrate to expose it.

This document is the full reference: what the benchmark measures, how, what it found, and how to reproduce.

---

## Why this benchmark exists

Public LLM evaluations measure knowledge (MMLU, GPQA) or single-shot problem solving (MATH, HumanEval). Both reward models with strong training data on the relevant domain. Neither isolates whether a model can **maintain coherent state and apply known rules across many turns** — the cognitive dimension that determines whether you can trust a model on long agentic tasks.

That gap matters in practice. The same model that scores 90% on a coding benchmark can confidently propose a 3-step refactor that references a function it imagined exists, then double down on the false reference when asked to revise. The same model that solves a math problem flawlessly will solve a similar-looking problem with one altered structural detail by applying the original problem's pattern. These are not knowledge failures — they're **state-reasoning failures** that public evals do not systematically catch.

**Chess is the cleanest probe.** A few properties make it ideal:

- A small, fully-specified 2D state (8×8 grid, ~32 typed entities) — every position is exhaustively described in a FEN string.
- Rules whose application is purely geometric — line-of-sight, paths, adjacency, attack-set membership.
- Two oracles: `python-chess` for deterministic rule-checking (no judgment), Stockfish for strength evaluation.
- A built-in difficulty calibration. The first 5-10 moves of nearly every game are saturated in training data. After that, branching combinatorics push games into positions essentially never repeated. **The same model on the same task shifts from pattern-recall to spatial reasoning within ~10 moves.** We see the cliff happen.

The benchmark is small, cheap (~$2-10 per model), and runs across any provider (Anthropic, OpenAI, Google, DeepSeek, OpenAI-API-compatible endpoints) through a single CLI.

---

## What the benchmark measures

Two scores per model, both bounded `[0, 1]`:

### ChessReliability

The model plays games vs Stockfish at skill 3. On an illegal proposed move, the model is told "that was illegal" and asked to try again — up to 10 retries per move. If all retries fail, the game forfeits.

Per-move score is a product of three factors:

```
per_move_score  =  move_quality(cp_loss)  ×  retry_cost(retries)  ×  game_phase_weight(ply)
```

Where:

- **`move_quality(cp_loss) = exp(-cp_loss / 150)`** — exponential decay over centipawn loss vs Stockfish's top move. Engine-level play (cp_loss ~5) scores ~0.97; grandmaster-level (cp_loss ~50) scores ~0.72; competent club (cp_loss ~150) scores ~0.37; blunder (cp_loss ~500) scores ~0.04. The exponential shape leaves real headroom at the top of the scale — a linear quality function (used in earlier iterations) squashed Stockfish-quality and intermediate play together at the top.

- **`retry_cost(retries) = 0.25 ^ retries`** — steep multiplicative penalty for needing the retry safety net. One retry costs 75% of the move's value; two retries cost 94%. In a real chess game, an illegal move is fatal; the benchmark is a generosity-graded approximation, and the 0.25 base ensures retried moves contribute almost nothing to the score.

- **`game_phase_weight(ply) = 1 / 2 / 4 / 8`** — geometric weighting by ply bucket (boundaries at ply 10, 20, 30). Opening positions are saturated in training data (weight 1); mid-game progressively novel (weights 2, 4); endgame essentially unique from training (weight 8). The geometric shape directly encodes the memorization-cliff thesis into the metric: reaching ply 30 is worth 8× more than playing ply 5 perfectly.

Per-game score:

```
per_game_score  =  sum(per_move_score for legal moves)  /  max_possible_weighted_score
```

The denominator (`max_possible_weighted_score = sum(game_phase_weight(p) for p in 1..max_moves_per_game)`) is constant for a given `max_moves_per_game`. Unplayed plies (after a forfeit) contribute 0 to the numerator but their phase weight is still in the denominator. So an early forfeit loses BOTH the missing per-move scores AND access to the high-weight late plies — forfeit penalty scales with how much of the game was missed.

ChessReliability = mean of per_game_score across N games. Standard config: N = 5 games, max 40 moves per game.

### PlayQuality

The model plays games vs Stockfish at skill 5 (harder opponent) in retry mode with max 3 retries per move, no per-retry penalty. PlayQuality measures **how strong the move is once a legal move is found** — it doesn't penalize retry use.

Per-move score:

```
per_move_score  =  move_quality(cp_loss)  ×  game_phase_weight(ply)
```

Per-game score:

```
per_game_score  =  sum(per_move_score for legal moves)  /  max_possible_weighted_score
```

PlayQuality = mean across N = 3 games, max 60 moves per game.

The difference from ChessReliability is intentional and conceptual: Reliability = "can the model play legal chess on first attempt"; PlayQuality = "given a legal move was found, how good was it." Both share the exponential quality decay and geometric phase weight, so their numbers are directly comparable at the per-move level.

### Why three multiplicative factors

Both metrics use multiplicative structure so any factor near zero collapses the score. A model that forfeits at move 5 can't rescue the score with great opening play (the high-weight late plies in the denominator dominate). A model that drags out long games of mediocre moves can't rescue the score with completion (move_quality is low). A model that needs many retries to find legal moves can't rescue the score with eventual success (retry_cost is exponentially punitive).

### Reference points

| Score | Equivalent |
|---|---|
| 1.000 | Stockfish playing itself (engine-quality moves, full survival) |
| 0.85-0.95 | Strong chess engine play across full games |
| 0.55-0.75 | Best current LLMs on this benchmark |
| 0.30-0.50 | LLMs that survive into mid-game but with mediocre quality |
| 0.10-0.20 | LLMs that struggle past opening |
| 0.00-0.05 | Forfeit before mid-game, or heavy retry dependence |

Current matrix range: **0.032 (Claude Haiku) to 0.639 (Gemini 3.1 Flash Lite)**.

### Average Centipawn Loss (ACPL) — diagnostic

ACPL is the standard chess-strength metric: per move, the centipawn difference between Stockfish's top move and what the model played. ACPL 50 = strong club player; ACPL 150 = intermediate; ACPL 500+ = blundering. We report ACPL **by phase** (opening / middlegame / endgame) alongside the composite scores. The ACPL gradient across phases is the most direct evidence of the memorization cliff.

---

## What it looks like in practice

[Levy Rozman's short "ChatGPT vs Meta AI: This Isn't Chess Anymore"](https://www.youtube.com/shorts/YlMWZNx93G4) shows the failure as comedy. Two LLMs trying to play a game produce nonsense after the opening: pieces appear from nowhere, captures are claimed on empty squares, the same illegal move keeps getting proposed, and both models confidently narrate their plans for pieces that no longer exist on the board.

The benchmark scores it. Numerically, the failure modes in that video look like:

- Per-move legality on static positions ≈ 80-90% — the move-by-move illegal rate is "only" 10-20%, easy to miss on isolated questions.
- ChessReliability across a full game ≈ 0.1-0.4 — because per-move rates compound across 30+ turns, almost no game completes cleanly.
- Illegal moves cluster on specific spatial computations: phantom pieces (the piece type exists somewhere but not where claimed), missed long-diagonal checks, pinned pieces moved anyway, sliding pieces moved through other pieces.
- The model commits to the same wrong belief for 3-5 consecutive turns because each turn is stateless and the same pattern-match keeps regenerating the same false geometry from the same FEN input.

The numbers below quantify what the video shows.

---

## Results

### Headline

**The strongest model in the matrix achieves ChessReliability 0.639 on a [0, 1] scale where Stockfish self-play is the 1.0 reference.** That's a budget non-reasoning model (Gemini 3.1 Flash Lite) playing legal-on-first-try ~88% of the time and reaching mid/endgame with mediocre move quality. Frontier reasoning models from Anthropic and DeepSeek score below it. The bottom of the matrix sits at 0.032 — models that forfeit before reaching mid-game.

GPT-5 has the **highest first-attempt-legal rate of any cell at 97.8%** but a lower composite score because the rare illegal-move cases burn more retry-cost recovery.

### The cross-family matrix

Eight cells — frontier and budget tier for each of four providers. Standardized config: N=5 Reliability games at Stockfish skill 3 with `max_retries=10`; N=3 PlayQuality games at skill 5 with `max_retries=3`; both at `max_tokens=65536` (per-provider override for Anthropic to 8192 — Anthropic's `max_tokens` counts output tokens only); alternating colors.

`gemini-3.1-pro-preview` is on a 250 req/day cap (Google's preview-track policy applies regardless of paid tier) — too tight for a CR+PS gauntlet with retries. We use the GA `gemini-2.5-pro` for the published frontier-Google cell.

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

Rows are sorted by Reliability. `first-try legal` is the fraction of plies where the model's very first proposal was legal (before any retry feedback). `avg retries/move` is total retries used across all plies divided by total plies attempted.

### Reading the matrix

The first-try legal column and avg retries column are essential. Two models with identical Reliability scores can mean very different things — one might be picking legal moves on first try and being graded mainly on move quality; another might be needing two retries per move and being penalized for the cost of finding the legal move. The diagnostic columns separate these readings.

The matrix sorts into three behavioral bands:

**Band 1 — "Plays legal chess on first attempt":** Gemini cells, GPT-5, DeepSeek-reasoner. First-try legal rate 78-98%, avg retries 0.07-0.27 per move. The Reliability score is mostly about *move quality* on the legal moves played. The composite reflects what these models actually do at the board.

**Band 2 — "Needs the retry safety net":** Claude Opus 4.7. First-try legal 63%, avg 0.79 retries per move. Roughly one move in three is illegal on first try; the harness feeds back errors before the model corrects. Even after recovery the 0.25^retries cost ensures retried moves contribute little.

**Band 3 — "Struggles to even propose legal moves":** Claude Haiku, DeepSeek-chat. First-try legal 38-54%, avg 1.7-2.7 retries per move. About half of all moves are illegal on first try, and the games rarely survive long enough to reach the high-weight late plies.

### Three findings

**1. Reasoning-tier supremacy doesn't hold for spatial state-tracking.** Across the four frontier reasoning models in the matrix, Reliability spans 0.16 to 0.53 — a 3× spread. The matrix-leader is a budget non-reasoning model. Reasoning-tier optimization helps when the reasoning fits in the budget AND when the model can apply that reasoning to spatial state — neither is guaranteed on this benchmark.

**2. The memorization cliff is universal and built into the score.** Every model that reaches mid-game shows the ACPL gradient: 25-130 cp opening → 55-241 cp middlegame → 50-130 cp endgame. Standard opening positions in our 20-position bank show 0% failure rate across every model tested. Mid-game and synthetic endgame positions show 33-67% failure rates on the harder examples. The geometric phase weight bakes this into the score: reaching ply 30 is worth 8× more than playing ply 5 well, so models that forfeit early lose the largest share of achievable score.

**3. Models form persistent wrong beliefs.** Across all retry-mode games, the same illegal SAN proposal recurs multiple times within a single game's plies. Opus proposed `Kxg4` five times across 11 plies of one game; Sonnet proposed `d7` three times to advance a pawn that doesn't exist. Each API call is independent and stateless from the model's side, but the same FEN deterministically reproduces the same wrong pattern-match. This is qualitatively different from typical "LLM hallucinations" — convergence on the same wrong belief across many invocations implies the belief is the model's deterministic response to a specific input, not a random error.

---

## A note on the Anthropic scores

Anthropic's scores are the lowest among the four providers in the matrix. This deserves explanation because the picture is easy to misread.

**Anthropic's numbers are not artifacts of methodology.** Anthropic was the only provider whose API responses were unaffected by a response-token-budget configuration issue that affected the other providers. Anthropic's `max_tokens` parameter counts *output tokens only*, with extended thinking on a separate budget. The numbers shown here reflect genuine model behavior at full reasoning budgets.

**What the matrix exposes is that Anthropic's reasoning models struggle specifically with sustained 2D spatial state-tracking** — exactly the cognitive failure mode this benchmark was designed to isolate. Anthropic excels on benchmarks that reward strong single-shot reasoning over textual or symbolic state (MMLU, GPQA, coding). It underperforms here because the benchmark is biased toward the dimension where Anthropic is comparatively weakest. The same models that fail to reach the endgame in chess routinely solve graduate-level math and write production code — chess-style spatial reasoning is a specific weakness, not a general one.

Quantitatively, three things compound for Anthropic:

1. Anthropic's first-try-legal rate is lower than the matrix top (63% Opus vs 98% GPT-5). Each illegal-first-try move pays the 0.25^retries cost.
2. Anthropic's middlegame ACPL is the highest in the matrix at ~240 cp for Opus. Under exponential quality decay this scores ~0.20 per move vs Flash Lite's ~0.65 per move.
3. Anthropic forfeits earlier than the matrix top — Opus reaches plies 1-30 in only some games, Haiku averages 11 plies per game. With high-weight plies (20+, 30+) contributing disproportionately to both numerator and denominator, models that forfeit early lose access to the score's largest contributions.

The matrix is not a ranking of "best AI." It's a ranking on one cognitive dimension. Anthropic is at the bottom of *this* dimension; other dimensions order the providers very differently.

---

## Deep dives

### The memorization cliff in numbers

ACPL by phase across the matrix (lower is better — engine-level play is ACPL ~10):

| Model | ACPL opening | ACPL middlegame | ACPL endgame |
|---|---|---|---|
| `gemini-3.1-flash-lite` | 62 | 53 | 128 |
| `gemini-2.5-pro` | 31 | 99 | 112 |
| `gpt-5` | _re-running_ | _re-running_ | _re-running_ |
| `deepseek-reasoner` | 131 | 140 | 70 |
| `claude-opus-4-7` | 94 | 240 | 0 (not reached) |
| `claude-haiku-4-5-20251001` | 70 | 0 (not reached) | 0 (not reached) |
| `deepseek-chat` | 80 | 0 (not reached) | 0 (not reached) |

Three readings:

- **Openings are roughly memorized for every model.** ACPL ranges from 31 to 131 — meaningfully different, but all in the "competent club player or stronger" band. No model is *bad* in the opening.
- **Middlegame is where models start to break.** ACPL roughly doubles or triples for every model that reaches mid-game. This is the position-space where training-distribution coverage starts to drop sharply.
- **Endgame is only reached by the strongest cells.** Three of the four budget cells (Haiku, deepseek-chat, gpt-5-mini) never reach endgame at all. Anthropic's frontier reaches endgame in only some games. The Gemini cells, GPT-5, and DeepSeek-reasoner are the only models that consistently produce endgame moves to measure.

### Persistent wrong belief — case studies

The most striking single qualitative finding: when a model produces an illegal move, the SAME illegal SAN often recurs multiple times within a single game across retry attempts and across later plies.

**Example 1 — Opus proposes the same illegal king-capture five times in one game:**

```
ply 25: rationale "grabs the pawn on g4"           — illegal: would put king in check
ply 32: rationale "captures the g4 pawn for free"
ply 33: rationale "wins the g4 pawn"
ply 35: rationale "to grab material"
ply 36: rationale "capture the g4 pawn to remove threat"
```

White's king cannot capture the pawn on g4 — doing so exposes it to check from a distant attacker. Opus regenerates this same wrong move five times across 11 plies. Stockfish-substituted around it on every preceding attempt, but the model has no memory of failure across stateless calls, and the same FEN reproduces the same wrong pattern-match.

**Example 2 — Sonnet proposes a pawn advance for a pawn that does not exist:**

```
"Advancing the passed pawn to d7 attacks the black rook..."
"Advancing d6 pawn to d7 puts tremendous pressure..."
"Advancing the d6 pawn to d7 puts tremendous pressure..."
```

There is no white pawn that can advance to d7 in this position. The model believes in a pawn configuration that doesn't exist on the board, and the wrong belief is encoded in its response to that specific FEN.

The pattern is qualitatively different from typical "LLM hallucinations." A standard hallucination is a one-shot plausible-sounding wrong fact. This is **convergence on the same specific wrong belief across many independent invocations** — implying the belief is the model's deterministic response to a specific input, not random error. Fixing it via prompt engineering alone is hard; the wrong mental model regenerates from the same FEN.

### Failure-mode taxonomy

Across 224 illegal moves classified, **~95% are spatial-reasoning failures**:

| Category | % of failures | What broke |
|---|---|---|
| `leaves_in_check` | 27% | Line-of-sight from enemy piece to own king. Moves a pinned piece, or in-check and doesn't address it. |
| `path_blocked` | 23% | Square-by-square enumeration of an intermediate path. Slides a queen/rook/bishop through other pieces. |
| `king_into_check` | 15% | Attack-set calculation. Walks the king into an attacked square. |
| `phantom_source` | 12% | Per-piece position tracking. Piece type exists somewhere — just not where the model thinks. |
| `wrong_pawn_move` | 10% | Direction + distance reasoning for piece type. |
| `king_adjacent` | 5% | Adjacency relation on a 2D grid. |
| Other | 8% | Castling state, target blocked by own piece, etc. |

The non-spatial 5% is castling state and miscellaneous inventory errors. Two clean models tested (Claude Opus and Sonnet) showed essentially identical distribution — these failure modes are not model-specific.

### Retry feedback efficacy — what it fixes vs what it doesn't

When a move is illegal, the next attempt is told "your move X was illegal, try again" and given the list of recent failed attempts (capped to the last 3 to keep prompt size bounded). Across one game's 22 retries, 17 of 22 iterations went to a different piece type (model genuinely updated). The remaining 5 went to the same piece type with a different target, or repeated the same move.

The forfeits cluster on plies where **all retry attempts shared the same structural mistake.** A representative case: at one ply, Black was in check from a distant bishop. Opus proposed 4 moves over its retry budget — none addressed the check, because Opus didn't perceive the check at all. Every retry proposed a "good move" that ignored the check, because the model's mental model was missing the bishop's attack on the king.

**Retry feedback fixes "wrong move from a roughly-correct mental model" but cannot fix "wrong mental model of what's on the board."** The model can change its move but it cannot, via text feedback alone, change what it thinks it's looking at.

### Reasoning effort vs move quality — suggestive evidence

Reasoning models from OpenAI and Gemini support an effort dial (`reasoning_effort` for OpenAI, `thinking_budget` for Gemini). The benchmark logs the effort level on each call; on length-failure errors, the harness steps the effort down for the next retry (`default → medium → low → minimal`). Aggregating cp_loss bucketed by the effort level on the final successful attempt:

| Model | default cp / n | medium cp / n | low cp / n | minimal cp / n |
|---|---|---|---|---|
| `gemini-2.5-pro` | 62 / 210 | 47 / 17 | 52 / 7 | (n/a) |
| `gpt-5` | 1 / 47 | (n/a) | 64 / 25 | 67 / 154 |
| `gpt-5-mini` | 33 / 108 | 51 / 3 | 86 / 41 | 93 / 9 |

The pattern: **quality at lower reasoning effort is comparable to quality at default effort.** For GPT-5 specifically, only 47 of 255 (~18%) of successful moves were emitted at full reasoning effort; the remaining 80% required stepdown to low/minimal to fit the response budget. The cp_loss on minimal-effort moves (67) is in the same range as Flash Lite's overall cp_loss — i.e., near the matrix median — and not catastrophically worse than the default-effort moves the same model produced.

**This is suggestive but has a confound:** the default and stepped-down buckets sample different distributions of positions. Default succeeded on easier positions where reasoning was short; minimal was the fallback for harder positions where reasoning was long. The data does not yet control for position difficulty across effort levels. A controlled experiment (same positions, varying effort per call, paired cp_loss) is planned as next work.

If the controlled experiment shows quality is genuinely flat across reasoning levels on this benchmark, it would mean reasoning budget is largely unused capacity for chess — an actionable deployment finding. If quality drops sharply at low effort, it would mean the current matrix systematically under-measures reasoning models forced into stepdown.

### Hardest positions in the bank

Aggregated across all runs on the 20-position hand-curated bank:

| Position class | Failure rate | What the model misses |
|---|---|---|
| Distant-attacker check | 67% | Long-diagonal line-of-sight |
| Endgame with enemy pawn attacking king-escape squares | 67% | Pawn attack geometry |
| Promotion blocked by own king (Lucena position) | 33% | Target-square occupation |
| Phantom pawn moves in middlegame | 33% | Per-pawn position tracking |
| Mate detection in K+Q vs K | 80% claim error | Mate-in-1 perception in simple endgames |
| Standard opening positions (Italian, Ruy Lopez, etc.) | **0%** | Memorized opening theory |

The 0% failure rate on standard openings vs 33-67% on synthetic/constructed positions is the cleanest single test of "is this model reasoning or recalling" the benchmark provides.

### Legality and rule-claim consistency expose different weaknesses

The eval suite includes two single-position evals — `legality` (does the model produce a legal SAN move?) and `consistency` (does the model correctly describe what its moves do?). Across the data:

- 6 positions fail both
- 4 positions fail legality only
- **19 positions fail consistency only** — model picks a legal move, but its claim about whether it's check, capture, or mate is wrong
- 11 positions pass both

Consistency-only failures are the largest bucket. **The model picks correctly but can't always tell you what the move does.** Two different facets of the same underlying spatial-reasoning weakness.

---

## Methodology constraints encountered

Reasoning-tier models share a token-budget quirk that bites anyone evaluating them with structured output (tool calling, JSON mode). This benchmark hit it; here's the diagnosis and the fix.

### The token-exhaustion failure mode

OpenAI's `max_completion_tokens` and Google's `max_output_tokens` parameters cap the total tokens a reasoning model can produce, **including internal reasoning**. If the model spends the entire budget reasoning, there are zero tokens left for the visible output — including the forced tool call. The model returns with `finish_reason='length'`, no tool call, no content.

A naive eval harness records this as "model failed to follow the schema" → forfeit. But the failure is in the harness's budget, not the model's capability.

Anthropic's API does not have this problem — `max_tokens` there means *output* tokens only, with thinking budget tracked separately.

### Gemini also has a non-length failure mode

Gemini Pro and 2.5 Pro have a second failure mode where reasoning runs long enough to corrupt the structured tool output rather than hit a length cap — the API reports `finish_reason='MALFORMED_FUNCTION_CALL'` instead of `'length'`. Different proximate cause, same root (reasoning overrun corrupts tool emission).

### The fix the benchmark applies

Four layers, in order of necessity:

1. **Generous `max_tokens` ceiling (65536) for reasoning-capable providers.** Well below OpenAI's 128K output limit, but far above the actual reasoning-token usage on >99% of calls. Closes the bug for typical positions.

2. **Per-provider ceiling for Anthropic at 8192.** Anthropic's SDK has a built-in guard that rejects calls with very high `max_tokens` unless the call is streamed (which the adapter doesn't currently support). 8192 is plenty for a chess tool call's output and stays under the streaming threshold.

3. **`previously_failed_sans` list capped to last 3 attempts.** Each retry passes the model the list of previous failed SAN attempts. After 5-10 retries the prompt grows long and reasoning expands to re-analyze everything. Capping to 3 keeps prompt size bounded without losing the "don't repeat these" signal.

4. **Gradual `reasoning_effort` fallback ladder on overrun failures.** If a call hits `finish_reason='length'` OR `MALFORMED_FUNCTION_CALL` despite the above, the next retry on that move drops reasoning effort one notch: `default → medium → low → minimal`. Adapters translate the cross-provider effort ladder into provider-specific budgets (`reasoning_effort` on OpenAI, `thinking_config.thinking_budget` on Gemini, no-op on Anthropic where output and thinking are budgeted separately). This activates *only after* the model demonstrates it can't fit at the current effort — the benchmark does NOT a-priori set low reasoning effort. A-priori reduction would handicap the model and produce data that under-measures its capability.

The combination makes the benchmark robust: most calls land at full reasoning effort with no token issues; pathological positions get a graceful fallback; and the data is honest about which calls used reduced reasoning (the `reasoning_effort_for_this_attempt` is logged per call).

### What the matrix exposes about the reasoning-budget battle

For OpenAI's reasoning tier in particular, the response-budget interaction with chess positions is real. With the corrected `max_tokens=65536`, GPT-5 produces a successful first-attempt tool call on 97.8% of plies — the highest in the matrix. The remaining 2.2% drives the avg-retries-per-move of 0.15 and a meaningful share of the composite-score deficit relative to the matrix top.

### Gemini Pro Preview's daily quota

`gemini-3.1-pro-preview` is on a 250 requests/day cap, applied per model regardless of paid tier. Google enforces this on preview-track models; only GA models (like `gemini-2.5-pro`) have unrestricted per-tier quotas. A Reliability + PlayQuality gauntlet on a reasoning-heavy model with retries can burn 300-400 requests in a single run, so the cap is binding. The benchmark publishes the `gemini-2.5-pro` substitute for the frontier-Google cell.

---

## Caveats and expectations

**Caveats:**

- **Sample sizes are small by default** (5 games for Reliability, 3 for PlayQuality). Enough for qualitative pattern signal but not for tight effect sizes. A 0.02 difference between two models could be sampling noise. Raise `--games` for publishable comparisons.
- **Stockfish version matters.** Move-quality scores depend on the engine's evaluation function. Lock the binary version when comparing across runs. This work used Stockfish 18.
- **Provider tool-calling differences.** Each provider's structured-output format works slightly differently. The adapters normalize these; a regression on a specific provider can show up as a benchmark regression. Always grep the run JSONL for `"did not call submit_move"` before drawing conclusions.
- **This is not a chess-skill benchmark.** A model that's bad at chess but rule-consistent would score well. The eval measures state-tracking and rule-following; ELO is incidental.

**Reasonable expectations:**

- LLMs on this benchmark span 0.03 to 0.64 on Reliability. The top is a budget non-reasoning model; the bottom is two budget models and one frontier model. Reasoning-tier optimization is neither necessary nor sufficient for high scores.
- The first-try legal rate column is the most diagnostic single number — it reads independently of how forgiving the harness is. Read it alongside the composite.
- Opening positions show **0% failure rate** for every model tested. Failures concentrate on positions outside training distribution.
- The scoring is designed to be hill-climbable. Today's top (0.64) leaves real room above. Engine-level play would score 0.95+. The gap is the cognitive headroom.

---

## How to reproduce

```powershell
# Install
git clone https://github.com/GlacianNex/llm-chess-eval.git
cd llm-chess-eval
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e '.[all,dev]'

# Stockfish required on PATH or STOCKFISH_PATH env var (use Stockfish 18 for reproducibility)

# Provider keys for whichever you want
$env:ANTHROPIC_API_KEY = "sk-ant-..."
$env:OPENAI_API_KEY    = "sk-..."
$env:GOOGLE_API_KEY    = "AIza..."
$env:DEEPSEEK_API_KEY  = "sk-..."

# Sanity check
llm-chess-eval check-env

# Single-model composite metrics
llm-chess-eval reliability   --model claude-opus-4-7 --games 5
llm-chess-eval play-strength --model claude-opus-4-7 --games 3

# Full cross-provider matrix (~$25-40 total)
llm-chess-eval benchmark --dry-run    # preview, no spend
llm-chess-eval benchmark

# Component evals for diagnostic depth
llm-chess-eval legality    --model claude-opus-4-7
llm-chess-eval consistency --model claude-opus-4-7

# Re-score existing runs with the canonical formula (no API spend)
python scripts/matrix_with_retry_context.py
```

### Cost and runtime (per model, default config)

| Model class | Reliability + PlayQuality cost | Wall time |
|---|---|---|
| Mid-tier non-reasoning (Haiku, Flash Lite) | ~$0.30-0.50 | ~6-15 min |
| Strong reasoning (Opus) | ~$3-5 | ~12-30 min |
| Slow reasoning (GPT-5, Gemini Pro Preview, DeepSeek-reasoner) | ~$5-15 | ~30-180 min |

Full matrix: **~$25-40 in API spend, ~3-6 hours wall time** if jobs run sequentially. Parallelize across providers to cut wall time roughly in half. Single Stockfish skill, single position bank — no per-cell methodological knobs to tune.

### Repository layout

```
src/llm_chess_eval/
  adapters/
    _shared.py                # SYSTEM_PROMPT, SUBMIT_MOVE_PARAMETERS, build_user_message
    base.py                   # ModelAdapter protocol
    claude.py / openai.py / gemini.py / openai_compat.py / factory.py
  evals/
    legality.py / consistency.py
    games.py                  # game loop (forfeit/substitute/retry) + per-move progress logging
    chess_reliability.py      # ChessReliability metric — canonical scoring
    play_strength.py          # PlayQuality metric — canonical scoring
  analytics/
    accumulation.py           # per-move error rate, survival curves
    illegal_taxonomy.py       # classifies why each illegal move failed
    report.py                 # auto-generated scorecard
  harness/
    runner.py / game_runner.py
  cli.py / config.py / types.py

data/positions/legality_v1.jsonl   # 20-position bank
runs/<timestamp>__<eval>__<model>/ # raw JSONL output of each run
scripts/                            # analysis + monitor helpers
```

### Adding a new provider

1. Implement an adapter in `src/llm_chess_eval/adapters/` exposing `propose_move(fen, prior_failed=None, augment_legal_moves=None, reasoning_effort_override=None) -> CallOutcome` (the `ModelAdapter` protocol in `adapters/base.py`).
2. Reuse the shared prompt/schema/parsing helpers in `_shared.py`. Wrap them in the provider's tool-call shape.
3. Register the provider in `provider_for_model()` and `KNOWN_MODELS` in `config.py`, and in `build_adapter()` in `adapters/factory.py`.

For OpenAI-API-compatible endpoints (Together, Groq, Ollama, vLLM, custom servers), reuse `OpenAICompatibleAdapter` in `adapters/openai_compat.py` with the right `base_url` and `api_key_env_var` — no new adapter file needed.

---

## Open questions

What would tighten the findings:

1. **Larger N per cell.** 5/3 games is enough for qualitative signal; 10-30 per cell is needed for tight effect sizes.
2. **Skill sweep.** Reliability and PlayQuality at multiple Stockfish skills — how does the cascade interact with opponent strength?
3. **Mid-game starting positions.** Does the cascade need game-length state accumulation, or does it appear immediately from a complex mid-game FEN? Separates "drift across turns" from "complex positions are harder regardless."
4. **Reasoning-trace inspection across retries.** Save every retry's full response (currently only the final). Measures whether the model genuinely updates between retries vs pattern-matching a different SAN.
5. **Controlled reasoning-effort experiment.** Same positions × varying effort per call × paired cp_loss. Resolves the confound in the current reasoning-effort-vs-quality observation.
6. **Other 2D-state substrates.** Build analogous benchmarks for grid-navigation, UI layout reasoning, or tile-based puzzles. Confirms the spatial-reasoning failure profile is universal across 2D-grounded tasks, not chess-specific.

---

## The bottom line

The eval measures whether LLMs can maintain a 2D state of typed entities and correctly compute geometric queries against it across many reasoning steps. The current matrix top is 0.639 — a budget non-reasoning model from Google playing legal-on-first-try ~88% of the time and reaching mid/endgame with mediocre move quality. Frontier reasoning models from Anthropic, OpenAI, and DeepSeek either struggle on first-try legality (Anthropic) or score below the budget top on the composite anyway (OpenAI, DeepSeek). The bottom of the matrix is 0.032 — models that forfeit before reaching mid-game.

Chess gives a domain with deterministic ground truth, calibrated difficulty (memorized openings vs unique mid/endgames), and rules whose application is purely geometric. Models describe these rules verbally with 99% accuracy while applying them spatially at much lower rates. The benchmark exposes this gap with a small, cheap, reproducible test — a structural weakness shared by frontier and budget models alike on a generalizable cognitive dimension.

The most striking single qualitative finding is the **persistent wrong belief** pattern: models don't make random illegal moves; they form coherent-but-wrong pictures of the position and commit to them across multiple stateless API calls. This is qualitatively different from typical hallucinations and suggests fixing it requires architectural changes — not just bigger context windows or more training data.

The scoring is designed to be **hill-climbable**. The current matrix top (0.64) leaves real room above 0.9 where engine-quality play would land. The headroom is not noise — it's the gap between today's frontier and a model that can keep its mental picture of an 8×8 board accurate for 40 moves.
