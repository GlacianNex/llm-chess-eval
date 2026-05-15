# LLM Chess Eval — Methodology

How the benchmark measures what it measures: the design hypothesis, scoring formulas, parameter choices, and what each diagnostic captures. For how to install and run the eval, see **[HOWTO.md](HOWTO.md)**. For the matrix and findings, see **[RESULTS.md](RESULTS.md)**.

---

## Why this benchmark exists

Public LLM evaluations measure knowledge (MMLU, GPQA) or single-shot problem solving (MATH, HumanEval). Both reward models with strong training data on the relevant domain. Neither isolates whether a model can **maintain coherent state and apply known rules across many turns** — the cognitive dimension that determines whether you can trust a model on long agentic tasks.

That gap matters in practice. The same model that scores 90% on a coding benchmark can confidently propose a 3-step refactor that references a function it imagined exists, then double down on the false reference when asked to revise. The same model that solves a math problem flawlessly will solve a similar-looking problem with one altered structural detail by applying the original problem's pattern. These are not knowledge failures — they're **state-reasoning failures** that public evals do not systematically catch.

**Chess is the cleanest probe.** A few properties make it ideal:

- A small, fully-specified 2D state (8×8 grid, ~32 typed entities) — every position is exhaustively described in a FEN string.
- Rules whose application is purely geometric — line-of-sight, paths, adjacency, attack-set membership.
- Two oracles: `python-chess` for deterministic rule-checking (no judgment), Stockfish for strength evaluation.
- A built-in difficulty calibration. The first 5-10 moves of nearly every game are saturated in training data. After that, branching combinatorics push games into positions essentially never repeated. **The same model on the same task shifts from pattern-recall to spatial reasoning within ~10 moves.** We see the cliff happen.

The benchmark runs across any provider (Anthropic, OpenAI, Google, DeepSeek, OpenAI-API-compatible endpoints) through a single CLI.

---

## The design hypothesis: cumulative coherence and the (model-specific) memorization cliff

The benchmark was designed around one specific claim about how LLMs play chess: **performance degrades across a game's length, not because positions get harder in absolute terms, but because the model has to sustain coherent state-tracking turn after turn.** The phase weight in the metric encodes this. The hypothesis came in two parts originally; the data supports one cleanly, the other only partially.

### Part 1 (supported): cumulative-coherence failure across turns

What the data shows: **most models that survive to all three phases of a game play progressively worse moves as the game proceeds.** ACPL by phase from the matrix:

| Model | opening | middlegame | endgame |
|---|---|---|---|
| `gemini-2.5-pro` | 34 | 78 | 168 |
| `gemini-3.1-flash-lite` | 45 | 77 | 114 |
| `claude-opus-4-7` | 66 | 103 | 123 |
| `gpt-5` | 85 | 111 | 102 |
| `deepseek-reasoner` | 85 | 172 | 118 |

Every cell that reaches the endgame shows the gradient, with one exception: GPT-5 is roughly flat across phases (consistently mid-quality everywhere). For weaker cells that never reach the endgame at all (Haiku, DeepSeek-chat — 80–95% forfeit rate), the cumulative-coherence claim is supported even more starkly: they can't survive long enough *for* the cliff to bite. The phase weight in the metric (1 / 1.5 / 2 / 3) is justified primarily by this: surviving to ply 30+ with maintained quality is harder than playing ply 5 well, and the metric should reward it.

### Part 2 (model-specific, not universal): the per-position memorization cliff

The stronger original hypothesis was that the cumulative degradation is driven by **training-data novelty**: opening positions are memorized, endgame positions are essentially unique, models lose their crutch and fall off a cliff. To test this directly we built four position banks of increasing training-novelty (T0 hand-curated → T1 real-play extracted → T2 random-opening + Stockfish continuation → T3 pure random-vs-random) and measured per-position legality. The result is messier than the hypothesis predicted:

| Model | T0 hand | T1 real-play | T2 random-open | T3 random-play |
|---|---|---|---|---|
| `gpt-5` | **1.000** | **1.000** | **1.000** | **1.000** |
| `gemini-3.1-flash-lite` | 0.950 | 0.900 | 0.900 | 0.900 |
| `claude-sonnet-4-6` | 0.750 | 0.500 | 0.750 | 0.700 |
| `claude-opus-4-7` | 0.900 | 0.300 | 0.775 | 0.800 |
| `deepseek-chat` | 0.500 | 0.200 | 0.200 | 0.250 |

GPT-5 and Flash Lite handle progressively-more-novel positions essentially as well as memorized ones. Sonnet is similarly flat. Only DeepSeek-chat shows a sharp per-position cliff. **The "training-novelty drives failure" claim is therefore model-specific, not universal.** Sonnet illustrates the gap most cleanly: zero per-position cliff (handles novel positions fine in isolation) but a massive in-game ACPL gradient (104 → 235 from opening to middlegame). For Sonnet, what fails is *cumulative coherence across turns*, not per-position novelty.

### What this means for the metric design

The phase weight rewards reaching late plies regardless of which mechanism makes that hard:

- For models with a per-position cliff (DeepSeek-chat, partly Opus), late plies are hard because the positions themselves drift out of training distribution.
- For models without a per-position cliff (GPT-5, Flash Lite, Sonnet), late plies are hard because state-tracking across many turns is hard — they handle individual novel positions fine but lose coherence over a 40-ply game.

Both mechanisms make "sustain quality into the endgame" the right thing to measure, and the phase weight encodes that without committing to which mechanism is doing the work. **The metric is right; the original "universal memorization cliff" framing was too strong.**

That nuance is also why we report two complementary tests: the per-game composites (which capture cumulative-coherence failure across turns) and the per-position bank gradient (which catches the subset of models with per-position novelty effects). See [RESULTS § The in-game cliff is NOT explained by position novelty](RESULTS.md#the-in-game-cliff-is-not-explained-by-position-novelty) for the deep dive.

---

## What the benchmark measures

Two scores per model, both bounded `[0, 1]`:

### PlayStrength (primary)

The model plays games vs Stockfish at skill 3 — approximately **1500 ELO, an intermediate amateur**. On an illegal proposed move, the model is told "that was illegal" and asked to try again — up to 10 retries per move. If all retries fail, the game forfeits.

The opponent is amateur-tier deliberately. PlayStrength measures the model's combined rule-following discipline and move quality across a full game — not chess strength against engines. A stronger opponent would just force the model into harder positions faster without telling us anything new about whether it can keep its mental picture of the board accurate.

Per-move score is a product of three factors:

```
per_move_score  =  move_quality(cp_loss)  ×  retry_cost(retries)  ×  game_phase_weight(ply)
```

Where:

- **`move_quality(cp_loss) = exp(-cp_loss / 150)`** — exponential decay over centipawn loss vs Stockfish's top move. Engine-level play (cp_loss ~5) scores ~0.97; grandmaster-level (cp_loss ~50) scores ~0.72; competent club (cp_loss ~150) scores ~0.37; blunder (cp_loss ~500) scores ~0.04. The exponential shape leaves real headroom at the top of the scale — a linear quality function (used in earlier iterations) squashed Stockfish-quality and intermediate play together at the top.

- **`retry_cost(retries) = 0.25 ^ retries`** — steep multiplicative penalty for needing the retry safety net. One retry costs 75% of the move's value; two retries cost 94%. In a real chess game, an illegal move is fatal; the benchmark is a generosity-graded approximation, and the 0.25 base ensures retried moves contribute almost nothing to the score.

- **`game_phase_weight(ply) = 1 / 1.5 / 2 / 3`** — softened weighting by ply bucket (boundaries at ply 10, 20, 30). Late plies are weighted more because reaching them with maintained quality requires sustained state-tracking — see [the design hypothesis](#the-design-hypothesis-cumulative-coherence-and-the-model-specific-memorization-cliff) for what the data does and doesn't support. The shape makes "reaching ply 30 with maintained quality" worth 3× more than "playing ply 5 perfectly" without letting the denominator be dominated by late plies — earlier iterations used 1/2/4/8 but that compressed scores too aggressively for models that broke down in middlegame.

Per-game score:

```
per_game_score  =  sum(per_move_score for legal moves)  /  max_possible_weighted_score
```

The denominator (`max_possible_weighted_score = sum(game_phase_weight(p) for p in 1..max_moves_per_game)`) is constant for a given `max_moves_per_game`. Unplayed plies (after a forfeit) contribute 0 to the numerator but their phase weight is still in the denominator. So an early forfeit loses BOTH the missing per-move scores AND access to the high-weight late plies — forfeit penalty scales with how much of the game was missed.

PlayStrength = mean of per_game_score across N games. Published matrix uses N ≥ 20 games per cell (some cells went to N = 100+); max 40 plies per game.

### PlayQuality (supplemental)

PlayQuality is the supplemental companion to PlayStrength. It strips the `retry_cost` factor out of the per-move formula and runs against a harder Stockfish opponent (skill 5, ~1700 ELO) with a tighter retry budget (max 3). PlayQuality scores are *not* "performance vs engine" — they're "move quality across full games when the opponent is amateur-tier and retries are free." PlayQuality measures **how strong the model's moves are once it has found a legal one** — it doesn't penalize retry use.

Use PlayQuality when you want to isolate move strength from rule-following discipline. PlayStrength is the right number for headline cross-model comparison; PlayQuality is the right number for answering "is this model weak on rules, or weak on chess?"

(For context on the full Stockfish skill range: skill 0 ≈ 1100 ELO beginner; skill 3 ≈ 1500 amateur — what PlayStrength uses; skill 5 ≈ 1700 intermediate amateur — what PlayQuality uses; skill 15 ≈ 2500 strong engine; skill 20 ≈ 2850 top engine. None of the current benchmark cells play against an engine-strength opponent.)

Per-move score:

```
per_move_score  =  move_quality(cp_loss)  ×  game_phase_weight(ply)
```

Per-game score:

```
per_game_score  =  sum(per_move_score for legal moves)  /  max_possible_weighted_score
```

PlayQuality = mean across N ≥ 10 games (most cells N = 50+), max 60 plies per game.

The difference from PlayStrength is intentional and conceptual: PlayStrength = "how well does the model play legal chess, with rule-following discipline penalized"; PlayQuality = "given a legal move was found, how good was it." Both share the exponential quality decay and softened phase weight (1/1.5/2/3), so their numbers are directly comparable at the per-move level.

### When a game forfeits

A game ends **naturally** when:

- One side delivers checkmate or causes stalemate
- The position reaches `max_plies` (40 for PlayStrength, 60 for PlayQuality)
- A draw rule fires (insufficient material, 50-move rule, threefold repetition)

A game ends in **forfeit** when the model cannot produce a legal move within the retry budget for a single ply. The mechanic, in detail:

1. Model proposes move M for the current FEN.
2. Harness validates M against `python-chess`. If illegal, M is added to a "recent failures" list and the model is re-prompted: *"Your move X was illegal because Y. Recent failed attempts: [last 3]. Try again."*
3. Steps 1–2 repeat up to `max_retries` times (10 for PlayStrength, 3 for PlayQuality).
4. If the model still hasn't produced a legal move after `max_retries`, the game terminates with result `forfeit_illegal`. The remaining plies up to `max_plies` are not played.

**A forfeit doesn't just end the game — it produces zero contribution to the score numerator from all remaining plies, while the denominator still accounts for the full max_plies.** So a forfeit at ply 5 of a 40-ply game keeps only ~6% of the achievable score; a forfeit at ply 20 keeps ~30%. Forfeits are *intentionally* counted at zero because forfeit-prone play is exactly the failure mode the benchmark exists to measure — we don't want to exclude forfeit games and report "score among games that completed," because that would hide the most important signal.

**Forfeit rate is the cleanest single diagnostic in the matrix.** GPT-5 and DeepSeek-reasoner forfeited 0% of games (0 / 166 and 0 / 112 respectively); Flash Lite 5% (8 / 160); Opus 30% (48 / 160); Sonnet 20% (29 / 142); Haiku 95% (168 / 176); DeepSeek-chat 80% (137 / 172). The forfeit rate ranks models without any scoring choices — it answers the simplest possible question: "can the model produce a legal move within the retry budget, every ply, end to end?"

### Why three multiplicative factors

Both metrics use multiplicative structure so any factor near zero collapses the score. The factor structure is deliberate:

- **A model that forfeits at move 5 can't rescue the score with great opening play.** The high-weight late plies are in the denominator regardless of whether they're played; the unplayed plies contribute 0 to the numerator. Forfeit penalty scales with how much game was left.
- **A model that drags out long games of mediocre moves can't rescue the score with completion.** `move_quality(cp_loss = 200) ≈ 0.26`; even a fully-completed game of such moves scores below 0.3.
- **A model that needs many retries to find legal moves can't rescue the score with eventual success.** `retry_cost = 0.25^n` makes a 2-retry move contribute 6% of its base value; a 4-retry move 0.4%. The recovery is recorded but doesn't compensate.

Additive scoring would let a model trade one bad axis for excellence on another. The cognitive failure we're measuring — coherent state-tracking across many turns — is *all-or-nothing*: a model with a 50% illegal-move rate isn't usefully "half-coherent," it's broken. The multiplicative collapse mirrors that.

### Design choices: parameter values and why

The non-obvious numerical choices in the metric all fall out of specific design pressures. None are arbitrary.

**Stockfish skill: 3 for PlayStrength, 5 for PlayQuality.**

- PlayStrength uses **skill 3 (~1500 ELO, amateur)** because we want games to *reach the endgame*. A harder opponent forces the LLM into difficult positions faster, where it forfeits or hits the ply cap before reaching the late-game phase weight. Skill 3 keeps games long enough that the phase weighting has signal across all plies.
- PlayQuality uses **skill 5 (~1700 ELO, intermediate amateur)** because once we're isolating move quality from rule-following, we want more discrimination at the top of the range. A harder opponent produces a wider cp_loss spread, which separates the strong-move-quality models more cleanly.
- Neither cell is engine-strength. We are not measuring "can the LLM beat Stockfish" — we are measuring "can the LLM play coherent chess across many turns." Opponent strength is calibrated to that goal.

**Retry budget: 10 for PlayStrength, 3 for PlayQuality.**

- PlayStrength's `retry_cost = 0.25^retries` factor means retries past 2–3 contribute almost nothing to the per-move score anyway (3 retries = 1.5% of move value). The 10-retry cap is generous on *survival* without changing the score — we want games to complete when possible so we have data on later plies. A model that needs all 10 retries on every move still finishes the game, but earns essentially zero per-move score.
- PlayQuality has *no* retry cost, so a generous retry budget would let weak-rule-following models brute-force their way to legal moves. We cap at 3 to prevent that. PlayQuality is asking "are your chosen moves good," not "can you eventually find something legal."

**Max plies: 40 for PlayStrength, 60 for PlayQuality.**

- 40 plies (20 LLM moves) covers opening + middlegame + early endgame for most games against skill 3. The phase weight peaks at ply 30+; 40 gives meaningful endgame exposure without exploding wall-clock cost on slow reasoning models.
- PlayQuality's 60-ply cap is longer because (a) skill 5 produces sharper game variance — some games end early, others extend — and (b) ACPL-by-phase diagnostics need games to actually reach the endgame phase to produce numbers.

**Sample size: N ≥ 20 per cell.**

- At N=5 (the default for solo runs), sampling noise can produce ±0.05–0.10 swings between models with similar true means. Two cells scoring 0.42 and 0.45 at N=5 might not be reliably different.
- The published matrix uses N ≥ 20 (most cells went to N = 50–170) which cuts sampling SD by at least 2× from the N=5 baseline. Score differences of ~0.02 are now meaningful.
- See the [τ sensitivity analysis](RESULTS.md#methodology-robustness--τ-sensitivity) for evidence that rankings are stable to scoring choices at this N.

**Retry mode (rather than forfeit-on-first-illegal).**

- A "one-shot legal" scoring (game ends on first illegal move) would be too noisy — a single tactical miscalculation early in the game would zero the entire game regardless of overall coherence.
- Retry-with-cost asks the model to *recover* from errors while penalizing recovery proportionally. The 0.25 base is steep enough that retried moves contribute negligibly; the 10-retry budget is large enough that games typically reach the high-weight late plies.
- This is the most informative test of state-tracking: did you make ONE mistake and recover, or are you systematically broken? One-shot scoring can't distinguish.

**Quality decay constant τ = 150.**

- `move_quality(cp_loss) = exp(-cp_loss/150)` anchors competent club play (cp_loss = 150) at quality = 0.37, grandmaster-level (cp_loss = 50) at 0.72, and engine-level (cp_loss = 5) at 0.97. The choice leaves real headroom at the top: a model improving from 50cp to 5cp average moves up from 0.72 to 0.97, which is visible in the composite.
- A linear quality function (used in earlier iterations) squashed Stockfish-quality and intermediate play together at the top.
- τ is not load-bearing: re-scoring at τ ∈ {100, 200, 300} produces identical model rankings with absolute scores shifting ~20%. See [RESULTS § τ sensitivity](RESULTS.md#methodology-robustness--τ-sensitivity).

### PlayStrength is NOT first-attempt-legal — they measure different things

The benchmark reports both **`first_attempt_legal_rate`** (a diagnostic — fraction of plies where the model's very first proposal was a legal SAN) and **`PlayStrength`** (the composite score). It's easy to assume "high first-attempt legality → high PlayStrength", but the composite has three factors and first-attempt-legality is only one of them. The other two can drag the score substantially even when legality is near-perfect:

- **`move_quality(cp_loss)`**: a legal move with cp_loss = 86 has quality = exp(−86/150) = 0.564, not 1.0. Reaching engine-level play (cp_loss ~5) requires move_quality ~0.97. Most LLMs play moves with cp_loss in the 50-200 range — competent club-player territory, scoring 0.27-0.72 per move on quality alone.
- **`game_phase_weight(ply)`**: ply-30+ moves are weighted 3×, ply-1-9 are weighted 1×. A model that forfeits early or stops at the max-ply cap before reaching late game loses access to the highest-weighted plies in the numerator while they remain in the denominator. The structural ceiling for a model that perfectly plays only to ply 20 (out of max 40) is ~0.30; perfectly to ply 30 is ~0.59; perfectly to ply 40 is 1.0.
- **A single full forfeit** (game ends at ply 1 because the model can't produce a legal move even after all retries) drops that game's contribution to 0 regardless of how the other games went. At N=20 games per cell, one forfeit subtracts ~0.05 from the mean — but cells with high forfeit rates (Haiku 95%, DeepSeek-chat 80%) have those zeroes dominating the mean.

A concrete example from the published matrix: **GPT-5 has 99.8% first-attempt-legal, 0.00 avg retries, and zero forfeits — but PlayStrength = 0.301.** Decomposing:

| Factor | Contribution |
|---|---|
| Mean cp_loss across legal moves: ~85 → quality = exp(−85/150) ≈ 0.57 | Caps the score around 0.57 even at perfect legality |
| Zero forfeits across 166 games | No penalty here |
| Games reach mean ply ~26 (vs max 40) | Misses some high-weight late plies; ~0.45 structural ceiling for typical-game-length |
| Retry cost (0.00 retries/move) | No penalty here |

The 0.301 composite is the product of "perfect legality × mid-range quality × partial late-game coverage." Read the columns together — first-attempt-legal alone is not a stand-in for PlayStrength. If GPT-5 played at engine quality (cp_loss ~5), the same legality and game-length profile would score ~0.93.

### Reference points

| Score | Equivalent |
|---|---|
| 1.000 | Theoretical max — engine-quality moves across all 40 plies, zero retries |
| 0.85-0.95 | Hypothetical: a model playing at engine-equivalent move quality |
| 0.40-0.55 | Best current LLMs on this benchmark (vs amateur-tier Stockfish opponent) |
| 0.25-0.40 | LLMs that survive games but play mid-range moves |
| 0.10-0.20 | LLMs that struggle past middlegame or have meaningful forfeit rates |
| 0.00-0.10 | High-forfeit-rate cells; the metric correctly reflects "model can't reliably complete games" |

The 1.0 reference is the theoretical maximum of the metric (engine-quality moves with no retries needed across a full game) — *not* a benchmark cell. The benchmark plays models against amateur-tier opponents (skill 3 / skill 5); the engine-quality calibration is the scoring anchor, not a comparison target.

Current matrix range: **0.074 (Claude Haiku) to 0.485 (Gemini 2.5 Pro)**. See [RESULTS.md](RESULTS.md) for the full matrix.

### Average Centipawn Loss (ACPL) — diagnostic

ACPL is the standard chess-strength metric: per move, the centipawn difference between Stockfish's top move and what the model played. ACPL 50 = strong club player; ACPL 150 = intermediate; ACPL 500+ = blundering. We report ACPL **by phase** (opening / middlegame / endgame) alongside the composite scores. The ACPL gradient across phases is the most direct evidence of cumulative-coherence failure — the metric the composite encodes structurally, broken out so you can see which phase a model degrades in.

---

## Caveats and expectations

**Caveats:**

- **Default `--games` is small** (5 for PlayStrength, 3 for PlayQuality). Enough for qualitative pattern signal but not for tight effect sizes. The published matrix used N ≥ 20 per cell (many cells N = 100+) for stable means. A 0.02 difference between two models at N=5 could be sampling noise; at N=20+ it's more durable.
- **Stockfish version matters.** Move-quality scores depend on the engine's evaluation function. Lock the binary version when comparing across runs. This work used Stockfish 18.
- **Provider tool-calling differences.** Each provider's structured-output format works slightly differently. The adapters normalize these; a regression on a specific provider can show up as a benchmark regression. Always grep the run JSONL for `"did not call submit_move"` before drawing conclusions.
- **Quota-corrupted games are dropped.** Per-game error filter removes games where any move hit a 429 / quota error mid-game (so the run isn't contaminated by half-played games where a provider's billing failed silently). Drop counts are reported alongside N per cell in the aggregated output.
- **This is not a chess-skill benchmark.** A model that's bad at chess but rule-consistent would score well. The eval measures state-tracking and rule-following; ELO is incidental.

**Reasonable expectations:**

- LLMs on this benchmark span 0.07 to 0.49 on PlayStrength. The top two are a Google frontier reasoning model and a Google budget non-reasoning model, essentially tied. Reasoning-tier optimization is neither necessary nor sufficient for high scores.
- The first-attempt-legal rate and forfeit-rate columns are the most diagnostic single numbers — first-legal measures the fraction of plies where the model's very first proposal (zero retries) was legal; forfeit-rate measures the fraction of games that ended because retries couldn't recover a legal move. Both read independently of scoring choices.
- Opening positions show **0% failure rate** for every model tested. Failures concentrate on positions outside training distribution.
- Choice of `move_quality` decay constant (τ=150) is not load-bearing: re-scoring at τ ∈ {100, 200, 300} produces identical model rankings with absolute scores shifting ~20%. See [RESULTS.md § τ sensitivity](RESULTS.md#methodology-robustness--τ-sensitivity).
- The scoring is designed to be hill-climbable. Today's top (0.49) leaves real room above. Engine-level play would score 0.95+. The gap is the cognitive headroom.

---

## Open questions

What would tighten the findings:

1. **Larger N per cell.** 5/3 games is enough for qualitative signal; 10-30 per cell is needed for tight effect sizes.
2. **Skill sweep.** PlayStrength and PlayQuality at multiple Stockfish skills — how does the cascade interact with opponent strength?
3. **Mid-game starting positions.** Does the cascade need game-length state accumulation, or does it appear immediately from a complex mid-game FEN? Separates "drift across turns" from "complex positions are harder regardless."
4. **Reasoning-trace inspection across retries.** Save every retry's full response (currently only the final). Measures whether the model genuinely updates between retries vs pattern-matching a different SAN.
5. **Controlled reasoning-effort experiment.** Same positions × varying effort per call × paired cp_loss. Resolves the confound in the current reasoning-effort-vs-quality observation.
6. **Other 2D-state substrates.** Build analogous benchmarks for grid-navigation, UI layout reasoning, or tile-based puzzles. Confirms the spatial-reasoning failure profile is universal across 2D-grounded tasks, not chess-specific.

---

For the matrix, findings, deep dives, and the Anthropic-score explanation, see **[RESULTS.md](RESULTS.md)**. For provider quirks and how to run the eval, see **[HOWTO.md](HOWTO.md)**.
