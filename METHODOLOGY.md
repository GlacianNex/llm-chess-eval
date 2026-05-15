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
- A built-in difficulty calibration. The first 5–10 moves of nearly every game are saturated in training data. After that, branching combinatorics push games into positions essentially never repeated. **The same model on the same task shifts from pattern-recall to spatial reasoning within ~10 moves.** We see the cliff happen.

The benchmark runs across any provider (Anthropic, OpenAI, Google, DeepSeek, OpenAI-API-compatible endpoints) through a single CLI.

---

## The design hypothesis

The benchmark is designed around one specific claim about how LLMs play chess: **performance degrades across a game's length, not because positions get harder in absolute terms, but because the model has to sustain coherent state-tracking turn after turn.** The phase weight in the metric encodes this.

Two specific mechanisms could produce that degradation. The benchmark separates them deliberately:

**Mechanism A — cumulative-coherence failure across turns.** State-tracking errors accumulate as the model makes more independent stateless inferences over a game. Even on positions the model could handle in isolation, the cumulative drift across 30–40 turns can corrupt its mental model of the board. Tested by **ACPL-by-phase within full games**: rising ACPL through opening → middlegame → endgame is the signal. The phase weight (1 / 1.5 / 2 / 3) rewards models that resist this drift.

**Mechanism B — per-position memorization cliff.** Chess has ~10^120 distinct positions; openings (~5,000 named) appear thousands of times in training corpora, mid/endgame positions are essentially unique. A model that pattern-matches without genuine state-tracking should show a *per-position* legality cliff as positions drift out of training distribution, regardless of how the position was reached. Tested by **per-position legality on banks of increasing training-novelty** (hand-curated → real-play → random-opening → pure random self-play).

The two mechanisms are independent in principle — a model can fail at one without failing at the other. We test both because the cognitive failure they imply is different: A is about drift across stateless calls; B is about training-data dependence on the position itself.

**The metric design is robust to which mechanism dominates.** The phase weight rewards "sustain quality into the endgame," which is the right thing to measure whether late plies are hard because of state-tracking drift (A) or because the positions themselves are out-of-distribution (B). Two diagnostics break the mechanism out:

- **ACPL by phase** — direct signal for mechanism A. Rising ACPL = cumulative drift.
- **Per-position legality across novelty-tiered banks** — direct signal for mechanism B. Per-position cliff = training-novelty effect.

For the data showing which mechanism dominates for each model in the matrix, see [RESULTS § The in-game cliff is NOT explained by position novelty](RESULTS.md#the-in-game-cliff-is-not-explained-by-position-novelty).

---

## What the benchmark measures

Two scores per model, both bounded `[0, 1]`:

- **PlayStrength** (primary) — composite that combines move quality, rule-following discipline, and game-phase progression into one number. Headline cross-model metric.
- **MoveQuality** (supplemental) — strips the rule-discipline factor and runs against a harder opponent. Isolates "how good are the moves themselves, once a legal one is found."

PlayStrength is what you compare across cells. MoveQuality is what you reach for when you need to ask "is this model weak on rules, or weak on chess?"

### PlayStrength formula

The model plays games vs Stockfish at skill 3 (~1500 ELO amateur). Per-move score is the product of three factors:

```
per_move_score  =  move_quality(cp_loss)  ×  retry_cost(retries)  ×  game_phase_weight(ply)
```

- **`move_quality(cp_loss) = exp(-cp_loss / 150)`** — exponential decay over centipawn loss vs Stockfish's top move. Engine-level play (cp_loss ~5) scores ~0.97; grandmaster-level (cp_loss ~50) scores ~0.72; competent club (cp_loss ~150) scores ~0.37; blunder (cp_loss ~500) scores ~0.04.

- **`retry_cost(retries) = 0.25 ^ retries`** — steep multiplicative penalty for needing the retry safety net. One retry costs 75% of the move's value; two retries cost 94%. The 0.25 base ensures retried moves contribute almost nothing.

- **`game_phase_weight(ply) = 1 / 1.5 / 2 / 3`** — boundaries at ply 10, 20, 30. Late plies count more because reaching them with maintained quality requires sustained state-tracking. The shape makes "reaching ply 30 with maintained quality" worth 3× more than "playing ply 5 perfectly" without letting the denominator be dominated by late plies.

Per-game score:

```
per_game_score  =  sum(per_move_score for legal moves)  /  max_possible_weighted_score
```

The denominator (`max_possible_weighted_score = sum(game_phase_weight(p) for p in 1..max_plies)`) is constant for a given `max_plies`. Unplayed plies (after a forfeit) contribute 0 to the numerator but their phase weight is still in the denominator — so an early forfeit loses BOTH the missing per-move scores AND access to the high-weight late plies.

PlayStrength = mean of per_game_score across N games. Published matrix uses N ≥ 20 games per cell (some cells went to N = 100+); max 40 plies per game.

### MoveQuality formula

MoveQuality strips the `retry_cost` factor and runs against a harder Stockfish opponent (skill 5, ~1700 ELO). Once a legal move is found, it's scored on its strength alone — retries are free.

```
per_move_score  =  move_quality(cp_loss)  ×  game_phase_weight(ply)
per_game_score  =  sum(per_move_score for legal moves)  /  max_possible_weighted_score
```

MoveQuality = mean across N ≥ 10 games (most cells N = 50+), max 60 plies per game.

Both metrics share the exponential quality decay and the softened phase weight (1/1.5/2/3), so their numbers are directly comparable at the per-move level. The conceptual split is intentional: PlayStrength = "how well does the model play legal chess with rule-following discipline penalized"; MoveQuality = "given a legal move was found, how good was it."

### Why three multiplicative factors

Both metrics use multiplicative structure so any factor near zero collapses the score. The factor structure is deliberate:

- **A model that forfeits at move 5 can't rescue the score with great opening play.** The high-weight late plies are in the denominator regardless of whether they're played; unplayed plies contribute 0 to the numerator. Forfeit penalty scales with how much game was left.
- **A model that drags out long games of mediocre moves can't rescue the score with completion.** `move_quality(cp_loss = 200) ≈ 0.26`; even a fully-completed game of such moves scores below 0.3.
- **A model that needs many retries to find legal moves can't rescue the score with eventual success.** `retry_cost = 0.25^n` makes a 2-retry move contribute 6% of its base value; a 4-retry move 0.4%. The recovery is recorded but doesn't compensate.

Additive scoring would let a model trade one bad axis for excellence on another. The cognitive failure we're measuring — coherent state-tracking across many turns — is *all-or-nothing*: a model with a 50% illegal-move rate isn't usefully "half-coherent," it's broken. The multiplicative collapse mirrors that.

### How games end

A game ends **naturally** when:

- One side delivers checkmate or causes stalemate
- The position reaches `max_plies` (40 for PlayStrength, 60 for MoveQuality)
- A draw rule fires (insufficient material, 50-move rule, threefold repetition)

A game ends in **forfeit** when the model cannot produce a legal move within the retry budget for a single ply. The mechanic:

1. Model proposes move M for the current FEN.
2. Harness validates M against `python-chess`. If illegal, M is added to a "recent failures" list and the model is re-prompted: *"Your move X was illegal because Y. Recent failed attempts: [last 3]. Try again."*
3. Steps 1–2 repeat up to `max_retries` times (10 for PlayStrength, 3 for MoveQuality).
4. If the model still hasn't produced a legal move after `max_retries`, the game terminates with result `forfeit_illegal`. The remaining plies up to `max_plies` are not played.

**A forfeit doesn't just end the game — it produces zero contribution to the numerator from all remaining plies, while the denominator still accounts for the full max_plies.** A forfeit at ply 5 of a 40-ply game keeps only ~6% of the achievable score; a forfeit at ply 20 keeps ~30%. Forfeits are *intentionally* counted at zero because forfeit-prone play is exactly the failure mode the benchmark exists to measure.

The benchmark reports **forfeit rate** as a top-line diagnostic alongside the composite. It ranks models without any scoring choices — it answers the simplest possible question: "can the model produce a legal move within the retry budget, every ply, end to end?"

### Reading the scores

What different score levels imply structurally about the model that produced them:

| Score | Implication |
|---|---|
| 1.000 | Theoretical max — engine-quality moves across all 40 plies, zero retries |
| 0.85–0.95 | Engine-equivalent move quality maintained across a full game |
| 0.40–0.55 | Reaches endgame consistently; plays at competent-club quality throughout |
| 0.25–0.40 | Completes most games but at mid-range move quality, or strong play that doesn't reach endgame |
| 0.10–0.20 | Struggles past middlegame, or has meaningful forfeit rates |
| 0.00–0.10 | High-forfeit-rate; the metric correctly reflects "model can't reliably complete games" |

The 1.0 reference is the theoretical maximum of the metric (engine-quality moves with no retries needed across a full game) — *not* a benchmark cell. The benchmark plays models against amateur-tier opponents (skill 3 / skill 5); the engine-quality calibration is the scoring anchor, not a comparison target.

For where actual cells land on this scale, see [RESULTS.md](RESULTS.md).

### ACPL by phase — the cliff diagnostic

ACPL (Average Centipawn Loss) is the standard chess-strength metric: per move, the centipawn difference between Stockfish's top move and what the model played. ACPL 50 = strong club player; ACPL 150 = intermediate; ACPL 500+ = blundering. We report ACPL **by phase** (opening / middlegame / endgame) alongside the composite scores. The ACPL gradient across phases is the most direct signal for cumulative-coherence failure — the metric the composite encodes structurally, broken out so you can see *which phase* a model degrades in.

---

## Design choices: parameter values and why

The non-obvious numerical choices in the metric all fall out of specific design pressures. None are arbitrary.

**Stockfish skill: 3 for PlayStrength, 5 for MoveQuality.**

- PlayStrength uses **skill 3 (~1500 ELO, amateur)** because we want games to *reach the endgame*. A harder opponent forces the LLM into difficult positions faster, where it forfeits or hits the ply cap before reaching the late-game phase weight. Skill 3 keeps games long enough that the phase weighting has signal across all plies.
- MoveQuality uses **skill 5 (~1700 ELO, intermediate amateur)** because once we're isolating move quality from rule-following, we want more discrimination at the top of the range. A harder opponent produces a wider cp_loss spread, which separates strong-move-quality models more cleanly.
- Neither cell is engine-strength. We are not measuring "can the LLM beat Stockfish" — we are measuring "can the LLM play coherent chess across many turns." Opponent strength is calibrated to that goal.

(For full context on the Stockfish skill range: skill 0 ≈ 1100 ELO beginner; skill 3 ≈ 1500 amateur — PlayStrength; skill 5 ≈ 1700 intermediate amateur — MoveQuality; skill 15 ≈ 2500 strong engine; skill 20 ≈ 2850 top engine. None of the current benchmark cells play against an engine-strength opponent.)

**Retry budget: 10 for PlayStrength, 3 for MoveQuality.**

- PlayStrength's `retry_cost = 0.25^retries` factor means retries past 2–3 contribute almost nothing to the per-move score anyway (3 retries = 1.5% of move value). The 10-retry cap is generous on *survival* without changing the score — we want games to complete when possible so we have data on later plies. A model that needs all 10 retries on every move still finishes the game, but earns essentially zero per-move score.
- MoveQuality has *no* retry cost, so a generous retry budget would let weak-rule-following models brute-force their way to legal moves. We cap at 3 to prevent that. MoveQuality is asking "are your chosen moves good," not "can you eventually find something legal."

**Max plies: 40 for PlayStrength, 60 for MoveQuality.**

- 40 plies (20 LLM moves) covers opening + middlegame + early endgame for most games against skill 3. The phase weight peaks at ply 30+; 40 gives meaningful endgame exposure without exploding wall-clock cost on slow reasoning models.
- MoveQuality's 60-ply cap is longer because (a) skill 5 produces sharper game variance — some games end early, others extend — and (b) ACPL-by-phase diagnostics need games to actually reach the endgame phase to produce numbers.

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

---

## Pitfalls when interpreting scores

### PlayStrength is NOT first-attempt-legal — they measure different things

The benchmark reports both **`first_attempt_legal_rate`** (a diagnostic — fraction of plies where the model's very first proposal was a legal SAN) and **`PlayStrength`** (the composite score). It's easy to assume "high first-attempt legality → high PlayStrength", but the composite has three factors and first-attempt-legality is only one of them. The other two can drag the score substantially even when legality is near-perfect:

- **`move_quality(cp_loss)`**: a legal move with cp_loss = 86 has quality = exp(−86/150) = 0.564, not 1.0. Reaching engine-level play (cp_loss ~5) requires move_quality ~0.97. Most LLMs play moves with cp_loss in the 50–200 range — competent club-player territory, scoring 0.27–0.72 per move on quality alone.
- **`game_phase_weight(ply)`**: ply-30+ moves are weighted 3×, ply-1–9 are weighted 1×. A model that forfeits early or stops at the max-ply cap before reaching late game loses access to the highest-weighted plies in the numerator while they remain in the denominator. The structural ceiling for a model that perfectly plays only to ply 20 (out of max 40) is ~0.30; perfectly to ply 30 is ~0.59; perfectly to ply 40 is 1.0.
- **A single full forfeit** (game ends at ply 1 because the model can't produce a legal move even after all retries) drops that game's contribution to 0 regardless of how the other games went. At N=20 games per cell, one forfeit subtracts ~0.05 from the mean.

A worked decomposition: a model with **100% first-attempt-legal, 0 avg retries, 0 forfeits, mean cp_loss of 85, and mean game length 26 plies** still scores PlayStrength = 0.30. Why:

| Factor | Contribution |
|---|---|
| Mean cp_loss 85 → quality = exp(−85/150) ≈ 0.57 | Caps the score around 0.57 even at perfect legality |
| Zero forfeits | No penalty here |
| Mean game length 26 plies (vs max 40) | Misses high-weight late plies; structural ceiling around ~0.45 |
| 0 avg retries | No penalty here |

"Perfect legality × mid-range quality × partial late-game coverage" multiplies to 0.30. Read the diagnostic columns together — first-attempt-legal alone isn't a stand-in for PlayStrength. The same legality and game-length profile at engine-quality move strength (cp_loss ~5) would score ~0.93.

### High PlayStrength does not mean strong vs an engine

The benchmark plays models against amateur-tier Stockfish (skill 3 / skill 5). A 0.50 PlayStrength against skill 3 says "competent play across a full game against an amateur opponent." It does NOT imply the model would win against skill 15 (engine-level), or against a top human, or against another LLM. The benchmark deliberately uses an amateur opponent so games last long enough to expose state-tracking failure — not as a yardstick for chess strength itself.

### PlayStrength and MoveQuality measure different things, on different opponents

The two metrics use different Stockfish skill levels and different retry budgets. **PlayStrength and MoveQuality scores from the same cell are not directly comparable** as a single "this model's chess level" reading. They answer different questions. See [What the benchmark measures](#what-the-benchmark-measures) above.

---

## Caveats and expectations

**Caveats:**

- **Default `--games` is small** (5 for PlayStrength, 3 for MoveQuality). Enough for qualitative pattern signal but not for tight effect sizes. The published matrix used N ≥ 20 per cell (many cells N = 100+) for stable means. A 0.02 difference between two models at N=5 could be sampling noise; at N=20+ it's more durable.
- **Stockfish version matters.** Move-quality scores depend on the engine's evaluation function. Lock the binary version when comparing across runs. This work used Stockfish 18.
- **Provider tool-calling differences.** Each provider's structured-output format works slightly differently. The adapters normalize these; a regression on a specific provider can show up as a benchmark regression. Always grep the run JSONL for `"did not call submit_move"` before drawing conclusions.
- **Quota-corrupted games are dropped.** Per-game error filter removes games where any move hit a 429 / quota error mid-game (so the run isn't contaminated by half-played games where a provider's billing failed silently). Drop counts are reported alongside N per cell in the aggregated output.
- **This is not a chess-skill benchmark.** A model that's bad at chess but rule-consistent would score well. The eval measures state-tracking and rule-following; ELO is incidental.

**Reasonable expectations:**

- The first-attempt-legal rate and forfeit-rate columns are the most diagnostic single numbers — first-legal measures the fraction of plies where the model's very first proposal (zero retries) was legal; forfeit-rate measures the fraction of games that ended because retries couldn't recover a legal move. Both read independently of scoring choices.
- The choice of `move_quality` decay constant (τ=150) is not load-bearing: re-scoring at τ ∈ {100, 200, 300} produces identical model rankings with absolute scores shifting ~20%. See [RESULTS § τ sensitivity](RESULTS.md#methodology-robustness--τ-sensitivity).
- The scoring is designed to be hill-climbable. The top of the [0, 1] scale (~0.95+, where engine-level play would land) is unoccupied by any current cell. The gap between the matrix top and 1.0 is the cognitive headroom this benchmark exists to quantify.

---

## Open questions

What would tighten the findings:

1. **Larger N per cell.** 5/3 games is enough for qualitative signal; 10–30 per cell is needed for tight effect sizes.
2. **Skill sweep.** PlayStrength and MoveQuality at multiple Stockfish skills — how does the cascade interact with opponent strength?
3. **Mid-game starting positions.** Does the cascade need game-length state accumulation, or does it appear immediately from a complex mid-game FEN? Separates "drift across turns" from "complex positions are harder regardless."
4. **Reasoning-trace inspection across retries.** Save every retry's full response (currently only the final). Measures whether the model genuinely updates between retries vs pattern-matching a different SAN.
5. **Controlled reasoning-effort experiment.** Same positions × varying effort per call × paired cp_loss. Resolves the confound in the current reasoning-effort-vs-quality observation.
6. **Other 2D-state substrates.** Build analogous benchmarks for grid-navigation, UI layout reasoning, or tile-based puzzles. Confirms the spatial-reasoning failure profile is universal across 2D-grounded tasks, not chess-specific.

---

## Chess glossary

Quick reference for chess-specific terms used throughout the methodology and results.

| Term | Meaning |
|---|---|
| **FEN** | Forsyth–Edwards Notation. A compact one-line string that fully encodes a chess position: piece placement, side to move, castling rights, en-passant target, halfmove clock, fullmove number. The benchmark passes the FEN to the model each turn — that's the entire context the model sees about board state. |
| **SAN** | Standard Algebraic Notation. The standard way to write chess moves: `e4`, `Nf3`, `Bxc5`, `O-O`, `e8=Q+`. The model proposes moves in SAN. |
| **PGN** | Portable Game Notation. The standard text format for recording a whole chess game (move list + tags). The benchmark stores raw game records as JSONL, not PGN. |
| **Ply** | A single half-move — one player's turn. One full move = 2 plies (white's move + black's response). The benchmark counts in plies throughout (`max_plies = 40` means 20 moves by each side). |
| **ELO** | The standard chess rating system. ~1100 = beginner, 1500 = amateur, 1700 = intermediate amateur, 2200 = club master, 2500+ = strong / IM / GM, 2850+ = top engine. The benchmark uses ELO to calibrate how strong the Stockfish opponent is. |
| **Centipawn (cp)** | 1/100 of a pawn. Standard unit for chess engine evaluations: +1.00 ≈ "white is up one pawn." Stockfish reports evaluations in centipawns. |
| **cp_loss** | Centipawn loss — the difference between Stockfish's evaluation of its top move and its evaluation of the move the LLM actually played. A move with cp_loss = 0 is the engine's best; cp_loss = 150 means the model gave up ~1.5 pawns of advantage. Used as the per-move quality input. |
| **ACPL** | Average Centipawn Loss. Mean cp_loss across the moves of a game (or phase, or cell). The classic chess-strength diagnostic: lower = stronger play. We report ACPL by phase (opening / middlegame / endgame) to expose where a model degrades. |
| **Stockfish skill level** | An integer 0–20 controlling Stockfish's `Skill Level` UCI option, which weakens the engine. Higher = stronger. The benchmark uses skill 3 (~1500 ELO) for PlayStrength and skill 5 (~1700 ELO) for MoveQuality — both amateur-tier on purpose. |
| **Opening / middlegame / endgame** | The three traditional phases of a chess game. The benchmark uses ply-based boundaries: opening ≈ plies 1–10, early middlegame ≈ 10–20, middlegame ≈ 20–30, endgame ≈ 30+. These are calibration thresholds, not hard chess theory — actual game phase varies. |
| **Forfeit** | In chess generally: losing by failing to make a legal move or running out of time. In this benchmark specifically: the game ends with result `forfeit_illegal` because the model failed to produce a legal SAN within the retry budget for a single ply. See [How games end](#how-games-end). |
| **Memorization cliff** | Our term for the transition from "recognize position from training data" to "infer next move from rules alone." The hypothesis is that LLMs play strong opening moves via memorization and weaker mid/endgame moves via inference; the data shows this is real for *some* models but not universal. See [The design hypothesis](#the-design-hypothesis). |
| **Cumulative-coherence failure** | Our term for performance degradation that accumulates across a long game — even on positions the model could handle in isolation — because it has to sustain a coherent mental picture of the board across many stateless API calls. Distinct from the per-position memorization cliff. |

---

For the matrix, findings, deep dives, and the Anthropic-score explanation, see **[RESULTS.md](RESULTS.md)**. For provider quirks and how to run the eval, see **[HOWTO.md](HOWTO.md)**.
