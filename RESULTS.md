# LLM Chess Eval — Results

What we found running the benchmark across eight model-tier cells from four providers (Anthropic, OpenAI, Google, DeepSeek). For the formulas, scoring rationale, methodology constraints, and reproduction recipe, see **[METHODOLOGY.md](METHODOLOGY.md)**.

---

## Scope

LLMs are pattern-matching transformers — closer to how humans think than to how chess engines search. Engines like Stockfish are exhaustive search systems; they don't need memory because they compute optimal moves from current state. Humans (and LLMs) pattern-match. For pattern-matching cognition, memory isn't a crutch — it's how the system works.

**This benchmark deliberately strips memory and tools to measure a specific cognitive primitive in isolation: pattern-matching to good moves from the current position alone, across many independent calls.** Real production usage compensates with context (history, prior reasoning, retrieval) — that's appropriate because LLMs are pattern-matchers, not deep searchers. The benchmark removes that scaffolding to test the primitive cleanly. If models score higher with memory or tools, that's a measure of how much cognitive work is being externalized to context — not a refutation of the benchmark.

The matrix below ranks models on this primitive. A model's score here doesn't predict its score on other benchmarks — chess-style spatial reasoning is one capability among many.

---

## The core finding: persistent wrong belief

When LLMs produce illegal chess moves, they don't make random errors. They form **coherent-but-wrong mental models of the position and commit to them across multiple stateless API calls.** Each call is independent from the model's side, but the same FEN deterministically reproduces the same wrong pattern-match.

**Example 1 — Claude Opus proposes the same illegal king-capture five times in one game:**

```
ply 25: rationale "grabs the pawn on g4"           — illegal: would put king in check
ply 32: rationale "captures the g4 pawn for free"
ply 33: rationale "wins the g4 pawn"
ply 35: rationale "to grab material"
ply 36: rationale "capture the g4 pawn to remove threat"
```

White's king cannot capture the pawn on g4 — doing so exposes it to check from a distant attacker. Opus regenerates this same wrong move five times across 11 plies. Stockfish substituted around it on every preceding attempt, but the model has no memory of failure across stateless calls. The same FEN keeps producing the same wrong pattern-match.

**Example 2 — Sonnet proposes a pawn advance for a pawn that does not exist:**

```
"Advancing the passed pawn to d7 attacks the black rook..."
"Advancing d6 pawn to d7 puts tremendous pressure..."
"Advancing the d6 pawn to d7 puts tremendous pressure..."
```

There is no white pawn on d6 in this position. The model believes in a pawn configuration that doesn't exist on the board, and the wrong belief is encoded in its response to that specific FEN.

**Why this matters.** A standard "LLM hallucination" is a one-shot plausible-sounding wrong fact. This is **convergence on the same specific wrong belief across many independent invocations** — implying the belief is the model's deterministic response to a specific input, not a random error. You cannot fix it by sampling more times or with a different temperature; the wrong mental model regenerates from the same FEN.

Retry feedback in the harness gives the model the list of recent failed attempts and asks it to try again. This fixes "wrong move from a roughly-correct mental model" (the model picks a different piece type), but it does NOT fix "wrong mental model of the board" — the model changes the move it commits to but cannot, via text feedback alone, change what it thinks it's looking at. Forfeits in our games cluster precisely on plies where all retry attempts share the same structural mistake (e.g., none of 4 retries addresses an existing check, because none of them perceive the check).

This is the deepest claim the benchmark makes, and it's the cognitive failure mode the metrics below quantify.

---

## What this looks like in practice

[Levy Rozman's short "ChatGPT vs Meta AI: This Isn't Chess Anymore"](https://www.youtube.com/shorts/YlMWZNx93G4) is the failure as comedy. Two LLMs trying to play a game produce nonsense after the opening: pieces appear from nowhere, captures are claimed on empty squares, the same illegal move keeps getting proposed, and both models confidently narrate plans for pieces no longer on the board. The benchmark scores numerically what the video shows visually.

---

## What's behind the in-game cliff — not memorization alone

The persistent-wrong-belief pattern doesn't show up evenly across game phases. Move quality (ACPL) roughly doubles or triples from opening to mid-game for every model that reaches it. This used to be called the **memorization cliff**: the position-space where models shift from pattern-recall to spatial reasoning. The story was clean and intuitive — but the data forced a revision.

### The cliff is NOT just about position novelty

To test whether the cliff is about position novelty (models perform well on positions they've seen, badly on unseen ones), we tested 5 models on 4 progressively-more-novel banks of 20 positions each:

- **T0 hand-curated**: 20 positions including named openings (Italian, Ruy Lopez, etc.) and textbook endgames. Saturated in training data.
- **T1 real-play extracted**: FENs extracted from real model-vs-Stockfish games where the model later made an illegal move. Out of training distribution, but selected for difficulty.
- **T2 random-opening + Stockfish continuation**: random 5-10 ply opening + 15 plies Stockfish skill-5 self-play. Mid-game positions with no theory shortcut, but engine-realistic structure.
- **T3 pure random-vs-random self-play**: 30 plies of random play. Maximally novel — no theory at any stage.

Per-position legality across all 5 models × 4 banks:

| Model | T0 hand | T1 real-play | T2 random-open | T3 random-play |
|---|---|---|---|---|
| `gpt-5` | **1.000** | **1.000** | **1.000** | **1.000** |
| `gemini-3.1-flash-lite` | 0.950 | 0.900 | 0.900 | 0.900 |
| `claude-sonnet-4-6` | 0.750 | 0.500 | 0.750 | 0.700 |
| `claude-opus-4-7` | 0.900 | 0.300 | 0.775 | 0.800 |
| `deepseek-chat` | 0.500 | 0.200 | 0.200 | 0.250 |

(T1 numbers are selection-biased — those positions came from games where the model had failed. T2 and T3 are unbiased random novel positions and give the cleanest "novelty" signal.)

**The picture is much more interesting than a universal cliff:**

- **GPT-5 has no cliff at all.** Perfect legality on every bank, including positions reached by random self-play. Whatever's dragging GPT-5's matrix Reliability down (0.41) is *not* per-position legality.
- **Gemini 3.1 Flash Lite has no cliff.** Stays at ~90% across all banks. The "best model in the matrix" is best because it reliably plays legal chess on positions it's never seen, not because it has the biggest opening book.
- **Sonnet has no cliff on unbiased banks.** Its T0 (0.75) and T2 (0.75) are identical. The T1 (0.5) drop reflects selection bias, not a real cliff.
- **Opus has a modest cliff** (~14% drop from hand-curated to T2). Some genuine memorization advantage on hand-curated positions, but mostly real chess ability underneath.
- **DeepSeek-chat has a sharp cliff** (~55% drop). Its hand-curated score IS largely from memorization; on truly novel positions it falls off.

The "memorization cliff" is **model-specific, not universal.** And it's not what drives the matrix Reliability cliff for the top-of-matrix models — those models pass single-position legality with flying colors on any bank we throw at them.

### So what IS driving the in-game cliff?

The matrix Reliability metric reflects cumulative game performance — legal moves played across 30-40 plies, with phase-weighted move quality. Several things compound in actual gameplay that single-position tests don't capture:

1. **Move quality (ACPL) degrades by phase even for models with no legality cliff.** Flash Lite plays at 62 cp loss in opening, 128 cp in endgame. Engine-level is ~5-20 cp. The cliff is in *move strength*, not legality — models play legal but mediocre moves on mid/endgame positions.
2. **Cumulative trajectory matters.** Real gameplay between an LLM and a stronger Stockfish opponent drifts into positions where the LLM is under pressure. T2 (Stockfish-vs-Stockfish from random opening) and T3 (random play) don't reproduce this — they produce balanced or neutrally-random positions. Real games create *asymmetric pressure* positions that test something different.
3. **Persistent wrong belief compounds within games.** A model that gets a wrong picture of the position on ply 24 carries it through retries on ply 24 and into ply 25's similar position. The Kxg4 ×5 case study is the cleanest example. Single-position evaluation doesn't capture this.

ACPL by phase across the matrix (lower is better — engine-level play is ACPL ~10):

| Model | ACPL opening | ACPL middlegame | ACPL endgame |
|---|---|---|---|
| `gemini-3.1-flash-lite` | 62 | 53 | 128 |
| `gemini-2.5-pro` | 31 | 99 | 112 |
| `gpt-5` | (low) | (mid) | (mid) |
| `deepseek-reasoner` | 131 | 140 | 70 |
| `claude-opus-4-7` | 94 | 240 | 0 (not reached) |
| `claude-haiku-4-5-20251001` | 70 | 0 (not reached) | 0 (not reached) |
| `deepseek-chat` | 80 | 0 (not reached) | 0 (not reached) |

The ACPL gradient is universal — every model that reaches mid-game shows it. Combined with the bank-comparison results above, this means: **the in-game cliff is about move *quality* and cumulative gameplay dynamics, not about whether the position is in training data.**

---

## How we measured this

Two scores per model, both bounded `[0, 1]`. Full formulas live in [METHODOLOGY.md](METHODOLOGY.md).

- **ChessReliability** — rule-following over full games vs Stockfish skill 3 (~1500 ELO, intermediate amateur), with up to 10 retries per illegal move at steep per-retry cost (`0.25^retries`). Move quality decays exponentially in centipawn loss (`exp(-cp_loss / 150)`). Each ply is weighted geometrically by game phase (1 / 2 / 4 / 8 at ply boundaries 10 / 20 / 30) so late-game plies, where training distribution ends, dominate the score.
- **PlayQuality** — move strength once a legal move is found. Same per-move quality and phase weighting as Reliability, but without the retry cost. Played at Stockfish skill 5 (~1700 ELO, intermediate amateur) with max 3 retries.

Two diagnostic columns alongside the composite scores make the matrix readable: **first-attempt-legal rate** (fraction of plies where the very first model proposal was legal) and **average retries per move**. Together they distinguish "scored high because moves were legal first try" from "scored high because the retry mechanism rescued the model."

The 1.0 scoring anchor is Stockfish self-play (engine-quality moves with no retries across a full game) — *not* an opponent the benchmark plays against. Both metrics use amateur-tier opponents deliberately; the benchmark is not a model-vs-engine comparison.

---

## The matrix

Nine cells: frontier and budget tier across four providers, plus Sonnet as Anthropic mid-tier. Sorted by ChessReliability.

| Provider | Tier | Model | Reliability | PlayQuality | first-attempt legal | avg retries/move |
|---|---|---|---|---|---|---|
| Google | budget | `gemini-3.1-flash-lite` | **0.639** | **0.320** | 87.8% | 0.16 |
| Google | frontier | `gemini-2.5-pro` | 0.527 | 0.274 | 93.8% | 0.07 |
| OpenAI | frontier | `gpt-5` | 0.410 [*ᶜ*](#footnote-c) | 0.200 | **97.8%** | 0.15 [*ᵈ*](#footnote-d) |
| DeepSeek | frontier | `deepseek-reasoner` | 0.373 | 0.086 [*ᵃ*](#footnote-a) | 78.3% | 0.27 |
| OpenAI | budget | `gpt-5-mini` | 0.301 | 0.079 | 90.2% | 0.11 |
| Anthropic | frontier | `claude-opus-4-7` | 0.164 | 0.068 [*ᵃ*](#footnote-a) | 63.1% | 0.79 |
| Anthropic | mid | `claude-sonnet-4-6` | 0.122 | 0.020 | 66.1% | 0.77 |
| DeepSeek | budget | `deepseek-chat` | 0.040 [*ᵇ*](#footnote-b) | 0.017 [*ᵇ*](#footnote-b) | 38.3% | 2.65 |
| Anthropic | budget | `claude-haiku-4-5-20251001` | 0.032 [*ᵇ*](#footnote-b) | 0.014 [*ᵇ*](#footnote-b) | 53.7% | 1.74 |

### Why some scores are extremely low

The geometric phase weight (`1 / 2 / 4 / 8` at ply boundaries 10 / 20 / 30) means roughly half of the achievable score lives in plies 20+. A model that forfeits before reaching mid-game loses access to that half in both the numerator AND has it weighing against them in the denominator. Combined with the `0.25^retries` cost and the exponential quality decay, scores below 0.10 are the natural floor for models that can't survive long.

<a id="footnote-a"></a>
***ᵃ*** **`0.068`–`0.086` (Opus PlayQuality, DeepSeek-reasoner PlayQuality)** — model produces legal moves on most first attempts (63–78%) and plays moves of reasonable quality (cp_loss 80–240 on the plies it reaches), but the games rarely survive into endgame at Stockfish skill 5. With only 3 retries permitted in PlayQuality mode (vs Reliability's 10), even occasional length-failures and run-of-bad-positions cause early forfeits. Result: the high-weight late plies are mostly missing from the numerator while still counting toward the denominator.

<a id="footnote-b"></a>
***ᵇ*** **`0.014`–`0.040` (Haiku, DeepSeek-chat — both metrics)** — first-attempt-legal rate of 38–54% means roughly half of all moves are illegal on the first try. Each of those moves pays the `0.25^retries` cost (1 retry → 25% credit, 2 retries → 6%). Combined with games averaging only 10–20 plies before forfeiting on illegal moves they can't recover from, the score is dominated by missing late-game plies and retry-cost-eroded mid-game plies. This isn't a metric pathology — it's the metric correctly reflecting "model cannot reliably play legal chess past the opening, even with the retry safety net."

<a id="footnote-c"></a>
***ᶜ*** **`0.410` (GPT-5 Reliability with 97.8% first-attempt-legal)** — the asterisk here is the *opposite* situation: the model produces a legal move on first attempt ~98% of the time, but the composite is still moderate because Reliability multiplies three factors and first-attempt-legality is only one of them. GPT-5's mean cp_loss across legal moves is 86 → move_quality = exp(−86/150) = 0.564 (competent club play, not engine-level). One of the 5 games forfeited at ply 1, contributing 0 to the mean and subtracting ~0.20 from the composite. The remaining 4 games averaged 27 plies (vs max 40), missing the highest-weighted late-game plies. **High first-attempt-legal does not imply high Reliability** — read the columns together. [METHODOLOGY § Reliability is NOT first-attempt-legal](METHODOLOGY.md#reliability-is-not-first-attempt-legal--they-measure-different-things) walks through the decomposition.

<a id="footnote-d"></a>
***ᵈ*** **`0.15` (GPT-5 avg retries/move) with 97.8% first-attempt-legal** — this number can read as a contradiction with Gemini 2.5 Pro's `0.07 / 93.8%`: GPT-5 fails *less often* yet shows *more retries on average*. The retry distributions explain it. Gemini's 11 failed plies recovered in 1–2 retries each (total 12 retries across 11 failures). GPT-5's 3 failed plies needed 1, 10, and 10 retries respectively (total 21 retries across 3 failures) — when GPT-5 fails it usually burns through the entire stepdown ladder (`default → medium → low → minimal`) before producing a legal SAN. Two qualitatively different failure profiles: Gemini's failures are quick to recover; GPT-5's are reasoning-budget exhaustion that exhaust the full retry budget. See [Reading the matrix § Avg retries/move can be misleading on its own](#avg-retriesmove-can-be-misleading-on-its-own--read-it-with-first-attempt-legal) for the detailed breakdown.

For specifics on the Anthropic cells in particular, see [the Anthropic note below](#a-note-on-the-anthropic-scores).

---

## Reading the matrix

The first-attempt legal column and avg retries column are essential. Two models with identical Reliability scores can mean very different things — one might be picking legal moves with zero retries and being graded mainly on move quality; another might be needing two retries per move and being penalized for the cost of finding the legal move. The diagnostic columns separate these readings.

"First-attempt legal" means **zero retries used** — the model proposed a legal SAN on its very first attempt for that move, before any harness feedback. A model with 88% first-attempt legal is producing a legal move with no retry feedback on 88% of plies.

### Avg retries/move can be misleading on its own — read it with first-attempt-legal

The `avg retries/move` column is the total retries used across all plies divided by total plies. It compresses two qualitatively different failure profiles into one number. A counterintuitive example from the matrix:

```
Gemini 2.5 Pro:  first-attempt legal 93.8%   avg retries 0.07
GPT-5:           first-attempt legal 97.8%   avg retries 0.15
```

GPT-5 fails *less often* (2.2% vs 6.2%) yet has *more retries on average* (0.15 vs 0.07). The retry distributions explain it:

```
Gemini 2.5 Pro:  11 plies needed retries → 10 took 1 retry, 1 took 2 retries
                 → failures recover in 1-2 retries

GPT-5:            3 plies needed retries → 1 took 1 retry, 2 took 10 retries (max)
                 → failures burn the entire stepdown ladder
```

Two different patterns: **Gemini fails more often but recovers fast** (1 retry usually fixes it — a typical "wrong piece-type choice" the retry feedback corrects). **GPT-5 fails rarely but catastrophically** — when it fails it usually exhausts the full retry budget because the failure is reasoning-budget exhaustion (length errors that force the harness all the way down the `default → medium → low → minimal` effort ladder before a legal move emerges).

The avg-retries column gives you the average cost; the first-attempt-legal column gives you the failure rate; the gap between them gives you the failure *depth*. Read all three together.

The matrix sorts into three behavioral bands:

**Band 1 — "Plays legal chess on first attempt":** Gemini cells, GPT-5, DeepSeek-reasoner. First-attempt legal rate 78-98%, avg retries 0.07-0.27 per move. The Reliability score is mostly about *move quality* on the legal moves played. The composite reflects what these models actually do at the board.

**Band 2 — "Needs the retry safety net":** Claude Opus 4.7. First-try legal 63%, avg 0.79 retries per move. Roughly one move in three is illegal on first try; the harness feeds back errors before the model corrects. Even after recovery the `0.25^retries` cost ensures retried moves contribute little.

**Band 3 — "Struggles to even propose legal moves":** Claude Haiku, DeepSeek-chat. First-try legal 38-54%, avg 1.7-2.7 retries per move. About half of all moves are illegal on first try, and the games rarely survive long enough to reach the high-weight late plies.

---

## Patterns from the matrix

Three observations the numbers support — these are *consequences* of the core finding, derived from the matrix, not the finding itself.

**1. Reasoning-tier supremacy doesn't hold for spatial state-tracking.** Across the four frontier reasoning models in the matrix, Reliability spans 0.16 to 0.53 — a 3× spread. The matrix-leader is a budget non-reasoning model. Reasoning-tier optimization helps when the reasoning fits in the budget AND when the model can apply that reasoning to spatial state — neither is guaranteed on this benchmark.

**2. Budget beats frontier in 1 of 4 providers.** Only OpenAI shows the inversion (`gpt-5-mini` re-running — preliminary data suggests it edges `gpt-5` on this dimension despite being the smaller model). DeepSeek and Anthropic show the expected direction (frontier > budget). Google's frontier (Gemini 2.5 Pro) leads its budget (Flash Lite) on PlayQuality but trails on Reliability — a split that reads as "2.5 Pro plays better moves but Flash Lite plays more legal moves."

**3. Failures concentrate on out-of-distribution positions.** Every standard opening position in our 20-position bank shows 0% failure rate across every model tested. Synthetic mid-game and endgame positions show 33-67% failure rates on the hardest examples. Failure isn't randomly distributed — it concentrates exactly where memorized chess theory runs out, supporting the memorization-cliff thesis baked into the score's phase weighting.

---

## A note on the Anthropic scores

Anthropic's scores are the lowest among the four providers in the matrix. This deserves explanation because the picture is easy to misread.

**Anthropic's numbers are not artifacts of methodology.** Anthropic was the only provider whose API responses were unaffected by the response-token-budget configuration issue that affected the other providers (see [METHODOLOGY.md](METHODOLOGY.md#methodology-constraints-encountered)). Anthropic's `max_tokens` parameter counts *output tokens only*, with extended thinking on a separate budget. The numbers shown here reflect genuine model behavior at full reasoning budgets.

**What the matrix exposes is that Anthropic's reasoning models struggle specifically with sustained 2D spatial state-tracking** — exactly the cognitive failure mode this benchmark was designed to isolate. Anthropic excels on benchmarks that reward strong single-shot reasoning over textual or symbolic state (MMLU, GPQA, coding). It underperforms here because the benchmark is biased toward the dimension where Anthropic is comparatively weakest. The same models that fail to reach the endgame in chess routinely solve graduate-level math and write production code — chess-style spatial reasoning is a specific weakness, not a general one.

Quantitatively, three things compound for Anthropic:

1. Anthropic's first-attempt-legal rate is lower than the matrix top (63% Opus vs 98% GPT-5). Each illegal-first-attempt move pays the `0.25^retries` cost.
2. Anthropic's middlegame ACPL is the highest in the matrix at ~240 cp for Opus. Under exponential quality decay this scores ~0.20 per move vs Flash Lite's ~0.65 per move.
3. Anthropic forfeits earlier than the matrix top — Opus reaches plies 1-30 in only some games, Haiku averages 11 plies per game. With high-weight plies (20+, 30+) contributing disproportionately to both numerator and denominator, models that forfeit early lose access to the score's largest contributions.

The matrix is not a ranking of "best AI." It's a ranking on one cognitive dimension. Anthropic is at the bottom of *this* dimension; other dimensions order the providers very differently.

---

## Further evidence

Additional analyses that support the core finding above.

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

When a move is illegal, the next attempt is told "your move X was illegal, try again" and given the list of recent failed attempts (capped to the last 3). Across one game's 22 retries, 17 of 22 iterations went to a different piece type (model genuinely updated). The remaining 5 went to the same piece type with a different target, or repeated the same move.

Forfeits cluster on plies where **all retry attempts share the same structural mistake.** A representative case: at one ply Black was in check from a distant bishop. Opus proposed 4 moves over its retry budget — none addressed the check, because Opus didn't perceive the check at all. Every retry proposed a "good move" that ignored the check.

**Retry feedback fixes "wrong move from a roughly-correct mental model" but cannot fix "wrong mental model of what's on the board."** The model can change its move but cannot, via text feedback alone, change what it thinks it's looking at.

### Reasoning effort vs move quality — suggestive evidence

Reasoning models from OpenAI and Gemini support an effort dial. The benchmark logs the effort level on each call; on length-failure errors, the harness steps the effort down for the next retry (`default → medium → low → minimal`). Aggregating cp_loss bucketed by the effort level on the final successful attempt:

| Model | default cp / n | medium cp / n | low cp / n | minimal cp / n |
|---|---|---|---|---|
| `gemini-2.5-pro` | 62 / 210 | 47 / 17 | 52 / 7 | (n/a) |
| `gpt-5` | 1 / 47 | (n/a) | 64 / 25 | 67 / 154 |
| `gpt-5-mini` | 33 / 108 | 51 / 3 | 86 / 41 | 93 / 9 |

The pattern: **quality at lower reasoning effort is comparable to quality at default effort.** For GPT-5 specifically, only 47 of 255 (~18%) of successful moves were emitted at full reasoning effort; the remaining 80% required stepdown to low/minimal to fit the response budget. The cp_loss on minimal-effort moves (67) is in the same range as Flash Lite's overall cp_loss — i.e., near the matrix median — and not catastrophically worse than the default-effort moves the same model produced.

**This is suggestive but has a confound:** the default and stepped-down buckets sample different distributions of positions. Default succeeded on easier positions where reasoning was short; minimal was the fallback for harder positions where reasoning was long. The data does not yet control for position difficulty across effort levels. A controlled experiment (same positions, varying effort per call, paired cp_loss) is planned as next work.

### Skill sweep — the metric isn't a single-skill artifact

The published matrix uses Stockfish skill 3 for Reliability. To check whether the metric is sensitive to opponent strength (and that the published score isn't a coincidence of one particular opponent), we ran Flash Lite at Stockfish skills 1, 5, 10, and 15:

```
Stockfish skill 1   (~1100 ELO, beginner):       CR 0.352
Stockfish skill 5   (~1700 ELO, amateur):        CR 0.645   ← peak
Stockfish skill 10  (~2200 ELO, club master):    CR 0.590
Stockfish skill 15  (~2500 ELO, strong engine):  CR 0.210
```

The curve is **not monotonic** — it's U-shaped. Flash Lite scores best against intermediate-amateur opponents (skill 5) and degrades in *both* directions:

- **Skill 15 collapse** (0.21) is expected: engine-grade pressure forces precise play the model can't deliver.
- **Skill 1 drop** (0.35) is the surprise: a beginner-random opponent makes such structurally weird moves that the games go off-theory faster than at amateur skill. **Random play breaks theoretical structure and creates novel mid-game positions on every move.**

Both ends push the model out of distribution. The model only thrives in the structured-amateur middle (skill 3-5), which is where the published Reliability runs. The score isn't a coincidence of opponent strength — it's roughly the best score the model can achieve under this metric.

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

## The bottom line

The benchmark measures one cognitive primitive: whether LLMs can pattern-match to good chess moves from current position alone, across stateless calls. The deepest claim is qualitative: **models form coherent-but-wrong mental models of the board and commit to them deterministically across stateless calls.** This is qualitatively different from typical hallucinations and is unlikely to be fixed by bigger context windows or more training data alone — it suggests something missing in how today's LLMs represent and update structured state when each call is independent.

Position novelty is NOT the explanation for in-game degradation in the top-of-matrix cells. Per-position legality tests across four progressively-more-novel banks show that Flash Lite, GPT-5, and Sonnet handle novel positions essentially as well as memorized ones (Opus has a modest cliff; only DeepSeek-chat has a sharp one). What drags matrix Reliability down for the top cells is move quality and cumulative gameplay dynamics — the cliff is in *cumulative coherence across turns*, not in *can the model handle one novel position*.

The quantitative matrix supports the qualitative finding. The strongest model scores 0.639 on a [0, 1] scale where engine-quality self-play is the 1.0 reference. That's a budget non-reasoning model from Google. Frontier reasoning models from Anthropic, OpenAI, and DeepSeek score below it. The ranking does not track "reasoning tier" or "frontier vs budget" — it tracks something more specific.

The scoring is designed to be **hill-climbable**. The current matrix top (0.64) leaves real room above 0.9, where engine-quality play would land. That headroom is not noise — it's the gap between today's frontier and a model that can keep its mental picture of an 8×8 board accurate across 40 moves, without external scaffolding.

For methodology, formulas, scoring rationale, and reproduction recipe, see **[METHODOLOGY.md](METHODOLOGY.md)**.
