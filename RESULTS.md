# LLM Chess Eval — Results

What we found running the benchmark across nine model-tier cells from four providers (Anthropic, OpenAI, Google, DeepSeek). For the formulas, scoring rationale, methodology constraints, and reproduction recipe, see **[METHODOLOGY.md](METHODOLOGY.md)**.

---

## Scope

LLMs are pattern-matching transformers — closer to how humans think than to how chess engines search. Engines like Stockfish are exhaustive search systems; they don't need memory because they compute optimal moves from current state. Humans (and LLMs) pattern-match. For pattern-matching cognition, memory isn't a crutch — it's how the system works.

**This benchmark deliberately strips memory and tools to measure a specific cognitive primitive in isolation: pattern-matching to good moves from the current position alone, across many independent calls.** Real production usage compensates with context (history, prior reasoning, retrieval) — that's appropriate because LLMs are pattern-matchers, not deep searchers. The benchmark removes that scaffolding to test the primitive cleanly. If models score higher with memory or tools, that's a measure of how much cognitive work is being externalized to context — not a refutation of the benchmark.

The matrix below ranks models on this primitive. A model's score here doesn't predict its score on other benchmarks — chess-style spatial reasoning is one capability among many.

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

Two scores per model, both bounded `[0, 1]`. **ChessReliability** measures rule-following over full games vs Stockfish skill 3 (~1500 ELO amateur) with up to 10 retries per illegal move at steep per-retry cost. **PlayQuality** measures move strength once a legal move is found, played against Stockfish skill 5 (~1700 ELO amateur). The 1.0 reference is Stockfish self-play. Diagnostic columns: **first-attempt legal** = fraction of plies where the very first model proposal was legal (zero retries); **avg retries/move** = total retries across all plies divided by total plies. Full formulas in [METHODOLOGY.md](METHODOLOGY.md).

### Reading the matrix

The first-attempt legal column and avg retries column are essential. Two models with identical Reliability scores can mean very different things — one might be picking legal moves with zero retries and being graded mainly on move quality; another might be needing retries each move and being penalized for the cost. The matrix sorts into three behavioral bands:

**Band 1 — "Plays legal chess on first attempt":** Gemini cells, GPT-5, GPT-5-mini, DeepSeek-reasoner. First-attempt legal rate 78-98%, avg retries 0.07-0.27 per move. The Reliability score here is mostly about *move quality* on the legal moves played.

**Band 2 — "Needs the retry safety net":** Claude Opus, Claude Sonnet. First-attempt legal 63-66%, avg 0.77-0.79 retries per move. Roughly one move in three is illegal on first try; the harness feeds back errors before the model corrects. Even after recovery the `0.25^retries` cost ensures retried moves contribute little.

**Band 3 — "Struggles to even propose legal moves":** Claude Haiku, DeepSeek-chat. First-attempt legal 38-54%, avg 1.7-2.7 retries per move. About half of all moves are illegal on first try, and the games rarely survive long enough to reach the high-weight late plies.

The full footnote explanations of [*ᵃ*](#footnote-a) [*ᵇ*](#footnote-b) [*ᶜ*](#footnote-c) [*ᵈ*](#footnote-d) are in [Footnotes on the matrix](#footnotes-on-the-matrix) below.

---

## A note on the Anthropic scores

Anthropic's scores are the lowest among the four providers in the matrix. This deserves explanation because the picture is easy to misread.

**Anthropic's numbers are not artifacts of methodology.** Anthropic was the only provider whose API responses were unaffected by the response-token-budget configuration issue that affected the other providers (see [METHODOLOGY.md](METHODOLOGY.md#methodology-constraints-encountered)). Anthropic's `max_tokens` parameter counts *output tokens only*, with extended thinking on a separate budget. The numbers shown here reflect genuine model behavior at full reasoning budgets.

**What the matrix exposes is that Anthropic's reasoning models struggle specifically with sustained 2D spatial state-tracking** — exactly the cognitive failure mode this benchmark was designed to isolate. Anthropic excels on benchmarks that reward strong single-shot reasoning over textual or symbolic state (MMLU, GPQA, coding). It underperforms here because the benchmark is biased toward the dimension where Anthropic is comparatively weakest. The same models that fail to reach the endgame in chess routinely solve graduate-level math and write production code — chess-style spatial reasoning is a specific weakness, not a general one.

Notably, **Sonnet 4.6 ranks below Haiku 4.5** within the Anthropic family on this dimension — the weakness doesn't track the standard "Haiku < Sonnet < Opus" capability ordering, supporting the "one specific dimension, not general capability" framing.

The matrix is not a ranking of "best AI." It's a ranking on one cognitive dimension.

---

## Conclusions

### The deepest finding: persistent wrong belief

When LLMs produce illegal chess moves, they don't make random errors. They form **coherent-but-wrong mental models of the position and commit to them across multiple stateless API calls.** Each call is independent from the model's side, but the same FEN deterministically reproduces the same wrong pattern-match.

This is qualitatively different from typical "LLM hallucinations." A standard hallucination is a one-shot plausible-sounding wrong fact. The pattern here is **convergence on the same specific wrong belief across many independent invocations** — implying the belief is the model's deterministic response to a specific input, not a random error. You cannot fix it by sampling more times or with a different temperature; the wrong mental model regenerates from the same FEN.

Retry feedback in the harness gives the model the list of recent failed attempts and asks it to try again. This fixes "wrong move from a roughly-correct mental model" (the model picks a different piece type), but it does NOT fix "wrong mental model of the board" — the model changes the move it commits to but cannot, via text feedback alone, change what it thinks it's looking at. Forfeits in games cluster precisely on plies where all retry attempts share the same structural mistake (none of 4 retries addresses an existing check, because none of them perceive the check). Concrete case studies are in [Supporting evidence](#supporting-evidence) below.

### The in-game cliff is NOT explained by position novelty

A natural reading of the matrix is that mid/endgame positions are "out of training distribution" and that's why models fail there. We tested this directly by running 5 models on 4 progressively-more-novel banks of 20 positions each. Per-position legality:

| Model | T0 hand-curated | T2 unbiased novel | Cliff |
|---|---|---|---|
| `gpt-5` | 1.000 | 1.000 | 0% |
| `gemini-3.1-flash-lite` | 0.950 | 0.900 | -5% |
| `claude-sonnet-4-6` | 0.750 | 0.750 | 0% |
| `claude-opus-4-7` | 0.900 | 0.775 | -14% |
| `deepseek-chat` | 0.500 | 0.225 | -55% |

**The cliff is model-specific.** GPT-5, Flash Lite, and Sonnet handle novel positions essentially as well as memorized ones. Opus has a modest cliff. Only DeepSeek-chat has a sharp one. **Position novelty is NOT what's dragging the matrix Reliability of top cells.** GPT-5 with 100% legality on truly random positions still only scores 0.41 on Reliability. The cliff is in *cumulative gameplay coherence*, not in single-position handling.

### What drives the in-game cliff instead

Three things compound across a 30-40 ply game that single-position tests don't capture:

1. **Move quality (ACPL) degrades by phase even for models with no per-position cliff.** Flash Lite plays at ~62 cp loss in opening, ~128 cp in endgame. Engine-level is ~5-20 cp. The cliff is in *move strength*, not legality — models play legal but mediocre moves on mid/endgame positions.
2. **Cumulative trajectory matters.** Real gameplay between an LLM and a stronger Stockfish opponent drifts into positions where the LLM is under pressure. Random novel banks don't reproduce this — they produce balanced or neutrally-random positions. Real games create *asymmetric pressure* positions that test something different from random novelty.
3. **Persistent wrong belief compounds within games.** A model that gets a wrong picture of the position on ply 24 carries it through retries and into ply 25's similar position. Single-position evaluation can't capture this.

### Other matrix-level patterns

**Reasoning-tier supremacy doesn't hold for spatial state-tracking.** Across the four frontier reasoning models, Reliability spans 0.12 to 0.53 — over a 4× spread. The matrix-leader is a budget non-reasoning model. Reasoning-tier optimization helps when the reasoning fits in the response budget AND when the model can apply it to spatial state — neither is guaranteed.

**Budget beats frontier in 1 of 4 providers.** Only OpenAI shows the inversion is preserved (`gpt-5-mini` 0.301 < `gpt-5` 0.410), but the gap is small. DeepSeek and Anthropic show the expected direction (frontier > budget). Google's two cells (Flash Lite > 2.5 Pro on Reliability, 2.5 Pro > Flash Lite on PlayQuality) split — Flash Lite plays more legal moves; 2.5 Pro plays better moves.

**Failures concentrate on out-of-distribution positions for SOME models.** This is the dimension where models genuinely differ. Standard opening positions in the hand-curated bank show 0% failure rate across every model tested. Synthetic mid-game and endgame positions show 33-67% failure rates on the hardest examples. But — per the bank comparison above — only Opus and DeepSeek-chat show the failure rate jumping when positions become novel. Top cells don't.

---

## Supporting evidence

The conclusions above are backed by these deep dives. Each section explains the data behind one of the claims.

### What this looks like in practice

[Levy Rozman's short "ChatGPT vs Meta AI: This Isn't Chess Anymore"](https://www.youtube.com/shorts/YlMWZNx93G4) is the failure as comedy. Two LLMs trying to play a game produce nonsense after the opening: pieces appear from nowhere, captures are claimed on empty squares, the same illegal move keeps getting proposed, and both models confidently narrate plans for pieces no longer on the board. The benchmark scores numerically what the video shows visually.

### Bank comparison details: 5 models × 4 banks

To test the "is the cliff about position novelty" hypothesis, we ran each of 5 models on 4 banks of 20 positions each:

- **T0 hand-curated**: 20 positions including named openings (Italian, Ruy Lopez, etc.) and textbook endgames. Saturated in training data.
- **T1 real-play extracted**: FENs extracted from real model-vs-Stockfish games where the model later made an illegal move. Out of training distribution, but selection-biased toward difficulty.
- **T2 random-opening + Stockfish continuation**: random 5-10 ply opening + 15 plies Stockfish skill-5 self-play. Mid-game positions with no theory shortcut, engine-realistic structure. **Unbiased.**
- **T3 pure random-vs-random self-play**: 30 plies of random play. Maximally novel — no theory at any stage. **Unbiased.**

Full per-position legality:

| Model | T0 hand | T1 real-play | T2 random-open | T3 random-play |
|---|---|---|---|---|
| `gpt-5` | **1.000** | **1.000** | **1.000** | **1.000** |
| `gemini-3.1-flash-lite` | 0.950 | 0.900 | 0.900 | 0.900 |
| `claude-sonnet-4-6` | 0.750 | 0.500 | 0.750 | 0.700 |
| `claude-opus-4-7` | 0.900 | 0.300 | 0.775 | 0.800 |
| `deepseek-chat` | 0.500 | 0.200 | 0.200 | 0.250 |

T1 numbers are selection-biased — those positions came from games where the model had failed, so they're hard *for that specific model*. T2 and T3 are unbiased random samples and are the cleanest "is this a novelty cliff" test.

### ACPL by phase across the matrix

Move-quality gradient as games proceed (lower is better — engine-level is ACPL ~10):

| Model | ACPL opening | ACPL middlegame | ACPL endgame |
|---|---|---|---|
| `gemini-3.1-flash-lite` | 62 | 53 | 128 |
| `gemini-2.5-pro` | 31 | 99 | 112 |
| `gpt-5` | 75 | 124 | 0 (not reached in 3 games) |
| `deepseek-reasoner` | 131 | 140 | 70 |
| `claude-opus-4-7` | 94 | 240 | 0 (not reached) |
| `claude-haiku-4-5-20251001` | 70 | 0 (not reached) | 0 (not reached) |
| `deepseek-chat` | 80 | 0 (not reached) | 0 (not reached) |

Openings: ACPL 31-131 — meaningfully different across models, but all in "competent club player or stronger" band. Middlegame: ACPL roughly doubles or triples. Endgame: only the strongest cells reach it at all; ACPL doesn't always fall (Flash Lite endgame ACPL is 128, the worst of its three phases).

### Skill sweep — opponent strength is one knob, not the cliff

Published Reliability uses Stockfish skill 3. We ran Flash Lite at skills 1, 5, 10, and 15 to see whether the score is sensitive to opponent strength:

```
Stockfish skill 1   (~1100 ELO, beginner):       CR 0.352
Stockfish skill 5   (~1700 ELO, amateur):        CR 0.645   ← peak
Stockfish skill 10  (~2200 ELO, club master):    CR 0.590
Stockfish skill 15  (~2500 ELO, strong engine):  CR 0.210
```

The curve is **U-shaped**, not monotonic. Score peaks against an intermediate-amateur opponent (skill 5) and degrades in *both* directions — skill 15 collapse from engine pressure, skill 1 drop from random-style opponent making weird moves that push positions out of any theoretical structure. Both ends create out-of-distribution positions through different mechanisms.

### Persistent wrong belief — case studies

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

### Retry feedback efficacy

When a move is illegal, the next attempt is told "your move X was illegal, try again" and given the list of recent failed attempts (capped to the last 3). Across one game's 22 retries, 17 of 22 iterations went to a different piece type (model genuinely updated). The remaining 5 went to the same piece type with a different target, or repeated the same move.

Forfeits cluster on plies where **all retry attempts share the same structural mistake.** A representative case: at one ply Black was in check from a distant bishop. Opus proposed 4 moves over its retry budget — none addressed the check, because Opus didn't perceive the check at all.

**Retry feedback fixes "wrong move from a roughly-correct mental model" but cannot fix "wrong mental model of what's on the board."**

### Reasoning effort vs move quality — suggestive evidence

Reasoning models from OpenAI and Gemini support an effort dial. The benchmark logs effort level on each call; on length-failure errors, the harness steps the effort down for the next retry. Aggregating cp_loss bucketed by the effort level on the final successful attempt:

| Model | default cp / n | medium cp / n | low cp / n | minimal cp / n |
|---|---|---|---|---|
| `gemini-2.5-pro` | 62 / 210 | 47 / 17 | 52 / 7 | (n/a) |
| `gpt-5` | 1 / 47 | (n/a) | 64 / 25 | 67 / 154 |
| `gpt-5-mini` | 33 / 108 | 51 / 3 | 86 / 41 | 93 / 9 |

The pattern: **quality at lower reasoning effort is comparable to quality at default effort.** Suggestive but has a confound — default and stepped-down buckets sample different positions. A controlled experiment (same positions, varying effort, paired cp_loss) is queued as post-launch work.

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

### Legality and rule-claim consistency expose different weaknesses

Across the 20-position hand-curated bank:

- 6 positions fail both legality and consistency
- 4 positions fail legality only
- **19 positions fail consistency only** — model picks a legal move, but its claim about whether it's check, capture, or mate is wrong
- 11 positions pass both

Consistency-only failures are the largest bucket. **The model picks correctly but can't always tell you what the move does.**

---

## Footnotes on the matrix

<a id="footnote-a"></a>
***ᵃ*** **`0.068`–`0.086` (Opus PlayQuality, DeepSeek-reasoner PlayQuality)** — model produces legal moves on most first attempts (63–78%) and plays moves of reasonable quality (cp_loss 80–240 on the plies it reaches), but the games rarely survive into endgame at Stockfish skill 5. With only 3 retries permitted in PlayQuality mode (vs Reliability's 10), even occasional length-failures and run-of-bad-positions cause early forfeits. Result: the high-weight late plies are mostly missing from the numerator while still counting toward the denominator.

<a id="footnote-b"></a>
***ᵇ*** **`0.014`–`0.040` (Haiku, DeepSeek-chat — both metrics)** — first-attempt-legal rate of 38–54% means roughly half of all moves are illegal on the first try. Each of those moves pays the `0.25^retries` cost (1 retry → 25% credit, 2 retries → 6%). Combined with games averaging only 10–20 plies before forfeiting on illegal moves they can't recover from, the score is dominated by missing late-game plies and retry-cost-eroded mid-game plies. This isn't a metric pathology — it's the metric correctly reflecting "model cannot reliably play legal chess past the opening, even with the retry safety net."

<a id="footnote-c"></a>
***ᶜ*** **`0.410` (GPT-5 Reliability with 97.8% first-attempt-legal)** — the asterisk here is the *opposite* situation: the model produces a legal move on first attempt ~98% of the time, but the composite is still moderate because Reliability multiplies three factors and first-attempt-legality is only one of them. GPT-5's mean cp_loss across legal moves is 86 → move_quality = exp(−86/150) = 0.564 (competent club play, not engine-level). One of the 5 games forfeited at ply 1, contributing 0 to the mean and subtracting ~0.20 from the composite. The remaining 4 games averaged 27 plies (vs max 40), missing the highest-weighted late-game plies. **High first-attempt-legal does not imply high Reliability** — read the columns together. [METHODOLOGY § Reliability is NOT first-attempt-legal](METHODOLOGY.md#reliability-is-not-first-attempt-legal--they-measure-different-things) walks through the decomposition.

<a id="footnote-d"></a>
***ᵈ*** **`0.15` (GPT-5 avg retries/move) with 97.8% first-attempt-legal** — this number can read as a contradiction with Gemini 2.5 Pro's `0.07 / 93.8%`: GPT-5 fails *less often* yet shows *more retries on average*. The retry distributions explain it. Gemini's 11 failed plies recovered in 1–2 retries each (total 12 retries across 11 failures). GPT-5's 3 failed plies needed 1, 10, and 10 retries respectively (total 21 retries across 3 failures) — when GPT-5 fails it usually burns through the entire stepdown ladder (`default → medium → low → minimal`) before producing a legal SAN. Two qualitatively different failure profiles: Gemini's failures are quick to recover; GPT-5's are reasoning-budget exhaustion that exhaust the full retry budget.

---

## The bottom line

The benchmark measures one cognitive primitive: whether LLMs can pattern-match to good chess moves from current position alone, across stateless calls. The deepest claim is qualitative: **models form coherent-but-wrong mental models of the board and commit to them deterministically across stateless calls.** This is qualitatively different from typical hallucinations and is unlikely to be fixed by bigger context windows or more training data alone.

Position novelty is NOT the explanation for in-game degradation in the top-of-matrix cells. Per-position legality tests across four progressively-more-novel banks show that Flash Lite, GPT-5, and Sonnet handle novel positions essentially as well as memorized ones (Opus has a modest cliff; only DeepSeek-chat has a sharp one). What drags matrix Reliability down for the top cells is move quality and cumulative gameplay dynamics — the cliff is in *cumulative coherence across turns*, not in *can the model handle one novel position*.

The strongest model scores 0.639 on a [0, 1] scale where engine-quality self-play is the 1.0 reference. The ranking does not track "reasoning tier" or "frontier vs budget" — it tracks something more specific about coherent state-tracking under stateless inference.

The scoring is designed to be **hill-climbable**. The current matrix top (0.64) leaves real room above 0.9, where engine-quality play would land. That headroom is not noise — it's the gap between today's frontier and a model that can keep its mental picture of an 8×8 board accurate across 40 moves, without external scaffolding.
