# LLM Chess Eval — Results

> **Want to see the failure mode this benchmark measures?** Watch [Levy Rozman's 2-minute short of ChatGPT vs Meta AI playing chess](https://www.youtube.com/shorts/YlMWZNx93G4) — pieces appear from nowhere, captures are claimed on empty squares, and both models confidently narrate plans for pieces no longer on the board. This benchmark scores numerically what that video shows visually.

What we found running the benchmark across nine model-tier cells from four providers (Anthropic, OpenAI, Google, DeepSeek). For the formulas, scoring rationale, and methodology constraints, see **[METHODOLOGY.md](METHODOLOGY.md)**. For install and run instructions, see **[HOWTO.md](HOWTO.md)**.

---

## Summary

Nine cells, four providers, ~1,300 games played end-to-end. Top scores:

- **Gemini 2.5 Pro (frontier)** and **Flash Lite (budget non-reasoning)** are essentially tied at the top with PlayStrength **0.485 / 0.477** on a [0, 1] scale where engine-quality self-play would be 1.0.
- Frontier reasoning models from OpenAI, DeepSeek, and Anthropic cluster around **0.28–0.30**.
- Budget cells from Anthropic and DeepSeek collapse under **0.10** due to high forfeit rates.

**Three key findings:**

1. **The deepest claim is qualitative.** When LLMs produce illegal chess moves they don't make random errors — they form coherent-but-wrong mental models of the board and commit to them deterministically across multiple stateless API calls. A failure mode distinct from typical hallucinations, and not fixable by sampling more times or with a different temperature.
2. **Reasoning-tier and frontier-vs-budget are not strong predictors.** The matrix co-leader is a budget non-reasoning model (Flash Lite). It *exceeds* its frontier sibling 2.4× on the supplemental MoveQuality metric.
3. **Position novelty is not the main driver of in-game failure** for top cells. Per-position legality on progressively-more-novel banks shows that GPT-5, Flash Lite, and Sonnet handle novel positions essentially as well as memorized ones. What drags the matrix down for top cells is *cumulative coherence failure across turns* — ACPL rising through opening → middlegame → endgame — not per-position novelty.

## Scope

LLMs are pattern-matching transformers — closer to how humans think than to how chess engines search. Engines like Stockfish are exhaustive search systems; they don't need memory because they compute optimal moves from current state. Humans (and LLMs) pattern-match. For pattern-matching cognition, memory isn't a crutch — it's how the system works.

**This benchmark deliberately strips memory and tools to measure a specific cognitive primitive in isolation: pattern-matching to good moves from the current position alone, across many independent calls.** Real production usage compensates with context (history, prior reasoning, retrieval) — that's appropriate because LLMs are pattern-matchers, not deep searchers. The benchmark removes that scaffolding to test the primitive cleanly. If models score higher with memory or tools, that's a measure of how much cognitive work is being externalized to context — not a refutation of the benchmark.

The matrix below ranks models on this primitive. A model's score here doesn't predict its score on other benchmarks — chess-style spatial reasoning is one capability among many.

---

## The matrix

> **Heads up — two counter-intuitions to know before reading the table:**
>
> **(1) High rule-following does NOT guarantee high PlayStrength.** Flash Lite (86.5% legal, 5% forfeit) scores higher than GPT-5 (99.8% legal, 0% forfeit) because PlayStrength is a composite — moves' quality and game depth matter alongside legality. Flash Lite plays substantially stronger moves (opening ACPL 45 vs 85) and reaches the endgame consistently (mean ply 43 vs 27). GPT-5 plays disciplined but mid-quality moves. **The composite ranks the stronger chess player, not the more rule-compliant one.** Full decomposition in [The Flash Lite outlier](#the-flash-lite-outlier--why-a-budget-non-reasoning-model-nearly-tops-the-matrix).
>
> **(2) PlayStrength and MoveQuality are *separate experiments*, not one metric with a factor on/off.** PS runs games vs Stockfish skill 3 (easier amateur, 10-retry budget, max 40 plies). MQ runs separate games vs Stockfish skill 5 (intermediate amateur, 3-retry budget, max 60 plies). A model can score high on one and low on the other. **Gemini 2.5 Pro is the extreme case** — tops the matrix on PS (0.485) but lands near bottom on MQ (0.192) because it plays clean disciplined games against skill 3 but breaks down under skill 5 pressure (mean MQ game length only 20.5 of 60 plies; endgame ACPL spikes to 168 = blunder territory). Flash Lite is the opposite — strong on both (0.477 / 0.466) because it plays consistently across difficulty. **The PS-vs-MQ gap is itself diagnostic**: a model with PS ≫ MQ is "strong under forgiving conditions, breaks under pressure"; a model with PS ≈ MQ is "robustly strong."

Nine cells: frontier and budget tier across four providers, plus Sonnet as Anthropic mid-tier. Each cell ran 20+ PlayStrength games and 10+ MoveQuality games (most ran much more — see N column). Sorted by PlayStrength.

| Provider | Tier | Model | PlayStrength | MoveQuality | first-legal | avg retries | forfeit rate | N (PS/MQ) |
|---|---|---|---|---|---|---|---|---|
| Google | frontier | `gemini-2.5-pro` | **0.485** | 0.192 | 93.2% | 0.10 | 0% | 160/88 |
| Google | budget | `gemini-3.1-flash-lite` | 0.477 | **0.466** | 86.5% | 0.20 | 5% | 160/80 |
| OpenAI | frontier | `gpt-5` | 0.301 [*ᶜ*](#footnote-c) | 0.237 | **99.8%** | **0.00** | **0%** | 166/49 |
| DeepSeek | frontier | `deepseek-reasoner` | 0.288 | 0.204 | 79.4% | 0.25 | 0% | 112/57 |
| Anthropic | frontier | `claude-opus-4-7` | 0.281 [*ᵃ*](#footnote-a) | 0.144 | 71.3% | 0.60 | 30% | 160/80 |
| OpenAI | budget | `gpt-5-mini` | 0.279 | 0.149 | 88.0% | 0.13 | 0% | 72/14 |
| Anthropic | mid | `claude-sonnet-4-6` | 0.149 | 0.086 | 63.3% | 0.69 | 20% | 142/53 |
| DeepSeek | budget | `deepseek-chat` | 0.097 [*ᵇ*](#footnote-b) | 0.046 [*ᵇ*](#footnote-b) | 33.9% | 2.29 | 80% | 172/80 |
| Anthropic | budget | `claude-haiku-4-5-20251001` | 0.074 [*ᵇ*](#footnote-b) | 0.053 [*ᵇ*](#footnote-b) | 42.9% | 1.95 | 95% | 176/80 |

Two scores per model, both bounded `[0, 1]`, **each from a separate set of games**. **PlayStrength** is the headline composite, run vs Stockfish skill 3 (~1500 ELO amateur, max 40 plies, 10-retry budget). It multiplies move quality, retry-cost discipline (`0.25^retries`), and a phase weight that counts late-game plies more. **MoveQuality** is the supplemental, run vs Stockfish skill 5 (~1700 ELO intermediate amateur, max 60 plies, 3-retry budget). Same formula minus the retry-cost factor — isolating "how good are the moves themselves" once a legal one is found. Stockfish self-play would score 1.0 on both; the published PlayStrength top is 0.485.

**Diagnostic columns** are from the PlayStrength runs (skill 3, 10 retries): **first-legal** = fraction of plies where the model's first proposal was legal (zero retries); **avg retries** = total retries divided by total plies; **forfeit rate** = fraction of games that forfeited within the 10-retry budget. Full formulas in [METHODOLOGY.md](METHODOLOGY.md).

### Reading the matrix

The first-legal, avg-retries, and forfeit-rate columns split the matrix into three clean behavioral bands:

**Band 1 — "Plays legal chess on first attempt":** GPT-5, Gemini Pro, GPT-5-mini, Flash Lite, DeepSeek-reasoner. First-legal 79–100%, avg retries 0.00–0.25, **0–5% forfeit rate**. The PlayStrength score here is mostly about *move quality* on the moves played — these models don't need the retry safety net.

**Band 2 — "Needs the retry safety net":** Claude Opus, Claude Sonnet. First-legal 63–71%, avg 0.60–0.69 retries per move, **20–30% forfeit rate**. Roughly one move in three is illegal on first try; the harness feeds back errors before the model corrects. Even after recovery the `0.25^retries` cost ensures retried moves contribute little to the composite. Some games never recover.

**Band 3 — "Struggles to even propose legal moves":** Claude Haiku, DeepSeek-chat. First-legal 34–43%, avg 1.95–2.29 retries per move, **80–95% forfeit rate**. About half of all moves are illegal on first try, and the games almost never survive to completion. Haiku forfeited 168 of 176 games (95%); DeepSeek-chat forfeited 137 of 172 (80%).

**Forfeit rate is the cleanest single diagnostic.** GPT-5 played 100% of games to completion; Haiku finished 5%. That gap reflects "can the model keep producing legal moves end-to-end" more directly than any composite score.

## Footnotes on the matrix

<a id="footnote-a"></a>
***ᵃ*** **`0.281` (Opus PlayStrength with 30% forfeit rate)** — Opus has 71.3% first-attempt-legal and 0.60 avg retries per move — meaningfully worse on rule-following than the Band-1 cells (79–100% first-legal). 48 of 160 games forfeited because retries couldn't recover a legal move. The PlayStrength composite is therefore a mix of "good Opus games that survive (mean ~21 plies on the games that complete)" with "lots of zero-scored forfeit games dragging down the mean." If we restricted to non-forfeit games only, the per-game mean would be substantially higher — but the metric is *intentionally* counting forfeits at 0 because forfeit-prone play is the failure mode the benchmark exists to measure.

<a id="footnote-b"></a>
***ᵇ*** **`0.046`–`0.097` (Haiku, DeepSeek-chat — both metrics)** — first-attempt-legal rate of 34–43% means the majority of moves are illegal on first try. Each of those moves pays the `0.25^retries` cost (1 retry → 25% credit, 2 retries → 6%). Worse: forfeit rates of 80–95% mean these models almost never complete a game, so most game-scores are at or near zero. Haiku forfeited 168 of 176 PlayStrength games; DeepSeek-chat forfeited 137 of 172. This isn't a metric pathology — it's the metric correctly reflecting "model cannot reliably play legal chess past the opening, even with a 10-retry safety net."

<a id="footnote-c"></a>
***ᶜ*** **`0.301` (GPT-5 PlayStrength with 99.8% first-attempt-legal, zero retries, zero forfeits)** — GPT-5 wins every rule-following column in the matrix yet scores *below* Flash Lite (0.477). Why: PlayStrength multiplies three factors, and rule-following is only one of them. GPT-5's mean cp_loss across legal moves is ~85 → move_quality = exp(−85/150) ≈ 0.57 (competent club, not engine-level). Flash Lite plays at opening ACPL 45 → quality ~0.74 per move, and reaches mean ply 43 in MoveQuality games (vs GPT-5's 27) so it captures the high-weight late-game phase. GPT-5's perfect retry-cost factor gives it a ~15% advantage on that single axis; Flash Lite's move-quality + game-depth advantages are worth ~3× that. **In chess terms: GPT-5 is a perfectly-disciplined intermediate-club player; Flash Lite is a strong-club player who occasionally needs a do-over.** The composite calls Flash Lite the stronger player — which is the right call by any standard chess-strength definition. **High first-attempt-legal does not imply high PlayStrength** — read the columns together. See [The Flash Lite outlier](#the-flash-lite-outlier--why-a-budget-non-reasoning-model-nearly-tops-the-matrix) for full decomposition, and [METHODOLOGY § PlayStrength is NOT first-attempt-legal](METHODOLOGY.md#playstrength-is-not-first-attempt-legal--they-measure-different-things) for the underlying math.

---

## Supporting data

The findings below are backed by these data tables and qualitative evidence. Each section provides the data for one of the claims made in [Findings and explanations](#findings-and-explanations).

### Per-position legality across novelty-tiered banks (5 models × 4 banks)

To test the "is the cliff about position novelty" hypothesis, we ran each of 5 models on 4 banks of 20 positions each:

- **T0 hand-curated**: 20 positions including named openings (Italian, Ruy Lopez, etc.) and textbook endgames. Saturated in training data.
- **T1 real-play extracted**: FENs extracted from real model-vs-Stockfish games where the model later made an illegal move. Out of training distribution, but selection-biased toward difficulty.
- **T2 random-opening + Stockfish continuation**: random 5–10 ply opening + 15 plies Stockfish skill-5 self-play. Mid-game positions with no theory shortcut, engine-realistic structure. **Unbiased.**
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
| `gemini-2.5-pro` | 34 | 78 | 168 |
| `gemini-3.1-flash-lite` | 45 | 77 | 114 |
| `deepseek-chat` | 65 | not reached | not reached |
| `claude-opus-4-7` | 66 | 103 | 123 |
| `gpt-5` | 85 | 111 | 102 |
| `deepseek-reasoner` | 85 | 172 | 118 |
| `gpt-5-mini` | 101 | 117 | not reached |
| `claude-haiku-4-5-20251001` | 103 | not reached | not reached |
| `claude-sonnet-4-6` | 104 | 235 | not reached |

Openings: ACPL 34–104 — meaningfully different across models, but all in "competent club player or stronger" territory. Middlegame: ACPL roughly doubles or triples for every model that reaches it. Endgame: only 5 of 9 cells reach it at all; ACPL doesn't always degrade further (Flash Lite endgame ACPL 114 is its worst phase but only modestly worse than middle 77; GPT-5's endgame ACPL 102 is actually *better* than its middle 111).

### Mean legal plies survived (MoveQuality, max 60)

How far into a game the model gets before forfeit or natural termination:

| Model | mean plies | of 60 |
|---|---|---|
| `gemini-3.1-flash-lite` | **43.1** | 72% |
| `gpt-5` | 26.9 | 45% |
| `deepseek-reasoner` | 25.3 | 42% |
| `gemini-2.5-pro` | 20.5 | 34% |
| `gpt-5-mini` | 19.9 | 33% |
| `claude-opus-4-7` | 19.2 | 32% |
| `claude-sonnet-4-6` | 14.9 | 25% |
| `claude-haiku-4-5-20251001` | 10.0 | 17% |
| `deepseek-chat` | 7.6 | 13% |

Flash Lite reaches the endgame consistently — its mean is **2× the runner-up**. Note Gemini 2.5 Pro's anomalous low number: it's the top PlayStrength scorer but only 20.5 plies on MoveQuality. MoveQuality uses a harder Stockfish opponent (skill 5 vs 3); Pro survives long enough to score reliably on opening/early-mid plies but doesn't push deep into harder positions. Flash Lite's larger games-played ceiling is why its MoveQuality (0.466) is 2.4× Pro's MoveQuality (0.192).

### Skill sweep — opponent strength is one knob, not the cliff

Published PlayStrength uses Stockfish skill 3. We ran Flash Lite at skills 1, 5, 10, and 15 to see whether the score is sensitive to opponent strength (numbers below are at the older 1/2/4/8 phase weighting but the curve shape is what matters):

```
Stockfish skill 1   (~1100 ELO, beginner):       PS 0.352
Stockfish skill 5   (~1700 ELO, amateur):        PS 0.645   ← peak
Stockfish skill 10  (~2200 ELO, club master):    PS 0.590
Stockfish skill 15  (~2500 ELO, strong engine):  PS 0.210
```

The curve is **U-shaped**, not monotonic. Score peaks against an intermediate-amateur opponent (skill 5) and degrades in *both* directions — skill 15 collapse from engine pressure, skill 1 drop from random-style opponent making weird moves that push positions out of any theoretical structure. Both ends create out-of-distribution positions through different mechanisms.

### Methodology robustness — τ sensitivity

The `move_quality` decay constant is τ=150. We re-scored the matrix at τ ∈ {100, 200, 300}. **Rankings are identical** across the τ range; absolute scores shift ~20%:

| Model | τ=100 | τ=150 (published) | τ=200 | τ=300 |
|---|---|---|---|---|
| `gemini-2.5-pro` | 0.443 | 0.485 | 0.512 | 0.545 |
| `gemini-3.1-flash-lite` | 0.434 | 0.477 | 0.504 | 0.537 |
| `gpt-5` | 0.275 | 0.301 | 0.318 | 0.339 |
| `deepseek-reasoner` | 0.262 | 0.288 | 0.305 | 0.326 |
| `claude-opus-4-7` | 0.254 | 0.281 | 0.299 | 0.322 |
| `gpt-5-mini` | 0.254 | 0.279 | 0.296 | 0.316 |
| `claude-sonnet-4-6` | 0.133 | 0.149 | 0.160 | 0.174 |
| `deepseek-chat` | 0.088 | 0.097 | 0.102 | 0.108 |
| `claude-haiku-4-5-20251001` | 0.067 | 0.074 | 0.078 | 0.084 |

The choice of τ=150 isn't load-bearing on any conclusion — pick τ=100 or τ=300, the relative ordering is invariant.

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

## Findings and explanations

### The deepest finding: persistent wrong belief

When LLMs produce illegal chess moves, they don't make random errors. They form **coherent-but-wrong mental models of the position and commit to them across multiple stateless API calls.** Each call is independent from the model's side, but the same FEN deterministically reproduces the same wrong pattern-match.

This is qualitatively different from typical "LLM hallucinations." A standard hallucination is a one-shot plausible-sounding wrong fact. The pattern here is **convergence on the same specific wrong belief across many independent invocations** — implying the belief is the model's deterministic response to a specific input, not a random error. You cannot fix it by sampling more times or with a different temperature; the wrong mental model regenerates from the same FEN.

Retry feedback in the harness gives the model the list of recent failed attempts and asks it to try again. This fixes "wrong move from a roughly-correct mental model" (the model picks a different piece type), but it does NOT fix "wrong mental model of the board" — the model changes the move it commits to but cannot, via text feedback alone, change what it thinks it's looking at. Forfeits in games cluster precisely on plies where all retry attempts share the same structural mistake — see the [case studies](#persistent-wrong-belief--case-studies) and [retry feedback efficacy](#retry-feedback-efficacy) data above.

### The Flash Lite outlier — why a budget non-reasoning model nearly tops the matrix

Flash Lite is the single most surprising cell. It's a budget non-reasoning model from Google that essentially **ties** the frontier reasoning model from the same family on PlayStrength (0.477 vs Gemini 2.5 Pro's 0.485) and **dominates** every other cell on MoveQuality (0.466 vs the next-best at 0.237). And it does this *despite* being demonstrably worse on rule-following than the top OpenAI cell:

| | Flash Lite | GPT-5 |
|---|---|---|
| first-attempt-legal | 86.5% | **99.8%** |
| avg retries / move | 0.20 | **0.00** |
| forfeit rate | 5% (8/160) | **0% (0/166)** |
| PlayStrength | **0.477** | 0.301 |
| MoveQuality | **0.466** | 0.237 |

How does the worse-on-legality model score 1.5–2× higher on the composite? Three contributors stack:

1. **Move quality is substantially better when Flash Lite plays a legal move.** Opening ACPL 45 vs GPT-5's 85; middlegame 77 vs 111. Flash Lite plays at strong club-player strength; GPT-5 plays at intermediate club strength. The `exp(-cp_loss/150)` move-quality function is steep at the middle of its range, so a 40-point ACPL gap produces a ~30% per-move quality difference that compounds across 25–40 plies per game.

2. **Flash Lite reaches the endgame consistently; GPT-5 does not.** MoveQuality mean legal plies: Flash Lite 43.1, GPT-5 26.9. The phase-weight function gives ply-30+ plies 3× the weight of opening plies, so a model that survives 43 plies sees substantially more high-weight contribution to its score than one that stops at 27.

3. **The 5% forfeit rate is real but not crippling.** 8 of 160 PlayStrength games forfeited — those games contribute 0 to the mean, subtracting ~0.025 from the composite. The remaining 152 games are enough to dominate. Compare Haiku's 95% forfeit rate, which mathematically caps the composite at ~0.05 regardless of how well the surviving games play.

**The story isn't "Flash Lite has bad rule-following despite good play."** It's that this composite metric weights move quality and game survival heavily, and Flash Lite trades off "near-perfect single-move legality" for "stronger moves overall and deeper games" — the trade pays off in the composite. Three factors multiply; legality is one of them.

A reasonable reading: **Flash Lite's pre-training has more chess-relevant pattern coverage per parameter than its reasoning-tier siblings.** The Gemini family carries strong chess priors (Flash Lite handles unbiased novel positions at 0.90 legality, only 0.05 below the T0 hand-curated bank), and the budget non-reasoning model gets to deploy that pattern-matching directly without paying for reasoning that, for GPT-5, doesn't translate into substantially better play on this task.

**This pattern doesn't generalize.** There's no rule that smaller models win on this benchmark. GPT-5-mini (also budget, also non-reasoning-tier-by-default for many calls) scores 0.279 on PlayStrength and 0.149 on MoveQuality — comparable to its frontier sibling, not exceeding it. DeepSeek-chat (budget non-reasoning) collapses at 0.097. The Flash Lite outlier is specifically about how Google's chess pre-training scales down to the budget cell. It is an *observation about Flash Lite*, not a claim about "budget tier beats frontier."

### A note on the Anthropic scores

Anthropic's scores are the lowest among the four providers on both PlayStrength and MoveQuality, except for Opus which is now competitive with mid-pack OpenAI/DeepSeek frontier cells. This deserves explanation because the picture is easy to misread.

**Anthropic's numbers are not artifacts of methodology.** During the benchmark build, OpenAI's `max_completion_tokens` and Google's `max_output_tokens` were initially set too low (2048) — both providers count reasoning + visible output in that budget, so reasoning models hit the cap before emitting the tool call. We discovered this, raised the budget to 65536, and re-ran the matrix. Anthropic was unaffected throughout: `max_tokens` on Anthropic counts output tokens only, with extended thinking on a separate budget. Anthropic was also capped at 8192 (vs 65536 for other providers) because Anthropic's SDK refuses non-streaming calls above that — but that ceiling is on *visible output*, not reasoning, and Anthropic's models comfortably emit their tool calls within it. The Anthropic numbers reflect genuine model behavior at full reasoning budgets.

**What the matrix exposes is that Anthropic's reasoning models struggle specifically with sustained 2D spatial state-tracking** — exactly the cognitive failure mode this benchmark was designed to isolate. Anthropic excels on benchmarks that reward strong single-shot reasoning over textual or symbolic state (MMLU, GPQA, coding). It underperforms here because the benchmark is biased toward the dimension where Anthropic is comparatively weakest. The same models that fail to reach the endgame in chess routinely solve graduate-level math and write production code — chess-style spatial reasoning is a specific weakness, not a general one.

Within the Anthropic family the standard "Haiku < Sonnet < Opus" capability ordering does hold here — Haiku 0.074, Sonnet 0.149, Opus 0.281 — but the spread is unusually wide. The model that's *strongest* on this dimension (Opus) is only mid-pack overall.

The matrix is not a ranking of "best AI." It's a ranking on one cognitive dimension.

### The in-game cliff is NOT explained by position novelty

A natural reading of the matrix is that mid/endgame positions are "out of training distribution" and that's why models fail there. We tested this directly — see [Per-position legality across novelty-tiered banks](#per-position-legality-across-novelty-tiered-banks-5-models--4-banks) above.

**The cliff is model-specific.** GPT-5, Flash Lite, and Sonnet handle novel positions essentially as well as memorized ones. Opus has a modest cliff. Only DeepSeek-chat has a sharp one. **Position novelty is NOT what's dragging the matrix PlayStrength of top cells.** GPT-5 with 100% legality on truly random positions still only scores 0.301 on PlayStrength — and it has 99.8% first-legal rate and zero forfeits across 166 games. Its score is dragged down purely by *move quality* (mean cp_loss ~85), not by failing to find legal moves. The cliff is in *cumulative gameplay coherence*, not in single-position handling.

### What drives the in-game cliff instead

Three things compound across a 30–40 ply game that single-position tests don't capture:

1. **Move quality (ACPL) degrades by phase even for models with no per-position cliff.** Flash Lite plays at ~45 cp loss in opening, ~114 cp in endgame. Engine-level is ~5–20 cp. The cliff is in *move strength*, not legality — models play legal but mediocre moves on mid/endgame positions. (See [ACPL by phase](#acpl-by-phase-across-the-matrix) above.)
2. **Cumulative trajectory matters.** Real gameplay between an LLM and Stockfish drifts into positions where the LLM is under pressure. Random novel banks don't reproduce this — they produce balanced or neutrally-random positions. Real games create *asymmetric pressure* positions that test something different from random novelty.
3. **Persistent wrong belief compounds within games.** A model that gets a wrong picture of the position on ply 24 carries it through retries and into ply 25's similar position. Single-position evaluation can't capture this.

### Other matrix-level patterns

**Reasoning-tier supremacy doesn't hold for spatial state-tracking.** Across the four frontier reasoning models, PlayStrength spans 0.28 to 0.49 — a ~1.7× spread that doesn't track the relative ordering on other benchmarks. Reasoning-tier optimization helps only when the reasoning fits in the response budget AND when the model can apply it to spatial state — neither is guaranteed. Within-family budget-vs-frontier comparisons: Google has Flash Lite essentially tied with Pro (Flash Lite outlier explained above); OpenAI has GPT-5-mini ~7% below GPT-5; DeepSeek and Anthropic preserve the expected frontier > budget direction with clear gaps. Tier is not a strong predictor on this benchmark.

**Failure rates concentrate on out-of-distribution positions for SOME models.** This is the dimension where models genuinely differ. Standard opening positions in the hand-curated bank show 0% failure rate across every model tested. Synthetic mid-game and endgame positions show 33–67% failure rates on the hardest examples. But — per the bank comparison above — only Opus and DeepSeek-chat show the failure rate jumping when positions become novel. Top cells don't.

---

## Related work

Several prior efforts have examined LLM chess capability and adjacent failure modes. This work builds on them; it does not exist in a vacuum.

- **Adam Karvonen** and others (2023–2024) showed that `gpt-3.5-turbo-instruct` plays at roughly 1700-1800 ELO when prompted with a chess-game-style prefix, with characteristic blunders concentrated in mid/endgame. That work focused on a single OpenAI model and used playing strength (ELO) as the headline metric.
- **DeepMind's "Grandmaster-Level Chess Without Search"** (Ruoss et al., 2024) trained a 270M-parameter transformer to play ~2300 ELO via supervised learning on Stockfish-labeled positions — demonstrating that transformer architectures *can* learn chess pattern-matching well when explicitly trained for it. General-purpose LLMs are not trained that way, which makes the comparison to chess-specialist transformers informative rather than damning.
- **Levy Rozman's viral video** (linked at the top) documented illegal-move sequences, phantom pieces, and incoherent plans in LLM-vs-LLM play. It's the qualitative version of what this benchmark scores numerically.
- **Various blog posts and Twitter threads** documenting illegal-move rates, fork blindness, and hallucinated piece positions in GPT-3.5/4 and Claude across 2023–2025. Most measured single-position legality without composing it into a cumulative-game score.

What this benchmark contributes beyond prior work:

1. **Cross-provider matrix at consistent methodology.** Most prior efforts evaluated single models with different prompts and metrics. The matrix here applies the same prompt schema, same retry budget, same scoring formula across nine cells from four providers.
2. **Persistent wrong belief as a framed phenomenon.** Convergence on the same wrong move across stateless API calls is documented in prior threads anecdotally, but not articulated as a deterministic-not-random failure mode that's qualitatively distinct from typical hallucinations. The framing is what's new; the data exists in other places.
3. **Multi-bank novelty test as direct experimental check.** Prior work inferred memorization indirectly from ACPL phase gradients. The 5-model × 4-bank gradient (hand-curated → real-play extracted → random-opening → pure random) tests memorization directly — and reveals that the cliff is model-specific, not universal. That nuance changes the story.
4. **Composite scoring with multiplicative factors and softened phase weight.** Most prior chess-LLM scoring used simple legality % or ELO. The scoring here is designed to (a) leave headroom at the top so the metric is hill-climbable, (b) penalize retry-as-crutch behavior, and (c) weight late-game plies higher to bake the memorization-cliff thesis into the score.

The contribution is more methodological than novel-result. The underlying observation that LLMs play chess poorly is not in dispute; the contribution is a structured way to measure how poorly, what mechanism drives the failure, and how the failure differs across models in ways the single-model literature doesn't capture.

---

## The bottom line

The benchmark measures one cognitive primitive: whether LLMs can pattern-match to good chess moves from current position alone, across stateless calls. The deepest claim is qualitative: **models form coherent-but-wrong mental models of the board and commit to them deterministically across stateless calls.** This is qualitatively different from typical hallucinations and is unlikely to be fixed by bigger context windows or more training data alone.

Position novelty is NOT the explanation for in-game degradation in the top-of-matrix cells. Per-position legality tests across four progressively-more-novel banks show that Flash Lite, GPT-5, and Sonnet handle novel positions essentially as well as memorized ones (Opus has a modest cliff; only DeepSeek-chat has a sharp one). What drags matrix PlayStrength down for the top cells is **move quality** (engine-level play scores ~0.95; current top cells play at ACPL ~50–80 = competent club, mid-0.5 quality) and **forfeit rate** (Band-2 and Band-3 models lose games entirely to retry exhaustion). The cliff is in *cumulative coherence across turns*, not in *can the model handle one novel position*.

The strongest model scores 0.485 on a [0, 1] scale where engine-quality self-play is the 1.0 reference. The ranking does not track "reasoning tier" or "frontier vs budget" — Flash Lite (budget non-reasoning) is essentially tied with Gemini 2.5 Pro (frontier reasoning) on PlayStrength, and exceeds it 2.4× on MoveQuality. The benchmark tracks something more specific about coherent state-tracking under stateless inference.

The scoring is designed to be **hill-climbable**. The current matrix top (0.485) leaves real room above 0.9, where engine-quality play would land. That headroom is not noise — it's the gap between today's frontier and a model that can keep its mental picture of an 8×8 board accurate across 40 moves, without external scaffolding.
