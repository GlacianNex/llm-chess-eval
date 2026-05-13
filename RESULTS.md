# LLM Chess Eval — Results

What we found running the benchmark across eight model-tier cells from four providers (Anthropic, OpenAI, Google, DeepSeek). For the formulas, scoring rationale, methodology constraints, and reproduction recipe, see **[METHODOLOGY.md](METHODOLOGY.md)**.

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

## Headline

**The strongest model in the matrix achieves ChessReliability 0.639 on a [0, 1] scale where Stockfish self-play is the 1.0 reference.** That's a budget non-reasoning model (Gemini 3.1 Flash Lite) producing a legal move on the first attempt (zero retries) ~88% of the time and reaching mid/endgame with mediocre move quality. Frontier reasoning models from Anthropic and DeepSeek score below it. The bottom of the matrix sits at 0.032 — models that forfeit before reaching mid-game.

GPT-5 has the **highest first-attempt-legal rate of any cell at 97.8%** but a lower composite score because the rare illegal-move cases burn more retry-cost recovery.

The ranking does not track "reasoning tier" or "frontier vs budget". It tracks something more specific: whether the model can produce a legal SAN move on first attempt while staying alive long enough to reach the positions outside its training distribution.

---

## What the metrics measure (brief)

Two scores per model, both bounded `[0, 1]`. Full formulas live in [METHODOLOGY.md](METHODOLOGY.md).

- **ChessReliability** — rule-following over full games vs Stockfish skill 3, with up to 10 retries per illegal move at steep per-retry cost (`0.25^retries`). Move quality decays exponentially in centipawn loss (`exp(-cp_loss / 150)`). Each ply is weighted geometrically by game phase (1 / 2 / 4 / 8 at ply boundaries 10 / 20 / 30) so late-game plies, where training distribution ends, dominate the score.
- **PlayQuality** — move strength once a legal move is found. Same per-move quality and phase weighting as Reliability, but without the retry cost (retries don't hurt the score once a legal move is found). Played at Stockfish skill 5 with max 3 retries.

Two diagnostic columns alongside the composite scores make the matrix readable: **first-attempt-legal rate** (fraction of plies where the very first model proposal was legal) and **average retries per move**. Together they distinguish "scored high because moves were legal first try" from "scored high because the retry mechanism rescued the model."

---

## The matrix

Eight cells: frontier and budget tier across four providers. Sorted by ChessReliability.

| Provider | Tier | Model | Reliability | PlayQuality | first-attempt legal | avg retries/move |
|---|---|---|---|---|---|---|
| Google | budget | `gemini-3.1-flash-lite` | **0.639** | 0.217 | 87.8% | 0.16 |
| Google | frontier | `gemini-2.5-pro` | 0.527 | **0.274** | 93.8% | 0.07 |
| OpenAI | frontier | `gpt-5` | 0.410 | _re-running_ | **97.8%** | 0.15 |
| DeepSeek | frontier | `deepseek-reasoner` | 0.373 | 0.086 *<sup>a</sup>* | 78.3% | 0.27 |
| Anthropic | frontier | `claude-opus-4-7` | 0.164 | 0.068 *<sup>a</sup>* | 63.1% | 0.79 |
| DeepSeek | budget | `deepseek-chat` | 0.040 *<sup>b</sup>* | 0.017 *<sup>b</sup>* | 38.3% | 2.65 |
| Anthropic | budget | `claude-haiku-4-5-20251001` | 0.032 *<sup>b</sup>* | 0.014 *<sup>b</sup>* | 53.7% | 1.74 |
| OpenAI | budget | `gpt-5-mini` | _re-running_ | _re-running_ | _re-running_ | _re-running_ |

`gemini-3.1-pro-preview` is on a 250 req/day cap (Google's preview-track policy applies regardless of paid tier) — too tight for a full gauntlet with retries. We use the GA `gemini-2.5-pro` for the published frontier-Google cell.

### Why some scores are extremely low

The geometric phase weight (`1 / 2 / 4 / 8` at ply boundaries 10 / 20 / 30) means roughly half of the achievable score lives in plies 20+. A model that forfeits before reaching mid-game loses access to that half in both the numerator AND has it weighing against them in the denominator. Combined with the `0.25^retries` cost and the exponential quality decay, scores below 0.10 are the natural floor for models that can't survive long.

***<sup>a</sup>*** **`0.068`–`0.086` (Opus PlayQuality, DeepSeek-reasoner PlayQuality)**: model produces legal moves on most first attempts (63–78%) and plays moves of reasonable quality (cp_loss 80–240 on the plies it reaches), but the games rarely survive into endgame at Stockfish skill 5. With only 3 retries permitted in PlayQuality mode (vs Reliability's 10), even occasional length-failures and run-of-bad-positions cause early forfeits. Result: the high-weight late plies are mostly missing from the numerator while still counting toward the denominator.

***<sup>b</sup>*** **`0.014`–`0.040` (Haiku, DeepSeek-chat — both metrics)**: first-attempt-legal rate of 38–54% means roughly half of all moves are illegal on the first try. Each of those moves pays the `0.25^retries` cost (1 retry → 25% credit, 2 retries → 6%). Combined with games averaging only 10–20 plies before forfeiting on illegal moves they can't recover from, the score is dominated by missing late-game plies and retry-cost-eroded mid-game plies. This isn't a metric pathology — it's the metric correctly reflecting "model cannot reliably play legal chess past the opening, even with the retry safety net."

For specifics on the Anthropic cells in particular, see [the Anthropic note below](#a-note-on-the-anthropic-scores).

---

## Reading the matrix

The first-attempt legal column and avg retries column are essential. Two models with identical Reliability scores can mean very different things — one might be picking legal moves with zero retries and being graded mainly on move quality; another might be needing two retries per move and being penalized for the cost of finding the legal move. The diagnostic columns separate these readings.

"First-attempt legal" means **zero retries used** — the model proposed a legal SAN on its very first attempt for that move, before any harness feedback. A model with 88% first-attempt legal is producing a legal move with no retry feedback on 88% of plies.

The matrix sorts into three behavioral bands:

**Band 1 — "Plays legal chess on first attempt":** Gemini cells, GPT-5, DeepSeek-reasoner. First-attempt legal rate 78-98%, avg retries 0.07-0.27 per move. The Reliability score is mostly about *move quality* on the legal moves played. The composite reflects what these models actually do at the board.

**Band 2 — "Needs the retry safety net":** Claude Opus 4.7. First-try legal 63%, avg 0.79 retries per move. Roughly one move in three is illegal on first try; the harness feeds back errors before the model corrects. Even after recovery the `0.25^retries` cost ensures retried moves contribute little.

**Band 3 — "Struggles to even propose legal moves":** Claude Haiku, DeepSeek-chat. First-try legal 38-54%, avg 1.7-2.7 retries per move. About half of all moves are illegal on first try, and the games rarely survive long enough to reach the high-weight late plies.

---

## Three findings

**1. Reasoning-tier supremacy doesn't hold for spatial state-tracking.** Across the four frontier reasoning models in the matrix, Reliability spans 0.16 to 0.53 — a 3× spread. The matrix-leader is a budget non-reasoning model. Reasoning-tier optimization helps when the reasoning fits in the budget AND when the model can apply that reasoning to spatial state — neither is guaranteed on this benchmark.

**2. The memorization cliff is universal and built into the score.** Every model that reaches mid-game shows the ACPL gradient: 25-130 cp opening → 55-241 cp middlegame → 50-130 cp endgame. Standard opening positions in our 20-position bank show 0% failure rate across every model tested. Mid-game and synthetic endgame positions show 33-67% failure rates on the harder examples. The geometric phase weight bakes this into the score: reaching ply 30 is worth 8× more than playing ply 5 well, so models that forfeit early lose the largest share of achievable score.

**3. Models form persistent wrong beliefs.** Across all retry-mode games, the same illegal SAN proposal recurs multiple times within a single game's plies. Opus proposed `Kxg4` five times across 11 plies of one game; Sonnet proposed `d7` three times to advance a pawn that doesn't exist. Each API call is independent and stateless from the model's side, but the same FEN deterministically reproduces the same wrong pattern-match. This is qualitatively different from typical "LLM hallucinations" — convergence on the same wrong belief across many invocations implies the belief is the model's deterministic response to a specific input, not a random error.

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

## The bottom line

The eval measures whether LLMs can maintain a 2D state of typed entities and correctly compute geometric queries against it across many reasoning steps. The current matrix top is 0.639 — a budget non-reasoning model from Google playing legal-on-first-try ~88% of the time and reaching mid/endgame with mediocre move quality. Frontier reasoning models from Anthropic, OpenAI, and DeepSeek either struggle on first-attempt legality (Anthropic) or score below the budget top on the composite anyway (OpenAI, DeepSeek). The bottom of the matrix is 0.032 — models that forfeit before reaching mid-game.

Chess gives a domain with deterministic ground truth, calibrated difficulty (memorized openings vs unique mid/endgames), and rules whose application is purely geometric. Models describe these rules verbally with 99% accuracy while applying them spatially at much lower rates. The benchmark exposes this gap with a small, cheap, reproducible test — a structural weakness shared by frontier and budget models alike on a generalizable cognitive dimension.

The most striking single qualitative finding is the **persistent wrong belief** pattern: models don't make random illegal moves; they form coherent-but-wrong pictures of the position and commit to them across multiple stateless API calls. This is qualitatively different from typical hallucinations and suggests fixing it requires architectural changes — not just bigger context windows or more training data.

The scoring is designed to be **hill-climbable**. The current matrix top (0.64) leaves real room above 0.9 where engine-quality play would land. The headroom is not noise — it's the gap between today's frontier and a model that can keep its mental picture of an 8×8 board accurate for 40 moves.

For methodology, formulas, scoring rationale, and reproduction recipe, see **[METHODOLOGY.md](METHODOLOGY.md)**.
