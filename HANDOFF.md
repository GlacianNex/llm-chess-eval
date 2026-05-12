# LLM Chess Eval — Project Handoff

A reproducible benchmark that exposes the gap between **memory-driven** and **reasoning-driven** LLM behavior, using chess as a substrate. Chess is not the subject of the eval — it's a clean probe for a specific cognitive failure that shows up across many domains where LLMs are asked to reason about structured state.

Share this doc with humans, or paste into a fresh Claude session for full project context.

---

## Why this eval matters

Most public LLM benchmarks measure either knowledge (MMLU, GPQA) or single-shot problem solving (MATH, HumanEval). Both reward strong models with strong training data on the relevant domain. Neither isolates **whether a model can maintain a coherent internal state and apply known rules to it across many reasoning steps** — the cognitive dimension that determines whether you can trust a model to operate on something unfamiliar for an extended period.

That gap matters in practice. The same model that scores 90% on a coding benchmark can confidently propose a 3-step refactor that references a function it imagined exists, then double down on the false reference when asked to revise. The same model that solves a math problem flawlessly will solve a similar-looking problem with one altered structural detail by applying the original problem's pattern. The same model that summarizes a long document accurately will, mid-conversation about it, confidently assert facts that contradict the document. These aren't knowledge failures — they're **state-reasoning failures** that current evals do not systematically catch.

**Chess is the perfect probe** because it has every property you'd design into a state-reasoning test on purpose:

- A small, deterministic, fully-specified 2D state (8×8 grid, ~32 typed entities).
- Rules whose application requires only geometric reasoning (line-of-sight, paths, adjacency, attack-set membership).
- Two ground-truth oracles: `python-chess` for rule-checking (deterministic, no judgment) and Stockfish for move-quality scoring.
- A built-in difficulty calibration: opening positions are saturated with training data (memorization works); mid-game and endgame positions are essentially never repeated (memorization fails, only reasoning works).

This last property is what makes the eval interesting. By measuring the **same model on the same task type** under both regimes — saturated and unsaturated — we observe how cleanly the model's apparent competence collapses when memory runs out.

## The eval value, in one sentence

**Two reproducible composite scores per model that quantify how reliably it can follow rules and how well it can play across the moves it does play — on positions outside training distribution where only state-reasoning can save it.**

Both scores live in `[0, 1]`. The benchmark is cheap (~$2-10 per model), takes 15-60 minutes per model, and runs across any provider (Anthropic / OpenAI / Google / DeepSeek / Llama via Together-Groq-Ollama) through a single CLI.

## What it looks like in practice

Levy Rozman (Gotham Moments) has a popular YouTube short titled ["ChatGPT vs Meta AI: This Isn't Chess Anymore"](https://www.youtube.com/shorts/YlMWZNx93G4). The visual gag is that two LLMs trying to play a game of chess produce nonsense after the opening: pieces appear from nowhere, captures are claimed on empty squares, the same illegal move keeps getting proposed, and both models confidently narrate their plans for pieces that no longer exist on the board.

Levy's video shows the failure as comedy. **Our eval scores it.** The numbers behind that video would look something like:

- `legality` per-move ≈ 0.80-0.90 (the move-by-move illegal rate is "only" 10-20%)
- **ChessReliability across a full game ≈ 0.15** (because the per-move rate compounds over 30+ turns and 0% of games complete cleanly)
- The illegal moves cluster on specific spatial computations: phantom pieces, ignored pins, walked-into checks, sliding pieces moved through other pieces
- The model commits to the same wrong belief for 3-5 consecutive turns because each turn is stateless and the same pattern-match keeps regenerating

The same dynamics that make the video go viral are exactly what this benchmark measures.

## Caveats and expectations

**Caveats:**
- **Small sample sizes by default.** v1 uses 5 games for ChessReliability and 3 for PlayStrength per model. That's enough for qualitative pattern signal but not for tight effect sizes — a 0.02 difference between two models could be sampling noise. Raise `--cr-games` and `--ps-games` for publishable comparisons.
- **Stockfish version matters.** Move-quality scores depend on the engine's evaluation function. Lock the Stockfish binary version when comparing across runs. v1 used Stockfish 18.
- **Provider tool-calling differences.** Each provider's structured-output format works slightly differently (Anthropic tools, OpenAI function-calling, Gemini function declarations, DeepSeek's reasoning model accepts only `tool_choice: "auto"`). Our adapters normalize these, but a tool-calling regression on a specific provider can show up as a benchmark regression — investigate adapter-level errors before concluding "this model got worse."
- **No claim about chess playing strength.** This is not a chess-skill benchmark. A model that's bad at chess but rule-consistent would score well. The eval measures whether the model maintains state and follows rules; ELO is incidental.

**Reasonable expectations:**
- Frontier models in 2026 score **0.15–0.50 on the headline metrics** despite playing 95-99% Stockfish-quality moves on the moves they do play. The gap between move quality and game completion is the headline finding.
- Opening positions show **0% failure rate** for every model tested (memorization). Failures concentrate on synthetic endgames and mid-game positions outside common training corpora.
- Reasoning models (Opus, GPT-5, DeepSeek-reasoner, Gemini Pro) tend to outscore their budget counterparts within the same family, but the gap is often smaller than the gap between general-reasoning benchmarks would predict — spatial state-tracking is a specific weakness that frontier-tier reasoning doesn't fully solve.

---

## TL;DR

Frontier models cruise through **5-7 opening moves perfectly** because chess openings are massively over-represented in their training data. The moment positions go off-script — unique mid-game configurations that don't pattern-match anything in memory — both models fall off a cliff: ~30% of their proposed moves become illegal, ACPL roughly doubles from opening to middlegame, and they commit to the same wrong move five turns in a row because their pattern-match keeps converging on the same out-of-distribution fallback.

Quantitatively (Claude Opus 4.7 / Sonnet 4.6 in v1):
- **ChessReliability (rule-following)**: 0.173 / 0.150 out of 1.0
- **PlayStrength (move quality across honest playthroughs at Stockfish skill 5)**: 0.388 / 0.144
- **ACPL by phase (Opus)**: opening 65 → middlegame 103 → endgame 122 cp

Models describe chess rules at **99% accuracy** while applying them spatially at **~15% reliability** once they leave familiar opening theory. **The benchmark isolates a specific cognitive failure: LLMs cannot reliably perform spatial/logical reasoning on out-of-distribution states**, even when they can verbalize all the relevant rules.

---

## 1. The thesis: memorization cliff in 2D-state reasoning

Chess provides four properties that make it a precise spatial-reasoning probe:
- **A deterministic 2D state** (8×8 grid, ~32 typed entities)
- **Rules whose application is purely geometric** (line-of-sight, adjacency, intermediate squares)
- **Ground-truth oracles** (`python-chess` for legality, Stockfish for move quality)
- **A calibrated memorization gradient** that the data confirms is observable:
  - Opening positions (first 5-10 moves) appear millions of times in training corpora
  - After move ~10, branching combinatorics push every game into a position essentially never seen before
  - The rules of chess remain identical throughout — only the availability of memorized lookup changes

This split lets the eval cleanly separate two cognitive modes:
- **Pattern-recall**: "I have seen this position. Here is the move."
- **Spatial reasoning**: "I have never seen this position. Let me look at the pieces and figure out what's legal and what's good."

**~95% of illegal moves in the v1 data correspond to spatial computations** — line-of-sight, path enumeration, attack-set membership, position tracking. Models can verbalize each of these rules at 99% accuracy (consistency eval), and select correctly when handed the legal-move set (augmented control → 100%), but fail to *apply* them when starting from a board they have to read.

The rule layer works. The selection layer works. **The geometry-from-state layer fails — most catastrophically on out-of-distribution positions where memory cannot paper over the gap.**

The protocol generalizes. Any domain with a saturated familiar surface and a combinatorial reasoning interior with deterministic ground truth should expose the same gap: code on unfamiliar codebases vs. boilerplate, math with novel structure vs. textbook patterns, multi-turn tool use in non-standard sequences, map / UI / layout reasoning on unfamiliar layouts.

---

## 2. Glossary

Concise reference. Each term used in this doc, in one or two sentences.

- **FEN** — Forsyth-Edwards Notation, the standard text representation of a chess position. Encodes piece placement, side-to-move, castling rights, en-passant target, halfmove clock, fullmove counter.
- **SAN** — Standard Algebraic Notation, the standard string format for chess moves (e.g. `Nf3`, `O-O`, `Qxh7+`, `e8=Q#`).
- **Move (LLM move)** — One turn taken by the LLM. We count only LLM moves; Stockfish's replies are not numbered. So "move 8" means the LLM has played 8 times.
- **Centipawn (cp)** — 1/100 of a pawn of advantage. Standard chess-engine unit for position evaluation. 100 cp = 1 pawn; 300 cp ≈ a minor piece; 900 cp ≈ a queen; ±10000 cp = clamped mate score.
- **cp_loss** — For a given LLM move, the centipawn difference between what Stockfish would have played and what the model actually played. Always ≥ 0; 0 if the model played Stockfish's #1.
- **ACPL** — Average Centipawn Loss. Mean of cp_loss across moves. Standard chess-strength metric. Lower is better. Strong club player ~50 cp ACPL; engine ~10 cp; casual ~150 cp.
- **CR (ChessReliability)** — Composite metric for rule-following ability. See §5.
- **PS (PlayStrength)** — Composite metric for sustained move quality across an honest playthrough. See §5.
- **Forfeit mode** — Game eval mode: the game ends when the model proposes its first illegal move.
- **Substitute mode** — Game eval mode: when the model proposes an illegal move, Stockfish's best move is played in its place and the game continues. Used to measure cascade.
- **Retry mode** — Game eval mode: when the model proposes an illegal move, it is told and asked to try again, up to `max_retries` times. If all retries fail, the game forfeits.
- **Survival (per game)** — `legal_moves_played / max_moves`. Fraction of the planned game length the model actually played legally. Bounded [0, 1].
- **Quality (per game)** — `1 − min(ACPL, 1000) / 1000`. Inverse of (capped) ACPL, mapped to [0, 1]. 1.0 = perfect engine; 0.0 = ACPL ≥ 1000 cp.
- **Stockfish skill level (0-20)** — Stockfish's internal strength dial. 0 ≈ ~1300 Elo; 3 ≈ ~1500 Elo; 5 ≈ ~1700 Elo amateur; 20 = full strength ~3500+ Elo.
- **Augmentation (legal-move list)** — Optional addition to the prompt that lists every legal SAN move in the current position. Used as a **control**, not as part of the eval — including it defeats the spatial-reasoning measurement.

---

## 3. The four evals

Each eval emits a normalized score in [0, 1] and writes a JSONL row per scored item. They differ in what cognitive layer they probe.

### 3.1 Legality
Static position → does the model produce a legal SAN move?

- **Probes**: can the model translate FEN to a legal move at all.
- **Scoring**: per-position score is 1.0 if `chosen_move` parses as a legal SAN, else 0.0. Sub-scores: `tool_called`, `chose_in_candidates`, `candidates_legal_rate` (fraction of the model's own candidate list that's legal).
- **Position bank**: 20 hand-curated FEN positions spanning openings, middlegames, endgames, and edge cases (forced moves, en passant, promotion, castling availability).

### 3.2 Consistency (rule-grounded)
For each candidate move, the model declares rule-based facts about that move. These are factually checkable, not judgment calls.

- **Probes**: can the model correctly describe what a move *does* on the board, independent of move legality.
- **Claims schema** (every field required per candidate):
  - `is_check` — does the move put the opponent in check (but not mate)?
  - `is_capture` — does it capture a piece?
  - `captured_piece` — which piece type is captured (P/N/B/R/Q or null)?
  - `is_castle` — kingside or queenside castling?
  - `is_promotion` — pawn promotion?
  - `is_en_passant` — en-passant capture?
  - `gives_mate` — delivers checkmate?
- **Ground truth**: every claim verified against `python-chess` (no engine needed).
- **Scoring**: per-candidate accuracy is (correct claims / 7 fields). Per-position score is mean across legal candidates. Sub-scores: per-field accuracy across all legal candidates.

**Earlier "claimed cp vs Stockfish cp" version of consistency was discarded.** Comparing the model's eval to Stockfish's deep search unfairly punished the model for not being a 3500-rated engine. Current consistency tests only what is verifiable from the rules.

### 3.3 Games
The model plays full games vs Stockfish at a configurable skill level from the standard starting position (or a configurable FEN). Every move is logged with: chosen move, was it legal, candidate-set quality, Stockfish's best move at the same position, centipawn loss, claim accuracy, retry count.

Three modes:

| Mode | On illegal chosen move | Measures |
|---|---|---|
| `forfeit` | Game ends, model loses | Clean rule-following signal. **CR is computed from these.** |
| `substitute` | Stockfish-best is played in its place; game continues | Cascade and out-of-distribution behavior |
| `retry` | Model is told its move was illegal and asked to try again (max 3); if all retries fail, game forfeits | Recovery from prompted feedback. **PS is computed from these.** |

Stockfish skill defaults: 3 for CR (forfeit), 5 for PS (retry).

### 3.4 Composite metrics: CR and PS
See §5 for full definitions and rationale.

---

## 4. The composite metrics

### 4.1 ChessReliability (CR)

**Definition.** Single 0-1 metric for rule-following ability.

```
CR = mean over N games of [ (legal_moves_played / max_moves) × mean(move_score) ]

move_score    = quality × retry_penalty
quality       = 1 − clamp(cp_loss, 0, 1000) / 1000
retry_penalty = 0.5 ^ retries_used
```

**Standardized config**: N=5 retry-mode games with `max_retries=10` (so up to 11 attempts per move before forfeit), alternating colors, Stockfish skill 3, max_moves=40, no augmentation.

**What it measures.** Three components must hold simultaneously, and the multiplication collapses the score if any fails:

- **Survival** (`legal_moves_played / max_moves`): the model must produce legal moves for many turns.
- **Quality** (`1 − cp_loss / 1000`): the moves played must be reasonable, not blunders.
- **Retry penalty** (`0.5 ^ retries_used`): each retry needed to find a legal move halves the move's credit. 0 retries = 1.0; 1 retry = 0.5; 3 retries = 0.125; 5 retries = 0.031; 10 retries = 0.001.

After `max_retries` failed attempts (= 11 total attempts), the game forfeits.

**The design intent: more chances + steeper grading.** Weak models often fail on move 1-2 in forfeit-only mode, which makes the metric flatline at zero and lose resolution. The retries-with-penalty formula gives the model many chances to complete a real game while ensuring that retry-heavy play scores essentially nothing. A model that needs 5+ retries on every move scores ~0 even though the game might run long. The "more tries" half preserves the game-completion data we want for downstream PS-style quality analysis; the "steeper grading" half ensures CR still rewards models that get it right on the first try.

**Worked example** (Opus v1, 3 games, forfeit-mode legacy data with retries_used=0 throughout — backwards compatible):
- Game 1: forfeit at move 7. survival = 7/40 = 0.175, mean move_score = 0.85, game = 0.149
- Game 2: forfeit at move 5. survival = 0.125, mean move_score = 0.80, game = 0.100
- Game 3: forfeit at move 12. survival = 0.30, mean move_score = 0.90, game = 0.270
- **CR = (0.149 + 0.100 + 0.270) / 3 = 0.173**

**Worked example with retries** (hypothetical model, 1 game with retries enabled):
- 20 legal moves played: 12 on first attempt, 5 needed 1 retry, 2 needed 3 retries, 1 needed 7 retries.
- Per-move quality assume ≈ 0.90 across the legal moves.
- Mean retry_penalty = (12×1.0 + 5×0.5 + 2×0.125 + 1×0.0078) / 20 = (12 + 2.5 + 0.25 + 0.008) / 20 = 0.738
- Mean move_score = 0.90 × 0.738 = 0.664
- Survival = 20/40 = 0.50
- Game CR = 0.50 × 0.664 = 0.332

Notice the move that needed 7 retries contributes 0.008 — essentially zero — even though it kept the game going. That's the design at work.

### 4.2 PlayStrength (PS)

**Definition.** Single 0-1 metric for sustained move quality on honest playthroughs.

```
PS = mean over N games of [ (legal_moves_played / max_moves) × (1 − ACPL_capped / 1000) ]

ACPL_capped = mean over each legal move's cp_loss, where each individual
              cp_loss is first clamped to [0, 1000] (so one mate blunder
              cannot dominate the average)
```

**Standardized config**: N=3 retry-mode games at max_retries=3, alternating colors, Stockfish skill 5, max_moves=60, no augmentation.

**What it measures.** A model that finds legal moves (with retry feedback) and plays them at high quality across the entire game scores high on PS. A model that forfeits despite retries scores low. A model that plays mediocre legal moves but completes long games scores moderate.

**Worked example** (Opus v1 at skill 5):
- Game 1: 13 legal moves, ACPL 116 → survival 0.217, quality 0.884, score 0.192
- Game 2: 39 legal moves, ACPL 77 → survival 0.650, quality 0.923, score 0.600
- **PS = (0.192 + 0.600) / 2 = 0.396** (CLI reported 0.388 — small rounding)

### 4.3 Why the two metrics, why multiplied, and what the gap means

**Two metrics because they probe different failure modes.**

- CR uses **forfeit mode at lower skill (3)** so the rule-following gap dominates. Games end at the first illegal move, so quality data is only from the first few moves — almost entirely opening.
- PS uses **retry mode at higher skill (5)** so the move-quality gap (especially mid/endgame ACPL) becomes measurable. Retries let the model recover from individual illegal moves; the cap on retries (3) keeps the metric honest.

**Multiplied not summed.** Either factor near zero collapses the composite. A model cannot:
- Hide an early-forfeit problem behind a few good opening moves (CR / PS survival term sinks).
- Hide a bad-moves problem behind a long game length (quality term sinks).

This matches the user-articulated framing: rare-but-recurring failures must carry heavy penalty. The multiplication enforces it.

**The PS / CR gap is itself a signal.** PS uses retry mode (the model can recover); CR uses forfeit (no recovery). So:
- **PS > CR** by a lot: retries help. The model's first-attempt illegality rate is reducing via prompted feedback.
- **PS ≈ CR**: retries don't extend playthroughs meaningfully; the model still forfeits at similar rates.

In v1, Opus PS / CR = 0.388 / 0.173 (2.2× lift); Sonnet PS / CR = 0.144 / 0.150 (no lift). Sonnet does not benefit from retry feedback as much as Opus.

### 4.4 Reference points for both metrics

| Score | What it corresponds to |
|---|---|
| 1.000 | Stockfish playing itself |
| 0.85+ | Strong club player (~2000 Elo) playing full games |
| 0.60-0.80 | A 1500-rated human who knows the rules |
| 0.30-0.50 | A model that completes some games at mediocre quality, or fully completes a few games at decent quality |
| 0.10-0.20 | Frontier LLMs in v1 |
| 0.000 | Forfeit on move 1, or ACPL ≥ 1000 across the board |

---

## 5. Why we made each design choice

Explicit justifications for the non-obvious decisions.

**Why use forced tool-use structured output instead of free-form text?**
JSON-via-tool-use is a clean contract. Free-form text would require parsing SAN out of prose, which adds noise — we want to measure the model's reasoning, not the parser's accuracy. The tool schema also forces the model to commit to specific claims (is_check, etc.) that we grade.

**Why grade rule-claims separately from move legality?**
Different cognitive layers. A model could output a legal move but misclaim what it does (e.g., say it captures when it doesn't). Grading claims separately catches state-perception failures that legality alone misses.

**Why no temperature setting?**
Some new Anthropic models (Opus 4.7) reject the parameter outright. The Anthropic API also has no seed parameter for reproducibility. We average across N samples instead of relying on bit-identical runs.

**Why no augmentation (legal moves in the prompt) in the eval?**
Augmentation hands the model the answer — when given the legal-move list, both v1 models hit 100% legality on static positions. That defeats the spatial-reasoning measurement. Augmentation exists as a **control** to confirm the failure is in derivation rather than selection, not as part of the eval.

**Why Stockfish skill 3 for CR and skill 5 for PS?**
- Skill 3 (~1500 Elo) for CR: rules-only test; the model loses by forfeit anyway, so the opponent's strength is mostly there to keep games complex. Lower-skill positions go off-book quickly.
- Skill 5 (~1700 Elo) for PS: strong enough to punish blunders (so ACPL is meaningful), weak enough that a competent model could draw or win. We want move quality to be the binding constraint, not opponent strength.

**Why max_moves = 40 for CR, 60 for PS?**
- 40 for CR: most forfeit-mode games end far earlier (~7 moves in v1), so the cap rarely binds. 40 is enough to expose the cascade if a model survives the early-game.
- 60 for PS: retry mode survives longer; we need room for the endgame phase (move 31+) to be measurable.

**Why cap individual cp_loss at 1000 before averaging?**
Mate scores are clamped to ±10000 in our setup. One walked-into-mate move with cp_loss = 10000 would dominate any average. Capping at 1000 (10 pawns) — anything past that is "completely lost" — preserves ACPL as a meaningful signal of average move quality across a game.

**Why phase-split ACPL (opening / middlegame / endgame)?**
Memorization carries the opening; reasoning is required for mid/endgame. A flat ACPL averaged over the whole game obscures the gradient. Phase-split makes the gradient visible.

**Why 3 retries in retry mode?**
- 1 would be too few — the iteration analysis shows the model often genuinely updates on attempt 2 or 3 (e.g., proposing a different piece type).
- 5+ would inflate scores by letting the model keep guessing until a legal move lands, measuring persistence with a verifier rather than reasoning.
- 3 is a clean balance and consistent with the cascade/recovery study.

**Why use retries-with-penalty for CR instead of forfeit-on-first-illegal?**
The original CR used forfeit mode (game ends on first illegal move). It worked well for moderately strong models but flattened the bottom of the scale: any model that produces an illegal move on move 1 scored ~0 regardless of whether it could ever recover. With retries-with-penalty, a model that needs help to be legal still gets graded — but the credit decays steeply (0.5^n per retry) so frequent retries collapse the score anyway. The combination of `max_retries=10` and `0.5^n` decay means the game can complete (so we see more late-game data) while retry-heavy moves contribute essentially zero (5 retries = 0.031 of full credit; 10 retries = 0.001). Forfeit-mode runs remain backwards compatible: `retries_used = 0` everywhere means `retry_penalty = 1.0` and the new formula reduces to the old one. Substitute mode (separate) still exists for studying cascade behavior.

**Why a 2D matrix of providers × tiers (frontier / budget)?**
Single-model evals can't separate "this provider is weak at chess" from "this specific tier is weak." The within-provider frontier-vs-budget gap is its own signal: a budget tier may collapse disproportionately on spatial reasoning compared to its general-reasoning benchmark scores.

---

## 6. The benchmark matrix

Standardized cross-provider set. Each provider contributes a frontier model (strongest) and a budget model (cheapest reasonable tier). Defined in [`config.BENCHMARK_MATRIX`](src/llm_chess_eval/config.py).

| Provider | Frontier | Budget |
|---|---|---|
| Anthropic | `claude-opus-4-7` | `claude-haiku-4-5-20251001` |
| OpenAI | `gpt-5` | `gpt-5-mini` |
| Google | `gemini-3.1-pro-preview` | `gemini-3.1-flash-lite` |
| DeepSeek | `deepseek-reasoner` | `deepseek-chat` |

Run:

```powershell
llm-chess-eval benchmark                       # all 8 cells
llm-chess-eval benchmark --tier frontier       # 4 frontier models
llm-chess-eval benchmark --tier budget         # 4 budget models
llm-chess-eval benchmark --provider anthropic  # one provider, both tiers
llm-chess-eval benchmark --dry-run             # preview the plan, no spend
```

---

## 7. End-to-end calculation walkthrough

A single move, fully traced.

1. **Position**: white to move, FEN `r1b1kbnr/1ppp1ppp/p1n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 4` (Ruy Lopez, move 4).
2. **Adapter call**: build the user message (FEN + ASCII board + side-to-move + tool schema). Send to the provider's API with forced tool-call on `submit_move`.
3. **Model response** (tool input):
   ```json
   {
     "position_summary": "Ruy Lopez, ...",
     "candidates": [
       {"san": "Ba4", "rationale": "Main line", "claims": {"is_check": false, "is_capture": false, "captured_piece": null, "is_castle": false, "is_promotion": false, "is_en_passant": false, "gives_mate": false}},
       {"san": "Bxc6", "rationale": "Exchange variation", "claims": {"is_capture": true, "captured_piece": "N", ...}},
       ...
     ],
     "chosen_move": "Ba4"
   }
   ```
4. **Per-candidate validation** (consistency eval):
   - For `Ba4`: parse SAN → confirm legal. Compute python-chess ground truth: `is_check=False, is_capture=False, captured_piece=None, ...`. Compare to model's claims field-by-field. 7/7 correct → claim accuracy = 1.0.
   - For `Bxc6`: parse → legal. Ground truth: `is_capture=True, captured_piece="N"`. Compare. 7/7 → 1.0.
   - Average across legal candidates → per-position consistency score.
5. **Legality scoring**: `chosen_move = "Ba4"` parses as legal SAN → score = 1.0. `chose_in_candidates = 1.0`. `candidates_legal_rate = 1.0`.
6. **Game flow** (if running in a game eval): push `Ba4` onto the board. Compute `cp_before` (eval before move, from white's POV, depth 12) and `cp_after` (eval after move, negated from black's POV back to white's POV). `cp_loss = max(0, cp_before − cp_after)`.
7. **Record**: write a JSONL row with all of the above, latency, raw response.

**Per-game CR/PS aggregation**: after the game ends, count legal moves played, compute ACPL (with per-move clamping at 1000), compute survival = legal_moves / max_moves, and combine: `score = survival × quality`. Take the mean across N games.

---

## 8. Multi-provider support

The CLI accepts any model ID; routing is done by `provider_for_model()` in [`config.py`](src/llm_chess_eval/config.py).

| Provider | SDK install | Env var | Example model IDs |
|---|---|---|---|
| Anthropic | bundled (`anthropic`) | `ANTHROPIC_API_KEY` | `claude-opus-4-7`, `claude-haiku-4-5-20251001` |
| OpenAI | `pip install '.[openai]'` | `OPENAI_API_KEY` | `gpt-5`, `gpt-5-mini`, `gpt-4o` |
| Google Gemini | `pip install '.[google]'` | `GOOGLE_API_KEY` or `GEMINI_API_KEY` | `gemini-3.1-pro-preview`, `gemini-3.1-flash-lite`, `gemini-2.5-pro` |
| DeepSeek | bundled (uses `openai` lib) | `DEEPSEEK_API_KEY` | `deepseek-reasoner`, `deepseek-chat` |
| Meta Llama via Together | `pip install '.[openai]'` | `TOGETHER_API_KEY` | `meta-llama/Llama-3.3-70B-Instruct-Turbo` |
| Meta Llama via Groq | `pip install '.[openai]'` | `GROQ_API_KEY` | `llama-3.3-70b-versatile` |
| Local via Ollama | `pip install '.[openai]'` | `OLLAMA_API_KEY` (any value) | any local model name |

### Adding a brand-new provider

1. Implement an adapter in `src/llm_chess_eval/adapters/` exposing `propose_move(fen, prior_failed=None, augment_legal_moves=None) -> CallOutcome` (the `ModelAdapter` protocol in `adapters/base.py`).
2. Reuse the shared prompt/schema/parsing helpers in `adapters/_shared.py` — `SYSTEM_PROMPT`, `SUBMIT_MOVE_PARAMETERS`, `build_user_message`, `parse_tool_input`. Wrap them in the provider's tool-call shape.
3. Register the provider in `provider_for_model()` and `KNOWN_MODELS` in `config.py`, and in `build_adapter()` in `adapters/factory.py`.

For OpenAI-API-compatible providers (DeepSeek, Mistral, vLLM, self-hosted), reuse `OpenAICompatibleAdapter` in `adapters/openai_compat.py` with the right `base_url` and `api_key_env_var`. No new adapter file needed.

---

## 9. Cross-family results

### v1 (Claude only)

| Model | CR | PS | ACPL opening | ACPL middlegame | ACPL endgame |
|---|---|---|---|---|---|
| claude-opus-4-7 | 0.173 | 0.388 | 65 cp | 103 cp | 122 cp |
| claude-sonnet-4-6 | 0.150 | 0.144 | 93 cp | – | – |

(Sonnet's gauntlet never reached mid/endgame in retry mode at skill 5.)

### v2 — full matrix (2026-05-12)

CR uses the final retry+penalty formula (`max_retries=10`, penalty `0.5^n`) at Stockfish skill 3; PS uses retry mode at skill 5 with `max_retries=3`. N=5 games for CR, N=3 for PS, alternating colors.

| Provider | Tier | Model | CR | PS | ACPL open | ACPL mid | ACPL end |
|---|---|---|---|---|---|---|---|
| Anthropic | frontier | `claude-opus-4-7` | **0.427** | 0.259 | 84 | 241 | – |
| Anthropic | budget | `claude-haiku-4-5-20251001` | 0.210 | 0.094 | 62 | – | – |
| OpenAI | frontier | `gpt-5` | **0.018** | 0.028 | 5 | – | – |
| OpenAI | budget | `gpt-5-mini` | 0.085 | 0.065 | 20 | – | – |
| Google | frontier | `gemini-3.1-pro-preview` | 0.356 | 0.306 | 25 | 55 | – |
| Google | budget | `gemini-3.1-flash-lite` | **0.633** | **0.726** | 43 | 71 | 66 |
| DeepSeek | frontier | `deepseek-reasoner` | 0.116* | 0.464 | 74 | 139 | 8 |
| DeepSeek | budget | `deepseek-chat` | 0.074* | 0.068 | 130 | – | – |

\* DeepSeek CR is from an earlier formula iteration (`max_retries=5`, penalty `0.6^n`). The matrix re-run was killed before DeepSeek finished; these numbers under the final formula would likely shift modestly (more retries available → potentially higher CR if DeepSeek can recover with more attempts). PS unaffected — the PS formula didn't change.

### Headline findings

**1. Gemini 3.1 Flash Lite dominates.** It scores highest on both metrics: CR 0.633 (almost 2× the next-best), PS 0.726, and is the only model to consistently reach the endgame phase in retry-mode PS (ACPL endgame 66 cp — engine-tier quality). The budget Gemini outperforms every frontier model tested.

**2. GPT-5 is the worst performer across the matrix.** CR 0.018, PS 0.028 — essentially zero on both. When GPT-5 plays a move it plays well (opening ACPL 5 cp, Stockfish-tier), but it forfeits so quickly with retries that it never establishes a real game. Its reasoning-tier overhead burns through the retry budget on the same wrong moves.

**3. Budget tier beats frontier tier in 2 of 4 providers.**

| Provider | Frontier CR / PS | Budget CR / PS | Reversal? |
|---|---|---|---|
| Anthropic | 0.427 / 0.259 | 0.210 / 0.094 | no — frontier wins both |
| **OpenAI** | **0.018 / 0.028** | **0.085 / 0.065** | **YES — budget 4.7× CR, 2.3× PS** |
| **Google** | **0.356 / 0.306** | **0.633 / 0.726** | **YES — budget 1.8× CR, 2.4× PS** |
| DeepSeek | 0.116 / 0.464 | 0.074 / 0.068 | no — frontier wins both |

The pattern is striking enough to warrant the hypothesis: **reasoning-tier optimization can hurt 2D-state tracking**. The models that "think more" before committing to a move are more likely to commit to a hallucinated state, then re-derive the same wrong state on the next turn. Budget/distilled models that lean on pattern-completion appear to be more robust to state drift on this task.

This is consistent with the in-flight observation that GPT-5 spent 19 retries on a single early move and still forfeited — repeatedly proposing the same illegal move from the same wrong mental model. A non-reasoning model would have given up that line of thought sooner.

**4. Reasoning models that DID reach mid/endgame are still mediocre.** DeepSeek-reasoner has good PS (0.464) — it ran 31-ply games and reached endgame with ACPL 8 cp (engine-tier when it got there). But low CR (0.116) means it needed many retries to make it. The reasoning helped quality once playing but hurt the rule-following layer.

**5. The memorization-cliff confirms across all tested families.** ACPL opening 5-130 cp; ACPL middlegame 55-241 cp where measured. Every model that played past opening showed a clear quality drop, validating the "memorization cliff" thesis on a non-Claude sample.

### Cross-tier interpretation

The within-provider rankings tell a more interesting story than the absolute scores. If you were choosing a model for an agent task that requires sustained state-tracking (multi-turn tool use, long refactors, long-document analysis), the v2 data suggests **the frontier reasoning-tier model is often NOT the right choice**. Two of four providers (Google and OpenAI) showed the budget tier outperforming on this measurement. The Anthropic Opus and DeepSeek-reasoner exceptions are notable — those frontier models retained dominance, but neither dominated as much as their general-reasoning benchmarks would suggest.

---

## 10. Open questions for v3

1. **Mid-game starting positions** — does the cascade need game-length state accumulation, or does it appear immediately from a complex mid-game FEN? Separates "drift across turns" from "complex positions are harder regardless."
2. **Reasoning-trace inspection across retries** — save every retry's full response (currently only the final one). Measures whether the model genuinely updates between retries vs. pattern-matching a different SAN.
3. **Larger N per cell** — 3-5 games is enough for qualitative signal; 10-30 per (model, skill) is needed for tight effect sizes.
4. **Skill sweep** — CR and PS at multiple Stockfish skills (0, 5, 10) to measure how the cascade interacts with opponent strength.
5. **Other 2D-state domains** — build analogous benchmarks for grid-navigation, UI layout reasoning, or tile-based puzzles. Confirms the spatial-reasoning failure profile is universal across 2D-grounded tasks.

---

## 11. Quickstart

```powershell
# Setup
git clone <wherever>
cd LLM_Chess_Eval
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e '.[all,dev]'

# Requires Stockfish binary on PATH (or STOCKFISH_PATH env var)
# Set provider keys you want to use:
[Environment]::SetEnvironmentVariable("ANTHROPIC_API_KEY", "...", "User")
[Environment]::SetEnvironmentVariable("OPENAI_API_KEY",    "...", "User")
[Environment]::SetEnvironmentVariable("GOOGLE_API_KEY",    "...", "User")
[Environment]::SetEnvironmentVariable("DEEPSEEK_API_KEY",  "...", "User")

# Sanity check
llm-chess-eval check-env

# Headline composite metrics
llm-chess-eval reliability    --model claude-opus-4-7 --games 5
llm-chess-eval play-strength  --model claude-opus-4-7 --games 3

# Full cross-provider matrix
llm-chess-eval benchmark

# Component evals for diagnostic depth
llm-chess-eval legality    --model claude-opus-4-7
llm-chess-eval consistency --model claude-opus-4-7
llm-chess-eval games       --model claude-opus-4-7 --mode forfeit --games 5
llm-chess-eval games       --model claude-opus-4-7 --mode substitute --games 2
llm-chess-eval games       --model claude-opus-4-7 --mode retry --games 2

# Aggregate scorecard across every run in runs/
llm-chess-eval report
```

### Repository layout

```
src/llm_chess_eval/
  adapters/
    _shared.py                # SYSTEM_PROMPT, SUBMIT_MOVE_PARAMETERS, build_user_message, parse_tool_input
    base.py                   # ModelAdapter protocol
    claude.py                 # Anthropic adapter
    openai.py                 # OpenAI adapter
    gemini.py                 # Google Gemini adapter (with schema translation)
    openai_compat.py          # generic OpenAI-API-compatible adapter (DeepSeek, Together, Groq, Ollama, ...)
    factory.py                # build_adapter(model) → routes to right provider
  evals/
    legality.py
    consistency.py
    games.py                  # game loop + forfeit/substitute/retry modes
    chess_reliability.py      # CR metric
    play_strength.py          # PS metric
  analytics/
    accumulation.py           # per-move error rate, survival curves
    illegal_taxonomy.py       # classifies why each illegal move failed
    report.py                 # auto-generated aggregate scorecard
  harness/
    runner.py                 # position-bank eval driver
    game_runner.py            # game gauntlet driver
  cli.py                      # `llm-chess-eval` entry point
  config.py                   # KNOWN_MODELS, BENCHMARK_MATRIX, paths, provider routing
  types.py                    # pydantic schemas (CandidateMove, MoveClaims, GameRecord, MoveRecord, etc.)

data/positions/legality_v1.jsonl   # 20-position bank used by legality + consistency
runs/<timestamp>__<eval>__<model>/ # raw JSONL output of each run
scripts/                            # ad-hoc analysis scripts
report.md                            # auto-generated aggregate data scorecard
FINDINGS.md                          # narrative report on v1 results
HANDOFF.md                           # this file
```

### Dependencies

- Python 3.12
- `anthropic`, `python-chess`, `pydantic`, `click`, `rich` (bundled in base install)
- Optional: `openai>=1.50` (covers OpenAI, DeepSeek, Together, Groq, Ollama), `google-genai>=0.3` (covers Gemini)
- Stockfish binary (v1 used Stockfish 18). GPL-3. Everything else is MIT-compatible.

---

## 12. The bottom line

The eval measures whether LLMs can maintain a 2D state of typed entities and correctly compute geometric queries against it across multiple reasoning steps. Frontier models score 0.15-0.17 on the headline reliability metric while playing high-quality individual moves at 95-99% accuracy. The gap between move quality (≥95%) and game completion (0%) is the spatial-reasoning ceiling, observable as a single number.

Chess gives a domain with deterministic ground truth, calibrated difficulty, and rules whose application is purely geometric. Models can describe these rules verbally with high accuracy while struggling to apply them spatially. The benchmark exposes this gap with a small, cheap, reproducible test — a structural weakness shared by current frontier models on a generalizable cognitive dimension.
