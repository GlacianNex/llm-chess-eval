# LLM Chess Eval — Methodology

How the benchmark measures what it measures: scoring formulas, design rationale, methodology constraints encountered, reproduction recipe, and code structure. For the matrix, findings, and deep dives, see **[RESULTS.md](RESULTS.md)**.

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

Current matrix range: **0.032 (Claude Haiku) to 0.639 (Gemini 3.1 Flash Lite)**. See [RESULTS.md](RESULTS.md) for the full matrix.

### Average Centipawn Loss (ACPL) — diagnostic

ACPL is the standard chess-strength metric: per move, the centipawn difference between Stockfish's top move and what the model played. ACPL 50 = strong club player; ACPL 150 = intermediate; ACPL 500+ = blundering. We report ACPL **by phase** (opening / middlegame / endgame) alongside the composite scores. The ACPL gradient across phases is the most direct evidence of the memorization cliff.

---

## Methodology constraints encountered

This section is the story of a real bug we hit, what it taught us, and the fix that's now in the code. It's left intact because the underlying token-budget interaction is something anyone evaluating reasoning models with structured output (tool calling, JSON mode) is likely to hit themselves.

### The bug: factory-level `max_tokens=2048` default silently capped reasoning models

The benchmark's adapter factory had a `max_tokens=2048` default parameter. The CLI commands all called `build_adapter(model=model)` without overriding it, so every adapter was instantiated with `max_tokens=2048` — regardless of each individual adapter's own (higher) default. The individual-adapter defaults were dead code; the factory's 2048 won every call.

For Anthropic this was fine — Anthropic's `max_tokens` counts *output tokens only*, with thinking on a separate budget. 2048 is plenty for a chess tool call's output.

For OpenAI and Gemini reasoning models, this was catastrophic. OpenAI's `max_completion_tokens` and Google's `max_output_tokens` cap the total tokens a reasoning model can produce, **including internal reasoning**. At 2048 total, the model spent the entire budget reasoning, had zero tokens left for the visible output, and returned `finish_reason='length'` with no tool call.

We caught this only after publishing a first matrix in which GPT-5 scored 0.033 — the worst cell. An audit of the GPT-5 run showed **162 of 166 first-attempt failures (97.6%) were token-budget exhaustion, not chess failures.** The model wasn't proposing illegal moves; it was running out of room to emit a tool call at all. With the bug fixed (`max_tokens=65536`), GPT-5 jumped to **0.41 with 97.8% first-attempt-legal rate** — the highest in the matrix on that diagnostic. The first published numbers were wrong by an order of magnitude on the affected cells.

The lesson worth keeping: if you're evaluating reasoning models with forced structured output, **the response-token budget is on the critical path and easy to misconfigure.** The individual-adapter defaults in this repo were set correctly. The factory wired them away. A trace of the actual call path at any point would have caught it; checking just the adapter constructors did not.

### Gemini exposes a related but distinct failure mode

Gemini Pro and 2.5 Pro have a second failure mode where reasoning runs long enough to corrupt the structured tool output rather than hit a length cap — the API reports `finish_reason='MALFORMED_FUNCTION_CALL'` instead of `'length'`. Different proximate cause, same root (reasoning overrun corrupts tool emission). The fallback ladder below triggers on both.

### The fix (now canonical)

Four layers, in order of necessity:

1. **Generous `max_tokens` ceiling (65536) for reasoning-capable providers.** Well below OpenAI's 128K output limit, but far above the actual reasoning-token usage on >99% of calls. Closes the bug for typical positions. **The factory now defaults to 65536**, not 2048.

2. **Per-provider ceiling for Anthropic at 8192.** Anthropic's SDK has a built-in guard that rejects calls with very high `max_tokens` unless the call is streamed (which the adapter doesn't currently support). 8192 is plenty for a chess tool call's output and stays under the streaming threshold.

3. **`previously_failed_sans` list capped to last 3 attempts.** Each retry passes the model the list of previous failed SAN attempts. After 5-10 retries the prompt grows long and reasoning expands to re-analyze everything. Capping to 3 keeps prompt size bounded without losing the "don't repeat these" signal.

4. **Gradual `reasoning_effort` fallback ladder on overrun failures.** If a call hits `finish_reason='length'` OR `MALFORMED_FUNCTION_CALL` despite the above, the next retry on that move drops reasoning effort one notch: `default → medium → low → minimal`. Adapters translate the cross-provider effort ladder into provider-specific budgets (`reasoning_effort` on OpenAI, `thinking_config.thinking_budget` on Gemini, no-op on Anthropic where output and thinking are budgeted separately). This activates *only after* the model demonstrates it can't fit at the current effort — the benchmark does NOT a-priori set low reasoning effort. A-priori reduction would handicap the model and produce data that under-measures its capability.

The combination makes the benchmark robust: most calls land at full reasoning effort with no token issues; pathological positions get a graceful fallback; and the data is honest about which calls used reduced reasoning (the `reasoning_effort_for_this_attempt` is logged per call).

### What the matrix still exposes about the reasoning-budget battle

Even with the corrected `max_tokens=65536`, the response-budget interaction with chess positions is real. GPT-5 produces a successful first-attempt tool call on 97.8% of plies — the highest in the matrix, but not 100%. The remaining 2.2% drives the avg-retries-per-move of 0.15 and a meaningful share of the composite-score deficit relative to the matrix top. The bug is fixed; the underlying tension between reasoning depth and response-token cap is not.

### How to detect this on your own provider

For anyone implementing or extending this benchmark on a new provider, the canary is in the run JSONLs:

```
1. Grep run JSONLs for "did not call submit_move".
   Any hits = a likely token-budget interaction.
2. Inspect the finish_reason in the error message.
   - "length" → bump max_tokens (start at 65536)
   - "MALFORMED_FUNCTION_CALL" → same; the stepdown ladder catches both
   - "content_filter" → safety refusal (different problem)
   - "stop" with no tool → model declined (likely tool_choice issue)
3. If still failing at 65536: the stepdown ladder will gracefully degrade,
   and the per-attempt reasoning_effort is logged so you can inspect.
4. If the underlying CLI call doesn't override max_tokens, check the
   factory default — that's where this bug lived.
```

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

For the matrix, findings, deep dives, and the Anthropic-score explanation, see **[RESULTS.md](RESULTS.md)**.
