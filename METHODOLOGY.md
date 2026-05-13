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

The model plays games vs Stockfish at skill 3 — approximately **1500 ELO, an intermediate amateur**. On an illegal proposed move, the model is told "that was illegal" and asked to try again — up to 10 retries per move. If all retries fail, the game forfeits.

The opponent is amateur-tier deliberately. Reliability measures rule-following, not chess strength against engines — a stronger opponent would just force the model into harder positions faster without telling us anything new about whether it can keep its mental picture of the board accurate.

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

The model plays games vs Stockfish at skill 5 — approximately **1700 ELO, a intermediate amateur**. This is deliberate: the benchmark is not comparing LLMs to a chess engine. Stockfish at skill 5 plays at roughly the level of a competent club player, the kind of opponent that creates real but tractable positions an LLM with genuine chess understanding should be able to navigate. PlayQuality scores are *not* "performance vs engine" — they're "move quality across full games when the opponent is amateur-tier." Retry mode max 3 retries, no per-retry penalty. PlayQuality measures **how strong the model's moves are once it has found a legal one** — it doesn't penalize retry use.

(For context on the full Stockfish skill range: skill 0 ≈ 1100 ELO beginner; skill 3 ≈ 1500 amateur — what Reliability uses; skill 5 ≈ 1700 intermediate amateur — what PlayQuality uses; skill 15 ≈ 2500 strong engine; skill 20 ≈ 2850 top engine. None of the current benchmark cells play against an engine-strength opponent.)

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

### Reliability is NOT first-attempt-legal — they measure different things

The benchmark reports both **`first_attempt_legal_rate`** (a diagnostic — fraction of plies where the model's very first proposal was a legal SAN) and **`ChessReliability`** (the composite score). It's easy to assume "high first-attempt legality → high Reliability", but the composite has three factors and first-attempt-legality is only one of them. The other two can drag the score substantially even when legality is near-perfect:

- **`move_quality(cp_loss)`**: a legal move with cp_loss = 86 has quality = exp(−86/150) = 0.564, not 1.0. Reaching engine-level play (cp_loss ~5) requires move_quality ~0.97. Most LLMs play moves with cp_loss in the 50-200 range — competent club-player territory, scoring 0.27-0.72 per move on quality alone.
- **`game_phase_weight(ply)`**: ply-30+ moves are weighted 8×, ply-1-9 are weighted 1×. A model that forfeits early or stops at the max-ply cap before reaching late game loses access to the highest-weighted plies in the numerator while they remain in the denominator. The structural ceiling for a model that perfectly plays only to ply 20 (out of max 40) is ~0.18; perfectly to ply 30 is ~0.44; perfectly to ply 40 is 1.0.
- **A single full forfeit** (game ends at ply 1 because the model can't produce a legal move even after all retries) drops that game's contribution to 0 regardless of how the other games went. In a 5-game gauntlet, one forfeit at ply 1 subtracts ~0.20 from the mean.

A concrete example from the published matrix: **GPT-5 has 97.8% first-attempt-legal but ChessReliability = 0.410.** Decomposing:

| Factor | Contribution |
|---|---|
| Mean cp_loss across legal moves: 86 → quality = 0.564 | Caps the score around 0.56 even if every move were legal |
| One game forfeited at ply 1 (1 of 5) | Subtracts ~0.20 from mean |
| Remaining 4 games averaged 27 plies played (vs max 40) | Misses the highest-weighted late plies; ~0.39 structural ceiling for play-to-ply-27 |
| Retry cost (0.01 retries/move) | Negligible |

The 0.41 composite is the product of "near-perfect legality × mid-range quality × partial late-game coverage." Read the columns together — first-attempt-legal alone is not a stand-in for Reliability.

### Reference points

| Score | Equivalent |
|---|---|
| 1.000 | Theoretical max — engine-quality moves across all 40 plies, zero retries |
| 0.85-0.95 | Hypothetical: a model playing at engine-equivalent move quality |
| 0.55-0.75 | Best current LLMs on this benchmark (vs amateur-tier Stockfish opponent) |
| 0.30-0.50 | LLMs that survive into mid-game but with mediocre quality |
| 0.10-0.20 | LLMs that struggle past opening |
| 0.00-0.05 | Forfeit before mid-game, or heavy retry dependence |

The 1.0 reference is the theoretical maximum of the metric (engine-quality moves with no retries needed across a full game) — *not* a benchmark cell. The benchmark plays models against amateur-tier opponents (skill 3 / skill 5); the engine-quality calibration is the scoring anchor, not a comparison target.

Current matrix range: **0.032 (Claude Haiku) to 0.639 (Gemini 3.1 Flash Lite)**. See [RESULTS.md](RESULTS.md) for the full matrix.

### Average Centipawn Loss (ACPL) — diagnostic

ACPL is the standard chess-strength metric: per move, the centipawn difference between Stockfish's top move and what the model played. ACPL 50 = strong club player; ACPL 150 = intermediate; ACPL 500+ = blundering. We report ACPL **by phase** (opening / middlegame / endgame) alongside the composite scores. The ACPL gradient across phases is the most direct evidence of the memorization cliff.

---

## Caveats and expectations

**Caveats:**

- **Sample sizes are small by default** (5 games for Reliability, 3 for PlayQuality). Enough for qualitative pattern signal but not for tight effect sizes. A 0.02 difference between two models could be sampling noise. Raise `--games` for publishable comparisons.
- **Stockfish version matters.** Move-quality scores depend on the engine's evaluation function. Lock the binary version when comparing across runs. This work used Stockfish 18.
- **Provider tool-calling differences.** Each provider's structured-output format works slightly differently. The adapters normalize these; a regression on a specific provider can show up as a benchmark regression. Always grep the run JSONL for `"did not call submit_move"` before drawing conclusions.
- **This is not a chess-skill benchmark.** A model that's bad at chess but rule-consistent would score well. The eval measures state-tracking and rule-following; ELO is incidental.

**Reasonable expectations:**

- LLMs on this benchmark span 0.03 to 0.64 on Reliability. The top is a budget non-reasoning model; the bottom is two budget models and one frontier model. Reasoning-tier optimization is neither necessary nor sufficient for high scores.
- The first-attempt legal rate column is the most diagnostic single number — it measures the fraction of plies where the model's very first proposal (zero retries) was legal. Reads independently of how forgiving the harness is. Read it alongside the composite.
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

## Things to be aware of

A few provider-specific behaviors to keep in mind when running this benchmark on reasoning-tier models. These aren't findings — just things that affect how the harness interacts with the APIs.

**`max_tokens` semantics differ by provider.** OpenAI's `max_completion_tokens` and Google's `max_output_tokens` cap reasoning + visible output combined. A reasoning model can spend its entire budget reasoning and have zero tokens left for the tool call, returning `finish_reason='length'`. Anthropic's `max_tokens` counts output only; thinking is on a separate budget. The benchmark defaults to 65536 for reasoning-capable providers (8192 for Anthropic, since the SDK refuses non-streaming calls above that). Set the ceiling lower at your own risk; audit run JSONLs for `did not call submit_move` entries to spot budget issues.

**Gemini has a second tool-output failure: `MALFORMED_FUNCTION_CALL`.** When reasoning runs long enough to corrupt the structured output without quite hitting the length cap, Gemini reports this finish_reason instead. The harness's reasoning-effort fallback ladder triggers on both — when an attempt hits either failure mode, the next retry on that move drops effort one notch (`default → medium → low → minimal`). The fallback activates only after the model demonstrates it can't fit at the current effort.

**Gemini Pro Preview has a 250-request-per-day cap** regardless of paid tier. Only GA models (e.g., `gemini-2.5-pro`) have unrestricted per-tier quotas. A Reliability + PlayQuality gauntlet on a reasoning-heavy model with retries can burn 300-400 requests, so the cap is binding. The benchmark uses `gemini-2.5-pro` for the published frontier-Google cell.

**Per-call `reasoning_effort_for_this_attempt` is logged** in `progress.jsonl` so you can audit which calls used reduced reasoning vs the model's default.

---

For the matrix, findings, deep dives, and the Anthropic-score explanation, see **[RESULTS.md](RESULTS.md)**.
