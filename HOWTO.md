# LLM Chess Eval — How to Run

Operational guide: install, configure providers, run the benchmark, add new providers, and provider-specific quirks to watch for.

For what the benchmark measures and why, see **[METHODOLOGY.md](METHODOLOGY.md)**. For the matrix and findings, see **[RESULTS.md](RESULTS.md)**.

---

## Install

```powershell
git clone https://github.com/GlacianNex/llm-chess-eval.git
cd llm-chess-eval
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e '.[all,dev]'
```

Stockfish must be on `PATH` or `STOCKFISH_PATH` must point at the binary. **Use Stockfish 18** for reproducibility — move-quality scores depend on the engine's evaluation function and lock to a specific binary version.

## Provider keys

Set whichever providers you want to evaluate:

```powershell
$env:ANTHROPIC_API_KEY = "sk-ant-..."
$env:OPENAI_API_KEY    = "sk-..."
$env:GOOGLE_API_KEY    = "AIza..."
$env:DEEPSEEK_API_KEY  = "sk-..."
```

Sanity check the environment:

```powershell
llm-chess-eval check-env
```

## Run

```powershell
# Single-model composite metrics
llm-chess-eval play-strength --model claude-opus-4-7 --games 5
llm-chess-eval play-quality  --model claude-opus-4-7 --games 3

# Full cross-provider matrix
llm-chess-eval benchmark --dry-run    # preview cells without invoking models
llm-chess-eval benchmark

# Component evals for diagnostic depth (per-position legality and rule-consistency)
llm-chess-eval legality    --model claude-opus-4-7
llm-chess-eval consistency --model claude-opus-4-7

# Re-score existing runs with the canonical formula (no model invocations)
python scripts/matrix_with_retry_context.py
```

Output is written to `runs/<timestamp>__<eval>__<model>/` as JSONL — one record per game (or per position for legality/consistency). Re-scoring scripts read this raw data and compute composite metrics; you can re-score historical runs whenever scoring formulas change without spending API again.

## Repository layout

```
src/llm_chess_eval/
  adapters/
    _shared.py                # SYSTEM_PROMPT, SUBMIT_MOVE_PARAMETERS, build_user_message
    base.py                   # ModelAdapter protocol
    claude.py / openai.py / gemini.py / openai_compat.py / factory.py
  evals/
    legality.py / consistency.py
    games.py                  # game loop (forfeit/substitute/retry) + per-move progress logging
    play_strength.py          # PlayStrength metric — canonical scoring (primary)
    play_quality.py           # PlayQuality metric — canonical scoring (supplemental)
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

## Adding a new provider

1. Implement an adapter in `src/llm_chess_eval/adapters/` exposing `propose_move(fen, prior_failed=None, augment_legal_moves=None, reasoning_effort_override=None) -> CallOutcome` (the `ModelAdapter` protocol in `adapters/base.py`).
2. Reuse the shared prompt/schema/parsing helpers in `_shared.py`. Wrap them in the provider's tool-call shape.
3. Register the provider in `provider_for_model()` and `KNOWN_MODELS` in `config.py`, and in `build_adapter()` in `adapters/factory.py`.

For OpenAI-API-compatible endpoints (Together, Groq, Ollama, vLLM, custom servers), reuse `OpenAICompatibleAdapter` in `adapters/openai_compat.py` with the right `base_url` and `api_key_env_var` — no new adapter file needed.

---

## Provider quirks to know about

These aren't findings about the models — just operational behaviors that affect how the harness interacts with the APIs. Worth knowing if you're running the eval or interpreting the JSONL output.

**`max_tokens` semantics differ by provider.** OpenAI's `max_completion_tokens` and Google's `max_output_tokens` cap reasoning + visible output combined. A reasoning model can spend its entire budget reasoning and have zero tokens left for the tool call, returning `finish_reason='length'`. Anthropic's `max_tokens` counts output only; thinking is on a separate budget. The benchmark defaults to 65536 for reasoning-capable providers (8192 for Anthropic, since the SDK refuses non-streaming calls above that). Set the ceiling lower at your own risk; audit run JSONLs for `"did not call submit_move"` entries to spot budget issues.

**Gemini has a second tool-output failure mode: `MALFORMED_FUNCTION_CALL`.** When reasoning runs long enough to corrupt the structured output without quite hitting the length cap, Gemini reports this `finish_reason` instead. The harness's reasoning-effort fallback ladder triggers on both — when an attempt hits either failure mode, the next retry on that move drops effort one notch (`default → medium → low → minimal`). The fallback activates only after the model demonstrates it can't fit at the current effort.

**Gemini Pro Preview has a 250-request-per-day cap** regardless of paid tier. Only GA models (e.g., `gemini-2.5-pro`) have unrestricted per-tier quotas. A PlayStrength + PlayQuality gauntlet on a reasoning-heavy model with retries can burn 300–400 requests, so the cap is binding. The benchmark uses `gemini-2.5-pro` for the published frontier-Google cell.

**Per-call `reasoning_effort_for_this_attempt` is logged** in `progress.jsonl` so you can audit which calls used reduced reasoning vs the model's default.

**Quota-corrupted games are dropped at aggregation time.** If a provider returns 429 / quota errors mid-game, the harness keeps writing what it has, but `scripts/aggregate_chunks.py` filters out any game that contains a quota error before scoring — so a billing failure mid-run doesn't contaminate the matrix. Drop counts are reported alongside N per cell in the aggregated output.

---

## Tips for interpreting runs

- **Always grep run JSONLs for `"did not call submit_move"`** before drawing conclusions. That string indicates the model's tool call failed structurally (length, malformed, etc.) and the harness fell through to a default — it's not a real chess failure, it's a provider-protocol failure.
- **Forfeit rate is the cleanest single diagnostic.** Compare across models without needing to debate scoring choices. See [METHODOLOGY § When a game forfeits](METHODOLOGY.md#when-a-game-forfeits).
- **First-attempt-legal rate is the second-cleanest.** Measures rule-following independent of retry budget.
- **ACPL by phase is the cliff signal.** Rising ACPL across opening → middlegame → endgame is the cumulative-coherence failure mode the benchmark is designed to expose.
