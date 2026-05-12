# LLM Chess Eval

A reproducible benchmark that measures whether LLMs can maintain coherent internal state and apply rules across many reasoning turns, using chess as a substrate. The cognitive failure it isolates — **state reconstruction and 2D spatial reasoning on out-of-distribution positions** — shows up in many domains. Chess is the cleanest substrate to expose it.

This document is the full reference: what the benchmark measures, how, what it found, and how to reproduce.

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

Two composite scores per model, both bounded `[0, 1]`:

### ChessReliability (CR)

```
CR = mean over N games of [ (legal_moves_played / max_moves) × mean(quality × retry_penalty) ]
quality        = 1 − clamp(cp_loss, 0, 1000) / 1000
retry_penalty  = 0.5 ^ retries_used
```

The model plays games vs Stockfish at skill 3. On an illegal proposed move, the model is told "that was illegal" and asked to try again — up to 10 retries per move. Each retry costs half of the move's credit (`0.5^n`). If all retries fail, the game forfeits and that move scores zero.

CR rewards **producing legal play cheaply.** A model that finds the right move on attempt 1 every turn scores high. A model that needs many retries every turn — even if it eventually finds legal moves — scores near zero because the penalty compounds. Standard config: N=5 games, max 40 LLM moves per game.

### PlayStrength (PS)

```
PS = mean over N games of [ (legal_moves_played / max_moves) × (1 − ACPL_capped / 1000) ]
ACPL_capped = mean over each move's cp_loss, each clamped to [0, 1000]
```

The model plays games vs Stockfish at skill 5 in retry mode (max 3 retries per move, no per-retry penalty). PS measures the **quality of moves played**, not the model's ability to find legal ones. Standard config: N=3 games, max 60 LLM moves per game.

### Why two metrics multiplied (not summed)

Both metrics use the same multiplicative structure: `survival × quality`. A model that forfeits early can't rescue the score with a few good opening moves (survival sinks). A model that drags out long games of mediocre moves can't rescue the score with completion (quality sinks). Both factors have to hold.

### Reference points

| Score | Equivalent |
|---|---|
| 1.000 | Stockfish playing itself |
| 0.85+ | Strong club player completing full games (~2000 Elo) |
| 0.60-0.80 | ~1500-rated human who knows the rules |
| 0.30-0.50 | Some games complete at mediocre quality, or short games at good quality |
| 0.10-0.20 | Most frontier LLMs |
| 0.000 | Forfeits on move 1, or ACPL ≥ 1000 throughout |

### ACPL — the third dimension

ACPL (Average Centipawn Loss) is the standard chess-strength metric: per move, the centipawn difference between Stockfish's best move and what the model played. ACPL 50 = strong club player; ACPL 150 = beginner-intermediate; ACPL 500+ = blundering.

Reporting ACPL **by phase** (opening / middlegame / endgame) is the most direct evidence of the memorization cliff. Every model that reaches mid-game shows the gradient.

---

## What it looks like in practice

[Levy Rozman's short "ChatGPT vs Meta AI: This Isn't Chess Anymore"](https://www.youtube.com/shorts/YlMWZNx93G4) shows the failure as comedy. Two LLMs trying to play a game produce nonsense after the opening: pieces appear from nowhere, captures are claimed on empty squares, the same illegal move keeps getting proposed, and both models confidently narrate their plans for pieces that no longer exist on the board.

The benchmark scores it. Numerically, the failure modes in that video look like:

- Per-move legality on static positions ≈ 80-90% — the move-by-move illegal rate is "only" 10-20%, easy to miss on isolated questions.
- ChessReliability across a full game ≈ 0.15-0.4 — because per-move rates compound across 30+ turns, almost no game completes cleanly.
- Illegal moves cluster on specific spatial computations: phantom pieces (the piece type exists somewhere but not where claimed), missed long-diagonal checks, pinned pieces moved anyway, sliding pieces moved through other pieces.
- The model commits to the same wrong belief for 3-5 consecutive turns because each turn is stateless and the same pattern-match keeps regenerating the same false geometry from the same FEN input.

The numbers below quantify what the video shows.

---

## Results

### The cross-family matrix

Eight cells — frontier and budget tier for each of four providers. Standardized config: N=5 CR games at Stockfish skill 3 with `max_retries=10` and `0.5^n` retry penalty; N=3 PS games at skill 5 with `max_retries=3`; both at `max_tokens=65536`, alternating colors.

| Provider | Tier | Model | CR | PS | ACPL open | ACPL mid | ACPL end |
|---|---|---|---|---|---|---|---|
| Anthropic | frontier | `claude-opus-4-7` | **0.427** | 0.259 | 84 | 241 | – |
| Anthropic | budget | `claude-haiku-4-5-20251001` | 0.210 | 0.094 | 62 | – | – |
| OpenAI | frontier | `gpt-5` | _re-running_ | _re-running_ | _re-running_ | _re-running_ | _re-running_ |
| OpenAI | budget | `gpt-5-mini` | _re-running_ | _re-running_ | _re-running_ | _re-running_ | _re-running_ |
| Google | frontier | `gemini-3.1-pro-preview` | _re-running_ | _re-running_ | _re-running_ | _re-running_ | _re-running_ |
| Google | budget | `gemini-3.1-flash-lite` | **0.633** | **0.726** | 43 | 71 | 66 |
| DeepSeek | frontier | `deepseek-reasoner` | _re-running_ | _re-running_ | _re-running_ | _re-running_ | _re-running_ |
| DeepSeek | budget | `deepseek-chat` | _re-running_ | _re-running_ | _re-running_ | _re-running_ | _re-running_ |

Six cells are pending re-run after a methodology bug was diagnosed and fixed (see "Methodology gotchas" below). Each cell will be filled in once the corresponding run lands.

### What the data shows so far

The clean cells (Anthropic + Gemini Flash Lite) already establish the headline pattern:

**1. Frontier-tier scores are not impressive.** The strongest reasoning-tier model in the clean data — Claude Opus 4.7 — scores 0.427 on rule-following and 0.259 on play strength. That's well below a ~1500-rated human. A budget-tier model with no extended thinking (Gemini 3.1 Flash Lite) outscores Opus on both metrics: CR 0.633, PS 0.726. The Flash Lite is the only model in the data to consistently reach the endgame phase under retry-mode play, with endgame ACPL of 66 cp — engine-tier quality when it plays.

**2. The memorization-cliff is universal.** Every model that reaches mid-game shows the ACPL gradient. Opus: 84 cp opening → 241 cp middlegame. Flash Lite: 43 → 71 → 66 cp. The first few moves are pattern-matched; the moment positions go off-script, move quality degrades sharply.

**3. The within-provider frontier-vs-budget comparison is the most interesting axis.** Anthropic's frontier beats its budget tier on both metrics — the expected pattern. Whether OpenAI and Google show the same direction or reverse it is the question the remaining cells will answer.

---

## Failure modes observed

The composite scores are the headline; the failures themselves are where the eval becomes interesting. Across 224 illegal moves classified across the data we have, **~95% of failures are spatial-reasoning errors**:

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

### The persistent wrong belief pattern

The most striking single qualitative finding: models don't make random illegal moves. They form **coherent-but-wrong mental models of the position and commit to them across consecutive moves.** Each API call is stateless from the model's side, but the same wrong geometry regenerates on each turn because the same input prompts the same pattern-match collapse.

Three documented examples from real games:

**Same illegal capture proposed 5 times in 11 plies (Opus):**
> ply 25: rationale "grabs the pawn on g4" (illegal: would put king in check)
> ply 32: "captures the g4 pawn for free"
> ply 33: "wins the g4 pawn"
> ply 35: "to grab material"
> ply 36: "capture the g4 pawn to remove threat"

White's king cannot capture the pawn on g4 — doing so exposes it to check. Opus regenerates this same wrong move five times. Stockfish-best substituted around it on every preceding attempt, but the model has no memory of failure across stateless calls.

**Pawn advance proposed 3 times for a pawn that doesn't exist (Sonnet):**
> "Advancing the passed pawn to d7 attacks the black rook..."
> "Advancing d6 pawn to d7 puts tremendous pressure..."
> "Advancing the d6 pawn to d7 puts tremendous pressure..."

There is no white pawn that can advance to d7 in this position. The model believes in a pawn configuration that doesn't exist, and the wrong belief is encoded in its response to that specific FEN.

This is qualitatively different from typical "LLM hallucinations." A standard hallucination is a one-shot plausible-sounding wrong fact. The pattern here is **convergence on the same specific wrong belief across many independent invocations** — implying the belief is the model's deterministic response to a specific input rather than random error.

### Retry feedback: helps for some failures, not others

When a move is illegal, the next attempt is told "your move X was illegal, try again" and given the list of recent failed attempts. Across one game's 22 retries, 17/22 iterations went to a different piece type (model genuinely updated). The remaining 5 went to the same piece type with a different target, or repeated the same move.

The forfeits cluster on plies where **all 4 retry attempts share the same structural mistake.** A representative case: at one ply Black was in check from a distant bishop. Opus proposed 4 moves over its retry budget — none addressed the check, because Opus didn't perceive the check at all. Every retry proposed a "good move" that ignored the check, because the model's mental model was missing the bishop's attack on the king.

**Feedback fixes "wrong move from a roughly-correct mental model" but cannot fix "wrong mental model of what's on the board."** The model can change its move but it cannot, via text feedback alone, change what it thinks it's looking at.

### Hardest positions in the bank

Aggregated across all runs on a 20-position hand-curated bank:

| Position class | Failure rate | What the model misses |
|---|---|---|
| Distant-attacker check (e.g., bishop on f1 checking king on d3) | 67% | Long-diagonal line-of-sight |
| Endgame with enemy pawn attacking king-escape squares | 67% | Pawn attack geometry |
| Promotion blocked by own king (Lucena position) | 33% | Target-square occupation |
| Phantom pawn moves in middlegame | 33% | Per-pawn position tracking |
| Mate detection in K+Q vs K | 80% claim error | Mate-in-1 perception in simple endgames |
| Standard opening positions (Italian, Ruy Lopez, King's Indian, etc.) | **0%** | Memorized opening theory |

The 0% failure rate on standard openings vs 33-67% on synthetic / constructed positions is the cleanest test of "is this model reasoning or recalling" the benchmark provides.

### Legality and consistency expose different failures

The eval suite includes two single-position evals — `legality` (does the model produce a legal SAN move?) and `consistency` (does the model correctly describe what its moves do?). Across the data:

- 6 positions fail both
- 4 positions fail legality only
- **19 positions fail consistency only** — model picks a legal move, but its claim about whether it's check, capture, or mate is wrong
- 11 positions pass both

Consistency-only failures are the largest bucket. The model picks correctly but can't always tell you what the move does. Two different facets of the same underlying weakness.

---

## Methodology gotchas (and how the benchmark handles them)

Reasoning-tier models share a token-budget quirk that bites anyone evaluating them with structured output (tool calling, JSON mode). This benchmark hits it; here's the diagnosis and the fix.

### The token-exhaustion failure mode

OpenAI's `max_completion_tokens` and Google's `max_output_tokens` parameters cap the total tokens a reasoning model can produce, **including internal reasoning**. If the model spends the entire budget reasoning, there are zero tokens left for the visible output — including the forced tool call. The model returns with `finish_reason='length'`, no tool call, no content.

A naive eval harness records this as "model failed to follow the schema" → forfeit. But the failure is in the harness's budget, not the model's capability. The model wanted to emit a tool call but ran out of room.

Anthropic's API does not have this problem — `max_tokens` there means *output* tokens only, with thinking budget tracked separately.

### The fix the benchmark applies

Three layers, in order of necessity:

1. **Generous `max_tokens` ceiling (65536).** Well below OpenAI's 128K output limit, but far above the actual reasoning-token usage on >99% of calls. Closes the bug for typical positions.

2. **`prior_failed` list capped to last 3 attempts.** Each retry passes the model the list of previous failed SAN attempts. After 5-10 retries, the prompt is long and reasoning expands to re-analyze everything. Capping to 3 keeps prompt size bounded.

3. **Gradual `reasoning_effort` stepdown on length failures.** If a call still hits `finish_reason='length'` despite the above, the next retry on that move drops reasoning effort one notch: `default → medium → low → minimal`. This activates *only after* the model demonstrates it can't fit at the current effort. The benchmark does NOT a-priori set `reasoning_effort='low'` — that would handicap the model and produce data that under-measures its capability.

The combination makes the benchmark robust: most calls land at full reasoning effort with no token issues, pathological positions get a graceful fallback, and the data is honest about which calls used reduced reasoning (logged per attempt).

### Reproducing — diagnostic and fix recipe

For anyone implementing or extending this benchmark on a new provider:

```
1. Grep run JSONLs for "did not call submit_move".
   Any hits = token-budget bug.
2. Inspect finish_reason in the error message.
   - "length" → bump max_tokens (start at 65536)
   - "content_filter" → safety refusal (different problem)
   - "stop" with no tool → model declined (likely tool_choice issue)
3. If still failing at 65536: enable the stepdown ladder.
4. If still failing: trim the prompt — prior_failed list is a usual suspect.
```

---

## Caveats and expectations

**Caveats:**

- **Sample sizes are small by default** (5 games for CR, 3 for PS). Enough for qualitative pattern signal but not for tight effect sizes. A 0.02 difference between two models could be sampling noise. Raise `--games` for publishable comparisons.
- **Stockfish version matters.** Move-quality scores depend on the engine's evaluation function. Lock the binary version when comparing across runs. This work used Stockfish 18.
- **Provider tool-calling differences.** Each provider's structured-output format works slightly differently. The adapters normalize these; a regression on a specific provider can show up as a benchmark regression. Always grep the run JSONL for `"did not call submit_move"` before drawing conclusions.
- **This is not a chess-skill benchmark.** A model that's bad at chess but rule-consistent would score well. The eval measures state-tracking and rule-following; ELO is incidental.

**Reasonable expectations:**

- Frontier reasoning-tier models score **0.15-0.50** on the headline composite metric, despite playing 95-99% Stockfish-quality moves *on the moves they do play*. The gap between move quality and game completion is the spatial-reasoning ceiling.
- Opening positions show **0% failure rate** for every model tested. Failures concentrate on positions outside training distribution.
- Within-provider frontier vs budget tiers do not always rank the way general-reasoning benchmarks predict. The chess substrate exposes a specific weakness that frontier-tier reasoning doesn't always solve, and sometimes hurts.

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
```

### Cost and runtime (per model, default config)

| Model class | CR + PS cost | Wall time |
|---|---|---|
| Mid-tier non-reasoning (Haiku, Flash Lite) | ~$0.30-0.50 | ~6-15 min |
| Strong reasoning (Opus, Sonnet) | ~$3-5 | ~12-30 min |
| Slow reasoning (GPT-5, Gemini Pro Preview, DeepSeek-reasoner) | ~$5-15 | ~30-90 min |

Full matrix: **~$25-40 in API spend, ~2-3 hours wall time** if jobs run sequentially. Parallelize across providers to cut wall time roughly in half. Single Stockfish skill, single position bank — no per-cell methodological knobs to tune.

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
    chess_reliability.py      # CR metric
    play_strength.py          # PS metric
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
2. **Skill sweep.** CR and PS at multiple Stockfish skills (0, 5, 10) — how does the cascade interact with opponent strength?
3. **Mid-game starting positions.** Does the cascade need game-length state accumulation, or does it appear immediately from a complex mid-game FEN? Separates "drift across turns" from "complex positions are harder regardless."
4. **Reasoning-trace inspection across retries.** Save every retry's full response (currently only the final). Measures whether the model genuinely updates between retries vs pattern-matching a different SAN.
5. **Other 2D-state substrates.** Build analogous benchmarks for grid-navigation, UI layout reasoning, or tile-based puzzles. Confirms the spatial-reasoning failure profile is universal across 2D-grounded tasks, not chess-specific.

---

## The bottom line

The eval measures whether LLMs can maintain a 2D state of typed entities and correctly compute geometric queries against it across many reasoning steps. Frontier reasoning-tier models score in the 0.15-0.45 range on the headline composite while playing 95-99% high-quality moves on the moves they do make. The gap between move quality and game completion is the spatial-reasoning ceiling — observable as a single number per model, comparable across providers.

Chess gives a domain with deterministic ground truth, calibrated difficulty (memorized openings vs unique mid/endgames), and rules whose application is purely geometric. Models describe these rules verbally with 99% accuracy while applying them spatially at much lower rates. The benchmark exposes this gap with a small, cheap, reproducible test — a structural weakness shared by frontier and budget models alike on a generalizable cognitive dimension.

The most striking single finding is the **persistent wrong belief** pattern: models don't make random illegal moves; they form coherent-but-wrong pictures of the position and commit to them across multiple stateless API calls. This is qualitatively different from typical hallucinations and suggests that fixing it requires architectural changes — not just bigger context windows or more training data.
