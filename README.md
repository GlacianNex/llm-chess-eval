# LLM Chess Eval

A reproducible benchmark that measures whether LLMs can maintain coherent state and apply rules across many reasoning turns, using chess as a substrate. The cognitive failure mode it isolates is **state reconstruction and 2D spatial reasoning on out-of-distribution positions** — a weakness that shows up wherever LLMs operate on structured state, not just chess.

**The strongest model in the cross-family matrix scores 0.64 on a [0, 1] ChessReliability scale where Stockfish self-play is the 1.0 reference.** That's a budget non-reasoning model. Frontier reasoning models from Anthropic, OpenAI, and DeepSeek either struggle on first-attempt legality or score below the budget top on the composite anyway.

📊 **[RESULTS.md](RESULTS.md)** — the matrix, findings, deep dives
📖 **[METHODOLOGY.md](METHODOLOGY.md)** — scoring formulas, methodology constraints, reproduction recipe

## What it measures

Two composite scores per model, both bounded in `[0, 1]`:

- **ChessReliability** — rule-following over full games vs Stockfish skill 3 (~1500 ELO, intermediate amateur), with up to 10 retries per illegal move at steep per-retry cost. Catches models that can't produce legal play on first attempt.
- **PlayQuality** — move strength once a legal move is found, against Stockfish skill 5 (~1700 ELO, strong amateur). Catches models that play legal-but-bad moves.

Both metrics use amateur-tier Stockfish opponents — this is not a model-vs-engine comparison. The 1.0 reference on the [0, 1] scale is Stockfish self-play as a scoring anchor, not the opponent the benchmark plays against.

Per-move score multiplies three factors: exponential decay over centipawn loss vs Stockfish's best, a `0.25^retries` cost on retries, and a geometric phase weight (1 / 2 / 4 / 8 at ply boundaries 10 / 20 / 30) that makes late-game plies dominate the score. The geometric phase weight bakes the memorization-cliff thesis directly into the metric.

See [METHODOLOGY.md](METHODOLOGY.md) for the full formulas and the rationale behind each factor.

## Quickstart

```powershell
# Install
git clone https://github.com/GlacianNex/llm-chess-eval.git
cd llm-chess-eval
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e '.[all,dev]'

# Stockfish required on PATH or set STOCKFISH_PATH (Stockfish 18 for reproducibility)

# Configure provider keys for whichever you want to use
$env:ANTHROPIC_API_KEY = "sk-ant-..."
$env:OPENAI_API_KEY    = "sk-..."
$env:GOOGLE_API_KEY    = "AIza..."
$env:DEEPSEEK_API_KEY  = "sk-..."

# Sanity check
llm-chess-eval check-env

# Single-model composite metrics
llm-chess-eval reliability   --model claude-opus-4-7 --games 5
llm-chess-eval play-strength --model claude-opus-4-7 --games 3

# Cross-provider 8-cell matrix (frontier + budget for 4 providers)
llm-chess-eval benchmark --dry-run    # preview, no spend
llm-chess-eval benchmark              # ~$25-40 to run all 8 cells
```

## Why this is useful

Public benchmarks measure knowledge (MMLU, GPQA) or single-shot problem solving (MATH, HumanEval). Neither isolates whether a model maintains coherent state and applies known rules across many reasoning steps. That's the cognitive dimension that determines whether you can trust a model on long agentic tasks.

Chess is the perfect probe because the first 5-10 moves of nearly every game are saturated with training data (memorization works), while mid-game and endgame positions are essentially unique (only reasoning works). The eval measures what's left when memory runs out.

Two clean ground-truth oracles — `python-chess` for rule-checking and Stockfish for move-quality scoring — make the test fully reproducible and deterministic against a fixed engine version.

## Multi-provider support

Adapters and routing for Anthropic (Claude), OpenAI (GPT), Google (Gemini), DeepSeek, and OpenAI-API-compatible endpoints (Together / Groq / vLLM / Ollama). New providers are easy to add — implement one `propose_move(fen) -> CallOutcome` method against the shared schema in `adapters/_shared.py`.

See [METHODOLOGY.md § Adding a new provider](METHODOLOGY.md#adding-a-new-provider) for the recipe.

## License

MIT. See [LICENSE](LICENSE).

Stockfish is GPL-3 and is invoked as an external binary (not bundled).

## Citation / sharing

The benchmark is designed to be shared and reproduced. The full cross-family matrix and methodology details live in [RESULTS.md](RESULTS.md) and [METHODOLOGY.md](METHODOLOGY.md). If you build on this, feel free to point at the repo.
