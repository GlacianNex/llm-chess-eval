# LLM Chess Eval

A reproducible benchmark that measures whether LLMs can maintain coherent state and apply rules across many reasoning turns, using chess as a substrate. The cognitive failure mode it isolates is **state reconstruction and 2D spatial reasoning on out-of-distribution positions** — a weakness that shows up wherever LLMs operate on structured state, not just chess.

**Frontier models score ~0.15–0.65 on the headline metrics**, while playing 95-99% Stockfish-quality moves on the moves they do play. The gap between move quality and game completion is the spatial-reasoning ceiling, observable as a single number per model.

📖 **Full project documentation, methodology, and findings: [HANDOFF.md](HANDOFF.md)**
📊 **Results narrative: [FINDINGS.md](FINDINGS.md)**

## What it measures

The benchmark provides two composite scores per model, both bounded in `[0, 1]`:

- **ChessReliability (CR)** — rule-following ability. The model plays vs. Stockfish at skill 3; each illegal-move attempt costs an exponential per-retry penalty (`0.5^n`); after 10 retries the game forfeits. CR rewards a model that produces legal play cheaply.
- **PlayStrength (PS)** — move quality across honest playthroughs at Stockfish skill 5 with retries allowed (no per-retry penalty). PS measures how good the moves are *given* the model finds them legal.

Both metrics multiply a **survival** factor (legal moves played / max moves) by a **quality** factor (ACPL-based). The multiplication means either failing on its own collapses the score.

## Quickstart

```powershell
# Install
git clone https://github.com/GlacianNex/llm-chess-eval.git
cd llm-chess-eval
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e '.[all,dev]'

# Stockfish required on PATH or set STOCKFISH_PATH

# Configure provider keys for whichever you want to use:
$env:ANTHROPIC_API_KEY = "sk-ant-..."
$env:OPENAI_API_KEY    = "sk-..."
$env:GOOGLE_API_KEY    = "AIza..."
$env:DEEPSEEK_API_KEY  = "sk-..."

# Sanity check
llm-chess-eval check-env

# Single-model
llm-chess-eval reliability   --model claude-opus-4-7 --games 5
llm-chess-eval play-strength --model claude-opus-4-7 --games 3

# Cross-provider 8-cell matrix (frontier + budget for 4 providers)
llm-chess-eval benchmark --dry-run    # preview, no spend
llm-chess-eval benchmark              # ~$30-50 to run all 8 cells
```

## Why this is useful

Public benchmarks measure knowledge (MMLU, GPQA) or single-shot problem solving (MATH, HumanEval). Neither isolates whether a model maintains coherent state and applies known rules across many reasoning steps. That's the cognitive dimension that determines whether you can trust a model on long agentic tasks.

Chess is the perfect probe because the first 5-10 moves of nearly every game are saturated with training data (memorization works), while mid-game and endgame positions are essentially unique (only reasoning works). The eval measures what's left when memory runs out.

Two clean ground-truth oracles — `python-chess` for rule-checking and Stockfish for move-quality scoring — make the test fully reproducible and deterministic against a fixed engine version.

## Multi-provider support

Adapters and routing for Anthropic (Claude), OpenAI (GPT), Google (Gemini), DeepSeek, and OpenAI-API-compatible endpoints (Together / Groq / vLLM / Ollama). New providers are easy to add — implement one `propose_move(fen) -> CallOutcome` method against the shared schema in `adapters/_shared.py`.

See [HANDOFF.md §8](HANDOFF.md#8-multi-provider-support) for the full provider list and how to add a new one.

## License

MIT. See [LICENSE](LICENSE).

Stockfish is GPL-3 and is invoked as an external binary (not bundled).

## Citation / sharing

The benchmark is designed to be shared and reproduced. The full v2 cross-family matrix and methodology details live in [HANDOFF.md](HANDOFF.md). If you build on this, feel free to point at the repo.
