# LLM Chess Eval

A reproducible benchmark that measures whether LLMs can maintain coherent state and apply rules across many reasoning turns, using chess as a substrate. The cognitive failure mode it isolates is **state reconstruction and 2D spatial reasoning on out-of-distribution positions** — a weakness that shows up wherever LLMs operate on structured state, not just chess.

**The strongest models in the cross-family matrix score 0.48 on a [0, 1] PlayStrength scale where Stockfish self-play is the 1.0 reference.** Gemini 2.5 Pro (frontier reasoning) and Flash Lite (budget non-reasoning) are essentially tied at the top — and Flash Lite *exceeds* Pro 2.4× on PlayQuality (the supplemental move-quality metric). Frontier reasoning models from Anthropic, OpenAI, and DeepSeek cluster around 0.28–0.30, with budget cells from Anthropic and DeepSeek collapsing under 0.10 due to high forfeit rates.

📊 **[RESULTS.md](RESULTS.md)** — the matrix, findings, deep dives
📖 **[METHODOLOGY.md](METHODOLOGY.md)** — design hypothesis, scoring formulas, parameter rationale
🛠 **[HOWTO.md](HOWTO.md)** — install, configure, run, add new providers, provider quirks

## What it measures

Two composite scores per model, both bounded in `[0, 1]`:

- **PlayStrength** (primary) — composite 0–1 score over full games vs Stockfish skill 3 (~1500 ELO, intermediate amateur). Combines move quality (centipawn loss vs Stockfish's best), retry-cost discipline (`0.25^retries`), and a phase weight that counts late-game plies more.
- **PlayQuality** (supplemental) — move strength once a legal move is found, against Stockfish skill 5 (~1700 ELO, intermediate amateur). Strips the retry-cost factor from PlayStrength. Isolates "how good are the moves themselves" from "how often does the model find a legal move on first try."

Both metrics use amateur-tier Stockfish opponents — this is not a model-vs-engine comparison. The 1.0 reference on the [0, 1] scale is Stockfish self-play as a scoring anchor, not the opponent the benchmark plays against.

Per-move score multiplies three factors: exponential decay over centipawn loss vs Stockfish's best, a `0.25^retries` cost on retries, and a phase weight (1 / 1.5 / 2 / 3 at ply boundaries 10 / 20 / 30) that makes late-game plies count more in the composite. The phase weight rewards sustained state-tracking across many turns — see [METHODOLOGY § The design hypothesis](METHODOLOGY.md#the-design-hypothesis-cumulative-coherence-and-the-model-specific-memorization-cliff) for what the data does and doesn't support about why this matters.

See [METHODOLOGY.md](METHODOLOGY.md) for the full formulas and the rationale behind each factor.

## Quickstart

```powershell
# Install + Stockfish 18 on PATH; configure provider keys
git clone https://github.com/GlacianNex/llm-chess-eval.git && cd llm-chess-eval
python -m venv .venv && .\.venv\Scripts\Activate.ps1
pip install -e '.[all,dev]'
$env:ANTHROPIC_API_KEY = "..."   # plus OPENAI_API_KEY / GOOGLE_API_KEY / DEEPSEEK_API_KEY as needed
llm-chess-eval check-env

# Single-model composite metrics
llm-chess-eval play-strength --model claude-opus-4-7 --games 5
llm-chess-eval play-quality  --model claude-opus-4-7 --games 3

# Cross-provider matrix
llm-chess-eval benchmark --dry-run    # preview cells without invoking models
llm-chess-eval benchmark
```

Full install steps, provider quirks, and "how to add a provider" recipe are in **[HOWTO.md](HOWTO.md)**.

## Why this is useful

Public benchmarks measure knowledge (MMLU, GPQA) or single-shot problem solving (MATH, HumanEval). Neither isolates whether a model maintains coherent state and applies known rules across many reasoning steps. That's the cognitive dimension that determines whether you can trust a model on long agentic tasks.

Chess is the right probe because the first 5–10 moves of nearly every game are saturated with training data while mid-game and endgame positions drift into territory the model hasn't seen — *and* state has to be tracked across many independent stateless API calls, which exposes coherence failures that single-shot evals can't see.

Two clean ground-truth oracles — `python-chess` for rule-checking and Stockfish for move-quality scoring — make the test fully reproducible and deterministic against a fixed engine version.

## Multi-provider support

Adapters and routing for Anthropic (Claude), OpenAI (GPT), Google (Gemini), DeepSeek, and OpenAI-API-compatible endpoints (Together / Groq / vLLM / Ollama). New providers are easy to add — see [HOWTO § Adding a new provider](HOWTO.md#adding-a-new-provider).

## License

MIT. See [LICENSE](LICENSE).

Stockfish is GPL-3 and is invoked as an external binary (not bundled).

## Citation / sharing

The benchmark is designed to be shared and reproduced. The full cross-family matrix and methodology details live in [RESULTS.md](RESULTS.md) and [METHODOLOGY.md](METHODOLOGY.md). If you build on this, feel free to point at the repo.
