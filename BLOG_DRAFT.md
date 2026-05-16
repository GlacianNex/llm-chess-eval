# I Made 9 AI Models Play 1,300 Games of Chess. The Winner Was a Budget Model — and Every Single One Hallucinated Pieces.

*A reproducible benchmark, a surprising leaderboard, and a cognitive failure mode that should worry anyone building agentic AI.*

---

## TL;DR

- **9 LLMs, 4 providers, ~1,300 games of chess** against Stockfish — scored on a single composite metric from 0 to 1.
- **The winner wasn't a frontier reasoning model.** Gemini Flash Lite — a budget non-reasoning model — tied Gemini Pro at the top, and outscored GPT-5 by 1.5×.
- **Every model hallucinated pieces.** They captured squares that were empty, advanced pawns that didn't exist, walked their king into check while confidently narrating why it was safe.
- **The failure mode is the headline.** Models form coherent-but-wrong mental models of the board and commit to them across many independent API calls. You can't fix it by sampling more or with a different temperature — the wrong belief regenerates from the same input every time.
- **It generalizes.** Chess is just a clean substrate. The same failure mode shows up wherever LLMs have to track state across many reasoning turns — coding, document writing, UI layout, anything agentic.
- **Open source:** [github.com/GlacianNex/llm-chess-eval](https://github.com/GlacianNex/llm-chess-eval) — reproduce the matrix against any provider in minutes.

---

## Watch this. It's 90 seconds.

[**Levy Rozman — "ChatGPT vs Meta AI: This Isn't Chess Anymore"**](https://www.youtube.com/shorts/YlMWZNx93G4)

Two of the most popular AI models in the world try to play a game of chess against each other. Within a few moves:

- A knight captures a piece that's already off the board.
- A bishop slides through three other pieces in a straight line.
- Both models narrate strategic plans involving pieces that aren't there.

It's funny once. It's unsettling the second time, because what you're watching isn't a "chess problem." It's a **state-tracking problem**. The models can talk about chess all day. But the moment they have to keep their mental picture of an 8×8 board accurate across many independent reasoning turns, the picture corrupts. Pieces drift. Captures lose their referents. Plans reference imagined positions.

I wanted to put numbers on it.

---

## The benchmark, briefly

I built **LLM Chess Eval**. It does one thing: it asks an LLM to play full games of chess against Stockfish (the chess engine, dialed down to amateur strength), and scores the model on a composite metric called **PlayStrength** that captures three things at once:

1. **Move quality** — how close to optimal was each move? (vs Stockfish's analysis, in centipawns.)
2. **Rule-following discipline** — did it produce a legal move on first attempt, or did it need retries? Retries cost steeply (`0.25^retries`).
3. **Game depth** — did the model reach the endgame, or break down in middlegame? Late plies count 3× more than opening plies.

The three factors multiply. A model that forfeits early can't rescue the score with great opening play. A model that drags out a long game of mediocre moves can't rescue the score with completion. A model that needs many retries can't rescue the score with eventual success. The composite is **all-or-nothing** in a way that matches the cognitive failure I'm trying to measure: coherent state-tracking across turns isn't 50%-coherent. It's either there or it isn't.

Each move, the model gets **only the FEN** — a one-line string encoding the full board state. No chat history. No memory of previous moves. Every turn, the model has to reconstruct its mental picture of the game from scratch.

If the model proposes an illegal move, the harness tells it: "Your move X was illegal. Try again." Up to 10 retries per move. If it can't find a legal move within the budget, the game forfeits.

Stockfish self-play would score 1.0 on this metric. A model that never produces a legal move scores 0.

I ran 9 cells across 4 providers — frontier and budget tier for each — totaling ~1,300 games. Here's where they landed.

---

## The matrix

| Provider | Tier | Model | PlayStrength | first-attempt-legal | forfeit rate |
|---|---|---|---|---|---|
| Google | frontier | gemini-2.5-pro | **0.485** | 93.2% | 0% |
| Google | budget | gemini-3.1-flash-lite | **0.477** | 86.5% | 5% |
| OpenAI | frontier | gpt-5 | 0.301 | **99.8%** | **0%** |
| DeepSeek | frontier | deepseek-reasoner | 0.288 | 79.4% | 0% |
| Anthropic | frontier | claude-opus-4-7 | 0.281 | 71.3% | 30% |
| OpenAI | budget | gpt-5-mini | 0.279 | 88.0% | 0% |
| Anthropic | mid | claude-sonnet-4-6 | 0.149 | 63.3% | 20% |
| DeepSeek | budget | deepseek-chat | 0.097 | 33.9% | 80% |
| Anthropic | budget | claude-haiku-4-5-20251001 | 0.074 | 42.9% | **95%** |

Three things to notice.

**The matrix is led by Google — and a budget model is tied with the frontier.** Gemini Flash Lite is a tiny, non-reasoning, $0.10-per-million-tokens model. It ties the frontier reasoning model from the same family. On the supplemental MoveQuality metric (which runs against a harder Stockfish opponent), Flash Lite beats Pro by 2.4×.

**The frontier reasoning cluster sits at 0.28–0.30.** GPT-5, Claude Opus, DeepSeek-reasoner — all frontier-tier reasoning models, all clustered well below the top. "Reasoning-tier optimization" is not a strong predictor of performance on this benchmark.

**The bottom three cells barely play chess.** Haiku forfeited 168 of 176 games — a 95% failure rate. It can't produce a legal sequence of chess moves end-to-end almost ever. DeepSeek-chat is at 80% forfeit. Claude Sonnet at 20%.

---

## The counter-intuitive winner

The most jarring thing about the matrix is the GPT-5 row. GPT-5 has **99.8% first-attempt-legal moves** — basically never makes an illegal move on first try. Zero forfeits across 166 games. Zero average retries per move. It is the most rule-compliant cell in the matrix by a wide margin.

It scores **0.301** on PlayStrength.

Flash Lite, by contrast, makes an illegal first-attempt move about **1 in 8 plies**. It forfeits 5% of its games. It needs 0.20 retries per move on average.

It scores **0.477** — over 1.5× higher than GPT-5.

**How does the less-rule-following model win the composite?**

PlayStrength has three factors. Rule-following is one of them. The other two are move quality and game depth. Flash Lite wins both by wide margins:

| | GPT-5 | Flash Lite |
|---|---|---|
| Opening ACPL (centipawn loss vs Stockfish's best) | 85 | **45** |
| Middlegame ACPL | 111 | **77** |
| Mean game length (out of 60 plies) | 26.9 | **43.1** |
| **PlayStrength composite** | 0.301 | **0.477** |

In chess terms: GPT-5 plays at **intermediate-club** strength but does so with perfect discipline. Flash Lite plays at **strong-club / weak-master** strength and occasionally needs a do-over.

Most chess players, asked which is the stronger player, would pick Flash Lite. **The composite ranks the stronger chess player, not the more rule-compliant one.**

The 5% forfeit rate costs Flash Lite about 0.025 on the composite. GPT-5's mid-quality moves cost it about 0.20. Move quality and game depth dominate by ~8×.

A reasonable hypothesis: Google's training corpus has more chess-relevant pattern coverage per parameter than its frontier siblings, and Flash Lite gets to deploy that pattern-matching directly without paying for expensive reasoning that doesn't translate to better moves in this domain. It doesn't generalize — GPT-5-mini doesn't beat GPT-5, DeepSeek-chat collapses. **The Flash Lite outlier is specifically about how Google's chess priors scale down to the budget cell.**

---

## The deepest finding: persistent wrong belief

Now the part you should care about even if you don't care about chess.

When LLMs produce illegal moves, they don't make random errors. They form **coherent-but-wrong mental models of the board, and commit to them deterministically across multiple stateless API calls.**

Here's Claude Opus, in one game, proposing the same illegal king-capture five times across 11 plies:

```
ply 25: rationale "grabs the pawn on g4"           — illegal: would put king in check
ply 32: rationale "captures the g4 pawn for free"
ply 33: rationale "wins the g4 pawn"
ply 35: rationale "to grab material"
ply 36: rationale "capture the g4 pawn to remove threat"
```

White's king cannot capture the pawn on g4 — doing so exposes the king to check from a distant attacker. The model regenerates this same wrong move five times. Each API call is fully independent (no chat history). But the same FEN keeps producing the same wrong pattern-match.

Here's Claude Sonnet, proposing to advance a pawn that doesn't exist:

```
"Advancing the passed pawn to d7 attacks the black rook..."
"Advancing d6 pawn to d7 puts tremendous pressure..."
"Advancing the d6 pawn to d7 puts tremendous pressure..."
```

There is no white pawn on d6. The model believes in a piece configuration that isn't on the board, and the wrong belief is encoded in its response to that specific input.

**This is qualitatively different from typical hallucinations.** A standard hallucination is a one-shot plausible-sounding wrong fact. The pattern here is **convergence on the same specific wrong belief across many independent invocations.** You cannot fix it by sampling more times. You cannot fix it with a different temperature. The wrong mental model regenerates deterministically from the same input.

The harness's retry mechanism gives the model the list of recent failed attempts and asks it to try again. This fixes "wrong move from a roughly-correct mental model" (the model switches to a different piece). It does NOT fix "wrong mental model of the board" — the model changes the move it commits to, but cannot change what it thinks it's looking at.

Across the 224 illegal moves I classified, **~95% are spatial-reasoning failures** — line-of-sight, attack-sets, path-blocking, phantom pieces. Two clean models tested (Opus and Sonnet) showed essentially identical failure-mode distributions. **This is not a model-specific quirk.** It's a structural feature of how current LLMs handle 2D spatial state.

---

## "Surely it's because the positions are out-of-distribution?"

The natural pushback is: openings are memorized; mid/endgame positions are novel; that's why models fail later in games.

I tested this directly. We built 4 banks of 20 chess positions each, ranging from "hand-curated textbook positions" (saturated in training data) to "pure random self-play" (maximally novel). Then ran 5 models on each bank.

Per-position legality:

| Model | Hand-curated | Random self-play | Cliff |
|---|---|---|---|
| gpt-5 | **100%** | **100%** | 0% |
| gemini-3.1-flash-lite | 95% | 90% | –5% |
| claude-sonnet-4-6 | 75% | 70% | –7% |
| claude-opus-4-7 | 90% | 80% | –11% |
| deepseek-chat | 50% | 25% | –50% |

**The cliff is model-specific, not universal.** GPT-5 handles totally random chess positions just as well as memorized ones. Flash Lite barely drops. Sonnet doesn't drop at all. Only DeepSeek-chat shows a sharp per-position novelty cliff.

So why do top models still play badly in the endgame phase of full games?

Because the failure isn't "is this single position out of training distribution?" It's **cumulative state-tracking across many turns.** Even on positions a model could handle in isolation, the drift across 30–40 turns of stateless reasoning corrupts its mental model. Errors compound. By ply 25, the model's internal picture of the board diverges from the actual board — and once that gap opens, every subsequent move is computed against the wrong picture.

This is the cognitive failure the benchmark is designed to expose. **It is real, it is universal, and it is invisible in any benchmark that doesn't make models maintain state across many independent calls.**

---

## Why this matters beyond chess

Chess is just the substrate. The cognitive primitive being tested is universal:

> **Can the model maintain coherent state and apply known rules across many independent reasoning turns?**

This primitive shows up everywhere in real-world agentic AI:

- A coding assistant proposes a refactor that references a function it imagined exists. When you point out the function doesn't exist, the model proposes a *different* function — pulling the imagined function from a *different* imagined location. The mental model of your codebase has drifted.
- A documentation-writing AI confidently describes APIs that have been deprecated, with perfect syntax. Its mental model of the API surface is wrong but coherent.
- An LLM doing UI layout places elements at coordinates that don't fit the screen and confidently explains why those coordinates are correct.
- An agentic AI running a multi-step workflow loses track of what it already accomplished and proposes redoing earlier steps.

These are all instances of the **same failure mode.** Chess is just clean enough to measure numerically because chess has a deterministic rule-checker (python-chess) and a strength oracle (Stockfish). The same cognitive collapse happens everywhere — it's just usually invisible because most tasks don't have a clean oracle.

The scoring is designed to be **hill-climbable.** Today's matrix top is 0.485. Engine-quality play would score 0.95+. The gap — between today's frontier and a model that can keep its mental picture of an 8×8 board accurate across 40 turns of stateless inference — is the **cognitive headroom** this benchmark exists to quantify.

That headroom matters. It's the gap between AI that can pass the bar exam and AI you can trust to drive a 50-step agent loop without losing track of what's happening.

---

## What's open-source

Everything: the benchmark, the matrix data, the scoring formulas, the analytics, the raw JSONL game records, and the harness that runs against any provider.

**[github.com/GlacianNex/llm-chess-eval](https://github.com/GlacianNex/llm-chess-eval)**

A single CLI runs against Anthropic, OpenAI, Google, DeepSeek, or any OpenAI-API-compatible endpoint. Adding a new provider is one adapter file. Re-scoring with different weights doesn't require re-running games — the raw data is saved as JSONL.

If you fork it, re-weight the scoring, add a provider, run your own cells, or find a failure case I missed — I'd love to hear about it.

The model that breaks this benchmark will be the model that can maintain coherent state across an agent loop. I'd really like to know which one gets there first.

---

*Methodology details, the full matrix with diagnostic columns, footnotes on every cell, the τ-sensitivity analysis, ACPL by phase tables, the chess glossary, and the reproduction recipe are all in the repo's METHODOLOGY.md and RESULTS.md.*
