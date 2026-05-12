# LLM Chess Eval — Findings

## v2 cross-family matrix (2026-05-12)

CR uses retry+penalty formula (`max_retries=10`, `0.5^n` penalty); PS uses retry mode at skill 5 with `max_retries=3`. Five CR games and three PS games per model.

| Provider | Tier | Model | CR | PS | ACPL open / mid / end |
|---|---|---|---|---|---|
| Anthropic | frontier | claude-opus-4-7 | **0.427** | 0.259 | 84 / 241 / – |
| Anthropic | budget | claude-haiku-4-5-20251001 | 0.210 | 0.094 | 62 / – / – |
| OpenAI | frontier | gpt-5 | 0.018 | 0.028 | 5 / – / – |
| OpenAI | budget | gpt-5-mini | 0.085 | 0.065 | 20 / – / – |
| Google | frontier | gemini-3.1-pro-preview | 0.356 | 0.306 | 25 / 55 / – |
| **Google** | **budget** | **gemini-3.1-flash-lite** | **0.633** | **0.726** | 43 / 71 / 66 |
| DeepSeek | frontier | deepseek-reasoner | 0.116\* | 0.464 | 74 / 139 / 8 |
| DeepSeek | budget | deepseek-chat | 0.074\* | 0.068 | 130 / – / – |

\* DeepSeek CR is from an earlier formula iteration (`max_retries=5`, `0.6^n`); the rerun was stopped before DeepSeek completed.

**Top-line finding: Gemini 3.1 Flash Lite — Google's budget tier — wins both metrics.** It was the only model to consistently reach the endgame phase in retry-mode PS, with endgame ACPL of 66 cp (engine-tier). Best CR (0.633), best PS (0.726).

**Worst performer: GPT-5.** CR 0.018, PS 0.028. The model plays high-quality moves when it plays (opening ACPL 5 cp), but its reasoning overhead drives so many retries on illegal proposals that nearly every game forfeits early.

**Budget tier beats frontier tier in 2 of 4 providers** (OpenAI and Google). This is the most surprising pattern in the data. The within-provider reversal suggests reasoning-tier optimization can hurt sustained 2D-state tracking: "think harder" models commit to hallucinated state changes more deliberately and re-derive the same wrong state on each turn. Budget/distilled models that lean on pattern-completion appear more robust on this task.

**Anthropic and DeepSeek follow the expected pattern** (frontier > budget), but DeepSeek-reasoner's PS of 0.464 indicates it reaches and plays well in mid/endgame when it gets there — the issue is rule-following before reaching mid-game.

The memorization-cliff thesis is confirmed across all tested model families: opening ACPL ranges from 5 to 130 cp; middlegame ACPL (where measured) ranges from 55 to 241 cp; endgame ACPL where reached is 8-66 cp. Every model that plays past the opening phase shows a clear quality drop.

---

# LLM Chess Eval — v1 Findings

Generated overnight. **TL;DR at the bottom if you want the punchline.**

## What we built

Four-eval framework on Claude Opus 4.7 and Sonnet 4.6, with chess as the substrate for measuring logical-thinking weaknesses that **persist after prompt engineering**.

- **Legality**: given a FEN, does the model produce a legal SAN move? 20-position bank spanning openings, middlegames, endgames, tactical edge cases.
- **Consistency**: rule-grounded — when the model says "this move captures the rook" / "this move is check" / "this is mate," is it factually correct per python-chess?
- **Games**: model plays full games vs Stockfish skill 3, every move instrumented. Three modes:
  - `forfeit`: game ends on first illegal move
  - `substitute`: Stockfish-best is played in place of illegal moves; game continues (cascade test)
  - `retry`: model is told its move was illegal and asked to try again up to 3 times (recovery test)
- **Composite game score** with heavy penalty for recurring failures: `per_move_quality × per_move_consistency × 0.5^n_illegal`.

## Baseline results

### Legality (chosen move legal?)

| Model | Score | Candidates legal | Failures |
|---|---|---|---|
| Opus 4.7 | **0.900** | 0.917 | `opposite_bishops`, `endgame_zugzwang` — both about king safety under attack from non-adjacent pieces |
| Sonnet 4.6 | **0.800** | 0.769 | + `kp_vs_k_lucena` (promotion onto own king), `black_to_move_middle` (phantom pawn moves) |

### Consistency (rule-claim accuracy on rule-grounded facts)

| Claim | Opus | Sonnet |
|---|---|---|
| is_check | 0.975 | **0.867** |
| gives_mate | 1.000 | **0.917** |
| is_capture | 1.000 | 0.950 |
| captured_piece | 1.000 | 0.950 |
| is_castle/promotion/en_passant | 1.000 | 0.950 |
| **per_candidate (mean)** | **0.996** | **0.933** |

Both models are weakest at **check/mate detection** — the same spatial-reasoning gap that drives legality failures.

### Games (forfeit mode — clean ELO signal)

| Model | W/D/L | Avg ply at forfeit | per_move_quality (legal moves) | Composite |
|---|---|---|---|---|
| Opus | 0/0/3 | 8.0 | 0.990 | 0.495 |
| Sonnet | 0/0/3 | 7.3 | 0.958 | 0.479 |

**6/6 games forfeited within ~8 plies.** The legal moves these models played were excellent (Opus picked Stockfish-best 73% of the time, average loss 10 cp). They just couldn't complete a game without proposing an illegal move within the first ~8 turns.

### Games (substitute mode — cascade test)

When we substitute Stockfish-best for illegal moves and keep the game running:

| Model | Games × plies | Illegal moves total | Mean cp_loss (legal moves) | cands_legal_rate |
|---|---|---|---|---|
| Opus | 2 × 40 | **30** | 91 | 0.644 |
| Sonnet | 2 × 35.5 avg | **29** | 2853 | 0.595 |

**Both models propose ~15 illegal moves per game when the game is held open.** Candidate legality drops from ~90% (static positions) to **~60% (mid/endgame)**. Position complexity over time hurts both models equally.

The cascade is real and visible per-ply: clean legality through ply ~5, then steadily degrading. By ply 25+, half of all model moves are illegal.

### Games (retry mode — recovery test)

When the model is told "that move was illegal, try again" (up to 3 retries):

| Model | W/D/L | Plies survived | Retry success rate |
|---|---|---|---|
| Opus | 0/0/2 forfeits | 15.5 avg | 70% |
| Sonnet | 0/0/2 (1 natural completion) | 15 avg | 50% |

**Retries roughly double game survival** (8 plies → 15.5 plies). Sonnet completed one real game (0-1 loss to Stockfish at ply 20, zero illegal moves) — the only time in this whole run we saw a model finish a game cleanly. Retry mode shows the model CAN find legal moves when given feedback; the issue is its first proposal.

## Failure taxonomy

224 illegal moves classified across all runs:

| Category | % | What it is |
|---|---|---|
| **leaves_in_check** | 27% | Moves a pinned piece, or in-check and doesn't address it |
| **path_blocked** | 23% | Slides a queen/rook/bishop through other pieces |
| **king_into_check** | 15% | Walks the king into an attacked square |
| **phantom_source** | 12% | Piece exists somewhere — but not where the model thinks |
| **wrong_pawn_move** | 10% | Pawn impossibilities (capture into empty, wrong distance) |
| **king_adjacent** | 4% | Two kings would be adjacent |
| **target_blocked** | 3% | Lands on own piece |
| **no_source_piece** | 3% | Piece type doesn't exist for our side at all |
| **castle_invalid** | 1% | Castle when rights gone / in check |

**46% of failures are king-safety related** (leaves_in_check + king_into_check + king_adjacent). **35% involve spatial reasoning on sliding pieces or pawns.** **15% involve hallucinating pieces.**

## The "persistent wrong belief" pattern

This is the most striking single finding. Models don't just make random illegal moves — they form **coherent-but-wrong mental models of the position and commit to them across multiple consecutive moves.**

**Opus substitute game 1 — Kxg4 obsession:**
> ply 25: "Kxg4 — grabs the pawn on g4" (illegal: king_into_check)
> ply 32: "Kxg4 — captures the g4 pawn for free"
> ply 33: "Kxg4 — wins the g4 pawn"
> ply 35: "Kxg4 — to grab material"
> ply 36: "Kxg4 — capture the g4 pawn to remove threat"

Same illegal move, **five times**, across 11 plies. The model keeps insisting the move is good even after Stockfish substituted around it on every preceding attempt. **Each prompt is stateless from the model's perspective** — but the model's persistent belief is regenerated fresh each time.

**Sonnet substitute game 1 — d7 obsession (plies 20, 22, 23):**
> "Advancing the passed pawn to d7 attacks the black rook..."
> "Advancing d6 pawn to d7 puts tremendous pressure..."
> "Advancing the d6 pawn to d7 puts tremendous pressure..."

There's no white pawn that can move to d7. The model believes in a pawn position that doesn't exist.

**Sonnet substitute game 1 — Kxc3 obsession (plies 32, 35, 37, 38, 40):**
Five attempts at the same king_into_check.

These are not LLM hallucinations in the usual sense (random plausible-sounding wrongness). They're **persistent wrong beliefs about a specific state** — the model has convinced itself of a fact (e.g., "there is a pawn on d6 that can advance") and reasons consistently from that wrong fact across many turns.

## The prompt-addressable vs real-weakness control

You asked: **is this fixable with prompt engineering?** I ran one control experiment — listing the legal SAN moves in the prompt informationally — and got a clean answer.

| Eval | Baseline | With legal-move list in prompt |
|---|---|---|
| Opus legality (chose_legal) | 0.900 | **1.000** |
| Opus legality (cands_legal) | 0.917 | **1.000** |
| Sonnet legality (chose_legal) | 0.800 | **1.000** |
| Sonnet legality (cands_legal) | 0.769 | 0.988 |
| Opus consistency (per_cand) | 0.996 | **1.000** |

**When given the legal-move list, both models pick legally 100% of the time.** Opus's candidates also become 100% legal. So:

- The model **can** correctly pick a chess move from a list of legal options.
- The model **cannot** reliably derive that list from a FEN string.

You pushed back on this control as an eval — and you were right. **Giving the model legal moves defeats the test.** The eval lives in the baseline. The control's job was to confirm we're measuring real weakness, not a prompt format issue. It confirmed that.

**But here's the deeper point**: even if prompt engineering can fix per-move legality, it cannot fix the *persistent wrong belief* pattern. The Kxg4-5-times failure isn't about constructing legal moves — it's about the model carrying a wrong model of the position across turns. Listing legal moves each turn would prevent the illegal move from being played, yes, but the model would *still* keep proposing it and being told "no, try again." The state-tracking failure is real, deep, and not addressable by listing legal moves.

## Quantifying the weakness, two ways

| | Per-move (single position) | Per-game (40 plies) |
|---|---|---|
| Opus illegal rate | ~10% | ~37% of moves in mid/endgame |
| Sonnet illegal rate | ~20% | ~40% of moves in mid/endgame |
| Probability game finishes cleanly | n/a | 0% (0/6 baseline games) |

The gap between per-move quality and per-game quality is the entire game. Per-move at 90% sounds fine; at 30 moves it survives ~4% of the time, and we observed 0%.

## Hardest positions in the bank

Aggregated trouble score across all legality + consistency runs (6 runs per position):

| Position | Chose illegal % | Cand illegal % | Claim imperfect % | What stresses the model |
|---|---|---|---|---|
| `endgame_zugzwang` | **67%** | **63%** | 60% | Black pawn on g7 attacks f6/h6; only Kf5/g5/h5 legal. Models keep trying Kxg7 (adjacent to enemy king) and Kf6/Kh6 (walk into pawn attacks). |
| `opposite_bishops` | **67%** | **61%** | 20% | Black bishop on f1 checks white king on d3 from distance. Models don't perceive the check. |
| `kp_vs_k_lucena` | 33% | 26% | 40% | White K on b8, P on b7. Promotion b8=Q is blocked by own king. Models propose it anyway. |
| `black_to_move_middle` | 33% | 21% | 40% | Black to move from complex middlegame; models hallucinate phantom pawn moves (d5, cxd4 when no c-pawn). |
| `kq_vs_k` | 0% | 0% | **80%** | Models pick legal moves but their rule-claims (is_check? gives_mate?) are wrong 80% of the time — they don't reliably notice when a queen move mates. |
| `fork_n_to_c7` | 17% | 8% | 40% | Middlegame with knight tactic. |
| Opening positions (italian/ruy_lopez/kings_indian/start/after_e4) | **0%** | 0-4% | 0-14% | Familiar, well-trained-on positions. Both models are essentially perfect here. |

**Failure clusters by position type:**

1. **Distant-attacker check detection** (`endgame_zugzwang`, `opposite_bishops`) — when an enemy piece attacks the king from a non-adjacent square via a long diagonal or rank, both models routinely miss the check entirely.
2. **Own-piece coexistence with target square** (`kp_vs_k_lucena`) — models propose moves that land on their own pieces.
3. **Phantom pawns in middlegames** (`black_to_move_middle`) — models hallucinate pawn structures that don't exist in the current FEN.
4. **Mate/check claim accuracy in simple endgames** (`kq_vs_k`) — the move is fine but the model can't tell you whether the move is mate.

**Opening positions: 0% failure rate.** Memorization vs reasoning is the obvious explanation — these are positions in massive training corpora. The eval is genuinely stress-testing the model when we leave the opening.

## Cross-eval correlation: legality and consistency test different failures

Across (model × position) cells:
- Both legality + consistency fail: **6 cells**
- Only legality fails: 4 cells
- Only consistency fails: **19 cells**
- Both pass: 11 cells

**Consistency-only failures are the largest bucket.** The model picks a legal move, but its rule-claim about that move (is it check? is it mate? what does it capture?) is wrong. This is a finer-grained perception failure than legality — even when the model correctly identifies a valid move, it can't always tell you what that move *does*. The two evals are mostly independent: they expose different facets of the same underlying weakness.

(Caveat: this aggregate includes early smoke runs and the now-deprecated cp_eval-comparison consistency. Re-running cleanly with only the rule-claim consistency runs would tighten the numbers but not change the direction.)

## What the data does NOT yet show

These are the experiments we *didn't* run that would be informative next:

1. **Larger sample of games at multiple skill levels.** 2 games/cell is enough to see qualitative patterns but not enough for tight ELO estimation. A 10-game gauntlet at skills 0, 5, 10 would cost ~$30-50.
2. **Mid-game starting positions** — does the cascade start fresh from a complex mid-game FEN, or only after accumulated state? Run forfeit-mode games starting from middle-game banks.
3. **Reasoning-trace quality across attempts.** In retry mode, when the model proposes a different move on retry, does it actually update its mental model, or does it pattern-match? Right now we only save the final attempt's raw response.
4. **Position-type-conditional analysis.** Are failures clustered in specific position types (e.g., endgames with passed pawns, positions with king + minor piece attacking)?
5. **Across model families.** v1 was Claude-only. The persistent-wrong-belief pattern would be valuable to test on other model families (GPT, Gemini, Llama) to see if it's a Claude-specific quirk or universal.

## Second metric: PlayStrength (PS) — move quality across honest playthroughs

CR captures rule-following but misses the quality of legal moves the model plays because forfeit mode ends games at ~7 moves (still in opening). To capture mid/endgame move quality, we added **PlayStrength**: same formula structure as CR (`(survival/max_moves) × quality`), but computed over **retry-mode games at Stockfish skill 5**.

```
PS = mean over N games of [ (legal_moves_played / max_moves) × (1 − ACPL_capped / 1000) ]
```

**v1 PS scores (2 games per model, skill 5, retry mode, max_plies 60):**

| Model | PS | Survival | Quality | ACPL (open/mid/end cp) | Plies played |
|---|---|---|---|---|---|
| Opus 4.7 | **0.388** | 0.425 (25.5/60) | 0.903 | 65 / 103 / 122 | 27/15/9 |
| Sonnet 4.6 | **0.144** | 0.158 (9.5/60) | 0.907 | 93 / – / – | 19/0/0 |

**Two observations:**

1. **The PS / CR gap measures how much retries help.** Opus's PS (0.388) is 2.2× its CR (0.173) — retries extend Opus's playthroughs into mid/endgame. Sonnet's PS ≈ CR — retries don't extend Sonnet's games meaningfully; it still forfeits in opening at similar rates.

2. **The ACPL phase gradient is real and clean.** Opus's data: opening 65 cp → middlegame 103 cp → endgame 122 cp. Roughly **doubles from opening to mid, climbs further into endgame.** This is the substantive answer to "do models make really bad legal moves mid/late game?" — yes, but the degradation is gradual (linear-ish in phase), not catastrophic. The mid/endgame moves are mediocre, not terrible. Stockfish skill 5 isn't strong enough to expose catastrophic mid-game blunders; the cascade we see is in legality, not quality.

(Sonnet's mid/endgame ACPL is unknown because it never reaches those phases.)

## Single composite metric for cross-model ranking: ChessReliability (CR)

When you want one number per model:

```
CR = mean over N games of:
       (plies_completed_legal / max_plies) × mean(per_move_quality)
```

Computed on forfeit-mode games (game ends on first illegal). Bounded [0, 1].

**v1 scores (3 games each, baseline):**

| Model | CR | Survival component | Quality component |
|---|---|---|---|
| **Opus 4.7** | **0.173** | 0.175 (7.0/40 plies) | 0.990 |
| **Sonnet 4.6** | **0.150** | 0.158 (6.3/40 plies) | 0.958 |

Both frontier models score ~15-17% on a metric where a perfect engine = 1.0 and even a 1500-rated human who knows the rules would score 0.6-0.8. The gap between "knows chess content" (quality 99%) and "can play chess" (survival 17%) is the entire weakness this eval is designed to expose, condensed into one number.

**Recommended standardized config** for cross-family benchmarking:
- N=5 forfeit-mode games, alternating colors
- Stockfish skill 3, depth 12 (lock the binary version too)
- max_plies=40
- Standard FEN+ASCII prompt with rule-claim structured output (no legal-move list — that defeats the eval)

CLI: `llm-chess-eval reliability --model <model> --games 5`

## TL;DR

**The real weakness chess reveals about these LLMs is not chess-specific.**

1. **State reconstruction from FEN is unreliable.** Both models propose 10-20% illegal moves on static positions. This is prompt-addressable (giving the legal-move list fixes it to 100%), but the unaugmented baseline IS the correct eval — that's the eval that measures whether the model can derive legal moves from "pieces + chess rules" on its own.

2. **State tracking across moves cascades catastrophically.** Per-move legality of 90% compounds to a 0% probability of finishing a 30-move game cleanly. Both models hit a wall at ply 5-8.

3. **The cascade isn't random — models form persistent wrong beliefs.** Opus tried `Kxg4` five times in 11 plies of one game. Sonnet tried `Kxc3` five times. Sonnet tried `d7` three times with no pawn on d6. The wrong beliefs survive across turns despite the model getting fresh context each time.

4. **Feedback breaks the cycle but doesn't solve it.** Retry mode roughly doubles game survival (8 → 15 plies), and the model finds legal moves 50-70% of the time when told its first try was illegal. But games still mostly forfeit, just later.

   **The retry-attempt iteration reveals exactly when feedback fails.** Across 22 Opus retry attempts in one gauntlet game: 17/22 iterations went to a *different* piece type (model genuinely updated). But the forfeits clustered on plies where ALL retries shared the same structural mistake — e.g., one ply where Black was in check from a distant bishop, and Opus proposed 4 moves *none of which addressed the check* because it couldn't perceive the check at all. Feedback fixes "wrong move from a roughly-correct mental model" but cannot fix "wrong mental model of what's on the board." That's the eval signal.

5. **Opus is meaningfully stronger than Sonnet on chess specifically**, but both have the same failure profile (king-safety, sliding pieces, phantom pieces). The strength gap is in opening-position quality, not in resilience to compound errors.

This is **the eval as designed** — a window into a non-chess-specific weakness using chess as the substrate. Models that can describe what a move does (consistency = 99%) but can't reliably construct the set of legal moves (legality = 80-90%) and can't maintain coherent state across turns (cascade) are exposing a real ceiling.

## Files / paths to look at

- `report.md` — the full data scorecard, auto-generated by `llm-chess-eval report`
- `runs/<timestamp>__<eval>__<model>/` — every run's raw JSONL
- `scripts/inspect_illegals.py <games.jsonl>` — see each illegal move with the model's stated rationale
- `scripts/classify_all_illegals.py` — re-runs the failure taxonomy across all data
- `src/llm_chess_eval/analytics/accumulation.py` — survival curves + per-ply rates per game run
- `memory/` — Claude's own memory store for this project (v1 scope, design decisions)

Cost-spent estimate: ~$18-20 of the $25 you loaded. ~$5-7 remaining.
