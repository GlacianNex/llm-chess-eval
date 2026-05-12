"""Walk every existing run and classify all illegal moves we've collected.

Gives a count by category per (model, eval) cell so we can answer:
  - What kind of errors do these models actually make?
"""
import json
from collections import Counter
from pathlib import Path

from llm_chess_eval.analytics.illegal_taxonomy import classify_illegal_san
import chess

RUNS = Path("C:/Users/igorc/Projects/LLM_Chess_Eval/runs")


def _is_actually_legal(board: chess.Board, san: str) -> bool:
    """Use parse_san — it's tolerant of missing +/# decorations, which is what we want.
    Earlier I was comparing SAN strings directly which generated false illegals."""
    try:
        board.parse_san(san)
        return True
    except Exception:
        return False


def collect_illegals_legality(jsonl: Path):
    """From a legality JSONL: each candidate SAN that's actually illegal, plus the chosen if illegal."""
    out = []
    for line in jsonl.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        fen = r["fen"]
        raw = r.get("raw_response") or {}
        cands = raw.get("candidates", [])
        board = chess.Board(fen)
        for c in cands:
            san = c.get("san")
            if san and not _is_actually_legal(board, san):
                out.append((fen, san, r.get("position_id"), "candidate"))
        chosen = raw.get("chosen_move")
        if chosen and not _is_actually_legal(board, chosen):
            out.append((fen, chosen, r.get("position_id"), "chosen"))
    return out


def collect_illegals_games(jsonl: Path):
    """From a games JSONL: chosen_san on each ply that ended up illegal (or had failed attempts)."""
    out = []
    for line in jsonl.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        g = json.loads(line)
        for m in g["moves"]:
            for failed_san in m.get("failed_attempts", []):
                out.append((m["fen_before"], failed_san, g["game_id"], "retry_attempt"))
            if not m["chosen_legal"] and m.get("chosen_san"):
                out.append((m["fen_before"], m["chosen_san"], g["game_id"], "terminal"))
    return out


def main():
    by_source: dict[str, Counter] = {}
    examples: dict[str, list[tuple[str, str, str]]] = {}

    for run_dir in sorted(RUNS.iterdir()):
        if not run_dir.is_dir():
            continue
        name = run_dir.name
        legality = run_dir / "legality.jsonl"
        games = run_dir / "games.jsonl"
        if legality.is_file():
            items = collect_illegals_legality(legality)
            key = f"legality :: {name}"
            counter = Counter()
            ex = []
            for fen, san, pid, kind in items:
                cat, _ = classify_illegal_san(chess.Board(fen), san)
                counter[cat] += 1
                if len(ex) < 5:
                    ex.append((pid or "?", san, cat))
            if items:
                by_source[key] = counter
                examples[key] = ex
        if games.is_file():
            items = collect_illegals_games(games)
            key = f"games   :: {name}"
            counter = Counter()
            ex = []
            for fen, san, pid, kind in items:
                cat, _ = classify_illegal_san(chess.Board(fen), san)
                counter[cat] += 1
                if len(ex) < 5:
                    ex.append((pid or "?", san, cat))
            if items:
                by_source[key] = counter
                examples[key] = ex

    print("=" * 96)
    print("Illegal-move classification by run")
    print("=" * 96)
    grand = Counter()
    for key in sorted(by_source):
        c = by_source[key]
        total = sum(c.values())
        grand.update(c)
        print(f"\n{key}  (total={total})")
        for cat, n in c.most_common():
            print(f"    {cat:20s} {n:>3}  ({n*100/total:.0f}%)")
        if examples[key]:
            print(f"    examples: " + "  |  ".join(f"{p}:{s}->{cat}" for (p, s, cat) in examples[key]))

    print()
    print("=" * 96)
    print(f"GRAND TOTAL (n={sum(grand.values())} illegal moves across all runs)")
    print("=" * 96)
    for cat, n in grand.most_common():
        print(f"  {cat:20s} {n:>4}  ({n*100/sum(grand.values()):.1f}%)")


if __name__ == "__main__":
    main()
