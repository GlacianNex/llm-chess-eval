"""Stockfish wrapper: synchronous helpers for ground-truth eval and move scoring."""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import chess
import chess.engine

from .config import stockfish_path


@contextmanager
def stockfish(skill_level: int | None = None, threads: int = 1) -> Iterator[chess.engine.SimpleEngine]:
    """Open a Stockfish UCI engine. Use as a context manager.

    skill_level: 0-20 if provided, else full strength.
    """
    engine = chess.engine.SimpleEngine.popen_uci(str(stockfish_path()))
    try:
        opts: dict[str, int | str] = {"Threads": threads}
        if skill_level is not None:
            if not 0 <= skill_level <= 20:
                raise ValueError(f"skill_level must be 0..20, got {skill_level}")
            opts["Skill Level"] = skill_level
        engine.configure(opts)
        yield engine
    finally:
        engine.quit()


def evaluate_cp(board: chess.Board, depth: int = 16) -> int:
    """Return Stockfish's eval in centipawns from side-to-move's perspective.

    Mate scores are mapped to ±100000 (clamped) — callers comparing magnitudes
    should treat |cp| >= 10000 as "decisive".
    """
    with stockfish() as eng:
        info = eng.analyse(board, chess.engine.Limit(depth=depth))
        score = info["score"].pov(board.turn)
        return score.score(mate_score=100000)


def best_move(board: chess.Board, depth: int = 16) -> chess.Move:
    with stockfish() as eng:
        result = eng.play(board, chess.engine.Limit(depth=depth))
        if result.move is None:
            raise RuntimeError("Stockfish returned no move")
        return result.move


def analyse_pov(eng: chess.engine.SimpleEngine, board: chess.Board, depth: int) -> int:
    """Return cp from side-to-move's POV using an open engine session. 0 at game over."""
    if board.is_game_over():
        result = board.result()
        if result == "1-0":
            return 100000 if board.turn == chess.WHITE else -100000
        if result == "0-1":
            return 100000 if board.turn == chess.BLACK else -100000
        return 0
    info = eng.analyse(board, chess.engine.Limit(depth=depth))
    return info["score"].pov(board.turn).score(mate_score=100000)
