"""Shared data types: model responses, eval results, position records."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field


class MoveClaims(BaseModel):
    """Rule-based facts the model asserts about a candidate move. All verifiable
    deterministically with python-chess; none are judgment calls."""
    is_check: bool
    is_capture: bool
    captured_piece: str | None = None  # "P"/"N"/"B"/"R"/"Q" or None
    is_castle: bool
    is_promotion: bool
    is_en_passant: bool
    gives_mate: bool


class CandidateMove(BaseModel):
    san: str
    rationale: str
    claims: MoveClaims


class MoveResponse(BaseModel):
    """Structured response produced by an LLM via tool-use."""
    position_summary: str
    candidates: list[CandidateMove]
    chosen_move: str


class EvalResultRow(BaseModel):
    """One row of eval output. JSONL-serializable."""
    run_id: str
    eval_name: Literal["legality", "consistency", "puzzles", "elo", "games"]
    model: str
    position_id: str
    fen: str
    score: float
    sub_scores: dict[str, float] = Field(default_factory=dict)
    raw_response: dict | None = None
    error: str | None = None
    latency_ms: int | None = None
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class MoveRecord(BaseModel):
    """One LLM move within a played game. Every move's data is logged for analytics."""
    ply: int                       # 1-indexed move number for the LLM (not full-move-number)
    fen_before: str
    sf_best_san: str               # Stockfish's pick at the search depth used
    cp_before: int                 # eval before LLM moves, from LLM's POV
    chosen_san: str | None         # what the model said on its FINAL attempt (after any retries)
    chosen_legal: bool             # whether the FINAL attempt was legal
    chose_in_candidates: bool
    candidates_legal_rate: float
    claim_consistency: float       # rule-claim accuracy for the final candidate
    actual_played_san: str | None  # actually pushed to the board (None if forfeit/dead end)
    chosen_was_top: bool
    cp_after: int | None           # eval AFTER what was actually played, from LLM's POV
    cp_loss: int                   # cp_before - cp_after; MATE_CLAMP if forfeited
    latency_ms: int                # total latency across all attempts on this ply
    model_error: str | None = None
    raw_response: dict | None = None
    # Mode-specific fields
    retries_used: int = 0          # how many illegal attempts before a legal one (0 if first try ok)
    failed_attempts: list[str] = Field(default_factory=list)  # SAN strings the model tried before giving up
    fallback_used: bool = False    # True if we substituted Stockfish-best for the LLM's illegal move


class GameRecord(BaseModel):
    """A full played game."""
    game_id: str
    model: str
    color: Literal["white", "black"]
    opponent: Literal["stockfish"]
    skill: int                      # Stockfish skill level 0..20
    sf_depth: int                   # depth used for ground-truth evaluation
    starting_fen: str
    moves: list[MoveRecord]
    final_fen: str
    result: Literal["1-0", "0-1", "1/2-1/2", "forfeit_illegal", "max_plies"]
    model_won: bool                 # convenience
    n_plies: int                    # how many LLM moves were played
    n_illegal: int


class PositionRecord(BaseModel):
    """One position in a bank (legality, consistency, puzzles)."""
    position_id: str
    fen: str
    source: str
    notes: str | None = None
    # Puzzle-specific (None for non-puzzle positions)
    expected_moves: list[str] | None = None
    rating: int | None = None
    themes: list[str] | None = None
