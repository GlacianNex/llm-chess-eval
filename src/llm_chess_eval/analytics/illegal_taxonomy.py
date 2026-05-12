"""Classify why a proposed SAN move is illegal in a given position.

Given a board and a candidate SAN, we try to figure out WHAT failed:
  - parse_error     : SAN can't even be parsed at all (malformed syntax)
  - no_source_piece : The SAN names a piece type that doesn't exist at any
                      legal source square for the moving side
  - phantom_source  : SAN's implied source square is empty (or wrong piece)
  - target_blocked  : Target square has the mover's own piece
  - path_blocked    : (sliding piece) path to target is obstructed
  - wrong_pawn_move : Pawn moving wrong distance, or capturing into empty,
                      or moving non-diagonally to a non-empty square
  - leaves_in_check : Would leave own king in check (pinned or just bad)
  - king_into_check : King moves into an attacked square
  - king_adjacent   : Move puts kings adjacent (forbidden)
  - castle_invalid  : Castling-specific failure (rights gone, in check, etc.)
  - other           : python-chess rejected it but we couldn't pinpoint why
"""
from __future__ import annotations

import re
from typing import Literal

import chess

Category = Literal[
    "parse_error",
    "no_source_piece",
    "phantom_source",
    "target_blocked",
    "path_blocked",
    "wrong_pawn_move",
    "leaves_in_check",
    "king_into_check",
    "king_adjacent",
    "castle_invalid",
    "other",
]

_PIECE_LETTER = re.compile(r"^([KQRBN])")
_SQUARE = re.compile(r"([a-h][1-8])")


def _piece_letter(san: str) -> str | None:
    m = _PIECE_LETTER.match(san.replace("x", ""))
    return m.group(1) if m else None


def classify_illegal_san(board: chess.Board, san: str) -> tuple[Category, dict]:
    """Return (category, details). Details may include 'target', 'piece', etc."""
    clean = san.replace("+", "").replace("#", "").replace("?", "").replace("!", "")
    details: dict = {"san_clean": clean}

    # Try to parse first
    try:
        move = board.parse_san(clean)
        # If it parses, then is_legal check
        if move in board.legal_moves:
            return ("other", {**details, "note": "actually legal — classifier shouldn't have been called"})
    except chess.InvalidMoveError:
        return ("parse_error", details)
    except chess.AmbiguousMoveError:
        return ("parse_error", {**details, "note": "ambiguous SAN"})
    except chess.IllegalMoveError:
        move = None  # fall through to deeper analysis
    except Exception as e:
        return ("parse_error", {**details, "note": f"unexpected {type(e).__name__}"})

    # Castling
    if clean in ("O-O", "O-O-O", "0-0", "0-0-0"):
        details["castle"] = clean
        return ("castle_invalid", details)

    # Pull target square
    sq_matches = _SQUARE.findall(clean)
    target_sq_name = sq_matches[-1] if sq_matches else None
    if target_sq_name:
        target_sq = chess.parse_square(target_sq_name)
        details["target"] = target_sq_name
    else:
        target_sq = None

    piece_letter = _piece_letter(clean)
    moving_color = board.turn

    # Pawn moves: no leading letter
    if piece_letter is None:
        # Possible pawn move. Find candidate source pawns.
        if target_sq is None:
            return ("parse_error", details)
        is_capture = "x" in clean
        # Find any of our pawns that could even consider this destination
        candidate_sources = []
        for src in chess.SQUARES:
            piece = board.piece_at(src)
            if piece is None or piece.color != moving_color or piece.piece_type != chess.PAWN:
                continue
            # Check if a legal pawn move from src lands on target
            for mv in board.pseudo_legal_moves:
                if mv.from_square == src and mv.to_square == target_sq:
                    candidate_sources.append(src)
                    break
        if not candidate_sources:
            # No pawn can even pseudo-legally reach target — wrong-pawn-move OR phantom source
            # If target has own piece → target_blocked
            tgt_piece = board.piece_at(target_sq)
            if tgt_piece and tgt_piece.color == moving_color:
                return ("target_blocked", details)
            if is_capture and tgt_piece is None and not _is_en_passant_square(board, target_sq):
                details["note"] = "capture into empty square"
                return ("wrong_pawn_move", details)
            if not is_capture and tgt_piece is not None:
                details["note"] = "straight pawn push into occupied square"
                return ("wrong_pawn_move", details)
            return ("wrong_pawn_move", details)
        # A pawn could pseudo-legally reach target but it isn't in legal_moves → king-pin / check
        return ("leaves_in_check", details)

    # Piece move
    piece_type_map = {"K": chess.KING, "Q": chess.QUEEN, "R": chess.ROOK,
                      "B": chess.BISHOP, "N": chess.KNIGHT}
    pt = piece_type_map[piece_letter]

    # Find all pieces of the right type and color
    our_pieces_of_type = [
        sq for sq in chess.SQUARES
        if (p := board.piece_at(sq)) is not None and p.color == moving_color and p.piece_type == pt
    ]
    if not our_pieces_of_type:
        return ("no_source_piece", {**details, "piece": piece_letter})

    if target_sq is None:
        return ("parse_error", details)

    # Target occupied by own piece?
    tgt_piece = board.piece_at(target_sq)
    if tgt_piece and tgt_piece.color == moving_color:
        return ("target_blocked", details)

    # Any of our pieces pseudo-legally reach target?
    pseudo_sources = []
    for src in our_pieces_of_type:
        for mv in board.pseudo_legal_moves:
            if mv.from_square == src and mv.to_square == target_sq:
                pseudo_sources.append(src)
                break

    if not pseudo_sources:
        # Piece exists but can't even pseudo-legally reach (path blocked, jumps over, etc.)
        if pt in (chess.QUEEN, chess.ROOK, chess.BISHOP):
            details["piece"] = piece_letter
            return ("path_blocked", details)
        details["piece"] = piece_letter
        return ("phantom_source", details)

    # Piece can pseudo-legally reach but move isn't legal → check / pin issue
    if pt == chess.KING:
        # King move into attacked square or adjacent to enemy king
        attackers = board.attackers(not moving_color, target_sq)
        if attackers:
            # Check if enemy king attacks target
            enemy_king_sq = board.king(not moving_color)
            if enemy_king_sq is not None and chess.square_distance(enemy_king_sq, target_sq) == 1:
                return ("king_adjacent", details)
            return ("king_into_check", details)
        return ("other", {**details, "note": "king move rejected but target not attacked"})

    # Non-king: must be pinning issue (would leave own king in check)
    return ("leaves_in_check", {**details, "piece": piece_letter})


def _is_en_passant_square(board: chess.Board, sq: int) -> bool:
    return board.ep_square is not None and board.ep_square == sq


def summarize_illegals(samples: list[tuple[str, str]]) -> dict[str, int]:
    """Given (fen, san) pairs, classify each and return category counts."""
    counts: dict[str, int] = {}
    for fen, san in samples:
        board = chess.Board(fen)
        cat, _ = classify_illegal_san(board, san)
        counts[cat] = counts.get(cat, 0) + 1
    return counts
