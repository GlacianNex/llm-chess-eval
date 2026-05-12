"""Unit test for ground_truth_claims: known moves on known positions."""
import chess

from llm_chess_eval.evals.consistency import ground_truth_claims


def check(fen: str, san: str, expected: dict) -> None:
    board = chess.Board(fen)
    move = board.parse_san(san)
    got = ground_truth_claims(board, move).model_dump()
    ok = all(got[k] == v for k, v in expected.items())
    flag = "OK" if ok else "FAIL"
    print(f"[{flag}] {fen[:30]:<30} {san:8s}  expected={expected}")
    if not ok:
        for k, v in expected.items():
            if got[k] != v:
                print(f"      {k}: expected {v!r}, got {got[k]!r}")


# 1. Plain quiet move
check(
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "Nf3",
    {"is_check": False, "is_capture": False, "captured_piece": None,
     "is_castle": False, "is_promotion": False, "is_en_passant": False, "gives_mate": False},
)

# 2. Capture
check(
    "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
    "exd5",
    {"is_capture": True, "captured_piece": "P", "is_check": False, "gives_mate": False},
)

# 3. Castle kingside
check(
    "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQK2R w KQkq - 4 5",
    "O-O",
    {"is_castle": True, "is_capture": False, "is_check": False, "gives_mate": False},
)

# 4. Promotion
check(
    "8/3P1k2/8/8/8/8/5K2/8 w - - 0 1",
    "d8=Q",
    {"is_promotion": True, "is_check": False, "is_capture": False, "gives_mate": False},
)

# (5 removed: invalid position causing python-chess to reject the move)

# 6. En passant — white captures black pawn that just moved d7-d5
check(
    "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",
    "exf6",
    {"is_capture": True, "is_en_passant": True, "captured_piece": "P", "is_check": False},
)

# 7. Check
check(
    "rnbqkbnr/ppp2ppp/3p4/4p3/2B1P3/8/PPPP1PPP/RNBQK1NR w KQkq - 0 3",
    "Bxf7+",
    {"is_capture": True, "captured_piece": "P", "is_check": True, "gives_mate": False},
)

# 8. Fool's mate — true mate-in-1 by black after 1. f3 e5 2. g4
check(
    "rnbqkbnr/pppp1ppp/8/4p3/6P1/5P2/PPPPP2P/RNBQKBNR b KQkq g3 0 2",
    "Qh4#",
    # is_check is the "check but not mate" flag, so it is False when gives_mate is True.
    {"gives_mate": True, "is_check": False, "is_capture": False, "is_castle": False},
)
