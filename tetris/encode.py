"""Game state -> the exact text + questions Laya sees. Shared by training and play so they never drift.

Every piece is two Laya decisions:
  1. TURN_Q   : how to turn the piece        (4 options), reading the skyline
  2. COLUMN_Q : where to drop the turned piece (10 options), reading the skyline + the turned shape
Neither text contains the result of any placement: Laya never tries a move before picking it.
"""
from typing import List

from .game import TURNS, W, Game, shape, underside, width

TURN_Q = {
    "turn": {
        "type": "choice",
        "instructions": "How should the falling Tetris piece be turned before it drops?",
        "criteria": {"spawn": "keep it as it spawns", "right": "turn it clockwise once",
                     "flip": "turn it upside down", "left": "turn it counter-clockwise once"},
    }
}
COLUMN_Q = {
    "column": {
        "type": "choice",
        "instructions": "In which column should the turned piece's leftmost block land to keep the stack flat and clear lines?",
        "criteria": {"c%d" % (i + 1): "column %d%s" % (i + 1, " (left wall)" if i == 0 else " (right wall)" if i == W - 1 else "")
                     for i in range(W)},
    }
}
TURN_KEYS = list(TURN_Q["turn"]["criteria"])
COLUMN_KEYS = list(COLUMN_Q["column"]["criteria"])
assert TURN_KEYS == TURNS


def _nums(xs: List[int]) -> str:
    return " ".join(str(x) for x in xs)


def _steps(hs: List[int]) -> str:
    return " ".join("%+d" % (b - a) if b != a else "0" for a, b in zip(hs, hs[1:]))


def _board(g: Game) -> List[str]:
    hs = g.heights()
    return ["heights: %s" % _nums(hs), "steps: %s" % _steps(hs), "holes: %s" % _nums(g.holes()),
            "max height: %d" % max(hs)]


def encode_turn_state(g: Game) -> str:
    return " | ".join(["piece: %s" % g.piece] + _board(g))


def encode_column_state(g: Game, turn: int) -> str:
    s = shape(g.piece, turn)
    head = ["piece: %s" % g.piece, "turn: %s" % TURNS[turn % 4],
            "shape: width %d, underside %s" % (width(s), _nums(underside(s)))]
    return " | ".join(head + _board(g))
