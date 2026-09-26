"""Tetris rules. Training, the server and eval all use this file, so they cannot drift apart.

Placement Tetris (the standard setup for Tetris AIs): each piece is placed with one decision,
a rotation plus the column of its leftmost cell, then hard-dropped straight down.

Board rows are ints (bit c = column c, bit 0 = left wall), row 0 is the top.
"""
import random
from typing import Dict, List, Optional, Tuple

W, H = 10, 20
FULL = (1 << W) - 1
PIECES = "IOTSZJL"
TURNS = ["spawn", "right", "flip", "left"]  # clockwise quarter turns from the spawn orientation
LINE_POINTS = [0, 100, 300, 500, 800]

_SPAWN = {
    "I": [(0, 0), (1, 0), (2, 0), (3, 0)],
    "O": [(0, 0), (1, 0), (0, 1), (1, 1)],
    "T": [(1, 0), (0, 1), (1, 1), (2, 1)],
    "S": [(1, 0), (2, 0), (0, 1), (1, 1)],
    "Z": [(0, 0), (1, 0), (1, 1), (2, 1)],
    "J": [(0, 0), (0, 1), (1, 1), (2, 1)],
    "L": [(2, 0), (0, 1), (1, 1), (2, 1)],
}

Shape = Tuple[Tuple[int, int], ...]


def _normalize(cells) -> Shape:
    mx, my = min(x for x, _ in cells), min(y for _, y in cells)
    return tuple(sorted((x - mx, y - my) for x, y in cells))


def _rotations(cells) -> List[Shape]:
    """Distinct shapes in clockwise order from spawn (O: 1, I/S/Z: 2, T/J/L: 4)."""
    out, cur = [], list(cells)
    for _ in range(4):
        s = _normalize(cur)
        if s not in out:
            out.append(s)
        cur = [(-y, x) for x, y in cur]  # clockwise on a y-down screen
    return out


SHAPES: Dict[str, List[Shape]] = {p: _rotations(c) for p, c in _SPAWN.items()}


def shape(piece: str, turn: int) -> Shape:
    """Any of the 4 turns maps onto a distinct shape (a flipped I is the same I)."""
    rots = SHAPES[piece]
    return rots[turn % len(rots)]


def width(s: Shape) -> int:
    return max(x for x, _ in s) + 1


def underside(s: Shape) -> List[int]:
    """Per column of the shape: how far its lowest cell sits above the shape's lowest point."""
    h = max(y for _, y in s)
    return [h - max(y for x, y in s if x == c) for c in range(width(s))]


def column_heights(rows: List[int]) -> List[int]:
    hs = [0] * W
    for c in range(W):
        bit = 1 << c
        for y, r in enumerate(rows):
            if r & bit:
                hs[c] = H - y
                break
    return hs


def column_holes(rows: List[int]) -> List[int]:
    """Empty cells with a filled cell somewhere above them, per column."""
    holes, covered = [0] * W, 0
    for r in rows:
        empty_under = covered & ~r
        if empty_under:
            for c in range(W):
                if empty_under >> c & 1:
                    holes[c] += 1
        covered |= r
    return holes


def drop(rows: List[int], s: Shape, col: int, heights: Optional[List[int]] = None):
    """Hard-drop shape `s` with its leftmost cell in `col`.

    Returns (new_rows, cells, cleared_row_indices, topped_out). `cells` are (x, y) board coords
    before the clear; y < 0 means the piece stuck out above the ceiling.
    """
    hs = heights or column_heights(rows)
    lows = {}
    for x, y in s:
        lows[x] = max(lows.get(x, -1), y)
    top = min(H - hs[col + x] - 1 - lows[x] for x in lows)  # y offset where the shape rests
    cells = [(col + x, top + y) for x, y in s]
    if any(y < 0 for _, y in cells):
        return rows, cells, [], True
    new = list(rows)
    for x, y in cells:
        new[y] |= 1 << x
    cleared = [y for y in sorted({y for _, y in cells}) if new[y] == FULL]
    if cleared:
        keep = [r for i, r in enumerate(new) if i not in cleared]
        new = [0] * len(cleared) + keep
    return new, cells, cleared, False


class Game:
    def __init__(self, seed: Optional[int] = None, max_pieces: Optional[int] = None):
        self.rng = random.Random(seed)
        self.rows: List[int] = [0] * H
        self.colors: List[List[str]] = [["."] * W for _ in range(H)]
        self.bag: List[str] = []
        self.piece = self._draw()
        self.next = self._draw()
        self.lines = self.score = self.pieces = 0
        self.clears = [0, 0, 0, 0, 0]  # index = lines cleared by one piece
        self.max_pieces = max_pieces
        self.done, self.end_cause = False, None
        self.last: Optional[Dict] = None

    def _draw(self) -> str:
        if not self.bag:  # 7-bag randomizer, like modern Tetris
            self.bag = list(PIECES)
            self.rng.shuffle(self.bag)
        return self.bag.pop()

    @property
    def level(self) -> int:
        return self.lines // 10 + 1

    def heights(self) -> List[int]:
        return column_heights(self.rows)

    def holes(self) -> List[int]:
        return column_holes(self.rows)

    def valid_cols(self, turn: int) -> range:
        return range(W - width(shape(self.piece, turn)) + 1)

    def is_valid(self, turn: int, col: int) -> bool:
        return col in self.valid_cols(turn)

    def tops_out(self, turn: int, col: int) -> bool:
        return drop(self.rows, shape(self.piece, turn), col)[3]

    def place(self, turn: int, col: int) -> int:
        """Drop the current piece. Returns lines cleared."""
        if self.done:
            raise ValueError("game over")
        if not self.is_valid(turn, col):
            raise ValueError("%s turned %s does not fit at column %d" % (self.piece, TURNS[turn % 4], col))
        s = shape(self.piece, turn)
        new, cells, cleared, topped = drop(self.rows, s, col)
        self.pieces += 1
        self.last = {"piece": self.piece, "turn": turn % len(SHAPES[self.piece]), "col": col,
                     "cells": cells, "cleared": cleared}
        if topped:
            self.done, self.end_cause = True, "topped out"
            return 0
        for x, y in cells:
            self.colors[y][x] = self.piece
        if cleared:
            self.colors = [["."] * W for _ in cleared] + [r for i, r in enumerate(self.colors) if i not in cleared]
        self.rows = new
        n = len(cleared)
        self.lines += n
        self.clears[n] += 1
        self.score += LINE_POINTS[n] * self.level
        self.piece, self.next = self.next, self._draw()
        if self.max_pieces and self.pieces >= self.max_pieces:
            self.done, self.end_cause = True, "piece limit"
        return n

    def to_dict(self) -> Dict:
        return {"w": W, "h": H, "board": ["".join(r) for r in self.colors], "piece": self.piece,
                "next": self.next, "lines": self.lines, "score": self.score, "level": self.level,
                "pieces": self.pieces, "clears": self.clears, "done": self.done, "end_cause": self.end_cause,
                "heights": self.heights(), "holes": self.holes(), "last": self.last,
                "shapes": {p: [list(map(list, s)) for s in SHAPES[p]] for p in (self.piece, self.next)}}
