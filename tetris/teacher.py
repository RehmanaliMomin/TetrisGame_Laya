"""El-Tetris teacher: labels every state for fine-tuning and scores Laya's agreement at play time.

One-piece search over every (turn, column) placement, scored with Pierre Dellacherie's six board
features and the El-Tetris weights (Islam, 2011). It is a script, not a model: it tries all ~34
placements for each piece, while Laya has to pick without trying any of them.
"""
import math
from typing import Dict, List, Tuple

from .game import FULL, H, SHAPES, TURNS, W, Game, column_heights, drop

WEIGHTS = {"landing_height": -4.500158825082766, "eroded_cells": 3.4181268101392694,
           "row_transitions": -3.2178882868487753, "col_transitions": -9.348695305445199,
           "holes": -7.899265427351652, "wells": -3.3855972247263626}
TOPPED = -1e6
TEACHER_TEMP = 3.0  # softness of the training targets: exp(score / temp) over placements
WALLS = 1 | (1 << (W + 1))


def _popcount(x: int) -> int:
    return bin(x).count("1")


def board_features(rows: List[int]) -> Dict[str, float]:
    row_t = col_t = holes = wells = 0
    covered = 0
    prev = 0  # the space above the board counts as empty
    run = [0] * W
    for r in rows:
        x = (r << 1) | WALLS  # walls count as filled
        row_t += _popcount((x ^ (x >> 1)) & ((1 << (W + 1)) - 1))
        col_t += _popcount(r ^ prev)
        prev = r
        holes += _popcount(covered & ~r & FULL)
        covered |= r
        well = ~r & ((r << 1) | 1) & ((r >> 1) | (1 << (W - 1))) & FULL  # empty, both sides filled
        for c in range(W):
            if well >> c & 1:
                run[c] += 1
                wells += run[c]
            else:
                run[c] = 0
    col_t += _popcount(~prev & FULL)  # the floor counts as filled
    return {"row_transitions": row_t, "col_transitions": col_t, "holes": holes, "wells": wells}


def score_placement(rows: List[int], heights: List[int], piece: str, turn: int, col: int) -> float:
    s = SHAPES[piece][turn]
    new, cells, cleared, topped = drop(rows, s, col, heights)
    if topped:
        return TOPPED
    ys = [H - y for _, y in cells]
    f = board_features(new)
    f["landing_height"] = (max(ys) + min(ys)) / 2 - 1
    f["eroded_cells"] = len(cleared) * sum(1 for _, y in cells if y in cleared)
    return sum(WEIGHTS[k] * v for k, v in f.items())


def all_scores(g: Game) -> Dict[Tuple[int, int], float]:
    """Score of every distinct placement (turn index < number of distinct shapes)."""
    hs = column_heights(g.rows)
    return {(t, c): score_placement(g.rows, hs, g.piece, t, c)
            for t, s in enumerate(SHAPES[g.piece])
            for c in range(W - (max(x for x, _ in s) + 1) + 1)}


def teacher_move(g: Game, scores=None) -> Tuple[int, int]:
    scores = scores or all_scores(g)
    return max(scores, key=lambda k: (scores[k], -k[0], -k[1]))  # ties: fewer turns, further left


def _soft(values: List[float], temp: float) -> List[float]:
    m = max(values)
    if m <= TOPPED:
        return [1.0 / len(values)] * len(values)
    ex = [math.exp((v - m) / temp) if v > TOPPED else 0.0 for v in values]
    z = sum(ex)
    return [e / z for e in ex]


def turn_target(g: Game, scores=None, temp: float = TEACHER_TEMP) -> List[float]:
    """Probability over the 4 turn options. Duplicate turns (a flipped I) get no mass: the model
    should name the canonical turn."""
    scores = scores or all_scores(g)
    n = len(SHAPES[g.piece])
    best = [max(v for (t, _), v in scores.items() if t == ti) for ti in range(n)]
    return _soft(best, temp) + [0.0] * (len(TURNS) - n)


def column_target(g: Game, turn: int, scores=None, temp: float = TEACHER_TEMP) -> List[float]:
    """Probability over the 10 column options for a given turn; columns the piece cannot use get 0."""
    scores = scores or all_scores(g)
    turn %= len(SHAPES[g.piece])
    cols = sorted(c for t, c in scores if t == turn)
    p = _soft([scores[(turn, c)] for c in cols], temp)
    out = [0.0] * W
    for c, v in zip(cols, p):
        out[c] = v
    return out
