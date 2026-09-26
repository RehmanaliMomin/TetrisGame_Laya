"""N-ply search over placements, and the measurement that says 2 plies is not worth it here.

Using the next piece means searching it too: value a placement by the best the board can be once
the next piece has also landed. Measured over 3 games of 2,500 pieces:

| teacher        | lines | Tetrises |
|----------------|-------|----------|
| eltetris 1-ply |   998 |      0.3 |
| tetris 1-ply   |   994 |     91.7 |
| tetris 2-ply   |   994 |     41.0 |

Two plies more than halves the Tetrises. The second ply maximises the board score after the next
piece, and filling the open well scores well right now, so the lookahead keeps cashing in the well
the first ply was holding open. Making that work needs a search that values the plan, not a deeper
greedy step, so the shipped teacher is 1-ply and the model never reads the next piece.
"""
from typing import Dict, Optional, Tuple

from .game import SHAPES, W, Game, column_heights, drop
from .teacher import OBJECTIVES, TOPPED


def placements(piece: str):
    for t, s in enumerate(SHAPES[piece]):
        for c in range(W - (max(x for x, _ in s) + 1) + 1):
            yield t, c


def all_scores_n(g: Game, plies: int = 1, objective: str = "eltetris",
                 next_piece: Optional[str] = None) -> Dict[Tuple[int, int], float]:
    score = OBJECTIVES[objective]
    hs = column_heights(g.rows)
    nxt = next_piece or g.next
    out = {}
    for t, c in placements(g.piece):
        base = score(g.rows, hs, g.piece, t, c)
        if plies < 2 or base <= TOPPED:
            out[(t, c)] = base
            continue
        rows2, _, _, topped = drop(g.rows, SHAPES[g.piece][t], c, hs)
        if topped:
            out[(t, c)] = TOPPED
            continue
        hs2 = column_heights(rows2)
        best = max((score(rows2, hs2, nxt, t2, c2) for t2, c2 in placements(nxt)), default=TOPPED)
        out[(t, c)] = best if best > TOPPED else base
    return out
