"""Generate teacher-labelled Tetris states for fine-tuning the Laya copy.

    python training/make_dataset.py                          # teacher rollouts -> data/tetris_{train,val}.jsonl
    python training/make_dataset.py --rollout laya-tetris \\
        --out data/tetris_dagger                             # DAgger: states the fine-tuned model visits

Each visited piece gives one `turn` row plus `column` rows (the teacher's turn, and one other turn so
the model can still place the piece well after picking a different turn).
Each line: {"q": "turn" | "column", "state": <text>, "target": [probs], "label": idx}
"""
import argparse
import json
import os
import random
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tetris.encode import encode_column_state, encode_turn_state  # noqa: E402
from tetris.game import FULL, H, SHAPES, Game  # noqa: E402
from tetris.teacher import all_scores, column_target, teacher_move, turn_target  # noqa: E402


def add_garbage(g: Game, rng: random.Random, rows: int):
    """Messy starting stack (one gap per row plus random extra holes) for off-teacher states."""
    for i in range(rows):
        y = H - 1 - i
        r = FULL & ~(1 << rng.randrange(10))
        for _ in range(rng.randrange(3)):
            r &= ~(1 << rng.randrange(10))
        g.rows[y] = r
        g.colors[y] = ["G" if r >> c & 1 else "." for c in range(10)]


def argmax(xs):
    return max(range(len(xs)), key=xs.__getitem__)


def rollout(seed, rng, eps, keep, policy=None, max_pieces=600):
    g = Game(seed=seed, max_pieces=max_pieces)
    if rng.random() < 0.5:
        add_garbage(g, rng, rng.randrange(1, 11))
    while not g.done:
        scores = all_scores(g)
        best_turn, best_col = teacher_move(g, scores)
        if rng.random() < keep:
            rows = []
            t = turn_target(g, scores)
            rows.append(("turn", encode_turn_state(g), t))
            turns = [best_turn]
            others = [x for x in range(len(SHAPES[g.piece])) if x != best_turn]
            if others:
                turns.append(rng.choice(others))
            for tt in turns:
                rows.append(("column", encode_column_state(g, tt), column_target(g, tt, scores)))
            yield rows
        if policy is not None:
            d = policy.decide(g)
            turn, col = d["turn"], d["col"]
        elif rng.random() < eps:
            turn, col = rng.choice(list(scores))
        else:
            turn, col = best_turn, best_col
        g.place(turn, col)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pieces", type=int, default=60_000, help="recorded pieces (each gives ~2.7 rows)")
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--eps", type=float, default=0.15, help="random placement rate (messier boards)")
    ap.add_argument("--keep", type=float, default=0.25, help="fraction of visited pieces recorded")
    ap.add_argument("--rollout", default="teacher", help="teacher | laya-tetris (DAgger)")
    ap.add_argument("--out", default="data/tetris")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    rng = random.Random(a.seed)
    policy = None
    if a.rollout != "teacher":
        from tetris.policy import make_policy
        policy = make_policy(a.rollout)
        a.keep = max(a.keep, 0.6)

    seen, pieces, game = set(), [], 0
    while len(pieces) < a.pieces:
        for rows in rollout(a.seed * 1_000_000 + game, rng, a.eps, a.keep, policy):
            if rows[0][1] not in seen:
                seen.add(rows[0][1])
                pieces.append((game, rows))
        game += 1
        print("\r%d games, %d pieces" % (game, len(pieces)), end="", flush=True)
    pieces = pieces[:a.pieces]
    print()

    cut = max(gm for gm, _ in pieces) * (1 - a.val_frac)  # split by game: no near-duplicate leakage
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    for split, part in (("train", [p for p in pieces if p[0] < cut]), ("val", [p for p in pieces if p[0] >= cut])):
        n, dist = 0, Counter()
        with open("%s_%s.jsonl" % (a.out, split), "w") as f:
            for _, rows in part:
                for q, state, target in rows:
                    f.write(json.dumps({"q": q, "state": state, "target": target, "label": argmax(target)}) + "\n")
                    dist[q] += 1
                    n += 1
        print("%-5s %6d pieces  %6d rows  %s" % (split, len(part), n, dict(dist)))


if __name__ == "__main__":
    main()
