"""Play N seeded games per policy without a UI and print a comparison table.

    python eval/eval_headless.py --games 10 --policy teacher,random,laya-base,laya-tetris
"""
import argparse
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tetris.game import SHAPES, Game  # noqa: E402
from tetris.policy import make_policy  # noqa: E402
from tetris.teacher import all_scores, teacher_move  # noqa: E402


def run(policy, games: int, mask: bool, max_pieces: int):
    lines, pieces, ends, lat, overrides, n = [], [], {}, [], 0, 0
    agree = agree_turn = agree_col = 0
    for seed in range(games):
        g = Game(seed=10_000 + seed, max_pieces=max_pieces)  # seeds disjoint from training data
        while not g.done:
            scores = all_scores(g)
            bt, bc = teacher_move(g, scores)
            d = policy.decide(g, mask=mask)
            turn = d["raw_turn"] % len(SHAPES[g.piece])
            # column agreement is judged against the teacher's best column for the turn Laya picked
            best_col = max((c for t, c in scores if t == turn), key=lambda c: scores[(turn, c)])
            agree_turn += turn == bt
            agree_col += d["raw_col"] == best_col
            agree += (turn, d["raw_col"]) == (bt, bc)
            overrides += d["masked"]
            lat.append(d["latency_ms"])
            n += 1
            g.place(d["turn"], d["col"])
        lines.append(g.lines)
        pieces.append(g.pieces)
        ends[g.end_cause] = ends.get(g.end_cause, 0) + 1
        print("  game %d: %d lines, %d pieces, %s" % (seed + 1, g.lines, g.pieces, g.end_cause), flush=True)
    lat.sort()
    return {
        "policy": policy.name, "device": policy.device,
        "avg lines": statistics.mean(lines), "best": max(lines), "avg pieces": statistics.mean(pieces),
        "agree%": 100 * agree / n, "turn%": 100 * agree_turn / n, "col%": 100 * agree_col / n,
        "mask/100": 100 * overrides / n,
        "p50ms": lat[len(lat) // 2], "p95ms": lat[int(len(lat) * 0.95)],
        "ends": " ".join("%s:%d" % kv for kv in sorted(ends.items())),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=10)
    ap.add_argument("--policy", default="teacher,random,laya-base")
    ap.add_argument("--no-mask", action="store_true", help="raw model play (no safety override)")
    ap.add_argument("--max-pieces", type=int, default=1000, help="game ends here (at most 400 lines)")
    a = ap.parse_args()
    rows = []
    for name in a.policy.split(","):
        print("running %s ..." % name, flush=True)
        rows.append(run(make_policy(name), a.games, not a.no_mask, a.max_pieces))
    cols = list(rows[0])
    print("\n| " + " | ".join(cols) + " |\n|" + "---|" * len(cols))
    for r in rows:
        print("| " + " | ".join(("%.1f" % r[c]) if isinstance(r[c], float) else str(r[c]) for c in cols) + " |")


if __name__ == "__main__":
    main()
