"""Policies that place the next piece. Every decide() returns the same Decision shape for the UI/eval."""
import os
import random
import time
from typing import Dict, List, Optional

from .encode import COLUMN_KEYS, COLUMN_Q, TURN_KEYS, TURN_Q, encode_column_state, encode_turn_state
from .game import SHAPES, W, Game
from .teacher import all_scores, column_target, teacher_move, turn_target

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIRS = {
    "laya-tetris": os.path.join(ROOT, "models", "laya-tetris"),          # taught by the El-Tetris teacher
    "laya-tetris-4": os.path.join(ROOT, "models", "laya-tetris-4"),      # taught to go for 4-line clears
    "laya-base": os.path.join(ROOT, "models", "laya-base"),              # untouched laya-multilingual
}


def decision(turn: int, col: int, turn_probs: List[float], col_probs: List[float], confidence: float,
             latency_ms: float, raw_turn: Optional[int] = None, raw_col: Optional[int] = None,
             masked: bool = False, passes_ms: Optional[List[float]] = None) -> Dict:
    return {"turn": turn, "col": col, "raw_turn": turn if raw_turn is None else raw_turn,
            "raw_col": col if raw_col is None else raw_col, "turn_probs": turn_probs, "col_probs": col_probs,
            "confidence": confidence, "latency_ms": latency_ms, "passes_ms": passes_ms or [], "masked": masked}


def apply_mask(g: Game, turn: int, col_probs: List[float]) -> Optional[int]:
    """Most probable column that fits and does not top out, or None if every one does."""
    for c in sorted(g.valid_cols(turn), key=lambda c: -col_probs[c]):
        if not g.tops_out(turn, c):
            return c
    return None


class TeacherPolicy:
    device = "cpu"

    def __init__(self, objective: str = "eltetris"):
        self.objective = objective
        self.name = "teacher" if objective == "eltetris" else "teacher-" + objective

    def decide(self, g: Game, mask: bool = True) -> Dict:
        t0 = time.perf_counter()
        scores = all_scores(g, self.objective)
        turn, col = teacher_move(g, scores)
        tp, cp = turn_target(g, scores), column_target(g, turn, scores)
        return decision(turn, col, tp, cp, 1.0, (time.perf_counter() - t0) * 1000)


class RandomPolicy:
    name, device = "random", "cpu"

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def decide(self, g: Game, mask: bool = True) -> Dict:
        turn = self.rng.randrange(4)
        raw = self.rng.randrange(W)
        col, masked = raw, False
        if mask and (not g.is_valid(turn, raw) or g.tops_out(turn, raw)):
            safe = apply_mask(g, turn, [self.rng.random() for _ in range(W)])
            if safe is not None:
                col, masked = safe, True
        if not g.is_valid(turn, col):
            col = min(col, max(g.valid_cols(turn)))
        return decision(turn, col, [0.25] * 4, [0.1] * W, 0.0, 0.0, turn, raw, masked)


class LayaPolicy:
    """Two forward passes per piece: turn, then column given that turn."""

    def __init__(self, name: str, path: Optional[str] = None, device: Optional[str] = None):
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        import laya  # heavy import, only when a Laya policy is actually used
        self.name = name
        self.path = path or MODEL_DIRS[name]
        if not os.path.exists(os.path.join(self.path, "model.safetensors")):
            raise FileNotFoundError("No checkpoint at %s (train it: see README.md)" % self.path)
        self.agent = laya.load(self.path, device=device)
        self.device = str(self.agent.device)
        for _ in range(2):
            self.decide(Game(seed=0))  # warm-up: first MPS calls compile kernels

    def _ask(self, state: str, q: Dict, keys: List[str]):
        t0 = time.perf_counter()
        ans = next(iter(self.agent.predict(state, q)["answers"].values()))
        ms = (time.perf_counter() - t0) * 1000
        return [float(ans["probabilities"][k]) for k in keys], keys.index(ans["choice"]), float(ans["confidence"]), ms

    def decide(self, g: Game, mask: bool = True) -> Dict:
        tp, raw_turn, tconf, t1 = self._ask(encode_turn_state(g), TURN_Q, TURN_KEYS)
        turn = raw_turn % len(SHAPES[g.piece])
        cp, raw_col, cconf, t2 = self._ask(encode_column_state(g, turn), COLUMN_Q, COLUMN_KEYS)
        col, masked = raw_col, False
        bad = not g.is_valid(turn, raw_col) or g.tops_out(turn, raw_col)
        if bad and mask:
            safe = apply_mask(g, turn, cp)
            if safe is not None:
                col, masked = safe, True
        if not g.is_valid(turn, col):  # unmasked play still has to put the piece somewhere on the board
            col = max(g.valid_cols(turn))
        return decision(turn, col, tp, cp, tconf * cconf, t1 + t2, raw_turn, raw_col, masked, [t1, t2])


def make_policy(name: str):
    if name == "teacher":
        return TeacherPolicy()
    if name == "teacher-tetris":
        return TeacherPolicy("tetris")
    if name == "random":
        return RandomPolicy()
    return LayaPolicy(name)
