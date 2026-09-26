"""Tetris x Laya: local game server.

    python server.py            -> http://127.0.0.1:7871

Python owns the game (tetris/game.py); the browser only draws frames and the metrics panel.
Listens on 127.0.0.1 only; models load from ./models with Hugging Face offline mode on.
"""
import json
import os
import threading
import warnings
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

os.environ["HF_HUB_OFFLINE"] = "1"
warnings.filterwarnings("ignore")

from tetris.encode import encode_column_state, encode_turn_state  # noqa: E402
from tetris.game import SHAPES, Game, drop, shape  # noqa: E402
from tetris.policy import MODEL_DIRS, make_policy  # noqa: E402
from tetris.teacher import all_scores, teacher_move  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
HOST, PORT = "127.0.0.1", int(os.environ.get("PORT", 7871))
POLICIES = ["laya-tetris-4", "laya-tetris", "laya-base", "teacher", "teacher-tetris", "random"]
STATIC = {"/": ("index.html", "text/html"), "/style.css": ("style.css", "text/css"),
          "/app.js": ("app.js", "application/javascript")}


def available(name):
    return name not in MODEL_DIRS or os.path.exists(os.path.join(MODEL_DIRS[name], "model.safetensors"))


class Session:
    """One game + running stats for the current policy. Switching policy clears the stats."""

    def __init__(self):
        self.policies = {}
        self.policy_name = None
        self.game = None
        self.clear_stats()

    def clear_stats(self):
        self.lines, self.ends = [], {}
        self.latencies, self.moves, self.agree, self.overrides = [], 0, 0, 0

    def policy(self):
        if self.policy_name not in self.policies:
            self.policies[self.policy_name] = make_policy(self.policy_name)
        return self.policies[self.policy_name]

    def reset(self, policy=None, seed=None):
        policy = policy or self.policy_name or next(p for p in POLICIES if available(p))
        if policy not in POLICIES:
            raise ValueError("Unknown policy %r" % policy)
        if not available(policy):
            raise ValueError("%s is not trained yet: see README.md (models/%s/)" % (policy, policy))
        if policy != self.policy_name:
            self.clear_stats()
        self.policy_name = policy
        self.policy()  # load now so the first step is not a 10 s stall
        self.game = Game(seed=seed)
        return self.frame(None)

    def step(self, mask=True):
        g = self.game
        if g is None or g.done:
            raise ValueError("No running game: reset first")
        states = {"turn": encode_turn_state(g)}
        d = self.policy().decide(g, mask=mask)
        states["column"] = encode_column_state(g, d["turn"])
        scores = all_scores(g)
        bt, bc = teacher_move(g, scores)
        d["teacher"] = {"turn": bt, "col": bc, "cells": drop(g.rows, shape(g.piece, bt), bc)[1]}
        d["piece"] = g.piece
        d["shape"] = [list(c) for c in SHAPES[g.piece][d["turn"] % len(SHAPES[g.piece])]]
        self.moves += 1
        self.agree += (d["raw_turn"] % len(SHAPES[g.piece]), d["raw_col"]) == (bt, bc)
        self.overrides += d["masked"]
        self.latencies = (self.latencies + [d["latency_ms"]])[-500:]
        g.place(d["turn"], d["col"])
        if g.done:
            self.lines.append(g.lines)
            self.ends[g.end_cause] = self.ends.get(g.end_cause, 0) + 1
        d["states"] = states
        return self.frame(d)

    def stats(self):
        lat = sorted(self.latencies)
        cur = self.game.lines if self.game else 0
        return {
            "policy": self.policy_name, "device": getattr(self.policy(), "device", "cpu"),
            "games": len(self.lines), "high": max(self.lines + [cur]),
            "avg": sum(self.lines) / len(self.lines) if self.lines else None,
            "moves": self.moves, "agree": self.agree / self.moves if self.moves else None,
            "overrides": self.overrides, "ends": self.ends,
            "lat_last": self.latencies[-1] if lat else None,
            "lat_avg": sum(lat) / len(lat) if lat else None,
            "lat_p95": lat[int(len(lat) * 0.95)] if lat else None,
        }

    def frame(self, decision):
        return {"game": self.game.to_dict(), "decision": decision, "stats": self.stats()}


session, lock = Session(), threading.Lock()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype + "; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path in STATIC:
            name, ctype = STATIC[self.path]
            with open(os.path.join(HERE, "ui", name), "rb") as f:
                return self._send(200, f.read(), ctype)
        if self.path == "/api/policies":
            return self._send(200, [{"name": p, "available": available(p)} for p in POLICIES])
        self._send(404, {"error": "Not found"})

    def do_POST(self):
        try:
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            with lock:
                if self.path == "/api/reset":
                    return self._send(200, session.reset(req.get("policy"), req.get("seed")))
                if self.path == "/api/step":
                    return self._send(200, session.step(bool(req.get("mask", True))))
            self._send(404, {"error": "Not found"})
        except (ValueError, KeyError, TypeError, FileNotFoundError) as e:
            self._send(400, {"error": str(e)})
        except Exception as e:  # surface anything else to the page instead of hanging
            self._send(500, {"error": "%s: %s" % (type(e).__name__, e)})


def main():
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print("Tetris x Laya running at http://%s:%d  (Ctrl+C to stop)" % (HOST, PORT), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
