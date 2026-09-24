# Tetris × Laya — design + build pipeline

## Context

Tetris where every piece is placed by [Laya](https://huggingface.co/convaiinnovations/laya), a
non-autoregressive decision model: give it a state as text and a typed question, it returns one
calibrated probability per option in a single forward pass (~20 ms), and it never generates text.

This is the second game in the series after [Snake × Laya](https://github.com/Okbatti/SnakeGame_Laya),
and it is deliberately a harder ask:

| | Snake | Tetris |
|---|---|---|
| decisions per turn | 1 | **2** (turn, then column) |
| options | 4 | 4 + **10** |
| chance accuracy | 25% | 25% and **10%** |
| a wrong move costs | usually one tick | a hole that lasts the rest of the game |

Base Laya checkpoints are near-chance zero-shot, so a **copy** of `laya-multilingual` (mmBERT-base,
322M) is fine-tuned on Tetris boards labelled by a scripted teacher. The original checkpoint is
never written to.

## Core idea: one piece = two Laya `choice` questions

Placement Tetris (the standard setup for Tetris AIs): the piece is turned, moved to a column and
hard-dropped in one decision, rather than steered a frame at a time.

1. **Turn** — 4 options (`spawn`, `right`, `flip`, `left`), reading the piece and the skyline.
2. **Column** — 10 options (`c1`…`c10`), reading the same skyline plus *the shape it is now holding*.

The column question is asked with the turn Laya just picked, so the second decision is conditioned
on the first — a two-step decision chain, not two independent guesses.

**State text** (`tetris/encode.py`), features rather than a raw grid, because a text encoder reads
features far better:

```
piece: T | turn: flip | shape: width 3, underside 1 0 1 | heights: 4 4 3 3 5 6 6 4 2 2
  | steps: 0 -1 0 +2 +1 0 -2 -2 0 | holes: 0 0 1 0 0 0 0 0 0 0 | max height: 6
```

`underside` is what makes an overhang visible: a `flip` T has undersides `1 0 1`, so it only sits
flush where the middle column is one lower than its neighbours. Nothing in the text is the *result*
of a placement: Laya never gets to try a move before choosing it.

**Safety mask** (toggleable): if Laya's column would top the stack out or the piece does not fit
there, play its most probable column that survives, and count the override.

## Teacher

`tetris/teacher.py` is a script, not a model: it tries all ~34 placements of the current piece and
scores each with Pierre Dellacherie's six features under the published
[El-Tetris](https://imake.ninja/el-tetris-an-improvement-on-pierre-dellacheries-algorithm/) weights
(landing height, eroded cells, row/column transitions, holes, wells). It clears ~800 lines per 2,000
pieces and takes 0.3 ms per piece.

Its placement scores become **soft targets**: `softmax(score / 3.0)` over the legal placements, so
the model is trained on how much better the best column is than the rest, not just which one won.
The loss is the expected log score of that target — a strictly proper scoring rule, the log term of
Laya's own RLCD reward — so honest probabilities are what minimises it.

Training rows per recorded piece:
* 1 `turn` row.
* 2 `column` rows: one for the teacher's turn, one for a different turn of the same piece, so the
  model can still place a piece well after choosing a turn the teacher would not have.

## Architecture

Python owns the game (a single `tetris/game.py` shared by dataset generation, training, the server
and eval, so train and serve cannot drift). The browser only renders.

```
TetrisGame_Laya/
  tetris/game.py      # 10x20 bitboard, 7-bag randomizer, hard drop, line clears
  tetris/teacher.py   # El-Tetris placement search + soft targets
  tetris/encode.py    # encode_turn_state / encode_column_state, TURN_Q, COLUMN_Q
  tetris/policy.py    # LayaPolicy (2 passes/piece), TeacherPolicy, RandomPolicy; timing + mask
  training/make_dataset.py         # teacher rollouts (+ garbage boards, eps-random) -> jsonl
  training/finetune.py             # MPS / CUDA / CPU; frozen embeddings by default
  training/laya_tetris_colab.ipynb # same run on a free Colab T4
  eval/eval_headless.py            # N games per policy -> comparison table
  server.py, ui/                   # 127.0.0.1:7871, canvas board + metrics panel
  tests/                           # pytest: game, teacher, encode
```

Server API: `POST /api/reset {policy, seed}`, `POST /api/step {mask}` → board, both probability
vectors, the teacher's placement, latency per pass, running stats. `GET /api/policies`.

## UI

* Board 10×20, next-piece box, lines / score / level.
* **Column probabilities are drawn as a strip directly under the board**, one bar per column, so
  Laya's distribution lines up with the place the piece would land.
* Turn probabilities as 4 bars, each with a small drawing of the piece in that turn; unusable turns
  (a flipped I is the same I) are dimmed.
* The teacher's placement is outlined on the board whenever it differs from what Laya played.
* Metrics panel: game stats, latency (last / avg / p95, and the two passes separately), confidence,
  teacher agreement, mask overrides.
* Both state texts are printed under the board, so what Laya read is always visible.

## Pipeline

| Step | Command |
|---|---|
| 1 Tests | `python -m pytest` |
| 2 Baseline | `python eval/eval_headless.py --policy teacher,random,laya-base` |
| 3 Dataset | `python training/make_dataset.py` |
| 4 Fine-tune | `python training/finetune.py --base models/laya-base --out models/laya-tetris` |
| 5 Eval | `python eval/eval_headless.py --games 10 --policy laya-tetris` |
| 6 Play | `python server.py` |
| 7 Optional DAgger | `make_dataset.py --rollout laya-tetris --out data/tetris_dagger`, fine-tune again |

## Notes from the build

* **Freeze the token embedding table.** It is 197M of the 322M parameters, and Tetris text uses a
  few hundred of the 256k tokens. Freezing it saves about 3 GB and changes nothing that matters.
  `--train-embeddings` turns it back on.
* **Split the dataset by game, not by row.** Consecutive boards in one game are near-duplicates, so
  a row-wise split leaks the validation set into training.
* **Half the rollouts start on a garbage stack.** Teacher games stay flat and tidy, so a model
  trained only on them never sees the messy boards its own mistakes produce.

## Verification

* `pytest` green (16 tests: rotations, hard drop, line clears, holes, top-out, seeded bag, teacher
  beats random by >5×, every scored placement legal, encoder output exact and deterministic).
* `eval/eval_headless.py` on seeds disjoint from the training data.
* The training sequence is asserted identical to `laya.common.build_sequence`'s inference layout on
  the first 50 rows of every run, so training and play cannot silently diverge.
