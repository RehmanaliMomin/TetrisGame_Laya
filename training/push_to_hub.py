"""Upload the fine-tuned checkpoint + a model card to the Hugging Face Hub.

    hf auth login                                   # once, in your own terminal
    python training/push_to_hub.py --repo <user>/laya-tetris

Reads the eval numbers out of models/laya-tetris/rl_agent_config.json, so the card cannot claim
results the checkpoint was not trained to. Pass --dry-run to write the card locally and stop.
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CARD = """---
license: apache-2.0
library_name: laya
base_model: convaiinnovations/laya-multilingual
pipeline_tag: text-classification
tags: [laya, tetris, game-playing, decision-model, imitation-learning, calibrated-decisions]
---

# laya-tetris

A fine-tuned copy of [`laya-multilingual`](https://huggingface.co/convaiinnovations/laya-multilingual)
(mmBERT-base, 322M) that plays Tetris. Every piece costs it two typed `choice` questions:

1. **how to turn the piece** — 4 options (chance: 25%)
2. **which column to drop it in** — 10 options, asked with the turn it just picked (chance: 10%)

No search and no lookahead: the board becomes a line of features, and one forward pass per question
returns a calibrated probability for every option.

Code, the game and the browser UI: **{github}**

## Validation accuracy

{val_rows} held-out boards, from games never trained on.

| question | options | chance | untuned | fine-tuned |
|---|---|---|---|---|
| turn | 4 | 25% | {turn_before} | **{turn_after}** |
| column | 10 | 10% | {col_before} | **{col_after}** |

## Game results

{results}

## Input format

The state is a single line of board features, and the question is fixed. Both must match the
strings in [`tetris/encode.py`]({github}/blob/main/tetris/encode.py) exactly.

```python
import laya

agent = laya.load("{repo}")

state = ("piece: T | turn: flip | shape: width 3, underside 1 0 1 "
         "| heights: 4 4 3 3 5 6 6 4 2 2 | steps: 0 -1 0 +2 +1 0 -2 -2 0 "
         "| holes: 0 0 1 0 0 0 0 0 0 0 | max height: 6")

question = {{"column": {{
    "type": "choice",
    "instructions": ("In which column should the turned piece's leftmost block land "
                     "to keep the stack flat and clear lines?"),
    "criteria": {{"c1": "column 1 (left wall)", "c2": "column 2", "c3": "column 3",
                 "c4": "column 4", "c5": "column 5", "c6": "column 6", "c7": "column 7",
                 "c8": "column 8", "c9": "column 9", "c10": "column 10 (right wall)"}},
}}}}

ans = agent.predict(state, question)["answers"]["column"]
print(ans["choice"], ans["probabilities"])
```

The turn question is the same shape with options `spawn` / `right` / `flip` / `left`.

## Training

* **Teacher**: a script, not a model — it tries all ~34 placements of the piece and scores each with
  Dellacherie's six board features under the published El-Tetris weights. Its `softmax(score / 3)`
  over the legal placements is the training target.
* **Loss**: expected log score of that soft target, a strictly proper scoring rule (the log term of
  Laya's own RLCD reward), so honest probabilities are what minimises it.
* **Data**: {train_rows} rows from teacher rollouts. Half of them start on a garbage stack, and 15%
  of placements are random, so the model also sees the messy boards its own mistakes produce.
  Train and validation are split by game, never by row.
* **Run**: {epochs} epochs, batch {bs}, {steps} steps on {device}. The 197M-parameter token
  embedding table stays frozen — Tetris text uses a few hundred of the 256k tokens.
* Calibration temperatures fitted on validation: {temps}.

## Limits

It was trained on boards the teacher and a noise policy reached. After a mistake of its own it can
find layouts it never saw, and its columns get worse exactly when the stack is already ugly. A DAgger
round (play with the model, relabel every board it visits with the teacher, fine-tune again) is the
fix, and the repo has the command for it.

## Credits

Laya is by Convai Innovations, Apache 2.0; these weights keep that license. The teacher uses the
[El-Tetris](https://imake.ninja/el-tetris-an-improvement-on-pierre-dellacheries-algorithm/) weights
by Islam Ahmed, an improvement on Pierre Dellacherie's algorithm.
"""


def pct(x):
    return "—" if x is None else "%.1f%%" % (100 * x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="e.g. yourname/laya-tetris")
    ap.add_argument("--model", default=os.path.join(ROOT, "models", "laya-tetris"))
    ap.add_argument("--github", default="https://github.com/RehmanaliMomin/TetrisGame_Laya")
    ap.add_argument("--results", default=os.path.join(ROOT, "data", "results.md"),
                    help="markdown table from eval_headless.py; omitted from the card if missing")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    cfg_path = os.path.join(a.model, "rl_agent_config.json")
    if not os.path.exists(cfg_path):
        sys.exit("No checkpoint at %s" % a.model)
    meta = json.load(open(cfg_path)).get("tetris", {})
    if not meta:
        sys.exit("%s has no 'tetris' training metadata: was it written by training/finetune.py?" % cfg_path)

    results = "_Not measured yet._"
    if os.path.exists(a.results):
        results = open(a.results).read().strip()

    card = CARD.format(
        repo=a.repo, github=a.github, results=results,
        val_rows="{:,}".format(meta["val_rows"]), train_rows="{:,}".format(meta["train_rows"]),
        turn_before=pct(meta["val_acc_before"]["turn"]), turn_after=pct(meta["val_acc_after"]["turn"]),
        col_before=pct(meta["val_acc_before"]["column"]), col_after=pct(meta["val_acc_after"]["column"]),
        epochs=meta["epochs"], bs=meta.get("bs", "?"), steps="{:,}".format(meta["steps"]),
        device=meta.get("device", "?"),
        temps=", ".join("%s %.2f" % (k, v) for k, v in meta["temperatures"].items()),
    )
    out = os.path.join(a.model, "README.md")
    with open(out, "w") as f:
        f.write(card)
    print("wrote %s" % out)
    if a.dry_run:
        return

    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(a.repo, repo_type="model", private=a.private, exist_ok=True)
    api.upload_folder(repo_id=a.repo, folder_path=a.model, repo_type="model",
                      commit_message="laya-tetris: Laya fine-tuned to place Tetris pieces")
    print("https://huggingface.co/%s" % a.repo)


if __name__ == "__main__":
    main()
