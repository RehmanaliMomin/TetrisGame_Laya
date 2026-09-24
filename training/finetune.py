"""Fine-tune a COPY of a Laya checkpoint to answer the two Tetris questions (turn, column).

    python training/finetune.py --base models/laya-base --out models/laya-tetris \\
        --data data/tetris_train.jsonl --val data/tetris_val.jsonl

Runs on Apple Silicon (MPS), CUDA (fp16 autocast) or CPU; --limit 256 is a quick smoke test.
The base directory is only read; everything is written to --out, in the layout laya.load() expects.

Loss = expected log score of the teacher's soft target (a strictly proper scoring rule: the log term
of Laya's own RLCD reward), so the model is pushed toward honest probabilities, not just argmax.
"""
import argparse
import json
import math
import os
import random
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import laya  # noqa: E402
from laya.common import QTYPES, build_sequence, collate_items, temp_bucket  # noqa: E402
from tetris.encode import COLUMN_Q, TURN_Q  # noqa: E402

QUESTIONS = {"turn": TURN_Q["turn"], "column": COLUMN_Q["column"]}


def load_jsonl(paths, limit=None):
    rows = []
    for path in paths.split(","):
        with open(path) as f:
            for line in f:
                rows.append(json.loads(line))
    random.Random(0).shuffle(rows)
    return rows[:limit] if limit else rows


class Encoder:
    """laya.common.build_sequence's exact layout, with each fixed question tokenized once."""

    def __init__(self, agent):
        self.tok, self.cfg = agent.tok, agent.cfg
        self.q = {k: agent._to_internal(v) for k, v in QUESTIONS.items()}
        self.head = {}
        for k, q in self.q.items():
            ids, markers = build_sequence(self.tok, "", q, self.cfg["max_len"], self.cfg["head_max_len"])
            self.head[k] = (ids[:-1], markers)  # drop the trailing [SEP]: the state goes there

    def encode(self, qkey, state_ids):
        ids, markers = self.head[qkey]
        room = max(0, self.cfg["max_len"] - len(ids) - 1)
        return ids + state_ids[:room] + [self.tok.sep_token_id], markers

    def check(self, qkey, state):
        """Guard against drift from the inference path."""
        ref = build_sequence(self.tok, state, self.q[qkey], self.cfg["max_len"], self.cfg["head_max_len"])
        assert ref == self.encode(qkey, self.tok(state, add_special_tokens=False)["input_ids"]), \
            "training sequence differs from laya's inference sequence"


def make_batch(enc, rows, state_ids, idx, pad_id):
    items = []
    for i in idx:
        ids, markers = enc.encode(rows[i]["q"], state_ids[i])
        items.append({"ids": ids, "markers": list(markers), "qtype": QTYPES["choice"],
                      "target": rows[i]["target"], "label": rows[i]["label"]})
    return collate_items([items], pad_id)


def forward(model, b, device):
    logits, _ = model(b["input_ids"].to(device), b["attention_mask"].to(device), b["marker_pos"].to(device),
                      b["marker_mask"].to(device), b["qtype"].to(device))
    return logits.float()


def soft_ce(logits, target):
    return -(target * F.log_softmax(logits, -1)).sum(-1)


@torch.no_grad()
def evaluate(model, enc, rows, state_ids, device, bs, pad_id, ctx):
    """Per question: mean loss, accuracy, and (logits, labels) for temperature fitting."""
    model.eval()
    out = {}
    for qkey in QUESTIONS:
        sel = [i for i, r in enumerate(rows) if r["q"] == qkey]
        logits, labels, loss = [], [], 0.0
        for s in range(0, len(sel), bs):
            b = make_batch(enc, rows, state_ids, sel[s:s + bs], pad_id)
            with ctx():
                lg = forward(model, b, device)
            loss += soft_ce(lg, b["target"].to(device)).sum().item()
            logits.append(lg.cpu())
            labels.append(b["label"])
        lg, lb = torch.cat(logits), torch.cat(labels)
        out[qkey] = {"loss": loss / max(1, len(sel)), "acc": (lg.argmax(-1) == lb).float().mean().item(),
                     "logits": lg, "labels": lb, "k": len(QUESTIONS[qkey]["criteria"])}
    model.train()
    if device.type == "mps":
        torch.mps.empty_cache()
    return out


def fit_temperature(logits, labels):
    """1-D grid search of the NLL-optimal temperature, kept inside laya's accepted range [0.5, 5]."""
    best = (math.inf, 1.0)
    for t in np.exp(np.linspace(math.log(0.5), math.log(5.0), 60)):
        nll = F.cross_entropy(logits / float(t), labels).item()
        best = min(best, (nll, float(t)))
    return best[1]


def save(model, cfg, base, out, meta):
    from safetensors import safe_open
    from safetensors.torch import save_file
    os.makedirs(out, exist_ok=True)
    with safe_open(os.path.join(base, "model.safetensors"), "pt") as f:  # keep the base's dtypes (fp16 ~ 650 MB)
        dtypes = {k: f.get_slice(k).get_dtype() for k in f.keys()}
    to_torch = {"F16": torch.float16, "BF16": torch.bfloat16, "F32": torch.float32}
    save_file({k: v.detach().cpu().to(to_torch.get(dtypes.get(k), v.dtype)).contiguous()
               for k, v in model.state_dict().items()},
              os.path.join(out, "model.safetensors"))
    for d in ("tokenizer", "encoder"):
        if os.path.isdir(os.path.join(base, d)):
            shutil.copytree(os.path.join(base, d), os.path.join(out, d), dirs_exist_ok=True)
    cfg = dict(cfg, model_name="laya-tetris", tetris=meta)
    with open(os.path.join(out, "rl_agent_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)


def report(tag, ev):
    return "%s: " % tag + " | ".join("%s loss %.3f acc %.3f" % (k, v["loss"], v["acc"]) for k, v in ev.items())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="Laya checkpoint dir to copy from (read only)")
    ap.add_argument("--out", default="models/laya-tetris")
    ap.add_argument("--data", default="data/tetris_train.jsonl", help="comma-separated jsonl files")
    ap.add_argument("--val", default="data/tetris_val.jsonl")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr-enc", type=float, default=3e-5)
    ap.add_argument("--lr-head", type=float, default=2e-4)
    ap.add_argument("--limit", type=int, default=None, help="use only N rows (smoke test)")
    ap.add_argument("--val-limit", type=int, default=6000)
    ap.add_argument("--train-embeddings", action="store_true",
                    help="also train the 197M-param token embedding table (off: saves ~3 GB, Tetris text uses few tokens)")
    ap.add_argument("--amp", choices=["auto", "off", "bf16", "fp16"], default="auto")
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    if os.path.abspath(a.out) == os.path.abspath(a.base):
        sys.exit("--out must differ from --base: the original checkpoint is never overwritten")
    random.seed(a.seed), np.random.seed(a.seed), torch.manual_seed(a.seed)
    rng = random.Random(a.seed)

    agent = laya.load(a.base, device=a.device)
    model, device, cfg = agent.model, agent.device, dict(agent.cfg)
    model.float().train()
    amp = {"auto": "fp16" if device.type == "cuda" else "bf16" if device.type == "mps" else "off"}.get(a.amp, a.amp)
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(amp)

    def ctx():
        if dtype is None:
            return torch.autocast(device_type="cpu", enabled=False)
        return torch.autocast(device_type=device.type, dtype=dtype)

    pad_id = agent.tok.pad_token_id
    print("device %s | amp %s" % (device, amp), flush=True)

    enc = Encoder(agent)
    train = load_jsonl(a.data, a.limit)
    val = load_jsonl(a.val, a.limit and max(64, a.limit // 4) or a.val_limit)
    for r in train[:50]:
        enc.check(r["q"], r["state"])
    t0 = time.time()
    tr_ids = [enc.tok(r["state"], add_special_tokens=False)["input_ids"] for r in train]
    va_ids = [enc.tok(r["state"], add_special_tokens=False)["input_ids"] for r in val]
    print("tokenized %d train / %d val in %.0fs (max state len %d)" %
          (len(train), len(val), time.time() - t0, max(map(len, tr_ids))), flush=True)

    ev0 = evaluate(model, enc, val, va_ids, device, a.bs * 2, pad_id, ctx)
    print(report("before", ev0), flush=True)

    if not a.train_embeddings:
        model.encoder.embeddings.tok_embeddings.weight.requires_grad_(False)
    enc_params = [p for n, p in model.named_parameters() if n.startswith("encoder.") and p.requires_grad]
    head_params = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
    opt = torch.optim.AdamW([{"params": enc_params, "lr": a.lr_enc}, {"params": head_params, "lr": a.lr_head}],
                            weight_decay=0.01)
    steps_per_epoch = math.ceil(len(train) / a.bs)
    total = max(1, int(steps_per_epoch * a.epochs))
    warm = max(1, total // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / warm, max(0.0, (total - s) / max(1, total - warm))))
    scaler = torch.amp.GradScaler("cuda", enabled=amp == "fp16" and device.type == "cuda")

    step, t0, run_loss = 0, time.time(), 0.0
    order = []
    while step < total:
        if not order:
            order = list(range(len(train)))
            rng.shuffle(order)
        idx, order = order[:a.bs], order[a.bs:]
        b = make_batch(enc, train, tr_ids, idx, pad_id)
        with ctx():
            logits = forward(model, b, device)
        loss = soft_ce(logits, b["target"].to(device)).mean()
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        step += 1
        run_loss = 0.98 * run_loss + 0.02 * loss.item() if step > 1 else loss.item()
        if step % 50 == 0 or step == total:
            el = time.time() - t0
            eta = (total - step) * el / step
            print("step %d/%d  loss %.3f  %.2f it/s  ETA %dm%02ds" % (step, total, run_loss, step / el, eta // 60, eta % 60),
                  flush=True)
        if step % steps_per_epoch == 0 and step < total:
            print(report("epoch %d" % (step // steps_per_epoch), evaluate(model, enc, val, va_ids, device, a.bs * 2, pad_id, ctx)),
                  flush=True)

    ev1 = evaluate(model, enc, val, va_ids, device, a.bs * 2, pad_id, ctx)
    temps = {k: fit_temperature(v["logits"], v["labels"]) for k, v in ev1.items()}
    print(report("after", ev1) + " | temperatures %s" % {k: round(t, 3) for k, t in temps.items()}, flush=True)

    by_opts = dict(cfg.get("temperature_by_options", {}))
    for k, v in ev1.items():
        by_opts[temp_bucket(QTYPES["choice"], v["k"])] = temps[k]
    cfg["temperature_by_options"] = by_opts
    meta = {"base": os.path.basename(os.path.normpath(a.base)), "train_rows": len(train), "val_rows": len(val),
            "epochs": a.epochs, "steps": total, "amp": amp, "device": str(device),
            "val_acc_before": {k: v["acc"] for k, v in ev0.items()},
            "val_acc_after": {k: v["acc"] for k, v in ev1.items()},
            "val_loss_after": {k: v["loss"] for k, v in ev1.items()}, "temperatures": temps,
            "minutes": (time.time() - t0) / 60, "questions": QUESTIONS}
    save(model, cfg, a.base, a.out, meta)
    print("saved fine-tuned copy to %s" % a.out, flush=True)


if __name__ == "__main__":
    main()
