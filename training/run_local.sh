#!/bin/sh
# Fine-tune locally, surviving the MPS driver dropping a command buffer mid-run.
# Each attempt picks up from the last resumable checkpoint (written every --save-every steps).
set -e
cd "$(dirname "$0")/.."
LOG=data/train.log
for attempt in 1 2 3 4 5 6 7 8 9 10; do
  echo "=== attempt $attempt $(date +%H:%M:%S)" >> "$LOG"
  if HF_HUB_OFFLINE=1 .venv/bin/python -u training/finetune.py \
        --base models/laya-base --out models/laya-tetris \
        --data data/tetris_train.jsonl --val data/tetris_val.jsonl \
        --epochs "${EPOCHS:-2}" --bs "${BS:-24}" --amp "${AMP:-off}" \
        --save-every 250 --resume >> "$LOG" 2>&1; then
    echo "=== done" >> "$LOG"; exit 0
  fi
  echo "=== attempt $attempt failed, retrying" >> "$LOG"
  sleep 20
done
echo "=== gave up after 10 attempts" >> "$LOG"; exit 1
