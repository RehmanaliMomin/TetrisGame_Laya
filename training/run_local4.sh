#!/bin/sh
# Same as run_local.sh, for the Tetris-seeking teacher's dataset.
set -e
cd "$(dirname "$0")/.."
LOG=data/train4.log
for attempt in 1 2 3 4 5 6 7 8 9 10; do
  echo "=== attempt $attempt $(date +%H:%M:%S)" >> "$LOG"
  if HF_HUB_OFFLINE=1 .venv/bin/python -u training/finetune.py \
        --base models/laya-base --out models/laya-tetris-4 \
        --data data/tetris4_train.jsonl --val data/tetris4_val.jsonl \
        --epochs "${EPOCHS:-1}" --bs "${BS:-32}" --amp "${AMP:-bf16}" \
        --save-every 250 --resume >> "$LOG" 2>&1; then
    echo "=== done" >> "$LOG"; exit 0
  fi
  echo "=== attempt $attempt failed, retrying" >> "$LOG"
  sleep 20
done
echo "=== gave up after 10 attempts" >> "$LOG"; exit 1
