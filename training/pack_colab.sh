#!/bin/sh
# Bundle the code + dataset Colab needs (no model weights: the notebook pulls those from HF).
set -e
cd "$(dirname "$0")/.."
zip -qr data/tetris_colab.zip tetris training/finetune.py data/tetris_train.jsonl data/tetris_val.jsonl \
    -x '*__pycache__*'
ls -lh data/tetris_colab.zip
