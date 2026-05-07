#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-python}"
"${PYTHON_BIN}" -m src.train_clip \
  --data-dir data/annotations \
  --image-dir data/raw_images \
  --out-dir results/clip_mlp \
  --device auto \
  --batch-size 16 \
  --epochs 80

