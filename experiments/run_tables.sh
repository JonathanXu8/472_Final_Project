#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-python}"
"${PYTHON_BIN}" -m src.dataset_stats --data-dir data/annotations --out results/dataset_stats.csv
"${PYTHON_BIN}" -m src.polygon_diagnostic --data-dir data/annotations --out results/polygon_diagnostic.json

