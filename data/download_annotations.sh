#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-python3}"
"${PYTHON_BIN}" - <<'PY'
from src.data import ensure_annotations
ensure_annotations("data/annotations")
print("Annotations are ready in data/annotations")
PY

