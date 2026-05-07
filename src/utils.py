from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .data import Example


def save_json(path: str | Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))


def prediction_rows(examples: list[Example], scores: np.ndarray, include_label: bool) -> list[dict]:
    rows = []
    for ex, score in zip(examples, scores):
        row = {"question_id": ex.question_id, "single_grounding": float(score)}
        if include_label:
            row["label"] = int(ex.label)
            row["source"] = ex.source
        rows.append(row)
    return rows


def standardize(train_x: np.ndarray, val_x: np.ndarray, test_x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = train_x.mean(axis=0, keepdims=True)
    sigma = train_x.std(axis=0, keepdims=True)
    sigma[sigma < 1e-6] = 1.0
    return (
        ((train_x - mu) / sigma).astype(np.float32),
        ((val_x - mu) / sigma).astype(np.float32),
        ((test_x - mu) / sigma).astype(np.float32),
    )


def balanced_weights(y: np.ndarray) -> np.ndarray:
    positives = max(float((y == 1).sum()), 1.0)
    negatives = max(float((y == 0).sum()), 1.0)
    return np.where(y == 1, len(y) / (2.0 * positives), len(y) / (2.0 * negatives)).astype(np.float32)

