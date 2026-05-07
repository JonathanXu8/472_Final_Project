from __future__ import annotations

import numpy as np


def precision_recall_f1(y_true: np.ndarray, scores: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred = (scores >= threshold).astype(np.int32)
    y_true = y_true.astype(np.int32)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    accuracy = (tp + tn) / max(len(y_true), 1)
    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def tune_threshold(y_true: np.ndarray, scores: np.ndarray) -> dict:
    candidates = np.unique(np.concatenate([np.linspace(0.01, 0.99, 99), scores]))
    best = None
    for threshold in candidates:
        metrics = precision_recall_f1(y_true, scores, float(threshold))
        if best is None or (metrics["f1"], metrics["precision"]) > (best["f1"], best["precision"]):
            best = metrics
    assert best is not None
    return best

