from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

from .data import Example


@lru_cache(maxsize=20000)
def find_image(root: str | Path, image_id: str) -> Path | None:
    root = Path(root)
    candidate = root / image_id
    if candidate.exists():
        return candidate

    stem = Path(image_id).stem
    for suffix in (".jpg", ".jpeg", ".png"):
        direct = root / f"{stem}{suffix}"
        if direct.exists():
            return direct

    matches = list(root.rglob(image_id))
    if matches:
        return matches[0]

    if stem.isdigit():
        padded = stem.zfill(12)
        for suffix in (".jpg", ".jpeg", ".png"):
            matches = list(root.rglob(f"*{padded}*{suffix}"))
            if matches:
                return matches[0]
    return None


def image_coverage(examples: list[Example], image_dir: str | Path, limit: int | None = None) -> tuple[int, int, float]:
    subset = examples if limit is None else examples[:limit]
    found = sum(find_image(image_dir, ex.image_id) is not None for ex in subset)
    return found, len(subset), found / max(len(subset), 1)


def image_grid_features(examples: list[Example], image_dir: str | Path, grid: int = 4) -> np.ndarray:
    rows = np.zeros((len(examples), 9), dtype=np.float32)
    for i, ex in enumerate(examples):
        path = find_image(image_dir, ex.image_id)
        if path is None:
            rows[i, -1] = 1.0
            continue
        arr = np.asarray(Image.open(path).convert("RGB").resize((224, 224)), dtype=np.float32) / 255.0
        patch_h, patch_w = arr.shape[0] // grid, arr.shape[1] // grid
        patch_stats = []
        centers = []
        for gy in range(grid):
            for gx in range(grid):
                patch = arr[gy * patch_h : (gy + 1) * patch_h, gx * patch_w : (gx + 1) * patch_w]
                gray = patch.mean(axis=2)
                contrast = float(gray.std())
                saturation = float((patch.max(axis=2) - patch.min(axis=2)).mean())
                edge = float(np.abs(np.diff(gray, axis=0)).mean() + np.abs(np.diff(gray, axis=1)).mean())
                patch_stats.append((contrast, saturation, edge))
                centers.append(((gx + 0.5) / grid, (gy + 0.5) / grid))
        stat_arr = np.array(patch_stats)
        salience = stat_arr.sum(axis=1)
        weights = np.exp(salience - salience.max())
        weights /= max(weights.sum(), 1e-12)
        sorted_w = np.sort(weights)[::-1]
        entropy = -float(np.sum(weights * np.log(weights + 1e-12)) / np.log(len(weights)))
        top_gap = float(sorted_w[0] - sorted_w[1])
        centers_arr = np.array(centers)
        center = (weights[:, None] * centers_arr).sum(axis=0)
        spatial_var = float((weights * ((centers_arr - center) ** 2).sum(axis=1)).sum())
        rows[i] = [
            entropy,
            top_gap,
            float((weights > 1 / len(weights)).sum()) / len(weights),
            spatial_var,
            stat_arr[:, 0].mean(),
            stat_arr[:, 1].mean(),
            stat_arr[:, 2].mean(),
            1.0,
            0.0,
        ]
    return rows

