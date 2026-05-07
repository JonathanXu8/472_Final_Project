from __future__ import annotations

import argparse

import numpy as np

from .data import ensure_annotations, labels, load_split
from .utils import save_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize ground-truth polygon dispersion.")
    parser.add_argument("--data-dir", default="data/annotations")
    parser.add_argument("--out", default="results/polygon_diagnostic.json")
    args = parser.parse_args()
    ensure_annotations(args.data_dir)
    val = load_split("val", args.data_dir)
    y_val = labels(val)
    summary = polygon_summary(val, y_val)
    save_json(args.out, {"note": "Analysis only; uses ground-truth polygons.", "summary": summary})
    print(summary)


def polygon_summary(examples, y):
    rows = []
    for label_value, label_name in [(1, "single"), (0, "multiple")]:
        subset = [ex for ex, yy in zip(examples, y) if yy == label_value]
        centroid_dists = []
        polygon_counts = []
        for ex in subset:
            polygons = ex.raw.get("grounding_labels") or []
            width = float(ex.raw.get("width") or 1.0)
            height = float(ex.raw.get("height") or 1.0)
            centroids = []
            for poly in polygons:
                pts = [(float(p["x"]) / width, float(p["y"]) / height) for p in poly if "x" in p and "y" in p]
                if pts:
                    centroids.append(np.array(pts).mean(axis=0))
            c = np.array(centroids) if centroids else np.zeros((0, 2))
            dists = [float(np.linalg.norm(c[a] - c[b])) for a in range(len(c)) for b in range(a + 1, len(c))]
            centroid_dists.append(float(np.mean(dists)) if dists else 0.0)
            polygon_counts.append(len(polygons))
        rows.append(
            {
                "label": label_name,
                "count": len(subset),
                "mean_num_polygons": float(np.mean(polygon_counts)),
                "mean_centroid_distance": float(np.mean(centroid_dists)),
            }
        )
    return rows


if __name__ == "__main__":
    main()

