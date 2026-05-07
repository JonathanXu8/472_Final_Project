from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

from .data import ensure_annotations, load_split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/annotations")
    parser.add_argument("--out", default="results/dataset_stats.csv")
    args = parser.parse_args()
    ensure_annotations(args.data_dir)
    rows = []
    for source in ("VizWiz", "VQA"):
        for split in ("train", "val", "test"):
            examples = load_split(split, args.data_dir, (source,))
            counts = Counter("unlabeled" if ex.label is None else "single" if ex.label == 1 else "multiple" for ex in examples)
            rows.append(
                {
                    "source": source,
                    "split": split,
                    "examples": len(examples),
                    "single": counts.get("single", 0),
                    "multiple": counts.get("multiple", 0),
                    "unlabeled": counts.get("unlabeled", 0),
                }
            )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.out).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()

