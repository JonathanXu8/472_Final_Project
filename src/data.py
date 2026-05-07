from __future__ import annotations

import json
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

ANNOTATION_URL = "https://vizwiz.cs.colorado.edu/VizWiz_AnswerTherapy/Annotation.zip"


@dataclass
class Example:
    question_id: str
    image_id: str
    question: str
    source: str
    label: int | None
    raw: dict


def ensure_annotations(data_dir: str | Path = "data/annotations") -> None:
    data_dir = Path(data_dir)
    expected = [
        data_dir / "VizWiz_train.json",
        data_dir / "VizWiz_val.json",
        data_dir / "VizWiz_test.json",
        data_dir / "VQA_train.json",
        data_dir / "VQA_val.json",
        data_dir / "VQA_test.json",
    ]
    if all(path.exists() for path in expected):
        return

    data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = data_dir.parent / "answertherapy_annotations.zip"
    urllib.request.urlretrieve(ANNOTATION_URL, zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(data_dir.parent)
    extracted = data_dir.parent / "Annotations"
    for path in extracted.glob("*.json"):
        path.replace(data_dir / path.name)
    try:
        extracted.rmdir()
    except OSError:
        pass


def load_split(
    split: str,
    data_dir: str | Path = "data/annotations",
    sources: Iterable[str] = ("VizWiz", "VQA"),
) -> list[Example]:
    data_dir = Path(data_dir)
    examples: list[Example] = []
    for source in sources:
        records = json.loads((data_dir / f"{source}_{split}.json").read_text())
        for record in records:
            label_text = record.get("binary_label")
            label = 1 if label_text == "single" else 0 if label_text == "multiple" else None
            question_id = str(record.get("question_id") or record.get("image_id"))
            image_id = str(record.get("image_id") or question_id)
            examples.append(
                Example(
                    question_id=question_id,
                    image_id=image_id,
                    question=str(record.get("question", "")),
                    source=source,
                    label=label,
                    raw=record,
                )
            )
    return examples


def labels(examples: list[Example]):
    import numpy as np

    if any(ex.label is None for ex in examples):
        raise ValueError("Some examples do not have labels.")
    return np.array([int(ex.label) for ex in examples], dtype=np.int32)

