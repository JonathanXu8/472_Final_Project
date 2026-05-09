from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .data import ensure_annotations, labels, load_split
from .image_utils import find_image, image_coverage, image_grid_features
from .metrics import precision_recall_f1, tune_threshold
from .utils import balanced_weights, prediction_rows, save_json, standardize


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train frozen CLIP + MLP for single answer grounding.")
    parser.add_argument("--data-dir", default="data/annotations")
    parser.add_argument("--image-dir", default="data/raw_images")
    parser.add_argument("--out-dir", default="results/clip_mlp")
    parser.add_argument("--model-name", default="openai/clip-vit-base-patch32")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=472)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_annotations(args.data_dir)
    image_dir = Path(args.image_dir)
    if not image_dir.exists() or not any(image_dir.iterdir()):
        raise FileNotFoundError("Raw images are required for the CLIP model. Put images under data/raw_images/.")

    try:
        import torch
        import torch.nn as nn
        from transformers import CLIPModel, CLIPProcessor
    except ImportError as exc:
        raise ImportError("Install dependencies with `pip install -r requirements.txt`.") from exc

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train = load_split("train", args.data_dir)
    val = load_split("val", args.data_dir)
    test = load_split("test", args.data_dir)
    y_train = labels(train)
    y_val = labels(val)

    print("train image coverage:", image_coverage(train, image_dir, limit=500))
    print("val image coverage:", image_coverage(val, image_dir, limit=500))

    device = choose_device(args.device, torch)
    print("Using device:", device)
    processor = CLIPProcessor.from_pretrained(args.model_name, use_fast=False)
    clip_model = CLIPModel.from_pretrained(args.model_name).to(device).eval()

    out_dir = Path(args.out_dir)
    cache_dir = out_dir / "feature_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    print("Extracting or loading CLIP features")
    cx_train = cached_clip_features("train", train, cache_dir, image_dir, processor, clip_model, device, args.batch_size)
    cx_val = cached_clip_features("val", val, cache_dir, image_dir, processor, clip_model, device, args.batch_size)
    cx_test = cached_clip_features("test", test, cache_dir, image_dir, processor, clip_model, device, args.batch_size)

    print("Computing grid-region features")
    gx_train = image_grid_features(train, image_dir)
    gx_val = image_grid_features(val, image_dir)
    gx_test = image_grid_features(test, image_dir)

    x_train = np.concatenate([cx_train, gx_train], axis=1)
    x_val = np.concatenate([cx_val, gx_val], axis=1)
    x_test = np.concatenate([cx_test, gx_test], axis=1)
    x_train, x_val, x_test = standardize(x_train, x_val, x_test)

    model = build_mlp(x_train.shape[1], args.hidden_dim, nn).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.BCEWithLogitsLoss(reduction="none")

    tx = torch.tensor(x_train, dtype=torch.float32, device=device)
    ty = torch.tensor(y_train.astype(np.float32), dtype=torch.float32, device=device)
    tw = torch.tensor(balanced_weights(y_train), dtype=torch.float32, device=device)
    vx = torch.tensor(x_val, dtype=torch.float32, device=device)

    best_f1 = -1.0
    best_scores = None
    best_state = None
    stale = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = torch.randperm(tx.shape[0], device=device)
        for start in range(0, tx.shape[0], args.batch_size):
            idx = order[start : start + args.batch_size]
            logits = model(tx[idx]).squeeze(1)
            loss = (criterion(logits, ty[idx]) * tw[idx]).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_scores = torch.sigmoid(model(vx).squeeze(1)).detach().cpu().numpy()
        current = tune_threshold(y_val, val_scores)
        if current["f1"] > best_f1:
            best_f1 = current["f1"]
            best_scores = val_scores
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if epoch == 1 or epoch % 10 == 0:
            print(f"epoch {epoch:03d} best_f1={best_f1:.4f}")
        if stale >= 24:
            break

    assert best_state is not None and best_scores is not None
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_scores = torch.sigmoid(model(torch.tensor(x_test, dtype=torch.float32, device=device)).squeeze(1)).cpu().numpy()

    majority_scores = np.full(len(val), y_train.mean(), dtype=np.float64)
    metrics = {
        "model": "frozen_clip_mlp",
        "train_examples": len(train),
        "val_examples": len(val),
        "test_examples": len(test),
        "majority_baseline": {
            "val_at_0.5": precision_recall_f1(y_val, majority_scores, 0.5),
            "val_best_threshold": tune_threshold(y_val, majority_scores),
        },
        "val_at_0.5": precision_recall_f1(y_val, best_scores, 0.5),
        "val_best_threshold": tune_threshold(y_val, best_scores),
        "model_note": vars(args),
    }
    save_json(out_dir / "metrics.json", metrics)
    save_json(out_dir / "val_predictions.json", prediction_rows(val, best_scores, True))
    save_json(out_dir / "test_submission.json", prediction_rows(test, test_scores, False))
    torch.save(best_state, out_dir / "classifier.pt")
    print(json.dumps(metrics, indent=2))


def choose_device(device_arg: str, torch) -> str:
    if device_arg != "auto":
        return device_arg
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def pooled_tensor(output):
    if hasattr(output, "pooler_output"):
        return output.pooler_output
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state[:, 0]
    if isinstance(output, (tuple, list)):
        return output[1] if len(output) > 1 else output[0]
    return output


def cached_clip_features(name, examples, cache_dir, image_dir, processor, clip_model, device, batch_size):
    path = cache_dir / f"{name}.npz"
    if path.exists():
        return np.load(path)["features"]
    features = clip_features(examples, image_dir, processor, clip_model, device, batch_size)
    np.savez_compressed(path, features=features)
    return features


def clip_features(examples, image_dir, processor, clip_model, device, batch_size):
    import torch

    feats = []
    for start in range(0, len(examples), batch_size):
        batch = examples[start : start + batch_size]
        images = []
        for ex in batch:
            path = find_image(image_dir, ex.image_id)
            images.append(Image.open(path).convert("RGB") if path else Image.new("RGB", (224, 224)))
        texts = [ex.question for ex in batch]
        inputs = processor(text=texts, images=images, return_tensors="pt", padding=True, truncation=True)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            img = pooled_tensor(clip_model.get_image_features(pixel_values=inputs["pixel_values"]))
            txt = pooled_tensor(
                clip_model.get_text_features(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
            )
        img = img / img.norm(dim=1, keepdim=True).clamp_min(1e-8)
        txt = txt / txt.norm(dim=1, keepdim=True).clamp_min(1e-8)
        cosine = (img * txt).sum(dim=1, keepdim=True)
        fused = torch.cat([img, txt, img * txt, torch.abs(img - txt), cosine], dim=1)
        feats.append(fused.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(feats, axis=0)


def build_mlp(input_dim, hidden_dim, nn):
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.GELU(),
        nn.Dropout(0.2),
        nn.Linear(hidden_dim, hidden_dim // 2),
        nn.GELU(),
        nn.Dropout(0.1),
        nn.Linear(hidden_dim // 2, 1),
    )


if __name__ == "__main__":
    main()

