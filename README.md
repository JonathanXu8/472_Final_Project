# CLIP Single Answer Grounding

Predict whether all valid answers to a visual question share the same grounding region. The main deployable model is a frozen CLIP vision-language encoder plus a small MLP classifier.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Download annotations:

```bash
bash data/download_annotations.sh
```

Download raw images following [data/README.md](data/README.md). The CLIP model will not run without images under `data/raw_images/`.

## Reproduce Main Results

Dataset stats and polygon diagnostic:

```bash
bash experiments/run_tables.sh
```

Train/evaluate the CLIP model:

```bash
bash experiments/run_clip.sh
```

Outputs:

```text
results/clip_mlp/metrics.json
results/clip_mlp/val_predictions.json
results/clip_mlp/test_submission.json
results/clip_mlp/classifier.pt
results/dataset_stats.csv
results/polygon_diagnostic.json
```

The key number is:

```text
results/clip_mlp/metrics.json -> val_best_threshold -> f1
```

The current saved Colab CLIP run achieved validation F1 `0.9220`, improving over the majority baseline F1 `0.9079`.

## Model

The model freezes `openai/clip-vit-base-patch32` and extracts:

- CLIP image embedding
- CLIP question/text embedding
- elementwise image-text product
- absolute image-text difference
- cosine similarity
- lightweight grid-region salience features

A PyTorch MLP is trained on those features with class-balanced binary cross entropy. The validation threshold is tuned for F1, matching the challenge metric.

## Notebook

The optional notebook version is in:

```text
notebooks/final_project_single_notebook.ipynb
```
