"""
Computes AUC, sensitivity, and specificity for each saved run.

Reads the {run_name}_predictions.pt files saved by train.py (each containing
{"labels": [...], "probs": [...]}) and prints a results table.

Sensitivity/specificity require a decision threshold on the TB-probability
score; 0.5 is used here (standard default, matches what argmax(logits) would
already have picked during training/eval).

Run this after you've generated the .pt files for whichever conditions you
want metrics for, e.g.:
    python compute_metrics.py
"""

import torch
from pathlib import Path
from sklearn.metrics import roc_auc_score, confusion_matrix

RUN_NAMES = [
    "baseline",
    "resnet18_scratch",
    "resnet18_pretrained",
    "densenet121_scratch",
    "densenet121_pretrained",
]

THRESHOLD = 0.5


def compute_metrics(labels, probs, threshold=THRESHOLD):
    preds = [1 if p >= threshold else 0 for p in probs]

    auc = roc_auc_score(labels, probs)

    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")

    return auc, sensitivity, specificity


if __name__ == "__main__":
    print(f"{'Run':<25} {'AUC':>8} {'Sensitivity':>12} {'Specificity':>12}")
    print("-" * 60)

    for run_name in RUN_NAMES:
        path = Path(f"{run_name}_predictions.pt")
        if not path.exists():
            print(f"{run_name:<25} (no saved predictions found, skipped)")
            continue

        data = torch.load(path)
        labels = data["labels"]
        probs = data["probs"]

        auc, sensitivity, specificity = compute_metrics(labels, probs)

        print(f"{run_name:<25} {auc:>8.4f} {sensitivity:>12.4f} {specificity:>12.4f}")