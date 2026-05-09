"""Evaluation script — computes metrics and saves artifacts."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from torch import nn

from visionserve.config import load_config
from visionserve.data import build_dataloaders
from visionserve.models import build_model
from visionserve.models.factory import load_checkpoint
from visionserve.utils import get_logger, resolve_device

logger = get_logger(__name__)


@torch.no_grad()
def compute_metrics(
    model: nn.Module,
    loader,
    device: torch.device,
    num_classes: int,
    topk: tuple[int, ...] = (1, 5),
) -> dict:
    """Compute top-k accuracy, per-class precision/recall/F1, and confusion matrix."""
    model.eval()
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)
    topk_correct = defaultdict(int)
    total = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        # top-k
        max_k = max(topk)
        _, pred = logits.topk(max_k, dim=1)
        for k in topk:
            correct_k = (pred[:, :k] == targets.unsqueeze(1)).any(dim=1).sum().item()
            topk_correct[k] += correct_k
        # confusion
        top1 = pred[:, 0].cpu()
        for t, p in zip(targets.cpu().tolist(), top1.tolist()):
            confusion[t, p] += 1
        total += targets.size(0)

    # Per-class metrics from confusion
    tp = confusion.diag().float()
    fp = confusion.sum(0).float() - tp
    fn = confusion.sum(1).float() - tp
    precision = (tp / (tp + fp).clamp(min=1)).tolist()
    recall = (tp / (tp + fn).clamp(min=1)).tolist()
    f1 = [
        (2 * p * r) / (p + r) if (p + r) > 0 else 0.0
        for p, r in zip(precision, recall)
    ]

    return {
        "total_samples": total,
        "topk_accuracy": {f"top{k}": topk_correct[k] / total for k in topk},
        "per_class": {
            "precision": precision,
            "recall": recall,
            "f1": f1,
        },
        "macro_f1": sum(f1) / len(f1),
        "confusion_matrix": confusion.tolist(),
    }


def save_confusion_plot(matrix: list[list[int]], class_names: list[str], path: Path) -> None:
    """Save confusion matrix heatmap. Requires matplotlib (extras: train)."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        logger.warning("matplotlib not installed; skipping confusion matrix plot")
        return

    arr = np.array(matrix)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(arr, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    logger.info(f"Saved confusion matrix to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained VisionServe model")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="./eval_output")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = resolve_device(cfg.train.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _, val_loader, class_names = build_dataloaders(cfg.data, seed=cfg.train.seed)

    model = build_model(
        backbone=cfg.model.backbone,
        num_classes=len(class_names),
        pretrained=False,
        dropout=cfg.model.dropout,
    ).to(device)
    load_checkpoint(model, args.checkpoint, device=device)

    metrics = compute_metrics(model, val_loader, device, num_classes=len(class_names))
    metrics["class_names"] = class_names

    out_json = output_dir / "metrics.json"
    with out_json.open("w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved metrics to {out_json}")
    logger.info(f"Top-1 acc: {metrics['topk_accuracy']['top1']:.4f}")
    logger.info(f"Top-5 acc: {metrics['topk_accuracy']['top5']:.4f}")
    logger.info(f"Macro F1: {metrics['macro_f1']:.4f}")

    save_confusion_plot(metrics["confusion_matrix"], class_names, output_dir / "confusion_matrix.png")


if __name__ == "__main__":
    main()
