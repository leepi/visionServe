"""Training entry point."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch import nn, optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from tqdm import tqdm

from visionserve.config import Config, load_config, save_config
from visionserve.data import build_dataloaders
from visionserve.models import build_model
from visionserve.models.factory import count_parameters
from visionserve.utils import get_logger, resolve_device, set_seed

logger = get_logger(__name__)


def build_optimizer(model: nn.Module, cfg) -> optim.Optimizer:
    params = [p for p in model.parameters() if p.requires_grad]
    if cfg.optimizer == "adamw":
        return optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.optimizer == "sgd":
        return optim.SGD(
            params, lr=cfg.lr, momentum=cfg.momentum,
            weight_decay=cfg.weight_decay, nesterov=True,
        )
    raise ValueError(f"Unknown optimizer '{cfg.optimizer}'")


def build_scheduler(optimizer: optim.Optimizer, cfg, num_epochs: int):
    if cfg.scheduler == "cosine":
        return CosineAnnealingLR(optimizer, T_max=num_epochs)
    if cfg.scheduler == "step":
        return StepLR(optimizer, step_size=max(1, num_epochs // 3), gamma=0.1)
    if cfg.scheduler == "none":
        return None
    raise ValueError(f"Unknown scheduler '{cfg.scheduler}'")


def train_one_epoch(
    model, loader, criterion, optimizer, device, grad_clip: float, epoch: int
) -> dict:
    model.train()
    total_loss, total_correct, total_seen = 0.0, 0, 0
    pbar = tqdm(loader, desc=f"Epoch {epoch} [train]", leave=False)
    for images, targets in pbar:
        images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        bs = targets.size(0)
        total_loss += loss.item() * bs
        total_correct += (logits.argmax(1) == targets).sum().item()
        total_seen += bs
        pbar.set_postfix(loss=f"{total_loss / total_seen:.4f}",
                         acc=f"{total_correct / total_seen:.4f}")
    return {"loss": total_loss / total_seen, "acc": total_correct / total_seen}


@torch.no_grad()
def evaluate(model, loader, criterion, device) -> dict:
    model.eval()
    total_loss, total_correct, total_seen = 0.0, 0, 0
    for images, targets in tqdm(loader, desc="eval", leave=False):
        images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, targets)
        bs = targets.size(0)
        total_loss += loss.item() * bs
        total_correct += (logits.argmax(1) == targets).sum().item()
        total_seen += bs
    return {"loss": total_loss / total_seen, "acc": total_correct / total_seen}


def train(cfg: Config) -> Path:
    """Run training. Returns path to best checkpoint."""
    set_seed(cfg.train.seed)
    device = resolve_device(cfg.train.device)
    logger.info(f"Using device: {device}")

    output_dir = Path(cfg.train.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, class_names = build_dataloaders(cfg.data, seed=cfg.train.seed)
    cfg.class_names = class_names
    cfg.model.num_classes = len(class_names)
    logger.info(f"Classes: {class_names}")
    logger.info(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    model = build_model(
        backbone=cfg.model.backbone,
        num_classes=cfg.model.num_classes,
        pretrained=cfg.model.pretrained,
        dropout=cfg.model.dropout,
    ).to(device)
    total, trainable = count_parameters(model)
    logger.info(f"Model: {cfg.model.backbone} | Params total={total:,} trainable={trainable:,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.train.label_smoothing)
    optimizer = build_optimizer(model, cfg.train)
    scheduler = build_scheduler(optimizer, cfg.train, cfg.train.epochs)

    best_acc = 0.0
    patience = 0
    history = []
    best_path = output_dir / "best.pt"

    for epoch in range(1, cfg.train.epochs + 1):
        t0 = time.time()
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, cfg.train.grad_clip, epoch
        )
        val_metrics = evaluate(model, val_loader, criterion, device)
        if scheduler is not None:
            scheduler.step()

        elapsed = time.time() - t0
        logger.info(
            f"Epoch {epoch}/{cfg.train.epochs} | "
            f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['acc']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['acc']:.4f} | "
            f"{elapsed:.1f}s"
        )
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics, "time": elapsed})

        if val_metrics["acc"] > best_acc:
            best_acc = val_metrics["acc"]
            patience = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_acc": val_metrics["acc"],
                    "val_loss": val_metrics["loss"],
                    "config": {
                        "model": cfg.model.__dict__,
                        "data": cfg.data.__dict__,
                    },
                    "class_names": class_names,
                },
                best_path,
            )
            logger.info(f"  ✓ Saved new best checkpoint (val_acc={best_acc:.4f})")
        else:
            patience += 1
            if patience >= cfg.train.early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch} epochs")
                break

    # Persist final config and history alongside checkpoint
    save_config(cfg, output_dir / "config.yaml")
    with (output_dir / "history.json").open("w") as f:
        json.dump(history, f, indent=2)

    logger.info(f"Training complete. Best val_acc={best_acc:.4f}. Checkpoint: {best_path}")
    return best_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a VisionServe model")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--backbone", type=str, default=None,
                        choices=["resnet18", "mobilenet_v3_small"])
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg.train.epochs = args.epochs
    if args.backbone is not None:
        cfg.model.backbone = args.backbone
    if args.output_dir is not None:
        cfg.train.output_dir = args.output_dir

    train(cfg)


if __name__ == "__main__":
    main()
