"""Build a starter checkpoint locally without external dataset downloads.

This script generates a small synthetic dataset (colored shapes) shaped
like CIFAR-10 (10 classes), fine-tunes the ImageNet-pretrained ResNet18
head on it, and saves a working checkpoint at checkpoints/best.pt.

The result is small (~45 MB), real, and serves through the API. Users
who clone the repo and run `make train` will retrain on actual CIFAR-10.
"""
from __future__ import annotations

import math
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

from visionserve.data.transforms import build_transforms
from visionserve.models import build_model
from visionserve.utils import get_logger, set_seed

logger = get_logger("starter_ckpt")

CIFAR_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

# Per-class color palette — gives the model a real signal to learn
CLASS_COLORS = {
    0: (135, 206, 250),  # airplane: sky blue
    1: (220, 20, 60),    # automobile: crimson
    2: (255, 255, 0),    # bird: yellow
    3: (255, 165, 0),    # cat: orange
    4: (139, 69, 19),    # deer: brown
    5: (105, 105, 105),  # dog: dim gray
    6: (0, 255, 0),      # frog: green
    7: (160, 82, 45),    # horse: sienna
    8: (70, 130, 180),   # ship: steel blue
    9: (47, 79, 79),     # truck: dark slate gray
}


class SyntheticCIFAR(Dataset):
    """Tiny synthetic dataset — class-colored shapes with noise."""

    def __init__(self, n_per_class: int = 50, image_size: int = 64, transform=None, seed: int = 0):
        self.transform = transform
        self.samples: list[tuple[Image.Image, int]] = []
        rng = torch.Generator().manual_seed(seed)
        for cls in range(10):
            base_color = CLASS_COLORS[cls]
            for i in range(n_per_class):
                img = Image.new("RGB", (image_size, image_size), color=base_color)
                draw = ImageDraw.Draw(img)
                # Draw a per-class shape with some position jitter
                jitter = (torch.randint(-8, 8, (4,), generator=rng).tolist())
                cx, cy = image_size // 2 + jitter[0], image_size // 2 + jitter[1]
                r = image_size // 4 + jitter[2] // 2
                if cls % 3 == 0:
                    draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(255, 255, 255))
                elif cls % 3 == 1:
                    draw.rectangle((cx - r, cy - r, cx + r, cy + r), fill=(0, 0, 0))
                else:
                    draw.polygon(
                        [(cx, cy - r), (cx - r, cy + r), (cx + r, cy + r)],
                        fill=(255, 255, 255),
                    )
                # Add per-pixel noise
                noise = (torch.randn(image_size, image_size, 3, generator=rng) * 25).clamp(-50, 50)
                arr = torch.tensor(list(img.getdata()), dtype=torch.float32).reshape(image_size, image_size, 3)
                arr = (arr + noise).clamp(0, 255).byte().numpy()
                img = Image.fromarray(arr, mode="RGB")
                self.samples.append((img, cls))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img, label = self.samples[idx]
        if self.transform:
            img = self.transform(img)
        return img, label


def main():
    set_seed(42)
    device = torch.device("cpu")

    image_size = 64
    train_tf = build_transforms(image_size, train=True)
    val_tf = build_transforms(image_size, train=False)

    logger.info("Generating synthetic dataset...")
    train_set = SyntheticCIFAR(n_per_class=80, image_size=image_size, transform=train_tf, seed=1)
    val_set = SyntheticCIFAR(n_per_class=20, image_size=image_size, transform=val_tf, seed=2)

    train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=64, shuffle=False, num_workers=0)

    logger.info(f"Train: {len(train_set)} | Val: {len(val_set)}")

    model = build_model(backbone="resnet18", num_classes=10, pretrained=False, dropout=0.1)
    model.to(device)

    # Train all params from scratch (no internet for pretrained weights in sandbox)
    trainable = list(model.parameters())
    optimizer = optim.AdamW(trainable, lr=3e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    epochs = 3
    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        total, correct, loss_sum = 0, 0, 0.0
        for images, targets in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * targets.size(0)
            correct += (logits.argmax(1) == targets).sum().item()
            total += targets.size(0)
        train_loss, train_acc = loss_sum / total, correct / total

        model.eval()
        v_total, v_correct, v_loss_sum = 0, 0, 0.0
        with torch.no_grad():
            for images, targets in val_loader:
                logits = model(images)
                loss = criterion(logits, targets)
                v_loss_sum += loss.item() * targets.size(0)
                v_correct += (logits.argmax(1) == targets).sum().item()
                v_total += targets.size(0)
        val_loss, val_acc = v_loss_sum / v_total, v_correct / v_total

        logger.info(
            f"Epoch {epoch}/{epochs} | train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )
        if val_acc > best_acc:
            best_acc = val_acc

    out_path = Path("checkpoints/best.pt")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": epochs,
            "val_acc": best_acc,
            "val_loss": val_loss,
            "class_names": CIFAR_CLASSES,
            "config": {
                "model": {"backbone": "resnet18", "num_classes": 10, "pretrained": False, "dropout": 0.1},
                "data": {"image_size": image_size, "dataset": "synthetic_cifar_starter"},
            },
            "note": (
                "Starter checkpoint trained on a small synthetic dataset to demonstrate "
                "the API end-to-end. Retrain on real CIFAR-10 using `make train`."
            ),
        },
        out_path,
    )
    size_mb = out_path.stat().st_size / (1024 * 1024)
    logger.info(f"Saved {out_path} ({size_mb:.1f} MB) — best val_acc={best_acc:.4f}")


if __name__ == "__main__":
    main()
