"""Dataset and DataLoader construction."""
from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets

from visionserve.config import DataConfig
from visionserve.data.transforms import build_transforms


def _build_cifar10(cfg: DataConfig) -> tuple[Dataset, Dataset, list[str]]:
    """Build CIFAR-10 train/val datasets."""
    Path(cfg.data_dir).mkdir(parents=True, exist_ok=True)
    train_tf = build_transforms(cfg.image_size, train=True)
    val_tf = build_transforms(cfg.image_size, train=False)

    full_train = datasets.CIFAR10(
        root=cfg.data_dir, train=True, download=True, transform=train_tf
    )
    test_set = datasets.CIFAR10(
        root=cfg.data_dir, train=False, download=True, transform=val_tf
    )
    return full_train, test_set, list(full_train.classes)


def _build_imagefolder(cfg: DataConfig) -> tuple[Dataset, Dataset, list[str]]:
    """Build datasets from an ImageFolder-style directory layout."""
    if cfg.train_dir is None or cfg.val_dir is None:
        raise ValueError(
            "data.train_dir and data.val_dir must be set when dataset='imagefolder'"
        )
    train_tf = build_transforms(cfg.image_size, train=True)
    val_tf = build_transforms(cfg.image_size, train=False)

    train_set = datasets.ImageFolder(cfg.train_dir, transform=train_tf)
    val_set = datasets.ImageFolder(cfg.val_dir, transform=val_tf)

    if train_set.classes != val_set.classes:
        raise ValueError("Train and val class lists differ")
    return train_set, val_set, list(train_set.classes)


def build_dataloaders(
    cfg: DataConfig,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, list[str]]:
    """Build train/val dataloaders. Returns (train_loader, val_loader, class_names)."""
    if cfg.dataset == "cifar10":
        train_full, val_set, class_names = _build_cifar10(cfg)
        # CIFAR-10's test set is our validation set; no extra split needed
        train_set: Dataset = train_full
    elif cfg.dataset == "imagefolder":
        train_set, val_set, class_names = _build_imagefolder(cfg)
    else:
        raise ValueError(f"Unknown dataset '{cfg.dataset}'")

    # If user wants an internal validation split from training set, support it
    if cfg.val_split > 0 and cfg.dataset == "imagefolder":
        n_val = int(len(train_set) * cfg.val_split)
        n_train = len(train_set) - n_val
        gen = torch.Generator().manual_seed(seed)
        train_set, _ = random_split(train_set, [n_train, n_val], generator=gen)

    train_loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=cfg.num_workers > 0,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=cfg.num_workers > 0,
    )
    return train_loader, val_loader, class_names
