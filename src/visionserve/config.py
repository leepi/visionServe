"""Configuration loading and validation."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    """Model architecture configuration."""

    backbone: str = "resnet18"  # resnet18 | mobilenet_v3_small
    num_classes: int = 10
    pretrained: bool = True
    dropout: float = 0.0


@dataclass
class DataConfig:
    """Data pipeline configuration."""

    dataset: str = "cifar10"  # cifar10 | imagefolder
    data_dir: str = "./data"
    image_size: int = 224
    batch_size: int = 128
    num_workers: int = 4
    val_split: float = 0.1
    # Used when dataset == "imagefolder"
    train_dir: str | None = None
    val_dir: str | None = None


@dataclass
class TrainConfig:
    """Training loop configuration."""

    epochs: int = 10
    lr: float = 1e-3
    weight_decay: float = 1e-4
    optimizer: str = "adamw"  # adamw | sgd
    scheduler: str = "cosine"  # cosine | step | none
    momentum: float = 0.9  # for SGD
    label_smoothing: float = 0.1
    grad_clip: float = 1.0
    seed: int = 42
    device: str = "auto"  # auto | cpu | cuda
    output_dir: str = "./checkpoints"
    log_dir: str = "./runs"
    early_stopping_patience: int = 5
    save_best_only: bool = True


@dataclass
class Config:
    """Top-level config."""

    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    class_names: list[str] = field(
        default_factory=lambda: [
            "airplane", "automobile", "bird", "cat", "deer",
            "dog", "frog", "horse", "ship", "truck",
        ]
    )


def _merge(default: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep merge override into default."""
    out = dict(default)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | Path | None = None) -> Config:
    """Load config from YAML, falling back to defaults."""
    cfg = Config()
    if path is None:
        return cfg

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r") as f:
        raw = yaml.safe_load(f) or {}

    return Config(
        model=ModelConfig(**_merge(cfg.model.__dict__, raw.get("model", {}))),
        data=DataConfig(**_merge(cfg.data.__dict__, raw.get("data", {}))),
        train=TrainConfig(**_merge(cfg.train.__dict__, raw.get("train", {}))),
        class_names=raw.get("class_names", cfg.class_names),
    )


def save_config(cfg: Config, path: str | Path) -> None:
    """Save config to YAML."""
    out = {
        "model": cfg.model.__dict__,
        "data": cfg.data.__dict__,
        "train": cfg.train.__dict__,
        "class_names": cfg.class_names,
    }
    with Path(path).open("w") as f:
        yaml.safe_dump(out, f, sort_keys=False)
