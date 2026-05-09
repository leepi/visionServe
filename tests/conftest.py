"""Shared pytest fixtures."""
from __future__ import annotations

import io
from pathlib import Path

import pytest
import torch
from PIL import Image

from visionserve.config import Config
from visionserve.models import build_model


@pytest.fixture
def tiny_image() -> Image.Image:
    """Return a small 64x64 random RGB image."""
    arr = (torch.rand(64, 64, 3) * 255).byte().numpy()
    return Image.fromarray(arr, mode="RGB")


@pytest.fixture
def tiny_image_bytes(tiny_image: Image.Image) -> bytes:
    buf = io.BytesIO()
    tiny_image.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def tiny_checkpoint(tmp_path: Path) -> Path:
    """Build an untrained tiny model and save a checkpoint for testing."""
    class_names = ["airplane", "automobile", "bird", "cat", "deer",
                   "dog", "frog", "horse", "ship", "truck"]
    model = build_model(backbone="resnet18", num_classes=10, pretrained=False)
    ckpt_path = tmp_path / "test_ckpt.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": 1,
            "val_acc": 0.5,
            "val_loss": 1.0,
            "class_names": class_names,
            "config": {
                "model": {"backbone": "resnet18", "num_classes": 10, "dropout": 0.0},
                "data": {"image_size": 224},
            },
        },
        ckpt_path,
    )
    return ckpt_path


@pytest.fixture
def default_config() -> Config:
    return Config()
