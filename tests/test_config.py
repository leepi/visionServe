"""Tests for config loading and saving."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from visionserve.config import Config, load_config, save_config


def test_default_config():
    cfg = load_config()
    assert cfg.model.backbone == "resnet18"
    assert cfg.data.dataset == "cifar10"
    assert cfg.train.epochs == 10
    assert len(cfg.class_names) == 10


def test_load_config_overrides(tmp_path: Path):
    yaml_path = tmp_path / "test.yaml"
    yaml_path.write_text(yaml.safe_dump({
        "model": {"backbone": "mobilenet_v3_small", "dropout": 0.3},
        "train": {"epochs": 5, "lr": 0.01},
    }))
    cfg = load_config(yaml_path)
    assert cfg.model.backbone == "mobilenet_v3_small"
    assert cfg.model.dropout == 0.3
    # Defaults preserved for unspecified
    assert cfg.model.pretrained is True
    assert cfg.train.epochs == 5
    assert cfg.train.lr == 0.01
    assert cfg.train.optimizer == "adamw"  # default


def test_save_and_load_roundtrip(tmp_path: Path):
    cfg = Config()
    cfg.train.epochs = 99
    out = tmp_path / "rt.yaml"
    save_config(cfg, out)
    loaded = load_config(out)
    assert loaded.train.epochs == 99


def test_missing_config_raises():
    with pytest.raises(FileNotFoundError):
        load_config("/nonexistent/path.yaml")
