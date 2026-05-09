"""Tests for the inference engine."""
from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from visionserve.inference import InferenceEngine


def test_engine_loads(tiny_checkpoint: Path):
    engine = InferenceEngine(tiny_checkpoint, device="cpu")
    assert len(engine.class_names) == 10
    assert engine.backbone == "resnet18"


def test_predict_single(tiny_checkpoint: Path, tiny_image: Image.Image):
    engine = InferenceEngine(tiny_checkpoint, device="cpu")
    preds = engine.predict(tiny_image, top_k=3)
    assert len(preds) == 3
    for p in preds:
        assert "class" in p
        assert "probability" in p
        assert 0.0 <= p["probability"] <= 1.0
    # Probabilities should be sorted descending
    probs = [p["probability"] for p in preds]
    assert probs == sorted(probs, reverse=True)


def test_predict_from_bytes(tiny_checkpoint: Path, tiny_image_bytes: bytes):
    engine = InferenceEngine(tiny_checkpoint, device="cpu")
    preds = engine.predict(tiny_image_bytes, top_k=1)
    assert len(preds) == 1


def test_predict_batch(tiny_checkpoint: Path, tiny_image: Image.Image):
    engine = InferenceEngine(tiny_checkpoint, device="cpu")
    results = engine.predict_batch([tiny_image, tiny_image, tiny_image], top_k=2)
    assert len(results) == 3
    for r in results:
        assert len(r) == 2


def test_predict_empty_batch(tiny_checkpoint: Path):
    engine = InferenceEngine(tiny_checkpoint, device="cpu")
    assert engine.predict_batch([]) == []


def test_topk_clamped_to_num_classes(tiny_checkpoint: Path, tiny_image: Image.Image):
    engine = InferenceEngine(tiny_checkpoint, device="cpu")
    preds = engine.predict(tiny_image, top_k=999)
    assert len(preds) == 10  # capped at num_classes


def test_grayscale_input_handled(tiny_checkpoint: Path):
    engine = InferenceEngine(tiny_checkpoint, device="cpu")
    gray = Image.new("L", (64, 64), 128)
    preds = engine.predict(gray, top_k=1)
    assert len(preds) == 1


def test_missing_checkpoint_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        InferenceEngine(tmp_path / "nonexistent.pt", device="cpu")
