"""Tests for data transforms."""
from __future__ import annotations

import torch
from PIL import Image

from visionserve.data.transforms import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    build_transforms,
    get_normalization,
)


def test_normalization_constants():
    norm = get_normalization()
    assert tuple(norm.mean) == IMAGENET_MEAN
    assert tuple(norm.std) == IMAGENET_STD


def test_eval_transform_shape(tiny_image: Image.Image):
    tf = build_transforms(image_size=224, train=False)
    out = tf(tiny_image)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (3, 224, 224)


def test_train_transform_shape(tiny_image: Image.Image):
    tf = build_transforms(image_size=224, train=True)
    out = tf(tiny_image)
    assert out.shape == (3, 224, 224)


def test_eval_transform_deterministic(tiny_image: Image.Image):
    tf = build_transforms(image_size=128, train=False)
    a = tf(tiny_image)
    b = tf(tiny_image)
    assert torch.equal(a, b)


def test_grayscale_image_handled():
    """Grayscale images should be convertible — done in inference engine, not transforms."""
    gray = Image.new("L", (64, 64), 128).convert("RGB")
    tf = build_transforms(image_size=224, train=False)
    out = tf(gray)
    assert out.shape == (3, 224, 224)
