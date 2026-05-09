"""Tests for model factory."""
from __future__ import annotations

import pytest
import torch

from visionserve.models import build_model
from visionserve.models.factory import count_parameters, load_checkpoint


@pytest.mark.parametrize("backbone", ["resnet18", "mobilenet_v3_small"])
def test_build_model_forward(backbone: str):
    model = build_model(backbone=backbone, num_classes=10, pretrained=False)
    model.eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 10)


def test_build_model_invalid_backbone():
    with pytest.raises(ValueError, match="Unknown backbone"):
        build_model(backbone="not_a_model")


def test_count_parameters():
    model = build_model(backbone="resnet18", num_classes=10, pretrained=False)
    total, trainable = count_parameters(model)
    assert total > 0
    assert trainable == total  # all trainable by default


def test_dropout_applied():
    model = build_model(backbone="resnet18", num_classes=10, pretrained=False, dropout=0.5)
    # Walk the head and check at least one Dropout layer exists
    has_dropout = any(isinstance(m, torch.nn.Dropout) for m in model.modules())
    assert has_dropout


def test_load_checkpoint(tmp_path):
    model_a = build_model(backbone="resnet18", num_classes=10, pretrained=False)
    ckpt_path = tmp_path / "ckpt.pt"
    torch.save({"model_state_dict": model_a.state_dict()}, ckpt_path)

    model_b = build_model(backbone="resnet18", num_classes=10, pretrained=False)
    load_checkpoint(model_b, str(ckpt_path), device="cpu")

    # Verify weights match
    for (n1, p1), (n2, p2) in zip(model_a.named_parameters(), model_b.named_parameters(), strict=False):
        assert n1 == n2
        assert torch.equal(p1, p2)
