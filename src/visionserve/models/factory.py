"""Model factory — builds backbones with configurable head."""
from __future__ import annotations

import torch
from torch import nn
from torchvision import models


def _build_resnet18(num_classes: int, pretrained: bool, dropout: float) -> nn.Module:
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    in_features = model.fc.in_features
    if dropout > 0:
        model.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes),
        )
    else:
        model.fc = nn.Linear(in_features, num_classes)
    return model


def _build_mobilenet_v3_small(num_classes: int, pretrained: bool, dropout: float) -> nn.Module:
    weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v3_small(weights=weights)
    in_features = model.classifier[-1].in_features
    # Replace final linear; keep the existing dropout layer in classifier[2]
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    if dropout > 0:
        # Tune the existing dropout
        for layer in model.classifier:
            if isinstance(layer, nn.Dropout):
                layer.p = dropout
    return model


_REGISTRY = {
    "resnet18": _build_resnet18,
    "mobilenet_v3_small": _build_mobilenet_v3_small,
}


def build_model(
    backbone: str = "resnet18",
    num_classes: int = 10,
    pretrained: bool = True,
    dropout: float = 0.0,
) -> nn.Module:
    """Build a classification model.

    Args:
        backbone: One of 'resnet18', 'mobilenet_v3_small'.
        num_classes: Number of output classes.
        pretrained: Use ImageNet-pretrained weights.
        dropout: Dropout probability before final linear.

    Returns:
        torch.nn.Module ready for training or inference.
    """
    if backbone not in _REGISTRY:
        raise ValueError(
            f"Unknown backbone '{backbone}'. Available: {list(_REGISTRY.keys())}"
        )
    return _REGISTRY[backbone](num_classes, pretrained, dropout)


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Return (total_params, trainable_params)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    device: str | torch.device = "cpu",
    strict: bool = True,
) -> dict:
    """Load a checkpoint into model, returning the checkpoint dict."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict, strict=strict)
    return ckpt
