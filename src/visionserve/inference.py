"""Inference engine — wraps a model for prediction."""
from __future__ import annotations

import io
from pathlib import Path

import torch
from PIL import Image
from torch import nn

from visionserve.data.transforms import build_transforms
from visionserve.models import build_model
from visionserve.utils import resolve_device


class InferenceEngine:
    """Wraps a trained model for production inference.

    Loads checkpoint metadata, builds the right architecture, and exposes
    `predict()` and `predict_batch()` against PIL Images or raw bytes.
    """

    def __init__(self, checkpoint_path: str | Path, device: str = "auto") -> None:
        self.device = resolve_device(device)
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        config = ckpt.get("config", {})
        model_cfg = config.get("model", {})
        data_cfg = config.get("data", {})

        self.class_names: list[str] = ckpt.get("class_names", [])
        if not self.class_names:
            raise ValueError("Checkpoint missing class_names")

        self.backbone: str = model_cfg.get("backbone", "resnet18")
        self.image_size: int = data_cfg.get("image_size", 224)

        self.model: nn.Module = build_model(
            backbone=self.backbone,
            num_classes=len(self.class_names),
            pretrained=False,
            dropout=model_cfg.get("dropout", 0.0),
        )
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(self.device).eval()

        self.transform = build_transforms(self.image_size, train=False)
        self.checkpoint_meta = {
            "epoch": ckpt.get("epoch"),
            "val_acc": ckpt.get("val_acc"),
            "val_loss": ckpt.get("val_loss"),
            "backbone": self.backbone,
        }

    def _prep(self, image: Image.Image | bytes) -> torch.Tensor:
        if isinstance(image, bytes):
            image = Image.open(io.BytesIO(image))
        if image.mode != "RGB":
            image = image.convert("RGB")
        return self.transform(image)

    @torch.no_grad()
    def predict(self, image: Image.Image | bytes, top_k: int = 5) -> list[dict]:
        """Predict top-k classes for a single image."""
        tensor = self._prep(image).unsqueeze(0).to(self.device)
        logits = self.model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
        top_k = min(top_k, len(self.class_names))
        values, indices = probs.topk(top_k)
        return [
            {"class": self.class_names[i.item()], "probability": float(v.item())}
            for v, i in zip(values, indices)
        ]

    @torch.no_grad()
    def predict_batch(
        self, images: list[Image.Image | bytes], top_k: int = 5
    ) -> list[list[dict]]:
        """Predict top-k for a batch of images."""
        if not images:
            return []
        tensors = torch.stack([self._prep(img) for img in images]).to(self.device)
        logits = self.model(tensors)
        probs = torch.softmax(logits, dim=1)
        top_k = min(top_k, len(self.class_names))
        values, indices = probs.topk(top_k, dim=1)

        results = []
        for vals, idxs in zip(values, indices):
            results.append(
                [
                    {"class": self.class_names[i.item()], "probability": float(v.item())}
                    for v, i in zip(vals, idxs)
                ]
            )
        return results
