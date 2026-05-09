"""Image preprocessing transforms."""
from __future__ import annotations

from torchvision import transforms

# ImageNet stats — backbones are pretrained on ImageNet
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_normalization() -> transforms.Normalize:
    return transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)


def build_transforms(image_size: int = 224, train: bool = False) -> transforms.Compose:
    """Build train or eval transforms.

    Train pipeline applies light augmentation appropriate for natural images.
    Eval pipeline is deterministic — resize, center-crop, normalize.
    """
    if train:
        return transforms.Compose(
            [
                transforms.Resize(int(image_size * 1.15)),
                transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.2, 0.2, 0.2),
                transforms.ToTensor(),
                get_normalization(),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.15)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            get_normalization(),
        ]
    )
