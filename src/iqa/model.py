"""EfficientNet-B0 regression model and checkpoint loading utilities."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0


ARCHITECTURE = "efficientnet_b0"
TARGET_MIN = 0.0
TARGET_MAX = 100.0
DEFAULT_INPUT_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class IQAModel(nn.Module):
    """Predict a normalized perceptual quality score for one or more images.

    The network produces values in ``[0, 1]``. KonIQ-10k MOS predictions in
    the dataset's usual 0-100 point scale are therefore ``output * 100``.
    """

    def __init__(self, *, pretrained: bool = True, dropout: float = 0.2) -> None:
        super().__init__()
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in the interval [0, 1)")

        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        self.network = efficientnet_b0(weights=weights)
        in_features = self.network.classifier[-1].in_features
        self.network.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(in_features, 1),
            nn.Sigmoid(),
        )
        self.dropout = float(dropout)

    def forward(self, images: Tensor) -> Tensor:
        """Return one normalized score per image with shape ``(batch,)``."""

        return self.network(images).flatten()

    def freeze_backbone(self) -> None:
        """Freeze all convolutional features and leave the regression head trainable."""

        for parameter in self.network.features.parameters():
            parameter.requires_grad = False
        for parameter in self.network.classifier.parameters():
            parameter.requires_grad = True

    def unfreeze_last_feature_blocks(self, blocks: int = 2) -> None:
        """Unfreeze the final EfficientNet feature blocks for low-rate fine-tuning."""

        feature_blocks = list(self.network.features.children())
        if not 1 <= blocks <= len(feature_blocks):
            raise ValueError(
                f"blocks must be between 1 and {len(feature_blocks)}, got {blocks}"
            )

        self.freeze_backbone()
        for block in feature_blocks[-blocks:]:
            for parameter in block.parameters():
                parameter.requires_grad = True

    def model_config(self) -> dict[str, Any]:
        """Return the architecture metadata required to reconstruct this model."""

        return {
            "architecture": ARCHITECTURE,
            "dropout": self.dropout,
            "output_activation": "sigmoid",
            "output_shape": 1,
        }

    @classmethod
    def from_config(
        cls, config: Mapping[str, Any] | None = None, *, pretrained: bool = False
    ) -> "IQAModel":
        """Build a model from checkpoint metadata without downloading weights."""

        config = config or {}
        architecture = config.get("architecture", ARCHITECTURE)
        if architecture != ARCHITECTURE:
            raise ValueError(f"unsupported architecture: {architecture!r}")
        dropout = float(config.get("dropout", 0.2))
        return cls(pretrained=pretrained, dropout=dropout)


def make_checkpoint(
    model: IQAModel,
    *,
    epoch: int,
    phase: str,
    metrics: Mapping[str, float],
    optimizer: torch.optim.Optimizer | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Create a portable training checkpoint with inference metadata."""

    checkpoint: dict[str, Any] = {
        "format_version": 1,
        "model_state_dict": model.state_dict(),
        "model_config": model.model_config(),
        "target": {
            "name": "MOS",
            "normalization": "divide_by_100",
            "min": TARGET_MIN,
            "max": TARGET_MAX,
        },
        "preprocessing": {
            "input_size": DEFAULT_INPUT_SIZE,
            "mean": list(IMAGENET_MEAN),
            "std": list(IMAGENET_STD),
            "color_mode": "RGB",
        },
        "epoch": int(epoch),
        "phase": phase,
        "metrics": {key: float(value) for key, value in metrics.items()},
    }
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    if seed is not None:
        checkpoint["seed"] = int(seed)
    return checkpoint


def load_iqa_checkpoint(
    checkpoint_path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> tuple[IQAModel, dict[str, Any]]:
    """Load a saved model and return it in evaluation mode with its metadata."""

    path = Path(checkpoint_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")

    target_device = torch.device(map_location)
    try:
        checkpoint = torch.load(path, map_location=target_device, weights_only=False)
    except TypeError:  # PyTorch versions before the weights_only argument.
        checkpoint = torch.load(path, map_location=target_device)

    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(
            "invalid checkpoint: expected a dictionary containing model_state_dict"
        )

    model = IQAModel.from_config(checkpoint.get("model_config"), pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
    model.to(target_device)
    model.eval()
    return model, checkpoint


def normalized_to_mos(scores: Tensor | float) -> Tensor | float:
    """Convert normalized model output to the KonIQ MOS point scale."""

    scale = TARGET_MAX - TARGET_MIN
    if isinstance(scores, Tensor):
        return scores * scale + TARGET_MIN
    return float(scores) * scale + TARGET_MIN
