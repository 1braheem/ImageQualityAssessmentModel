"""Deterministic single-image inference for the IQA regression model."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from PIL import Image, ImageOps
from torch import nn
from torchvision.transforms import InterpolationMode
from torchvision.transforms import v2

from .model import (
    DEFAULT_INPUT_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    load_iqa_checkpoint,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "efficientnet_b0_koniq10k.pt"
DEFAULT_QUALITY_THRESHOLD = 0.60
DEFAULT_MINIMUM_WIDTH = 224
DEFAULT_MINIMUM_HEIGHT = 224


@dataclass(frozen=True)
class QualityPrediction:
    """One model prediction plus transparent, non-learned image checks."""

    quality_score: float
    mos_equivalent: float
    suitable: bool
    quality_threshold: float
    passes_quality_threshold: bool
    minimum_width: int
    minimum_height: int
    passes_resolution_check: bool
    image_width: int
    image_height: int
    image_format: str | None


def resolve_model_path(model_path: str | Path | None = None) -> Path:
    """Resolve an explicit path, ``IQA_MODEL_PATH``, or the project default."""

    selected = model_path or os.getenv("IQA_MODEL_PATH") or DEFAULT_MODEL_PATH
    return Path(selected).expanduser().resolve()


def _three_floats(
    value: object, *, name: str, fallback: Sequence[float]
) -> tuple[float, float, float]:
    if value is None:
        value = fallback
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"checkpoint preprocessing {name!r} must contain 3 numbers")
    numbers = tuple(float(item) for item in value)
    if len(numbers) != 3 or not all(math.isfinite(item) for item in numbers):
        raise ValueError(f"checkpoint preprocessing {name!r} must contain 3 numbers")
    return numbers  # type: ignore[return-value]


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


class IQAPredictor:
    """Load a local checkpoint and predict normalized perceptual quality.

    Loading always reconstructs EfficientNet-B0 without pretrained weights, so
    creating this predictor never performs a network download.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        *,
        device: str | torch.device = "cpu",
        quality_threshold: float = DEFAULT_QUALITY_THRESHOLD,
        minimum_width: int = DEFAULT_MINIMUM_WIDTH,
        minimum_height: int = DEFAULT_MINIMUM_HEIGHT,
    ) -> None:
        if not 0.0 <= quality_threshold <= 1.0:
            raise ValueError("quality_threshold must be in the interval [0, 1]")
        if minimum_width < 1 or minimum_height < 1:
            raise ValueError("minimum image dimensions must be positive")

        self.model_path = resolve_model_path(model_path)
        self.device = torch.device(device)
        self.quality_threshold = float(quality_threshold)
        self.minimum_width = int(minimum_width)
        self.minimum_height = int(minimum_height)

        model, checkpoint = load_iqa_checkpoint(
            self.model_path, map_location=self.device
        )
        self.model: nn.Module = model
        self.checkpoint_metadata = checkpoint

        preprocessing = _mapping(checkpoint.get("preprocessing"))
        input_size = int(preprocessing.get("input_size", DEFAULT_INPUT_SIZE))
        if input_size < 1:
            raise ValueError("checkpoint preprocessing input_size must be positive")
        color_mode = str(preprocessing.get("color_mode", "RGB"))
        if color_mode.upper() != "RGB":
            raise ValueError(
                f"unsupported checkpoint preprocessing color mode: {color_mode!r}"
            )
        mean = _three_floats(
            preprocessing.get("mean"), name="mean", fallback=IMAGENET_MEAN
        )
        std = _three_floats(
            preprocessing.get("std"), name="std", fallback=IMAGENET_STD
        )
        if any(value <= 0 for value in std):
            raise ValueError("checkpoint preprocessing standard deviations must be positive")

        self.input_size = input_size
        self.preprocess = v2.Compose(
            [
                v2.Resize(
                    (input_size, input_size),
                    interpolation=InterpolationMode.BICUBIC,
                    antialias=True,
                ),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=mean, std=std),
            ]
        )

        target = _mapping(checkpoint.get("target"))
        self.target_min = float(target.get("min", 0.0))
        self.target_max = float(target.get("max", 100.0))
        if not (
            math.isfinite(self.target_min)
            and math.isfinite(self.target_max)
            and self.target_max > self.target_min
        ):
            raise ValueError("checkpoint target range must have a finite max greater than min")

    def predict(self, image: Image.Image) -> QualityPrediction:
        """Predict an image without changing the dimensions reported to callers."""

        if not isinstance(image, Image.Image):
            raise TypeError("image must be a PIL Image")

        oriented = ImageOps.exif_transpose(image)
        image_width, image_height = oriented.size
        if image_width < 1 or image_height < 1:
            raise ValueError("image dimensions must be positive")

        image_format = image.format.upper() if image.format else None
        rgb_image = oriented.convert("RGB")
        model_input = self.preprocess(rgb_image).unsqueeze(0).to(self.device)

        self.model.eval()
        with torch.inference_mode():
            output = self.model(model_input)
        if not isinstance(output, torch.Tensor) or output.numel() != 1:
            raise RuntimeError("IQA model must return exactly one score per image")

        score = float(output.reshape(-1)[0].detach().cpu().item())
        if not math.isfinite(score):
            raise RuntimeError("IQA model returned a non-finite quality score")
        score = min(max(score, 0.0), 1.0)
        mos_equivalent = self.target_min + score * (
            self.target_max - self.target_min
        )

        passes_quality = score >= self.quality_threshold
        passes_resolution = (
            image_width >= self.minimum_width
            and image_height >= self.minimum_height
        )
        return QualityPrediction(
            quality_score=score,
            mos_equivalent=mos_equivalent,
            suitable=passes_quality and passes_resolution,
            quality_threshold=self.quality_threshold,
            passes_quality_threshold=passes_quality,
            minimum_width=self.minimum_width,
            minimum_height=self.minimum_height,
            passes_resolution_check=passes_resolution,
            image_width=image_width,
            image_height=image_height,
            image_format=image_format,
        )


__all__ = [
    "DEFAULT_MINIMUM_HEIGHT",
    "DEFAULT_MINIMUM_WIDTH",
    "DEFAULT_MODEL_PATH",
    "DEFAULT_QUALITY_THRESHOLD",
    "IQAPredictor",
    "QualityPrediction",
    "resolve_model_path",
]
