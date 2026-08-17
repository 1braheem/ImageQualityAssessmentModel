"""Small, dependency-free metric helpers for IQA regression."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class RegressionAccumulator:
    """Accumulate MAE and RMSE without retaining every batch."""

    count: int = 0
    absolute_error_sum: float = 0.0
    squared_error_sum: float = 0.0

    def update(self, predictions: Tensor, targets: Tensor) -> None:
        predictions = predictions.detach().float().reshape(-1)
        targets = targets.detach().float().reshape(-1)
        if predictions.numel() != targets.numel():
            raise ValueError("predictions and targets must contain the same number of values")

        errors = predictions - targets
        self.count += errors.numel()
        self.absolute_error_sum += errors.abs().sum().item()
        self.squared_error_sum += errors.square().sum().item()

    def compute(self, *, scale: float = 1.0) -> dict[str, float]:
        if self.count == 0:
            raise ValueError("cannot compute metrics without any observations")
        mae = self.absolute_error_sum / self.count
        rmse = math.sqrt(self.squared_error_sum / self.count)
        return {"mae": mae * scale, "rmse": rmse * scale}


def regression_metrics(
    predictions: Tensor, targets: Tensor, *, scale: float = 1.0
) -> dict[str, float]:
    """Compute MAE and RMSE for tensors, optionally rescaled to MOS points."""

    accumulator = RegressionAccumulator()
    accumulator.update(predictions, targets)
    return accumulator.compute(scale=scale)
