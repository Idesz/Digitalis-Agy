"""Online uncertainty scoring from predictive surprise and representation drift."""

from collections import deque
import math
from typing import Optional

import torch


class RollingUncertaintyCalibrator:
    """Maps anomaly scores to rolling empirical percentiles after a warm-up."""

    def __init__(self, window_size: int = 256, min_samples: int = 30):
        if type(window_size) is not int or window_size < 1:
            raise ValueError("window_size must be a positive integer")
        if type(min_samples) is not int or not 1 <= min_samples <= window_size:
            raise ValueError("min_samples must be in [1, window_size]")
        self.window_size = window_size
        self.min_samples = min_samples
        self._history: deque[float] = deque(maxlen=window_size)

    def update(self, score: float) -> tuple[float, bool]:
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise ValueError("score must be a finite number in [0, 1]")
        value = float(score)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("score must be a finite number in [0, 1]")

        sample_count = len(self._history)
        calibrated = sample_count >= self.min_samples
        if calibrated:
            lower = sum(previous < value for previous in self._history)
            ties = sum(previous == value for previous in self._history)
            result = (lower + 0.5 * ties + 0.5) / (sample_count + 1)
        else:
            result = value
        self._history.append(value)
        return float(result), calibrated


def representation_drift(
    previous: Optional[torch.Tensor],
    current: torch.Tensor,
) -> float:
    """Return mean absolute change between bounded firing-rate vectors."""
    if not isinstance(current, torch.Tensor):
        raise ValueError("current representation must be a tensor")
    if (
        current.ndim != 1
        or current.numel() == 0
        or not torch.isfinite(current).all()
        or torch.any((current < 0.0) | (current > 1.0))
    ):
        raise ValueError("current representation must be a finite vector in [0, 1]")
    if previous is None:
        return 0.0
    if not isinstance(previous, torch.Tensor):
        raise ValueError("previous representation must be a tensor")
    if previous.shape != current.shape or previous.ndim != 1:
        raise ValueError("representations must be vectors with matching dimensions")
    if not torch.isfinite(previous).all() or torch.any((previous < 0.0) | (previous > 1.0)):
        raise ValueError("previous representation must be a finite vector in [0, 1]")
    return float(torch.mean(torch.abs(previous - current)).item())
