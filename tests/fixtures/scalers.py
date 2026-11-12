"""A minimal stand-in for a fitted scikit-learn ``StandardScaler``."""

from __future__ import annotations

import numpy as np


class AffineScaler:
    """``(x - mean) / scale``; anything with ``transform`` is accepted as ``scaler=``."""

    def __init__(self, mean: np.ndarray, scale: np.ndarray) -> None:
        self.mean, self.scale = mean, scale

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (np.asarray(X) - self.mean) / self.scale
