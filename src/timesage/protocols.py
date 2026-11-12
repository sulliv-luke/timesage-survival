"""The ``SurvivalModel`` protocol (public).

The single contract between TimeSAGE and any model: a method
``predict_survival(X, times)`` returning survival probabilities of shape
``(n_samples, n_times)`` in ``[0, 1]`` and non-increasing in time. A plain
callable with the same signature is also accepted wherever a ``SurvivalModel``
is.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = ["SurvivalModel"]


@runtime_checkable
class SurvivalModel(Protocol):
    """A fitted survival model that predicts ``S_hat(t | x)``."""

    def predict_survival(self, X: NDArray[Any], times: NDArray[np.float64]) -> ArrayLike:
        """Return ``S_hat(t | x)`` for every row of ``X`` and every time in ``times``.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Covariates. TimeSAGE passes float32 arrays, with columns in the
            order of the explained data.
        times : ndarray of shape (n_times,)
            Evaluation times, increasing.

        Returns
        -------
        array_like of shape (n_samples, n_times)
            Survival probabilities in ``[0, 1]``, non-increasing along the
            time axis.
        """
        ...
