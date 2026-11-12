r"""Coalition value function with marginal imputation (private).

Implements the subset predictor of Eq. 13 of the dissertation,

    S_hat_S(t | x_S) = E_{X_{M\S} ~ p(X_{M\S})} [ S_hat(t | x_S, X_{M\S}) ],

approximated by averaging predictions over a fixed background sample whose
rows replace the features outside ``S``. Replacement values do not depend on
``x_S``, so a feature the model ignores never changes the prediction and gets
exactly zero importance. The per-sample IPCW Brier loss of the imputed
predictions (Eq. 14-15) is the negative coalition value ``-v_t(S)`` per row.

Ported from the joint-times value function ``TimeSAGE._fit_joint`` of the
original research implementation (``time_sage_survival.py``), which
combined ``sage.imputers.MarginalImputer`` (vendored in
:mod:`timesage._vendor.sage`) with ``ipcw_brier_loss_multi``.

The whole background is used on every evaluation; nothing is resampled. Any
subsampling of a larger reference set, and any float32 casting, is the
caller's responsibility (the original ``TimeSAGE.fit`` cast both to float32).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from timesage._ipcw import CensoringKM, ipcw_brier_loss
from timesage._vendor.sage import MarginalImputer

__all__ = ["CoalitionValue", "PredictSurvivalFn"]

PredictSurvivalFn = Callable[[NDArray[Any], NDArray[np.float64]], ArrayLike]
"""``predict_survival(X, times) -> (n_samples, n_times)`` survival probabilities."""


class CoalitionValue:
    """Joint-times coalition value function under marginal imputation (Eq. 13-15).

    Parameters
    ----------
    predict_survival : callable
        ``predict_survival(X, times)`` returning survival probabilities of
        shape ``(len(X), len(times))``.
    background : array_like of shape (m, p)
        Background rows used to impute features outside the coalition. Copied;
        its dtype is preserved.
    times : array_like of shape (K,)
        Evaluation horizons ``t_1..t_K``.
    censoring : CensoringKM
        Fitted censoring survival ``G_hat`` used by the IPCW loss.

    Raises
    ------
    ValueError
        If ``background`` is not a non-empty 2-D array or ``times`` is not a
        non-empty 1-D array. Also, on evaluation, if ``predict_survival``
        returns the wrong shape or non-finite values.
    """

    def __init__(
        self,
        predict_survival: PredictSurvivalFn,
        background: ArrayLike,
        times: ArrayLike,
        censoring: CensoringKM,
    ) -> None:
        bg = np.array(background)
        if bg.ndim != 2 or len(bg) == 0:
            raise ValueError(
                f"background must be a non-empty 2-D array (m, p); got shape {bg.shape}."
            )
        t_grid = np.array(times, dtype=float)
        if t_grid.ndim != 1 or len(t_grid) == 0:
            raise ValueError(f"times must be a non-empty 1-D array; got shape {t_grid.shape}.")

        self._times: NDArray[np.float64] = t_grid
        self._censoring = censoring
        self._censoring_at_times = censoring(t_grid)

        n_times = len(t_grid)

        def model_all_times(X_in: NDArray[Any]) -> NDArray[Any]:
            out = np.asarray(predict_survival(X_in, t_grid))
            if out.shape != (len(X_in), n_times):
                raise ValueError(
                    f"predict_survival returned shape {out.shape}; "
                    f"expected {(len(X_in), n_times)} (n_samples, n_times)."
                )
            # Cheap next to the model call; a NaN here would silently corrupt phi.
            if not np.all(np.isfinite(out)):
                raise ValueError(
                    "predict_survival returned non-finite values on imputed rows, which "
                    "combine feature values from X with values from the background. The "
                    "model probably fails on feature combinations unlike the data it was "
                    "fitted on (marginal imputation evaluates off the data manifold)."
                )
            return out

        self._imputer = MarginalImputer(model_all_times, bg)

    @property
    def num_features(self) -> int:
        """Number of features ``p``."""
        return int(self._imputer.num_groups)

    @property
    def n_background(self) -> int:
        """Number of background rows ``m``."""
        return int(self._imputer.samples)

    @property
    def times(self) -> NDArray[np.float64]:
        """Evaluation horizons (a copy)."""
        out: NDArray[np.float64] = self._times.copy()
        return out

    def predict(self, x: ArrayLike, mask: ArrayLike) -> NDArray[np.float64]:
        """Imputed survival predictions ``S_hat_S(t_k | x_S)`` (Eq. 13).

        Parameters
        ----------
        x : array_like of shape (n, p)
            Rows to evaluate.
        mask : array_like of bool, shape (n, p)
            ``mask[i, j]`` is True if feature ``j`` is in row ``i``'s coalition.
            Rows may have different coalitions.

        Returns
        -------
        ndarray of shape (n, K)
            Predictions averaged over the background rows.
        """
        x_arr = np.asarray(x)
        mask_arr = np.asarray(mask)
        p = self.num_features
        if x_arr.ndim != 2 or x_arr.shape[1] != p:
            raise ValueError(
                f"x must have shape (n, {p}) to match the background's {p} features; "
                f"got {x_arr.shape}."
            )
        if mask_arr.dtype != np.bool_:
            raise TypeError(f"mask must be a boolean array; got dtype {mask_arr.dtype}.")
        if mask_arr.shape != x_arr.shape:
            raise ValueError(
                f"mask must have the same shape as x {x_arr.shape}; got {mask_arr.shape}."
            )
        pred: NDArray[np.float64] = self._imputer(x_arr, mask_arr)
        return pred

    def loss(
        self, x: ArrayLike, mask: ArrayLike, time: ArrayLike, event: ArrayLike
    ) -> NDArray[np.float64]:
        """Per-sample IPCW Brier loss of the coalition's predictions (Eq. 14-15).

        Averaging over rows gives ``BS_IPCW(t_k; S) = -v_{t_k}(S)``.

        Parameters
        ----------
        x : array_like of shape (n, p)
            Rows to evaluate.
        mask : array_like of bool, shape (n, p)
            Coalition membership per row.
        time : array_like of shape (n,)
            Observed times of the rows.
        event : array_like of shape (n,)
            Event indicators of the rows (1 = event, 0 = censored).

        Returns
        -------
        ndarray of shape (n, K)
            IPCW Brier loss per row and horizon.
        """
        return ipcw_brier_loss(
            self.predict(x, mask),
            time,
            event,
            self._times,
            self._censoring,
            self._censoring_at_times,
        )
