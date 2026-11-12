"""Adapter for DeepSurv, ``pycox.models.CoxPH`` (extra: ``pycox``).

Neither ``pycox`` nor ``torch`` is imported here: the adapter only calls
methods of an already fitted model.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from timesage.adapters._common import (
    FloatArray,
    apply_scaler,
    as_matrix,
    check_scaler,
    check_times,
    interp_linear,
    predict_surv_df,
    step,
    warn_if_not_cpu,
)

__all__ = ["DeepSurvAdapter"]


class DeepSurvAdapter:
    """Survival predictions from a fitted DeepSurv model (``pycox.models.CoxPH``).

    Parameters
    ----------
    model : pycox.models.CoxPH
        A fitted model on which ``compute_baseline_hazards`` has been called.
    scaler : object with a ``transform`` method, optional
        Preprocessing fitted on the training data (for example a
        ``StandardScaler``), applied to ``X`` before prediction. Pass ``X``
        and the background on the original scale, and importances are
        reported for the original features. Marginal imputation commutes with
        a per-feature affine scaler, so the importances equal those from
        pre-scaled inputs up to float32 rounding. Without a scaler, ``X``
        must already be on the scale the network was trained on.
    legacy_parity : bool, default False
        Reproduce the original research implementation's arithmetic
        (``deepsurv_adapter.py``), which differs from pycox in two ways: it
        interpolates pycox's survival linearly between its time points
        instead of returning the step function pycox estimates, and it clamps
        query times to the first and last training time. Use to reproduce
        results computed with that implementation.

    Notes
    -----
    **Times between, before and after the training times.** pycox estimates
    the baseline hazard (Breslow) at every unique training duration, events
    and censorings alike, and its survival ``exp(-H)`` is a right-continuous
    step function on those times. The adapter returns that step function:

    - it is right-continuous: at a training time ``g_k`` survival already
      includes the drop at ``g_k``;
    - between training times, survival is that of the last training time at
      or before ``t``; it drops only at the event causing the drop;
    - before the first training time, S = 1;
    - after the last training time, survival is held constant.

    This differs from :class:`CoxPHAdapter`, which follows lifelines' linear
    interpolation of the same kind of Breslow estimate, and matches
    :class:`CoxTimeVaryingAdapter`. Each adapter returns what its library's
    own estimate says. With ``legacy_parity=True`` survival is linear between
    training times and query times are clamped to the first and last
    training time (so before the first, S is the value at the first).

    **Evaluation.** Predictions go through ``model.predict_surv_df``, which
    runs the network in ``eval()`` mode without gradients, so dropout and
    batch normalisation are always in inference mode. torchtuples then calls
    ``net.train()`` whatever the previous mode was; the adapter records
    ``net.training`` before each prediction and restores it afterwards, so
    the model is left in the mode it was in. Inputs are cast to float32. The
    model should be on the CPU for reproducible results; a warning is issued
    otherwise, and the model is not moved.
    """

    def __init__(self, model: Any, *, scaler: Any = None, legacy_parity: bool = False) -> None:
        _check_model(model)
        self.model = model
        self.scaler = check_scaler(scaler)
        self.legacy_parity = bool(legacy_parity)

    def predict_survival(self, X: npt.ArrayLike | pd.DataFrame, times: npt.ArrayLike) -> FloatArray:
        """Predict survival probabilities ``S(t | x)``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Covariates, in the column order used for training; on the original
            scale if ``scaler`` is set.
        times : array-like of shape (n_times,)
            Non-negative evaluation times, in any order.

        Returns
        -------
        numpy.ndarray of shape (n_samples, n_times)
            Survival probabilities in ``[0, 1]``, non-increasing in time.
        """
        X_mat = as_matrix(X, None, None, dtype=np.float32)
        t = check_times(times)
        if self.scaler is not None:
            X_mat = apply_scaler(self.scaler, X_mat, None, dtype=np.float32)
        surv = predict_surv_df(self.model, X_mat)  # index: training times; one column per row
        grid = surv.index.to_numpy(dtype=float)
        S_grid = surv.to_numpy(dtype=float).T  # (n, G)
        S = interp_linear(grid, S_grid, t) if self.legacy_parity else step(grid, S_grid, t)
        out: FloatArray = np.clip(S, 0.0, 1.0)
        return out


def _check_model(model: Any) -> None:
    if not callable(getattr(model, "predict_surv_df", None)):
        raise ValueError("model must be a fitted pycox model (missing predict_surv_df).")
    if getattr(model, "baseline_hazards_", None) is None:
        raise ValueError(
            "model has no baseline hazards. Call "
            "model.compute_baseline_hazards(X_train, (durations, events)) first."
        )
    warn_if_not_cpu(model)
