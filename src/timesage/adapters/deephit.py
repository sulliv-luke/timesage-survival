"""Adapter for DeepHit, ``pycox.models.DeepHitSingle`` (extra: ``pycox``).

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
    warn_if_not_cpu,
)

__all__ = ["DeepHitAdapter"]


class DeepHitAdapter:
    """Survival predictions from a fitted DeepHit model (``pycox.models.DeepHitSingle``).

    Parameters
    ----------
    model : pycox.models.DeepHitSingle
        A fitted single-risk model with ``duration_index`` set to the
        discretisation cuts used for training (``labtrans.cuts``).
    scaler : object with a ``transform`` method, optional
        Preprocessing fitted on the training data (for example a
        ``StandardScaler``), applied to ``X`` before prediction. Pass ``X``
        and the background on the original scale, and importances are
        reported for the original features. Marginal imputation commutes with
        a per-feature affine scaler, so the importances equal those from
        pre-scaled inputs up to float32 rounding. Without a scaler, ``X``
        must already be on the scale the network was trained on.

    Notes
    -----
    **A discrete-time model.** DeepHit predicts a probability mass function
    over its cuts ``c_0 = 0 < c_1 < ... < c_m`` (75 equidistant cuts in the
    original study), plus the mass beyond the last cut. Survival
    ``S(c_k) = 1 - sum_{l <= k} pmf_l`` is defined only at the cuts; any value
    in between is an interpolation scheme, which is a modelling choice.

    The adapter interpolates **linearly in S** between cuts. This is pycox's
    default scheme, ``const_pdf`` (``model.interpolate()``, constant density
    within each interval), and the original research implementation's
    (``deephit_adapter.py``), so there is no ``legacy_parity`` option.
    Log-linear interpolation (constant hazard within each interval,
    ``const_hazard``, which pycox 0.3 does not implement for this model) or
    the step function through the cuts would give different values between
    cuts.

    - **At 0:** ``S(0) = 1 - pmf_0`` is below 1. pycox maps an event in
      ``(c_0, c_1]`` to cut 1, so no training event lies at cut 0, but the
      softmax output still gives cut 0 positive mass. A grid that includes 0
      sees ``S < 1`` there, and grid points before ``c_1`` are interpolated
      from it, so survival at a grid's first point is not exactly 1 even
      when that point is early.
    - **Between cuts:** linear in S, as above.
    - **After the last cut:** survival is held at its value there, which is
      the mass beyond the last cut (not necessarily 0). The model says
      nothing about the shape of survival there.

    **Evaluation.** Predictions go through ``model.predict_surv_df``, which
    runs the network in ``eval()`` mode without gradients, so dropout and
    batch normalisation are always in inference mode. torchtuples then calls
    ``net.train()`` whatever the previous mode was; the adapter records
    ``net.training`` before each prediction and restores it afterwards, so
    the model is left in the mode it was in. Inputs are cast to float32. The
    model should be on the CPU for reproducible results; a warning is issued
    otherwise, and the model is not moved.
    """

    def __init__(self, model: Any, *, scaler: Any = None) -> None:
        _check_model(model)
        self.model = model
        self.scaler = check_scaler(scaler)

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
        surv = predict_surv_df(self.model, X_mat)  # index: the cuts; one column per row
        grid = surv.index.to_numpy(dtype=float)
        S_grid = surv.to_numpy(dtype=float).T  # (n, n_cuts)
        out: FloatArray = np.clip(interp_linear(grid, S_grid, t), 0.0, 1.0)
        return out


def _check_model(model: Any) -> None:
    if not callable(getattr(model, "predict_surv_df", None)):
        raise ValueError("model must be a fitted pycox model (missing predict_surv_df).")
    cuts = getattr(model, "duration_index", None)
    if cuts is None:
        raise ValueError(
            "model.duration_index is not set, so pycox would index survival by cut "
            "number instead of time. Pass duration_index=labtrans.cuts when building "
            "the model."
        )
    cuts = np.asarray(cuts, dtype=float)
    if cuts.ndim != 1 or cuts.size < 2 or np.any(np.diff(cuts) <= 0):
        raise ValueError("model.duration_index must be strictly increasing cut points.")
    warn_if_not_cpu(model)
