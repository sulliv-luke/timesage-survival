"""Adapter for ``lifelines.CoxPHFitter`` (extra: ``lifelines``).

``lifelines`` is not imported here: the adapter only calls methods of an
already fitted model.
"""

from __future__ import annotations

from collections.abc import Sequence
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
)

__all__ = ["CoxPHAdapter"]


class CoxPHAdapter:
    """Survival predictions from a fitted ``lifelines.CoxPHFitter``.

    Returns exactly what ``model.predict_survival_function(X, times=times)``
    returns, clipped to ``[0, 1]`` as in the original research
    implementation (``cph_adapter.py``). There is no ``legacy_parity`` option
    because that adapter does no arithmetic of its own.

    Parameters
    ----------
    model : lifelines.CoxPHFitter
        A fitted, unstratified model.
    feature_names : sequence of str, optional
        Column order of ``X`` when it is an array. Defaults to the model's
        covariate order, ``model.params_.index``. Must contain exactly the
        model's covariates. ``TimeSAGE`` passes plain arrays in the column
        order of the data given to ``explain``, so if that order differs from
        the model's, set ``feature_names`` to it: a wrong order is not
        detectable from the arrays and silently gives wrong importances.
    scaler : object with a ``transform`` method, optional
        Preprocessing fitted on the training data (for example a
        ``StandardScaler``), applied to ``X`` before prediction. Pass ``X``
        and the background on the original scale, and importances are
        reported for the original features. Marginal imputation commutes with
        a per-feature affine scaler, so the importances equal those from
        pre-scaled inputs up to float32 rounding (``TimeSAGE`` casts inputs to
        float32 before the adapter scales them).

    Notes
    -----
    **Times between, before and after the training times.** lifelines
    estimates the baseline cumulative hazard (Breslow) as a step function on
    all unique training durations, events and censorings alike, and
    interpolates it **linearly** between them (``np.interp``). The adapter
    keeps that convention, so its output matches lifelines everywhere:

    - between training times, survival declines linearly in the cumulative
      hazard, so part of an event's drop appears before the event time. On the
      9-feature METABRIC model this differs from the step function by up to
      0.065 between training times and 0.003 on the 50-point grid of
      ``examples/coxph_metabric.py``;
    - before the first training time, lifelines returns the value at that
      time: S = 1 if the earliest duration is a censoring, slightly below 1 if
      it is an event;
    - after the last training time, survival is held constant.

    The other adapters follow their own libraries' estimates, which are step
    functions: :class:`CoxTimeVaryingAdapter` and ``DeepSurvAdapter`` do not
    interpolate. A one-bin time-varying Cox model therefore agrees with this
    adapter at observed times but not between them.
    """

    def __init__(
        self,
        model: Any,
        feature_names: Sequence[str] | None = None,
        *,
        scaler: Any = None,
    ) -> None:
        self.model = model
        model_names = _check_model(model)
        self.feature_names = _check_feature_names(feature_names, model_names)
        self.scaler = check_scaler(scaler)

    def predict_survival(self, X: npt.ArrayLike | pd.DataFrame, times: npt.ArrayLike) -> FloatArray:
        """Predict survival probabilities ``S(t | x)``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or pandas.DataFrame
            Covariates, on the original scale if ``scaler`` is set. A
            DataFrame is aligned by ``feature_names``; an array must have its
            columns in that order.
        times : array-like of shape (n_times,)
            Non-negative evaluation times, in any order.

        Returns
        -------
        numpy.ndarray of shape (n_samples, n_times)
            Survival probabilities in ``[0, 1]``, non-increasing in time.
        """
        X_mat = as_matrix(X, self.feature_names, len(self.feature_names))
        t = check_times(times)
        if self.scaler is not None:
            X_mat = apply_scaler(self.scaler, X_mat, self.feature_names)
        X_df = pd.DataFrame(X_mat, columns=list(self.feature_names), copy=False)
        sf = self.model.predict_survival_function(X_df, times=t)  # (n_times, n_samples)
        S: FloatArray = np.clip(sf.T.to_numpy(dtype=float), 0.0, 1.0)
        return S


def _check_model(model: Any) -> tuple[str, ...]:
    params = getattr(model, "params_", None)
    if not isinstance(params, pd.Series) or not callable(
        getattr(model, "predict_survival_function", None)
    ):
        raise ValueError(
            "model must be a fitted lifelines CoxPHFitter "
            "(missing params_ or predict_survival_function)."
        )
    if getattr(model, "strata", None):
        raise ValueError(f"Stratified models are not supported (strata={model.strata!r}).")
    return tuple(str(f) for f in params.index)


def _check_feature_names(
    feature_names: Sequence[str] | None, model_names: tuple[str, ...]
) -> tuple[str, ...]:
    if feature_names is None:
        return model_names
    names = tuple(str(f) for f in feature_names)
    if len(set(names)) != len(names) or sorted(names) != sorted(model_names):
        raise ValueError(
            "feature_names must contain exactly the model's covariates.\n"
            f"  feature_names: {list(names)}\n"
            f"  model.params_: {list(model_names)}"
        )
    return names
