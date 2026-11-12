r"""Adapter for ``lifelines.CoxTimeVaryingFitter`` with piecewise bins (extra: ``lifelines``).

The model is a Cox regression whose coefficients are piecewise constant in time:
on bin ``b = (c_b, c_{b+1}]`` the log partial hazard is ``x' beta_b``. It is fitted
with ``CoxTimeVaryingFitter`` on long-format data in which each baseline covariate
``f`` is replicated as a column ``"{f}__bin{b}"`` that equals ``x_f`` on the
subject's bin-``b`` interval and 0 elsewhere.

Survival is obtained by piecewise integration of lifelines' Breslow baseline
cumulative hazard:

.. math::

    H(t \mid x) = \sum_b \exp\big((x - \bar x)'\beta_b\big)
        \big[\hat H_0(\min(t, c_{b+1})) - \hat H_0(c_b)\big]_{t > c_b},
    \qquad S(t \mid x) = \exp(-H(t \mid x)),

where ``(x - x_bar)' beta_b`` is exactly what lifelines' ``predict_log_partial_hazard``
returns for the bin-``b`` long-format row. lifelines centres the long-format
design at its row mean before building ``baseline_cumulative_hazard_``, so the
centring is required for the baseline and the linear predictor to match.

``lifelines`` is not imported here: the adapter only uses attributes of an
already fitted model.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from timesage.adapters._common import apply_scaler, as_matrix, check_scaler, check_times

__all__ = ["CoxTimeVaryingAdapter"]

FloatArray = npt.NDArray[np.float64]


class CoxTimeVaryingAdapter:
    """Survival predictions from a fitted ``lifelines.CoxTimeVaryingFitter``.

    Parameters
    ----------
    model : lifelines.CoxTimeVaryingFitter
        A fitted, unstratified model whose covariates are exactly the
        bin-specific columns ``"{feature}__bin{b}"`` for every feature in
        ``feature_names`` and every bin ``b = 0, ..., len(cut_points) - 2``.
    cut_points : sequence of float
        Bin boundaries used to build the training long-format data, starting
        at 0 and strictly increasing. Bin ``b`` is the interval
        ``(cut_points[b], cut_points[b + 1]]``, matching lifelines' risk sets
        ``start < t <= stop``.
    feature_names : sequence of str
        Original (wide) feature names, in the column order of ``X`` passed to
        :meth:`predict_survival` when ``X`` is an array.
    legacy_parity : bool, default False
        Reproduce the original research implementation's arithmetic
        (``coxtv_adapter.py``), which differs from lifelines in four ways: an
        uncentred linear predictor, the first baseline jump dropped, linear
        interpolation of the baseline between event times, and clipping of
        the linear predictor to ``[-legacy_eta_clip, legacy_eta_clip]``. Use
        to reproduce results computed with that implementation.
    legacy_eta_clip : float, default 10.0
        Clip applied to the linear predictor when ``legacy_parity=True``.
        Ignored otherwise.
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
    With ``legacy_parity=False`` (the default) predictions are faithful to the
    fitted lifelines model:

    - the bin-``b`` log partial hazard is ``x' beta_b - c`` with
      ``c = x_bar' beta`` taken from lifelines itself, as minus
      ``predict_log_partial_hazard`` of an all-zero long-format row, so it is
      identical to ``predict_log_partial_hazard`` on the bin-``b`` row;
    - ``H_0`` is lifelines' ``baseline_cumulative_hazard_`` evaluated as the
      right-continuous step function it estimates (0 before the first event
      time, constant after the last);
    - no clipping is applied.

    Between event times survival is therefore a step function: it drops
    only at the event that causes the drop. For a single bin this coincides
    with ``CoxPHFitter`` fitted on the equivalent wide data at every observed
    time, but not between observed times, where lifelines (and
    :class:`CoxPHAdapter`, which follows it) interpolates the baseline
    cumulative hazard linearly.
    """

    def __init__(
        self,
        model: Any,
        cut_points: Sequence[float] | npt.ArrayLike,
        feature_names: Sequence[str],
        *,
        legacy_parity: bool = False,
        legacy_eta_clip: float = 10.0,
        scaler: Any = None,
    ) -> None:
        self.model = model
        self.scaler = check_scaler(scaler)
        self.feature_names = _check_feature_names(feature_names)
        self.cut_points = _check_cut_points(cut_points)
        self.legacy_parity = bool(legacy_parity)
        self.legacy_eta_clip = float(legacy_eta_clip)
        if self.legacy_parity and not self.legacy_eta_clip > 0:
            raise ValueError("legacy_eta_clip must be positive.")

        params, H0 = _check_model(model)
        n_bins = len(self.cut_points) - 1
        self._betas = _bin_coefficients(params, self.feature_names, n_bins)  # (n_bins, p)
        self._H0_times = H0.index.to_numpy(dtype=float)
        self._H0_vals = H0.iloc[:, 0].to_numpy(dtype=float)
        if self._H0_times.size == 0:
            raise ValueError("model.baseline_cumulative_hazard_ is empty (no events?).")

        if self.legacy_parity:
            self.centring_constant = 0.0
            # Precomputation of the original coxtv_adapter.py.
            t_min, t_max = float(self._H0_times[0]), float(self._H0_times[-1])
            self._legacy_t_range = (t_min, t_max)
            self._legacy_cuts = np.clip(self.cut_points, t_min, t_max)
            self._legacy_H0_at_cuts = np.interp(self._legacy_cuts, self._H0_times, self._H0_vals)
        else:
            self.centring_constant = _centring_constant(model, params)
            self._H0_at_cuts = self._step_H0(self.cut_points)

    # ------------------------------------------------------------------ public

    def predict_survival(self, X: npt.ArrayLike | pd.DataFrame, times: npt.ArrayLike) -> FloatArray:
        """Predict survival probabilities ``S(t | x)``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or pandas.DataFrame
            Covariates, on the original scale if ``scaler`` is set (otherwise
            on the scale used for fitting). A DataFrame is aligned by
            ``feature_names``; an array must have its columns in that order.
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
        if self.legacy_parity:
            return self._predict_legacy(X_mat, t)
        return self._predict(X_mat, t)

    # ----------------------------------------------------------------- private

    def _step_H0(self, t: FloatArray) -> FloatArray:
        """Right-continuous step evaluation of lifelines' baseline cumulative hazard."""
        idx = np.searchsorted(self._H0_times, t, side="right") - 1
        out: FloatArray = np.where(idx >= 0, self._H0_vals[np.maximum(idx, 0)], 0.0)
        return out

    def _predict(self, X: FloatArray, t: FloatArray) -> FloatArray:
        cuts = self.cut_points
        n_bins = len(cuts) - 1
        # dH0[b, k] = H0(min(t_k, c_{b+1})) - H0(c_b) for t_k > c_b, else 0.
        upper = np.minimum(t[None, :], cuts[1:, None])  # (n_bins, K)
        dH0 = self._step_H0(upper.ravel()).reshape(n_bins, -1) - self._H0_at_cuts[:-1, None]
        dH0 = np.where(t[None, :] > cuts[:-1, None], dH0, 0.0)

        # (n, n_bins): lifelines' log partial hazard of each bin's long-format row.
        eta = X @ self._betas.T - self.centring_constant
        with np.errstate(over="ignore"):
            risk = np.exp(eta)
        # exp overflow gives inf, and inf * 0 = nan for bins not yet reached;
        # capping at the largest float keeps those terms 0 and the rest inf.
        risk = np.minimum(risk, np.finfo(np.float64).max)
        with np.errstate(over="ignore"):
            H = risk @ dH0
        S: FloatArray = np.exp(-H)
        return S

    def _predict_legacy(self, X: FloatArray, t: FloatArray) -> FloatArray:
        # Arithmetic copied from the original coxtv_adapter.py, in the same
        # order, so results are bit-identical to the original implementation.
        t_min, t_max = self._legacy_t_range
        times_clipped = np.clip(t, t_min, t_max)
        H0_at_t = np.interp(times_clipped, self._H0_times, self._H0_vals)
        eta_bins = X @ self._betas.T
        exp_eta_bins = np.exp(np.clip(eta_bins, -self.legacy_eta_clip, self.legacy_eta_clip))
        H = np.zeros((X.shape[0], len(times_clipped)), dtype=float)
        cuts = self._legacy_cuts
        H0_cuts = self._legacy_H0_at_cuts
        for b in range(self._betas.shape[0]):
            left = float(cuts[b])
            right = float(cuts[b + 1])
            H0_left = float(H0_cuts[b])
            H0_right = float(H0_cuts[b + 1])
            contrib = np.zeros(len(times_clipped), dtype=float)
            mask_mid = (times_clipped > left) & (times_clipped < right)
            contrib[mask_mid] = H0_at_t[mask_mid] - H0_left
            mask_hi = times_clipped >= right
            contrib[mask_hi] = H0_right - H0_left
            H += exp_eta_bins[:, [b]] * contrib[np.newaxis, :]
        S: FloatArray = np.clip(np.exp(-H), 0.0, 1.0)
        return S


def _check_feature_names(feature_names: Sequence[str]) -> tuple[str, ...]:
    names = tuple(str(f) for f in feature_names)
    if not names:
        raise ValueError("feature_names must not be empty.")
    if len(set(names)) != len(names):
        raise ValueError("feature_names must be unique.")
    return names


def _check_cut_points(cut_points: Sequence[float] | npt.ArrayLike) -> FloatArray:
    cuts = np.asarray(cut_points, dtype=float)
    if cuts.ndim != 1 or cuts.size < 2:
        raise ValueError("cut_points must be one-dimensional with at least two entries.")
    if not np.all(np.isfinite(cuts)):
        raise ValueError("cut_points must be finite.")
    if cuts[0] != 0.0:
        raise ValueError("cut_points must start at 0.0, as in the training long format.")
    if np.any(np.diff(cuts) <= 0):
        raise ValueError("cut_points must be strictly increasing.")
    return cuts


def _check_model(model: Any) -> tuple[pd.Series, pd.DataFrame]:
    params = getattr(model, "params_", None)
    H0 = getattr(model, "baseline_cumulative_hazard_", None)
    if not isinstance(params, pd.Series) or not isinstance(H0, pd.DataFrame):
        raise ValueError(
            "model must be a fitted lifelines CoxTimeVaryingFitter "
            "(missing params_ or baseline_cumulative_hazard_)."
        )
    if H0.shape[1] != 1:
        raise ValueError(
            "Stratified models (several baseline hazards) are not supported: "
            f"baseline_cumulative_hazard_ has {H0.shape[1]} columns."
        )
    if not H0.index.is_monotonic_increasing:
        raise ValueError("baseline_cumulative_hazard_ index must be increasing.")
    return params, H0


def _bin_coefficients(params: pd.Series, feature_names: tuple[str, ...], n_bins: int) -> FloatArray:
    expected = [f"{f}__bin{b}" for b in range(n_bins) for f in feature_names]
    present = set(params.index)
    missing = [k for k in expected if k not in present]
    unexpected = sorted(present - set(expected))
    if missing or unexpected:
        raise ValueError(
            "model covariates must be exactly '{feature}__bin{b}' for every feature and bin; "
            f"missing={missing[:10]}, unexpected={unexpected[:10]}."
        )
    betas: FloatArray = np.array(
        [[float(params[f"{f}__bin{b}"]) for f in feature_names] for b in range(n_bins)],
        dtype=float,
    )
    return betas


def _centring_constant(model: Any, params: pd.Series) -> float:
    """``c = x_bar' beta`` as lifelines applies it: minus the log partial hazard of a zero row."""
    zero_row = pd.DataFrame(np.zeros((1, len(params))), columns=params.index)
    log_ph = model.predict_log_partial_hazard(zero_row)
    return -float(np.asarray(log_ph, dtype=float).ravel()[0])
