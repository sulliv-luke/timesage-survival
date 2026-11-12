"""Input handling shared by the adapters (private)."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

FloatArray = npt.NDArray[np.float64]


def check_times(times: npt.ArrayLike) -> FloatArray:
    """Return times as a 1-D float64 array; finite and non-negative, in any order."""
    t = np.asarray(times, dtype=float)
    if t.ndim != 1:
        raise ValueError(f"times must be one-dimensional, got shape {t.shape}.")
    if not np.all(np.isfinite(t)) or np.any(t < 0):
        raise ValueError("times must be finite and non-negative.")
    return t


def as_matrix(
    X: npt.ArrayLike | pd.DataFrame,
    feature_names: Sequence[str] | None,
    n_features: int | None,
    dtype: type[np.floating[Any]] = np.float64,
) -> npt.NDArray[Any]:
    """Return ``X`` as a 2-D array of ``dtype`` with columns in model order.

    A DataFrame is aligned by ``feature_names`` when they are known; an array
    must already have its columns in that order. ``n_features=None`` accepts
    any number of columns.
    """
    if isinstance(X, pd.DataFrame) and feature_names is not None:
        missing = [f for f in feature_names if f not in X.columns]
        if missing:
            raise ValueError(f"X is missing feature columns: {missing}.")
        X_mat: npt.NDArray[Any] = X.loc[:, list(feature_names)].to_numpy(dtype=dtype)
    else:
        X_mat = np.asarray(X, dtype=dtype)
    if X_mat.ndim != 2 or (n_features is not None and X_mat.shape[1] != n_features):
        width = "n_features" if n_features is None else n_features
        raise ValueError(f"X must have shape (n_samples, {width}), got {X_mat.shape}.")
    if not np.all(np.isfinite(X_mat)):
        raise ValueError("X contains non-finite values.")
    return X_mat


def check_scaler(scaler: Any) -> Any:
    if scaler is not None and not callable(getattr(scaler, "transform", None)):
        raise TypeError(
            "scaler must have a transform(X) method (for example a fitted StandardScaler)."
        )
    return scaler


def apply_scaler(
    scaler: Any,
    X: npt.NDArray[Any],
    feature_names: Sequence[str] | None,
    dtype: type[np.floating[Any]] = np.float64,
) -> npt.NDArray[Any]:
    """Return ``scaler.transform(X)`` as a ``dtype`` array of the same shape.

    A scaler fitted on a DataFrame (scikit-learn sets ``feature_names_in_``)
    receives a DataFrame with ``feature_names``, or with its own names when the
    adapter does not know them, so that scikit-learn checks the column order.
    """
    names = getattr(scaler, "feature_names_in_", None)
    X_in: npt.NDArray[Any] | pd.DataFrame = X
    if names is not None:
        X_in = pd.DataFrame(X, columns=list(feature_names if feature_names is not None else names))
    out = np.asarray(scaler.transform(X_in), dtype=dtype)
    if out.shape != X.shape:
        raise ValueError(f"scaler.transform changed the shape of X: {X.shape} -> {out.shape}.")
    if not np.all(np.isfinite(out)):
        raise ValueError("scaler.transform returned non-finite values.")
    return out


def step(grid: FloatArray, S_grid: FloatArray, t: FloatArray) -> FloatArray:
    """Right-continuous step function through ``(grid, S_grid)``; 1 before ``grid[0]``.

    ``S_grid`` has shape (n_samples, len(grid)); the result (n_samples, len(t)).
    Constant after ``grid[-1]``.
    """
    idx = np.searchsorted(grid, t, side="right") - 1
    out: FloatArray = np.where(idx[None, :] >= 0, S_grid[:, np.maximum(idx, 0)], 1.0)
    return out


def interp_linear(grid: FloatArray, S_grid: FloatArray, t: FloatArray) -> FloatArray:
    """Linear interpolation in S, with ``t`` clamped to ``[grid[0], grid[-1]]``.

    The arithmetic of ``deepsurv_adapter.py`` and ``deephit_adapter.py`` of the
    original research implementation, in the same order, so results are
    bit-identical to it.
    """
    t_query = np.clip(t, float(grid[0]), float(grid[-1]))
    out = np.empty((S_grid.shape[0], len(t_query)), dtype=float)
    for i in range(S_grid.shape[0]):
        out[i] = np.interp(t_query, grid, S_grid[i])
    return out


def warn_if_not_cpu(model: Any) -> None:
    """Warn, without moving it, if a torchtuples model is not on the CPU."""
    device = getattr(model, "device", None)
    if device is not None and getattr(device, "type", str(device)) != "cpu":
        warnings.warn(
            f"model is on device {device!s}; results are reproducible only on the CPU. "
            "The adapter does not move the model.",
            UserWarning,
            stacklevel=4,  # warn_if_not_cpu <- _check_model <- __init__ <- caller
        )


def predict_surv_df(model: Any, X: npt.NDArray[Any]) -> pd.DataFrame:
    """``model.predict_surv_df(X)``, leaving ``model.net`` in the mode it was in.

    pycox predicts in ``eval()`` mode without gradients, but torchtuples then
    calls ``net.train()`` whatever the previous mode was. Restoring the mode
    keeps the adapter from changing the user's model.
    """
    net = getattr(model, "net", None)
    was_training = getattr(net, "training", None)
    try:
        surv: pd.DataFrame = model.predict_surv_df(X)
    finally:
        if net is not None and was_training is not None:
            net.train(bool(was_training))
    return surv
