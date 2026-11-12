r"""Attribution-quality metrics and the stable support window (public).

Implements the evaluation quantities of Section 7.7.3 of the dissertation:

- :func:`stable_window`, the stable support window of Eq. 31,
  ``W = {t : R_hat(t) >= 0.10 and G_hat(t) >= 0.05}``;
- comparisons of an estimated importance curve with a reference curve (for
  example an oracle's): :func:`curve_correlation`, :func:`nwiae`,
  :func:`high_importance_iou` and :func:`peak_time_error`;
- :func:`rank_correlation` of integrated importances.

Ported from ``metrics.py`` and ``truth.py`` of the original synthetic study.
Study-specific aggregates (for example the mean NWIAE over the signal features
x1-x4) are not part of the package.

**Undefined values.** Where a metric is undefined (an empty window, a constant
curve, a reference curve with no mass), the functions return NaN and emit a
``UserWarning``, so that a NaN in a results table is not missed. The original
research implementation returned NaN in some of these cases and a number in
others; ``legacy_parity=True`` reproduces its numbers where they differ.

**``legacy_parity``** means what it means across the package: reproduce the
original research implementation's arithmetic where the package deliberately
departs from it (see ``CONVENTIONS.md``). Here it selects that
implementation's at-risk fraction ``1{Y > t}`` in :func:`stable_window` and its
values for undefined cases.
"""

from __future__ import annotations

import warnings
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import pearsonr, spearmanr

from timesage._ipcw import _as_survival_data, fit_censoring_km
from timesage.estimator import _validate_grid
from timesage.integrate import _trapezoid_weights

__all__ = [
    "curve_correlation",
    "high_importance_iou",
    "nwiae",
    "peak_time_error",
    "rank_correlation",
    "stable_window",
]

_CONSTANT_STD = 1e-12  # as _safe_corr: a curve with std <= 1e-12 is constant
_NWIAE_EPS = 1e-12  # as normalized_weighted_curve_error
_PEAK_FLOOR = 1e-12  # as high_importance_region


def _undefined(metric: str, reason: str) -> None:
    warnings.warn(f"{metric} is undefined: {reason}; returning NaN.", UserWarning, stacklevel=3)


def _curve(values: ArrayLike, name: str, n: int | None = None) -> NDArray[np.float64]:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1 or len(arr) == 0:
        raise ValueError(f"{name} must be a non-empty 1-D array; got shape {arr.shape}.")
    if n is not None and len(arr) != n:
        raise ValueError(f"{name} has {len(arr)} values; expected {n}.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values.")
    return arr


def _pair(estimate: ArrayLike, reference: ArrayLike) -> tuple[NDArray[np.float64], ...]:
    est = _curve(estimate, "estimate")
    return est, _curve(reference, "reference", len(est))


def _window(window: ArrayLike | None, n: int) -> NDArray[np.bool_] | None:
    if window is None:
        return None
    mask = np.asarray(window)
    if mask.dtype != np.bool_:
        raise TypeError(f"window must be a boolean array; got dtype {mask.dtype}.")
    if mask.shape != (n,):
        raise ValueError(f"window must have shape ({n},); got {mask.shape}.")
    return mask


# ---------------------------------------------------------------- stable window


def stable_window(
    times: ArrayLike,
    time: ArrayLike,
    event: ArrayLike,
    *,
    censoring_time: ArrayLike | None = None,
    censoring_event: ArrayLike | None = None,
    min_at_risk: float = 0.10,
    min_censoring_survival: float = 0.05,
    legacy_parity: bool = False,
) -> NDArray[np.bool_]:
    """Stable support window on a time grid (Eq. 31).

    ``W = {t_k : R_hat(t_k) >= min_at_risk and G_hat(t_k) >= min_censoring_survival}``,
    where ``R_hat`` is the at-risk fraction of ``time`` and ``G_hat`` the
    censoring Kaplan-Meier curve (Eq. 16). Late horizons outside ``W`` have a
    small risk set or large IPCW weights, so curve metrics there are noisy.

    Parameters
    ----------
    times : array_like of shape (K,)
        Strictly increasing evaluation grid.
    time, event : array_like of shape (n,)
        Observed times and event indicators (1 = event) of the evaluated rows.
        ``R_hat`` is computed from ``time``; ``G_hat`` from ``(time, event)``
        unless censoring data are given.
    censoring_time, censoring_event : array_like of shape (n_c,), optional
        Data for ``G_hat`` instead, given together. The original synthetic
        study fitted ``G_hat`` on the training set and ``R_hat`` on
        the test set: ``stable_window(grid, T_test, E_test,
        censoring_time=T_train, censoring_event=E_train, legacy_parity=True)``
        reproduces its window exactly.
    min_at_risk, min_censoring_survival : float
        Thresholds; the defaults are those of Eq. 31.
    legacy_parity : bool, default False
        Use the original implementation's ``R_hat(t) = (1/n) sum_i 1{Y_i > t}``
        instead of the textbook ``1{Y_i >= t}``. The two differ only at grid times
        equal to an observed time.

    Returns
    -------
    ndarray of bool, shape (K,)
        ``True`` inside the window. A ``UserWarning`` is emitted if it is empty.
    """
    grid = _validate_grid(times)
    T, delta = _as_survival_data(time, event)
    if (censoring_time is None) != (censoring_event is None):
        raise ValueError("Pass censoring_time and censoring_event together, or neither.")
    if censoring_time is not None and censoring_event is not None:
        censoring = fit_censoring_km(*_as_survival_data(censoring_time, censoring_event))
    else:
        censoring = fit_censoring_km(T, delta)
    if legacy_parity:
        at_risk = np.mean(T[:, None] > grid[None, :], axis=0)
    else:
        at_risk = np.mean(T[:, None] >= grid[None, :], axis=0)
    return _stable_mask(at_risk, censoring(grid), min_at_risk, min_censoring_survival)


def _stable_mask(
    at_risk: NDArray[np.float64],
    censoring_survival: NDArray[np.float64],
    min_at_risk: float,
    min_censoring_survival: float,
) -> NDArray[np.bool_]:
    """Shared by :func:`stable_window` and ``TimeSAGEResult.stable_window``."""
    mask: NDArray[np.bool_] = (censoring_survival >= float(min_censoring_survival)) & (
        at_risk >= float(min_at_risk)
    )
    if not mask.any():
        warnings.warn(
            "The stable window is empty: no grid time has at-risk fraction >= "
            f"{min_at_risk} and G_hat >= {min_censoring_survival}.",
            UserWarning,
            stacklevel=3,
        )
    return mask


# ---------------------------------------------------------------- curve comparisons


def curve_correlation(
    estimate: ArrayLike, reference: ArrayLike, *, window: ArrayLike | None = None
) -> float:
    """Pearson correlation of two importance curves, optionally on a window.

    Measures whether the estimated curve has the reference's shape, ignoring
    its scale and level. Undefined (NaN, with a warning) if either curve is
    constant on the window (standard deviation <= 1e-12), or the
    window has fewer than two times.

    Parameters
    ----------
    estimate, reference : array_like of shape (K,)
        Importance curves on the same grid, e.g. ``result.phi[j]`` and an
        oracle curve.
    window : array_like of bool, shape (K,), optional
        Restrict to these grid times, e.g. :func:`stable_window`.

    Returns
    -------
    float
    """
    est, ref = _pair(estimate, reference)
    mask = _window(window, len(est))
    if mask is not None:
        est, ref = est[mask], ref[mask]
    return _correlation("pearson", est, ref, "curve_correlation")


def nwiae(
    estimate: ArrayLike,
    reference: ArrayLike,
    times: ArrayLike,
    *,
    weights: ArrayLike | None = None,
    window: ArrayLike | None = None,
    legacy_parity: bool = False,
) -> float:
    r"""Normalised weighted integrated absolute error between two curves.

    .. math::
        \mathrm{NWIAE} = \frac{\sum_k \omega_k |\hat\phi(t_k) - \phi^*(t_k)|}
                              {\sum_k \omega_k |\phi^*(t_k)| + 10^{-12}},
        \qquad \omega_k \propto \Delta t_k\, w(t_k)\, 1\{t_k \in W\},

    with ``dt_k`` the trapezoid weights of the grid, normalised to sum to one.
    0 is a perfect match; 1 is an error as large as the reference itself.

    Parameters
    ----------
    estimate, reference : array_like of shape (K,)
        Importance curves on the same grid.
    times : array_like of shape (K,)
        The grid.
    weights : array_like of shape (K,), optional
        Per-time weight ``w(t_k) >= 0``, multiplied by the trapezoid weights.
        The default, ``w = 1``, is plain trapezoid weighting. The original
        synthetic study used the data-generating
        process's event density ``h0(t) S0(t)``, which only exists when the
        true model is known, e.g. from :mod:`timesage.datasets.synthetic`.
    window : array_like of bool, shape (K,), optional
        Restrict to these grid times, e.g. :func:`stable_window`.
    legacy_parity : bool, default False
        Undefined cases (all weights zero on the window, or a reference with
        zero weighted mass) return NaN with a warning. With ``True`` they
        return the original implementation's values instead: 0 for zero
        weights, and the numerator divided by 1e-12 for a zero reference.

    Returns
    -------
    float
    """
    est, ref = _pair(estimate, reference)
    grid = _validate_grid(times)
    if len(grid) != len(est):
        raise ValueError(f"times has {len(grid)} values; the curves have {len(est)}.")
    w = _trapezoid_weights(grid)
    if weights is not None:
        density = _curve(weights, "weights", len(est))
        if np.any(density < 0):
            raise ValueError("weights must be non-negative.")
        w = w * density
    mask = _window(window, len(est))
    if mask is not None:
        w = w * mask
    # As weighted_point_weights: normalise, or all zeros if the total is ~0.
    total = float(np.sum(w))
    if total <= 1e-12:
        if not legacy_parity:
            _undefined("nwiae", "all weights are zero on the window")
            return float("nan")
        w = np.zeros_like(w)
    else:
        w = w / total
    numerator = float(np.sum(w * np.abs(est - ref)))
    denominator = float(np.sum(w * np.abs(ref)))
    if denominator <= _NWIAE_EPS and not legacy_parity:
        _undefined("nwiae", "the reference curve has no weighted mass")
        return float("nan")
    return numerator / (denominator + _NWIAE_EPS)


def high_importance_iou(
    estimate: ArrayLike,
    reference: ArrayLike,
    *,
    threshold: float = 0.8,
    window: ArrayLike | None = None,
    legacy_parity: bool = False,
) -> float:
    """Overlap of the high-importance regions of two curves (IoU@q80).

    A curve's high-importance region is the set of grid times where it is at
    least ``threshold`` times its maximum. The maximum is taken over the
    **whole** curve, and the region is then intersected with ``window``, as
    defined in Section 7.7.3. For a noisy estimated curve, a spike outside the
    window can therefore raise the cut-off and shrink its region inside it.
    Returns the intersection over union of the two regions.

    Parameters
    ----------
    estimate, reference : array_like of shape (K,)
        Importance curves on the same grid.
    threshold : float, default 0.8
        Fraction of the maximum, in ``(0, 1]``.
    window : array_like of bool, shape (K,), optional
        Restrict to these grid times, e.g. :func:`stable_window`.
    legacy_parity : bool, default False
        If both regions are empty (an empty window, curves with no positive
        values, or both regions outside the window), the IoU is undefined: NaN
        with a warning. The original implementation returned 0. This is not
        rare: in a test fixture of random hump-shaped curves, 3 of 20 pairs
        have both high-importance regions entirely outside the stable window.

    Returns
    -------
    float
        In ``[0, 1]``, or NaN.
    """
    est, ref = _pair(estimate, reference)
    if not 0 < float(threshold) <= 1:
        raise ValueError(f"threshold must lie in (0, 1]; got {threshold}.")
    est_region = est >= float(threshold) * max(float(np.max(est)), _PEAK_FLOOR)
    ref_region = ref >= float(threshold) * max(float(np.max(ref)), _PEAK_FLOOR)
    mask = _window(window, len(est))
    if mask is not None:
        est_region, ref_region = est_region & mask, ref_region & mask
    union = float(np.logical_or(est_region, ref_region).sum())
    if union <= 0.0:
        if legacy_parity:
            return 0.0
        _undefined("high_importance_iou", "both high-importance regions are empty")
        return float("nan")
    return float(np.logical_and(est_region, ref_region).sum()) / union


def peak_time_error(
    estimate: ArrayLike,
    reference: ArrayLike,
    times: ArrayLike,
    *,
    window: ArrayLike | None = None,
) -> float:
    """Absolute difference between the times at which two curves peak.

    The peak is the first grid time of the maximum, within ``window`` if
    given. Undefined (NaN, with a warning) for an empty window.

    Parameters
    ----------
    estimate, reference : array_like of shape (K,)
        Importance curves on the same grid.
    times : array_like of shape (K,)
        The grid.
    window : array_like of bool, shape (K,), optional
        Restrict to these grid times, e.g. :func:`stable_window`.

    Returns
    -------
    float
        In the units of ``times``.
    """
    est, ref = _pair(estimate, reference)
    grid = _validate_grid(times)
    if len(grid) != len(est):
        raise ValueError(f"times has {len(grid)} values; the curves have {len(est)}.")
    mask = _window(window, len(est))
    if mask is not None:
        if not mask.any():
            _undefined("peak_time_error", "the window is empty")
            return float("nan")
        grid, est, ref = grid[mask], est[mask], ref[mask]
    return float(abs(grid[int(np.argmax(est))] - grid[int(np.argmax(ref))]))


# ---------------------------------------------------------------- rankings


def rank_correlation(
    estimate: ArrayLike,
    reference: ArrayLike,
    method: Literal["spearman", "pearson"] = "spearman",
) -> float:
    """Correlation between two sets of per-feature scores, e.g. integrated importances.

    Spearman (default) compares the rankings; Pearson the values. Undefined
    (NaN, with a warning) if either set is constant (standard deviation
    <= 1e-12).

    Parameters
    ----------
    estimate, reference : array_like of shape (p,)
        One score per feature, in the same feature order, e.g.
        ``result.integrated().to_numpy()``.
    method : {"spearman", "pearson"}, default "spearman"

    Returns
    -------
    float
    """
    if method not in ("spearman", "pearson"):
        raise ValueError(f"method must be 'spearman' or 'pearson'; got {method!r}.")
    est, ref = _pair(estimate, reference)
    return _correlation(method, est, ref, "rank_correlation")


def _correlation(method: str, x: NDArray[np.float64], y: NDArray[np.float64], metric: str) -> float:
    """As the original ``_safe_corr``, with a warning instead of a silent NaN."""
    if len(x) < 2:
        _undefined(metric, "fewer than two points")
        return float("nan")
    if np.std(x) <= _CONSTANT_STD or np.std(y) <= _CONSTANT_STD:
        _undefined(metric, "a curve is constant")
        return float("nan")
    result = spearmanr(x, y) if method == "spearman" else pearsonr(x, y)
    return float(result.statistic)
