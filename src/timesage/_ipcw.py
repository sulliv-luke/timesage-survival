"""Censoring survival and IPCW Brier loss (private).

Implements the Kaplan-Meier estimate of the censoring survival function
``G_hat`` with censoring as the event (Eq. 16 of the dissertation) and the
per-sample IPCW Brier loss (Eq. 14-15).

Conventions, the same as in the original research implementation
(``time_sage_survival.py``):

- ``G_hat`` is evaluated right-continuously, so an event's weight is
  ``1 / G_hat(Y_i)``, as Eq. 15 is written, not the left limit
  ``1 / G_hat(Y_i-)`` of Graf et al. (1999). The two differ only when a
  censoring occurs at the same time as an event.
- At a time shared by events and censorings, the event subjects remain in the
  risk set for the censoring step, as in
  ``lifelines.KaplanMeierFitter(T, 1 - event)``.
- ``G_hat`` is clipped to ``[1e-12, 1]`` on evaluation. If the largest observed
  time is censored, ``G_hat`` reaches 0 and the clip yields weights up to 1e12
  without warning; callers must keep evaluation times away from that tail.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = ["CensoringKM", "fit_censoring_km", "ipcw_brier_loss", "ipcw_weights"]

_G_FLOOR = 1e-12


@dataclass(frozen=True)
class CensoringKM:
    """Kaplan-Meier estimate of the censoring survival ``G(t) = P(C > t)`` (Eq. 16).

    A right-continuous step function with a value at every unique observed
    time.

    Attributes
    ----------
    times : ndarray of shape (n_unique,)
        Sorted unique observed times (events and censorings).
    survival : ndarray of shape (n_unique,)
        ``G_hat`` at each of ``times``.
    """

    times: NDArray[np.float64]
    survival: NDArray[np.float64]

    def __call__(self, t: ArrayLike) -> NDArray[np.float64]:
        """Evaluate ``G_hat`` at ``t``, right-continuously, clipped to ``[1e-12, 1]``.

        Parameters
        ----------
        t : array_like
            Evaluation time(s). A scalar returns a 0-d array.

        Returns
        -------
        ndarray
            ``G_hat(t)``, same shape as ``t``. Equal to 1 before the first
            observed time and to the last value after the last observed time.
        """
        t_arr = np.asarray(t, dtype=float)
        idx = np.searchsorted(self.times, t_arr, side="right") - 1
        out = np.ones_like(t_arr, dtype=float)
        valid = idx >= 0
        out[valid] = self.survival[idx[valid]]
        clipped: NDArray[np.float64] = np.clip(out, _G_FLOOR, 1.0)
        return clipped


def _as_survival_data(
    time: ArrayLike, event: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.int_]]:
    """Validate ``(time, event)`` and return them as float and int arrays (copies).

    Raises
    ------
    ValueError
        If the arrays are not 1-D, differ in length, are empty, contain
        non-finite times, or ``event`` has values other than 0 and 1.
    """
    T = np.array(time, dtype=float)
    raw_event = np.asarray(event)
    if T.ndim != 1 or raw_event.ndim != 1:
        raise ValueError(f"time and event must be 1-D; got shapes {T.shape} and {raw_event.shape}.")
    if len(T) != len(raw_event):
        raise ValueError(
            f"time and event must have the same length; got {len(T)} and {len(raw_event)}."
        )
    if len(T) == 0:
        raise ValueError("time and event must contain at least one observation.")
    if not np.all(np.isfinite(T)):
        raise ValueError("time must contain only finite values.")
    if not np.all((raw_event == 0) | (raw_event == 1)):
        bad = np.unique(raw_event[(raw_event != 0) & (raw_event != 1)])[:5]
        raise ValueError(
            "event must take values in {0, 1} (1 = event observed, 0 = censored); "
            f"found {bad.tolist()}."
        )
    return T, raw_event.astype(int)


def fit_censoring_km(time: ArrayLike, event: ArrayLike) -> CensoringKM:
    """Fit the Kaplan-Meier censoring survival function (Eq. 16).

    Implements ``G_hat(t) = prod_{u <= t} (1 - d_u^C / n_u)``, with
    ``d_u^C`` the number of censorings at ``u`` and ``n_u`` the number of
    subjects with observed time ``>= u``. Events tied with censorings at ``u``
    are therefore in the risk set for the censoring step.

    Parameters
    ----------
    time : array_like of shape (n,)
        Observed times ``Y_i`` (event or censoring).
    event : array_like of shape (n,)
        Event indicator ``delta_i``: 1 if the event was observed, 0 if censored.

    Returns
    -------
    CensoringKM
        The fitted step function, with a value at every unique observed time.

    Raises
    ------
    ValueError
        On invalid ``time`` or ``event`` (see ``_as_survival_data``).

    References
    ----------
    Kaplan, E. L. & Meier, P. (1958). JASA, 53(282), 457-481.
    Graf, E. et al. (1999). Statistics in Medicine, 18(17-18), 2529-2545.
    """
    T, delta = _as_survival_data(time, event)
    uniq_times, inverse, counts = np.unique(T, return_inverse=True, return_counts=True)
    n_censored = np.bincount(inverse, weights=(delta == 0)).astype(float)
    # Number at risk just before each unique time.
    n_at_risk = len(T) - np.concatenate(([0], np.cumsum(counts)[:-1]))
    survival = np.cumprod(1.0 - n_censored / n_at_risk)
    return CensoringKM(uniq_times, survival)


def ipcw_weights(
    time: ArrayLike,
    event: ArrayLike,
    times: ArrayLike,
    censoring: CensoringKM,
    censoring_at_times: ArrayLike | None = None,
) -> NDArray[np.float64]:
    """IPCW weights ``W_i(t_k)`` (Eq. 15).

    ``W_i(t) = 1{Y_i <= t, delta_i = 1} / G_hat(Y_i) + 1{Y_i > t} / G_hat(t)``.
    Subjects censored at or before ``t`` get weight 0. ``G_hat`` is evaluated
    right-continuously, so an event at ``Y_i`` uses ``G_hat(Y_i)`` (see the
    module notes on ties).

    Parameters
    ----------
    time : array_like of shape (n,)
        Observed times ``Y_i``.
    event : array_like of shape (n,)
        Event indicator (1 = event, 0 = censored).
    times : array_like of shape (K,)
        Evaluation horizons ``t_k``.
    censoring : CensoringKM
        Fitted censoring survival ``G_hat``.
    censoring_at_times : array_like of shape (K,), optional
        Precomputed ``censoring(times)``; computed if omitted.

    Returns
    -------
    ndarray of shape (n, K)
        ``W_i(t_k)``.
    """
    T, delta = _as_survival_data(time, event)
    t_grid = np.asarray(times, dtype=float)
    if t_grid.ndim != 1:
        raise ValueError(f"times must be 1-D; got shape {t_grid.shape}.")
    if censoring_at_times is None:
        G_at_times = censoring(t_grid)
    else:
        G_at_times = np.asarray(censoring_at_times, dtype=float)
        if G_at_times.shape != t_grid.shape:
            raise ValueError(
                f"censoring_at_times must have shape {t_grid.shape}; got {G_at_times.shape}."
            )

    inv_G_T = 1.0 / censoring(T)
    inv_G_times = 1.0 / G_at_times

    # The two masks are mutually exclusive.
    mask_event = (T[:, None] <= t_grid[None, :]) & (delta[:, None] == 1)  # w = 1/G(Y_i)
    mask_risk = T[:, None] > t_grid[None, :]  # w = 1/G(t_k)

    w = np.where(mask_event, inv_G_T[:, None], 0.0)
    weights: NDArray[np.float64] = np.where(mask_risk, inv_G_times[None, :], w)
    return weights


def ipcw_brier_loss(
    pred_surv: ArrayLike,
    time: ArrayLike,
    event: ArrayLike,
    times: ArrayLike,
    censoring: CensoringKM,
    censoring_at_times: ArrayLike | None = None,
) -> NDArray[np.float64]:
    """Per-sample IPCW Brier loss at K evaluation times (Eq. 14-15).

    ``loss[i, k] = W_i(t_k) * (1{Y_i > t_k} - S_hat(t_k | x_i))**2``, with
    ``W_i`` from :func:`ipcw_weights`. Averaging over ``i`` gives
    ``BS_IPCW(t_k)`` (Eq. 14). With no censoring all weights are 1 and this is
    the plain Brier score.

    Parameters
    ----------
    pred_surv : array_like of shape (n, K)
        Predicted survival probabilities ``S_hat(t_k | x_i)``.
    time : array_like of shape (n,)
        Observed times ``Y_i``.
    event : array_like of shape (n,)
        Event indicator (1 = event, 0 = censored).
    times : array_like of shape (K,)
        Evaluation horizons ``t_k``.
    censoring : CensoringKM
        Fitted censoring survival ``G_hat``.
    censoring_at_times : array_like of shape (K,), optional
        Precomputed ``censoring(times)``; computed if omitted.

    Returns
    -------
    ndarray of shape (n, K)
        Per-sample, per-time IPCW Brier loss.

    References
    ----------
    Graf, E. et al. (1999); Gerds, T. A. & Schumacher, M. (2006).
    """
    T = np.asarray(time, dtype=float)
    t_grid = np.asarray(times, dtype=float)
    S = np.asarray(pred_surv, dtype=float)
    w = ipcw_weights(T, event, t_grid, censoring, censoring_at_times)
    if S.shape != w.shape:
        raise ValueError(f"pred_surv must have shape {w.shape} (n, K); got {S.shape}.")
    y = (T[:, None] > t_grid[None, :]).astype(float)  # 1{Y_i > t_k}
    loss: NDArray[np.float64] = w * (y - S) ** 2
    return loss
