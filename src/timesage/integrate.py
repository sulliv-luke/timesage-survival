r"""Integrated importance scalars (public).

Implements Section 5.8 of the dissertation (Eq. 20-24): each feature's
importance curve is collapsed to one number,

    phi_j^int[w] = sum_k w(t_k) phi_j(t_k) dt_k,

where ``dt_k`` are the trapezoid quadrature weights of the grid, so
``w = 1`` is the trapezoidal area under the curve. Three weights are
available:

=============  ===============================  =====================================
``method``     ``w(t_k)``                       Notes
=============  ===============================  =====================================
``trapezoid``  1                                Area under the importance curve
``at_risk``    ``n_hat(t_k)``, the at-risk      Down-weights late horizons where the
               fraction of the explained rows   risk set is small
``censoring``  ``G_hat(t_k)``                   **Default.** Counteracts the ``1/G_hat``
                                                inflation of the IPCW weights
=============  ===============================  =====================================

As in the original research implementation
(``TimeSAGE.integrated_importance`` in ``time_sage_survival.py``), the
weighted variants multiply the trapezoid weights pointwise rather than
integrating ``w * phi`` trapezoidally, and the integral runs from the first
grid time ``t_1``, not from 0.

``legacy_parity=True`` reproduces the original research implementation's
arithmetic: it changes the ``at_risk`` fraction from the textbook
``(1/n) sum_i 1{Y_i >= t}`` to that implementation's ``1{Y_i > t}``. The two
differ only at grid times equal to an
observed time. The flag has no effect on the other variants.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

if TYPE_CHECKING:
    from timesage.result import TimeSAGEResult

__all__ = ["IntegrationMethod", "integrate"]

IntegrationMethod = Literal["trapezoid", "at_risk", "censoring"]
_METHODS: tuple[str, ...] = ("trapezoid", "at_risk", "censoring")


def _trapezoid_weights(t: NDArray[np.float64]) -> NDArray[np.float64]:
    """Point weights ``w`` with ``w @ f == np.trapezoid(f, x=t)``.

    Same arithmetic as the original ``_trapezoid_weights``. Needs at least two times.
    """
    w = np.empty_like(t, dtype=np.float64)
    w[0] = 0.5 * (t[1] - t[0])
    w[-1] = 0.5 * (t[-1] - t[-2])
    if len(t) > 2:
        w[1:-1] = 0.5 * (t[2:] - t[:-2])
    return w


def integrate(
    result: TimeSAGEResult,
    method: IntegrationMethod = "censoring",
    *,
    window: ArrayLike | None = None,
    legacy_parity: bool = False,
) -> pd.Series:
    """Collapse each feature's importance curve to a scalar (Eq. 20-24).

    Parameters
    ----------
    result : TimeSAGEResult
        Output of :meth:`timesage.TimeSAGE.explain`, on a grid of at least two
        times.
    method : {"censoring", "at_risk", "trapezoid"}, default "censoring"
        Weight ``w(t)`` applied to the trapezoid weights; see the module
        docstring. On synthetic data with a known oracle, ``censoring`` and
        ``at_risk`` recovered the oracle ranking slightly better than
        ``trapezoid``, more so under heavy censoring, and were about equal to
        each other. This has not been checked on real data or in high
        dimensions.
    window : array_like of bool, shape (K,), optional
        Integrate only over these grid times, e.g. ``result.stable_window()``.
        As in the original implementation (``truth.integrate_curve``), the
        trapezoid weights are
        computed on the **full** grid and set to zero outside the window. They
        are not recomputed on the windowed sub-grid, so a window's first and
        last points keep half-intervals that reach outside it, and the edge
        points are slightly overweighted. For example, on the grid
        ``1, 2, 4, 7, 11`` a window selecting ``2, 4, 7`` gives a constant
        curve of 1 an area of 7.5, not the sub-grid's 5. With the
        synthetic study's stable window and ``method="trapezoid"``, this is
        the integration the original synthetic study used for its rank
        correlation. ``None`` (default)
        integrates over the whole grid.
    legacy_parity : bool, default False
        Use the original implementation's at-risk fraction ``1{Y_i > t}``
        instead of ``1{Y_i >= t}``. Only affects ``method="at_risk"``.

    Returns
    -------
    pandas.Series
        One value per feature, indexed by feature name, named
        ``"integrated_<method>"``. Units are those of ``phi`` times time.

    Raises
    ------
    ValueError
        If ``method`` is unknown, the grid has fewer than two times, the
        window does not match the grid or selects fewer than two times, or
        ``legacy_parity=True`` is requested for a result that does not hold
        the at-risk fraction ``legacy_parity`` needs (one not created by
        ``explain``).

    Examples
    --------
    >>> result.integrated()                     # censoring-weighted
    >>> integrate(result, method="trapezoid")   # area under each curve
    >>> result.integrated(window=result.stable_window())
    """
    if method not in _METHODS:
        raise ValueError(f"Unknown method {method!r}; choose one of {', '.join(_METHODS)}.")
    times = result.times
    if len(times) < 2:
        raise ValueError("Integration needs a grid of at least two times.")
    dt = _trapezoid_weights(times)
    if window is not None:
        mask = np.asarray(window)
        if mask.dtype != np.bool_:
            raise TypeError(f"window must be a boolean array; got dtype {mask.dtype}.")
        if mask.shape != times.shape:
            raise ValueError(f"window must have shape {times.shape}; got {mask.shape}.")
        if int(mask.sum()) < 2:
            raise ValueError(
                f"window must select at least two times; it selects {int(mask.sum())}."
            )
        dt = dt * mask  # as integrate_curve: full-grid weights, zeroed outside

    if method == "trapezoid":
        weights = dt
    elif method == "at_risk":
        if legacy_parity:
            fraction = result._at_risk_fraction_legacy
            if fraction is None:
                raise ValueError(
                    "This result does not hold the legacy at-risk fraction; "
                    "legacy_parity=True needs a result created by TimeSAGE.explain."
                )
        else:
            fraction = result.at_risk_fraction
        weights = fraction * dt
    else:
        weights = result.censoring_survival * dt

    # Same expression as the original implementation, so the summation order is unchanged.
    values = np.sum(result.phi * weights[None, :], axis=1)
    out: pd.Series = pd.Series(
        values, index=pd.Index(result.feature_names, name="feature"), name=f"integrated_{method}"
    )
    return out
