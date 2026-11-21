"""Optional plotting helpers (public).

Two figures for a :class:`~timesage.TimeSAGEResult`:

- :func:`plot_importance_curves`: the time-indexed importance curves
  ``phi_j(t)`` (Eq. 19) with Monte Carlo bands, the stable support window
  (Eq. 31) shaded;
- :func:`plot_integrated`: a bar chart of integrated importance (Eq. 20-24).

Matplotlib is imported only when a function is called, so it remains an
optional dependency: ``pip install "timesage-survival[plot]"``. This module is
not imported by ``import timesage``.

Colours follow a fixed eight-colour categorical order, checked for
colour-vision-deficiency separation between neighbouring series. Colour
identifies a series *within one figure*: features are coloured in the order
they are plotted, not by a global feature index. At most eight curves can be
drawn in one plot.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from timesage.integrate import IntegrationMethod, integrate

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from timesage.result import TimeSAGEResult

__all__ = ["plot_importance_curves", "plot_integrated"]

# Categorical order: blue, orange, aqua, yellow, magenta, green, violet, red.
_SERIES_COLOURS: tuple[str, ...] = (
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#6250d6",
    "#e34948",
)
_MAX_SERIES = len(_SERIES_COLOURS)
_WINDOW_COLOUR = "#8a8a85"
_REFERENCE_COLOUR = "#52514e"
_BAND_Z = 1.96


def _pyplot() -> Any:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - exercised by monkeypatching
        raise ImportError(
            "timesage.plotting needs matplotlib. Install it with "
            'pip install "timesage-survival[plot]".'
        ) from exc
    return plt


def _axes(ax: Axes | None) -> Axes:
    if ax is not None:
        return ax
    new_ax: Axes = _pyplot().subplots()[1]
    return new_ax


def _check_top(top: int | None) -> None:
    if top is not None and (isinstance(top, bool) or not isinstance(top, int) or top < 1):
        raise ValueError(f"top must be a positive integer or None; got {top!r}.")


def _select_features(
    result: TimeSAGEResult, features: Sequence[str] | None, top: int | None
) -> list[int]:
    names = list(result.feature_names)
    if features is not None:
        if isinstance(features, str):
            features = [features]
        chosen = list(features)
        unknown = [f for f in chosen if f not in names]
        if unknown:
            raise ValueError(f"Unknown feature names: {unknown}.")
        if len(set(chosen)) != len(chosen):
            raise ValueError("features contains duplicates.")
        if not chosen:
            raise ValueError("features is empty.")
        return [names.index(f) for f in chosen]
    _check_top(top)
    order = np.argsort(-result.integrated().to_numpy(), kind="stable")
    return [int(j) for j in (order if top is None else order[:top])]


def _window_mask(
    result: TimeSAGEResult, window: ArrayLike | Literal["stable"] | None
) -> NDArray[np.bool_] | None:
    if window is None:
        return None
    if isinstance(window, str):
        if window != "stable":
            raise ValueError(f'window must be "stable", a boolean mask or None; got {window!r}.')
        return result.stable_window()
    mask = np.asarray(window)
    if mask.dtype != np.bool_:
        raise TypeError(f"window must be a boolean mask; got dtype {mask.dtype}.")
    if mask.shape != result.times.shape:
        raise ValueError(
            f"window must have shape {result.times.shape} (one entry per time); got {mask.shape}."
        )
    return mask


def _runs(mask: NDArray[np.bool_]) -> list[tuple[int, int]]:
    """Inclusive index ranges ``(a, b)`` of the contiguous ``True`` runs of ``mask``."""
    padded = np.concatenate(([False], mask, [False])).astype(np.int8)
    edges = np.flatnonzero(np.diff(padded))
    return [(int(a), int(b) - 1) for a, b in zip(edges[::2], edges[1::2], strict=True)]


def plot_importance_curves(
    result: TimeSAGEResult,
    *,
    features: Sequence[str] | None = None,
    top: int | None = 8,
    window: ArrayLike | Literal["stable"] | None = "stable",
    band: bool = True,
    ax: Axes | None = None,
) -> Axes:
    r"""Plot time-indexed importance curves ``phi_j(t)`` (Eq. 19).

    Positive values mean the feature reduces the IPCW Brier loss at that
    horizon; negative values are legitimate and are drawn as they are.

    Parameters
    ----------
    result : TimeSAGEResult
    features : sequence of str, optional
        Features to plot, in this order (at most eight). Default: the ``top``
        features by integrated importance (the default ``censoring``
        variant, :meth:`TimeSAGEResult.integrated`).
    top : int or None, default 8
        Used only when ``features`` is not given; ``None`` means all
        features, which must then number at most eight.
    window : "stable", boolean array of shape (K,) or None, default "stable"
        Times to shade. ``"stable"`` shades the stable support window
        :meth:`TimeSAGEResult.stable_window` (Eq. 31), where the at-risk set
        and ``G_hat`` are large enough for the curves to be reliable; late
        horizons outside it are noisy. ``None`` shades nothing. Each
        contiguous run of the window is shaded from midway to the previous
        grid time to midway to the next one.
    band : bool, default True
        Shade ``phi +/- 1.96 * phi_se`` around each curve: pointwise Monte
        Carlo error of the permutation estimator only. It does not reflect
        sampling variation of the data or the model.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; a new figure is created if omitted.

    Returns
    -------
    matplotlib.axes.Axes
    """
    idx = _select_features(result, features, top)
    if len(idx) > _MAX_SERIES:
        raise ValueError(
            f"At most {_MAX_SERIES} curves can be told apart in one plot; got {len(idx)}. "
            "Pass features=... or top <= 8, or draw several plots."
        )
    mask = _window_mask(result, window)
    ax = _axes(ax)
    t = result.times

    if mask is not None and mask.any():
        mid = (t[1:] + t[:-1]) / 2 if len(t) > 1 else t[:0]
        for n, (a, b) in enumerate(_runs(mask)):
            left = mid[a - 1] if a > 0 else t[a]
            right = mid[b] if b < len(t) - 1 else t[b]
            ax.axvspan(
                left,
                right,
                color=_WINDOW_COLOUR,
                alpha=0.12,
                linewidth=0,
                label="stable window" if n == 0 else None,
                gid="timesage-window",
                zorder=0,
            )
    ax.axhline(0.0, color=_REFERENCE_COLOUR, linewidth=0.8, zorder=1)

    for colour, j in zip(_SERIES_COLOURS, idx, strict=False):
        if band:
            half = _BAND_Z * result.phi_se[j]
            ax.fill_between(
                t,
                result.phi[j] - half,
                result.phi[j] + half,
                color=colour,
                alpha=0.15,
                linewidth=0,
                zorder=2,
            )
        ax.plot(
            t, result.phi[j], color=colour, linewidth=2.0, label=result.feature_names[j], zorder=3
        )

    ax.set_xlabel("Time")
    ax.set_ylabel(r"Importance $\phi_j(t)$ (reduction in IPCW Brier score)")
    ax.grid(True, axis="y", linewidth=0.5, alpha=0.4)
    ax.legend(frameon=False)
    return ax


def plot_integrated(
    result: TimeSAGEResult,
    method: IntegrationMethod = "censoring",
    *,
    window: ArrayLike | None = None,
    top: int | None = 20,
    legacy_parity: bool = False,
    ax: Axes | None = None,
) -> Axes:
    """Bar chart of integrated importance (Eq. 20-24), largest at the top.

    Bars come from :func:`timesage.integrate` and are drawn as they are:
    negative importances point left of the zero line.

    No error bars are drawn: ``phi_se`` is a pointwise standard error and the
    errors are correlated across times, so it does not give a valid standard
    error for the integral. To judge stability, compare results from several
    ``random_state`` values.

    Parameters
    ----------
    result : TimeSAGEResult
    method : {"censoring", "at_risk", "trapezoid"}, default "censoring"
        Integration weight; see :func:`timesage.integrate`.
    window : boolean array of shape (K,), optional
        Integrate only over these times, e.g. ``result.stable_window()``.
    top : int or None, default 20
        Show the ``top`` features with the largest integrated importance;
        ``None`` shows all.
    legacy_parity : bool, default False
        Passed to :func:`timesage.integrate`.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; a new figure is created if omitted.

    Returns
    -------
    matplotlib.axes.Axes
    """
    _check_top(top)
    values = integrate(result, method, window=window, legacy_parity=legacy_parity)
    values = values.sort_values(ascending=False, kind="stable")
    if top is not None:
        values = values.iloc[:top]
    ax = _axes(ax)
    pos = np.arange(len(values))
    ax.barh(pos, values.to_numpy(), height=0.7, color=_SERIES_COLOURS[0], zorder=2)
    ax.set_yticks(pos, labels=[str(n) for n in values.index])
    ax.invert_yaxis()
    ax.axvline(0.0, color=_REFERENCE_COLOUR, linewidth=0.8, zorder=3)
    label = f"Integrated importance ({method}"
    ax.set_xlabel(label + (", windowed)" if window is not None else ")"))
    ax.grid(True, axis="x", linewidth=0.5, alpha=0.4)
    return ax
