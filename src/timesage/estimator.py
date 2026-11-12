r"""The ``TimeSAGE`` estimator (public).

Implements the joint-times permutation Monte Carlo estimator of Eq. 18-19:
one permutation's coalitions are shared across the whole time grid, and each
permutation is paired with one evaluation row sampled with replacement, as in
``PermutationEstimatorMultiOutput`` of the original research implementation
(``time_sage_survival.py``).

Randomness follows the original implementation's ``SeedSequence`` tree, so a
seeded run reproduces one of its runs with ``n_jobs=1`` exactly::

    root = SeedSequence(seed)
    batch b:  child_b = root.spawn(1)[0];  idx_seq, perm_seq = child_b.spawn(2)
              rows = default_rng(idx_seq).choice(n, batch_size)    # with replacement
              rng = default_rng(perm_seq); rng.shuffle(row) for each permutation row

``SeedSequence.spawn`` numbers children sequentially, so batch ``b`` gets the
same child whatever ``n_jobs`` the original implementation used (checked
with ``n_jobs=2``). Differences from it: exactly ``n_permutations`` are used
(it rounded up to a multiple of ``batch_size * n_jobs``; the last batch here
may be smaller), and early stopping is checked after every batch, as it was
with ``n_jobs=1``.
"""

from __future__ import annotations

import logging
import numbers
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from timesage._ipcw import CensoringKM, _as_survival_data, fit_censoring_km
from timesage._value import CoalitionValue, PredictSurvivalFn
from timesage._vendor.sage import ImportanceTracker
from timesage.protocols import SurvivalModel
from timesage.result import TimeSAGEResult

__all__ = ["TimeSAGE"]

logger = logging.getLogger("timesage")

ProgressFn = Callable[[int, int], None]
"""``progress(done, total)``: permutations done so far and the cap ``n_permutations``."""

_RANGE_FLOOR = 1e-12
_STABLE_G_MIN = 0.05  # G_hat(t) >= 0.05: the censoring half of the stable window (Eq. 31)


@dataclass(frozen=True)
class _SamplingOutcome:
    """Copies of the tracker's final state and how the run ended."""

    values: NDArray[np.float64]
    std: NDArray[np.float64]
    n_permutations: int
    converged: bool | None
    convergence_ratio: float


def _convergence_ratio(tracker: ImportanceTracker) -> float:
    """Return the stopping measure: the worst over times of ``max_j SE / range_j phi``.

    For each time ``t_k``, the largest standard error over features divided by
    the spread of the current estimates (floored at 1e-12); the maximum over
    ``k`` is returned. Same arithmetic as the original implementation's
    ``PermutationEstimatorMultiOutput._convergence_ratio``.
    """
    vals = tracker.values
    if not isinstance(vals, np.ndarray) or vals.ndim < 2:
        return 1.0
    max_std_k = np.max(tracker.std, axis=0)
    range_k = np.max(vals, axis=0) - np.min(vals, axis=0)
    return float(np.max(max_std_k / np.maximum(range_k, _RANGE_FLOOR)))


def _batch_scores(
    value: CoalitionValue,
    x: NDArray[np.floating],
    time: NDArray[np.float64],
    event: NDArray[np.int_],
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Marginal contributions for one batch, one permutation per row (Eq. 18).

    Returns ``scores`` of shape ``(batch, p, K)`` with
    ``scores[i, j] = loss_i(S_pi(j)) - loss_i(S_pi(j) + {j})`` for the
    permutation ``pi`` drawn for row ``i``. Same operations, in the same order,
    as the original implementation's
    ``PermutationEstimatorMultiOutput._process_sample`` with the full
    coalition range.
    """
    batch, p = x.shape
    arange = np.arange(batch)
    scores = np.zeros((batch, p, len(value.times)))
    mask = np.zeros((batch, p), dtype=bool)
    permutations = np.tile(np.arange(p), (batch, 1))
    for i in range(batch):
        rng.shuffle(permutations[i])

    prev_loss = value.loss(x, mask, time, event)
    for i in range(p):
        inds = permutations[:, i]
        mask[arange, inds] = True
        loss = value.loss(x, mask, time, event)
        scores[arange, inds, :] = prev_loss - loss
        prev_loss = loss
    return scores


def _sample_permutations(
    value: CoalitionValue,
    x: NDArray[np.floating],
    time: NDArray[np.float64],
    event: NDArray[np.int_],
    *,
    n_permutations: int,
    batch_size: int,
    convergence_threshold: float | None,
    seed: int,
    progress: ProgressFn | None = None,
) -> _SamplingOutcome:
    """Run the permutation estimator (Eq. 19) and return ``phi`` and its standard errors.

    Parameters
    ----------
    value : CoalitionValue
        Coalition value function on the evaluation grid.
    x : ndarray of shape (n, p)
        Evaluation rows, already cast as the model should see them.
    time, event : ndarray of shape (n,)
        Observed times and event indicators of the rows.
    n_permutations : int
        Number of permutations; a cap when early stopping is on.
    batch_size : int
        Permutations per batch. The last batch may be smaller.
    convergence_threshold : float or None
        Stop once :func:`_convergence_ratio` falls below this value, checked
        after each batch. ``None`` turns early stopping off.
    seed : int
        Entropy of the root ``SeedSequence``.
    progress : callable, optional
        Called as ``progress(done, n_permutations)`` after each batch.

    Returns
    -------
    _SamplingOutcome
        Copies of the tracker's mean and standard error, both ``(p, K)``,
        which never share memory with the tracker.
    """
    n = len(x)
    root = np.random.SeedSequence(seed)
    tracker = ImportanceTracker()
    done = 0
    ratio = 1.0
    converged: bool | None = None if convergence_threshold is None else False

    while done < n_permutations:
        size = min(batch_size, n_permutations - done)
        (child,) = root.spawn(1)
        idx_seq, perm_seq = child.spawn(2)
        idx = np.random.default_rng(idx_seq).choice(n, size)
        scores = _batch_scores(
            value, x[idx], time[idx], event[idx], np.random.default_rng(perm_seq)
        )
        tracker.update(scores)
        done += size

        ratio = _convergence_ratio(tracker)
        logger.debug("permutations %d/%d, convergence ratio %.4f", done, n_permutations, ratio)
        if progress is not None:
            progress(done, n_permutations)
        if convergence_threshold is not None and ratio < convergence_threshold:
            converged = True
            logger.debug("converged after %d permutations", done)
            break

    # ImportanceTracker.update changes `mean` in place: store copies, never views.
    values: NDArray[np.float64] = np.array(tracker.values, dtype=np.float64, copy=True)
    std: NDArray[np.float64] = np.array(tracker.std, dtype=np.float64, copy=True)
    return _SamplingOutcome(
        values=values,
        std=std,
        n_permutations=done,
        converged=converged,
        convergence_ratio=ratio,
    )


# ---------------------------------------------------------------- public estimator


def _resolve_predict(model: SurvivalModel | PredictSurvivalFn) -> PredictSurvivalFn:
    method = getattr(model, "predict_survival", None)
    if callable(method):
        return cast(PredictSurvivalFn, method)
    if callable(model):
        return model
    raise TypeError(
        "model must have a predict_survival(X, times) method or be a callable "
        f"(X, times) -> survival probabilities; got {type(model).__name__}."
    )


def _check_int(name: str, value: object, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise TypeError(f"{name} must be an integer; got {type(value).__name__}.")
    if int(value) < minimum:
        raise ValueError(f"{name} must be >= {minimum}; got {value}.")
    return int(value)


def _as_float32_matrix(data: object, name: str) -> NDArray[np.float32]:
    """Cast values to float64, then to float32 (as the original ``TimeSAGE.fit`` did)."""
    try:
        values = np.asarray(np.asarray(data, dtype=float), dtype=np.float32)
    except (TypeError, ValueError) as err:
        raise TypeError(f"{name} must contain only numeric values.") from err
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty 2-D array; got shape {values.shape}.")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain only finite values.")
    return values


def _column_names(data: object) -> tuple[str, ...] | None:
    if isinstance(data, pd.DataFrame):
        return tuple(str(c) for c in data.columns)
    return None


def _validate_grid(times: ArrayLike) -> NDArray[np.float64]:
    grid = np.array(times, dtype=float)
    if grid.ndim != 1 or len(grid) == 0:
        raise ValueError(f"times must be a non-empty 1-D array; got shape {grid.shape}.")
    if not np.all(np.isfinite(grid)) or np.any(grid < 0):
        raise ValueError("times must contain only finite, non-negative values.")
    if np.any(np.diff(grid) <= 0):
        raise ValueError("times must be strictly increasing.")
    return grid


def _check_follow_up(grid: NDArray[np.float64], observed: NDArray[np.float64], what: str) -> None:
    largest = float(np.max(observed))
    if grid[-1] > largest:
        raise ValueError(
            f"times extend beyond the observed follow-up of {what}: the last grid time is "
            f"{grid[-1]:g} but the largest observed time is {largest:g}."
        )


def _check_censoring_positive(censoring: CensoringKM, grid: NDArray[np.float64]) -> None:
    """Refuse grids where ``G_hat(t_k) = 0``, since the IPCW weights are undefined there.

    ``G_hat`` is non-increasing and events only use ``G_hat(Y_i)`` with
    ``Y_i <= t_k``, so checking the grid covers every weight in use.
    """
    idx = np.searchsorted(censoring.times, grid, side="right") - 1
    raw = np.where(idx >= 0, censoring.survival[np.maximum(idx, 0)], 1.0)
    if np.any(raw <= 0):
        first = grid[np.argmax(raw <= 0)]
        raise ValueError(
            "The censoring survival G_hat(t) is 0 at grid times from "
            f"{first:g} on (the largest observed time is censored), so the IPCW "
            "weights 1/G_hat(t) are undefined there. End the grid before that time."
        )


def _warn_low_censoring_survival(g_grid: NDArray[np.float64], grid: NDArray[np.float64]) -> None:
    """Warn once if ``G_hat(t_k) < 0.05`` anywhere: outside the stable window (Eq. 31)."""
    low = grid[g_grid < _STABLE_G_MIN]
    if len(low):
        warnings.warn(
            f"The censoring survival G_hat(t) is below {_STABLE_G_MIN} at grid times "
            f"{', '.join(f'{t:g}' for t in low)}. IPCW weights there exceed "
            f"{1 / _STABLE_G_MIN:g}, so importance estimates at these times are "
            "unreliable (outside the stable support window, Eq. 31).",
            UserWarning,
            stacklevel=3,
        )


def _check_trial_predictions(pred: NDArray[Any], n: int, K: int) -> None:
    """Shape, finiteness and range checks on one model call; warns if curves rise."""
    if pred.shape != (n, K):
        raise ValueError(
            f"predict_survival returned shape {pred.shape}; expected {(n, K)} (n_samples, n_times)."
        )
    dtype = pred.dtype if np.issubdtype(pred.dtype, np.floating) else np.dtype(np.float64)
    values = pred.astype(dtype, copy=False)
    if not np.all(np.isfinite(values)):
        raise ValueError("predict_survival returned non-finite values.")
    # Rounding noise scales with the output precision (tol is about 1.2e-6 for float32).
    tol = 10 * float(np.finfo(dtype).eps)
    if values.min() < -tol or values.max() > 1 + tol:
        raise ValueError(
            "predict_survival must return probabilities in [0, 1]; got values in "
            f"[{values.min():g}, {values.max():g}]."
        )
    if K > 1:
        rise = float(np.max(np.diff(values, axis=1)))
        if rise > tol:
            warnings.warn(
                "predict_survival returned survival curves that increase over time "
                f"(largest rise {rise:.3g}); survival probabilities should be "
                "non-increasing in t.",
                UserWarning,
                stacklevel=3,
            )


class TimeSAGE:
    r"""Time-indexed SAGE importances for a survival model (Eq. 11-19).

    For each horizon ``t_k`` on a grid, TimeSAGE treats the features as players
    in the game ``v_t(S) = -BS_IPCW(t; S)`` and estimates each feature's
    Shapley value by permutation sampling. A positive ``phi_j(t)`` means that
    feature ``j`` reduces the model's IPCW Brier loss at ``t``. It explains
    predictive performance, not individual predictions.

    Features outside a coalition are replaced by background rows (marginal
    imputation, Eq. 13), independently of the retained features. A feature
    the model ignores therefore gets exactly zero importance, but the model is
    evaluated off the data manifold, and correlated features need not share
    credit equally.

    Parameters
    ----------
    model : SurvivalModel or callable
        An object with ``predict_survival(X, times)``, or a callable with that
        signature, returning ``(n_samples, n_times)`` survival probabilities.
        It receives float32 numpy arrays.
    background : array_like or pandas.DataFrame of shape (m, p)
        Rows used to impute removed features, typically training data or a
        summary of it (e.g. k-means centroids). Used as given unless
        ``n_background`` is set. Cast to float32.
    n_background : int, optional
        If given, use ``min(n_background, m)`` background rows drawn without
        replacement with ``np.random.default_rng(random_state)``. ``None``
        (default) uses the whole background.
    n_permutations : int, default 768
        Number of permutations ``B``; the cap when early stopping is on.
        Rankings at low budgets are noisy and can change between seeds; use
        larger budgets when rankings matter (the original study used 2,048
        for METABRIC).
    batch_size : int, default 64
        Permutations evaluated together; each batch calls the model ``p + 1``
        times with ``batch_size * m`` rows. With a fixed seed, results depend
        on ``batch_size``.
    convergence_threshold : float, optional
        Turn on early stopping: stop once, at every horizon, the largest
        standard error across features is below this fraction of the spread
        of the estimates across features. Checked after every batch. Must lie
        in ``(0, 1)``; the original synthetic study used 0.025. ``None``
        (default) always uses ``n_permutations``.
    random_state : int, optional
        Seed for all randomness: background subsampling and the permutation
        sampler (a ``numpy.random.SeedSequence`` tree matching the original
        research implementation). ``None`` draws a seed once, at construction; it is recorded in
        the result, and repeated calls of :meth:`explain` reuse it.
    progress : callable, optional
        ``progress(done, total)`` is called after each batch with the number of
        permutations done so far and ``n_permutations``.

    Notes
    -----
    Inputs are cast to float32 before they reach the model, as in the
    original research implementation, so the model sees float32-rounded
    features.

    The background should come from the training data, not from the rows
    being explained. The IPCW weights use a Kaplan-Meier estimate of the
    censoring distribution, which assumes censoring is independent of the
    event time and of the features; where censoring is plausibly informative,
    the loss, and so the importances, can be biased.

    Examples
    --------
    >>> explainer = TimeSAGE(model, background=X_train, n_background=96, random_state=0)
    >>> result = explainer.explain(X_test, time=y_time, event=y_event, times=grid)
    >>> result.phi.shape  # (p, K)
    """

    def __init__(
        self,
        model: SurvivalModel | PredictSurvivalFn,
        *,
        background: ArrayLike | pd.DataFrame,
        n_background: int | None = None,
        n_permutations: int = 768,
        batch_size: int = 64,
        convergence_threshold: float | None = None,
        random_state: int | None = None,
        progress: ProgressFn | None = None,
    ) -> None:
        self._predict = _resolve_predict(model)
        self.n_permutations = _check_int("n_permutations", n_permutations, 1)
        self.batch_size = _check_int("batch_size", batch_size, 1)
        if convergence_threshold is not None:
            convergence_threshold = float(convergence_threshold)
            if not 0 < convergence_threshold < 1:
                raise ValueError(
                    f"convergence_threshold must lie in (0, 1); got {convergence_threshold}."
                )
        self.convergence_threshold = convergence_threshold
        if progress is not None and not callable(progress):
            raise TypeError("progress must be callable as progress(done, total).")
        self.progress = progress

        if random_state is None:
            self.random_state = int(cast(int, np.random.SeedSequence().entropy))
        else:
            self.random_state = _check_int("random_state", random_state, 0)

        self._background_columns = _column_names(background)
        bg = _as_float32_matrix(background, "background")
        if n_background is not None:
            size = min(_check_int("n_background", n_background, 1), len(bg))
            # Same call as the original synthetic study's runner.
            idx = np.random.default_rng(self.random_state).choice(len(bg), size=size, replace=False)
            bg = bg[idx]
        self._background = bg

    @property
    def background(self) -> NDArray[np.float32]:
        """The background rows in use, after any subsampling (a copy)."""
        out: NDArray[np.float32] = self._background.copy()
        return out

    def _names_and_background(
        self, X: object, p: int, feature_names: Sequence[str] | None
    ) -> tuple[tuple[str, ...], NDArray[np.float32]]:
        """Feature names for ``X`` and the background with its columns in the same order."""
        bg = self._background
        if bg.shape[1] != p:
            raise ValueError(f"X has {p} features but the background has {bg.shape[1]}.")
        names = _column_names(X)
        if feature_names is not None:
            given = tuple(str(name) for name in feature_names)
            if len(given) != p:
                raise ValueError(f"feature_names has {len(given)} entries; X has {p} columns.")
            if names is not None and given != names:
                raise ValueError("feature_names does not match the columns of X.")
            names = given
        if names is not None and len(set(names)) != p:
            raise ValueError("feature names must be unique.")

        bg_cols = self._background_columns
        if bg_cols is None:
            return (names if names is not None else tuple(f"x{j}" for j in range(p))), bg
        if names is None:
            return bg_cols, bg
        if set(names) != set(bg_cols):
            raise ValueError(
                "The columns of X and of the background differ: "
                f"only in X {sorted(set(names) - set(bg_cols))}, "
                f"only in background {sorted(set(bg_cols) - set(names))}."
            )
        order = [bg_cols.index(name) for name in names]
        return names, bg[:, order]

    def explain(
        self,
        X: ArrayLike | pd.DataFrame,
        time: ArrayLike,
        event: ArrayLike,
        times: ArrayLike,
        *,
        censoring_time: ArrayLike | None = None,
        censoring_event: ArrayLike | None = None,
        feature_names: Sequence[str] | None = None,
    ) -> TimeSAGEResult:
        """Estimate time-indexed importances on an evaluation set.

        Parameters
        ----------
        X : array_like or pandas.DataFrame of shape (n, p)
            Rows whose IPCW Brier loss is decomposed, e.g. a test set. A
            DataFrame supplies feature names; if the background is also a
            DataFrame, its columns are matched to ``X`` by name.
        time : array_like of shape (n,)
            Observed times ``Y_i = min(T_i, C_i)``.
        event : array_like of shape (n,)
            Event indicators: 1 if the event was observed, 0 if censored.
        times : array_like of shape (K,)
            Strictly increasing, non-negative evaluation horizons, no later
            than the largest observed time.
        censoring_time, censoring_event : array_like of shape (n_c,), optional
            Data for fitting the censoring Kaplan-Meier curve ``G_hat`` (Eq.
            16). Give both or neither. By default ``G_hat`` is fitted on
            ``(time, event)``, which is standard for IPCW evaluation and is
            what the original study did. With a small evaluation set (a few
            hundred rows) ``G_hat``, and so the weights ``1/G_hat``, are
            noisy; a larger sample from the same censoring mechanism, such as
            the training data, can be passed here instead.
        feature_names : sequence of str, optional
            Names for the columns of ``X``. Defaults to the DataFrame columns,
            then the background's columns, then ``x0, x1, ...``.

        Returns
        -------
        TimeSAGEResult

        Raises
        ------
        ValueError
            On shape mismatches, event values outside {0, 1}, an invalid grid,
            grid times beyond the observed follow-up (of the evaluation set or
            of the censoring data), ``G_hat(t_k) = 0`` at a grid time, or model
            output of the wrong shape, non-finite, or outside ``[0, 1]``. Shape
            and finiteness are checked on every model call, including calls on
            imputed rows; the range check runs on ``X`` only.

        Warns
        -----
        UserWarning
            If the model's survival curves on ``X`` increase over time by more
            than rounding error. Separately, if ``G_hat(t_k) < 0.05`` at any grid
            time (listed in the warning): the weights ``1/G_hat`` exceed 20 there
            and the estimates are unreliable. Grids running to a high quantile
            of follow-up often trigger this; ``result.censoring_survival`` holds
            ``G_hat`` on the grid.
        """
        x = _as_float32_matrix(X, "X")
        n, p = x.shape
        T, delta = _as_survival_data(time, event)
        if len(T) != n:
            raise ValueError(f"X has {n} rows but time and event have {len(T)}.")
        names, bg = self._names_and_background(X, p, feature_names)
        grid = _validate_grid(times)
        _check_follow_up(grid, T, "the explained data")

        if (censoring_time is None) != (censoring_event is None):
            raise ValueError("Pass censoring_time and censoring_event together, or neither.")
        if censoring_time is not None and censoring_event is not None:
            T_c, delta_c = _as_survival_data(censoring_time, censoring_event)
            _check_follow_up(grid, T_c, "the censoring data")
            censoring = fit_censoring_km(T_c, delta_c)
        else:
            censoring = fit_censoring_km(T, delta)
        _check_censoring_positive(censoring, grid)
        g_grid = censoring(grid)
        _warn_low_censoring_survival(g_grid, grid)

        K = len(grid)
        _check_trial_predictions(np.asarray(self._predict(x, grid)), n, K)
        value = CoalitionValue(self._predict, bg, grid, censoring)
        loss_empty = value.loss(x, np.zeros((n, p), dtype=bool), T, delta).mean(axis=0)
        loss_full = value.loss(x, np.ones((n, p), dtype=bool), T, delta).mean(axis=0)

        logger.info(
            "TimeSAGE: n=%d, p=%d, K=%d, m=%d, B=%d, batch_size=%d, random_state=%d",
            n, p, K, len(bg), self.n_permutations, self.batch_size, self.random_state,
        )  # fmt: skip
        outcome = _sample_permutations(
            value,
            x,
            T,
            delta,
            n_permutations=self.n_permutations,
            batch_size=self.batch_size,
            convergence_threshold=self.convergence_threshold,
            seed=self.random_state,
            progress=self.progress,
        )
        return TimeSAGEResult(
            phi=outcome.values,
            phi_se=outcome.std,
            times=grid,
            feature_names=names,
            censoring_survival=g_grid,
            at_risk_fraction=np.mean(T[:, None] >= grid[None, :], axis=0),
            _at_risk_fraction_legacy=np.mean(T[:, None] > grid[None, :], axis=0),
            loss_empty=loss_empty,
            loss_full=loss_full,
            n_permutations=outcome.n_permutations,
            converged=outcome.converged,
            convergence_ratio=outcome.convergence_ratio,
            random_state=self.random_state,
        )
