"""The ``TimeSAGEResult`` container (public).

Holds the ``(p, K)`` importance matrix ``Phi`` with its Monte Carlo standard
errors, the time grid, feature names, the per-time weights that weighted
integration needs (Section 5.8), the empty- and full-coalition losses for the
efficiency check, and a record of how the run ended. It contains only
summaries, never patient-level rows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from timesage.integrate import IntegrationMethod, integrate

__all__ = ["TimeSAGEResult"]


def _frozen(a: ArrayLike, name: str, shape: tuple[int, ...]) -> NDArray[np.float64]:
    out = np.array(a, dtype=np.float64, copy=True)
    if out.shape != shape:
        raise ValueError(f"{name} must have shape {shape}; got {out.shape}.")
    out.setflags(write=False)
    return out


@dataclass(frozen=True, eq=False)
class TimeSAGEResult:
    """Time-indexed TimeSAGE importances and the information needed to summarise them.

    Created by :meth:`timesage.TimeSAGE.explain`. All arrays are read-only
    copies.

    Attributes
    ----------
    phi : ndarray of shape (p, K)
        ``phi[j, k]`` estimates the Shapley value of feature ``j`` in the game
        ``v_{t_k}(S) = -BS_IPCW(t_k; S)`` (Eq. 11-12, 19). Positive means the
        feature reduces IPCW Brier loss at ``t_k``. Negative values are
        legitimate.
    phi_se : ndarray of shape (p, K)
        Monte Carlo standard error of each entry of ``phi``: the square root
        of the per-permutation variance divided by the number of
        permutations. As in sage-importance, the variance divides by ``N``
        rather than ``N - 1``.
    times : ndarray of shape (K,)
        Evaluation horizons ``t_1 < ... < t_K``.
    feature_names : tuple of str
        Feature names, in the order of the rows of ``phi``.
    censoring_survival : ndarray of shape (K,)
        ``G_hat(t_k)``, the censoring Kaplan-Meier curve used by the IPCW
        weights (Eq. 16). Weight of the ``censoring`` integration variant.
    at_risk_fraction : ndarray of shape (K,)
        ``(1/n) sum_i 1{Y_i >= t_k}`` over the explained rows, the textbook
        risk set. Weight of the ``at_risk`` integration variant. The original
        research implementation used ``1{Y_i > t_k}``; results from
        :meth:`~timesage.TimeSAGE.explain` also keep that fraction privately for
        ``integrated(..., legacy_parity=True)``.
    loss_empty, loss_full : ndarray of shape (K,)
        ``BS_IPCW(t_k; S)`` over the explained rows for the empty and the full
        coalition, computed through the same marginal imputation and float32
        inputs as the estimator.
    n_permutations : int
        Permutations actually used. Below the requested cap only if early
        stopping fired.
    converged : bool or None
        Whether early stopping fired; ``None`` if it was off.
    convergence_ratio : float
        Final value of the stopping measure: over times, the worst ratio of
        the largest standard error to the spread of ``phi`` across features.
    random_state : int
        Seed actually used. Passing it back as ``random_state`` reproduces the
        run.
    """

    phi: NDArray[np.float64]
    phi_se: NDArray[np.float64]
    times: NDArray[np.float64]
    feature_names: tuple[str, ...]
    censoring_survival: NDArray[np.float64]
    at_risk_fraction: NDArray[np.float64]
    loss_empty: NDArray[np.float64]
    loss_full: NDArray[np.float64]
    n_permutations: int
    converged: bool | None
    convergence_ratio: float
    random_state: int
    _at_risk_fraction_legacy: NDArray[np.float64] | None = field(default=None, repr=False)
    _p_k: tuple[int, int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate shapes and replace arrays with read-only float64 copies."""
        phi = np.asarray(self.phi)
        if phi.ndim != 2:
            raise ValueError(f"phi must be 2-D (p, K); got shape {phi.shape}.")
        p, K = phi.shape
        names = tuple(str(n) for n in self.feature_names)
        if len(names) != p:
            raise ValueError(f"feature_names has {len(names)} entries; phi has {p} rows.")
        values: dict[str, Any] = {
            "phi": _frozen(phi, "phi", (p, K)),
            "phi_se": _frozen(self.phi_se, "phi_se", (p, K)),
            "times": _frozen(self.times, "times", (K,)),
            "feature_names": names,
            "censoring_survival": _frozen(self.censoring_survival, "censoring_survival", (K,)),
            "at_risk_fraction": _frozen(self.at_risk_fraction, "at_risk_fraction", (K,)),
            "loss_empty": _frozen(self.loss_empty, "loss_empty", (K,)),
            "loss_full": _frozen(self.loss_full, "loss_full", (K,)),
            "n_permutations": int(self.n_permutations),
            "convergence_ratio": float(self.convergence_ratio),
            "random_state": int(self.random_state),
            "_at_risk_fraction_legacy": (
                None
                if self._at_risk_fraction_legacy is None
                else _frozen(self._at_risk_fraction_legacy, "_at_risk_fraction_legacy", (K,))
            ),
            "_p_k": (p, K),
        }
        for name, value in values.items():
            object.__setattr__(self, name, value)

    @property
    def n_features(self) -> int:
        """Number of features ``p``."""
        return self._p_k[0]

    @property
    def n_times(self) -> int:
        """Number of evaluation horizons ``K``."""
        return self._p_k[1]

    def efficiency_residual(self) -> NDArray[np.float64]:
        """Absolute efficiency residual at each horizon (Eq. 39).

        ``|sum_j phi_j(t_k) - (BS_IPCW(t_k; empty) - BS_IPCW(t_k; full))|``.
        The Shapley values of the dataset-level game sum exactly to the loss
        reduction, so the residual measures Monte Carlo error only. Its 95th
        percentile over the grid is ``eff_p95``.

        Both losses use the estimator's float32 inputs. The original
        synthetic study computed the full-model loss from the original float64 inputs,
        so its ``eff_p95`` also contains float32 rounding (of order 1e-7).

        Returns
        -------
        ndarray of shape (K,)
        """
        out: NDArray[np.float64] = np.abs(self.phi.sum(axis=0) - (self.loss_empty - self.loss_full))
        return out

    def integrated(
        self,
        method: IntegrationMethod = "censoring",
        *,
        window: ArrayLike | None = None,
        legacy_parity: bool = False,
    ) -> pd.Series:
        """Integrated importance per feature (Eq. 20-24); see :func:`timesage.integrate`.

        Parameters
        ----------
        method : {"censoring", "at_risk", "trapezoid"}, default "censoring"
            Weight applied to the trapezoid weights of the grid.
        window : array_like of bool, shape (K,), optional
            Integrate only over these grid times, e.g. :meth:`stable_window`.
            The weights are full-grid trapezoid weights zeroed outside the
            window.
        legacy_parity : bool, default False
            Use the original implementation's at-risk fraction ``1{Y_i > t}``.

        Returns
        -------
        pandas.Series
            One value per feature, indexed by feature name.
        """
        return integrate(self, method, window=window, legacy_parity=legacy_parity)

    def stable_window(
        self,
        *,
        min_at_risk: float = 0.10,
        min_censoring_survival: float = 0.05,
        legacy_parity: bool = False,
    ) -> NDArray[np.bool_]:
        """Stable support window (Eq. 31) from this result's stored weights.

        ``W = {t_k : R_hat(t_k) >= min_at_risk and G_hat(t_k) >= min_censoring_survival}``,
        using :attr:`at_risk_fraction` and :attr:`censoring_survival`. Both
        come from the explained rows, unless ``explain`` was given separate
        censoring data, in which case ``G_hat`` is that curve.

        The original synthetic study fitted ``G_hat`` on the training
        set instead, so its window can differ from this one. Use
        :func:`timesage.metrics.stable_window` with ``censoring_time`` and
        ``censoring_event`` to reproduce it.

        Parameters
        ----------
        min_at_risk, min_censoring_survival : float
            Thresholds; the defaults are those of Eq. 31.
        legacy_parity : bool, default False
            Use the original implementation's at-risk fraction ``1{Y > t}``.

        Returns
        -------
        ndarray of bool, shape (K,)
        """
        from timesage.metrics import _stable_mask  # metrics imports the estimator

        at_risk = self._at_risk_fraction_legacy if legacy_parity else self.at_risk_fraction
        if at_risk is None:
            raise ValueError(
                "This result does not hold the legacy at-risk fraction; "
                "legacy_parity=True needs a result created by TimeSAGE.explain."
            )
        return _stable_mask(at_risk, self.censoring_survival, min_at_risk, min_censoring_survival)

    def to_frame(self) -> pd.DataFrame:
        """Long-format table with columns ``feature``, ``time``, ``phi``, ``se``.

        Rows are ordered by feature, then time.
        """
        p, K = self._p_k
        frame: pd.DataFrame = pd.DataFrame(
            {
                "feature": np.repeat(np.array(self.feature_names, dtype=object), K),
                "time": np.tile(self.times, p),
                "phi": self.phi.ravel(),
                "se": self.phi_se.ravel(),
            }
        )
        return frame

    def __repr__(self) -> str:
        """Short summary: shape, grid range and permutations used."""
        p, K = self._p_k
        stop = "" if self.converged is None else f", converged={self.converged}"
        return (
            f"TimeSAGEResult(p={p}, K={K}, times=[{self.times[0]:g}, {self.times[-1]:g}], "
            f"n_permutations={self.n_permutations}{stop}, random_state={self.random_state})"
        )
