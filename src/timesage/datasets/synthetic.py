r"""Synthetic survival data with a known time-varying effect (Section 7.7).

A port of the synthetic study of the original research implementation
(``scenarios.py``, ``hazards.py``, ``censoring.py``, ``data_generation.py``).
Data are generated on the fly from a seed; nothing is bundled.

**Data-generating process.** Six standard-normal features ``x1..x6``
(correlated only between ``x1`` and ``x2``, in E3 and H1). Event times follow
a proportional-hazards model with time-varying coefficients,

.. math::

    h(t \mid x) = h_0(t) \exp\big(\operatorname{clip}(x' \beta(t), -4, 4)\big),

with a piecewise-constant baseline ``h_0`` ("piecewise_irregular": cuts 0,
35, 90, 150, 240 months). ``x1`` has a logistic-hump coefficient, ``x2..x4``
constant coefficients, and ``x5, x6`` none (pure noise). Event times are drawn
by inverting the cumulative hazard (Bender, Augustin and Blettner, 2005).
Censoring is exponential and independent of the features, with its rate
chosen so that the *expected* censored fraction, given the sample's event
times, equals the scenario's target.

**What the generator actually uses.** The hazard is integrated exactly on the
intervals between the cuts {0, 35, 60, 90, 120, 150, 240} (the baseline's and
the scenarios' cuts), with ``beta`` evaluated **at each interval's midpoint**.
So the coefficients that generate the data are the smooth curves of
:meth:`Scenario.beta` sampled at six midpoints and held constant in between,
and the hazard after 240 months continues at the last interval's rate.
:meth:`Scenario.predict_survival` is the exact survival function of *that*
process.

Limitations
-----------
- The linear predictor is clipped to ``[-4, 4]``, so this is not exactly a
  Cox model for extreme feature values.
- The censoring rate is solved from each sample's own event times, so it
  varies slightly between seeds; censoring is still independent of the
  features.
- Only proportional-hazards-compatible scenarios: no crossing hazards, no
  competing risks.
- Reproduced bit for bit on the platform where it was checked. Feature draws
  use ``Generator.multivariate_normal(method="svd")``, whose output depends on
  the linear-algebra library, so bit-identical data on other platforms is not
  guaranteed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
import pandas as pd

if TYPE_CHECKING:
    from timesage.result import TimeSAGEResult

__all__ = [
    "FEATURE_NAMES",
    "SCENARIOS",
    "OracleResult",
    "Scenario",
    "SyntheticDataset",
    "event_density",
    "get_scenario",
    "make_synthetic",
    "oracle_grid",
    "oracle_importance",
]

FloatArray = npt.NDArray[np.float64]

FEATURE_NAMES: tuple[str, ...] = ("x1", "x2", "x3", "x4", "x5", "x6")
SCENARIOS: tuple[str, ...] = ("E1", "E2", "E3", "E4", "H1")

# configs/defaults.yaml and hazards.py of the original synthetic study.
MAX_FOLLOWUP = 240.0
SPLIT_SIZES: tuple[int, int, int] = (500, 150, 200)  # train, validation, test
_SCENARIO_CUTS = np.array([0.0, 60.0, 120.0, 240.0])
_BASELINE_CUTS = np.array([0.0, 35.0, 90.0, 150.0, 240.0])
_BASELINE_RATES = np.array([0.0060, 0.0120, 0.0048, 0.0100])
_CENSORING_TARGETS = {"low": 0.25, "medium": 0.45, "moderate_high": 0.55}
_LP_CLIP = 4.0


# ------------------------------------------------------------------ scenarios


@dataclass(frozen=True)
class Scenario:
    """One synthetic scenario (Table 7.6).

    Also a ``SurvivalModel``: :meth:`predict_survival` is the exact survival
    function of the data-generating process, on raw (unscaled) features.

    Attributes
    ----------
    id, name : str
        For example ``"E1"`` and ``"E1_easy_control"``.
    x1_start, x1_end, x1_rise_width, x1_fall_width, x1_amplitude, x1_baseline : float
        Parameters of ``x1``'s logistic-hump coefficient.
    x2_effect, x3_effect, x4_effect : float
        Constant coefficients of ``x2..x4``. ``x5`` and ``x6`` have none.
    corr_x1_x2 : float
        Correlation between ``x1`` and ``x2``; all other features are
        independent with unit variance.
    censoring : str
        Censoring regime: ``"low"`` (target 25% censored), ``"medium"`` (45%)
        or ``"moderate_high"`` (55%).
    """

    id: str
    name: str
    x1_start: float
    x1_end: float
    x1_rise_width: float
    x1_fall_width: float
    x1_amplitude: float
    x1_baseline: float
    x2_effect: float
    x3_effect: float
    x4_effect: float
    corr_x1_x2: float
    censoring: str

    @property
    def feature_names(self) -> tuple[str, ...]:
        """``("x1", ..., "x6")``."""
        return FEATURE_NAMES

    @property
    def covariance(self) -> FloatArray:
        """Feature covariance: identity, plus ``corr_x1_x2`` between x1 and x2."""
        cov = np.eye(len(FEATURE_NAMES), dtype=float)
        if abs(self.corr_x1_x2) > 0.0:
            cov[0, 1] = cov[1, 0] = self.corr_x1_x2
        return cov

    @property
    def censoring_target(self) -> float:
        """Target censored fraction of the scenario's censoring regime."""
        return _CENSORING_TARGETS[self.censoring]

    def beta(self, times: npt.ArrayLike) -> FloatArray:
        """Design coefficients ``beta(t)``, shape ``(6, n_times)``.

        These are the smooth curves; the generator uses them at interval
        midpoints only (see the module docstring).
        """
        t = np.asarray(times, dtype=float)
        rise = 1.0 / (1.0 + np.exp(-(t - self.x1_start) / max(self.x1_rise_width, 1e-6)))
        fall = 1.0 / (1.0 + np.exp((t - self.x1_end) / max(self.x1_fall_width, 1e-6)))
        x1 = self.x1_baseline + self.x1_amplitude * rise * fall
        consts = [self.x2_effect, self.x3_effect, self.x4_effect, 0.0, 0.0]
        return np.vstack([x1, *(np.full_like(t, fill_value=c, dtype=float) for c in consts)])

    def predict_survival(self, X: npt.ArrayLike, times: npt.ArrayLike) -> FloatArray:
        """Exact survival ``S(t | x)`` of the data-generating process.

        Parameters
        ----------
        X : array-like of shape (n_samples, 6)
            Raw features ``x1..x6`` (not standardised).
        times : array-like of shape (n_times,)
            Non-negative times, in any order.

        Returns
        -------
        numpy.ndarray of shape (n_samples, n_times)
            ``exp(-H(t | x))``, with ``H`` integrated exactly over the
            generator's piecewise-constant hazard (S(0) = 1).
        """
        X_mat = np.asarray(X, dtype=float)
        if X_mat.ndim != 2 or X_mat.shape[1] != len(FEATURE_NAMES):
            raise ValueError(f"X must have shape (n_samples, 6), got {X_mat.shape}.")
        t = np.asarray(times, dtype=float)
        if t.ndim != 1 or not np.all(np.isfinite(t)) or np.any(t < 0):
            raise ValueError(
                "times must be a one-dimensional array of finite, non-negative values."
            )
        cuts, rates = _piecewise_rates(self, X_mat)  # (J + 1,), (n, J)
        left, right = cuts[:-1], cuts[1:]
        exposure = np.clip(np.minimum(t[:, None], right[None, :]) - left[None, :], 0.0, None)
        exposure = np.minimum(exposure, (right - left)[None, :])  # (K, J)
        H = rates @ exposure.T + rates[:, -1:] * np.clip(t - cuts[-1], 0.0, None)[None, :]
        S: FloatArray = np.exp(-H)
        return S


def _scenario(
    id: str,
    name: str,
    *,
    x2: float,
    corr: float,
    censoring: str,
    amplitude: float,
    start: float = 34.0,
    end: float = 118.0,
    rise: float = 11.0,
    fall: float = 15.0,
    baseline: float = 0.08,
) -> Scenario:
    return Scenario(
        id=id,
        name=name,
        x1_start=start,
        x1_end=end,
        x1_rise_width=rise,
        x1_fall_width=fall,
        x1_amplitude=amplitude,
        x1_baseline=baseline,
        x2_effect=x2,
        x3_effect=0.22,
        x4_effect=0.08,
        corr_x1_x2=corr,
        censoring=censoring,
    )


# As scenarios.py: scenario_library() of the original synthetic study.
_SCENARIO_LIBRARY: dict[str, Scenario] = {
    s.id: s
    for s in (
        _scenario("E1", "E1_easy_control", x2=0.30, corr=0.0, censoring="low", amplitude=1.5),
        _scenario(
            "E2", "E2_easy_strong_competitor", x2=1.30, corr=0.0, censoring="low", amplitude=0.92
        ),
        _scenario(
            "E3",
            "E3_easy_moderate_correlation",
            x2=0.85,
            corr=0.30,
            censoring="low",
            amplitude=0.92,
        ),
        _scenario(
            "E4",
            "E4_easy_moderate_censoring",
            x2=0.85,
            corr=0.0,
            censoring="medium",
            amplitude=0.90,
            end=112.0,
            rise=10.0,
            fall=14.0,
        ),
        _scenario(
            "H1",
            "H1_combined_moderate",
            x2=1.20,
            corr=0.45,
            censoring="moderate_high",
            amplitude=0.96,
            start=62.0,
            end=150.0,
            rise=13.0,
            fall=18.0,
            baseline=0.06,
        ),
    )
}


def get_scenario(scenario: str) -> Scenario:
    """Look up a scenario by id (``"E1"``) or full name (``"E1_easy_control"``)."""
    for s in _SCENARIO_LIBRARY.values():
        if scenario in (s.id, s.name):
            return s
    raise ValueError(f"Unknown scenario {scenario!r}; expected one of {list(SCENARIOS)}.")


# ------------------------------------------------------------------- hazards


def _baseline_hazard_at(times: FloatArray) -> FloatArray:
    idx = np.searchsorted(_BASELINE_CUTS[1:], times, side="right")
    idx = np.clip(idx, 0, len(_BASELINE_RATES) - 1)
    out: FloatArray = _BASELINE_RATES[idx]
    return out


def _baseline_cumulative_hazard(times: FloatArray) -> FloatArray:
    out = np.zeros_like(times, dtype=float)
    for i, rate in enumerate(_BASELINE_RATES):
        left, right = _BASELINE_CUTS[i], _BASELINE_CUTS[i + 1]
        out += rate * np.clip(np.minimum(times, right) - left, 0.0, right - left)
    out += np.clip(times - _BASELINE_CUTS[-1], 0.0, None) * _BASELINE_RATES[-1]
    return out


def event_density(times: npt.ArrayLike) -> FloatArray:
    """Baseline event density ``h0(t) S0(t)``: the study's NWIAE weights.

    Pass as ``weights=`` to :func:`timesage.metrics.nwiae` to reproduce the
    synthetic study's curve metrics (``hazards.event_density``). At a
    cut point the hazard of the following interval is used.
    """
    t = np.asarray(times, dtype=float)
    out: FloatArray = _baseline_hazard_at(t) * np.exp(-_baseline_cumulative_hazard(t))
    return out


def _union_cuts(scenario: Scenario) -> FloatArray:
    return np.unique(np.concatenate([[0.0, MAX_FOLLOWUP], _BASELINE_CUTS, _SCENARIO_CUTS]))


def _piecewise_rates(scenario: Scenario, X: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Cuts and per-subject hazard rates of the generator (as ``_simulate_event_times``)."""
    cuts = _union_cuts(scenario)
    widths = np.diff(cuts)
    midpoints = cuts[:-1] + 0.5 * widths
    lp = X @ scenario.beta(midpoints)
    rates: FloatArray = _baseline_hazard_at(midpoints)[None, :] * np.exp(
        np.clip(lp, -_LP_CLIP, _LP_CLIP)
    )
    return cuts, rates


# ------------------------------------------------------------------ sampling


def _sample_event_times(X: FloatArray, scenario: Scenario, rng: np.random.Generator) -> FloatArray:
    # Arithmetic and draw order of the original data_generation._simulate_event_times.
    cuts, rates = _piecewise_rates(scenario, X)
    widths = np.diff(cuts)
    n = len(X)
    expo_draws = rng.exponential(scale=1.0, size=n)
    cumulative = np.cumsum(rates * widths[None, :], axis=1)
    hit_mask = cumulative >= expo_draws[:, None]
    has_hit = hit_mask.any(axis=1)
    hit_idx = np.where(has_hit, hit_mask.argmax(axis=1), len(widths) - 1)
    prev_cum = np.where(hit_idx > 0, cumulative[np.arange(n), np.clip(hit_idx - 1, 0, None)], 0.0)
    event_times = cuts[hit_idx] + (expo_draws - prev_cum) / rates[np.arange(n), hit_idx]
    no_hit = ~has_hit
    if np.any(no_hit):
        residual = expo_draws[no_hit] - cumulative[no_hit, -1]
        event_times[no_hit] = cuts[-1] + residual / rates[no_hit, -1]
    out: FloatArray = event_times.astype(float)
    return out


def _expected_censored_fraction(event_times: FloatArray, rate: float) -> float:
    return float(np.mean(1.0 - np.exp(-rate * event_times)))


def _solve_censoring_rate(event_times: FloatArray, target: float, max_iter: int = 80) -> float:
    # Bisection of the original censoring.solve_exponential_censoring_rate.
    if target == 0.0:
        return 0.0
    low, high = 0.0, 0.01
    while _expected_censored_fraction(event_times, high) < target:
        high *= 2.0
        if high > 100.0:
            break
    for _ in range(max_iter):
        mid = 0.5 * (low + high)
        if _expected_censored_fraction(event_times, mid) < target:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


def _sample_cohort(
    n: int, scenario: Scenario, rng: np.random.Generator
) -> tuple[pd.DataFrame, float]:
    """Features, latent times and observed data (as ``sample_observed_synthetic_cohort``)."""
    X = rng.multivariate_normal(
        mean=np.zeros(len(FEATURE_NAMES)), cov=scenario.covariance, size=n, method="svd"
    )
    event_time = _sample_event_times(X, scenario, rng)
    rate = _solve_censoring_rate(event_time, scenario.censoring_target)
    censoring_time = (
        np.full(n, np.inf) if rate == 0.0 else rng.exponential(scale=1.0 / rate, size=n)
    )
    frame = pd.DataFrame(X, columns=list(FEATURE_NAMES))
    frame["time"] = np.minimum(event_time, censoring_time).astype(float)
    frame["event"] = (event_time <= censoring_time).astype(np.int64)
    frame["event_time"] = event_time
    frame["censoring_time"] = censoring_time.astype(float)
    return frame, rate


@dataclass(frozen=True)
class SyntheticDataset:
    """One synthetic dataset, split 500 / 150 / 200 as in the study.

    Each split has columns ``x1..x6``, ``time``, ``event`` and the latent
    ``event_time`` and ``censoring_time``. The latent columns are known only
    because the data are simulated: **never pass them to**
    :meth:`TimeSAGE.explain` **or a model**; use ``time`` and ``event``.

    Attributes
    ----------
    train, validation, test : pandas.DataFrame
    scenario : Scenario
    seed : int
    censoring_rate : float
        Rate of the exponential censoring distribution (per month).
    censored_fraction : float
        Fraction of all 850 subjects that are censored.
    """

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    scenario: Scenario
    seed: int
    censoring_rate: float
    censored_fraction: float

    @property
    def feature_names(self) -> tuple[str, ...]:
        """``("x1", ..., "x6")``."""
        return FEATURE_NAMES

    def arrays(self, split: str) -> tuple[FloatArray, FloatArray, npt.NDArray[np.int64]]:
        """``(X, time, event)`` of one split, ready for a model or ``TimeSAGE.explain``."""
        frame = {"train": self.train, "validation": self.validation, "test": self.test}[split]
        return (
            frame[list(FEATURE_NAMES)].to_numpy(dtype=float),
            frame["time"].to_numpy(dtype=float),
            frame["event"].to_numpy(dtype=np.int64),
        )


def make_synthetic(scenario: str = "E1", seed: int = 101) -> SyntheticDataset:
    """Generate one dataset of the synthetic study.

    Reproduces the original synthetic study's generated datasets for the
    same scenario and seed (the study used seeds 101, 202, ..., 1010).

    Parameters
    ----------
    scenario : str, default "E1"
        Scenario id or name; see :data:`SCENARIOS`.
    seed : int, default 101
        Seed of ``numpy.random.default_rng``; the only source of randomness.

    Returns
    -------
    SyntheticDataset
    """
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise TypeError(f"seed must be an integer, got {type(seed).__name__}.")
    spec = get_scenario(scenario)
    rng = np.random.default_rng(int(seed))
    n_train, n_val, n_test = SPLIT_SIZES
    full, rate = _sample_cohort(n_train + n_val + n_test, spec, rng)
    if np.any(full["time"].to_numpy() <= 0):  # the original study asserts this
        raise AssertionError("Observed times must be strictly positive.")
    censored = float(1.0 - full["event"].mean())
    perm = rng.permutation(len(full))
    full = full.iloc[perm].reset_index(drop=True)
    parts: list[Any] = [
        full.iloc[a:b].reset_index(drop=True)
        for a, b in ((0, n_train), (n_train, n_train + n_val), (n_train + n_val, len(full)))
    ]
    return SyntheticDataset(
        train=parts[0],
        validation=parts[1],
        test=parts[2],
        scenario=spec,
        seed=int(seed),
        censoring_rate=rate,
        censored_fraction=censored,
    )


# -------------------------------------------------------------------- oracle

# Oracle settings of the original research implementation: oracle_truth and budgets.main.
_ORACLE_SEED_OFFSET = 50_000
_ORACLE_GRID_SIZE = 24
_ORACLE_MIN_POSITIVE_TIME = 0.001


class _Standardiser:
    """``(x - mean) / std`` fitted on training rows, as scikit-learn's ``StandardScaler``.

    Population standard deviation, and a zero scale replaced by 1, as
    scikit-learn does. On every training set of the study, mean and scale are
    bit-identical to scikit-learn 1.7.2 and 1.9.1 (checked in the tests).
    """

    def __init__(self, X: FloatArray) -> None:
        self.mean = X.mean(axis=0)
        scale = X.std(axis=0)
        self.scale = np.where(scale == 0.0, 1.0, scale)

    def transform(self, X: FloatArray) -> FloatArray:
        out: FloatArray = (np.array(X, dtype=float) - self.mean) / self.scale
        return out

    def inverse_transform(self, X: npt.ArrayLike) -> FloatArray:
        out = np.array(X, dtype=float)
        out *= self.scale
        out += self.mean
        return out


def _legacy_oracle_survival(scenario: Scenario, X_raw: FloatArray, times: FloatArray) -> FloatArray:
    # Arithmetic of the original truth.oracle_survival_from_dgp, in the same
    # order: S is read one breakpoint after each query time, on a finer
    # partition than the generator's.
    union = np.unique(np.concatenate([[0.0], _BASELINE_CUTS, _SCENARIO_CUTS, times]))
    widths = np.diff(union)
    midpoints = union[:-1] + 0.5 * widths
    lp = X_raw @ scenario.beta(midpoints)
    hazard = _baseline_hazard_at(midpoints)[None, :] * np.exp(np.clip(lp, -_LP_CLIP, _LP_CLIP))
    cumulative = np.cumsum(hazard * widths[None, :], axis=1)
    idx = np.searchsorted(union[1:], times, side="right")
    idx = np.clip(idx, 0, cumulative.shape[1] - 1)
    out: FloatArray = np.exp(-cumulative[:, idx])
    return out


class _OraclePredictor:
    """True survival on standardised inputs: unscale, then the DGP's survival."""

    def __init__(self, scenario: Scenario, scaler: _Standardiser, legacy_parity: bool) -> None:
        self.scenario, self.scaler, self.legacy_parity = scenario, scaler, legacy_parity

    def predict_survival(self, X: npt.ArrayLike, times: npt.ArrayLike) -> FloatArray:
        X_raw = self.scaler.inverse_transform(X)
        t = np.asarray(times, dtype=float)
        if self.legacy_parity:
            return _legacy_oracle_survival(self.scenario, X_raw, t)
        return self.scenario.predict_survival(X_raw, t)


def oracle_grid(data: SyntheticDataset) -> FloatArray:
    """Return the study's 24-point oracle grid, from the first positive test time to 240."""
    t = data.test["time"].to_numpy(dtype=float)
    start = max(float(np.min(t[t > 0])), _ORACLE_MIN_POSITIVE_TIME)
    return np.linspace(start, MAX_FOLLOWUP, _ORACLE_GRID_SIZE)


@dataclass(frozen=True)
class OracleResult:
    """Oracle importance for one synthetic dataset.

    Attributes
    ----------
    result : TimeSAGEResult
        TimeSAGE on the true survival function: ``result.phi`` are the oracle
        curves and ``result.phi_se`` their Monte Carlo standard errors.
    stable_window : numpy.ndarray of bool
        The study's stable window (Eq. 31): ``G_hat`` fitted on the training
        split, the at-risk fraction on the test split.
    integrated : pandas.Series
        Trapezoid integral over the stable window, the original study's
        integrated target.
    """

    result: TimeSAGEResult
    stable_window: npt.NDArray[np.bool_]
    integrated: pd.Series

    @property
    def phi(self) -> FloatArray:
        """Oracle importance curves, shape ``(6, n_times)``."""
        return self.result.phi

    @property
    def phi_se(self) -> FloatArray:
        """Monte Carlo standard errors of :attr:`phi`."""
        return self.result.phi_se

    @property
    def times(self) -> FloatArray:
        """The evaluation grid."""
        return self.result.times


def oracle_importance(
    data: SyntheticDataset,
    *,
    times: npt.ArrayLike | None = None,
    n_eval: int = 1024,
    n_background: int = 256,
    n_permutations: int = 4096,
    batch_size: int = 128,
    legacy_parity: bool = False,
) -> OracleResult:
    """Oracle importance curves: TimeSAGE applied to the true survival function.

    Implements the synthetic study's oracle (Section 7.7; ``oracle_runner``
    of the original research implementation). A fresh cohort of ``n_eval`` subjects and a background
    of ``n_background`` are drawn from the same scenario with seed
    ``data.seed + 50000``. Features are standardised with the training
    split's mean and standard deviation and cast to float32, as for the
    models, and TimeSAGE (IPCW Brier loss, marginal imputation, joint times,
    no early stopping) explains a predictor that maps them back to raw
    features and returns the DGP's survival, :meth:`Scenario.predict_survival`.

    The oracle is an estimate, not an exact Shapley value: it averages
    ``n_permutations`` sampled permutations and so carries Monte Carlo error,
    reported in ``phi_se``. Features without an effect (x5, x6) get exactly
    zero. The weak feature x4 (and x3 in some H1 datasets) has an integrated
    importance close to zero, whose sign can change with small changes to the
    oracle.

    Parameters
    ----------
    data : SyntheticDataset
    times : array-like, optional
        Evaluation grid; default :func:`oracle_grid` (24 points to 240
        months, as in the study).
    n_eval, n_background, n_permutations, batch_size : int
        Default to the study's oracle settings.
    legacy_parity : bool, default False
        Use the survival function of the original research implementation's
        oracle, which reads the cumulative hazard one breakpoint after each
        query time and uses a finer partition than the generator, and its
        at-risk fraction ``1{Y > t}`` in the stable window. Reproduces the
        oracle of the original research implementation, which its synthetic
        evaluation used.

    Returns
    -------
    OracleResult
    """
    from timesage.estimator import TimeSAGE
    from timesage.integrate import integrate
    from timesage.metrics import stable_window

    grid = oracle_grid(data) if times is None else np.asarray(times, dtype=float)
    seed = data.seed + _ORACLE_SEED_OFFSET
    rng = np.random.default_rng(seed)
    eval_df, _ = _sample_cohort(int(n_eval), data.scenario, rng)
    bg_df, _ = _sample_cohort(int(n_background), data.scenario, rng)
    names = list(FEATURE_NAMES)
    scaler = _Standardiser(data.train[names].to_numpy(dtype=float))
    X_eval = scaler.transform(eval_df[names].to_numpy(dtype=float)).astype(np.float32)
    X_bg = scaler.transform(bg_df[names].to_numpy(dtype=float)).astype(np.float32)

    explainer = TimeSAGE(
        _OraclePredictor(data.scenario, scaler, bool(legacy_parity)),
        background=X_bg,
        n_permutations=int(n_permutations),
        batch_size=int(batch_size),
        random_state=seed,
    )
    result = explainer.explain(
        X_eval,
        eval_df["time"].to_numpy(dtype=float),
        eval_df["event"].to_numpy(dtype=np.int64),
        grid,
        feature_names=names,
    )
    window = stable_window(
        grid,
        data.test["time"].to_numpy(dtype=float),
        data.test["event"].to_numpy(dtype=np.int64),
        censoring_time=data.train["time"].to_numpy(dtype=float),
        censoring_event=data.train["event"].to_numpy(dtype=np.int64),
        legacy_parity=bool(legacy_parity),
    )
    integrated = integrate(result, "trapezoid", window=window).rename("integrated_oracle")
    return OracleResult(result=result, stable_window=window, integrated=integrated)
