"""Unit tests for ``timesage.datasets.synthetic``.

Parity with the original study's datasets is in
``tests/characterisation/test_synthetic.py``. Here: structure, determinism,
and that ``Scenario.predict_survival`` is the survival function of the data
the generator draws (probability integral transform).
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import kstest

from timesage import SurvivalModel, TimeSAGE
from timesage.datasets.synthetic import (
    FEATURE_NAMES,
    SCENARIOS,
    SyntheticDataset,
    _sample_cohort,
    event_density,
    get_scenario,
    make_synthetic,
)

COLUMNS = [*FEATURE_NAMES, "time", "event", "event_time", "censoring_time"]


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_structure(scenario: str) -> None:
    data = make_synthetic(scenario, seed=101)
    assert isinstance(data, SyntheticDataset)
    assert [len(data.train), len(data.validation), len(data.test)] == [500, 150, 200]
    for frame in (data.train, data.validation, data.test):
        assert list(frame.columns) == COLUMNS
        np.testing.assert_array_equal(
            frame["time"], np.minimum(frame["event_time"], frame["censoring_time"])
        )
        np.testing.assert_array_equal(
            frame["event"], (frame["event_time"] <= frame["censoring_time"]).astype(int)
        )
        assert (frame["time"] > 0).all()
        assert set(frame["event"].unique()) <= {0, 1}
    # Censoring is tuned to the target in expectation (the original study's tolerance 0.08).
    assert abs(data.censored_fraction - data.scenario.censoring_target) < 0.08
    assert data.censoring_rate > 0


def test_deterministic_and_seed_dependent() -> None:
    a, b, c = make_synthetic("E3", 7), make_synthetic("E3", 7), make_synthetic("E3", 8)
    assert a.train.equals(b.train) and a.test.equals(b.test)
    assert not a.train.equals(c.train)


def test_arrays() -> None:
    data = make_synthetic("E1", 101)
    X, T, E = data.arrays("test")
    assert X.shape == (200, 6) and T.shape == E.shape == (200,)
    assert E.dtype == np.int64


def test_scenario_lookup_and_validation() -> None:
    assert get_scenario("E1") is get_scenario("E1_easy_control")
    assert get_scenario("H1").covariance[0, 1] == 0.45
    with pytest.raises(ValueError, match="Unknown scenario"):
        get_scenario("E9")
    with pytest.raises(TypeError, match="integer"):
        make_synthetic("E1", seed=1.5)  # type: ignore[arg-type]


# ------------------------------------------------------------ true survival


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_predict_survival_is_the_generating_survival(scenario: str) -> None:
    """S(T | x) of the generator's own event times is Uniform(0, 1)."""
    spec = get_scenario(scenario)
    cohort, _ = _sample_cohort(20_000, spec, np.random.default_rng(0))
    X = cohort[list(FEATURE_NAMES)].to_numpy()
    T = cohort["event_time"].to_numpy()
    u = np.array([spec.predict_survival(X[i : i + 1], T[i : i + 1])[0, 0] for i in range(2000)])
    assert kstest(u, "uniform").pvalue > 0.01


def test_predict_survival_matches_numerical_integration() -> None:
    spec = get_scenario("H1")
    x = np.random.default_rng(1).standard_normal(6)
    cuts = np.array([0.0, 35.0, 60.0, 90.0, 120.0, 150.0, 240.0])
    mids = 0.5 * (cuts[:-1] + cuts[1:])
    eta = np.clip(x @ spec.beta(mids), -4, 4)
    h0 = np.array([0.006, 0.012, 0.012, 0.0048, 0.0048, 0.01])  # baseline on these intervals

    def hazard(s: float) -> float:
        j = min(int(np.searchsorted(cuts[1:], s, side="right")), len(mids) - 1)
        return float(h0[j] * np.exp(eta[j]))

    times = np.array([0.0, 20.0, 35.0, 100.0, 240.0, 300.0])
    expected = [np.exp(-quad(hazard, 0, t, points=cuts[1:-1], limit=200)[0]) for t in times]
    np.testing.assert_allclose(spec.predict_survival(x[None, :], times)[0], expected, rtol=1e-9)


def test_scenario_is_a_survival_model() -> None:
    spec = get_scenario("E2")
    assert isinstance(spec, SurvivalModel)
    X = np.random.default_rng(2).standard_normal((50, 6)).astype(np.float32)
    times = np.array([240.0, 0.0, 60.0, 61.0, 500.0])
    S = spec.predict_survival(X, times)
    assert S.shape == (50, 5)
    assert np.all((S >= 0) & (S <= 1))
    assert np.all(S[:, 1] == 1.0)
    order = np.argsort(times)
    assert np.all(np.diff(S[:, order], axis=1) <= 0)
    with pytest.raises(ValueError, match="shape"):
        spec.predict_survival(X[:, :5], times)
    with pytest.raises(ValueError, match="non-negative"):
        spec.predict_survival(X, np.array([-1.0]))


def test_noise_features_get_zero_importance() -> None:
    """x5 and x6 do not enter the true survival, so TimeSAGE gives them exactly 0."""
    data = make_synthetic("E1", 101)
    X, T, E = data.arrays("test")
    bg, _, _ = data.arrays("train")
    result = TimeSAGE(data.scenario, background=bg[:16], n_permutations=8, random_state=0).explain(
        X[:60], T[:60], E[:60], np.array([30.0, 60.0, 90.0])
    )
    np.testing.assert_array_equal(result.phi[4:], 0.0)


def test_event_density() -> None:
    t = np.array([0.0, 10.0, 35.0, 100.0, 300.0])
    d = event_density(t)
    h0 = np.array([0.006, 0.006, 0.012, 0.0048, 0.01])  # at a cut: the next interval's rate
    H0 = np.array([0.0, 0.06, 0.21, 0.21 + 55 * 0.012 + 10 * 0.0048, 0.0])
    H0[4] = 0.21 + 0.66 + 60 * 0.0048 + 90 * 0.01 + 60 * 0.01
    np.testing.assert_allclose(d, h0 * np.exp(-H0), rtol=1e-12)


# -------------------------------------------------------------------- oracle


def test_oracle_predictor_uses_the_generator_survival() -> None:
    """The oracle's survival on standardised inputs is Scenario.predict_survival on raw ones."""
    from timesage.datasets.synthetic import _OraclePredictor, _Standardiser

    data = make_synthetic("H1", 101)
    X = data.test[list(FEATURE_NAMES)].to_numpy()
    scaler = _Standardiser(data.train[list(FEATURE_NAMES)].to_numpy())
    times = np.array([0.0, 12.5, 35.0, 100.0, 240.0])
    exact = _OraclePredictor(data.scenario, scaler, legacy_parity=False)
    np.testing.assert_allclose(
        exact.predict_survival(scaler.transform(X), times),
        data.scenario.predict_survival(X, times),
        rtol=1e-12,
    )


def test_legacy_oracle_reads_the_next_breakpoint() -> None:
    """Pinned: the original oracle returns S at the next breakpoint after t."""
    from timesage.datasets.synthetic import _legacy_oracle_survival

    spec = get_scenario("E1")
    X = np.random.default_rng(3).standard_normal((4, 6))
    times = np.array([0.0, 10.0, 20.0])  # breakpoints 0, 10, 20, 35, ...
    legacy = _legacy_oracle_survival(spec, X, times)
    assert np.all(legacy[:, 0] < 1.0)  # exact S(0) is 1
    # Returned at 0 and 10: survival at 10 and 20 on the finer partition.
    union = np.array([0.0, 10.0, 20.0, 35.0])
    mids = 0.5 * (union[:-1] + union[1:])
    eta = np.clip(X @ spec.beta(mids), -4, 4)
    H = np.cumsum(0.006 * np.exp(eta) * np.diff(union)[None, :], axis=1)
    np.testing.assert_allclose(legacy, np.exp(-H), rtol=1e-12)


def test_oracle_importance_small_budget() -> None:
    from timesage.datasets.synthetic import oracle_grid, oracle_importance

    data = make_synthetic("E1", 101)
    grid = oracle_grid(data)
    assert len(grid) == 24 and grid[-1] == 240.0
    oracle = oracle_importance(data, n_eval=128, n_background=32, n_permutations=16)
    assert oracle.phi.shape == oracle.phi_se.shape == (6, 24)
    np.testing.assert_array_equal(oracle.times, grid)
    assert np.all(oracle.phi_se >= 0)
    np.testing.assert_array_equal(oracle.phi[4:], 0.0)  # x5, x6 have no effect
    assert list(oracle.integrated.index) == list(FEATURE_NAMES)
    assert oracle.stable_window.dtype == bool
    assert oracle.result.n_permutations == 16
