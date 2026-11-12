"""Unit tests for the permutation sampler in ``timesage.estimator`` (Eq. 18-19)."""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

import timesage.estimator as estimator_module
from tests.fixtures.toy_models import toy_predict_survival
from timesage._ipcw import fit_censoring_km
from timesage._value import CoalitionValue
from timesage._vendor.sage import ImportanceTracker
from timesage.estimator import _batch_scores, _sample_permutations

P = 4
IGNORED = 3  # toy model coefficient on feature 3 is 0


def _setup(seed: int = 0, n: int = 60, m: int = 10, K: int = 5):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, P)).astype(np.float32)
    bg = rng.normal(size=(m, P)).astype(np.float32)
    t_ev = 50.0 * rng.weibull(1.3, n) / np.exp(0.5 * X[:, 0])
    t_c = rng.exponential(80.0, n)
    T = np.minimum(t_ev, t_c)
    delta = (t_ev <= t_c).astype(int)
    times = np.linspace(np.percentile(T, 10), np.percentile(T, 80), K)
    value = CoalitionValue(toy_predict_survival, bg, times, fit_censoring_km(T, delta))
    return value, X, T, delta


def _sample(value, X, T, delta, **kwargs):
    defaults = {"n_permutations": 128, "batch_size": 32, "convergence_threshold": None, "seed": 0}
    kwargs = defaults | kwargs
    return _sample_permutations(value, X, T, delta, **kwargs)


def test_ignored_feature_gets_exactly_zero() -> None:
    out = _sample(*_setup())
    np.testing.assert_array_equal(out.values[IGNORED], 0.0)
    np.testing.assert_array_equal(out.std[IGNORED], 0.0)
    assert np.all(np.abs(out.values[:IGNORED]) > 0)


def test_contributions_telescope_to_empty_minus_full_loss() -> None:
    # Within one permutation the marginal contributions sum to loss(empty) - loss(full).
    value, X, T, delta = _setup()
    scores = _batch_scores(value, X, T, delta, np.random.default_rng(1))
    empty = value.loss(X, np.zeros_like(X, dtype=bool), T, delta)
    full = value.loss(X, np.ones_like(X, dtype=bool), T, delta)
    np.testing.assert_allclose(scores.sum(axis=1), empty - full, rtol=0, atol=1e-12)


def _exact_shapley(value: CoalitionValue, X, T, delta) -> np.ndarray:
    """Shapley values of v(S) = -mean_i loss_i(S) by subset enumeration, shape (p, K)."""
    n = len(X)
    v = {}
    for bits in itertools.product([False, True], repeat=P):
        mask = np.tile(np.array(bits), (n, 1))
        v[bits] = -value.loss(X, mask, T, delta).mean(axis=0)
    phi = np.zeros((P, len(value.times)))
    for j in range(P):
        for bits in v:
            if bits[j]:
                continue
            size = sum(bits)
            weight = math.factorial(size) * math.factorial(P - size - 1) / math.factorial(P)
            with_j = tuple(b or k == j for k, b in enumerate(bits))
            phi[j] += weight * (v[with_j] - v[bits])
    return phi


def test_exact_shapley_is_efficient() -> None:
    value, X, T, delta = _setup()
    phi = _exact_shapley(value, X, T, delta)
    empty = value.loss(X, np.zeros_like(X, dtype=bool), T, delta).mean(axis=0)
    full = value.loss(X, np.ones_like(X, dtype=bool), T, delta).mean(axis=0)
    np.testing.assert_allclose(phi.sum(axis=0), empty - full, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(phi[IGNORED], 0.0, atol=1e-15)


def test_estimator_agrees_with_exact_shapley_as_b_grows() -> None:
    # The estimator is unbiased for the exact Shapley values of the dataset-level
    # game (rows are drawn uniformly). Check |error| <= 4 SE everywhere, and that
    # the error shrinks with B.
    value, X, T, delta = _setup()
    exact = _exact_shapley(value, X, T, delta)
    errors = []
    for B in (256, 4096):
        out = _sample(value, X, T, delta, n_permutations=B, batch_size=128, seed=3)
        err = np.abs(out.values - exact)
        se = out.std[:IGNORED]
        assert np.all(err[:IGNORED] <= 4 * se), f"B={B}: max z = {np.max(err[:IGNORED] / se):.2f}"
        errors.append(err.max())
    assert errors[1] < errors[0]


def test_same_seed_reproduces_and_different_seed_differs() -> None:
    args = _setup()
    a, b, c = _sample(*args, seed=5), _sample(*args, seed=5), _sample(*args, seed=6)
    np.testing.assert_array_equal(a.values, b.values)
    np.testing.assert_array_equal(a.std, b.std)
    assert not np.array_equal(a.values, c.values)


def test_exact_budget_with_smaller_final_batch() -> None:
    calls: list[tuple[int, int]] = []
    out = _sample(
        *_setup(), n_permutations=100, batch_size=64, progress=lambda d, t: calls.append((d, t))
    )
    assert out.n_permutations == 100
    assert out.converged is None
    assert calls == [(64, 100), (100, 100)]


def test_early_stopping_records_where_it_stopped() -> None:
    calls: list[tuple[int, int]] = []
    out = _sample(
        *_setup(),
        n_permutations=8192,
        batch_size=32,
        convergence_threshold=0.5,
        progress=lambda d, t: calls.append((d, t)),
    )
    assert out.converged is True
    assert out.n_permutations < 8192
    assert out.n_permutations % 32 == 0
    assert out.convergence_ratio < 0.5
    assert calls[-1] == (out.n_permutations, 8192)
    assert [d for d, _ in calls] == list(range(32, out.n_permutations + 1, 32))


def test_early_stopping_that_never_fires_reports_not_converged() -> None:
    out = _sample(*_setup(), n_permutations=64, batch_size=32, convergence_threshold=1e-6)
    assert out.converged is False
    assert out.n_permutations == 64
    assert out.convergence_ratio >= 1e-6


def test_outputs_do_not_share_memory_with_the_tracker(monkeypatch: pytest.MonkeyPatch) -> None:
    # ImportanceTracker.update mutates its mean in place (see legacy_notes.md §9).
    trackers: list[ImportanceTracker] = []

    class Recording(ImportanceTracker):
        def __init__(self) -> None:
            super().__init__()
            trackers.append(self)

    monkeypatch.setattr(estimator_module, "ImportanceTracker", Recording)
    out = _sample(*_setup())
    (tracker,) = trackers
    for internal in (tracker.mean, tracker.sum_squares, tracker.values):
        assert not np.shares_memory(out.values, internal)
        assert not np.shares_memory(out.std, internal)
    before = out.values.copy()
    tracker.update(np.ones((4, P, out.values.shape[1])))
    np.testing.assert_array_equal(out.values, before)
