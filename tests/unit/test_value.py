"""Unit tests for ``timesage._value`` (marginal imputation, Eq. 13; loss, Eq. 14)."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from tests.fixtures.toy_models import toy_predict_survival
from timesage._ipcw import fit_censoring_km, ipcw_brier_loss
from timesage._value import CoalitionValue

P = 4
IGNORED = 3  # toy model coefficient on feature 3 is 0
TIMES = np.array([5.0, 20.0, 40.0, 80.0])


def _setup(seed: int = 0, n: int = 30, m: int = 12):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, P))
    bg = rng.normal(size=(m, P))
    T = rng.exponential(40.0, n)
    delta = rng.integers(0, 2, n)
    value = CoalitionValue(toy_predict_survival, bg, TIMES, fit_censoring_km(T, delta))
    return value, X, bg, T, delta


def _all_masks(n: int) -> list[np.ndarray]:
    return [
        np.tile(np.array(bits, dtype=bool), (n, 1)) for bits in itertools.product([0, 1], repeat=P)
    ]


def test_ignored_feature_never_changes_predictions() -> None:
    # The property that gives unused features exactly zero importance.
    value, X, *_ = _setup()
    for mask in _all_masks(len(X)):
        with_j, without_j = mask.copy(), mask.copy()
        with_j[:, IGNORED], without_j[:, IGNORED] = True, False
        np.testing.assert_array_equal(value.predict(X, with_j), value.predict(X, without_j))


def test_full_coalition_reproduces_the_model() -> None:
    value, X, *_ = _setup()
    full = np.ones_like(X, dtype=bool)
    # Averaging m identical predictions can differ from one by rounding only.
    np.testing.assert_allclose(
        value.predict(X, full), toy_predict_survival(X, TIMES), rtol=1e-14, atol=0
    )


def test_empty_coalition_does_not_depend_on_x() -> None:
    value, X, bg, *_ = _setup()
    pred = value.predict(X, np.zeros_like(X, dtype=bool))
    expected = toy_predict_survival(bg, TIMES).mean(axis=0)
    np.testing.assert_allclose(pred, np.tile(expected, (len(X), 1)), rtol=1e-14, atol=0)


def test_replacement_values_are_independent_of_retained_features() -> None:
    # Eq. 13: S_hat_S(t | x_S) = mean over background rows b of S_hat(t | x_S, b_{M\S}).
    value, X, bg, *_ = _setup()
    mask = np.zeros_like(X, dtype=bool)
    mask[:, [0, 2]] = True
    pred = value.predict(X, mask)
    for i in (0, 7, 19):
        rows = bg.copy()
        rows[:, [0, 2]] = X[i, [0, 2]]
        expected = toy_predict_survival(rows, TIMES).mean(axis=0)
        np.testing.assert_allclose(pred[i], expected, rtol=1e-14, atol=0)


def test_rows_may_have_different_coalitions() -> None:
    value, X, *_ = _setup()
    rng = np.random.default_rng(1)
    mask = rng.random(X.shape) < 0.5
    pred = value.predict(X, mask)
    for i in range(len(X)):
        # Batch size may change BLAS summation order, hence a 1e-14 tolerance.
        np.testing.assert_allclose(pred[i], value.predict(X[[i]], mask[[i]])[0], rtol=1e-14, atol=0)


def test_loss_is_ipcw_brier_loss_of_imputed_predictions() -> None:
    value, X, _, T, delta = _setup()
    mask = np.random.default_rng(2).random(X.shape) < 0.5
    expected = ipcw_brier_loss(value.predict(X, mask), T, delta, TIMES, fit_censoring_km(T, delta))
    np.testing.assert_array_equal(value.loss(X, mask, T, delta), expected)


def test_inputs_are_not_mutated_and_background_is_copied() -> None:
    value, X, bg, T, delta = _setup()
    mask = np.random.default_rng(3).random(X.shape) < 0.5
    copies = [a.copy() for a in (X, mask, T, delta)]
    before = value.predict(X, mask)
    value.loss(X, mask, T, delta)
    for original, copy in zip((X, mask, T, delta), copies, strict=True):
        np.testing.assert_array_equal(original, copy)
    bg[:] = 1e6  # changing the caller's array must not affect the value function
    np.testing.assert_array_equal(value.predict(X, mask), before)


def test_properties() -> None:
    value, *_ = _setup(m=12)
    assert value.num_features == P
    assert value.n_background == 12
    np.testing.assert_array_equal(value.times, TIMES)


# ---------------------------------------------------------------- validation


def test_rejects_non_boolean_mask() -> None:
    value, X, *_ = _setup()
    with pytest.raises(TypeError, match="boolean"):
        value.predict(X, np.ones_like(X, dtype=int))


def test_rejects_mask_shape_mismatch() -> None:
    value, X, *_ = _setup()
    with pytest.raises(ValueError, match="mask"):
        value.predict(X, np.ones((len(X), P - 1), dtype=bool))


def test_rejects_feature_count_mismatch() -> None:
    value, X, *_ = _setup()
    with pytest.raises(ValueError, match="4 features"):
        value.predict(X[:, :3], np.ones((len(X), 3), dtype=bool))


def test_rejects_model_with_wrong_output_shape() -> None:
    _, X, bg, T, delta = _setup()
    value = CoalitionValue(
        lambda X_in, t: np.ones((len(X_in), len(t) + 1)), bg, TIMES, fit_censoring_km(T, delta)
    )
    with pytest.raises(ValueError, match="predict_survival returned shape"):
        value.predict(X, np.ones_like(X, dtype=bool))


def test_rejects_non_finite_model_output_on_every_call() -> None:
    _, X, bg, T, delta = _setup()

    def model(X_in: np.ndarray, t: np.ndarray) -> np.ndarray:
        S = toy_predict_survival(X_in, t)
        S[X_in[:, 0] > 1.0] = np.nan  # fails only for some feature values
        return S

    value = CoalitionValue(model, bg, TIMES, fit_censoring_km(T, delta))
    rows = X[X[:, 0] <= 1.0]
    keep = np.ones_like(rows, dtype=bool)
    assert np.all(np.isfinite(value.predict(rows, keep)))
    rows[0, 0] = 2.0
    with pytest.raises(ValueError, match="non-finite values on imputed rows"):
        value.predict(rows, keep)


@pytest.mark.parametrize(
    ("background", "times", "match"),
    [
        (np.zeros(4), TIMES, "background"),
        (np.zeros((0, 4)), TIMES, "background"),
        (np.zeros((3, 4)), np.array([[1.0]]), "times"),
        (np.zeros((3, 4)), np.array([]), "times"),
    ],
)
def test_rejects_invalid_construction(
    background: np.ndarray, times: np.ndarray, match: str
) -> None:
    km = fit_censoring_km([1.0, 2.0], [1, 0])
    with pytest.raises(ValueError, match=match):
        CoalitionValue(toy_predict_survival, background, times, km)
