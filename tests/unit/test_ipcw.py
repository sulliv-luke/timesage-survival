"""Unit tests for ``timesage._ipcw`` (Eq. 14-16)."""

from __future__ import annotations

import numpy as np
import pytest

from timesage._ipcw import CensoringKM, fit_censoring_km, ipcw_brier_loss, ipcw_weights

# Hand-built data. Unique times 1, 2, 3, 4; at t = 2 one event and one
# censoring are tied; subject 4 is censored at 3.
T_HAND = np.array([1.0, 2.0, 2.0, 3.0, 4.0])
D_HAND = np.array([1, 0, 1, 0, 1])
# Censoring KM by hand (risk set n_u counts everyone with T >= u):
#   u=1: 1 - 0/5 = 1;  u=2: 1 - 1/4 = 0.75;  u=3: 0.75 * (1 - 1/2) = 0.375;
#   u=4: 0.375 * (1 - 0/1) = 0.375.
G_HAND = np.array([1.0, 0.75, 0.375, 0.375])


def _random_data(seed: int, n: int = 300) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    t_ev = np.ceil(rng.exponential(10.0, n))  # integer times: many ties
    t_c = np.ceil(rng.exponential(12.0, n))
    return np.minimum(t_ev, t_c), (t_ev < t_c).astype(int)


# ---------------------------------------------------------------- KM (Eq. 16)


def test_km_by_hand() -> None:
    km = fit_censoring_km(T_HAND, D_HAND)
    np.testing.assert_array_equal(km.times, [1.0, 2.0, 3.0, 4.0])
    np.testing.assert_allclose(km.survival, G_HAND, rtol=0, atol=1e-15)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_km_matches_lifelines_on_censoring_indicator(seed: int) -> None:
    lifelines = pytest.importorskip("lifelines")
    T, delta = _random_data(seed)
    kmf = lifelines.KaplanMeierFitter().fit(T, event_observed=1 - delta)
    sf = kmf.survival_function_.iloc[:, 0]
    km = fit_censoring_km(T, delta)
    expected = sf.loc[km.times].to_numpy()
    np.testing.assert_allclose(km.survival, expected, rtol=1e-10, atol=1e-12)


def test_km_is_right_continuous_step_function() -> None:
    km = fit_censoring_km(T_HAND, D_HAND)
    # Before the first time: 1. At a jump: the post-jump value. Between jumps:
    # the previous value. After the last time: the last value.
    t = np.array([0.0, 0.999, 1.0, 1.999, 2.0, 2.5, 3.0, 100.0])
    np.testing.assert_allclose(km(t), [1, 1, 1, 1, 0.75, 0.75, 0.375, 0.375])


def test_km_scalar_evaluation_returns_0d_array() -> None:
    value = fit_censoring_km(T_HAND, D_HAND)(2.0)
    assert value.shape == ()
    assert float(value) == 0.75


def test_km_reaching_zero_is_clipped_to_floor() -> None:
    # Largest time censored: G_hat drops to exactly 0. Behaviour kept from the original:
    # clip to 1e-12 on evaluation, no error.
    km = fit_censoring_km([1.0, 2.0, 3.0], [1, 0, 0])
    assert km.survival[-1] == 0.0
    assert float(km(3.0)) == 1e-12


def test_km_without_censoring_is_one() -> None:
    km = fit_censoring_km([3.0, 1.0, 2.0], [1, 1, 1])
    np.testing.assert_array_equal(km.survival, 1.0)


# ---------------------------------------------------------------- weights (Eq. 15)


def test_ipcw_weights_by_hand() -> None:
    km = fit_censoring_km(T_HAND, D_HAND)
    times = np.array([0.5, 2.0, 3.0, 4.0])
    w = ipcw_weights(T_HAND, D_HAND, times, km)
    expected = np.array(
        [
            # t=0.5: everyone still at risk -> 1 / G(0.5) = 1.
            [1.0, 1.0, 1.0, 1.0, 1.0],
            # t=2: event at 1 -> 1/G(1); censored at t -> 0; event at t tied with
            # a censoring -> 1/G(2) = 4/3 (right-continuous; the left limit
            # G(2-) would give 1); at risk -> 1/G(2).
            [1.0, 0.0, 4 / 3, 4 / 3, 4 / 3],
            # t=3: subject 4 censored exactly at t -> 0.
            [1.0, 0.0, 4 / 3, 0.0, 8 / 3],
            # t=4: event exactly at t -> 1/G(4).
            [1.0, 0.0, 4 / 3, 0.0, 8 / 3],
        ]
    ).T
    np.testing.assert_allclose(w, expected, rtol=0, atol=1e-15)


def test_ipcw_weights_zero_exactly_for_subjects_censored_by_t() -> None:
    T, delta = _random_data(3)
    times = np.array([2.0, 5.0, 10.0, 20.0])
    w = ipcw_weights(T, delta, times, fit_censoring_km(T, delta))
    censored_by_t = (T[:, None] <= times[None, :]) & (delta[:, None] == 0)
    assert np.all(w[censored_by_t] == 0.0)
    assert np.all(w[~censored_by_t] >= 1.0)


# ---------------------------------------------------------------- loss (Eq. 14)


def test_loss_equals_plain_brier_without_censoring() -> None:
    rng = np.random.default_rng(4)
    T = rng.exponential(10.0, 50)
    delta = np.ones(50, dtype=int)
    times = np.array([1.0, 5.0, 10.0, 30.0])
    pred = -np.sort(-rng.uniform(size=(50, 4)), axis=1)
    loss = ipcw_brier_loss(pred, T, delta, times, fit_censoring_km(T, delta))
    plain = ((T[:, None] > times[None, :]).astype(float) - pred) ** 2
    np.testing.assert_array_equal(loss, plain)


def test_loss_is_weight_times_squared_error() -> None:
    T, delta = _random_data(5, n=40)
    times = np.array([1.0, 4.0, 9.0])
    pred = np.full((40, 3), 0.6)
    km = fit_censoring_km(T, delta)
    expected = (
        ipcw_weights(T, delta, times, km) * ((T[:, None] > times[None, :]).astype(float) - 0.6) ** 2
    )
    np.testing.assert_array_equal(ipcw_brier_loss(pred, T, delta, times, km), expected)


def test_inputs_are_not_mutated() -> None:
    T, delta = _random_data(6, n=30)
    times = np.array([2.0, 6.0])
    pred = np.full((30, 2), 0.5)
    copies = [a.copy() for a in (T, delta, times, pred)]
    km = fit_censoring_km(T, delta)
    ipcw_brier_loss(pred, T, delta, times, km)
    for original, copy in zip((T, delta, times, pred), copies, strict=True):
        np.testing.assert_array_equal(original, copy)


def test_accepts_boolean_and_float_event_indicators() -> None:
    km_int = fit_censoring_km(T_HAND, D_HAND)
    for event in (D_HAND.astype(bool), D_HAND.astype(float)):
        km = fit_censoring_km(T_HAND, event)
        np.testing.assert_array_equal(km.survival, km_int.survival)
        np.testing.assert_array_equal(
            ipcw_weights(T_HAND, event, [2.0], km), ipcw_weights(T_HAND, D_HAND, [2.0], km_int)
        )


# ---------------------------------------------------------------- validation


@pytest.mark.parametrize(
    ("time", "event", "match"),
    [
        ([1.0, 2.0], [1, 0, 1], "same length"),
        ([[1.0, 2.0]], [[1, 0]], "1-D"),
        ([], [], "at least one"),
        ([1.0, np.nan], [1, 0], "finite"),
        ([1.0, 2.0], [1, 2], r"\{0, 1\}"),
        ([1.0, 2.0], [1.0, 0.5], r"\{0, 1\}"),
    ],
)
def test_fit_rejects_invalid_survival_data(time: list, event: list, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        fit_censoring_km(time, event)


def test_loss_rejects_prediction_shape_mismatch() -> None:
    km = fit_censoring_km(T_HAND, D_HAND)
    with pytest.raises(ValueError, match=r"pred_surv must have shape \(5, 2\)"):
        ipcw_brier_loss(np.zeros((5, 3)), T_HAND, D_HAND, [1.0, 2.0], km)


def test_loss_rejects_censoring_at_times_shape_mismatch() -> None:
    km = fit_censoring_km(T_HAND, D_HAND)
    with pytest.raises(ValueError, match="censoring_at_times"):
        ipcw_brier_loss(np.zeros((5, 2)), T_HAND, D_HAND, [1.0, 2.0], km, [1.0])


def test_censoring_km_is_immutable() -> None:
    km = CensoringKM(np.array([1.0]), np.array([1.0]))
    with pytest.raises(AttributeError):
        km.times = np.array([2.0])  # type: ignore[misc]
