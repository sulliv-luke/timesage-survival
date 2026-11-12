"""Unit tests for ``timesage.metrics`` on hand-built curves."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from tests.fixtures.toy_models import toy_predict_survival
from timesage import TimeSAGE
from timesage._ipcw import fit_censoring_km
from timesage.metrics import (
    curve_correlation,
    high_importance_iou,
    nwiae,
    peak_time_error,
    rank_correlation,
    stable_window,
)

TIMES = np.array([1.0, 2.0, 4.0, 7.0, 11.0, 16.0])
HUMP = np.array([0.0, 0.2, 1.0, 0.6, 0.1, 0.0])


# ---------------------------------------------------------------- stable window


def test_stable_window_thresholds() -> None:
    # 10 subjects at times 1..10; all censored at 10 except the last event.
    T = np.arange(1.0, 11.0)
    E = np.ones(10, dtype=int)
    grid = np.array([0.5, 5.0, 9.5, 10.0])
    # R_hat(>=) = 1.0, 0.6, 0.1, 0.1 -> all >= 0.10; G_hat = 1 (no censoring).
    np.testing.assert_array_equal(stable_window(grid, T, E), [True, True, True, True])
    # Original 1{T > t}: R_hat(10) = 0 -> outside.
    np.testing.assert_array_equal(
        stable_window(grid, T, E, legacy_parity=True), [True, True, True, False]
    )
    np.testing.assert_array_equal(
        stable_window(grid, T, E, min_at_risk=0.5), [True, True, False, False]
    )


def test_stable_window_uses_the_censoring_curve() -> None:
    T = np.arange(1.0, 21.0)
    E = np.ones(20, dtype=int)
    grid = np.array([2.0, 10.0, 15.0])
    # Censoring data in which nearly everyone is censored by t = 12.
    T_c = np.concatenate([np.linspace(0.5, 12.0, 40), [30.0]])
    E_c = np.concatenate([np.zeros(40, dtype=int), [1]])
    G = fit_censoring_km(T_c, E_c)(grid)
    expected = G >= 0.05
    assert expected.any() and not expected.all()
    mask = stable_window(grid, T, E, censoring_time=T_c, censoring_event=E_c)
    np.testing.assert_array_equal(mask, expected)


def test_stable_window_empty_warns() -> None:
    T = np.arange(1.0, 11.0)
    with pytest.warns(UserWarning, match="stable window is empty"):
        mask = stable_window(np.array([9.5, 10.0]), T, np.ones(10, dtype=int), min_at_risk=0.5)
    assert not mask.any()


def test_stable_window_validation() -> None:
    T, E = np.arange(1.0, 11.0), np.ones(10, dtype=int)
    with pytest.raises(ValueError, match="together"):
        stable_window(TIMES, T, E, censoring_time=T)
    with pytest.raises(ValueError, match="strictly increasing"):
        stable_window(np.array([2.0, 1.0]), T, E)
    with pytest.raises(ValueError):
        stable_window(TIMES, T, np.full(10, 2))


def test_result_stable_window_matches_the_function_form() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 4))
    T = rng.exponential(40.0, 120)
    E = rng.integers(0, 2, 120)
    grid = np.linspace(np.percentile(T, 5), np.percentile(T, 95), 12)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # G_hat < 0.05 near the end is expected here
        res = TimeSAGE(
            toy_predict_survival, background=X[:10], n_permutations=8, random_state=0
        ).explain(X, T, E, grid)
        for lp in (False, True):
            np.testing.assert_array_equal(
                res.stable_window(legacy_parity=lp), stable_window(grid, T, E, legacy_parity=lp)
            )


# ---------------------------------------------------------------- curve comparisons


def test_curve_correlation() -> None:
    assert curve_correlation(HUMP, 3 * HUMP + 1) == pytest.approx(1.0)
    assert curve_correlation(HUMP, -HUMP) == pytest.approx(-1.0)
    window = np.array([True, True, True, False, False, False])
    assert curve_correlation(HUMP, HUMP**2, window=window) == pytest.approx(
        np.corrcoef(HUMP[:3], HUMP[:3] ** 2)[0, 1]
    )


@pytest.mark.parametrize(
    ("estimate", "window", "reason"),
    [
        (np.full(6, 0.3), None, "constant"),
        (HUMP, np.array([False, False, True, False, False, False]), "fewer than two"),
    ],
)
def test_curve_correlation_undefined_warns(estimate, window, reason) -> None:
    with pytest.warns(UserWarning, match=f"curve_correlation is undefined: .*{reason}"):
        assert np.isnan(curve_correlation(estimate, HUMP, window=window))


def test_nwiae_known_values() -> None:
    assert nwiae(HUMP, HUMP, TIMES) == pytest.approx(0.0, abs=1e-12)
    assert nwiae(2 * HUMP, HUMP, TIMES) == pytest.approx(1.0)
    assert nwiae(np.zeros(6), HUMP, TIMES) == pytest.approx(1.0)
    # Weights matter: put all weight on the peak, where the error is zero.
    est = HUMP.copy()
    est[0] = 0.5
    only_peak = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    assert nwiae(est, HUMP, TIMES, weights=only_peak) == pytest.approx(0.0, abs=1e-12)
    assert nwiae(est, HUMP, TIMES) > 0


def test_nwiae_weights_and_window_agree() -> None:
    est = HUMP + np.array([0.05, -0.02, 0.1, 0.0, 0.03, 0.01])
    window = np.array([False, True, True, True, False, False])
    assert nwiae(est, HUMP, TIMES, window=window) == pytest.approx(
        nwiae(est, HUMP, TIMES, weights=window.astype(float))
    )


def test_nwiae_undefined_warns() -> None:
    with pytest.warns(UserWarning, match="no weighted mass"):
        assert np.isnan(nwiae(HUMP, np.zeros(6), TIMES))
    with pytest.warns(UserWarning, match="all weights are zero"):
        assert np.isnan(nwiae(HUMP, HUMP, TIMES, window=np.zeros(6, dtype=bool)))


def test_high_importance_iou() -> None:
    assert high_importance_iou(HUMP, HUMP) == 1.0
    shifted = np.roll(HUMP, 2)  # regions {t_3} and {t_5}: disjoint
    assert high_importance_iou(HUMP, shifted) == 0.0
    wide = np.array([0.0, 0.85, 1.0, 0.9, 0.1, 0.0])  # region {t_2, t_3, t_4}
    assert high_importance_iou(HUMP, wide) == pytest.approx(1 / 3)
    assert high_importance_iou(HUMP, wide, threshold=0.5) == pytest.approx(2 / 3)


def test_high_importance_iou_peak_is_taken_over_the_whole_curve() -> None:
    # Section 7.7.3: a late spike outside the window raises the estimate's
    # cut-off, so its region inside the window empties.
    est = HUMP.copy()
    est[-1] = 2.0
    window = np.array([True, True, True, True, False, False])
    with pytest.warns(UserWarning, match="undefined"):
        assert np.isnan(high_importance_iou(est, est, window=window))
    assert high_importance_iou(est, HUMP, window=window) == 0.0  # reference still has {t_3}


def test_peak_time_error() -> None:
    assert peak_time_error(HUMP, HUMP, TIMES) == 0.0
    assert peak_time_error(np.roll(HUMP, 1), HUMP, TIMES) == pytest.approx(7.0 - 4.0)
    window = np.array([True, True, False, True, True, True])
    # Without t_3 = 4, both curves peak at t_4 = 7.
    assert peak_time_error(HUMP, np.roll(HUMP, 1), TIMES, window=window) == 0.0
    with pytest.warns(UserWarning, match="window is empty"):
        assert np.isnan(peak_time_error(HUMP, HUMP, TIMES, window=np.zeros(6, dtype=bool)))


# ---------------------------------------------------------------- rankings


def test_rank_correlation() -> None:
    a = np.array([3.0, 1.0, 2.0, 0.5])
    assert rank_correlation(a, a**3) == pytest.approx(1.0)
    assert rank_correlation(a, -a) == pytest.approx(-1.0)
    assert rank_correlation(a, a**3, "pearson") < 1.0
    with pytest.warns(UserWarning, match="constant"):
        assert np.isnan(rank_correlation(a, np.ones(4)))
    with pytest.raises(ValueError, match="method"):
        rank_correlation(a, a, "kendall")  # type: ignore[arg-type]


# ---------------------------------------------------------------- validation


@pytest.mark.parametrize(
    ("call", "error", "match"),
    [
        (lambda: curve_correlation(HUMP, HUMP[:5]), ValueError, "expected 6"),
        (lambda: curve_correlation(HUMP, np.full(6, np.nan)), ValueError, "finite"),
        (lambda: curve_correlation(HUMP, HUMP, window=np.ones(6)), TypeError, "boolean"),
        (lambda: curve_correlation(HUMP, HUMP, window=np.ones(5, bool)), ValueError, "shape"),
        (lambda: nwiae(HUMP, HUMP, TIMES[:5]), ValueError, "times has 5"),
        (lambda: nwiae(HUMP, HUMP, TIMES, weights=-np.ones(6)), ValueError, "non-negative"),
        (lambda: high_importance_iou(HUMP, HUMP, threshold=0.0), ValueError, "threshold"),
        (lambda: peak_time_error(HUMP, HUMP, TIMES[::-1]), ValueError, "increasing"),
    ],
)
def test_validation(call, error, match) -> None:
    with pytest.raises(error, match=match):
        call()
