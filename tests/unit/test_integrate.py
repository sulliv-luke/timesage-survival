"""Unit tests for ``timesage.integrate`` on known curves (Eq. 20-24)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from timesage import TimeSAGEResult, integrate
from timesage.integrate import _trapezoid_weights

TIMES = np.array([1.0, 2.0, 4.0, 7.0, 11.0])  # uneven on purpose


def _result(phi: np.ndarray, times: np.ndarray = TIMES, **kwargs: object) -> TimeSAGEResult:
    p, K = phi.shape
    fields: dict = {
        "phi": phi,
        "phi_se": np.zeros((p, K)),
        "times": times,
        "feature_names": [f"f{j}" for j in range(p)],
        "censoring_survival": np.linspace(1.0, 0.5, K),
        "at_risk_fraction": np.linspace(0.9, 0.2, K),
        "loss_empty": np.zeros(K),
        "loss_full": np.zeros(K),
        "n_permutations": 1,
        "converged": None,
        "convergence_ratio": 0.0,
        "random_state": 0,
    }
    return TimeSAGEResult(**(fields | kwargs))


def test_trapezoid_weights_reproduce_numpy_trapezoid() -> None:
    f = np.random.default_rng(0).normal(size=len(TIMES))
    assert _trapezoid_weights(TIMES) @ f == pytest.approx(np.trapezoid(f, x=TIMES), rel=1e-14)
    np.testing.assert_array_equal(_trapezoid_weights(np.array([0.0, 3.0])), [1.5, 1.5])


def test_trapezoid_is_exact_for_linear_curves() -> None:
    # Constant 1 -> t_K - t_1 (the integral starts at t_1, not 0); phi = t -> (t_K^2 - t_1^2) / 2.
    phi = np.vstack([np.ones_like(TIMES), TIMES, -2 * TIMES])
    out = integrate(_result(phi), "trapezoid")
    np.testing.assert_allclose(out.to_numpy(), [10.0, 60.0, -120.0], rtol=1e-14)


def test_weighted_variants_scale_the_trapezoid_weights_pointwise() -> None:
    # Eq. 20-24: sum_k w(t_k) phi(t_k) dt_k, not a trapezoid of w * phi.
    rng = np.random.default_rng(1)
    phi = rng.normal(size=(3, len(TIMES)))
    G = np.array([1.0, 0.9, 0.7, 0.4, 0.2])
    R = np.array([0.95, 0.8, 0.6, 0.3, 0.1])
    res = _result(phi, censoring_survival=G, at_risk_fraction=R)
    dt = _trapezoid_weights(TIMES)
    np.testing.assert_allclose(integrate(res, "censoring").to_numpy(), phi @ (G * dt), rtol=1e-14)
    np.testing.assert_allclose(integrate(res, "at_risk").to_numpy(), phi @ (R * dt), rtol=1e-14)


def test_constant_weight_is_a_rescaled_trapezoid() -> None:
    phi = np.random.default_rng(2).normal(size=(2, len(TIMES)))
    res = _result(phi, censoring_survival=np.full(len(TIMES), 0.6))
    np.testing.assert_allclose(
        integrate(res, "censoring").to_numpy(), 0.6 * integrate(res, "trapezoid").to_numpy()
    )


def test_ignored_feature_integrates_to_zero_and_signs_are_kept() -> None:
    phi = np.vstack([np.zeros_like(TIMES), -np.ones_like(TIMES)])
    for method in ("trapezoid", "at_risk", "censoring"):
        out = integrate(_result(phi), method)  # type: ignore[arg-type]
        assert out.iloc[0] == 0.0
        assert out.iloc[1] < 0


def test_default_is_censoring_and_output_is_labelled() -> None:
    phi = np.random.default_rng(3).normal(size=(3, len(TIMES)))
    res = _result(phi)
    out = res.integrated()
    assert isinstance(out, pd.Series)
    assert out.equals(integrate(res, "censoring"))
    assert out.name == "integrated_censoring"
    assert list(out.index) == ["f0", "f1", "f2"]
    assert out.index.name == "feature"


def test_legacy_parity_only_changes_at_risk() -> None:
    phi = np.random.default_rng(4).normal(size=(2, len(TIMES)))
    legacy = np.array([0.85, 0.7, 0.5, 0.25, 0.05])
    res = _result(phi, _at_risk_fraction_legacy=legacy)
    dt = _trapezoid_weights(TIMES)
    np.testing.assert_allclose(
        res.integrated("at_risk", legacy_parity=True).to_numpy(), phi @ (legacy * dt), rtol=1e-14
    )
    for method in ("trapezoid", "censoring"):
        assert res.integrated(method, legacy_parity=True).equals(res.integrated(method))  # type: ignore[arg-type]


def test_legacy_parity_needs_the_legacy_fraction() -> None:
    res = _result(np.ones((1, len(TIMES))))
    with pytest.raises(ValueError, match="legacy at-risk fraction"):
        res.integrated("at_risk", legacy_parity=True)


def test_legacy_fraction_is_a_read_only_copy() -> None:
    legacy = np.linspace(0.9, 0.1, len(TIMES))
    res = _result(np.ones((1, len(TIMES))), _at_risk_fraction_legacy=legacy)
    stored = res._at_risk_fraction_legacy
    assert stored is not None
    assert not stored.flags.writeable
    assert not np.shares_memory(stored, legacy)
    with pytest.raises(ValueError, match="_at_risk_fraction_legacy"):
        _result(np.ones((1, len(TIMES))), _at_risk_fraction_legacy=np.ones(2))


def test_validation() -> None:
    res = _result(np.ones((1, len(TIMES))))
    with pytest.raises(ValueError, match="Unknown method 'auc'"):
        integrate(res, "auc")  # type: ignore[arg-type]
    single = _result(np.ones((1, 1)), times=np.array([5.0]))
    with pytest.raises(ValueError, match="at least two times"):
        single.integrated()


# ---------------------------------------------------------------- window


def test_window_none_is_unchanged_and_all_true_equals_none() -> None:
    phi = np.random.default_rng(5).normal(size=(3, len(TIMES)))
    res = _result(phi, _at_risk_fraction_legacy=np.linspace(0.85, 0.15, len(TIMES)))
    everywhere = np.ones(len(TIMES), dtype=bool)
    dt = _trapezoid_weights(TIMES)
    for method in ("trapezoid", "at_risk", "censoring"):
        for lp in (False, True):
            default = res.integrated(method, legacy_parity=lp)  # type: ignore[arg-type]
            # Bit-identical to the expression used before window= existed.
            weight = {
                "trapezoid": dt,
                "at_risk": (res._at_risk_fraction_legacy if lp else res.at_risk_fraction) * dt,  # type: ignore[operator]
                "censoring": res.censoring_survival * dt,
            }[method]
            np.testing.assert_array_equal(default.to_numpy(), np.sum(phi * weight[None, :], axis=1))
            full = res.integrated(method, window=everywhere, legacy_parity=lp)  # type: ignore[arg-type]
            assert full.equals(default)


def test_window_zeroes_full_grid_weights_outside() -> None:
    # As integrate_curve: the window's edge points keep their full-grid
    # half-intervals, which reach outside the window.
    phi = np.ones((1, len(TIMES)))  # TIMES = 1, 2, 4, 7, 11
    window = np.array([False, True, True, True, False])
    out = integrate(_result(phi), "trapezoid", window=window).iloc[0]
    # dt at t=2, 4, 7: (4-1)/2 + (7-2)/2 + (11-4)/2 = 7.5, not the sub-grid area 7 - 2 = 5.
    assert out == pytest.approx(7.5)
    G = np.array([1.0, 0.9, 0.7, 0.4, 0.2])
    res = _result(phi, censoring_survival=G)
    expected = float(np.sum(G * _trapezoid_weights(TIMES) * window))
    assert res.integrated(window=window).iloc[0] == pytest.approx(expected)


@pytest.mark.parametrize(
    ("window", "error", "match"),
    [
        (np.ones(len(TIMES)), TypeError, "boolean"),
        (np.ones(len(TIMES) - 1, dtype=bool), ValueError, "shape"),
        (np.array([False, False, True, False, False]), ValueError, "at least two"),
    ],
)
def test_window_validation(window: np.ndarray, error: type[Exception], match: str) -> None:
    with pytest.raises(error, match=match):
        integrate(_result(np.ones((1, len(TIMES)))), window=window)
