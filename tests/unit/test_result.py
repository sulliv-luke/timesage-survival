"""Unit tests for ``timesage.result.TimeSAGEResult``."""

from __future__ import annotations

import numpy as np
import pytest

from timesage.result import TimeSAGEResult


def _kwargs(p: int = 3, K: int = 4) -> dict:
    rng = np.random.default_rng(0)
    return {
        "phi": rng.normal(size=(p, K)),
        "phi_se": rng.random((p, K)),
        "times": np.linspace(1.0, 10.0, K),
        "feature_names": [f"f{j}" for j in range(p)],
        "censoring_survival": np.linspace(1.0, 0.5, K),
        "at_risk_fraction": np.linspace(0.9, 0.2, K),
        "loss_empty": np.full(K, 0.25),
        "loss_full": np.full(K, 0.20),
        "n_permutations": 128,
        "converged": None,
        "convergence_ratio": 0.3,
        "random_state": 42,
    }


def test_arrays_are_read_only_copies() -> None:
    kw = _kwargs()
    res = TimeSAGEResult(**kw)
    for name in ("phi", "phi_se", "times", "censoring_survival", "at_risk_fraction"):
        arr = getattr(res, name)
        assert not arr.flags.writeable
        assert not np.shares_memory(arr, kw[name])
        with pytest.raises(ValueError, match="read-only"):
            arr[...] = 0.0
    kw["phi"][:] = 99.0  # the caller's array no longer matters
    assert not np.any(res.phi == 99.0)
    assert res.feature_names == ("f0", "f1", "f2")
    assert (res.n_features, res.n_times) == (3, 4)


def test_is_frozen() -> None:
    res = TimeSAGEResult(**_kwargs())
    with pytest.raises(AttributeError):
        res.phi = np.zeros((3, 4))  # type: ignore[misc]


def test_efficiency_residual() -> None:
    kw = _kwargs()
    kw["phi"] = np.array([[0.02, 0.01, 0.0, 0.03], [0.03, 0.05, 0.05, 0.0], [0.0] * 4])
    res = TimeSAGEResult(**kw)
    # loss reduction is 0.05 at every time
    np.testing.assert_allclose(res.efficiency_residual(), [0.0, 0.01, 0.0, 0.02], atol=1e-15)


def test_to_frame_is_long_and_ordered_by_feature_then_time() -> None:
    res = TimeSAGEResult(**_kwargs())
    df = res.to_frame()
    assert list(df.columns) == ["feature", "time", "phi", "se"]
    assert len(df) == 12
    assert df["feature"].tolist()[:5] == ["f0"] * 4 + ["f1"]
    np.testing.assert_array_equal(df["phi"].to_numpy().reshape(3, 4), res.phi)
    np.testing.assert_array_equal(df["se"].to_numpy().reshape(3, 4), res.phi_se)
    np.testing.assert_array_equal(df["time"].to_numpy()[:4], res.times)


def test_repr_is_short() -> None:
    assert repr(TimeSAGEResult(**_kwargs())) == (
        "TimeSAGEResult(p=3, K=4, times=[1, 10], n_permutations=128, random_state=42)"
    )
    kw = _kwargs() | {"converged": True}
    assert "converged=True" in repr(TimeSAGEResult(**kw))


@pytest.mark.parametrize(
    ("name", "bad", "match"),
    [
        ("phi", np.zeros(4), "phi must be 2-D"),
        ("phi_se", np.zeros((3, 5)), "phi_se must have shape"),
        ("times", np.zeros(5), "times must have shape"),
        ("feature_names", ["a", "b"], "feature_names has 2"),
        ("loss_full", np.zeros(3), "loss_full must have shape"),
    ],
)
def test_rejects_inconsistent_shapes(name: str, bad: object, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        TimeSAGEResult(**(_kwargs() | {name: bad}))
