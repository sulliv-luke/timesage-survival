"""Unit tests for ``timesage.plotting`` (skipped without matplotlib)."""

from __future__ import annotations

import builtins
from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest

from timesage import integrate
from timesage.result import TimeSAGEResult

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402

from timesage import plotting  # noqa: E402


@pytest.fixture(autouse=True)
def _close_figures() -> Iterator[None]:
    yield
    plt.close("all")


def _result(p: int = 4, K: int = 6, **overrides: Any) -> TimeSAGEResult:
    rng = np.random.default_rng(0)
    # Feature f{j} has constant importance p - j, except f1 which is negative.
    phi = np.repeat(np.arange(p, 0, -1, dtype=float)[:, None], K, axis=1)
    phi[1] = -0.5
    kw: dict[str, Any] = {
        "phi": phi,
        "phi_se": 0.01 * rng.random((p, K)),
        "times": np.linspace(1.0, 11.0, K),
        "feature_names": [f"f{j}" for j in range(p)],
        "censoring_survival": np.linspace(1.0, 0.3, K),
        # at-risk falls below 0.10 at the last two times: stable window = first K-2
        "at_risk_fraction": np.array([0.9, 0.7, 0.5, 0.3, 0.08, 0.05])[:K],
        "loss_empty": np.full(K, 0.25),
        "loss_full": np.full(K, 0.20),
        "n_permutations": 128,
        "converged": None,
        "convergence_ratio": 0.3,
        "random_state": 0,
    }
    kw.update(overrides)
    return TimeSAGEResult(**kw)


def _spans(ax: Any) -> list[Any]:
    return [c for c in ax.patches if c.get_gid() == "timesage-window"]


def _x_extent(span: Any) -> tuple[float, float]:
    # axvspan returns a Polygon before matplotlib 3.9 and a Rectangle after.
    x = np.asarray(span.get_path().transformed(span.get_patch_transform()).vertices)[:, 0]
    if hasattr(span, "get_x"):
        x = np.array([span.get_x(), span.get_x() + span.get_width()])
    return float(x.min()), float(x.max())


# ------------------------------------------------------------------ curves


def test_curves_default_plots_all_small_p_by_integrated_rank() -> None:
    res = _result()
    ax = plotting.plot_importance_curves(res)
    assert [ln.get_label() for ln in ax.get_lines() if not ln.get_label().startswith("_")] == [
        "f0",
        "f2",
        "f3",
        "f1",
    ]
    np.testing.assert_array_equal(ax.get_lines()[1].get_ydata(), res.phi[0])  # [0] is y = 0


def test_curves_negative_values_are_not_clipped() -> None:
    res = _result()
    ax = plotting.plot_importance_curves(res, features=["f1"])
    (curve,) = [ln for ln in ax.get_lines() if ln.get_label() == "f1"]
    assert np.all(np.asarray(curve.get_ydata()) == -0.5)


def test_curves_explicit_features_and_top() -> None:
    res = _result()
    ax = plotting.plot_importance_curves(res, features=["f3", "f0"], band=False)
    assert [ln.get_label() for ln in ax.get_lines()][1:] == ["f3", "f0"]
    ax = plotting.plot_importance_curves(res, top=2)
    assert [ln.get_label() for ln in ax.get_lines()][1:] == ["f0", "f2"]


def test_curves_band_toggle() -> None:
    res = _result()
    with_band = plotting.plot_importance_curves(res, window=None)
    assert sum(isinstance(c, PolyCollection) for c in with_band.collections) == 4
    without = plotting.plot_importance_curves(res, window=None, band=False)
    assert not any(isinstance(c, PolyCollection) for c in without.collections)


def test_curves_stable_window_shaded_by_default() -> None:
    res = _result()
    assert res.stable_window().tolist() == [True] * 4 + [False] * 2
    ax = plotting.plot_importance_curves(res)
    (span,) = _spans(ax)
    t = res.times
    assert _x_extent(span) == pytest.approx((t[0], (t[3] + t[4]) / 2))
    assert "stable window" in [h.get_label() for h in ax.get_legend().legend_handles]


def test_curves_window_none_and_custom_mask_runs() -> None:
    res = _result()
    assert _spans(plotting.plot_importance_curves(res, window=None)) == []
    mask = np.array([True, False, True, True, False, True])
    spans = _spans(plotting.plot_importance_curves(res, window=mask))
    assert len(spans) == 3


def test_curves_uses_given_axes() -> None:
    _, ax = plt.subplots()
    assert plotting.plot_importance_curves(_result(), ax=ax) is ax


def test_curves_more_than_eight_rejected() -> None:
    res = _result(p=9)
    with pytest.raises(ValueError, match="At most 8"):
        plotting.plot_importance_curves(res, top=None)
    plotting.plot_importance_curves(res)  # default top=8 is fine


@pytest.mark.parametrize(
    ("kwargs", "exc", "match"),
    [
        ({"features": ["nope"]}, ValueError, "Unknown feature"),
        ({"features": ["f0", "f0"]}, ValueError, "duplicates"),
        ({"features": []}, ValueError, "empty"),
        ({"top": 0}, ValueError, "top must be"),
        ({"window": "full"}, ValueError, "window must be"),
        ({"window": np.ones(5, dtype=bool)}, ValueError, "shape"),
        ({"window": np.ones(6)}, TypeError, "boolean"),
    ],
)
def test_curves_validation(kwargs: dict[str, Any], exc: type[Exception], match: str) -> None:
    with pytest.raises(exc, match=match):
        plotting.plot_importance_curves(_result(), **kwargs)


# ---------------------------------------------------------------- integrated


def test_integrated_sorted_largest_on_top_and_matches_integrate() -> None:
    res = _result()
    ax = plotting.plot_integrated(res)
    labels = [t.get_text() for t in ax.get_yticklabels()]
    widths = [b.get_width() for b in ax.patches]
    expected = integrate(res, "censoring").sort_values(ascending=False)
    assert labels == list(expected.index)
    np.testing.assert_allclose(widths, expected.to_numpy())
    assert ax.yaxis_inverted()
    assert min(widths) < 0  # negative importance drawn as is


def test_integrated_top_method_window_and_axes() -> None:
    res = _result()
    _, given = plt.subplots()
    window = res.stable_window()
    ax = plotting.plot_integrated(res, "trapezoid", window=window, top=2, ax=given)
    assert ax is given
    expected = integrate(res, "trapezoid", window=window).sort_values(ascending=False)
    np.testing.assert_allclose([b.get_width() for b in ax.patches], expected.to_numpy()[:2])
    assert "trapezoid" in ax.get_xlabel()
    with pytest.raises(ValueError, match="top must be"):
        plotting.plot_integrated(res, top=-1)


# --------------------------------------------------------------- dependency


def test_missing_matplotlib_message(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.startswith("matplotlib"):
            raise ImportError("no matplotlib")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match=r"timesage-survival\[plot\]"):
        plotting.plot_integrated(_result())


def test_import_timesage_does_not_import_matplotlib() -> None:
    import subprocess
    import sys

    code = "import sys, timesage; print('matplotlib' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"
