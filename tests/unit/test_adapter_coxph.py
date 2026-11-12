"""Unit tests for ``timesage.adapters.coxph.CoxPHAdapter``.

The adapter must return exactly lifelines' own ``predict_survival_function``,
including lifelines' linear interpolation between training times, and satisfy
the adapter contract (shape, values in [0, 1], non-increasing in time, any
query times). ``scaler=`` must give the same importances as pre-scaled inputs:
checked bit for bit with a scaler whose arithmetic is exact in float32.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
import pytest

from tests.fixtures.scalers import AffineScaler
from timesage import TimeSAGE
from timesage.adapters import CoxPHAdapter

lifelines = pytest.importorskip("lifelines")

FEATURES = ["a", "b", "c"]
BETA_TRUE = np.array([0.8, -0.5, 0.0])


def _make_wide(seed: int = 3, n: int = 200, shift: float = 5.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, len(FEATURES)))
    T = rng.exponential(1.0 / (0.05 * np.exp(X @ BETA_TRUE)))
    C = rng.uniform(0.0, 60.0, size=n)
    df = pd.DataFrame(X + shift, columns=FEATURES)
    df["duration"] = np.minimum(T, C)
    df["event"] = (T <= C).astype(int)
    return df


def _fit(df: pd.DataFrame, features: list[str] = FEATURES) -> Any:
    cph = lifelines.CoxPHFitter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cph.fit(df[[*features, "duration", "event"]], duration_col="duration", event_col="event")
    return cph


def _query_times(df: pd.DataFrame) -> np.ndarray:
    """0, observed times, midpoints between them, and times beyond follow-up."""
    obs = np.sort(df["duration"].to_numpy())
    mids = 0.5 * (obs[:-1] + obs[1:])
    return np.unique(np.concatenate([[0.0], obs[::5], mids[::7], [obs[-1] * 1.5]]))


@pytest.fixture(scope="module")
def wide() -> pd.DataFrame:
    return _make_wide()


@pytest.fixture(scope="module")
def cph(wide: pd.DataFrame) -> Any:
    return _fit(wide)


# ---------------------------------------------------------------- faithfulness


def test_returns_lifelines_survival(cph: Any, wide: pd.DataFrame) -> None:
    times = _query_times(wide)
    expected = cph.predict_survival_function(wide[FEATURES], times=times).to_numpy().T
    S = CoxPHAdapter(cph).predict_survival(wide[FEATURES].to_numpy(), times)
    np.testing.assert_array_equal(S, expected)


def test_interpolates_linearly_between_training_times(cph: Any, wide: pd.DataFrame) -> None:
    # lifelines' convention, documented in the docstring: not a step function.
    H0 = cph.baseline_cumulative_hazard_.iloc[:, 0]
    g = H0.index.to_numpy()
    k = int(np.argmax(np.diff(H0.to_numpy()) > 0)) + 1  # g[k] is an event time
    mid = 0.5 * (g[k - 1] + g[k])
    S = CoxPHAdapter(cph).predict_survival(wide[FEATURES].iloc[:1], np.array([g[k - 1], mid, g[k]]))
    assert S[0, 0] > S[0, 1] > S[0, 2]
    np.testing.assert_allclose(np.log(S[0, 1]), 0.5 * (np.log(S[0, 0]) + np.log(S[0, 2])))


# ------------------------------------------------------------------- contract


def test_output_contract(cph: Any, wide: pd.DataFrame) -> None:
    times = _query_times(wide)
    order = np.random.default_rng(0).permutation(len(times))
    adapter = CoxPHAdapter(cph)
    X = wide[FEATURES].to_numpy(dtype=np.float32)  # as TimeSAGE passes it
    S = adapter.predict_survival(X, times)
    assert S.shape == (len(wide), len(times))
    assert S.dtype == np.float64
    assert np.all((S >= 0) & (S <= 1))
    assert np.all(np.diff(S, axis=1) <= 0)
    assert np.all(S[:, -1] == S[:, -2])  # constant after the last training time
    np.testing.assert_array_equal(adapter.predict_survival(X, times[order]), S[:, order])


def test_before_first_training_time_is_lifelines_first_value(cph: Any, wide: pd.DataFrame) -> None:
    g1 = float(cph.baseline_cumulative_hazard_.index[0])
    S = CoxPHAdapter(cph).predict_survival(wide[FEATURES], np.array([0.0, 0.5 * g1, g1]))
    np.testing.assert_array_equal(S[:, 0], S[:, 2])
    np.testing.assert_array_equal(S[:, 1], S[:, 2])


# ------------------------------------------------------------- feature order


def test_feature_names_reorder_array_columns(cph: Any, wide: pd.DataFrame) -> None:
    times = _query_times(wide)
    expected = CoxPHAdapter(cph).predict_survival(wide[FEATURES].to_numpy(), times)
    order = ["c", "a", "b"]
    adapter = CoxPHAdapter(cph, feature_names=order)
    np.testing.assert_array_equal(adapter.predict_survival(wide[order].to_numpy(), times), expected)
    # A DataFrame is aligned by name whatever its column order.
    np.testing.assert_array_equal(adapter.predict_survival(wide[FEATURES], times), expected)


@pytest.mark.parametrize(
    "names", [["a", "b"], ["a", "b", "d"], ["a", "a", "b"], ["a", "b", "c", "d"]]
)
def test_feature_names_must_match_the_model(cph: Any, names: list[str]) -> None:
    with pytest.raises(ValueError, match="exactly the model's covariates") as info:
        CoxPHAdapter(cph, feature_names=names)
    assert str(names) in str(info.value)
    assert str(FEATURES) in str(info.value)  # both orders are printed


def test_dataframe_inputs_are_not_mutated(cph: Any, wide: pd.DataFrame) -> None:
    X = wide[FEATURES].copy()
    before = X.copy()
    CoxPHAdapter(cph, scaler=AffineScaler(np.full(3, 5.0), np.full(3, 2.0))).predict_survival(
        X, np.array([1.0, 2.0])
    )
    pd.testing.assert_frame_equal(X, before)


# --------------------------------------------------------------------- scaler


def test_scaler_equals_prescaled_inputs(wide: pd.DataFrame) -> None:
    scaler = AffineScaler(np.full(3, 5.0), np.full(3, 2.0))
    scaled = wide.copy()
    scaled[FEATURES] = scaler.transform(wide[FEATURES].to_numpy())
    cph = _fit(scaled)
    times = _query_times(wide)
    on_scaled = CoxPHAdapter(cph).predict_survival(scaled[FEATURES], times)
    on_raw = CoxPHAdapter(cph, scaler=scaler).predict_survival(wide[FEATURES], times)
    np.testing.assert_array_equal(on_raw, on_scaled)


def test_scaler_gives_the_same_importances(wide: pd.DataFrame) -> None:
    """Marginal imputation commutes with a per-feature affine scaler.

    TimeSAGE casts inputs to float32 before the adapter scales them, so the
    paths agree bit for bit only when scaling is exact in float32: raw values
    that are float32 numbers, and a scale of 2.
    """
    scaler = AffineScaler(np.zeros(3), np.full(3, 2.0))
    wide = wide.copy()
    wide[FEATURES] = wide[FEATURES].astype(np.float32).astype(np.float64)
    scaled = wide.copy()
    scaled[FEATURES] = scaler.transform(wide[FEATURES].to_numpy())
    cph = _fit(scaled)
    T, E = wide["duration"].to_numpy(), wide["event"].to_numpy()
    grid = np.quantile(T, [0.2, 0.4, 0.6])

    def run(model: CoxPHAdapter, X: pd.DataFrame) -> np.ndarray:
        explainer = TimeSAGE(model, background=X.iloc[:20], n_permutations=8, random_state=1)
        return explainer.explain(X.iloc[20:80], T[20:80], E[20:80], grid).phi

    phi_scaled = run(CoxPHAdapter(cph), scaled[FEATURES])
    phi_raw = run(CoxPHAdapter(cph, scaler=scaler), wide[FEATURES])
    np.testing.assert_array_equal(phi_raw, phi_scaled)
    assert np.abs(phi_raw).max() > 0


def test_scaler_validation(cph: Any, wide: pd.DataFrame) -> None:
    with pytest.raises(TypeError, match="transform"):
        CoxPHAdapter(cph, scaler=object())

    class Bad:
        def transform(self, X: np.ndarray) -> np.ndarray:
            return np.asarray(X)[:, :2]

    with pytest.raises(ValueError, match="changed the shape"):
        CoxPHAdapter(cph, scaler=Bad()).predict_survival(wide[FEATURES], np.array([1.0]))


# ----------------------------------------------------------------- validation


def test_constructor_validation(wide: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="fitted lifelines CoxPHFitter"):
        CoxPHAdapter(object())
    strat = wide.copy()
    strat["s"] = (strat["a"] > 5).astype(int)
    model = lifelines.CoxPHFitter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(strat[["a", "b", "s", "duration", "event"]], "duration", "event", strata=["s"])
    with pytest.raises(ValueError, match="Stratified"):
        CoxPHAdapter(model)


@pytest.mark.parametrize(
    ("X", "times", "match"),
    [
        (np.ones((2, 2)), [1.0], "shape"),
        (np.array([[1.0, np.nan, 0.0]]), [1.0], "non-finite"),
        (np.ones((2, 3)), [[1.0]], "one-dimensional"),
        (np.ones((2, 3)), [-1.0], "non-negative"),
        (pd.DataFrame({"a": [1.0], "b": [1.0]}), [1.0], "missing feature columns"),
    ],
)
def test_predict_validation(cph: Any, X: Any, times: Any, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        CoxPHAdapter(cph).predict_survival(X, times)
