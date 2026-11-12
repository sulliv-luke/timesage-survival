"""Unit tests for ``timesage.adapters.coxtv.CoxTimeVaryingAdapter``.

The default (faithful) mode is checked against lifelines itself:

- the bin-wise linear predictor equals ``predict_log_partial_hazard`` on the
  long-format row (exact up to rounding);
- survival equals an independent long-format reference built from
  ``predict_partial_hazard`` and the step function ``baseline_cumulative_hazard_``;
- with one bin, survival equals ``CoxPHFitter`` on the wide data at every
  observed time (atol 1e-4: separate Newton fits of the same Efron likelihood);
- survival is invariant to shifting the covariates before fitting (atol 1e-6),
  which fails without lifelines' centring.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
import pytest

from timesage.adapters.coxtv import CoxTimeVaryingAdapter

lifelines = pytest.importorskip("lifelines")

FEATURES = ["a", "b", "c"]
BETA_TRUE = np.array([0.8, -0.5, 0.3])


def _make_wide(seed: int = 7, n: int = 250) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, len(FEATURES)))
    T = rng.exponential(1.0 / (0.05 * np.exp(X @ BETA_TRUE)))
    C = rng.uniform(0.0, 60.0, size=n)
    df = pd.DataFrame(X, columns=FEATURES)
    df["duration"] = np.minimum(T, C)
    df["event"] = (T <= C).astype(int)
    df["id"] = np.arange(n)
    return df


def _cut_points(durations: np.ndarray, n_bins: int) -> np.ndarray:
    cuts = np.quantile(durations, np.linspace(0, 1, n_bins + 1))
    cuts[0] = 0.0
    return np.unique(cuts)


def _make_long(df: pd.DataFrame, cuts: np.ndarray) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, r in df.iterrows():
        t, e = float(r["duration"]), int(r["event"])
        for b in range(len(cuts) - 1):
            if cuts[b] >= t:
                break
            stop = min(float(cuts[b + 1]), t)
            row: dict[str, Any] = {
                "id": int(r["id"]),
                "start": float(cuts[b]),
                "stop": stop,
                "event": int(e == 1 and stop == t),
            }
            row.update({f"{f}__bin{b}": float(r[f]) for f in FEATURES})
            rows.append(row)
    tv = pd.DataFrame(rows)
    bin_cols = [c for c in tv.columns if "__bin" in c]
    tv[bin_cols] = tv[bin_cols].fillna(0.0)
    return tv


def _fit(df: pd.DataFrame, n_bins: int) -> tuple[Any, np.ndarray]:
    cuts = _cut_points(df["duration"].to_numpy(), n_bins)
    ctv = lifelines.CoxTimeVaryingFitter(penalizer=0.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ctv.fit(
            _make_long(df, cuts), id_col="id", start_col="start", stop_col="stop", event_col="event"
        )
    return ctv, cuts


def _query_times(df: pd.DataFrame, cuts: np.ndarray) -> np.ndarray:
    """Event times, censoring times, cut points, 0, between-event and beyond-follow-up times."""
    obs = np.sort(df["duration"].to_numpy())
    mids = 0.5 * (obs[:-1] + obs[1:])
    return np.unique(np.concatenate([[0.0], obs[::7], mids[::11], cuts, [obs[-1] * 1.5]]))


@pytest.fixture(scope="module")
def wide() -> pd.DataFrame:
    df = _make_wide()
    df[FEATURES] = df[FEATURES] + 5.0  # non-centred covariates, so x_bar' beta != 0
    return df


@pytest.fixture(scope="module")
def fitted(wide: pd.DataFrame) -> tuple[Any, np.ndarray, CoxTimeVaryingAdapter]:
    ctv, cuts = _fit(wide, n_bins=3)
    return ctv, cuts, CoxTimeVaryingAdapter(ctv, cuts, FEATURES)


# ---------------------------------------------------------------- faithfulness


def test_linear_predictor_matches_lifelines(fitted: Any, wide: pd.DataFrame) -> None:
    ctv, cuts, adapter = fitted
    X = wide[FEATURES].to_numpy()
    assert abs(adapter.centring_constant) > 0.1
    for b in range(len(cuts) - 1):
        rows = pd.DataFrame(0.0, index=range(len(X)), columns=ctv.params_.index)
        rows[[f"{f}__bin{b}" for f in FEATURES]] = X
        expected = ctv.predict_log_partial_hazard(rows).to_numpy()
        got = X @ adapter._betas[b] - adapter.centring_constant
        np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-12)


def _reference_survival(ctv: Any, cuts: np.ndarray, X: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Independent reference: integrate over long-format rows (start, min(stop, t)]."""
    H0 = ctv.baseline_cumulative_hazard_.iloc[:, 0]

    def H0_step(s: float) -> float:
        before = H0[H0.index <= s]
        return float(before.iloc[-1]) if len(before) else 0.0

    S = np.empty((len(X), len(times)))
    for i, x in enumerate(X):
        rows = pd.DataFrame(0.0, index=range(len(cuts) - 1), columns=ctv.params_.index)
        for b in range(len(cuts) - 1):
            rows.loc[b, [f"{f}__bin{b}" for f in FEATURES]] = x
        ph = ctv.predict_partial_hazard(rows).to_numpy()
        for k, t in enumerate(times):
            H = 0.0
            for b in range(len(cuts) - 1):
                if t > cuts[b]:
                    H += ph[b] * (H0_step(min(t, cuts[b + 1])) - H0_step(cuts[b]))
            S[i, k] = np.exp(-H)
    return S


def test_matches_independent_long_format_reference(fitted: Any, wide: pd.DataFrame) -> None:
    ctv, cuts, adapter = fitted
    X = wide[FEATURES].to_numpy()[::25]
    times = _query_times(wide, cuts)
    np.testing.assert_allclose(
        adapter.predict_survival(X, times),
        _reference_survival(ctv, cuts, X, times),
        rtol=1e-12,
        atol=1e-14,
    )


@pytest.mark.parametrize("shift", [0.0, 5.0])
def test_single_bin_matches_coxph(shift: float) -> None:
    df = _make_wide()
    df[FEATURES] = df[FEATURES] + shift
    ctv, cuts = _fit(df, n_bins=1)
    adapter = CoxTimeVaryingAdapter(ctv, cuts, FEATURES)
    cph = lifelines.CoxPHFitter(penalizer=0.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cph.fit(df[[*FEATURES, "duration", "event"]], duration_col="duration", event_col="event")
    times = np.unique(df["duration"].to_numpy())  # every observed time
    expected = cph.predict_survival_function(df[FEATURES], times=times).to_numpy().T
    np.testing.assert_allclose(adapter.predict_survival(df[FEATURES], times), expected, atol=1e-4)


def test_invariant_to_covariate_shift() -> None:
    """Cox survival does not depend on the covariate origin; only correct centring achieves this."""
    base = _make_wide()
    shifted = base.copy()
    shifted[FEATURES] = shifted[FEATURES] + 5.0
    ctv0, cuts = _fit(base, n_bins=3)
    ctv5, _ = _fit(shifted, n_bins=3)
    times = _query_times(base, cuts)
    S0 = CoxTimeVaryingAdapter(ctv0, cuts, FEATURES).predict_survival(base[FEATURES], times)
    S5 = CoxTimeVaryingAdapter(ctv5, cuts, FEATURES).predict_survival(shifted[FEATURES], times)
    np.testing.assert_allclose(S5, S0, atol=1e-6)
    legacy5 = CoxTimeVaryingAdapter(ctv5, cuts, FEATURES, legacy_parity=True)
    assert np.abs(legacy5.predict_survival(shifted[FEATURES], times) - S0).max() > 0.01


# ------------------------------------------------------------------- contract


def test_output_contract(fitted: Any, wide: pd.DataFrame) -> None:
    _, cuts, adapter = fitted
    times = _query_times(wide, cuts)
    rng = np.random.default_rng(0)
    order = rng.permutation(len(times))
    S = adapter.predict_survival(wide[FEATURES].to_numpy(), times)
    assert S.shape == (len(wide), len(times))
    assert S.dtype == np.float64
    assert np.all((S >= 0) & (S <= 1))
    assert np.all(np.diff(S, axis=1) <= 0)
    assert np.all(S[:, 0] == 1.0)  # times[0] == 0
    assert np.all(S[:, -1] == S[:, -2])  # flat beyond the last cut / event time
    # Unsorted times give the same values, column for column.
    S_perm = adapter.predict_survival(wide[FEATURES].to_numpy(), times[order])
    np.testing.assert_array_equal(S_perm, S[:, order])


def test_dataframe_aligned_by_name_and_inputs_not_mutated(fitted: Any, wide: pd.DataFrame) -> None:
    _, _, adapter = fitted
    df = wide[FEATURES[::-1]].copy()
    before = df.copy()
    times = np.array([5.0, 20.0, 40.0])
    np.testing.assert_array_equal(
        adapter.predict_survival(df, times),
        adapter.predict_survival(wide[FEATURES].to_numpy(), times),
    )
    pd.testing.assert_frame_equal(df, before)


def test_extreme_linear_predictor_gives_no_nan(fitted: Any) -> None:
    _, _, adapter = fitted
    X = np.array([[1e6, -1e6, 1e6], [-1e6, 1e6, -1e6]])
    S = adapter.predict_survival(X, np.array([0.0, 1.0, 30.0, 100.0]))
    assert np.all(np.isfinite(S))
    assert np.all((S >= 0) & (S <= 1))


# ----------------------------------------------------------------- validation


class _FakeModel:
    def __init__(self, params: pd.Series, H0: pd.DataFrame) -> None:
        self.params_ = params
        self.baseline_cumulative_hazard_ = H0

    def predict_log_partial_hazard(self, X: pd.DataFrame) -> pd.Series:
        return pd.Series(np.zeros(len(X)))


def _fake(columns: list[str], n_strata: int = 1) -> _FakeModel:
    H0 = pd.DataFrame(np.ones((3, n_strata)), index=[1.0, 2.0, 3.0])
    return _FakeModel(pd.Series(0.1, index=columns), H0)


GOOD_COLS = [f"{f}__bin{b}" for b in range(2) for f in ["x", "y"]]


@pytest.mark.parametrize(
    ("model", "cuts", "match"),
    [
        (object(), [0.0, 1.0, 2.0], "fitted lifelines"),
        (_fake(GOOD_COLS, n_strata=2), [0.0, 1.0, 2.0], "Stratified"),
        (_fake([*GOOD_COLS, "z__bin0"]), [0.0, 1.0, 2.0], "unexpected"),
        (_fake(GOOD_COLS[:-1]), [0.0, 1.0, 2.0], "missing"),
        (_fake(GOOD_COLS), [1.0, 2.0, 3.0], "start at 0"),
        (_fake(GOOD_COLS), [0.0, 2.0, 2.0], "strictly increasing"),
        (_fake(GOOD_COLS), [0.0], "at least two"),
    ],
)
def test_constructor_validation(model: Any, cuts: list[float], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        CoxTimeVaryingAdapter(model, cuts, ["x", "y"])


@pytest.mark.parametrize(
    ("X", "times", "match"),
    [
        (np.zeros((2, 3)), [1.0], "shape"),
        (np.array([[np.nan, 0.0]]), [1.0], "non-finite"),
        (np.zeros((2, 2)), [-1.0], "non-negative"),
        (np.zeros((2, 2)), [[1.0]], "one-dimensional"),
        (pd.DataFrame({"x": [0.0]}), [1.0], "missing feature"),
    ],
)
def test_predict_validation(X: Any, times: Any, match: str) -> None:
    adapter = CoxTimeVaryingAdapter(_fake(GOOD_COLS), [0.0, 1.0, 2.0], ["x", "y"])
    with pytest.raises(ValueError, match=match):
        adapter.predict_survival(X, times)


# --------------------------------------------------------------------- scaler


@pytest.mark.parametrize("legacy_parity", [False, True])
def test_scaler_equals_prescaled_inputs(wide: pd.DataFrame, legacy_parity: bool) -> None:
    from tests.fixtures.scalers import AffineScaler

    raw = wide[FEATURES].to_numpy()
    scaler = AffineScaler(raw.mean(axis=0), raw.std(axis=0))
    scaled = wide.copy()
    scaled[FEATURES] = scaler.transform(raw)
    ctv, cuts = _fit(scaled, n_bins=3)
    times = _query_times(wide, cuts)
    plain = CoxTimeVaryingAdapter(ctv, cuts, FEATURES, legacy_parity=legacy_parity)
    with_scaler = CoxTimeVaryingAdapter(
        ctv, cuts, FEATURES, legacy_parity=legacy_parity, scaler=scaler
    )
    np.testing.assert_array_equal(
        with_scaler.predict_survival(wide[FEATURES], times),
        plain.predict_survival(scaled[FEATURES], times),
    )


def test_scaler_validation(fitted: Any) -> None:
    ctv, cuts, _ = fitted
    with pytest.raises(TypeError, match="transform"):
        CoxTimeVaryingAdapter(ctv, cuts, FEATURES, scaler=object())
