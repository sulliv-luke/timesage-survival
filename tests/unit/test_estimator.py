"""Unit tests for the public ``timesage.TimeSAGE`` estimator."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from tests.fixtures.toy_models import toy_predict_survival
from timesage import SurvivalModel, TimeSAGE, TimeSAGEResult
from timesage._ipcw import fit_censoring_km, ipcw_brier_loss

P = 4
IGNORED = 3  # toy model coefficient on feature 3 is 0
NAMES = ["age", "grade", "size", "noise"]


def _data(seed: int = 0, n: int = 80, m: int = 15, K: int = 5):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, P))
    bg = rng.normal(size=(m, P))
    t_ev = 50.0 * rng.weibull(1.3, n) / np.exp(0.5 * X[:, 0])
    t_c = rng.exponential(80.0, n)
    T = np.minimum(t_ev, t_c)
    delta = (t_ev <= t_c).astype(int)
    times = np.linspace(np.percentile(T, 10), np.percentile(T, 80), K)
    return X, bg, T, delta, times


def _explain(model=toy_predict_survival, *, data=None, explain_kwargs=None, **kwargs):
    X, bg, T, delta, times = data if data is not None else _data()
    params = {"n_permutations": 64, "batch_size": 32, "random_state": 0} | kwargs
    params.setdefault("background", bg)
    return TimeSAGE(model, **params).explain(X, T, delta, times, **(explain_kwargs or {}))


class _Model:
    def predict_survival(self, X: np.ndarray, times: np.ndarray) -> np.ndarray:
        return toy_predict_survival(X, times)


def test_protocol_object_and_callable_give_identical_results() -> None:
    assert isinstance(_Model(), SurvivalModel)
    a, b = _explain(_Model()), _explain(toy_predict_survival)
    assert isinstance(a, TimeSAGEResult)
    np.testing.assert_array_equal(a.phi, b.phi)


def test_result_contents() -> None:
    X, _, T, delta, times = data = _data()
    res = _explain(data=data)
    assert res.phi.shape == res.phi_se.shape == (P, len(times))
    np.testing.assert_array_equal(res.phi[IGNORED], 0.0)
    assert res.feature_names == ("x0", "x1", "x2", "x3")
    np.testing.assert_array_equal(res.times, times)
    assert (res.n_permutations, res.converged, res.random_state) == (64, None, 0)
    km = fit_censoring_km(T, delta)
    np.testing.assert_array_equal(res.censoring_survival, km(times))
    # Textbook risk set by default; the original strict inequality kept privately.
    np.testing.assert_array_equal(res.at_risk_fraction, (T[:, None] >= times).mean(axis=0))
    np.testing.assert_array_equal(res._at_risk_fraction_legacy, (T[:, None] > times).mean(axis=0))
    # Full coalition through the imputer = the model on float32 inputs, up to averaging error.
    pred = toy_predict_survival(X.astype(np.float32), times)
    full = ipcw_brier_loss(pred, T, delta, times, km).mean(axis=0)
    np.testing.assert_allclose(res.loss_full, full, rtol=1e-12)
    assert np.all(res.loss_empty > res.loss_full)


def test_model_receives_float32() -> None:
    dtypes = set()

    def model(X: np.ndarray, times: np.ndarray) -> np.ndarray:
        dtypes.add(X.dtype)
        return toy_predict_survival(X, times)

    _explain(model)
    assert dtypes == {np.dtype(np.float32)}


def test_seed_is_recorded_and_reproduces_the_run() -> None:
    X, bg, T, delta, times = _data()
    explainer = TimeSAGE(toy_predict_survival, background=bg, n_permutations=64, batch_size=32)
    first = explainer.explain(X, T, delta, times)
    again = explainer.explain(X, T, delta, times)
    np.testing.assert_array_equal(first.phi, again.phi)
    replay = _explain(random_state=first.random_state)
    np.testing.assert_array_equal(first.phi, replay.phi)
    assert not np.array_equal(first.phi, _explain(random_state=first.random_state + 1).phi)


def test_progress_callback() -> None:
    calls: list[tuple[int, int]] = []
    _explain(n_permutations=70, batch_size=32, progress=lambda d, t: calls.append((d, t)))
    assert calls == [(32, 70), (64, 70), (70, 70)]


def test_early_stopping_is_recorded() -> None:
    res = _explain(n_permutations=8192, batch_size=32, convergence_threshold=0.5)
    assert res.converged is True
    assert res.n_permutations < 8192
    assert res.n_permutations % 32 == 0
    assert res.convergence_ratio < 0.5


def test_background_subsampling() -> None:
    _, bg, *_ = _data(m=15)
    explainer = TimeSAGE(toy_predict_survival, background=bg, n_background=6, random_state=4)
    idx = np.random.default_rng(4).choice(15, size=6, replace=False)
    np.testing.assert_array_equal(explainer.background, bg[idx].astype(np.float32))
    whole = TimeSAGE(toy_predict_survival, background=bg, n_background=100, random_state=4)
    assert whole.background.shape == (15, P)
    as_given = TimeSAGE(toy_predict_survival, background=bg)
    np.testing.assert_array_equal(as_given.background, bg.astype(np.float32))


# ---------------------------------------------------------------- DataFrames


def test_dataframe_names_are_used() -> None:
    X, bg, T, delta, times = _data()
    explainer = TimeSAGE(
        toy_predict_survival, background=bg, n_permutations=64, batch_size=32, random_state=0
    )
    res = explainer.explain(pd.DataFrame(X, columns=NAMES), T, delta, times)
    assert res.feature_names == tuple(NAMES)
    np.testing.assert_array_equal(res.phi, _explain().phi)


def test_dataframe_background_in_another_column_order_is_matched_by_name() -> None:
    X, bg, T, delta, times = _data()
    shuffled = pd.DataFrame(bg, columns=NAMES)[["size", "noise", "age", "grade"]]
    by_name = TimeSAGE(toy_predict_survival, background=shuffled, random_state=0).explain(
        pd.DataFrame(X, columns=NAMES), T, delta, times
    )
    reference = TimeSAGE(toy_predict_survival, background=bg, random_state=0).explain(
        X, T, delta, times, feature_names=NAMES
    )
    np.testing.assert_array_equal(by_name.phi, reference.phi)
    np.testing.assert_array_equal(by_name.phi_se, reference.phi_se)
    assert by_name.feature_names == tuple(NAMES)


def test_array_x_with_dataframe_background_takes_background_names() -> None:
    X, bg, T, delta, times = _data()
    explainer = TimeSAGE(toy_predict_survival, background=pd.DataFrame(bg, columns=NAMES))
    assert explainer.explain(X, T, delta, times).feature_names == tuple(NAMES)


def test_column_mismatch_raises() -> None:
    X, bg, T, delta, times = _data()
    explainer = TimeSAGE(toy_predict_survival, background=pd.DataFrame(bg, columns=NAMES))
    other = pd.DataFrame(X, columns=["age", "grade", "size", "other"])
    with pytest.raises(ValueError, match="columns of X and of the background differ"):
        explainer.explain(other, T, delta, times)


def test_feature_names_conflicting_with_dataframe_raise() -> None:
    X, bg, T, delta, times = _data()
    with pytest.raises(ValueError, match="feature_names does not match"):
        TimeSAGE(toy_predict_survival, background=bg).explain(
            pd.DataFrame(X, columns=NAMES), T, delta, times, feature_names=list("abcd")
        )


def test_inputs_are_not_mutated() -> None:
    X, bg, T, delta, times = _data()
    X_df, bg_df = pd.DataFrame(X, columns=NAMES), pd.DataFrame(bg, columns=NAMES)
    arrays = (X, bg, T, delta, times)
    copies = [a.copy() for a in arrays]
    frames = (X_df.copy(), bg_df.copy())
    TimeSAGE(toy_predict_survival, background=bg, n_background=5, random_state=0).explain(
        X, T, delta, times, censoring_time=T, censoring_event=delta
    )
    TimeSAGE(toy_predict_survival, background=bg_df[NAMES[::-1]]).explain(X_df, T, delta, times)
    for a, c in zip(arrays, copies, strict=True):
        np.testing.assert_array_equal(a, c)
    pd.testing.assert_frame_equal(X_df, frames[0])
    pd.testing.assert_frame_equal(bg_df, frames[1])


# ---------------------------------------------------------------- censoring curve


def test_censoring_override_is_used() -> None:
    _, _, T, _, times = data = _data()
    rng = np.random.default_rng(9)
    T_c, d_c = rng.exponential(60.0, 500) + 1.0, rng.integers(0, 2, 500)
    T_c[0], d_c[0] = T.max() + 1.0, 1
    res = _explain(data=data, explain_kwargs={"censoring_time": T_c, "censoring_event": d_c})
    np.testing.assert_array_equal(res.censoring_survival, fit_censoring_km(T_c, d_c)(times))
    assert not np.array_equal(res.phi, _explain(data=data).phi)


@pytest.mark.parametrize("given", ["censoring_time", "censoring_event"])
def test_censoring_override_needs_both_arrays(given: str) -> None:
    X, bg, T, delta, times = _data()
    value = T if given == "censoring_time" else delta
    with pytest.raises(ValueError, match="together"):
        TimeSAGE(toy_predict_survival, background=bg).explain(X, T, delta, times, **{given: value})


def test_censoring_override_must_cover_the_grid() -> None:
    X, bg, T, delta, times = _data()
    with pytest.raises(ValueError, match="follow-up of the censoring data"):
        TimeSAGE(toy_predict_survival, background=bg).explain(
            X, T, delta, times, censoring_time=[1.0, times[-1] / 2], censoring_event=[0, 1]
        )


def _censored_last(T: np.ndarray, delta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    delta = delta.copy()
    delta[np.argmax(T)] = 0  # largest time censored -> G_hat reaches 0 there
    return T, delta


def test_zero_censoring_survival_on_the_grid_raises() -> None:
    # The original implementation clipped G_hat = 0 to 1e-12 silently.
    X, bg, T, delta, _ = _data()
    T, delta = _censored_last(T, delta)
    times = np.array([np.percentile(T, 50), T.max()])
    with pytest.raises(ValueError, match="G_hat"):
        TimeSAGE(toy_predict_survival, background=bg).explain(X, T, delta, times)
    # Ending the grid just before the last time is fine.
    TimeSAGE(toy_predict_survival, background=bg, n_permutations=8).explain(
        X, T, delta, np.array([np.percentile(T, 50), np.sort(T)[-2]])
    )


def test_zero_censoring_survival_from_override_raises() -> None:
    X, bg, T, delta, times = _data()
    # Censoring data whose largest time, at the last grid time, is censored.
    T_c = np.concatenate([T[T < times[-1]], [times[-1]]])
    d_c = np.concatenate([np.ones(len(T_c) - 1, dtype=int), [0]])
    with pytest.raises(ValueError, match="G_hat"):
        TimeSAGE(toy_predict_survival, background=bg).explain(
            X, T, delta, times, censoring_time=T_c, censoring_event=d_c
        )


def test_low_censoring_survival_warns_once_listing_the_times() -> None:
    # Eq. 31: G_hat(t) >= 0.05 bounds the stable window; below it 1/G_hat > 20.
    *_, times = _data()
    T_c = np.linspace(0.01, 1.01 * times[-1], 400)  # censoring almost everyone by t_K
    d_c = np.zeros(len(T_c), dtype=int)
    d_c[-1] = 1  # keeps G_hat > 0 on the grid
    g = fit_censoring_km(T_c, d_c)(times)
    low = times[g < 0.05]
    assert 0 < len(low) < len(times)
    with pytest.warns(UserWarning, match=r"G_hat\(t\) is below 0.05") as record:
        res = _explain(explain_kwargs={"censoring_time": T_c, "censoring_event": d_c})
    assert len(record) == 1
    message = str(record[0].message)
    for t in times:
        assert (f"{t:g}" in message) == (t in low)
    np.testing.assert_array_equal(res.censoring_survival, g)  # numbers are unaffected


def test_no_censoring_warning_inside_the_stable_window() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = _explain()
    assert np.all(res.censoring_survival >= 0.05)


# ---------------------------------------------------------------- model output checks


def _scaled(factor: float = 1.0, shift: float = 0.0, rise: float = 0.0, dtype=np.float64):
    def model(X: np.ndarray, times: np.ndarray) -> np.ndarray:
        S = toy_predict_survival(X, times) * factor + shift
        S[:, -1] += rise
        return S.astype(dtype)

    return model


def test_model_output_outside_unit_interval_raises() -> None:
    with pytest.raises(ValueError, match=r"probabilities in \[0, 1\]"):
        _explain(_scaled(factor=1.5))
    with pytest.raises(ValueError, match=r"probabilities in \[0, 1\]"):
        _explain(_scaled(shift=-0.01))


def test_float32_rounding_noise_is_tolerated() -> None:
    # About 1e-7 outside [0, 1] and rising by 1e-7: float32 rounding, not a broken model.
    def model(X: np.ndarray, times: np.ndarray) -> np.ndarray:
        S = toy_predict_survival(X, times).astype(np.float32)
        S[:, 0] = np.float32(1.0) + np.float32(1.2e-7)
        S[:, 1] = S[:, 0]
        return S

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _explain(model)


def test_rising_survival_curves_warn_once() -> None:
    with pytest.warns(UserWarning, match="increase over time") as record:
        _explain(_scaled(factor=0.5, rise=0.2))
    assert len(record) == 1


def test_model_output_shape_and_finiteness_are_checked() -> None:
    with pytest.raises(ValueError, match="returned shape"):
        _explain(lambda X, t: np.ones((len(X), len(t) + 1)))
    with pytest.raises(ValueError, match="non-finite"):
        _explain(lambda X, t: np.full((len(X), len(t)), np.nan))


def test_non_finite_output_on_imputed_rows_raises() -> None:
    # Finite on X itself (x0 == x1 in every row), NaN once imputation mixes in
    # background values: the trial call cannot see this, the per-call check must.
    X, bg, T, delta, times = _data()
    X[:, 1] = X[:, 0]

    def model(X_in: np.ndarray, t: np.ndarray) -> np.ndarray:
        S = toy_predict_survival(X_in, t)
        S[X_in[:, 0] != X_in[:, 1]] = np.nan
        return S

    with pytest.raises(ValueError, match="non-finite values on imputed rows"):
        _explain(model, data=(X, bg, T, delta, times))


# ---------------------------------------------------------------- argument validation


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        ({"n_permutations": 10.0}, TypeError, "n_permutations must be an integer"),
        ({"n_permutations": 0}, ValueError, "n_permutations must be >= 1"),
        ({"batch_size": True}, TypeError, "batch_size must be an integer"),
        ({"n_background": 0}, ValueError, "n_background must be >= 1"),
        ({"convergence_threshold": 1.0}, ValueError, "convergence_threshold"),
        ({"convergence_threshold": 0.0}, ValueError, "convergence_threshold"),
        ({"random_state": -1}, ValueError, "random_state must be >= 0"),
        ({"random_state": np.random.default_rng(0)}, TypeError, "random_state"),
        ({"progress": "yes"}, TypeError, "progress"),
    ],
)
def test_constructor_validation(kwargs: dict, error: type[Exception], match: str) -> None:
    _, bg, *_ = _data()
    with pytest.raises(error, match=match):
        TimeSAGE(toy_predict_survival, background=bg, **kwargs)


def test_model_must_be_callable() -> None:
    with pytest.raises(TypeError, match="predict_survival"):
        TimeSAGE(object(), background=np.zeros((3, P)))  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("background", "match"),
    [
        (np.zeros(4), "background must be a non-empty 2-D"),
        (np.zeros((0, 4)), "background must be a non-empty 2-D"),
        (np.array([[np.nan, 0, 0, 0]]), "background must contain only finite"),
        (pd.DataFrame({"a": ["x", "y"]}), "background must contain only numeric"),
    ],
)
def test_background_validation(background: object, match: str) -> None:
    with pytest.raises((ValueError, TypeError), match=match):
        TimeSAGE(toy_predict_survival, background=background)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("name", "make", "match"),
    [
        ("X", lambda X, T, t: X[:, :3], "X has 3 features but the background has 4"),
        ("X", lambda X, T, t: X[:-1], "X has 79 rows but time and event have 80"),
        ("X", lambda X, T, t: np.where(X > 2, np.inf, X), "X must contain only finite"),
        ("event", lambda X, T, t: np.full(len(T), 2), r"event must take values in \{0, 1\}"),
        ("times", lambda X, T, t: t[::-1], "strictly increasing"),
        ("times", lambda X, T, t: np.array([-1.0, 1.0]), "non-negative"),
        ("times", lambda X, T, t: np.array([]), "non-empty 1-D"),
        ("times", lambda X, T, t: np.array([1.0, T.max() + 1]), "beyond the observed follow-up"),
    ],
)
def test_explain_validation(name: str, make, match: str) -> None:
    X, bg, T, delta, times = _data()
    args = {"X": X, "time": T, "event": delta, "times": times}
    args[name] = make(X, T, times)
    with pytest.raises(ValueError, match=match):
        TimeSAGE(toy_predict_survival, background=bg).explain(**args)  # type: ignore[arg-type]


def test_duplicate_feature_names_raise() -> None:
    X, bg, T, delta, times = _data()
    with pytest.raises(ValueError, match="unique"):
        TimeSAGE(toy_predict_survival, background=bg).explain(
            X, T, delta, times, feature_names=["a", "a", "b", "c"]
        )
