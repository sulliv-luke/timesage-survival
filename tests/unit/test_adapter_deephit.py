"""Unit tests for ``timesage.adapters.deephit.DeepHitAdapter``.

The adapter must return pycox's survival at the cuts, pycox's own
``const_pdf`` interpolation between them, and satisfy the adapter contract
(shape, values in [0, 1], non-increasing in time, any query times).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from tests.fixtures.scalers import AffineScaler
from timesage import TimeSAGE

torch = pytest.importorskip("torch")
pytest.importorskip("pycox")

from timesage.adapters import DeepHitAdapter  # noqa: E402

P = 3
N_CUTS = 10
# pycox's interpolate() computes in float32 torch arithmetic; the adapter in
# float64 (np.interp on the float32 survival). They agree to float32 rounding.
PYCOX_F32_ATOL = 1e-6


def _data(seed: int = 1, n: int = 160) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, P)) * 2.0 + 5.0  # raw scale, not standardised
    T = rng.exponential(10.0 / np.exp(0.6 * (X[:, 0] - 5.0)))
    C = rng.uniform(0.0, 30.0, size=n)
    return {"X": X, "T": np.minimum(T, C), "E": (T <= C).astype(int)}


def fit_deephit(X: np.ndarray, T: np.ndarray, E: np.ndarray, seed: int = 0) -> Any:
    import torchtuples as tt
    from pycox.models import DeepHitSingle

    torch.manual_seed(seed)
    labtrans = DeepHitSingle.label_transform(N_CUTS)
    y = labtrans.fit_transform(T, E)
    net = tt.practical.MLPVanilla(
        X.shape[1], [16], labtrans.out_features, batch_norm=True, dropout=0.1
    )
    model = DeepHitSingle(
        net, tt.optim.Adam(1e-2), alpha=0.2, sigma=0.1, duration_index=labtrans.cuts
    )
    model.fit(X.astype(np.float32), y, batch_size=32, epochs=5, verbose=False)
    return model


@pytest.fixture(scope="module")
def data() -> dict[str, np.ndarray]:
    return _data()


@pytest.fixture(scope="module")
def scaler(data: dict[str, np.ndarray]) -> AffineScaler:
    return AffineScaler(data["X"].mean(axis=0), data["X"].std(axis=0))


@pytest.fixture(scope="module")
def model(data: dict[str, np.ndarray], scaler: AffineScaler) -> Any:
    return fit_deephit(scaler.transform(data["X"]), data["T"], data["E"])


def _query_times(model: Any) -> np.ndarray:
    cuts = np.asarray(model.duration_index, dtype=float)
    mids = 0.5 * (cuts[:-1] + cuts[1:])
    return np.unique(np.concatenate([cuts, mids, [0.3 * cuts[1], 1.5 * cuts[-1]]]))


def _scaled(data: dict[str, np.ndarray], scaler: AffineScaler) -> np.ndarray:
    return scaler.transform(data["X"]).astype(np.float32)


# ---------------------------------------------------------------- faithfulness


def test_returns_pycox_survival_at_the_cuts(model: Any, data: dict, scaler: Any) -> None:
    X = _scaled(data, scaler)
    surv = model.predict_surv_df(X)
    S = DeepHitAdapter(model).predict_survival(X, surv.index.to_numpy(dtype=float))
    np.testing.assert_array_equal(S, surv.to_numpy(dtype=float).T)


def test_between_cuts_is_pycox_const_pdf(model: Any, data: dict, scaler: Any) -> None:
    """pycox's default interpolation scheme, computed by pycox itself."""
    X = _scaled(data, scaler)
    interp = model.interpolate(10, scheme="const_pdf").predict_surv_df(X)
    S = DeepHitAdapter(model).predict_survival(X, interp.index.to_numpy(dtype=float))
    np.testing.assert_allclose(S, interp.to_numpy(dtype=float).T, rtol=0, atol=PYCOX_F32_ATOL)


def test_survival_at_zero_is_below_one(model: Any, data: dict, scaler: Any) -> None:
    """Pins pycox's convention S(0) = 1 - pmf_0 < 1; fails loudly if pycox changes it."""
    X = _scaled(data, scaler)
    S0 = DeepHitAdapter(model).predict_survival(X, np.array([0.0]))[:, 0]
    pmf0 = np.asarray(model.predict_pmf(X), dtype=float)[:, 0]
    assert np.all(S0 < 1.0)
    np.testing.assert_allclose(S0, 1.0 - pmf0, rtol=0, atol=PYCOX_F32_ATOL)


def test_constant_after_the_last_cut(model: Any, data: dict, scaler: Any) -> None:
    X = _scaled(data, scaler)
    last = float(model.duration_index[-1])
    S = DeepHitAdapter(model).predict_survival(X, np.array([last, 1.5 * last, 10.0 * last]))
    np.testing.assert_array_equal(S[:, 1], S[:, 0])
    np.testing.assert_array_equal(S[:, 2], S[:, 0])


# ------------------------------------------------------------------- contract


def test_output_contract(model: Any, data: dict, scaler: Any) -> None:
    times = _query_times(model)
    order = np.random.default_rng(0).permutation(len(times))
    adapter = DeepHitAdapter(model, scaler=scaler)
    X = data["X"].astype(np.float32)  # as TimeSAGE passes it
    S = adapter.predict_survival(X, times)
    assert S.shape == (len(X), len(times))
    assert S.dtype == np.float64
    assert np.all((S >= 0) & (S <= 1))
    assert np.all(np.diff(S, axis=1) <= 0)
    np.testing.assert_array_equal(adapter.predict_survival(X, times[order]), S[:, order])


def test_predictions_do_not_depend_on_network_mode(model: Any, data: dict[str, np.ndarray]) -> None:
    """Contract: pycox predicts in eval() mode whatever mode the net is in.

    With dropout present, a prediction in training mode would be random, so
    repeated calls on a net in training mode would differ. Guards the
    pycox/torchtuples bounds in the ``[pycox]`` extra.
    """
    assert any(isinstance(m, torch.nn.Dropout) for m in model.net.modules())
    adapter = DeepHitAdapter(model)
    X, times = data["X"][:20], np.array([1.0, 5.0])
    model.net.train()
    first = adapter.predict_survival(X, times)
    np.testing.assert_array_equal(adapter.predict_survival(X, times), first)
    model.net.eval()
    np.testing.assert_array_equal(adapter.predict_survival(X, times), first)


@pytest.mark.parametrize("training", [False, True])
def test_network_mode_is_preserved(model: Any, data: dict[str, np.ndarray], training: bool) -> None:
    """torchtuples calls net.train() after predicting; the adapter restores the mode."""
    adapter = DeepHitAdapter(model)
    model.net.train(training)
    for _ in range(3):
        adapter.predict_survival(data["X"][:5], np.array([1.0]))
        assert model.net.training is training
    model.net.eval()


# --------------------------------------------------------------------- scaler


def test_scaler_equals_prescaled_inputs(model: Any, data: dict) -> None:
    # Exact in float32: float32 raw values, zero mean shift and a scale of 2.
    X = data["X"].astype(np.float32)
    exact = AffineScaler(np.zeros(P), np.full(P, 2.0))
    times = _query_times(model)
    np.testing.assert_array_equal(
        DeepHitAdapter(model, scaler=exact).predict_survival(X, times),
        DeepHitAdapter(model).predict_survival(exact.transform(X), times),
    )


def test_scaler_gives_the_same_importances(model: Any, data: dict) -> None:
    X = data["X"].astype(np.float32).astype(np.float64)
    exact = AffineScaler(np.zeros(P), np.full(P, 2.0))
    T, E = data["T"], data["E"]
    grid = np.quantile(T, [0.2, 0.4, 0.6])

    def run(adapter: DeepHitAdapter, Z: np.ndarray) -> np.ndarray:
        explainer = TimeSAGE(adapter, background=Z[:20], n_permutations=8, random_state=1)
        return explainer.explain(Z[20:80], T[20:80], E[20:80], grid).phi

    phi_raw = run(DeepHitAdapter(model, scaler=exact), X)
    np.testing.assert_array_equal(phi_raw, run(DeepHitAdapter(model), exact.transform(X)))
    assert np.abs(phi_raw).max() > 0


# ----------------------------------------------------------------- validation


def test_constructor_validation(model: Any) -> None:
    import torchtuples as tt
    from pycox.models import DeepHitSingle

    with pytest.raises(ValueError, match="predict_surv_df"):
        DeepHitAdapter(object())
    no_index = DeepHitSingle(tt.practical.MLPVanilla(P, [4], N_CUTS), tt.optim.Adam(1e-2))
    with pytest.raises(ValueError, match="duration_index"):
        DeepHitAdapter(no_index)


def test_warns_when_not_on_cpu(model: Any) -> None:
    class OnGPU:
        device = torch.device("cuda")
        duration_index = model.duration_index

        def predict_surv_df(self, X: np.ndarray) -> pd.DataFrame:  # pragma: no cover
            raise AssertionError

    with pytest.warns(UserWarning, match="device cuda") as record:
        DeepHitAdapter(OnGPU())
    assert record[0].filename == __file__  # points at the caller


@pytest.mark.parametrize(
    ("X", "times", "match"),
    [
        (np.ones(3), [1.0], "shape"),
        (np.array([[1.0, np.nan, 0.0]]), [1.0], "non-finite"),
        (np.ones((2, 3)), [[1.0]], "one-dimensional"),
        (np.ones((2, 3)), [-1.0], "non-negative"),
    ],
)
def test_predict_validation(model: Any, X: Any, times: Any, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        DeepHitAdapter(model).predict_survival(X, times)
