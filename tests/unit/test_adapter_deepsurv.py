"""Unit tests for ``timesage.adapters.deepsurv.DeepSurvAdapter``.

The default must return pycox's own survival, a right-continuous step
function on the training times, and satisfy the adapter contract
(shape, values in [0, 1], non-increasing in time, any query times).
``legacy_parity=True`` interpolates linearly.
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

from timesage.adapters import DeepSurvAdapter  # noqa: E402

P = 3


def _data(seed: int = 0, n: int = 150) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, P)) * 2.0 + 5.0  # raw scale, not standardised
    T = rng.exponential(10.0 / np.exp(0.6 * (X[:, 0] - 5.0)))
    C = rng.uniform(0.0, 30.0, size=n)
    return {"X": X, "T": np.minimum(T, C), "E": (T <= C).astype(np.float32)}


def fit_deepsurv(X: np.ndarray, T: np.ndarray, E: np.ndarray, seed: int = 0) -> Any:
    import torchtuples as tt
    from pycox.models import CoxPH

    torch.manual_seed(seed)
    net = tt.practical.MLPVanilla(X.shape[1], [8], 1, batch_norm=True, dropout=0.1)
    model = CoxPH(net, tt.optim.Adam(1e-2))
    X32 = X.astype(np.float32)
    model.fit(X32, (T, E), batch_size=32, epochs=5, verbose=False)
    model.compute_baseline_hazards(X32, (T, E))
    return model


@pytest.fixture(scope="module")
def data() -> dict[str, np.ndarray]:
    return _data()


@pytest.fixture(scope="module")
def scaler(data: dict[str, np.ndarray]) -> AffineScaler:
    return AffineScaler(data["X"].mean(axis=0), data["X"].std(axis=0))


@pytest.fixture(scope="module")
def model(data: dict[str, np.ndarray], scaler: AffineScaler) -> Any:
    return fit_deepsurv(scaler.transform(data["X"]), data["T"], data["E"])


def _query_times(T: np.ndarray) -> np.ndarray:
    obs = np.sort(T)
    mids = 0.5 * (obs[:-1] + obs[1:])
    return np.unique(np.concatenate([[0.0, 0.5 * obs[0]], obs[::5], mids[::3], [obs[-1] * 1.5]]))


# ---------------------------------------------------------------- faithfulness


def test_returns_pycox_step_function(model: Any, data: dict[str, np.ndarray], scaler: Any) -> None:
    X = scaler.transform(data["X"]).astype(np.float32)
    surv = model.predict_surv_df(X)
    grid, S_grid = surv.index.to_numpy(dtype=float), surv.to_numpy(dtype=float).T
    adapter = DeepSurvAdapter(model)
    # At pycox's own time points: exactly pycox.
    np.testing.assert_array_equal(adapter.predict_survival(X, grid), S_grid)
    # Between them: the value at the previous time point (right-continuous step).
    mids = 0.5 * (grid[:-1] + grid[1:])
    np.testing.assert_array_equal(adapter.predict_survival(X, mids), S_grid[:, :-1])
    # Before the first: 1. After the last: constant.
    S_edges = adapter.predict_survival(X, np.array([0.0, 0.5 * grid[0], 2.0 * grid[-1]]))
    np.testing.assert_array_equal(S_edges[:, :2], 1.0)
    np.testing.assert_array_equal(S_edges[:, 2], S_grid[:, -1])


def test_legacy_parity_interpolates_linearly(
    model: Any, data: dict[str, np.ndarray], scaler: Any
) -> None:
    X = scaler.transform(data["X"]).astype(np.float32)
    surv = model.predict_surv_df(X)
    grid, S_grid = surv.index.to_numpy(dtype=float), surv.to_numpy(dtype=float).T
    mids = 0.5 * (grid[:-1] + grid[1:])
    S = DeepSurvAdapter(model, legacy_parity=True).predict_survival(X, mids)
    np.testing.assert_allclose(S, 0.5 * (S_grid[:, :-1] + S_grid[:, 1:]), rtol=1e-12)
    # Clamped before the first time point: the value there, not 1.
    S0 = DeepSurvAdapter(model, legacy_parity=True).predict_survival(X, np.array([0.0]))
    np.testing.assert_array_equal(S0[:, 0], S_grid[:, 0])


# ------------------------------------------------------------------- contract


@pytest.mark.parametrize("legacy_parity", [False, True])
def test_output_contract(
    model: Any, data: dict[str, np.ndarray], scaler: Any, legacy_parity: bool
) -> None:
    times = _query_times(data["T"])
    order = np.random.default_rng(0).permutation(len(times))
    adapter = DeepSurvAdapter(model, scaler=scaler, legacy_parity=legacy_parity)
    X = data["X"].astype(np.float32)  # as TimeSAGE passes it
    S = adapter.predict_survival(X, times)
    assert S.shape == (len(X), len(times))
    assert S.dtype == np.float64
    assert np.all((S >= 0) & (S <= 1))
    assert np.all(np.diff(S, axis=1) <= 0)
    assert np.all(S[:, -1] == S[:, -2])  # constant after the last training time
    np.testing.assert_array_equal(adapter.predict_survival(X, times[order]), S[:, order])


def test_predictions_do_not_depend_on_network_mode(model: Any, data: dict[str, np.ndarray]) -> None:
    """Contract: pycox predicts in eval() mode whatever mode the net is in.

    With dropout present, a prediction in training mode would be random, so
    repeated calls on a net in training mode would differ. Guards the
    pycox/torchtuples bounds in the ``[pycox]`` extra.
    """
    assert any(isinstance(m, torch.nn.Dropout) for m in model.net.modules())
    adapter = DeepSurvAdapter(model)
    X, times = data["X"][:20], np.array([1.0, 5.0])
    model.net.train()
    first = adapter.predict_survival(X, times)
    np.testing.assert_array_equal(adapter.predict_survival(X, times), first)
    model.net.eval()
    np.testing.assert_array_equal(adapter.predict_survival(X, times), first)


@pytest.mark.parametrize("training", [False, True])
def test_network_mode_is_preserved(model: Any, data: dict[str, np.ndarray], training: bool) -> None:
    """torchtuples calls net.train() after predicting; the adapter restores the mode."""
    adapter = DeepSurvAdapter(model)
    model.net.train(training)
    for _ in range(3):
        adapter.predict_survival(data["X"][:5], np.array([1.0]))
        assert model.net.training is training
    model.net.eval()


# --------------------------------------------------------------------- scaler


def test_scaler_equals_prescaled_inputs(model: Any, data: dict[str, np.ndarray]) -> None:
    # Exact in float32: float32 raw values, zero mean shift and a scale of 2.
    X = data["X"].astype(np.float32)
    exact = AffineScaler(np.zeros(P), np.full(P, 2.0))
    times = _query_times(data["T"])
    np.testing.assert_array_equal(
        DeepSurvAdapter(model, scaler=exact).predict_survival(X, times),
        DeepSurvAdapter(model).predict_survival(exact.transform(X), times),
    )


def test_scaler_gives_the_same_importances(model: Any, data: dict[str, np.ndarray]) -> None:
    X = data["X"].astype(np.float32).astype(np.float64)
    exact = AffineScaler(np.zeros(P), np.full(P, 2.0))
    T, E = data["T"], data["E"].astype(int)
    grid = np.quantile(T, [0.2, 0.4, 0.6])

    def run(adapter: DeepSurvAdapter, Z: np.ndarray) -> np.ndarray:
        explainer = TimeSAGE(adapter, background=Z[:20], n_permutations=8, random_state=1)
        return explainer.explain(Z[20:80], T[20:80], E[20:80], grid).phi

    phi_raw = run(DeepSurvAdapter(model, scaler=exact), X)
    phi_scaled = run(DeepSurvAdapter(model), exact.transform(X))
    np.testing.assert_array_equal(phi_raw, phi_scaled)
    assert np.abs(phi_raw).max() > 0


# ----------------------------------------------------------------- validation


def test_constructor_validation(data: dict[str, np.ndarray]) -> None:
    import torchtuples as tt
    from pycox.models import CoxPH

    with pytest.raises(ValueError, match="predict_surv_df"):
        DeepSurvAdapter(object())
    unfitted = CoxPH(tt.practical.MLPVanilla(P, [4], 1), tt.optim.Adam(1e-2))
    with pytest.raises(ValueError, match="compute_baseline_hazards"):
        DeepSurvAdapter(unfitted)


def test_warns_when_not_on_cpu(model: Any) -> None:
    class OnGPU:
        device = torch.device("cuda")
        baseline_hazards_ = model.baseline_hazards_

        def predict_surv_df(self, X: np.ndarray) -> pd.DataFrame:  # pragma: no cover
            raise AssertionError

    with pytest.warns(UserWarning, match="device cuda") as record:
        DeepSurvAdapter(OnGPU())
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
        DeepSurvAdapter(model).predict_survival(X, times)
