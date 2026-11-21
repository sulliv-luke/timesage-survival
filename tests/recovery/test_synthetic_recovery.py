"""Synthetic recovery: does TimeSAGE find the features that matter?

Unlike the characterisation tests, which check agreement with the original
research implementation, these check the method against the truth: scenario
E1 of the synthetic study, where x1 has a time-varying (logistic-hump) effect,
x2..x4 constant effects of decreasing size and x5, x6 none.

Setup (the study's main tier, one CoxPH cell per seed)
------------------------------------------------------
- Data: ``make_synthetic("E1", seed)`` for the ten study seeds 101..1010.
- Model: lifelines ``CoxPHFitter(penalizer=0.05, l1_ratio=0.0)`` fitted on the
  training split, features standardised with the training mean and standard
  deviation (``ddof=0``, as scikit-learn's ``StandardScaler``) and cast to
  float32.
- TimeSAGE: background of 96 training rows drawn with
  ``default_rng(seed).choice(500, 96, replace=False)``, B = 768 with
  convergence detection at 0.025, batch 64, ``random_state=seed``, the test
  split explained on the 24-point study grid (:func:`oracle_grid`).
- Reference: :func:`oracle_importance` with its default exact survival
  function (``legacy_parity=False``), which differs slightly from the oracle
  of the original research implementation.

Assertions and where their thresholds come from
-----------------------------------------------
The thresholds are set from the E1 CoxPH runs of the original research
implementation's synthetic pipeline (all ten seeds, re-run unchanged, against
its own oracle), with a margin. "Original" below refers to those runs. The
thresholds were fixed before this test was first run and were not tuned to
the package's output.

1. x1 ranks first in every seed. It did in all 30 original E1 runs (three
   models x ten seeds).
2. x5 and x6 rank below x2 and x3 in every seed. **Not** "x5 and x6 last":
   x4's true importance is essentially zero (oracle about 0.006 against 9 for
   x1), so its order relative to the noise features is a coin toss. The
   original runs put a noise feature above x4 in 3 of 10 E1 CoxPH seeds
   (101, 404, 1010).
3. ``|integrated(x5)|`` and ``|integrated(x6)|`` are below ``NOISE_FRACTION``
   (10%) of x1's. Original worst case: about 5% (seed 808).
4. The x1 curve correlation with the oracle on the stable window has a mean
   >= ``CURVE_CORR_MEAN`` (0.90) and a minimum >= ``CURVE_CORR_MIN`` (0.80).
   Original: mean 0.944, minimum 0.857.
5. The Spearman correlation of stable-window trapezoid integrals with the
   oracle's (the original study's ranking metric) has a mean >= ``SPEARMAN_MEAN`` (0.75).
   Original: mean 0.850, minimum 0.754. With six features a single swap
   costs about 0.03-0.09, so no per-seed floor is asserted.

Rankings (1-3) use the package's default integration (``censoring``).

Observed when first run (2026-11-13, exact oracle): curve correlation mean
0.943, minimum 0.844 (seed 202); Spearman mean 0.867, minimum 0.754; largest
noise share 4.8% (x5, seed 808). The censoring-weighted integrals agree with
the original runs' to the three decimals they were recorded at. The small
differences in curve correlation and Spearman come from the oracle.

Marked ``slow`` (several minutes, mostly the ten oracle runs); skipped when
lifelines is not installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from timesage import TimeSAGE, integrate
from timesage.datasets.synthetic import (
    FEATURE_NAMES,
    make_synthetic,
    oracle_grid,
    oracle_importance,
)
from timesage.metrics import curve_correlation, rank_correlation, stable_window

pytestmark = pytest.mark.slow

SEEDS = (101, 202, 303, 404, 505, 606, 707, 808, 909, 1010)

# Study settings (configs/defaults.yaml of the original synthetic study).
PENALIZER = 0.05
N_BACKGROUND = 96
N_PERMUTATIONS = 768
BATCH_SIZE = 64
CONVERGENCE_THRESHOLD = 0.025

# Thresholds; see the module docstring for their justification.
NOISE_FRACTION = 0.10
CURVE_CORR_MEAN = 0.90
CURVE_CORR_MIN = 0.80
SPEARMAN_MEAN = 0.75

NOISE = ("x5", "x6")
SIGNAL = ("x2", "x3")


@dataclass(frozen=True)
class Run:
    seed: int
    integrated: Any  # pd.Series, censoring-weighted, full grid
    curve_corr: float
    spearman: float


def _fit_and_explain(seed: int) -> Run:
    lifelines = pytest.importorskip("lifelines")
    import pandas as pd

    data = make_synthetic("E1", seed)
    names = list(FEATURE_NAMES)
    X_train, t_train, e_train = data.arrays("train")
    X_test, t_test, e_test = data.arrays("test")
    mean, std = X_train.mean(axis=0), X_train.std(axis=0)
    Z_train = ((X_train - mean) / std).astype(np.float32)
    Z_test = ((X_test - mean) / std).astype(np.float32)

    frame = pd.DataFrame(Z_train.astype(float), columns=names)
    frame["duration"] = t_train
    frame["event"] = e_train
    cph = lifelines.CoxPHFitter(penalizer=PENALIZER, l1_ratio=0.0)
    cph.fit(frame, duration_col="duration", event_col="event")

    from timesage.adapters import CoxPHAdapter

    rng = np.random.default_rng(seed)
    background = Z_train[rng.choice(len(Z_train), size=N_BACKGROUND, replace=False)]
    grid = oracle_grid(data)
    result = TimeSAGE(
        CoxPHAdapter(cph, feature_names=names),
        background=background,
        n_permutations=N_PERMUTATIONS,
        batch_size=BATCH_SIZE,
        convergence_threshold=CONVERGENCE_THRESHOLD,
        random_state=seed,
    ).explain(Z_test, t_test, e_test, grid, feature_names=names)

    oracle = oracle_importance(data, times=grid)
    window = stable_window(grid, t_test, e_test, censoring_time=t_train, censoring_event=e_train)
    assert np.array_equal(window, oracle.stable_window)
    x1 = names.index("x1")
    curve_corr = curve_correlation(result.phi[x1], oracle.phi[x1], window=window)
    stable_int = integrate(result, "trapezoid", window=window)
    spearman = rank_correlation(stable_int.to_numpy(), oracle.integrated.to_numpy())
    return Run(
        seed=seed,
        integrated=integrate(result, "censoring"),
        curve_corr=curve_corr,
        spearman=spearman,
    )


@pytest.fixture(scope="module")
def runs() -> list[Run]:
    out = [_fit_and_explain(seed) for seed in SEEDS]
    # Shown with -rA / -s; recorded in the test log for comparison with the
    # original runs' figures quoted in the module docstring.
    for r in out:
        print(
            f"seed {r.seed}: curve_corr={r.curve_corr:.3f} spearman={r.spearman:.3f} "
            f"integrated={np.round(r.integrated.to_numpy(), 3).tolist()}"
        )
    return out


def test_x1_ranks_first(runs: list[Run]) -> None:
    for r in runs:
        assert r.integrated.idxmax() == "x1", (r.seed, r.integrated.to_dict())


def test_noise_ranks_below_signal(runs: list[Run]) -> None:
    for r in runs:
        lowest_signal = min(r.integrated[f] for f in SIGNAL)
        for f in NOISE:
            assert r.integrated[f] < lowest_signal, (r.seed, f, r.integrated.to_dict())


def test_noise_is_small(runs: list[Run]) -> None:
    for r in runs:
        bound = NOISE_FRACTION * r.integrated["x1"]
        for f in NOISE:
            assert abs(r.integrated[f]) < bound, (r.seed, f, r.integrated.to_dict())


def test_x1_curve_matches_oracle(runs: list[Run]) -> None:
    corr = np.array([r.curve_corr for r in runs])
    assert np.all(np.isfinite(corr)), corr
    assert corr.mean() >= CURVE_CORR_MEAN, corr
    assert corr.min() >= CURVE_CORR_MIN, corr


def test_ranking_matches_oracle(runs: list[Run]) -> None:
    rho = np.array([r.spearman for r in runs])
    assert np.all(np.isfinite(rho)), rho
    assert rho.mean() >= SPEARMAN_MEAN, rho
