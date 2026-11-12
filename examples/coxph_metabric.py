"""TimeSAGE for a Cox model on METABRIC: the original study's 9-feature CoxPH analysis.

Usage::

    pip install "timesage-survival[examples]"
    python examples/coxph_metabric.py path/to/metabric_with_feature_names.csv
    python examples/coxph_metabric.py path/to/metabric_with_feature_names.csv --quick

You supply the METABRIC CSV (see ``DATA.md``; ODbL). Nothing is downloaded or
bundled. The example follows ``02_coxph.ipynb`` of the original research
implementation:

1. ``load_metabric`` (content-verified) and the fixed split ``metabric_split``.
2. ``lifelines.CoxPHFitter(penalizer=0.01)`` on the 1,188 training rows,
   with raw (unstandardised) features, as in the original study. Nothing else
   is fitted, so no scaler is involved; any preprocessing you add must be
   fitted on the training split only.
3. Background: 40 k-means centroids of the training rows, each coordinate
   moved to the nearest observed value (what ``shap.kmeans`` does; shap is
   not needed). **scikit-learn's KMeans result depends on the number of
   threads**: with one thread, one of the 40 centroids differs and the seed
   means below shift by up to about 0.01. The original study used all cores,
   so this example does too (scikit-learn's default). If you limit threads
   (``OMP_NUM_THREADS``, ``threadpoolctl``), expect small differences.
4. TimeSAGE on the training rows, 50 times from the first positive time to
   the 90th percentile, B = 2,048 permutations, seeds 0-9, no early stopping.
5. Integrated importance with the trapezoid rule (the variant this notebook
   used; the package default is ``censoring``).

**What it reproduces exactly** (checked against the notebook's saved
outputs):

- the seed means of trapezoid-integrated importance (``02_coxph`` cell 37,
  printed to 6 decimals);
- the mean pairwise Spearman between seeds, 0.907 (cell 31).

**LOFO comparison.** ``examples/data/coxph_lofo_summary.csv`` holds the
leave-one-feature-out (LOFO) results the original study's CoxPH notebook
compared against (feature-level aggregates only; provenance in
``examples/data/README.md``). They come from the original research
implementation and are **compared against, not recomputed** here. The
original study reported Spearman 0.73 (LOFO integrated or mean degradation)
and 0.77 (LOFO maximum degradation over time); this example reproduces both.

**Runtime.** About a minute: 47 s of TimeSAGE for 10 seeds at B = 2,048
(52 s in all) where this was checked (Windows, Python 3.11, 16-core CPU;
TimeSAGE is sequential, k-means and lifelines may use several cores).
``--quick`` runs 2 seeds at B = 128 in a few seconds: a demonstration only,
which does **not** reproduce the reference numbers.
"""

from __future__ import annotations

import argparse
import time
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from timesage import TimeSAGE
from timesage.adapters import CoxPHAdapter
from timesage.datasets import load_metabric, metabric_split

DATA = Path(__file__).resolve().parent / "data"

# 02_coxph cell 37: trapezoid-integrated importance, mean over seeds 0-9.
SAVED_SEED_MEANS = {
    "AGE_AT_DIAGNOSIS": 2.113004,
    "CHEMOTHERAPY": 1.017749,
    "MKI67": 0.348556,
    "PGR": 0.331872,
    "ERBB2": 0.265456,
    "HORMONE_THERAPY": 0.211341,
    "EGFR": 0.006170,
    "RADIO_THERAPY": -0.016120,
    "ER_POSITIVE": -0.057543,
}
SAVED_STABILITY = 0.9070  # 02_coxph cell 31, mean pairwise Spearman over 10 seeds

REFERENCE = {"n_permutations": 2048, "seeds": tuple(range(10))}
QUICK = {"n_permutations": 128, "seeds": (0, 1)}


def kmeans_background(X: np.ndarray, k: int = 40) -> np.ndarray:
    """``shap.kmeans(X, k).data``: k-means centroids snapped to observed values."""
    from sklearn.cluster import KMeans

    centres = KMeans(n_clusters=k, random_state=0, n_init=10).fit(X).cluster_centers_
    for j in range(X.shape[1]):
        nearest = np.argmin(np.abs(X[:, j][None, :] - centres[:, j][:, None]), axis=1)
        centres[:, j] = X[nearest, j]
    return centres.astype(np.float32)


def run(csv: str | Path, *, n_permutations: int, seeds: tuple[int, ...]) -> dict:
    """Fit, explain and compare; returns everything the report prints."""
    from lifelines import CoxPHFitter

    data = load_metabric(csv)  # verify="content": the original study's data
    train = metabric_split(data)["train"]
    X = data.X.iloc[train].reset_index(drop=True)
    T, E = data.time[train], data.event[train]
    names = list(X.columns)

    cph = CoxPHFitter(penalizer=0.01)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # lifelines convergence chatter
        cph.fit(X.assign(duration=T, event=E), duration_col="duration", event_col="event")

    background = kmeans_background(X.to_numpy(dtype=float))
    T_pos = T[T > 0]
    times = np.linspace(max(float(T_pos.min()), 1e-6), float(np.percentile(T, 90)), 50)

    integrated = []
    start = time.perf_counter()
    for seed in seeds:
        explainer = TimeSAGE(
            CoxPHAdapter(cph, feature_names=names),
            background=background,
            n_permutations=n_permutations,
            batch_size=64,
            random_state=seed,
        )
        result = explainer.explain(X, T, E, times)
        integrated.append(result.integrated("trapezoid").to_numpy())
    seconds = time.perf_counter() - start

    integ = np.array(integrated)
    means = pd.Series(integ.mean(axis=0), index=names)
    pairs = list(combinations(range(len(seeds)), 2))
    stability = float(np.mean([spearmanr(integ[i], integ[j])[0] for i, j in pairs]))
    lofo = pd.read_csv(DATA / "coxph_lofo_summary.csv", comment="#").set_index("feature")
    return {
        "seed_means": means,
        "stability": stability,
        "lofo_spearman": {
            c: float(spearmanr(means[names], lofo.loc[names, c])[0])
            for c in ("integrated_delta", "max_delta")
        },
        "seconds": seconds,
        "n_seeds": len(seeds),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("csv", help="metabric_with_feature_names.csv (user-supplied)")
    parser.add_argument(
        "--quick", action="store_true", help="2 seeds, B = 128; not the reference numbers"
    )
    args = parser.parse_args()
    budget = QUICK if args.quick else REFERENCE
    out = run(args.csv, **budget)

    if args.quick:
        print("QUICK MODE (2 seeds, B = 128): a demonstration only.")
        print("It does NOT reproduce the reference numbers.\n")
    n_perm = budget["n_permutations"]
    print(f"TimeSAGE: {out['n_seeds']} seeds x B = {n_perm}, {out['seconds']:.0f} s\n")
    means = out["seed_means"].sort_values(ascending=False)
    table = pd.DataFrame({"seed mean": means, "saved (cell 37)": pd.Series(SAVED_SEED_MEANS)})
    table["difference"] = table["seed mean"] - table["saved (cell 37)"]
    print("Trapezoid-integrated importance, mean over seeds:")
    print(table.round(6).to_string(), "\n")
    stab = out["stability"]
    print(f"Seed stability (mean pairwise Spearman): {stab:.4f}   saved: {SAVED_STABILITY}")
    print(
        "Spearman vs LOFO (compared against the shipped LOFO summary, not recomputed): "
        f"integrated {out['lofo_spearman']['integrated_delta']:.4f} (reference 0.73), "
        f"max {out['lofo_spearman']['max_delta']:.4f} (reference 0.77)"
    )
    if not args.quick:
        worst = float(table["difference"].abs().max())
        print(f"\nLargest difference from the saved seed means: {worst:.1e} (saved to 6 dp)")


if __name__ == "__main__":
    main()
