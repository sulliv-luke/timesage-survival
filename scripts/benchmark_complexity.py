"""Wall-clock benchmark: TimeSAGE vs leave-one-feature-out (LOFO) refitting.

Usage::

    python scripts/benchmark_complexity.py                 # p = 209, B = 768 (the headline cell)
    python scripts/benchmark_complexity.py --lofo          # ... plus the LOFO refits (~3 min)
    python scripts/benchmark_complexity.py --all --lofo    # the full original grid
    python scripts/benchmark_complexity.py --p 50 100 --B 256 768 --repeats 1

Needs lifelines (``pip install "timesage-survival[lifelines]"``). Not a test:
timings depend on the machine, so nothing is asserted. Use it before and after
performance-sensitive changes, on the same machine, and compare.

Setup (the original study's complexity benchmark, ``complexity_benchmark.ipynb``)
---------------------------------------------------------------------------------
- Data: scenario E1, 500 training and 200 test subjects (here the training
  and test splits of ``make_synthetic("E1", seed=42)``; the original drew one
  700-subject cohort from ``default_rng(42)``, so the values differ but the
  shapes and distribution are the same). For ``p > 6``, ``p - 6`` independent
  standard-normal noise features are appended, drawn from
  ``default_rng(42 + 10000 + p)``; for ``p = 5``, x6 is dropped. Features are
  not standardised.
- Model: lifelines ``CoxPHFitter(penalizer=0.05, l1_ratio=0.0)``, one per
  ``p``, fitted before timing (not timed).
- TimeSAGE: background of m = 32 training rows from
  ``default_rng(42 + p).choice(500, 32, replace=False)``; K = 20 times on
  ``linspace(12, 200, 20)``; batch 64; exactly B permutations (no early
  stopping); ``random_state = 42 + repeat``; the 200 test rows explained.
  Timed: ``TimeSAGE(...).explain(...)``.
- LOFO: p refits of the same CoxPH model, each without one feature. Timed:
  the refits only (the cost that grows with p).
- Each cell is the mean of ``--repeats`` runs (default 3).

The original timings (one machine, single process, 2026) are printed
alongside for reference. At p = 209: TimeSAGE B = 768 took 51.8 s, LOFO
174.9 s.

Raw seconds are not comparable across machines. With ``--lofo``, the script
also prints TimeSAGE time divided by LOFO time for the same ``p``: LOFO is
pure lifelines fitting, so the ratio largely cancels machine speed (though
not differences in lifelines or numpy versions). It is the number to compare
with the original (0.296 at p = 209, B = 768).
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import timesage
from timesage import TimeSAGE
from timesage.datasets.synthetic import make_synthetic

SEED = 42
N_BACKGROUND = 32
TIMES = np.linspace(12.0, 200.0, 20)
BATCH_SIZE = 64
PENALIZER = 0.05
BASE_FEATURES = ["x1", "x2", "x3", "x4", "x5", "x6"]
GRID_P = (5, 10, 20, 50, 100, 209)
GRID_B = (128, 256, 512, 768)

# Mean wall-clock seconds from the original benchmark (complexity_benchmark.ipynb).
REFERENCE_TIMESAGE: dict[tuple[int, int], float] = {
    (5, 128): 0.099, (5, 256): 0.168, (5, 512): 0.325, (5, 768): 0.450,
    (10, 128): 0.170, (10, 256): 0.293, (10, 512): 0.561, (10, 768): 0.825,
    (20, 128): 0.298, (20, 256): 0.559, (20, 512): 1.096, (20, 768): 1.636,
    (50, 128): 0.845, (50, 256): 1.679, (50, 512): 3.327, (50, 768): 5.016,
    (100, 128): 2.459, (100, 256): 4.737, (100, 512): 11.009, (100, 768): 17.111,
    (209, 128): 8.533, (209, 256): 17.258, (209, 512): 34.180, (209, 768): 51.757,
}  # fmt: skip
REFERENCE_LOFO: dict[int, float] = {
    5: 0.324, 10: 0.680, 20: 1.592, 50: 6.559, 100: 24.460, 209: 174.852,
}  # fmt: skip


def _features(
    p: int, X_train: np.ndarray, X_test: np.ndarray
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    if p < 5:
        raise SystemExit(f"p must be at least 5; got {p}.")
    if p == 5:
        return X_train[:, :5], X_test[:, :5], BASE_FEATURES[:5]
    cols = list(BASE_FEATURES)
    if p > 6:
        rng = np.random.default_rng(SEED + 10_000 + p)
        n_null = p - 6
        X_train = np.hstack([X_train, rng.standard_normal((len(X_train), n_null))])
        X_test = np.hstack([X_test, rng.standard_normal((len(X_test), n_null))])
        cols += [f"null_{i + 1}" for i in range(n_null)]
    return X_train, X_test, cols


def _fit_cox(X: np.ndarray, time_: np.ndarray, event: np.ndarray, cols: list[str]) -> object:
    from lifelines import CoxPHFitter

    df = pd.DataFrame(X, columns=cols)
    df["duration"] = time_
    df["event"] = event
    model = CoxPHFitter(penalizer=PENALIZER, l1_ratio=0.0)
    model.fit(df, duration_col="duration", event_col="event")
    return model


def _time_lofo(X: np.ndarray, time_: np.ndarray, event: np.ndarray, cols: list[str]) -> float:
    t0 = time.perf_counter()
    for j in range(len(cols)):
        keep = [i for i in range(len(cols)) if i != j]
        _fit_cox(X[:, keep], time_, event, [cols[i] for i in keep])
    return time.perf_counter() - t0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--p", type=int, nargs="+", default=[209], help="feature counts")
    parser.add_argument("--B", type=int, nargs="+", default=[768], help="permutation budgets")
    parser.add_argument("--all", action="store_true", help="the original grid of p and B")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--lofo", action="store_true", help="also time the LOFO refits")
    parser.add_argument("--json", type=Path, help="write the results to this file")
    args = parser.parse_args(argv)
    if args.repeats < 1:
        parser.error("--repeats must be at least 1.")
    try:
        import lifelines
    except ImportError:
        parser.error('needs lifelines: pip install "timesage-survival[lifelines]"')
    p_values = list(GRID_P) if args.all else args.p
    b_values = list(GRID_B) if args.all else args.B

    data = make_synthetic("E1", seed=SEED)
    X_train0, t_train, e_train = data.arrays("train")
    X_test0, t_test, e_test = data.arrays("test")

    print(
        f"timesage {timesage.__version__}, lifelines {lifelines.__version__}, "
        f"numpy {np.__version__}, Python {platform.python_version()}, {platform.platform()}"
    )
    print(
        f"m = {N_BACKGROUND}, K = {len(TIMES)}, n_explain = {len(X_test0)}, "
        f"repeats = {args.repeats}\n"
    )
    print(f"{'method':<9} {'p':>4} {'B':>5} {'seconds':>9} {'original':>9} {'ratio':>6}")

    records: list[dict[str, Any]] = []

    def report(method: str, p: int, B: int | None, seconds: float, ref: float | None) -> None:
        ratio = f"{seconds / ref:6.2f}" if ref else "     -"
        ref_s = f"{ref:9.2f}" if ref else "        -"
        b_s = f"{B:5d}" if B is not None else "    -"
        print(f"{method:<9} {p:4d} {b_s} {seconds:9.2f} {ref_s} {ratio}", flush=True)
        records.append(
            {"method": method, "p": p, "B": B, "mean_seconds": seconds, "original_seconds": ref}
        )

    for p in p_values:
        X_train, X_test, cols = _features(p, X_train0, X_test0)
        model = _fit_cox(X_train, t_train, e_train, cols)
        from timesage.adapters import CoxPHAdapter

        adapter = CoxPHAdapter(model, feature_names=cols)
        idx = np.random.default_rng(SEED + p).choice(len(X_train), N_BACKGROUND, replace=False)
        background = X_train[idx]

        if args.lofo:
            secs = [_time_lofo(X_train, t_train, e_train, cols) for _ in range(args.repeats)]
            report("LOFO", p, None, float(np.mean(secs)), REFERENCE_LOFO.get(p))

        for B in b_values:
            secs = []
            for rep in range(args.repeats):
                explainer = TimeSAGE(
                    adapter,
                    background=background,
                    n_permutations=B,
                    batch_size=BATCH_SIZE,
                    random_state=SEED + rep,
                )
                t0 = time.perf_counter()
                explainer.explain(X_test, t_test, e_test, TIMES, feature_names=cols)
                secs.append(time.perf_counter() - t0)
            report("TimeSAGE", p, B, float(np.mean(secs)), REFERENCE_TIMESAGE.get((p, B)))

    if args.lofo:
        print("\nTimeSAGE / LOFO time (machine-normalised):")
        lofo = {r["p"]: r["mean_seconds"] for r in records if r["method"] == "LOFO"}
        for r in records:
            if r["method"] != "TimeSAGE":
                continue
            p_, b_ = r["p"], r["B"]
            ratio = r["mean_seconds"] / lofo[p_]
            ref_ts, ref_lofo = REFERENCE_TIMESAGE.get((p_, b_)), REFERENCE_LOFO.get(p_)
            ref = f"{ref_ts / ref_lofo:.3f}" if ref_ts and ref_lofo else "-"
            print(f"  p = {p_:4d}, B = {b_:4d}: {ratio:.3f}   (original {ref})")

    if args.json is not None:
        args.json.write_text(json.dumps(records, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
