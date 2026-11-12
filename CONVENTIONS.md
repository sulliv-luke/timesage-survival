# Conventions and reproducing the reference implementation

TimeSAGE's numbers depend on conventions that survival-analysis software
handles in different ways. This page states the package's choices.

`legacy_parity=True`, available wherever the package's default differs,
reproduces the arithmetic of the original research implementation (the code
behind the dissertation in which the method was introduced), so results
computed with that implementation can be matched exactly. The default
follows the textbook definition or the fitted library's own estimate.

## IPCW Brier score

- **Censoring survival `G_hat`.** Kaplan–Meier with censoring as the event,
  fitted on the explained data (or on `censoring_time`/`censoring_event` if
  given), evaluated as a right-continuous step function.
- **Event weights use `G_hat(Y_i)`**, the value at the event time, as in the
  method's definition (Eq. 15). Graf et al. (1999) use the left limit
  `G_hat(Y_i−)`. The two differ only when a censoring and an event share a
  time. There is no option for the left limit.
- **`G_hat = 0` is an error.** If `G_hat` is 0 at any grid time, the IPCW
  weights are undefined and `explain` raises `ValueError`; shorten the grid.
- **`G_hat < 0.05` gives a warning** (once per call, listing the times):
  weights above 20 make late horizons noisy.

## At-risk fraction

`R_hat(t) = (1/n) Σ 1{Y_i ≥ t}` (the textbook risk set). It is the weight of
`integrate(method="at_risk")` and one of the two conditions of the stable
window. With `legacy_parity=True` the strict `1{Y_i > t}` is used. The two
differ only at grid times equal to an observed time.

## Integration

- `integrate(result, method)` approximates `∫ w(t) φ_j(t) dt` by
  `Σ_k w(t_k) φ_j(t_k) Δt_k`, with trapezoid weights `Δt_k` on the grid, from
  the first grid time (not from 0). `method` is `"censoring"` (`w = G_hat`,
  the default), `"at_risk"` (`w = R_hat`) or `"trapezoid"` (`w = 1`).
- **Windows.** With `window=` (for example `result.stable_window()`), the
  trapezoid weights are computed on the whole grid and set to zero outside
  the window. The window's first and last points keep half-intervals that
  reach outside it, so they are slightly overweighted: on the grid
  `1, 2, 4, 7, 11`, a window selecting `2, 4, 7` gives a constant curve of 1
  an area of 7.5, not 5.
- **Stable window** (Eq. 31): grid times with `R_hat(t) ≥ 0.10` and
  `G_hat(t) ≥ 0.05`.

## Inputs and randomness

- Features are cast to **float32** before they reach the model, for every
  model, as in the original implementation. Adapters that take `scaler=`
  apply it after this cast.
- All randomness comes from `random_state` through a `numpy.random.SeedSequence`
  tree; the same seed gives the same result. Exactly `n_permutations`
  permutations are used (fewer if early stopping is on).

## Model adapters: survival between and beyond the model's own times

Each adapter returns what its library's own estimate says.

| Adapter | Between the model's time points | Before the first | After the last |
|---|---|---|---|
| `CoxPHAdapter` (lifelines) | linear in the cumulative hazard, as lifelines interpolates | lifelines' value at the first training time (S = 1 if that time is a censoring) | constant |
| `CoxTimeVaryingAdapter` (lifelines) | step function (Breslow) | S = 1 | constant |
| `DeepSurvAdapter` (pycox) | step function, right-continuous | S = 1 | constant |
| `DeepHitAdapter` (pycox) | linear in S between cuts (pycox's default `const_pdf`) | — | constant at the last cut's value |

- **DeepHit is a discrete-time model.** It defines survival only at its cuts;
  the interpolation between them is a modelling choice. `S(0) = 1 − pmf_0` is
  below 1, because the network's softmax always gives the first cut some
  probability, so a grid that includes 0, or starts before the second cut,
  sees `S < 1` at early times. After the last cut, survival stays at the
  probability mass beyond it, which need not be 0.
- **pycox networks** are evaluated in `eval()` mode without gradients by
  pycox; the adapters restore the network's previous mode afterwards and
  warn if the model is not on the CPU.
- **`legacy_parity=True`** on `CoxTimeVaryingAdapter` and `DeepSurvAdapter`
  reproduces the original implementation's survival functions (for
  example, linear interpolation between time points and clamping to the
  first and last time).

## Synthetic data and oracle

- `make_synthetic(scenario, seed)` reproduces the original study's
  synthetic datasets. Their columns `event_time` and `censoring_time` are
  latent and must never be passed to `explain` or a model.
- `oracle_importance(data)` applies TimeSAGE to the **true** survival
  function of the data-generating process (`Scenario.predict_survival`). It
  is a Monte Carlo estimate over 4,096 sampled permutations, with standard
  errors in `phi_se`. With `legacy_parity=True` it reproduces the
  original implementation's oracle, which evaluates survival at the next
  breakpoint after each query time on a finer partition of the coefficient
  curves.

## Metrics

`timesage.metrics` returns **NaN with a warning** wherever a metric is
undefined: an empty window, a constant curve, a reference curve with no mass,
or (for IoU) no high-importance region inside the window. `legacy_parity=True`
returns the original implementation's numbers in those cases (for example 0
for an empty IoU).

**Spearman against an oracle.** Of the six synthetic features, x5 and x6
have exactly zero true importance and x4 almost zero. The rank correlation of
integrated importances is therefore driven largely by how features whose
true importance is about zero are ordered, which neither the estimate nor
the oracle determines. Curve correlation and NWIAE for x1, the feature with
a time-varying effect, are more informative measures of recovery.

## Tests that need data

Tests marked `needs_data` use files that are not in the repository. They
skip, naming the missing file, unless these environment variables point at
them:

- `TIMESAGE_METABRIC_CSV`: the 9-feature METABRIC CSV (see `DATA.md`);
- `TIMESAGE_CBIOPORTAL_DIR`: the cBioPortal `brca_metabric` folder, to
  rebuild that CSV with `scripts/metabric_from_cbioportal.py`;
- `TIMESAGE_SYNTHETIC_DIR`: parquet files written by the original synthetic
  study, compared with `make_synthetic`'s output.

`make_synthetic` is checked on every platform against event counts and column
sums of the original study's datasets. Its output can differ in the last bits
between machines (CPU-dependent linear-algebra kernels in
`multivariate_normal`), so the bit-exact SHA-256 check runs only with
`TIMESAGE_EXACT_DIGESTS=1`.

## k-means backgrounds

`examples/coxph_metabric.py` summarises the training data by 40 k-means
centroids, as the original study did. scikit-learn's `KMeans` result depends
on the number of threads: with one thread, one of the 40 centroids differs
and the seed means shift by up to about 0.01. The example uses all cores, as
the original study did.
