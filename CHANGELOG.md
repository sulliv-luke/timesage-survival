# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

Not yet released on PyPI.

### Added
- `TimeSAGE` estimator: time-indexed importance curves from the IPCW Brier
  score, with marginal imputation from a background sample and joint-times
  permutation sampling. Options for a background subsample
  (`n_background`), optional early stopping (`convergence_threshold`), a
  progress callback and a single `random_state` for reproducible results.
- `TimeSAGEResult`: the importance matrix `phi` with Monte Carlo standard
  errors `phi_se`, the censoring survival and at-risk fraction on the grid,
  `efficiency_residual()`, `stable_window()`, `integrated()` and
  `to_frame()`.
- `integrate` with `censoring` (default), `at_risk` and `trapezoid` weights,
  optionally over a window such as the stable window.
- `SurvivalModel` protocol: any object with
  `predict_survival(X, times)`, or a plain function with that signature,
  can be explained.
- Adapters for lifelines `CoxPHFitter` and `CoxTimeVaryingFitter` and pycox
  DeepSurv (`CoxPH`) and `DeepHitSingle`, with an optional `scaler=`.
- `timesage.metrics`: stable window, curve correlation, NWIAE, IoU of
  high-importance regions, peak-time error and rank correlation. Undefined
  values return NaN with a warning.
- `timesage.datasets`: synthetic scenarios with a known time-varying effect
  (`make_synthetic`) and their oracle importance (`oracle_importance`), and
  a loader for a user-supplied 9-feature METABRIC file (`load_metabric`)
  with its fixed train/validation/test split (`metabric_split`).
- `legacy_parity=True` options reproducing the original research
  implementation's arithmetic where the package's defaults differ.
- Input validation: clear errors for shape mismatches, invalid event
  indicators, grids beyond the observed follow-up, `G_hat = 0` on the grid
  and invalid model output; a warning when `G_hat < 0.05` on the grid.
- Example: CoxPH on METABRIC (`examples/coxph_metabric.py`), and
  `scripts/metabric_from_cbioportal.py` to build its input file from the
  cBioPortal release.
- Documentation: `README.md`, `CONVENTIONS.md` and `DATA.md`.
- Optional extras `[lifelines]`, `[pycox]` and `[examples]`. Type
  information is included (`py.typed`).
- Tested on Python 3.10–3.12 (Linux) and 3.11 (Windows).
