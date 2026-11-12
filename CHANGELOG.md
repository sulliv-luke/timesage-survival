# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- `TimeSAGE` estimator and `TimeSAGEResult`: time-indexed importance curves
  from the IPCW Brier score with marginal imputation and joint-times
  permutation sampling, with standard errors and an efficiency residual.
- `integrate` with `censoring` (default), `at_risk` and `trapezoid` weights,
  optionally over a window such as the stable window.
- `timesage.metrics`: stable window, curve correlation, NWIAE, IoU of
  high-importance regions, peak-time error and rank correlation.
- Adapters for lifelines `CoxPHFitter` and `CoxTimeVaryingFitter` and pycox
  DeepSurv and `DeepHitSingle`.
- `timesage.datasets`: synthetic scenarios with an oracle, and a loader for
  a user-supplied METABRIC file with its fixed split.
- `legacy_parity=True` options reproducing the original research
  implementation's arithmetic; see `CONVENTIONS.md`.
- Example: CoxPH on METABRIC (`examples/coxph_metabric.py`).
