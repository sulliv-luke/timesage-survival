<!-- PROJECT HEADER -->
<br />
<p align="center">
  <h3 align="center">TimeSAGE</h3>

  <p align="center">
    Time-indexed, loss-calibrated global feature attribution for survival models.
    <br />
    <a href="https://drive.google.com/file/d/1frWXf60dzRwxJ_MNlHRtz0w7xC42irFq/view?usp=sharing"><strong>Read the research »</strong></a>
    <br />
    <br />
    <a href="https://github.com/sulliv-luke/timesage-survival#usage">Quick start</a>
    ·
    <a href="https://github.com/sulliv-luke/timesage-survival/issues">Bug Reports/Feature Requests</a>
  </p>
</p>

<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About the project</a>
      <ul>
        <li><a href="#built-with">Built with</a></li>
        <li><a href="#limitations">Limitations</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting started</a>
      <ul>
        <li><a href="#dependencies">Dependencies</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li>
      <a href="#usage">Usage</a>
      <ul>
        <li><a href="#quick-start">Quick start</a></li>
        <li><a href="#explaining-your-own-model">Explaining your own model</a></li>
        <li><a href="#reproducing-the-coxph-metabric-analysis">Reproducing the CoxPH METABRIC analysis</a></li>
        <li><a href="#conventions">Conventions</a></li>
      </ul>
    </li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#citation">Citation</a></li>
    <li><a href="#licence">Licence</a></li>
    <li><a href="#authors">Authors</a></li>
    <li><a href="#acknowledgements">Acknowledgements</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About the project

TimeSAGE is a global feature-attribution method for survival models. It
extends SAGE (Covert et al., 2020) to right-censored data: at each time on a
grid it attributes a fitted model's IPCW Brier score to its input features,
giving one importance curve per feature. It explains a model's **predictive
performance**, not its individual predictions. Positive importance at time
`t` means that knowing the feature reduces the model's prediction error at
`t`; negative values are possible and are not clipped.

Each curve can be collapsed to a single score per feature for ranking, with
weights that down-weight sparse late horizons (`censoring`, the default, or
`at_risk`) or none (`trapezoid`).

The method was introduced in my MSc dissertation at Trinity College Dublin:

[Sullivan, L. (2026). *TimeSAGE: A Loss-Calibrated Global Feature Attribution Method for Survival Models.*](https://drive.google.com/file/d/1frWXf60dzRwxJ_MNlHRtz0w7xC42irFq/view?usp=sharing)

This package reimplements the original research code and is tested against
it. Where its defaults differ, `legacy_parity=True` reproduces the original
arithmetic (see [Conventions](#conventions)).

### Built with

* [NumPy](https://numpy.org), [pandas](https://pandas.pydata.org) and
  [SciPy](https://scipy.org): the only required dependencies.
* [sage-importance](https://github.com/iancovert/sage) 0.0.6 (MIT): its
  marginal imputer and importance tracker are vendored in
  `timesage/_vendor/`, so it is not a dependency.
* Optional adapters for [lifelines](https://lifelines.readthedocs.io) and
  [pycox](https://github.com/havakv/pycox), and plotting with
  [Matplotlib](https://matplotlib.org).

### Limitations

- **Marginal imputation.** Removed features are replaced by draws from a
  background sample, independently of the retained features. Importance
  measures the fitted model's dependence on each input. Predictions are
  evaluated off the data manifold, and correlated features may not share
  credit evenly.
- **Independent censoring.** The censoring distribution is estimated by
  Kaplan–Meier without covariates, so the IPCW weights assume censoring is
  independent of both the event times and the features. Informative or very
  heavy censoring makes the weights, and the importances, unreliable.
- **Late horizons are noisy.** The risk set shrinks and IPCW weights grow;
  restrict to the stable window (`result.stable_window()`).
- **Timing is weakly localised.** The curves recover where a feature matters
  only roughly; do not read precise peak times from them.
- **Validation.** Checked only on synthetic scenarios compatible with
  proportional hazards, not on crossing hazards or competing risks.
- **Budget.** The estimate is a Monte Carlo average over sampled
  permutations.  Fine-grained rankings, especially with many features, need large budgets.
- **Privacy.** Time-indexed attribution curves are derived from the
  explained data and can be sensitive.



<!-- GETTING STARTED -->
## Getting started

### Dependencies

Python 3.10 or later. The core package needs only:

* numpy >= 1.26
* pandas >= 2.0
* scipy >= 1.11

Optional extras add what the adapters, plots and examples need:

| Extra | Installs | For |
|---|---|---|
| `[lifelines]` | lifelines >= 0.27 | CoxPH and time-varying Cox adapters |
| `[pycox]` | pycox 0.3, torchtuples 0.2, torch >= 2.0 | DeepSurv and DeepHit adapters |
| `[plot]` | matplotlib >= 3.7 | `timesage.plotting` |
| `[examples]` | scikit-learn >= 1.3, lifelines >= 0.27 | the scripts in `examples/` |

### Installation

From PyPI (the import name is `timesage`):

```bash
pip install timesage-survival                 # core
pip install "timesage-survival[lifelines]"    # with the Cox adapters
pip install "timesage-survival[lifelines,plot]"
```

From source:

1. Clone the repository
   ```bash
   git clone https://github.com/sulliv-luke/timesage-survival.git
   cd timesage-survival
   ```
2. Install it, with the development tools if you want to run the tests
   ```bash
   pip install -e ".[dev,lifelines,plot]"
   ```



<!-- USAGE EXAMPLES -->
## Usage

### Quick start

```python
import numpy as np
from lifelines import CoxPHFitter

from timesage import TimeSAGE
from timesage.adapters import CoxPHAdapter
from timesage.datasets import make_synthetic

data = make_synthetic("E1", seed=101)  # simulated; x1 has a time-varying effect
cph = CoxPHFitter(penalizer=0.05).fit(
    data.train.drop(columns=["event_time", "censoring_time"]),
    duration_col="time",
    event_col="event",
)

X, T, E = data.arrays("test")
background, _, _ = data.arrays("train")
times = np.linspace(10, 180, 18)

explainer = TimeSAGE(
    CoxPHAdapter(cph), background=background[:96], n_permutations=768, random_state=0
)
result = explainer.explain(X, T, E, times, feature_names=data.feature_names)

result.phi  # (6 features, 18 times)
result.integrated()  # censoring-weighted integral, by feature
result.integrated(window=result.stable_window())
```

With the `[plot]` extra, `timesage.plotting.plot_importance_curves(result)`
draws the curves with the stable window shaded, and
`timesage.plotting.plot_integrated(result)` a bar chart of the integrals.

### Explaining your own model

Any model works if it provides `predict_survival(X, times)` returning
survival probabilities of shape `(n_samples, n_times)`; a plain function with
that signature works too. Adapters are included for lifelines `CoxPHFitter`
and `CoxTimeVaryingFitter`, and pycox DeepSurv (`CoxPH`) and `DeepHitSingle`.

### Reproducing the CoxPH METABRIC analysis

`examples/coxph_metabric.py` runs TimeSAGE on a Cox model of the 9-feature
METABRIC data (10 seeds, 2,048 permutations, about a minute). No data are
bundled or downloaded: see
[DATA.md](https://github.com/sulliv-luke/timesage-survival/blob/main/DATA.md)
for the source, licence and how to build the file. With that file the
example reproduces the original study's CoxPH results exactly:

- the seed means of trapezoid-integrated importance, led by
  AGE_AT_DIAGNOSIS (2.113), CHEMOTHERAPY (1.018), MKI67 (0.349) and PGR
  (0.332), equal to the original output's six printed decimals;
- the mean pairwise Spearman correlation between seeds, 0.907.

`--quick` runs a reduced budget in a few seconds; it does not reproduce these
numbers.

### Conventions

Survival estimators differ in conventions that change numbers without
changing results qualitatively. The package's choices, and how
`legacy_parity=True` reproduces the original research implementation's
arithmetic, are described in
[CONVENTIONS.md](https://github.com/sulliv-luke/timesage-survival/blob/main/CONVENTIONS.md).



<!-- ROADMAP -->
## Roadmap

See the [open issues](https://github.com/sulliv-luke/timesage-survival/issues)
for proposed features and known issues. Changes are listed in
[CHANGELOG.md](https://github.com/sulliv-luke/timesage-survival/blob/main/CHANGELOG.md).



<!-- CONTRIBUTING -->
## Contributing

Bug reports, questions and pull requests are welcome. For anything beyond a
small fix, please open an issue first to discuss it.

1. Fork the project
2. Create a feature branch (`git checkout -b feature/my-change`)
3. Install the development tools (`pip install -e ".[dev,lifelines,plot]"`)
4. Make your change, with tests, and check that these pass:
   ```bash
   pytest -q
   ruff check . && ruff format --check .
   mypy src
   ```
5. Commit your changes and push the branch
6. Open a pull request



<!-- CITATION -->
## Citation

If you use TimeSAGE in your work, please cite:

> Sullivan, L. (2026). *TimeSAGE: A Loss-Calibrated Global Feature
> Attribution Method for Survival Models.* MSc dissertation, Trinity College
> Dublin.



<!-- LICENCE -->
## Licence

Distributed under the MIT Licence. See
[LICENSE](https://github.com/sulliv-luke/timesage-survival/blob/main/LICENSE)
for more information. Portions of `sage-importance` 0.0.6 (MIT) are vendored
in `timesage/_vendor/`.



<!-- AUTHORS -->
## Authors

Luke Sullivan - [@sulliv-luke](https://github.com/sulliv-luke)

Project link: [https://github.com/sulliv-luke/timesage-survival](https://github.com/sulliv-luke/timesage-survival)



<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements

* Ian Covert, Scott Lundberg and Su-In Lee, for SAGE and
  [sage-importance](https://github.com/iancovert/sage)
* The METABRIC study (Curtis et al., 2012), distributed by
  [cBioPortal](https://www.cbioportal.org)
* [lifelines](https://lifelines.readthedocs.io) and
  [pycox](https://github.com/havakv/pycox)
* README layout adapted from
  [README-template](https://github.com/catiaspsilva/README-template) by Catia Silva
