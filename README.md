# timesage-survival

**Status: alpha.** The API may change before 1.0.

TimeSAGE is a global feature-attribution method for survival models. It
extends SAGE (Covert et al., 2020) to right-censored data: at each time on a
grid it attributes a fitted model's IPCW Brier score to its input features,
giving one importance curve per feature. It explains a model's **predictive
performance**, not its individual predictions. Positive importance at time
`t` means that knowing the feature reduces the model's prediction error at
`t`; negative values are possible and are not clipped.

## Installation

Not yet on PyPI. Until the first release, install from a clone of this
repository (`pip install .`); after it:

```bash
pip install timesage-survival                 # core: numpy, pandas, scipy
pip install "timesage-survival[lifelines]"    # CoxPH and time-varying Cox adapters
pip install "timesage-survival[pycox]"        # DeepSurv and DeepHit adapters (PyTorch)
pip install "timesage-survival[examples]"     # what examples/ needs
```

The import name is `timesage`. Python 3.10 or later.

## Quick start

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

Any model works if it provides `predict_survival(X, times)` returning
survival probabilities of shape `(n_samples, n_times)`; a plain function with
that signature works too. Adapters are included for lifelines `CoxPHFitter`
and `CoxTimeVaryingFitter`, and pycox DeepSurv (`CoxPH`) and `DeepHitSingle`.

## Reproducing the CoxPH METABRIC analysis

`examples/coxph_metabric.py` runs TimeSAGE on a Cox model of the 9-feature
METABRIC data (10 seeds, 2,048 permutations, about a minute). METABRIC is not
included; see [DATA.md](DATA.md) for the source, licence and how to build the
file. With that file the example reproduces the original study's CoxPH
results exactly:

- the seed means of trapezoid-integrated importance, led by
  AGE_AT_DIAGNOSIS (2.113), CHEMOTHERAPY (1.018), MKI67 (0.349) and PGR
  (0.332), equal to the original output's six printed decimals;
- the mean pairwise Spearman correlation between seeds, 0.907.

`--quick` runs a reduced budget in a few seconds; it does not reproduce these
numbers.

## Conventions

Survival estimators differ in conventions that change numbers without
changing results qualitatively. The package's choices, and how
`legacy_parity=True` reproduces the original research implementation's
arithmetic, are described in [CONVENTIONS.md](CONVENTIONS.md).

## Limitations

- **Marginal imputation.** Removed features are replaced by draws from a
  background sample, independently of the retained features. Importance
  measures the fitted model's dependence on each input; predictions are
  evaluated off the data manifold, and correlated features need not share
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
  proportional hazards; not on crossing hazards or competing risks.
- **Budget.** The estimate is a Monte Carlo average over sampled
  permutations (`result.phi_se` gives standard errors); fine-grained
  rankings, especially with many features, need large budgets.
- **Privacy.** Time-indexed attribution curves are derived from the
  explained data and can be sensitive.

## Data

No data are bundled or downloaded. See [DATA.md](DATA.md).

## Citation

The original research can be accessed [here](https://drive.google.com/file/d/1frWXf60dzRwxJ_MNlHRtz0w7xC42irFq/view?usp=sharing). If mentioning TimeSAGE in your own work, please reference as:

> Sullivan, L. (2026). *TimeSAGE: A Loss-Calibrated Global Feature
> Attribution Method for Survival Models.* MSc dissertation, Trinity College
> Dublin.

## Licence

MIT. See [LICENSE](LICENSE). Portions of `sage-importance` 0.0.6 (MIT) are
vendored in `timesage/_vendor/`.
