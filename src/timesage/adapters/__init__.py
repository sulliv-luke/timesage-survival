"""Model adapters (public).

Each adapter translates a library's native survival output into the
``SurvivalModel`` interface and contains no attribution logic. Heavy
dependencies (lifelines, pycox/torch) are imported lazily inside each adapter.
"""

from timesage.adapters.coxph import CoxPHAdapter
from timesage.adapters.coxtv import CoxTimeVaryingAdapter
from timesage.adapters.deephit import DeepHitAdapter
from timesage.adapters.deepsurv import DeepSurvAdapter

__all__ = ["CoxPHAdapter", "CoxTimeVaryingAdapter", "DeepHitAdapter", "DeepSurvAdapter"]
