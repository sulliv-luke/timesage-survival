"""TimeSAGE: loss-calibrated, time-indexed global feature attribution for survival models.

TimeSAGE extends SAGE (Covert et al., 2020) to right-censored survival data. For
each horizon ``t`` on a time grid it defines a cooperative game whose value is
the negative IPCW Brier score of a fitted model (Eq. 11-12 of the dissertation)
and estimates each feature's Shapley value in that game.

Public API: :class:`TimeSAGE` (the estimator), :class:`TimeSAGEResult` (its
output), :func:`integrate` (integrated importance scalars, Eq. 20-24) and
:class:`SurvivalModel` (the model protocol).
"""

from timesage.estimator import TimeSAGE
from timesage.integrate import integrate
from timesage.protocols import SurvivalModel
from timesage.result import TimeSAGEResult

__version__ = "0.0.0.dev0"

__all__ = ["SurvivalModel", "TimeSAGE", "TimeSAGEResult", "__version__", "integrate"]
