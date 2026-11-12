"""Pure-numpy survival models shared by the tests and the golden generators.

The golden generators run in a separate environment, so this module must
depend on numpy only.
"""

from __future__ import annotations

import numpy as np

# Feature 3 is ignored by the model, so its TimeSAGE importance must be exactly 0.
TOY_BETA = np.array([0.8, -0.5, 0.3, 0.0])
TOY_INTERACTION = 0.4  # coefficient on x0 * x1, so the model is not additive


def toy_predict_survival(X: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Weibull-type proportional hazards with an interaction.

    ``S(t | x) = exp(-(t / 50) ** 1.3 * exp(x @ beta + 0.4 * x0 * x1))``.
    Returns shape ``(n, len(times))``, in ``[0, 1]`` and non-increasing in time.
    """
    X = np.asarray(X)
    t = np.asarray(times, dtype=float)
    eta = X @ TOY_BETA + TOY_INTERACTION * X[:, 0] * X[:, 1]
    return np.exp(-((t[None, :] / 50.0) ** 1.3) * np.exp(eta)[:, None])
