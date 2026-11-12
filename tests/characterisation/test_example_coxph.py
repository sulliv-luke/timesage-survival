"""``examples/coxph_metabric.py`` reproduces the original study's CoxPH METABRIC outputs.

``needs_data``: reads the METABRIC CSV from ``$TIMESAGE_METABRIC_CSV`` and
skips when the variable is unset or the file is absent. Needs the
``[examples]`` extra (lifelines, scikit-learn).

- quick mode runs (a smoke test; it does not reproduce the reference numbers);
- at the reference budget (``slow``, about a minute) the seed means match
  the notebook's saved 6-decimal output, seed stability is 0.907, and the
  Spearmans against the shipped LOFO aggregate are 0.733 and 0.767.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
EXAMPLE = REPO / "examples" / "coxph_metabric.py"
CSV = Path(os.environ.get("TIMESAGE_METABRIC_CSV", "metabric_with_feature_names.csv"))

pytestmark = pytest.mark.needs_data


@pytest.fixture(scope="module")
def example() -> ModuleType:
    if "TIMESAGE_METABRIC_CSV" not in os.environ or not CSV.exists():
        pytest.skip(f"needs_data: missing {CSV} (set TIMESAGE_METABRIC_CSV)")
    pytest.importorskip("lifelines")
    pytest.importorskip("sklearn")
    spec = importlib.util.spec_from_file_location("_coxph_metabric_example", EXAMPLE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    dont_write = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = dont_write
    return module


def test_quick_mode_runs(example: ModuleType) -> None:
    out = example.run(CSV, **example.QUICK)
    assert out["n_seeds"] == 2
    assert set(out["seed_means"].index) == set(example.SAVED_SEED_MEANS)  # all 9 features


@pytest.mark.slow
def test_reference_budget_reproduces_saved_outputs(example: ModuleType) -> None:
    out = example.run(CSV, **example.REFERENCE)
    means = out["seed_means"]
    for feature, saved in example.SAVED_SEED_MEANS.items():
        assert abs(means[feature] - saved) <= 5e-7, feature  # saved to 6 decimals
    assert round(out["stability"], 4) == example.SAVED_STABILITY
    np.testing.assert_allclose(
        [out["lofo_spearman"]["integrated_delta"], out["lofo_spearman"]["max_delta"]],
        [0.7333, 0.7667],
        atol=5e-5,
    )
