"""``load_metabric`` and ``metabric_split`` on the original study's METABRIC CSV.

``needs_data``: reads the CSV from ``$TIMESAGE_METABRIC_CSV`` and skips when the
variable is unset or the file is absent. Copies written for the re-save checks
are deleted at once.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from importlib import resources
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from timesage.datasets.metabric import FEATURE_NAMES, load_metabric, metabric_split

CSV = Path(os.environ.get("TIMESAGE_METABRIC_CSV", "metabric_with_feature_names.csv"))
SPLITS = resources.files("timesage.datasets").joinpath("_data/metabric_split_indices.json")

pytestmark = pytest.mark.needs_data


@pytest.fixture(scope="module", autouse=True)
def _data_present() -> None:
    if "TIMESAGE_METABRIC_CSV" not in os.environ or not CSV.exists():
        pytest.skip(f"needs_data: missing {CSV} (set TIMESAGE_METABRIC_CSV)")


@pytest.fixture
def copy_of(tmp_path: Path) -> Iterator[Path]:
    path = tmp_path / "metabric_copy.csv"
    yield path
    path.unlink(missing_ok=True)  # never leave patient-level copies behind


def test_both_verifications_pass_and_values_match() -> None:
    reference = pd.read_csv(CSV, sep=",", comment="#")
    for verify in ("content", "file"):
        data = load_metabric(CSV, verify=verify)  # type: ignore[arg-type]
        assert data.verified == verify
        pd.testing.assert_frame_equal(data.X, reference[list(FEATURE_NAMES)])
        np.testing.assert_array_equal(data.time, reference["duration"])
        np.testing.assert_array_equal(data.event, reference["event"])
        assert data.patient_id.tolist() == reference["PATIENT_ID"].tolist()


def test_split_matches_the_stored_indices() -> None:
    stored = json.loads(SPLITS.read_text())
    split = metabric_split(load_metabric(CSV))
    for ours, theirs in (("train", "train_idx"), ("validation", "val_idx"), ("test", "test_idx")):
        np.testing.assert_array_equal(split[ours], stored[theirs])
    data = load_metabric(CSV)
    rates = [float(data.event[idx].mean()) for idx in split.values()]
    assert max(rates) - min(rates) < 0.005  # stratified on the event indicator


def test_resaved_copy_passes_content_but_not_file(copy_of: Path) -> None:
    copy_of.write_bytes(CSV.read_bytes().replace(b"\r\n", b"\n"))
    assert load_metabric(copy_of).verified == "content"
    with pytest.raises(ValueError, match="byte-identical"):
        load_metabric(copy_of, verify="file")


def test_changed_value_or_row_order_fails_content(copy_of: Path) -> None:
    df = pd.read_csv(CSV, sep=",", comment="#")
    df.loc[10, "AGE_AT_DIAGNOSIS"] += 0.01
    df.to_csv(copy_of, index=False)
    with pytest.raises(ValueError, match="differ from the original study"):
        load_metabric(copy_of)
    df = pd.read_csv(CSV, sep=",", comment="#").iloc[::-1]
    df.to_csv(copy_of, index=False)
    with pytest.raises(ValueError, match="row order"):
        load_metabric(copy_of)
