"""Unit tests for ``timesage.datasets.metabric`` on a fake, randomly generated CSV.

Checks against the real data are in ``tests/characterisation/test_metabric.py``
(``needs_data``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from timesage.datasets.metabric import (
    FEATURE_NAMES,
    N_ROWS,
    MetabricData,
    load_metabric,
    metabric_split,
)


def _fake(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"PATIENT_ID": [f"FAKE-{i:04d}" for i in range(N_ROWS)]})
    for col in FEATURE_NAMES:
        if col in ("HORMONE_THERAPY", "RADIO_THERAPY", "CHEMOTHERAPY", "ER_POSITIVE"):
            df[col] = rng.integers(0, 2, N_ROWS)
        else:
            df[col] = rng.normal(6.0, 1.0, N_ROWS)
    df["duration"] = rng.exponential(100.0, N_ROWS)
    df["event"] = rng.integers(0, 2, N_ROWS)
    return df


def _write(df: pd.DataFrame, tmp_path: Path) -> Path:
    path = tmp_path / "metabric.csv"
    df.to_csv(path, index=False)
    return path


def test_loads_structure_without_verification(tmp_path: Path) -> None:
    df = _fake()
    data = load_metabric(_write(df, tmp_path), verify=None)
    assert isinstance(data, MetabricData)
    assert list(data.X.columns) == list(FEATURE_NAMES)
    assert "PATIENT_ID" not in data.X.columns
    np.testing.assert_array_equal(data.time, df["duration"])
    np.testing.assert_array_equal(data.event, df["event"])
    assert data.event.dtype == np.int64
    assert data.patient_id.tolist() == df["PATIENT_ID"].tolist()
    assert data.verified is None


def test_fake_data_fail_both_hashes(tmp_path: Path) -> None:
    path = _write(_fake(), tmp_path)
    with pytest.raises(ValueError, match="differ from the original study"):
        load_metabric(path)  # default verify="content"
    with pytest.raises(ValueError, match="byte-identical"):
        load_metabric(path, verify="file")
    with pytest.raises(ValueError, match="verify must be"):
        load_metabric(path, verify="yes")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("change", "match"),
    [
        (lambda d: d[["PGR", "EGFR", *d.columns[3:]]].copy(), "Unexpected columns"),
        (lambda d: d.drop(columns=["MKI67"]), "Unexpected columns"),
        (lambda d: d.iloc[:-1], "Expected 1980 rows"),
        (lambda d: d.assign(PGR=d["PGR"].where(d.index != 5)), "missing values"),
        (lambda d: d.assign(PATIENT_ID=["X"] * len(d)), "not unique"),
        (lambda d: d.assign(CHEMOTHERAPY=d["CHEMOTHERAPY"] * 2), "only 0 and 1"),
        (lambda d: d.assign(event=d["event"] + 1), "only 0 and 1"),
        (lambda d: d.assign(duration=-d["duration"]), "non-negative"),
        (lambda d: d.assign(EGFR="high"), "Non-numeric"),
    ],
)
def test_structure_errors(tmp_path: Path, change, match: str) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ValueError, match=match):
        load_metabric(_write(change(_fake()), tmp_path), verify=None)


def test_split_refuses_unverified_data(tmp_path: Path) -> None:
    data = load_metabric(_write(_fake(), tmp_path), verify=None)
    with pytest.raises(ValueError, match="verify='content'"):
        metabric_split(data)


def test_packaged_split_is_a_partition(tmp_path: Path) -> None:
    # The indices ship with the package; their own hash is checked inside.
    unverified = load_metabric(_write(_fake(), tmp_path), verify=None)
    data = MetabricData(
        X=unverified.X,
        time=unverified.time,
        event=unverified.event,
        patient_id=unverified.patient_id,
        verified="content",
    )
    split = metabric_split(data)
    assert {k: len(v) for k, v in split.items()} == {"train": 1188, "validation": 396, "test": 396}
    every = np.concatenate(list(split.values()))
    np.testing.assert_array_equal(np.sort(every), np.arange(N_ROWS))
    assert all(v.dtype == np.int64 for v in split.values())
