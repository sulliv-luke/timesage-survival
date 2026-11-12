"""Loader for the original study's 9-feature METABRIC dataset.

**Loader only.** The package never downloads or bundles METABRIC data. You
need the CSV the original study used (``metabric_with_feature_names.csv``),
which ``scripts/metabric_from_cbioportal.py`` in the source repository builds
from the cBioPortal ``brca_metabric`` release.

**Licence and attribution.** METABRIC data from cBioPortal are available
under the Open Database License (ODbL 1.0). Attribute any public use of the
data or of works produced from it, keep derived datasets open, and share them
under the same licence. The study's citations, as listed in the release's
``meta_study.txt``: Curtis et al., Nature 2012 (PMID 22522925); Pereira et
al., Nat Commun 2016 (PMID 27161491); Rueda et al., Nature 2019 (PMID
30867590). Also cite cBioPortal, as its website asks. Full attribution:
``DATA.md`` in the source distribution.

**The dataset.** 1,980 patients, the four gene-expression features of the
pycox METABRIC set (EGFR, PGR, ERBB2, MKI67) and five clinical features, with
overall survival as the endpoint (``OS_MONTHS``, ``OS_STATUS``: death from any
cause, in months).

Limitations
-----------
- Overall survival, not disease-specific survival.
- 1,980 patients, not the 1,904 of pycox's preprocessed METABRIC.
- One patient (in the validation split) has a follow-up time of 0.
- Nothing is scaled. Fit any scaler on the training split only; the
  original study standardised features for DeepSurv, DeepHit and the
  time-varying Cox model, and used raw features for CoxPH.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from importlib import resources
from typing import Literal

import numpy as np
import numpy.typing as npt
import pandas as pd

__all__ = ["FEATURE_NAMES", "MetabricData", "load_metabric", "metabric_split"]

FEATURE_NAMES: tuple[str, ...] = (
    "EGFR",
    "PGR",
    "ERBB2",
    "MKI67",
    "HORMONE_THERAPY",
    "RADIO_THERAPY",
    "CHEMOTHERAPY",
    "ER_POSITIVE",
    "AGE_AT_DIAGNOSIS",
)
_BINARY = ("HORMONE_THERAPY", "RADIO_THERAPY", "CHEMOTHERAPY", "ER_POSITIVE")
_COLUMNS = ("PATIENT_ID", *FEATURE_NAMES, "duration", "event")
N_ROWS = 1980

# SHA-256 of the original study's CSV, byte for byte (CRLF line endings).
_FILE_SHA256 = "d227efe688d799dab457fa1d5e5e22cdaeb4972793b06c74ed01e9c8f6a3e76b"
# SHA-256 of its parsed content (_content_digest): column names, patient IDs in
# row order and every value as float64. Independent of line endings and quoting.
_CONTENT_SHA256 = "46b14f03fe0fea3d585a8d1574ae1e84ca1f1ef60142ea1c013a35b4a19aa024"
# SHA-256 of _data/metabric_split_indices.json (the original study's split).
_SPLIT_SHA256 = "5689f6898b35a7faad0a4ac5be095be1753d3a6e3fb5f11882a9a180a95d12aa"
_SPLIT_SIZES = {"train": 1188, "validation": 396, "test": 396}

Verification = Literal["content", "file"]


@dataclass(frozen=True)
class MetabricData:
    """The 9-feature METABRIC dataset, in the CSV's row order.

    Attributes
    ----------
    X : pandas.DataFrame
        The nine features, in :data:`FEATURE_NAMES` order. Binary columns are
        integers.
    time : numpy.ndarray
        Follow-up in months (overall survival).
    event : numpy.ndarray
        1 if death was observed, 0 if censored.
    patient_id : pandas.Series
        cBioPortal patient identifiers, kept out of ``X``.
    verified : {"content", "file"} or None
        How the file was checked against the original study's data; None if it
        was not.
    """

    X: pd.DataFrame
    time: npt.NDArray[np.float64]
    event: npt.NDArray[np.int64]
    patient_id: pd.Series
    verified: Verification | None


def load_metabric(
    path: str | os.PathLike[str], *, verify: Verification | None = "content"
) -> MetabricData:
    """Load the original study's 9-feature METABRIC CSV and check it.

    Parameters
    ----------
    path : str or path-like
        ``metabric_with_feature_names.csv`` in the original study's format: columns
        ``PATIENT_ID``, the nine features, ``duration``, ``event``.
    verify : {"content", "file"} or None, default "content"
        ``"content"`` checks a SHA-256 of the parsed data (column names,
        patient IDs in row order and all values), so a re-saved copy with
        different line endings passes. ``"file"`` checks the file's bytes,
        which is stricter. ``None`` checks the structure only; such data
        cannot be split with :func:`metabric_split`.

    Returns
    -------
    MetabricData

    Raises
    ------
    ValueError
        If the columns, their order, the row count, a value or a hash does
        not match the original study's data.
    """
    if verify not in ("content", "file", None):
        raise ValueError(f"verify must be 'content', 'file' or None, got {verify!r}.")
    path = os.fspath(path)
    if verify == "file":
        digest = _sha256_file(path)
        if digest != _FILE_SHA256:
            raise ValueError(
                f"{path} is not byte-identical to the original study's CSV "
                f"(SHA-256 {digest}, expected {_FILE_SHA256}). A re-saved copy can "
                "still pass verify='content'."
            )
    # Correctly rounded parsing; on the original study's file this gives the same
    # values as pandas' default parser, which that study used.
    df = pd.read_csv(path, sep=",", comment="#", float_precision="round_trip")
    _check_structure(df)
    if verify == "content":
        digest = _content_digest(df)
        if digest != _CONTENT_SHA256:
            raise ValueError(
                f"The data in {path} differ from the original study's METABRIC data "
                f"(content SHA-256 {digest}, expected {_CONTENT_SHA256}): values, "
                "patient IDs or row order have changed."
            )
    return MetabricData(
        X=df.loc[:, list(FEATURE_NAMES)].copy(),
        time=df["duration"].to_numpy(dtype=np.float64),
        event=df["event"].to_numpy(dtype=np.int64),
        patient_id=df["PATIENT_ID"].astype(str).rename("patient_id"),
        verified=verify,
    )


def metabric_split(data: MetabricData) -> dict[str, npt.NDArray[np.int64]]:
    """Return the original study's fixed train / validation / test split (60 / 20 / 20).

    Positional row indices into ``data`` (1,188 / 396 / 396), stratified on
    the event indicator as in the original study (``01_data_preparation``). The
    indices ship with the package and are checked against a pinned SHA-256.
    They refer to row positions, so ``data`` must have been verified.

    Returns
    -------
    dict
        ``{"train": ..., "validation": ..., "test": ...}``, integer arrays in
        the original study's order.
    """
    if data.verified is None:
        raise ValueError(
            "metabric_split needs data loaded with verify='content' or verify='file': "
            "the indices refer to the original study's row order."
        )
    raw = resources.files("timesage.datasets").joinpath("_data/metabric_split_indices.json")
    blob = raw.read_bytes()
    if hashlib.sha256(blob).hexdigest() != _SPLIT_SHA256:
        raise RuntimeError("The packaged METABRIC split indices are corrupt (hash mismatch).")
    stored = json.loads(blob)
    split = {
        "train": np.asarray(stored["train_idx"], dtype=np.int64),
        "validation": np.asarray(stored["val_idx"], dtype=np.int64),
        "test": np.asarray(stored["test_idx"], dtype=np.int64),
    }
    every = np.concatenate(list(split.values()))
    if {k: len(v) for k, v in split.items()} != _SPLIT_SIZES or not np.array_equal(
        np.sort(every), np.arange(len(data.time))
    ):
        raise RuntimeError("The packaged METABRIC split indices are inconsistent.")
    return split


# ------------------------------------------------------------------ private


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _content_digest(df: pd.DataFrame) -> str:
    h = hashlib.sha256()
    h.update(",".join(df.columns).encode("utf-8"))
    h.update("\n".join(df["PATIENT_ID"].astype(str)).encode("utf-8"))
    for col in df.columns[1:]:
        h.update(np.ascontiguousarray(df[col].to_numpy(dtype="<f8")).tobytes())
    return h.hexdigest()


def _check_structure(df: pd.DataFrame) -> None:
    if tuple(df.columns) != _COLUMNS:
        raise ValueError(
            "Unexpected columns. Expected, in this order:\n"
            f"  {list(_COLUMNS)}\ngot:\n  {list(df.columns)}"
        )
    if len(df) != N_ROWS:
        raise ValueError(f"Expected {N_ROWS} rows, got {len(df)}.")
    if df.isna().any().any():
        raise ValueError("The data contain missing values.")
    if not df["PATIENT_ID"].is_unique:
        raise ValueError("PATIENT_ID values are not unique.")
    numeric = df.drop(columns=["PATIENT_ID"])
    non_numeric = [c for c in numeric.columns if not pd.api.types.is_numeric_dtype(numeric[c])]
    if non_numeric:
        raise ValueError(f"Non-numeric columns: {non_numeric}.")
    if not np.all(np.isfinite(numeric.to_numpy(dtype=float))):
        raise ValueError("The data contain non-finite values.")
    for col in (*_BINARY, "event"):
        if not df[col].isin([0, 1]).all():
            raise ValueError(f"{col} must contain only 0 and 1.")
    if (df["duration"] < 0).any():
        raise ValueError("duration must be non-negative.")
