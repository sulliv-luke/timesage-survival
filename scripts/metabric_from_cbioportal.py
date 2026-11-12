"""Build the original study's 9-feature METABRIC CSV from the cBioPortal release.

Usage::

    python scripts/metabric_from_cbioportal.py path/to/brca_metabric out.csv

``brca_metabric`` is the cBioPortal study folder (ODbL 1.0), downloaded by
you; this repository never includes it. The script needs
``data_clinical_patient.txt`` and ``data_mrna_illumina_microarray.txt``. It
ports the original study's ``create_metabric_dataset.py``:

- four expression features of the pycox METABRIC set, in pycox's x0-x3 order:
  EGFR, PGR, ERBB2, MKI67 (``data_mrna_illumina_microarray.txt``);
- HORMONE_THERAPY, RADIO_THERAPY, CHEMOTHERAPY (``YES`` -> 1), ER_POSITIVE
  (``ER_IHC`` is "Positive" or the release's typo "Positve" -> 1) and
  AGE_AT_DIAGNOSIS;
- overall survival: ``duration`` = ``OS_MONTHS``, ``event`` = 1 if
  ``OS_STATUS`` is ``1:DECEASED``;
- patients present in both files, in the expression file's column order;
  rows with any missing value dropped.

**What has been verified (2026-10-28; Python 3.11.0, pandas 2.3.3,
numpy 2.4.6).** The script was run end to end on the cBioPortal release with

- ``data_clinical_patient.txt``, SHA-256
  ``2a6bbadc6e4b2b07c0505559d438da959fc5c9aba55c7194abc394e2d6454995``
  (identical to the copy used by the original study, as are the other 12
  release files it kept);
- ``data_mrna_illumina_microarray.txt``, SHA-256
  ``b8aa84ea4d205483e4b2c2794a5eac6e73d4644e0f13e74c940f8e3d81d93dd6``.

Both steps were verified: its output is byte-identical to the
original study's ``metabric_with_feature_names.csv`` (same columns, the same
1,980 patients in the same order, every value equal), so it passes both
``verify="content"`` and ``verify="file"``. It streams the expression file
line by line (about 3 s and 85 MB peak memory on the machine where this was
checked). ``tests/characterisation/test_metabric_build.py`` repeats the check
when the release is present.

The output is then checked with ``timesage.datasets.load_metabric``
against the original study's data. Lines end in CRLF, as in that study's
file; with a pandas version that formats floats differently, only
``verify="content"`` may pass.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

GENES = ["EGFR", "PGR", "ERBB2", "MKI67"]
CLINICAL = ["HORMONE_THERAPY", "RADIO_THERAPY", "CHEMOTHERAPY", "ER_POSITIVE", "AGE_AT_DIAGNOSIS"]


def clinical_features(release: Path) -> pd.DataFrame:
    """Clinical features and overall survival, one row per patient."""
    clinical = pd.read_csv(release / "data_clinical_patient.txt", sep="\t", comment="#")
    clin = clinical[
        [
            "PATIENT_ID",
            "HORMONE_THERAPY",
            "RADIO_THERAPY",
            "CHEMOTHERAPY",
            "ER_IHC",
            "AGE_AT_DIAGNOSIS",
            "OS_MONTHS",
            "OS_STATUS",
        ]
    ].copy()
    for col in ["HORMONE_THERAPY", "RADIO_THERAPY", "CHEMOTHERAPY"]:
        clin[col] = (clin[col] == "YES").astype(int)
    clin["ER_POSITIVE"] = clin["ER_IHC"].isin(["Positive", "Positve"]).astype(int)
    clin = clin.drop(columns="ER_IHC").rename(
        columns={"OS_MONTHS": "duration", "OS_STATUS": "event"}
    )
    clin["event"] = clin["event"].str.contains("1:DECEASED", na=False).astype(int)
    return clin


def build(release: Path) -> pd.DataFrame:
    """The 9-feature dataset from a ``brca_metabric`` folder (the original study's logic)."""
    clin = clinical_features(release)
    genes: dict[str, pd.Series] = {}
    with open(release / "data_mrna_illumina_microarray.txt", encoding="utf-8") as f:
        line = f.readline()
        while line.startswith("#"):
            line = f.readline()
        patient_ids = line.strip().split("\t")[2:]  # after Hugo_Symbol, Entrez_Gene_Id
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3 or parts[0] not in GENES:
                continue
            genes[parts[0]] = pd.Series([float(x) for x in parts[2:]], index=patient_ids)
            if len(genes) == len(GENES):
                break
    missing = [g for g in GENES if g not in genes]
    if missing:
        raise SystemExit(f"Genes not found in the expression file: {missing}")
    gene_df = pd.DataFrame(genes)
    gene_df.index.name = "PATIENT_ID"
    gene_df = gene_df.reset_index()

    common = set(clin["PATIENT_ID"]) & set(gene_df["PATIENT_ID"])
    merged = pd.merge(
        gene_df[gene_df["PATIENT_ID"].isin(common)],
        clin[clin["PATIENT_ID"].isin(common)],
        on="PATIENT_ID",
    )
    return merged[["PATIENT_ID", *GENES, *CLINICAL, "duration", "event"]].dropna()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("release", type=Path, help="cBioPortal brca_metabric folder")
    parser.add_argument("output", type=Path, help="CSV to write")
    args = parser.parse_args()
    df = build(args.release)
    df.to_csv(args.output, index=False, lineterminator="\r\n")
    print(f"wrote {args.output}: {len(df)} patients")

    from timesage.datasets.metabric import load_metabric

    for verify in ("content", "file"):
        try:
            load_metabric(args.output, verify=verify)  # type: ignore[arg-type]
            print(f"verify={verify!r}: matches the original study's data")
        except ValueError as err:
            print(f"verify={verify!r}: {err}")
            if verify == "content":
                return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
