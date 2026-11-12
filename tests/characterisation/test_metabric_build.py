"""``scripts/metabric_from_cbioportal.py`` rebuilds the original study's METABRIC CSV.

``needs_data``: needs the cBioPortal ``brca_metabric`` release folder, given by
``$TIMESAGE_CBIOPORTAL_DIR`` (never committed). Builds the 9-feature CSV in a
temporary directory, checks it against ``load_metabric``'s pinned hashes and
deletes it. Prints nothing from the data.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType

import pytest

from timesage.datasets.metabric import N_ROWS, load_metabric

REPO = Path(__file__).resolve().parents[2]
RELEASE = Path(os.environ.get("TIMESAGE_CBIOPORTAL_DIR", "brca_metabric"))
SCRIPT = REPO / "scripts" / "metabric_from_cbioportal.py"
NEEDED = ("data_clinical_patient.txt", "data_mrna_illumina_microarray.txt")

pytestmark = pytest.mark.needs_data


@pytest.fixture(scope="module")
def script() -> ModuleType:
    missing = [f for f in NEEDED if not (RELEASE / f).exists()]
    if "TIMESAGE_CBIOPORTAL_DIR" not in os.environ or missing:
        pytest.skip(
            f"needs_data: missing {', '.join(missing) or RELEASE} "
            "(set TIMESAGE_CBIOPORTAL_DIR to the cBioPortal brca_metabric folder)"
        )
    spec = importlib.util.spec_from_file_location("_metabric_from_cbioportal", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    dont_write = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = dont_write
    return module


@pytest.fixture
def built(tmp_path: Path, script: ModuleType) -> Iterator[Path]:
    path = tmp_path / "metabric_built.csv"
    df = script.build(RELEASE)
    df.to_csv(path, index=False, lineterminator="\r\n")
    yield path
    path.unlink(missing_ok=True)  # never leave patient-level copies behind


def test_build_matches_the_original_csv(built: Path) -> None:
    data = load_metabric(built, verify="content")
    assert len(data.time) == N_ROWS
    # The script writes CRLF like the original study's file; with this pandas the
    # bytes are identical too.
    assert load_metabric(built, verify="file").verified == "file"
