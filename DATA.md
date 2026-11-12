# Data

**No data are bundled or downloaded.** `timesage-survival` contains no
patient-level data and never fetches any. Synthetic data are generated on the
fly from a seed (`timesage.datasets.make_synthetic`); METABRIC must be
obtained and supplied by the user (`timesage.datasets.load_metabric`).

## METABRIC

**Source.** The cBioPortal study `brca_metabric`, "Breast Cancer (METABRIC,
Nature 2012 & Nat Commun 2016)". The original study's 9-feature file
(`metabric_with_feature_names.csv`: EGFR, PGR, ERBB2, MKI67,
HORMONE_THERAPY, RADIO_THERAPY, CHEMOTHERAPY, ER_POSITIVE, AGE_AT_DIAGNOSIS,
overall survival) can be rebuilt from your own download with
`scripts/metabric_from_cbioportal.py` (in the source repository, not the
installed package). On 2026-10-28 the script was run on the release
(`data_clinical_patient.txt` SHA-256 `2a6bbadc...6454995`,
`data_mrna_illumina_microarray.txt` SHA-256 `b8aa84ea...e1d93dd6`; full
hashes in the script) with Python 3.11.0, pandas 2.3.3 and numpy 2.4.6, and
its output was byte-identical to the original study's file.

**Licence.** The data are available under the Open Data Commons Open
Database License (ODbL) 1.0, http://opendatacommons.org/licenses/odbl/1.0/.
In the words of the release's `LICENSE` file: you are free to share and
modify the data so long as you attribute any public use of the database, or
works produced from the database; keep the resulting data-sets open; and
offer your shared or adapted version of the data-set under the same ODbL
license.

**Citation.** As listed in the release's `meta_study.txt`:

- Curtis et al., Nature 2012 (PMID 22522925)
- Pereira et al., Nat Commun 2016 (PMID 27161491)
- Rueda et al., Nature 2019 (PMID 30867590)

Also cite cBioPortal as its website asks. These entries are copied from
`meta_study.txt`; they have not been checked against the original papers.

**What ships with the package.** Only the original study's fixed train /
validation / test split: 1,980 integer row positions in
`timesage/datasets/_data/metabric_split_indices.json`, with a pinned SHA-256.
They identify rows of the user's own file and contain no patient data.
Patient identifiers, values and outcomes are never included.

## Synthetic data

`timesage.datasets.synthetic` reimplements the synthetic study of the
original research implementation (scenarios E1-H1). Its data are generated from seeds. The test
fixtures in `tests/fixtures/golden/` are synthetic only: digests of the
generated datasets, their first rows, and oracle curves.
