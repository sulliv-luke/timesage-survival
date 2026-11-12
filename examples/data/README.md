# Example data (feature-level aggregates only)

## coxph_lofo_summary.csv

Leave-one-feature-out (LOFO) results for the 9-feature METABRIC CoxPH model,
one row per feature: the C-index and Brier scores with the feature removed
(model refitted), and their degradation relative to the full model
(`mean_delta`, `integrated_delta`, `max_delta` = largest degradation over
time).

- **Provenance:** a byte-for-byte copy of `results/coxph_ablation_summary.csv`
  from the original research implementation (SHA-256
  `74924f1c902c793a2e95c217f94208516aa337801cded1c1c1ff5731c9b2a713`). Its
  `02_coxph.ipynb` notebook (cell 36) reads it for its LOFO comparison.
- **Use:** `examples/coxph_metabric.py` compares TimeSAGE against it. It is
  not recomputed here.
- **Contents:** 9 rows of model-performance aggregates; no patient-level data.
- **Licence:** a work produced from METABRIC (ODbL 1.0); see `DATA.md` for
  attribution.
