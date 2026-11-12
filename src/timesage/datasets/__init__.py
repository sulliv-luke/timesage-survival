"""Datasets (public): the synthetic study's generator and the METABRIC loader.

Nothing is downloaded or bundled: synthetic data are generated from a seed,
and METABRIC must be supplied by the user (ODbL; see
:mod:`timesage.datasets.metabric`).
"""

from timesage.datasets.metabric import MetabricData, load_metabric, metabric_split
from timesage.datasets.synthetic import (
    OracleResult,
    Scenario,
    SyntheticDataset,
    get_scenario,
    make_synthetic,
    oracle_importance,
)

__all__ = [
    "MetabricData",
    "OracleResult",
    "Scenario",
    "SyntheticDataset",
    "get_scenario",
    "load_metabric",
    "make_synthetic",
    "metabric_split",
    "oracle_importance",
]
