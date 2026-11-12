import importlib

import pytest

import timesage

SUBMODULES = [
    "timesage.estimator",
    "timesage.result",
    "timesage._ipcw",
    "timesage._value",
    "timesage.integrate",
    "timesage.metrics",
    "timesage.protocols",
    "timesage.plotting",
    "timesage.adapters",
    "timesage.adapters.coxph",
    "timesage.adapters.coxtv",
    "timesage.adapters.deepsurv",
    "timesage.adapters.deephit",
    "timesage.datasets",
    "timesage.datasets.synthetic",
    "timesage.datasets.metabric",
]


def test_version_is_a_string() -> None:
    assert isinstance(timesage.__version__, str)


@pytest.mark.parametrize("name", SUBMODULES)
def test_submodule_imports_without_optional_dependencies(name: str) -> None:
    importlib.import_module(name)
