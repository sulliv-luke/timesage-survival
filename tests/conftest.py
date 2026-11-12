"""Shared pytest configuration.

Test layout
-----------
unit/              Fast, hand-checkable tests of the package (run by default).
characterisation/  Golden-output tests pinning the original research
                   implementation's behaviour; goldens live in fixtures/golden/.
parity/            Tolerances (parity/tolerances.toml) shared by those tests.

Slow tests are deselected by default; run them with ``pytest -m slow``.
Tests marked ``needs_data`` read METABRIC or other files that are not in the
repository, located by environment variables (``TIMESAGE_METABRIC_CSV``,
``TIMESAGE_CBIOPORTAL_DIR``, ``TIMESAGE_SYNTHETIC_DIR``). They run by default
and skip, naming what is missing, when those are unset or absent.
"""
