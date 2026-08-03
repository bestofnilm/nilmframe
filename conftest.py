"""Root fixtures, shared by the test suite and by the doctests in ``src/``.

Docstring examples are collected from ``src/`` and tests from ``tests/``, so the
namespace they run in has to be defined here rather than under either. Sphinx
gets the same names from ``doctest_global_setup`` in ``docs/conf.py``, so an
example that passes under pytest passes in the built documentation too.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _doctest_namespace(doctest_namespace):
    import numpy as np
    import torch

    import nilmframe as nf

    doctest_namespace["np"] = np
    doctest_namespace["torch"] = torch
    doctest_namespace["nf"] = nf
    doctest_namespace["store"] = nf.example_store()
