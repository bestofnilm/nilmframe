"""Phase 0 acceptance: the package installs, imports, and exposes a CLI.

The repository this replaces had *no* importable module -- every `nilm.*` import
raised `ImportError: attempted relative import beyond top-level package`. These
tests exist so that never regresses.
"""

from __future__ import annotations

import subprocess
import sys

import pytest


def test_import_and_version():
    import nilmframe

    assert nilmframe.__version__


def test_no_cpp_extension_on_the_import_path():
    """Importing the package must not require a compiled extension.

    The old code did `from fitps import FITPS` at module scope in
    `data/highfreq/entity.py`, so an unbuilt pybind11 extension made the whole
    data layer unimportable. Cycle alignment is pure torch now.
    """
    import nilmframe  # noqa: F401

    assert "fitps" not in sys.modules


@pytest.mark.parametrize(
    "module",
    [
        "nilmframe.store",
        "nilmframe.readers",
        "nilmframe.data",
        "nilmframe.nn",
        "nilmframe.eval",
        "nilmframe.cli",
    ],
)
def test_subpackages_import(module):
    __import__(module)


def test_unknown_attribute_raises_attribute_error():
    import nilmframe

    with pytest.raises(AttributeError, match="no attribute"):
        getattr(nilmframe, "definitely_not_a_real_symbol")  # noqa: B009


def test_all_matches_lazy_export_table():
    """`__all__` is a literal for static analysers; keep it honest."""
    import nilmframe

    assert set(nilmframe.__all__) == set(nilmframe._EXPORTS) | {"__version__"}


def test_every_exported_symbol_resolves():
    """Every advertised name must actually be importable -- no stale entries."""
    import nilmframe

    for name in nilmframe._EXPORTS:
        assert getattr(nilmframe, name) is not None


def test_cli_version():
    out = subprocess.run(
        [sys.executable, "-m", "nilmframe.cli", "--version"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "nilmframe" in out.stdout


def test_optional_extras_degrade_to_a_legible_message(monkeypatch):
    """A missing extra must break one subpackage, not the whole package.

    Setting a name to None in sys.modules makes `import x` raise ImportError, so
    this reproduces an install without the optional dependencies. The CI
    `core-only` job checks the same thing against a real bare install.
    """
    import importlib

    for name in ("torchmetrics", "lightning"):
        monkeypatch.setitem(sys.modules, name, None)
    for name in list(sys.modules):
        if name.startswith("nilmframe.eval"):
            monkeypatch.delitem(sys.modules, name)

    # The core still imports.
    importlib.import_module("nilmframe.nn.align")
    importlib.import_module("nilmframe.data")

    with pytest.raises(ImportError, match=r"nilmframe\[eval\]"):
        importlib.import_module("nilmframe.eval.metrics")
