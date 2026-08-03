"""Shared fixtures.

The stores are session-scoped: building them is fast, but rebuilding one per test
would dominate the suite's runtime for no benefit, since every consumer treats
them as read-only.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent / "fixtures"))

from synthetic import make_plaid, make_ukdale, make_whited


@pytest.fixture(scope="session")
def plaid_source(tmp_path_factory) -> tuple[Path, Path]:
    """A miniature dataset in PLAID's on-disk format."""
    return make_plaid(tmp_path_factory.mktemp("plaid_src"))


@pytest.fixture(scope="session")
def whited_source(tmp_path_factory) -> Path:
    """A miniature dataset in WHITED's on-disk format."""
    pytest.importorskip("soundfile")
    return make_whited(tmp_path_factory.mktemp("whited_src"))


@pytest.fixture(scope="session")
def plaid_store(tmp_path_factory, plaid_source):
    """A canonical store converted from the PLAID fixture."""
    from nilmframe.readers import PLAID
    from nilmframe.store import Store, StoreWriter

    csv_dir, metadata = plaid_source
    path = tmp_path_factory.mktemp("plaid_store") / "store"
    with StoreWriter(path, source="synthetic-plaid") as writer:
        writer.extend(PLAID(csv_dir, metadata))
    return Store(path)


@pytest.fixture(scope="session")
def whited_store(tmp_path_factory, whited_source):
    """A canonical store converted from the WHITED fixture."""
    from nilmframe.readers import WHITED
    from nilmframe.store import Store, StoreWriter

    path = tmp_path_factory.mktemp("whited_store") / "store"
    with StoreWriter(path, source="synthetic-whited") as writer:
        writer.extend(WHITED(whited_source))
    return Store(path)


@pytest.fixture(scope="session")
def combined_store(tmp_path_factory, plaid_source, whited_source):
    """One store holding both datasets, for cross-dataset protocols."""
    from nilmframe.readers import PLAID, WHITED
    from nilmframe.store import Store, StoreWriter

    csv_dir, metadata = plaid_source
    path = tmp_path_factory.mktemp("combined_store") / "store"
    with StoreWriter(path, source="synthetic-plaid+whited") as writer:
        writer.extend(PLAID(csv_dir, metadata))
        writer.extend(WHITED(whited_source))
    return Store(path)


@pytest.fixture(scope="session")
def ukdale_source(tmp_path_factory) -> Path:
    """A miniature dataset in UK-DALE's on-disk format."""
    pytest.importorskip("soundfile")
    return make_ukdale(tmp_path_factory.mktemp("ukdale_src"))


@pytest.fixture(scope="session")
def ukdale_store(tmp_path_factory, ukdale_source):
    """A canonical store converted from the UK-DALE fixture."""
    from nilmframe.readers import UKDALE
    from nilmframe.store import Store, StoreWriter

    path = tmp_path_factory.mktemp("ukdale_store") / "store"
    with StoreWriter(path, source="synthetic-ukdale") as writer:
        writer.extend(UKDALE(ukdale_source, rate_hz=1 / 6))
    return Store(path)
