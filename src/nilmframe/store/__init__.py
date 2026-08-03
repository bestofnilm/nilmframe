"""Canonical store: tabular metadata plus memory-mapped signals."""

from __future__ import annotations

from nilmframe.store.merge import merge_stores
from nilmframe.store.reader import Store
from nilmframe.store.schema import (
    DEFAULT_ON_THRESHOLD_W,
    STORE_FORMAT_VERSION,
    Activation,
    ChannelKind,
    Quantity,
    Recording,
)
from nilmframe.store.writer import StoreWriter, sha256_file

__all__ = [
    "DEFAULT_ON_THRESHOLD_W",
    "STORE_FORMAT_VERSION",
    "Activation",
    "ChannelKind",
    "Quantity",
    "Recording",
    "Store",
    "StoreWriter",
    "merge_stores",
    "sha256_file",
]
