"""nilmframe: PyTorch-native end-to-end non-intrusive load monitoring.

The design in three sentences:

1. Data lives in a **canonical store** -- tabular metadata plus memory-mapped signal
   arrays -- not in a graph of Python objects. Nothing is materialised until a
   window is requested.
2. A **view** derives the model's input from the store's highest-rate signal
   through one code path, so a low-frequency and a high-frequency experiment
   differ only by a config flag and share their preprocessing exactly.
3. Models emit **presence logits and non-negative power** separately, take the
   measured aggregate as an input rather than the sum of labels, and are tied to
   a measurable physical quantity by a conservation term in the loss.

See ``plan.html`` and ``REFRAME.md`` at the repository root for the rationale.
"""

from __future__ import annotations

import importlib

__version__ = "0.1.0"

# Public name -> module it lives in. Resolved lazily so that importing nilmframe
# never drags in torchmetrics/lightning/soundfile, and so a partially installed
# environment fails at the point of use with a legible message.
#
# This table only ever lists symbols that exist: tests/test_package.py resolves
# every entry. It grows as each phase of plan.html lands.
_EXPORTS: dict[str, str] = {
    "Activation": "nilmframe.store",
    "Compose": "nilmframe.data",
    "CrossDataset": "nilmframe.data",
    "GainJitter": "nilmframe.data",
    "APPLIANCES": "nilmframe.docdata",
    "CompatibilityReport": "nilmframe.compat",
    "Measurement": "nilmframe.measurement",
    "compatibility": "nilmframe.compat",
    "example_measurement": "nilmframe.docdata",
    "example_store": "nilmframe.docdata",
    "merge_stores": "nilmframe.store",
    "Taxonomy": "nilmframe.taxonomy",
    "default_taxonomy": "nilmframe.taxonomy",
    "MixAggregate": "nilmframe.data",
    "VoltageJitter": "nilmframe.data",
    "HighFreqView": "nilmframe.data",
    "LeaveBrandOut": "nilmframe.data",
    "LeaveHouseOut": "nilmframe.data",
    "LowFreqView": "nilmframe.data",
    "RandomSplit": "nilmframe.data",
    "Split": "nilmframe.data",
    "UnseenAppliance": "nilmframe.data",
    "WindowDataset": "nilmframe.data",
    "WindowIndex": "nilmframe.data",
    "check_leakage": "nilmframe.data",
    "collate_windows": "nilmframe.data",
    "ChannelKind": "nilmframe.store",
    "Quantity": "nilmframe.store",
    "Recording": "nilmframe.store",
    "Store": "nilmframe.store",
    "StoreWriter": "nilmframe.store",
}

# Spelled out rather than derived from _EXPORTS so static analysers can see it;
# tests/test_package.py asserts the two stay in sync.
__all__ = [
    "APPLIANCES",
    "Activation",
    "ChannelKind",
    "CompatibilityReport",
    "Compose",
    "CrossDataset",
    "GainJitter",
    "HighFreqView",
    "LeaveBrandOut",
    "LeaveHouseOut",
    "LowFreqView",
    "Measurement",
    "MixAggregate",
    "Quantity",
    "RandomSplit",
    "Recording",
    "Split",
    "Store",
    "StoreWriter",
    "Taxonomy",
    "UnseenAppliance",
    "VoltageJitter",
    "WindowDataset",
    "WindowIndex",
    "__version__",
    "check_leakage",
    "collate_windows",
    "compatibility",
    "default_taxonomy",
    "example_measurement",
    "example_store",
    "merge_stores",
]


def __getattr__(name: str):
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module 'nilmframe' has no attribute {name!r}")
    return getattr(importlib.import_module(module), name)


def __dir__() -> list[str]:
    return sorted([*_EXPORTS, "__version__", "nn", "eval", "data", "store", "readers", "sources"])
