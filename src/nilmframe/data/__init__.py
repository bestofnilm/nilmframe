"""Data layer: window index, torch Dataset, rate views, split protocols."""

from __future__ import annotations

from nilmframe.data.dataset import WindowDataset, collate_windows
from nilmframe.data.mixing import Compose, GainJitter, MixAggregate, VoltageJitter, materialize
from nilmframe.data.splits import (
    CrossDataset,
    LeaveBrandOut,
    LeaveHouseOut,
    RandomSplit,
    Split,
    SplitProtocol,
    UnseenAppliance,
    check_leakage,
)
from nilmframe.data.views import HighFreqView, LowFreqView, View, active_power
from nilmframe.data.windows import WindowIndex

__all__ = [
    "Compose",
    "CrossDataset",
    "GainJitter",
    "HighFreqView",
    "LeaveBrandOut",
    "LeaveHouseOut",
    "LowFreqView",
    "MixAggregate",
    "RandomSplit",
    "Split",
    "SplitProtocol",
    "UnseenAppliance",
    "View",
    "VoltageJitter",
    "WindowDataset",
    "WindowIndex",
    "active_power",
    "check_leakage",
    "collate_windows",
    "materialize",
]
