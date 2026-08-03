"""Evaluation: metric families, routing, and reports.

Needs the ``eval`` extra: ``pip install 'nilmframe[eval]'``.
"""

from __future__ import annotations

from nilmframe.eval.evaluator import Evaluator
from nilmframe.eval.events import (
    EventCounts,
    EventF1,
    EventTimingError,
    match_events,
)
from nilmframe.eval.metrics import (
    SAE,
    TECA,
    CalibrationError,
    DetectionF1,
    MatthewsCorrCoef,
    MeanAbsoluteError,
    ModifiedF1,
    ModifiedJaccard,
    NormalisedDisaggregationError,
    UnknownAUROC,
    default_collection,
)
from nilmframe.eval.report import HEADLINE_COLUMNS, compare, format_table, load_results

__all__ = [
    "HEADLINE_COLUMNS",
    "SAE",
    "TECA",
    "CalibrationError",
    "DetectionF1",
    "Evaluator",
    "EventCounts",
    "EventF1",
    "EventTimingError",
    "MatthewsCorrCoef",
    "MeanAbsoluteError",
    "ModifiedF1",
    "ModifiedJaccard",
    "NormalisedDisaggregationError",
    "UnknownAUROC",
    "compare",
    "default_collection",
    "format_table",
    "load_results",
    "match_events",
]
