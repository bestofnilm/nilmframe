"""Signal-side building blocks: alignment, representations, event detection.

Everything here is an ``nn.Module`` operating on batched tensors, so the same object
runs per-sample inside a DataLoader worker and per-batch on the GPU -- training and
deployment share one code path.

These are the transforms that turn a measurement into something a model can eat,
and the detectors that decide where to look. The reference architectures live in
:mod:`nilmframe.nn.models`, behind one shared call signature.
"""

from __future__ import annotations

from nilmframe.nn import models
from nilmframe.nn.align import (
    CycleAlign,
    cycle_align,
    estimate_f0,
    rising_zero_crossings,
    samples_for_cycles,
)
from nilmframe.nn.repr import (
    DFIA,
    PAA,
    DistanceMatrix,
    Downsample,
    Fryze,
    HarmonicLowpass,
    Patchify,
    ReIm,
    Spectrogram,
    StandardScale,
    VITrajectory,
)
from nilmframe.nn.segment import (
    ActiveSectionDetector,
    AdaptiveThresholdDetector,
    CusumDetector,
    GLRDetector,
    GoodnessOfFitDetector,
    MultivariateDetector,
    ZScoreDetector,
    segments_from_mask,
)

__all__ = [
    "DFIA",
    "PAA",
    "ActiveSectionDetector",
    "AdaptiveThresholdDetector",
    "CusumDetector",
    "CycleAlign",
    "DistanceMatrix",
    "Downsample",
    "Fryze",
    "GLRDetector",
    "GoodnessOfFitDetector",
    "HarmonicLowpass",
    "MultivariateDetector",
    "Patchify",
    "ReIm",
    "Spectrogram",
    "StandardScale",
    "VITrajectory",
    "ZScoreDetector",
    "cycle_align",
    "estimate_f0",
    "models",
    "rising_zero_crossings",
    "samples_for_cycles",
    "segments_from_mask",
]
