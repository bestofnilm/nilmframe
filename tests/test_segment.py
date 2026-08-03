"""Event detection.

`legacy/events/` was five empty files that the dataset layer already called into.
These tests exist so the replacements are real rather than merely present.
"""

from __future__ import annotations

from itertools import pairwise

import pytest
import torch

from nilmframe.nn.segment import CusumDetector, ZScoreDetector, segments_from_mask


def step_signal(level: float = 100.0, n: int = 400, at: int = 200, noise: float = 1.0):
    torch.manual_seed(0)
    x = torch.zeros(n)
    x[at:] = level
    return x + noise * torch.randn(n)


def ramp_signal(level: float = 100.0, n: int = 400, start: int = 150, length: int = 120):
    """A change too gradual for any single sample to look anomalous."""
    torch.manual_seed(0)
    x = torch.zeros(n)
    x[start : start + length] = torch.linspace(0, level, length)
    x[start + length :] = level
    return x + 0.5 * torch.randn(n)


# --------------------------------------------------------------------------- #
# Z-score
# --------------------------------------------------------------------------- #


def test_zscore_finds_a_step():
    events = ZScoreDetector(window=32, threshold=4.0)(step_signal(at=200))
    found = events.nonzero().flatten()
    assert found.numel() >= 1
    assert (found - 200).abs().min() < 40, "the event should land near the step"


def test_zscore_is_quiet_on_a_flat_signal():
    torch.manual_seed(1)
    flat = 50.0 + 0.5 * torch.randn(500)
    events = ZScoreDetector(window=32, threshold=6.0, min_delta=5.0)(flat)
    assert events.sum() == 0, "noise on an idle channel must not read as activity"


def test_min_delta_ignores_relatively_large_but_absolutely_tiny_steps():
    """A 0.01 W wobble on a dead channel is many deviations and still not an event."""
    torch.manual_seed(2)
    tiny = 0.001 * torch.randn(400)
    tiny[200:] += 0.01
    assert ZScoreDetector(window=32, threshold=3.0, min_delta=0.0)(tiny).sum() > 0
    assert ZScoreDetector(window=32, threshold=3.0, min_delta=1.0)(tiny).sum() == 0


def test_min_gap_enforces_a_minimum_spacing_between_events():
    """A switching edge otherwise reports one appliance turning on as a dozen.

    The contract is the spacing, not a particular count: with min_gap=1 this
    signal yields 11 events one sample apart; with min_gap=10 it yields 5, none
    closer together than the gap.
    """
    signal = step_signal(at=200, noise=0.5)
    dense = ZScoreDetector(window=32, threshold=3.0, min_gap=1)(signal)
    sparse = ZScoreDetector(window=32, threshold=3.0, min_gap=10)(signal)

    assert sparse.sum() < dense.sum()
    kept = sparse.nonzero().flatten().tolist()
    spacings = [b - a for a, b in pairwise(kept)]
    assert all(gap >= 10 for gap in spacings), spacings


# --------------------------------------------------------------------------- #
# CUSUM
# --------------------------------------------------------------------------- #


def test_cusum_finds_a_step():
    events = CusumDetector(window=64, threshold=200.0, drift=1.0)(step_signal(at=200))
    assert events.sum() >= 1
    assert events.nonzero().flatten().min() > 190


def test_cusum_catches_a_ramp_a_zscore_misses():
    """The reason to have two detectors: a compressor ramp is not a kettle edge."""
    signal = ramp_signal()
    zscore = ZScoreDetector(window=32, threshold=6.0, min_delta=5.0)(signal)
    cusum = CusumDetector(window=32, threshold=150.0, drift=1.0)(signal)
    assert cusum.sum() > zscore.sum()


def test_cusum_drift_raises_the_bar_for_what_counts_as_a_change():
    """`drift` is a dead band: each deviation must beat it before it accumulates.

    Measured on the ramp: drift 0 gives 10 events, 5 gives 5, and 20 gives none,
    because a gradient shallower than the dead band never accumulates at all.
    """
    signal = ramp_signal()
    counts = [
        int(CusumDetector(window=32, threshold=150.0, drift=d)(signal).sum())
        for d in (0.0, 5.0, 20.0)
    ]
    assert counts == sorted(counts, reverse=True), counts
    assert counts[0] > 0 and counts[-1] == 0


def test_cusum_is_quiet_on_stationary_noise():
    """The running mean tracks the level, so zero-mean noise must not accumulate."""
    torch.manual_seed(3)
    quiet = 10.0 + torch.randn(2000)
    assert CusumDetector(window=64, threshold=100.0, drift=1.0)(quiet).sum() == 0


# --------------------------------------------------------------------------- #
# Contract
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "detector",
    [ZScoreDetector(window=16), CusumDetector(window=16, threshold=100.0)],
    ids=["zscore", "cusum"],
)
def test_batched_equals_per_item(detector):
    signals = [step_signal(at=a) for a in (100, 200, 300)]
    batched = detector(torch.stack(signals))
    for k, signal in enumerate(signals):
        assert torch.equal(batched[k], detector(signal))


@pytest.mark.parametrize(
    "detector",
    [ZScoreDetector(window=16), CusumDetector(window=16, threshold=100.0)],
    ids=["zscore", "cusum"],
)
def test_output_shape_matches_input(detector):
    assert detector(torch.randn(400)).shape == (400,)
    assert detector(torch.randn(3, 400)).shape == (3, 400)
    assert detector(torch.randn(3, 400)).dtype == torch.bool


def test_detectors_validate_their_window():
    with pytest.raises(ValueError, match="window"):
        ZScoreDetector(window=1)
    with pytest.raises(ValueError, match="window"):
        CusumDetector(window=1)


# --------------------------------------------------------------------------- #
# Segments
# --------------------------------------------------------------------------- #


def test_segments_from_mask():
    mask = torch.zeros(10, dtype=torch.bool)
    mask[3] = mask[7] = True
    assert segments_from_mask(mask) == [(0, 3), (3, 7), (7, 10)]


def test_segments_respect_a_minimum_length():
    mask = torch.zeros(10, dtype=torch.bool)
    mask[1] = mask[2] = mask[8] = True
    assert segments_from_mask(mask, min_length=4) == [(2, 8)]


def test_segments_of_an_eventless_signal_are_the_whole_signal():
    assert segments_from_mask(torch.zeros(10, dtype=torch.bool)) == [(0, 10)]


def test_segments_batched():
    mask = torch.zeros(2, 10, dtype=torch.bool)
    mask[0, 5] = True
    spans = segments_from_mask(mask)
    assert spans[0] == [(0, 5), (5, 10)]
    assert spans[1] == [(0, 10)]


def test_detector_and_segments_compose_on_a_real_window(plaid_store):
    """End to end on stored data: detect on the power envelope, cut into spans."""
    from nilmframe.data import LowFreqView, WindowDataset

    ids = plaid_store.mains()["channel_id"].tolist()
    item = WindowDataset(plaid_store, ids, view=LowFreqView(rate_hz=200.0, n_steps=64))[0]

    events = ZScoreDetector(window=8, threshold=3.0, min_gap=4)(item["p"])
    spans = segments_from_mask(events, min_length=2)
    assert spans and spans[0][0] == 0 and spans[-1][1] == 64
    assert all(b > a for a, b in spans)
