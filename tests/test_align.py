"""Phase 1 acceptance: cycle alignment is correct, batched, and dependency-free.

The acceptance criterion in plan.html is max error below 1e-3 against an analytic
signal, plus agreement with a *fixed* transcription of the original C++.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

from nilmframe.nn.align import (
    CycleAlign,
    cycle_align,
    estimate_f0,
    rising_zero_crossings,
    samples_for_cycles,
)

sys.path.insert(0, str(Path(__file__).parent / "reference"))
from fitps_reference import FITPSReference, allocate_fixed, allocate_original

FS = 6000.0


def sine(f0: float, n: int, fs: float = FS, phase: float = 0.0, amp: float = 1.0) -> torch.Tensor:
    t = torch.arange(n, dtype=torch.float64) / fs
    return amp * torch.sin(2 * math.pi * f0 * t + phase)


def ideal_cycle(cycle_size: int, phase_offset: float = 0.0) -> torch.Tensor:
    m = torch.arange(cycle_size, dtype=torch.float64) / cycle_size
    return torch.sin(2 * math.pi * m + phase_offset)


# --------------------------------------------------------------------------- #
# Acceptance criterion
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("f0", [49.3, 50.0, 50.7, 59.4, 60.0])
@pytest.mark.parametrize("cycle_size", [64, 118, 128])
def test_analytic_sine_within_tolerance(f0, cycle_size):
    """Every aligned cycle of a pure sine must equal one period of a sine.

    This is the property the whole representation rests on: the mains frequency
    is off-nominal and non-integer in samples, and alignment has to remove that.
    """
    v = sine(f0, 4000).unsqueeze(0)
    i = 0.5 * v
    vc, ic, mask = cycle_align(v, i, fs=FS, cycle_size=cycle_size, f0=f0)

    assert mask.all(), "a clean sine should yield no rejected cycles"
    assert vc.shape[1] >= 15

    target = ideal_cycle(cycle_size)
    err = (vc[0] - target).abs().max().item()
    assert err < 1e-3, f"max error {err:.2e} exceeds the 1e-3 acceptance bound"
    assert (ic[0] - 0.5 * target).abs().max().item() < 1e-3


def _allocate_rms(allocate, phase: float, cycle_size: int = 118) -> float:
    """RMS error of one allocated cycle against an ideal period, at a given phase."""
    buf = sine(50.7, 400, phase=phase).tolist()
    zc = [k for k in range(len(buf) - 1) if buf[k] < 0 <= buf[k + 1]]
    z0, z1 = zc[0], zc[1]
    shifts = [-buf[z] / (buf[z + 1] - buf[z] + 1e-9) for z in (z0, z1)]
    out = torch.tensor(allocate(buf, [z0, z1], shifts, cycle_size))
    return (out - ideal_cycle(cycle_size)).pow(2).mean().sqrt().item()


# Zero-crossing phases sampled across a period; the original's error depends on
# where in the sample grid the crossing falls, so a single phase understates it.
_PHASES = (0.0, 0.3, 0.6, 1.2, 2.1, 3.0)


@pytest.mark.parametrize("phase", _PHASES)
def test_fixed_weight_beats_the_shipped_cpp_weight_at_every_phase(phase):
    """Measured on this signal the ratio ranges ~31x (phase 0.0) to ~259x (3.0)."""
    rms_original = _allocate_rms(allocate_original, phase)
    rms_fixed = _allocate_rms(allocate_fixed, phase)

    assert rms_fixed < 1e-3
    assert rms_original > 5e-3
    assert rms_original > 20 * rms_fixed


def test_original_cpp_weight_is_phase_dependent():
    """Regression artefact: the shipped weight varies with crossing phase.

    The distortion tracks the sub-sample offset of the cycle's first zero
    crossing, so it changes from cycle to cycle -- precisely the jitter FITPS
    exists to remove. The corrected weight is phase-insensitive. If someone
    reinstates the C++ formula, this fails.
    """
    original = [_allocate_rms(allocate_original, p) for p in _PHASES]
    fixed = [_allocate_rms(allocate_fixed, p) for p in _PHASES]

    spread_original = max(original) - min(original)
    spread_fixed = max(fixed) - min(fixed)

    assert spread_original > 5e-3
    assert spread_fixed < 1e-3
    assert spread_original > 20 * spread_fixed


# --------------------------------------------------------------------------- #
# Agreement with the fixed C++ transcription
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("f0", [49.5, 50.0, 51.0])
def test_matches_fixed_cpp_reference(f0):
    """The torch rewrite must reproduce the corrected original algorithm."""
    cycle_size = round(FS / 50.0)
    v = sine(f0, 3000)
    i = 0.3 * v + 0.1 * sine(3 * f0, 3000)

    ref = FITPSReference(cycle_size, int(1.25 * cycle_size), thresh=20, fixed=True)
    ref_v, ref_i = ref.transform(v.tolist(), i.tolist())
    assert len(ref_v) > 10

    vc, ic, mask = cycle_align(
        v.unsqueeze(0), i.unsqueeze(0), fs=FS, cycle_size=cycle_size, f0=50.0, tol=20 / cycle_size
    )
    got_v = vc[0][mask[0]]
    got_i = ic[0][mask[0]]

    n = min(len(ref_v), got_v.shape[0])
    assert n >= 10
    ref_v_t = torch.tensor(ref_v[:n], dtype=torch.float64)
    ref_i_t = torch.tensor(ref_i[:n], dtype=torch.float64)

    # The reference streams and resets its buffer, so it drops the cycle following
    # each reset; compare the cycles both produce, not the counts.
    assert (got_v[:n] - ref_v_t).abs().max() < 1e-6
    assert (got_i[:n] - ref_i_t).abs().max() < 1e-6


# --------------------------------------------------------------------------- #
# Contract
# --------------------------------------------------------------------------- #


def test_batched_equals_per_item():
    """Batching must not change results -- no cross-item leakage through padding."""
    signals = [sine(49.6, 2000), sine(50.4, 2000, phase=0.7), sine(60.0, 2000)]
    v = torch.stack(signals)
    i = torch.stack([0.2 * s for s in signals])

    vb, ib, mb = cycle_align(v, i, fs=FS, cycle_size=64, n_cycles=8)
    for k, s in enumerate(signals):
        vs, is_, ms = cycle_align(
            s.unsqueeze(0), (0.2 * s).unsqueeze(0), fs=FS, cycle_size=64, n_cycles=8
        )
        assert torch.equal(mb[k : k + 1], ms)
        assert (vb[k : k + 1] - vs).abs().max() < 1e-9
        assert (ib[k : k + 1] - is_).abs().max() < 1e-9


def test_multicomponent_current_preserves_superposition():
    """Components are resampled on the voltage's grid, so their sum is preserved.

    This is what makes synthetic aggregation exact: aligning a mixture equals
    aligning the parts and adding them.
    """
    v = sine(50.3, 2000).unsqueeze(0)
    comps = torch.stack([0.4 * v[0], 0.25 * sine(50.3, 2000, phase=0.6), 0.1 * v[0]]).unsqueeze(0)

    _, ic_parts, mask = cycle_align(v, comps, fs=FS, cycle_size=96, n_cycles=6)
    _, ic_total, _ = cycle_align(v, comps.sum(1), fs=FS, cycle_size=96, n_cycles=6)

    assert (ic_parts.sum(1) - ic_total).abs().max() < 1e-9
    assert mask.all()


def test_rejects_out_of_tolerance_cycles():
    """Cycles whose period is off must be dropped, not stretched onto the grid.

    Half the record is 50 Hz (120 samples at 6 kHz) and half is 40 Hz (150
    samples). Against a nominal 50 Hz with 10% tolerance, only the first half is
    admissible -- so alignment must not silently squeeze 40 Hz cycles onto the
    same grid, which would make them look like a different appliance.
    """
    v = torch.cat([sine(50.0, 600), sine(40.0, 600)]).unsqueeze(0)

    _, _, strict = cycle_align(v, v.clone(), fs=FS, cycle_size=120, f0=50.0, tol=0.1)
    _, _, loose = cycle_align(v, v.clone(), fs=FS, cycle_size=120, f0=50.0, tol=0.5)

    assert int(strict.sum()) == 4, "expected only the 50 Hz half to be admitted"
    assert int(loose.sum()) > int(strict.sum()), "a wider tolerance must admit more cycles"


def test_requested_cycles_are_never_padded_with_duplicates():
    """Regression: the shortfall padding must not re-index a valid cycle.

    An earlier version filled the padded selection slots with index 0, so a
    request for more cycles than the signal contains returned copies of the first
    cycle marked valid -- silently fabricating data.
    """
    # 400 samples at 6 kHz is 3.3 periods of 50 Hz. The signal starts at zero
    # going up, which is not a *rising* crossing (it needs v[k] < 0), so the
    # crossings land near 120, 240 and 360 -- three crossings, two cycles.
    short = sine(50.0, 400).unsqueeze(0)
    vc, _, mask = cycle_align(short, short.clone(), fs=FS, cycle_size=64, n_cycles=20)

    assert int(mask.sum()) == 2
    padded = vc[0][~mask[0]]
    assert torch.equal(padded, torch.zeros_like(padded))


def test_n_cycles_gives_fixed_shape_and_masks_the_shortfall():
    """Fixed output shape by construction is what removes ragged batching."""
    short = sine(50.0, 400).unsqueeze(0)  # ~3 cycles at 6 kHz
    vc, ic, mask = cycle_align(short, short.clone(), fs=FS, cycle_size=64, n_cycles=20)

    assert vc.shape == (1, 20, 64)
    assert ic.shape == (1, 20, 64)
    assert mask.shape == (1, 20)
    assert 0 < int(mask.sum()) < 20
    assert torch.equal(vc[0][~mask[0]], torch.zeros_like(vc[0][~mask[0]]))
    # Real cycles come first, padding last -- so a consumer can slice by count.
    assert bool(mask[0, 0]) and not bool(mask[0, -1])


def test_no_crossings_returns_empty_not_an_exception():
    v = torch.ones(1, 500, dtype=torch.float64)  # never crosses zero
    vc, _, mask = cycle_align(v, v.clone(), fs=FS, cycle_size=32, n_cycles=4)
    assert vc.shape == (1, 4, 32)
    assert not mask.any()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_dtype_is_preserved(dtype):
    v = sine(50.0, 1200).to(dtype).unsqueeze(0)
    vc, ic, mask = cycle_align(v, v.clone(), fs=FS, cycle_size=64, n_cycles=4)
    assert vc.dtype == dtype and ic.dtype == dtype and mask.dtype == torch.bool


def test_is_differentiable():
    """Alignment sits inside the model, so gradients must flow through it."""
    v = sine(50.0, 1200).unsqueeze(0)
    i = (0.5 * sine(50.0, 1200)).unsqueeze(0).requires_grad_(True)
    _, ic, _ = cycle_align(v, i, fs=FS, cycle_size=64, n_cycles=4)
    ic.sum().backward()
    assert i.grad is not None and torch.isfinite(i.grad).all() and i.grad.abs().sum() > 0


def test_module_matches_function():
    v = sine(50.2, 1500).unsqueeze(0)
    mod = CycleAlign(cycle_size=64, n_cycles=5, f0=50.0)
    a = mod(v, v.clone(), fs=FS)
    b = cycle_align(v, v.clone(), fs=FS, cycle_size=64, n_cycles=5, f0=50.0)
    for x, y in zip(a, b, strict=True):
        assert torch.equal(x, y)


def test_auto_f0_handles_mixed_mains_in_one_batch():
    """f0=None infers the period per item, so 50 Hz and 60 Hz can share a batch."""
    v = torch.stack([sine(50.0, 2000), sine(60.0, 2000)])
    _, _, mask = cycle_align(v, v.clone(), fs=FS, cycle_size=64, n_cycles=10, f0=None)
    assert mask.all()

    # A single nominal f0 cannot satisfy both.
    _, _, mask50 = cycle_align(v, v.clone(), fs=FS, cycle_size=64, n_cycles=10, f0=50.0, tol=0.05)
    assert mask50[0].all() and not mask50[1].any()


def test_rejects_bad_shapes():
    v = sine(50.0, 500).unsqueeze(0)
    with pytest.raises(ValueError, match="cycle_size"):
        cycle_align(v, v.clone(), fs=FS, cycle_size=1)
    with pytest.raises(ValueError, match="voltage"):
        cycle_align(v[0], v[0].clone(), fs=FS, cycle_size=32)
    with pytest.raises(ValueError, match="not compatible"):
        cycle_align(v, v[:, :100].clone(), fs=FS, cycle_size=32)
    with pytest.raises(ValueError, match="tol"):
        cycle_align(v, v.clone(), fs=FS, cycle_size=32, tol=5.0)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def test_rising_zero_crossings_are_sub_sample_accurate():
    f0 = 50.0
    v = sine(f0, 1000).unsqueeze(0)
    pos, mask = rising_zero_crossings(v)
    found = pos[0][mask[0]]
    period = FS / f0
    spacing = (found[1:] - found[:-1]).abs()
    assert (spacing - period).abs().max() < 1e-6


@pytest.mark.parametrize("f0", [50.0, 60.0])
def test_estimate_f0(f0):
    v = sine(f0, 6000)
    assert abs(estimate_f0(v, FS).item() - f0) < 1.0


def test_samples_for_cycles_is_enough():
    n = samples_for_cycles(20, FS, 50.0)
    v = sine(49.0, n).unsqueeze(0)  # worst case: slowest plausible mains
    _, _, mask = cycle_align(v, v.clone(), fs=FS, cycle_size=64, n_cycles=20)
    assert mask.all(), "window sizing must guarantee the requested cycle count"
