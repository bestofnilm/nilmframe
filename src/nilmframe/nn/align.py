"""Cycle-invariant alignment (FITPS) in pure PyTorch.

Mains-referenced appliance signatures are far easier to compare when every mains
cycle is resampled onto the same number of points: the mains frequency drifts, so
a fixed-length slice of a waveform is not phase-aligned with the next one. FITPS
("frequency-invariant transformation of periodic signals") locates rising zero
crossings of the voltage to sub-sample precision and resamples the span between
consecutive crossings onto a fixed grid.

This module replaces the pybind11/C++ extension in ``legacy/data/highfreq/core``.
The rewrite is not cosmetic:

* **The C++ interpolated with the wrong weight.** ``fitps.h::allocate`` computed
  ``buffer[k2] + (buffer[k3] - buffer[k2]) * zero_crossing_shifts[0]``, using the
  sub-sample offset of the cycle's *first* zero crossing as the interpolation
  weight for every point in the cycle. The weight must be the fractional part
  ``k1 - k2``. Measured on a 50.7 Hz sine at 6 kHz resampled to 118 points, the
  shipped weight gives 0.0074-0.0212 RMS error against 0.0001-0.0002 for the
  correct one -- 31x to 259x worse depending on where the zero crossing falls
  between samples. That phase dependence is the real damage: the distortion
  changes from cycle to cycle, re-injecting exactly the jitter FITPS exists to
  remove. See ``tests/test_align.py::test_original_cpp_weight_is_phase_dependent``.
* The C++ also decremented ``zero_crossings[0]`` on a ``deque<size_t>`` without
  checking it was non-empty (out-of-range access and unsigned underflow) and
  called ``erase(begin())`` per sample once the buffer filled, which is O(n^2).
* Being a compiled extension, it had to be built before *any* module that
  imported the data layer could load.

The torch version is batched, runs on whatever device the tensors are on -- so it
can sit inside the model and be shared by training and deployment -- and is
differentiable with respect to the signal.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

__all__ = ["CycleAlign", "cycle_align", "estimate_f0", "rising_zero_crossings"]


def estimate_f0(v: Tensor, fs: float) -> Tensor:
    """Estimate the fundamental frequency from the FFT peak.

    Ports ``legacy/data/highfreq/utils.py::fundamental`` to batched torch. Used to
    stamp ``f0`` onto store metadata; alignment itself does not need it (it can
    infer the expected period from the crossing spacing).

    Args:
        v: ``(..., T)`` voltage.
        fs: sampling rate in Hz.

    Returns:
        ``(...)`` fundamental frequency in Hz.

    Example:
        >>> m = nf.example_measurement()
        >>> round(float(nf.nn.estimate_f0(m.v, fs=6000.0)), 1)
        50.0
    """
    if v.shape[-1] < 2:
        raise ValueError("need at least 2 samples to estimate f0")
    spectrum = torch.fft.rfft(v.to(torch.float32) - v.mean(-1, keepdim=True), dim=-1).abs()
    freqs = torch.fft.rfftfreq(v.shape[-1], 1.0 / fs, device=v.device)
    return freqs[spectrum.argmax(-1)]


def rising_zero_crossings(v: Tensor) -> tuple[Tensor, Tensor]:
    """Locate rising zero crossings to sub-sample precision.

    A rising crossing lies between samples ``k`` and ``k+1`` when ``v[k] < 0`` and
    ``v[k+1] >= 0``. Linear interpolation puts it at
    ``k + v[k] / (v[k] - v[k+1])``.

    Args:
        v: ``(B, T)`` voltage.

    Returns:
        ``(positions, mask)``, both ``(B, T-1)``. ``positions`` is meaningful only
        where ``mask`` is true.

    Example:
        >>> m = nf.example_measurement()
        >>> pos, mask = nf.nn.rising_zero_crossings(m.v.unsqueeze(0))
        >>> int(mask.sum())
        24
        >>> [round(float(x), 2) for x in pos[0][mask[0]][:3]]
        [120.0, 240.0, 360.0]
    """
    if v.ndim != 2:
        raise ValueError(f"expected (B, T) voltage, got shape {tuple(v.shape)}")
    prev, nxt = v[:, :-1], v[:, 1:]
    mask = (prev < 0) & (nxt >= 0)
    denom = prev - nxt
    frac = prev / torch.where(denom.abs() < 1e-12, torch.full_like(denom, 1e-12), denom)
    idx = torch.arange(v.shape[1] - 1, device=v.device, dtype=v.dtype)
    return idx.unsqueeze(0) + frac.clamp(0.0, 1.0), mask


def _pad_crossings(pos: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
    """Compact per-item crossing positions into a dense ``(B, Zmax)`` tensor."""
    b, _ = mask.shape
    counts = mask.sum(1)
    zmax = int(counts.max().item())
    if zmax == 0:
        empty = pos.new_zeros((b, 0))
        return empty, empty.to(torch.bool)

    rank = mask.cumsum(1) - 1  # 0-based index of each crossing within its item
    bi, ti = mask.nonzero(as_tuple=True)
    out = pos.new_zeros((b, zmax))
    ok = torch.zeros((b, zmax), dtype=torch.bool, device=pos.device)
    out[bi, rank[bi, ti]] = pos[bi, ti]
    ok[bi, rank[bi, ti]] = True
    return out, ok


def _gather_lerp(x: Tensor, grid: Tensor) -> Tensor:
    """Sample ``x`` at fractional positions ``grid`` with linear interpolation.

    Args:
        x: ``(B, R, T)`` signal, ``R`` an arbitrary flattened channel/component axis.
        grid: ``(B, C, S)`` fractional sample positions.

    Returns:
        ``(B, R, C, S)``.
    """
    b, r, t = x.shape
    c, s = grid.shape[1], grid.shape[2]
    g = grid.reshape(b, 1, c * s).expand(b, r, c * s)
    lo = g.floor()
    w = (g - lo).to(x.dtype)
    lo = lo.long().clamp_(0, max(t - 2, 0))
    hi = (lo + 1).clamp_(0, t - 1)
    x_lo = torch.gather(x, 2, lo)
    x_hi = torch.gather(x, 2, hi)
    # The fractional part -- this is the line the C++ implementation got wrong.
    return torch.lerp(x_lo, x_hi, w).reshape(b, r, c, s)


def cycle_align(
    v: Tensor,
    i: Tensor,
    fs: float,
    cycle_size: int,
    n_cycles: int | None = None,
    f0: float | None = None,
    tol: float = 0.2,
) -> tuple[Tensor, Tensor, Tensor]:
    """Resample each mains cycle onto a fixed-length grid.

    Args:
        v: ``(B, T)`` voltage. Zero crossings are taken from this signal.
        i: ``(B, T)`` or ``(B, K, T)`` current. ``K`` is the per-appliance component
            axis of a submetered aggregate; every component is resampled on the
            same grid, so superposition is preserved exactly.
        fs: sampling rate in Hz.
        cycle_size: samples per output cycle.
        n_cycles: if given, return exactly this many cycles, zero-padded (and
            masked) when fewer valid cycles are present. If ``None``, return as
            many as the longest item in the batch yields.
        f0: nominal mains frequency. If ``None``, the expected period is the
            median observed crossing spacing of each item, which tolerates
            50/60 Hz mixes and off-nominal recordings.
        tol: a cycle is rejected when its length deviates from the expected period
            by more than this fraction. ``0.2`` at 6 kHz/50 Hz is ~24 samples,
            comparable to the C++ default of 20.

    Returns:
        ``(v_cycles, i_cycles, mask)`` where ``v_cycles`` is ``(B, C, cycle_size)``,
        ``i_cycles`` is ``(B, C, cycle_size)`` or ``(B, K, C, cycle_size)``, and
        ``mask`` is ``(B, C)`` true for cycles that are real rather than padding.

    Note:
        Cycles are counted between *consecutive* rising crossings, so a cycle
        spanning a rejected interval is dropped rather than stretched. This
        matches the C++ behaviour of resetting its buffer on an out-of-tolerance
        period.

    Example:
        >>> m = nf.example_measurement('kettle')
        >>> v, i = m.v.unsqueeze(0), m.i.unsqueeze(0)
        >>> vc, ic, mask = nf.nn.cycle_align(v, i, fs=6000.0, cycle_size=128, n_cycles=10)
        >>> tuple(vc.shape), tuple(ic.shape), tuple(mask.shape)
        ((1, 10, 128), (1, 10, 128), (1, 10))
        >>> int(mask.sum())
        10
    """
    if cycle_size < 2:
        raise ValueError(f"cycle_size must be >= 2, got {cycle_size}")
    if v.ndim != 2:
        raise ValueError(f"expected (B, T) voltage, got shape {tuple(v.shape)}")
    if i.shape[0] != v.shape[0] or i.shape[-1] != v.shape[-1]:
        raise ValueError(
            f"current {tuple(i.shape)} is not compatible with voltage {tuple(v.shape)}"
        )
    if not (0.0 < tol < 1.0):
        raise ValueError(f"tol is a fraction of the expected period, got {tol}")

    b, t = v.shape
    squeeze_components = i.ndim == 2
    i3 = i.unsqueeze(1) if squeeze_components else i
    if i3.ndim != 3:
        raise ValueError(f"expected (B, T) or (B, K, T) current, got shape {tuple(i.shape)}")

    pos_all, zc_mask = rising_zero_crossings(v)
    pos, ok = _pad_crossings(pos_all, zc_mask)

    # Candidate cycles are consecutive crossing pairs.
    if pos.shape[1] < 2:
        n_out = n_cycles or 0
        shape_i = (b, i3.shape[1], n_out, cycle_size)
        return (
            v.new_zeros((b, n_out, cycle_size)),
            (
                v.new_zeros(shape_i[:1] + shape_i[2:])
                if squeeze_components
                else v.new_zeros(shape_i)
            ),
            torch.zeros((b, n_out), dtype=torch.bool, device=v.device),
        )

    start, end = pos[:, :-1], pos[:, 1:]
    pair_ok = ok[:, :-1] & ok[:, 1:]
    length = end - start

    if f0 is not None:
        expected = torch.full_like(length, fs / float(f0))
    else:
        # Median observed spacing, ignoring padding.
        masked = torch.where(pair_ok, length, torch.full_like(length, float("nan")))
        expected = masked.nanmedian(dim=1, keepdim=True).values.expand_as(length)
        expected = torch.nan_to_num(expected, nan=float(t))

    valid = pair_ok & ((length - expected).abs() <= tol * expected) & (length > 1.0)

    # Take the first `n_cycles` valid pairs per item, preserving order. A stable
    # argsort on the negated validity flag puts valid pairs first without a
    # per-item Python loop.
    order = torch.argsort((~valid).to(torch.int8), dim=1, stable=True)
    n_valid = int(valid.sum(1).max().item())
    keep = n_valid if n_cycles is None else n_cycles
    keep = max(keep, 0)
    if keep == 0:
        shape_i = (b, i3.shape[1], 0, cycle_size)
        return (
            v.new_zeros((b, 0, cycle_size)),
            (v.new_zeros((b, 0, cycle_size)) if squeeze_components else v.new_zeros(shape_i)),
            torch.zeros((b, 0), dtype=torch.bool, device=v.device),
        )

    if order.shape[1] < keep:  # fewer candidate pairs than requested
        n_pairs = order.shape[1]
        pad = keep - n_pairs
        # The padded slots must *index the appended invalid entries*. Filling them
        # with zeros would point every one of them at pair 0, which is usually
        # valid, and silently duplicate a real cycle into the padding.
        pad_idx = torch.arange(n_pairs, n_pairs + pad, device=v.device, dtype=order.dtype)
        order = torch.cat([order, pad_idx.unsqueeze(0).expand(b, pad)], dim=1)
        valid = torch.cat([valid, torch.zeros((b, pad), dtype=torch.bool, device=v.device)], 1)
        start = torch.cat([start, start.new_zeros((b, pad))], dim=1)
        end = torch.cat([end, end.new_zeros((b, pad))], dim=1)

    sel = order[:, :keep]
    mask = torch.gather(valid, 1, sel)
    sel_start = torch.gather(start, 1, sel)
    sel_end = torch.gather(end, 1, sel)

    # One period sampled at `cycle_size` points, endpoint excluded: the signal is
    # periodic, so including both crossings would duplicate a sample.
    phase = torch.arange(cycle_size, device=v.device, dtype=v.dtype) / cycle_size
    grid = sel_start.unsqueeze(-1) + (sel_end - sel_start).unsqueeze(-1) * phase
    grid = torch.where(mask.unsqueeze(-1), grid, torch.zeros_like(grid))

    v_cycles = _gather_lerp(v.unsqueeze(1), grid).squeeze(1)
    i_cycles = _gather_lerp(i3, grid)
    zero = mask.unsqueeze(-1)
    v_cycles = v_cycles * zero
    i_cycles = i_cycles * zero.unsqueeze(1)
    if squeeze_components:
        i_cycles = i_cycles.squeeze(1)
    return v_cycles, i_cycles, mask


def samples_for_cycles(n_cycles: int, fs: float, f0: float, slack: float = 1.25) -> int:
    """Window length in samples needed to yield ``n_cycles`` aligned cycles.

    Windows are defined in *cycles* rather than samples so that alignment produces
    a fixed output shape; this is what removes ragged batching from the pipeline.
    ``slack`` covers the partial cycles at both ends plus frequency drift.

    Args:
        n_cycles: cycles the window must yield after alignment.
        fs: sampling rate in Hz.
        f0: nominal mains frequency.
        slack: oversizing factor, covering the partial cycles at both ends plus
            frequency drift.

    Returns:
        Window length in samples.

    Example:
        >>> nf.nn.samples_for_cycles(20, fs=6000.0, f0=50.0)
        3150
    """
    return math.ceil((n_cycles + 1) * slack * fs / f0)


class CycleAlign(nn.Module):
    """``nn.Module`` wrapper around :func:`cycle_align`.

    Being a module means the alignment can live inside the model and run on the
    GPU, so training and deployment share one code path.

    Example:
        >>> align = nf.nn.CycleAlign(cycle_size=64, n_cycles=8, f0=50.0)
        >>> align
        CycleAlign(cycle_size=64, n_cycles=8, f0=50.0, tol=0.2)
        >>> m = nf.example_measurement()
        >>> vc, ic, mask = align(m.v.unsqueeze(0), m.i.unsqueeze(0), fs=6000.0)
        >>> tuple(ic.shape)
        (1, 8, 64)
    """

    def __init__(
        self,
        cycle_size: int = 128,
        n_cycles: int | None = None,
        f0: float | None = None,
        tol: float = 0.2,
    ) -> None:
        super().__init__()
        self.cycle_size = int(cycle_size)
        self.n_cycles = n_cycles
        self.f0 = f0
        self.tol = float(tol)

    def forward(self, v: Tensor, i: Tensor, fs: float) -> tuple[Tensor, Tensor, Tensor]:
        """Align a batch of waveforms. See :func:`cycle_align` for the details.

        Args:
            v: ``(B, T)`` voltage; zero crossings are taken from it.
            i: ``(B, T)`` or ``(B, K, T)`` current.
            fs: sampling rate in Hz.

        Returns:
            ``(v_cycles, i_cycles, mask)`` -- see :func:`cycle_align`.
        """
        return cycle_align(
            v,
            i,
            fs=fs,
            cycle_size=self.cycle_size,
            n_cycles=self.n_cycles,
            f0=self.f0,
            tol=self.tol,
        )

    def extra_repr(self) -> str:
        return (
            f"cycle_size={self.cycle_size}, n_cycles={self.n_cycles}, f0={self.f0}, tol={self.tol}"
        )
