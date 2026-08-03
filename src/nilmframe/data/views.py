"""Rate views: how a window of stored signal becomes model input.

This module is where the paper's central experiment lives. ``edframe_concept_note``
asks for a fair comparison of low- and high-frequency NILM "using exactly the same
underlying data and samples". That is only fair if the low-frequency series is
*derived from* the same waveform the high-frequency model sees, by the same code,
over the same window boundaries -- otherwise the two arms differ by an
uncontrolled preprocessing pipeline and the comparison proves nothing.

So a view is a small object that answers three questions:

* how many samples of stored signal does one window need, given a channel's rate?
* what tensors does the model get?
* what is the measured aggregate power of this window?

That last one matters more than it looks. The predecessor rescaled predictions by
``Y_true.sum(1)`` -- the sum of the labels -- which is not available at inference
and inflated every reported metric. The aggregate power computed here comes from
the *input signal*, is available at deployment, and is what the conservation term
in the loss is anchored to.

Three views, one harness:

``HighFreqView(align="fitps")``
    Cycle-aligned waveform, ``(n_cycles, cycle_size)``.
``HighFreqView(align=None)``
    The same window chunked at a fixed length and resampled to the same shape --
    the no-alignment control. Identical tensor shape, so the identical backbone
    consumes both and any difference is attributable to alignment.
``LowFreqView(rate_hz=1.0)``
    Active power at a low rate, derived from the same waveform when the channel
    stores one, or read directly when the dataset is natively low-frequency.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch
from torch import Tensor

from nilmframe.nn.align import cycle_align, samples_for_cycles

__all__ = ["HighFreqView", "LowFreqView", "View", "active_power", "resample_to"]


def active_power(v: Tensor, i: Tensor, dim: int = -1) -> Tensor:
    """Real power in watts: the mean of the instantaneous product.

    This is the quantity a meter reports, and the one the framework treats as an
    *input* rather than a label -- a conservation term states its
    conservation term over it.

    Args:
        v: voltage.
        i: current, broadcastable against ``v``.
        dim: axis to average over. The last by default, which is time.

    Returns:
        ``v`` and ``i`` with ``dim`` reduced away.

    Example:
        >>> from nilmframe.data.views import active_power
        >>> m = nf.example_measurement('kettle')
        >>> round(float(active_power(m.v, m.i)), 1)
        2926.1
    """
    return (v * i).mean(dim)


def apparent_power(v: Tensor, i: Tensor, dim: int = -1) -> Tensor:
    """Product of the RMS values."""
    return v.pow(2).mean(dim).sqrt() * i.pow(2).mean(dim).sqrt()


def resample_to(x: Tensor, size: int) -> Tensor:
    """Linearly resample the last axis to ``size`` points.

    Used by the no-alignment control so that both high-frequency arms produce the
    same tensor shape and can share a backbone.

    Example:
        >>> from nilmframe.data.views import resample_to
        >>> tuple(resample_to(torch.randn(3, 5, 100), 32).shape)
        (3, 5, 32)
    """
    shape = x.shape
    flat = x.reshape(-1, 1, shape[-1])
    out = torch.nn.functional.interpolate(flat, size=size, mode="linear", align_corners=False)
    return out.reshape(*shape[:-1], size)


@runtime_checkable
class View(Protocol):
    """What a dataset needs from a view.

    A view is the one place that decides how a stretch of stored samples becomes
    model input. :class:`WindowDataset` never touches raw signals itself -- it asks
    the view which quantities to read, how wide a window is, and what tensors come
    out. Swapping :class:`HighFreqView` for :class:`LowFreqView` therefore changes
    the entire input pipeline without changing the dataset, the split, or the
    training loop.

    Note:
        A structural protocol, not a base class. Anything implementing these four
        methods works -- there is nothing to subclass and nothing to register.

    Example:
        Both shipped views satisfy it, and so does a plain object of your own:

        >>> from nilmframe.data import View
        >>> isinstance(nf.HighFreqView(n_cycles=4, cycle_size=64), View)
        True
        >>> isinstance(nf.LowFreqView(rate_hz=1.0, n_steps=60), View)
        True

        >>> class Envelope:  # window of |i|, decimated: the crudest useful view
        ...     def required_quantities(self, available):
        ...         return ('i',) if 'i' in available else ()
        ...     def supports(self, quantities, fs, f0):
        ...         return 'i' in quantities
        ...     def window_samples(self, fs, f0):
        ...         return int(fs)
        ...     def __call__(self, signals, fs, f0):
        ...         x = signals['i'].abs()
        ...         return {'p': x.reshape(-1, 100).mean(-1), 'p_total': x.mean()}
        >>> isinstance(Envelope(), View)
        True
    """

    def required_quantities(self, available: set[str]) -> tuple[str, ...]:
        """Store quantities this view wants, most-preferred first.

        Args:
            available: what the channel actually has, e.g. ``{"v", "i"}``.

        Returns:
            The quantities to read, in preference order. Empty means this view
            cannot work with what the channel offers.
        """
        ...

    def supports(self, quantities: set[str], fs: float, f0: float) -> bool:
        """Whether this view can run on a channel with these properties.

        Used by :class:`WindowDataset` to skip channels rather than fail on them,
        so a mixed-rate store yields whatever subset the view can handle.

        Args:
            quantities: what the channel stores.
            fs: the channel's sampling rate, in hertz.
            f0: the channel's mains frequency estimate, in hertz.

        Returns:
            True when a window from this channel can be produced.
        """
        ...

    def window_samples(self, fs: float, f0: float) -> int:
        """Window width in stored samples, at this channel's rate.

        Windows are specified in physical units -- cycles or seconds -- so the
        sample count differs per channel. The index calls this once per channel.

        Args:
            fs: the channel's sampling rate, in hertz.
            f0: the channel's mains frequency estimate, in hertz.

        Returns:
            Number of consecutive samples one window spans.
        """
        ...

    def __call__(self, signals: dict[str, Tensor], fs: float, f0: float) -> dict[str, Tensor]:
        """Turn one window of raw samples into model input.

        Args:
            signals: raw window, keyed by quantity, each ``(window_samples,)``.
            fs: the channel's sampling rate, in hertz.
            f0: the channel's mains frequency estimate, in hertz.

        Returns:
            Model input tensors. Must include ``p_total``, the measured active
            power of the window -- the conservation term in
            a conservation term is computed against it.
        """
        ...


@dataclass(frozen=True)
class HighFreqView:
    """Waveform view: one window becomes ``(n_cycles, cycle_size)`` per quantity.

    Args:
        n_cycles: mains cycles per window. Windows are specified in cycles rather
            than samples so that alignment yields a fixed shape and ragged batching
            never arises.
        cycle_size: samples per cycle after resampling.
        align: ``"fitps"`` for zero-crossing alignment, ``None`` for the
            fixed-length control.
        f0: nominal mains frequency. ``None`` uses the channel's stored estimate.
        tol: alignment tolerance, as a fraction of the expected period.
        slack: window oversizing factor, to guarantee ``n_cycles`` survive
            alignment even with frequency drift and partial cycles at the edges.

    Returns from ``__call__``:
        ``v``, ``i``: ``(n_cycles, cycle_size)``
        ``cycle_mask``: ``(n_cycles,)`` true where a real cycle was found
        ``p_total``: scalar, measured active power of the window

    Example:
        >>> from nilmframe.data import HighFreqView, LowFreqView, WindowDataset, collate_windows
        >>> view = HighFreqView(n_cycles=20, cycle_size=128, align='fitps')
        >>> view.output_shape()
        (20, 128)
        >>> view.window_samples(fs=6000.0, f0=50.0)
        3276
        >>> HighFreqView(align=None).describe()
        {'view': 'highfreq', 'n_cycles': 20, 'cycle_size': 128, 'align': 'none'}
    """

    n_cycles: int = 20
    cycle_size: int = 128
    align: str | None = "fitps"
    f0: float | None = None
    tol: float = 0.2
    slack: float = 1.3

    name = "highfreq"

    def __post_init__(self) -> None:
        if self.align not in (None, "fitps"):
            raise ValueError(f"align must be 'fitps' or None, got {self.align!r}")
        if self.n_cycles < 1 or self.cycle_size < 2:
            raise ValueError("n_cycles must be >= 1 and cycle_size >= 2")

    def required_quantities(self, available: set[str]) -> tuple[str, ...]:
        """Which stored quantities this view reads.

        Raises:
            ValueError: when the channel cannot serve it.

        Example:
            >>> from nilmframe.data import HighFreqView, LowFreqView, WindowDataset, collate_windows
            >>> HighFreqView().required_quantities({'v', 'i', 'p'})
            ('v', 'i')
        """
        if not {"v", "i"} <= available:
            raise ValueError(
                f"HighFreqView needs voltage and current; channel offers {sorted(available)}"
            )
        return ("v", "i")

    def supports(self, quantities: set[str], fs: float, f0: float) -> bool:
        """Can this view render such a channel at all?

        A store legitimately holds a house's 16 kHz mains next to its 1/6 Hz
        submeters, and a split spans both. Each arm of a sweep then has to select
        the channels it can actually read, rather than failing on the first one it
        cannot -- which is what makes one split serve every arm.

        Example:
            >>> from nilmframe.data import HighFreqView, LowFreqView, WindowDataset, collate_windows
            >>> view = HighFreqView()
            >>> view.supports({'v', 'i'}, fs=6000.0, f0=50.0)
            True
            >>> view.supports({'p'}, fs=6000.0, f0=50.0)
            False
            >>> view.supports({'v', 'i'}, fs=60.0, f0=50.0)
            False
        """
        if not {"v", "i"} <= quantities:
            return False
        # Two samples per mains cycle is the absolute floor; below it there is no
        # waveform left to align.
        return fs >= 2 * (self.f0 or f0 or 50.0)

    def window_samples(self, fs: float, f0: float) -> int:
        """Window length in samples -- *the same whether or not alignment is on*.

        Alignment needs slack: it discards partial cycles at the edges and any
        whose period is out of tolerance, so it must be handed more signal than
        ``n_cycles`` worth. If the unaligned control took exactly ``n_cycles``
        nominal periods instead, the two arms would consume different spans and
        produce different numbers of windows -- a confound in the one comparison
        this framework exists to make. Both take the oversized window; the control
        simply chunks its leading portion and ignores the tail.

        Example:
            >>> from nilmframe.data import HighFreqView, LowFreqView, WindowDataset, collate_windows
            >>> HighFreqView(n_cycles=10).window_samples(fs=6000.0, f0=50.0)
            1716
            >>> HighFreqView(n_cycles=10, align=None).window_samples(fs=6000.0, f0=50.0)
            1716
        """
        f0 = self.f0 or f0
        if round(fs / f0) < 2:
            raise ValueError(
                f"a {fs} Hz channel cannot carry a {f0} Hz waveform; "
                "check View.supports() before building a window index"
            )
        return samples_for_cycles(self.n_cycles, fs, f0, slack=self.slack)

    def __call__(self, signals: dict[str, Tensor], fs: float, f0: float) -> dict[str, Tensor]:
        v, i = signals["v"], signals["i"]
        batched = v.ndim == 2
        if not batched:
            v, i = v.unsqueeze(0), i.unsqueeze(0)

        if self.align is None:
            period = round(fs / (self.f0 or f0))
            usable = self.n_cycles * period
            if v.shape[-1] < usable:
                raise ValueError(
                    f"window of {v.shape[-1]} samples is too short for "
                    f"{self.n_cycles} cycles of {period} samples"
                )
            chunks_v = v[:, :usable].reshape(v.shape[0], self.n_cycles, period)
            chunks_i = i[:, :usable].reshape(i.shape[0], self.n_cycles, period)
            vc = resample_to(chunks_v, self.cycle_size)
            ic = resample_to(chunks_i, self.cycle_size)
            mask = torch.ones(v.shape[0], self.n_cycles, dtype=torch.bool, device=v.device)
        else:
            vc, ic, mask = cycle_align(
                v,
                i,
                fs=fs,
                cycle_size=self.cycle_size,
                n_cycles=self.n_cycles,
                f0=self.f0 or (f0 if math.isfinite(f0) else None),
                tol=self.tol,
            )

        # Measure the aggregate from the *represented* signal, not the raw window.
        # The raw window is deliberately oversized so alignment can discard partial
        # and out-of-tolerance cycles, so its power describes samples the model
        # never sees. Anchoring p_total to the returned tensors is what makes
        # `sum(per-appliance power) == p_total` hold exactly through superposition
        # -- see nilmframe.data.mixing.
        n_valid = mask.sum(-1).clamp_min(1)
        p_total = (vc * ic).sum(dim=(-2, -1)) / (n_valid * self.cycle_size)

        out = {"v": vc, "i": ic, "cycle_mask": mask, "p_total": p_total}
        return out if batched else {k: t.squeeze(0) for k, t in out.items()}

    def output_shape(self) -> tuple[int, int]:
        """Shape of one window's tensors, ``(n_cycles, cycle_size)``.

        Example:
            >>> from nilmframe.data import HighFreqView, LowFreqView, WindowDataset, collate_windows
            >>> HighFreqView(n_cycles=12, cycle_size=64).output_shape()
            (12, 64)
        """
        return (self.n_cycles, self.cycle_size)

    def describe(self) -> dict:
        """The view's settings, for a results table or a manifest.

        Example:
            >>> from nilmframe.data import HighFreqView, LowFreqView, WindowDataset, collate_windows
            >>> HighFreqView(n_cycles=4, cycle_size=32).describe()
            {'view': 'highfreq', 'n_cycles': 4, 'cycle_size': 32, 'align': 'fitps'}
        """
        return {
            "view": self.name,
            "n_cycles": self.n_cycles,
            "cycle_size": self.cycle_size,
            "align": self.align or "none",
        }


@dataclass(frozen=True)
class LowFreqView:
    """Low-rate power view: one window becomes ``(n_steps,)`` watts.

    When the channel stores a waveform, the series is *derived* from it by block
    averaging -- the same samples the high-frequency arm sees, reduced by this
    code. When the channel natively stores power (a smart-meter dataset), it is
    block-averaged to the requested rate instead. Either way the window boundaries
    are the same, which is the condition for the comparison to mean anything.

    Args:
        rate_hz: output rate.
        n_steps: output samples per window, so the window spans
            ``n_steps / rate_hz`` seconds.
        quantity: ``"active"`` (mean of v*i) or ``"apparent"`` (product of RMS).

    Returns from ``__call__``:
        ``p``: ``(n_steps,)`` watts
        ``p_total``: scalar, the window mean of ``p``

    Example:
        >>> from nilmframe.data import HighFreqView, LowFreqView, WindowDataset, collate_windows
        >>> view = LowFreqView(rate_hz=1.0, n_steps=60)
        >>> view.output_shape()
        (60,)
        >>> view.window_seconds()
        60.0
        >>> view.window_samples(fs=6000.0, f0=50.0)
        360000
    """

    rate_hz: float = 1.0
    n_steps: int = 60
    quantity: str = "active"

    name = "lowfreq"

    def __post_init__(self) -> None:
        if self.quantity not in ("active", "apparent"):
            raise ValueError(f"quantity must be 'active' or 'apparent', got {self.quantity!r}")
        if self.rate_hz <= 0 or self.n_steps < 1:
            raise ValueError("rate_hz must be positive and n_steps >= 1")

    def required_quantities(self, available: set[str]) -> tuple[str, ...]:
        """Prefer the waveform when there is one; fall back to a stored power series.

        Example:
            >>> from nilmframe.data import HighFreqView, LowFreqView, WindowDataset, collate_windows
            >>> LowFreqView().required_quantities({'v', 'i'})
            ('v', 'i')
            >>> LowFreqView().required_quantities({'p'})
            ('p',)
        """
        if {"v", "i"} <= available:
            return ("v", "i")
        if "p" in available:
            if self.quantity == "apparent":
                raise ValueError("apparent power needs voltage and current, not a stored p")
            return ("p",)
        raise ValueError(
            f"LowFreqView needs either (v, i) or p; channel offers {sorted(available)}"
        )

    def supports(self, quantities: set[str], fs: float, f0: float) -> bool:
        """Can this view render such a channel.

        A power view reads either a waveform, by reducing it, or a stored power
        series -- but it cannot upsample one that is already slower than its rate.

        Example:
            >>> from nilmframe.data import HighFreqView, LowFreqView, WindowDataset, collate_windows
            >>> view = LowFreqView(rate_hz=1.0)
            >>> view.supports({'p'}, fs=1.0, f0=50.0)
            True
            >>> view.supports({'v', 'i'}, fs=6000.0, f0=50.0)
            True
            >>> view.supports({'p'}, fs=0.1, f0=50.0)
            False
        """
        if not ({"v", "i"} <= quantities or "p" in quantities):
            return False
        if self.quantity == "apparent" and not {"v", "i"} <= quantities:
            return False
        # Downsampling only: a 1/6 Hz channel cannot produce a 1 Hz series.
        return fs >= self.rate_hz

    def window_seconds(self) -> float:
        """Seconds of signal one window spans.

        Example:
            >>> from nilmframe.data import HighFreqView, LowFreqView, WindowDataset, collate_windows
            >>> LowFreqView(rate_hz=2.0, n_steps=30).window_seconds()
            15.0
        """
        return self.n_steps / self.rate_hz

    def window_samples(self, fs: float, f0: float) -> int:
        """Stored samples one window needs at this channel's rate.

        Example:
            >>> from nilmframe.data import HighFreqView, LowFreqView, WindowDataset, collate_windows
            >>> LowFreqView(rate_hz=1.0, n_steps=10).window_samples(fs=6000.0, f0=50.0)
            60000
        """
        # Round up so every output step has at least one input sample, then let the
        # block reduction trim.
        return math.ceil(self.window_seconds() * fs)

    def __call__(self, signals: dict[str, Tensor], fs: float, f0: float) -> dict[str, Tensor]:
        if "v" in signals and "i" in signals:
            v, i = signals["v"], signals["i"]
            batched = v.ndim == 2
            if not batched:
                v, i = v.unsqueeze(0), i.unsqueeze(0)
            block = max(1, v.shape[-1] // self.n_steps)
            usable = block * self.n_steps
            vb = v[..., :usable].reshape(*v.shape[:-1], self.n_steps, block)
            ib = i[..., :usable].reshape(*i.shape[:-1], self.n_steps, block)
            p = active_power(vb, ib) if self.quantity == "active" else apparent_power(vb, ib)
        else:
            raw = signals["p"]
            batched = raw.ndim == 2
            if not batched:
                raw = raw.unsqueeze(0)
            block = max(1, raw.shape[-1] // self.n_steps)
            usable = block * self.n_steps
            p = raw[..., :usable].reshape(*raw.shape[:-1], self.n_steps, block).mean(-1)

        out = {"p": p, "p_total": p.mean(-1)}
        return out if batched else {k: t.squeeze(0) for k, t in out.items()}

    def output_shape(self) -> tuple[int]:
        """Shape of one window's series, ``(n_steps,)``.

        Example:
            >>> from nilmframe.data import HighFreqView, LowFreqView, WindowDataset, collate_windows
            >>> LowFreqView(n_steps=42).output_shape()
            (42,)
        """
        return (self.n_steps,)

    def describe(self) -> dict:
        """The view's settings, for a results table or a manifest.

        Example:
            >>> from nilmframe.data import HighFreqView, LowFreqView, WindowDataset, collate_windows
            >>> LowFreqView(rate_hz=0.5, n_steps=8).describe()
            {'view': 'lowfreq', 'rate_hz': 0.5, 'n_steps': 8, 'quantity': 'active'}
        """
        return {
            "view": self.name,
            "rate_hz": self.rate_hz,
            "n_steps": self.n_steps,
            "quantity": self.quantity,
        }
