"""``Measurement`` -- one concrete measurement you can hold and poke at.

The training path hands back dicts of tensors, which is right for a ``DataLoader``
and wrong for sitting in a notebook trying to understand a signal. This is the
object for the second case: dot into it, chain operations, plot it.

It is deliberately **not** what ``HighFreqSample`` was. That class *was* the
storage -- the dataset held a Python list of them, so the whole corpus lived in
memory -- and it carried mutable pending edits (``__v_modified__``, ``.save()``)
that made every method's behaviour depend on invisible state. Three of its bugs
came straight out of that design: ``n_components`` read the shape of an
already-summed array so every recording looked aggregated, ``copy()`` collapsed
multi-component samples down to one, and ``+`` raised on the first call because it
forgot to pass ``fs``.

This one is a **lens**. The store stays lazy and memory-mapped; a measurement is
created on demand over a window of it, holds plain torch tensors, and is
immutable -- every operation returns a new one. Nothing you do to it can corrupt
the corpus, because it does not own the corpus.

    >>> m = store.measurement("ukdale-house_1-run_0-mains#2", seconds=2)   # doctest: +SKIP
    >>> m.aligned(cycle_size=128).lowpass(8).active_power()               # doctest: +SKIP
    >>> (m1 + m2).vi_image()                                              # doctest: +SKIP

``.v`` and ``.i`` are torch tensors, so anything torch does still works; ``.batch()``
hands the same window to a model in the shape the dataset would have produced.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field, replace
from typing import Any

import torch
from torch import Tensor

__all__ = ["Measurement"]


def _as_tensor(x: Any, name: str) -> Tensor:
    if x is None:
        return None
    if isinstance(x, Tensor):
        return x.detach()
    out = torch.as_tensor(x)
    if not out.is_floating_point():
        out = out.float()
    return out


@dataclass(frozen=True)
class Measurement:
    """A window of one channel, as an object rather than a dict.

    Attributes:
        fs: sampling rate in Hz.
        f0: mains frequency, where known.
        t0: absolute start time in seconds, where known.
        appliances: what is in this measurement, one name per current component.
        source: provenance -- channel id, start sample, whatever produced it.
        meta: anything else worth carrying.

    Note:
        The current is stored with an explicit component axis, always. ``.i`` sums
        it (the aggregate, which is what a meter sees) while ``.n_components`` and
        ``.components`` read the axis itself. Conflating those two is exactly the
        bug that made the predecessor report every single-appliance recording as
        aggregated.

    Example:
        >>> m = nf.example_measurement('kettle')
        >>> m
        Measurement(waveform raw, 6000Hz, 0.500s, kettle, 2926W)
        >>> m.n_components
        1
        >>> round(float(m.active_power()), 1)
        2926.1
        >>> m.aligned(cycle_size=128).n_cycles
        23
    """

    _v: Tensor | None
    _i: Tensor | None
    _p: Tensor | None
    fs: float
    f0: float | None = None
    t0: float = 0.0
    appliances: tuple[str, ...] = ()
    source: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    # -- construction ------------------------------------------------------- #

    @classmethod
    def from_vi(
        cls,
        v,
        i,
        fs: float,
        *,
        f0: float | None = None,
        t0: float = 0.0,
        appliances: tuple[str, ...] | list[str] = (),
        source: str = "",
        **meta,
    ) -> Measurement:
        """Build from a voltage and a current.

        ``i`` may be ``(T,)``/``(C, S)`` for one appliance or ``(K, T)``/``(K, C, S)``
        for a superposition of ``K`` of them.

        Example:
            >>> import math
            >>> t = torch.arange(600) / 6000
            >>> v = 325 * torch.sin(2 * math.pi * 50 * t)
            >>> i = 9 * torch.sin(2 * math.pi * 50 * t)
            >>> nf.Measurement.from_vi(v, i, fs=6000, f0=50, appliances=['kettle'])
            Measurement(waveform raw, 6000Hz, 0.100s, kettle, 1462W)
        """
        v_t, i_t = _as_tensor(v, "v"), _as_tensor(i, "i")
        if v_t is None or i_t is None:
            raise ValueError("from_vi needs both a voltage and a current")
        if i_t.ndim == v_t.ndim:  # no component axis given
            i_t = i_t.unsqueeze(0)
        if i_t.ndim != v_t.ndim + 1:
            raise ValueError(
                f"current {tuple(i_t.shape)} is not compatible with voltage {tuple(v_t.shape)}"
            )
        if i_t.shape[1:] != v_t.shape:
            raise ValueError(
                f"each current component must match the voltage: "
                f"{tuple(i_t.shape[1:])} != {tuple(v_t.shape)}"
            )
        return cls(
            _v=v_t,
            _i=i_t,
            _p=None,
            fs=float(fs),
            f0=f0,
            t0=t0,
            appliances=tuple(appliances),
            source=source,
            meta=meta,
        )

    @classmethod
    def from_power(
        cls,
        p,
        fs: float,
        *,
        t0: float = 0.0,
        appliances: tuple[str, ...] | list[str] = (),
        source: str = "",
        **meta,
    ) -> Measurement:
        """Build from a power series, for a low-rate channel.

        Example:
            >>> watts = torch.tensor([0., 0., 2000., 2000., 0.])
            >>> m = nf.Measurement.from_power(watts, fs=1/6, appliances=['kettle'])
            >>> m.kind, m.n_samples
            ('power', 5)
            >>> round(float(m.active_power()), 1)
            800.0
        """
        p_t = _as_tensor(p, "p")
        if p_t.ndim == 1:
            p_t = p_t.unsqueeze(0)
        return cls(
            _v=None,
            _i=None,
            _p=p_t,
            fs=float(fs),
            f0=None,
            t0=t0,
            appliances=tuple(appliances),
            source=source,
            meta=meta,
        )

    @classmethod
    def from_item(cls, item: dict, *, fs: float, f0: float | None = None) -> Measurement:
        """Build from a :class:`~nilmframe.data.WindowDataset` item.

        Example:
            >>> from nilmframe.data import HighFreqView, WindowDataset
            >>> view = HighFreqView(n_cycles=5, cycle_size=64)
            >>> ds = WindowDataset(store, store.submeters().channel_id.tolist(), view=view)
            >>> nf.Measurement.from_item(ds[0], fs=6000.0, f0=50.0)
            Measurement(waveform 5x64, 6000Hz, 0.053s, 2925W)
        """
        source = f"{item.get('channel', '')}@{item.get('start', 0)}"
        if "v" in item and "i" in item:
            return cls.from_vi(item["v"], item["i"], fs, f0=f0, source=source)
        return cls.from_power(item["p"], fs, source=source)

    # -- what is it --------------------------------------------------------- #

    @property
    def kind(self) -> str:
        """``"waveform"`` if this carries v and i, ``"power"`` if a power series.

        Example:
            >>> nf.example_measurement().kind
            'waveform'
            >>> nf.Measurement.from_power(torch.ones(10), fs=1.0).kind
            'power'
        """
        return "waveform" if self._v is not None else "power"

    @property
    def is_waveform(self) -> bool:
        return self._v is not None

    @property
    def aligned_(self) -> bool:
        """Has this been cycle-aligned? Aligned data is ``(n_cycles, cycle_size)``.

        Example:
            >>> m = nf.example_measurement()
            >>> m.aligned_
            False
            >>> m.aligned(cycle_size=64).aligned_
            True
        """
        return self.is_waveform and self._v.ndim == 2

    @property
    def n_components(self) -> int:
        """Number of superposed appliances. Reads the component axis, not the data.

        Example:
            >>> a = nf.example_measurement('kettle')
            >>> b = nf.example_measurement('fridge')
            >>> a.n_components
            1
            >>> (a + b).n_components
            2
        """
        return int((self._i if self.is_waveform else self._p).shape[0])

    @property
    def n_samples(self) -> int:
        """Samples of signal, counting every cycle when aligned.

        Example:
            >>> nf.example_measurement(seconds=0.5).n_samples
            3000
        """
        base = self._v if self.is_waveform else self._p[0]
        return int(base.numel()) if self.is_waveform else int(base.shape[-1])

    @property
    def n_cycles(self) -> int:
        """Mains cycles, once aligned.

        Example:
            >>> nf.example_measurement(seconds=0.5).aligned(cycle_size=128).n_cycles
            23
        """
        if not self.aligned_:
            raise AttributeError("not cycle-aligned; call .aligned() first")
        return int(self._v.shape[0])

    @property
    def cycle_size(self) -> int:
        """Points per cycle, once aligned.

        Example:
            >>> nf.example_measurement().aligned(cycle_size=96).cycle_size
            96
        """
        if not self.aligned_:
            raise AttributeError("not cycle-aligned; call .aligned() first")
        return int(self._v.shape[1])

    @property
    def duration(self) -> float:
        """Seconds of signal.

        Example:
            >>> round(nf.example_measurement(seconds=0.25).duration, 3)
            0.25
        """
        return self.n_samples / self.fs

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the stored signal, component axis first.

        Example:
            >>> nf.example_measurement(seconds=0.1).shape
            (1, 600)
        """
        return tuple((self._i if self.is_waveform else self._p).shape)

    # -- the signals -------------------------------------------------------- #

    @property
    def v(self) -> Tensor:
        """Voltage.

        Example:
            >>> v = nf.example_measurement().v
            >>> type(v).__name__, tuple(v.shape)
            ('Tensor', (3000,))
            >>> round(float(v.abs().max()), 1)
            325.3
        """
        if self._v is None:
            raise AttributeError("this is a power measurement; it has no voltage")
        return self._v

    @property
    def i(self) -> Tensor:
        """Total current -- the superposition, which is what a meter sees.

        Example:
            >>> a, b = nf.example_measurement('kettle'), nf.example_measurement('fridge')
            >>> mix = a + b
            >>> bool(torch.allclose(mix.i, a.i + b.i))
            True
        """
        if self._i is None:
            raise AttributeError("this is a power measurement; it has no current")
        return self._i.sum(0)

    @property
    def p(self) -> Tensor:
        """Instantaneous power for a waveform, or the stored series for a power channel.

        Example:
            >>> p = nf.example_measurement().p
            >>> tuple(p.shape)
            (3000,)
            >>> round(float(p.mean()), 1)
            2926.1
        """
        if self._p is not None:
            return self._p.sum(0)
        return self.v * self.i

    @property
    def components(self) -> tuple[Measurement, ...]:
        """One measurement per superposed appliance.

        Example:
            >>> mix = nf.example_measurement('kettle') + nf.example_measurement('fridge')
            >>> [c.appliances[0] for c in mix.components]
            ['kettle', 'fridge']
            >>> [c.n_components for c in mix.components]
            [1, 1]
        """
        names = self.appliances or ("",) * self.n_components
        if self.is_waveform:
            return tuple(
                replace(self, _i=self._i[k : k + 1], appliances=(names[k],))
                for k in range(self.n_components)
            )
        return tuple(
            replace(self, _p=self._p[k : k + 1], appliances=(names[k],))
            for k in range(self.n_components)
        )

    # -- electrical quantities ---------------------------------------------- #

    @property
    def vrms(self) -> Tensor:
        """RMS voltage in volts.

        Example:
            >>> round(float(nf.example_measurement().vrms), 1)
            230.0
        """
        return self.v.pow(2).mean().sqrt()

    @property
    def irms(self) -> Tensor:
        """RMS current in amperes, of the superposition.

        Example:
            >>> round(float(nf.example_measurement('kettle').irms), 2)
            12.73
        """
        return self.i.pow(2).mean().sqrt()

    def active_power(self, per_component: bool = False) -> Tensor:
        """Real power in watts. ``per_component`` breaks it down per appliance.

        Example:
            >>> mix = nf.example_measurement('kettle') + nf.example_measurement('fridge')
            >>> round(float(mix.active_power()), 1)
            3163.6
            >>> [round(float(x), 1) for x in mix.active_power(per_component=True)]
            [2926.1, 237.5]
        """
        if not self.is_waveform:
            series = self._p if per_component else self._p.sum(0, keepdim=True)
            out = series.flatten(1).mean(-1)
            return out if per_component else out.squeeze(0)
        if per_component:
            return (self._v.unsqueeze(0) * self._i).flatten(1).mean(-1)
        return (self.v * self.i).mean()

    def apparent_power(self) -> Tensor:
        """Product of the RMS values, in volt-amperes.

        Example:
            >>> round(float(nf.example_measurement('fridge').apparent_power()), 1)
            292.7
        """
        return self.vrms * self.irms

    def reactive_power(self) -> Tensor:
        """The part of the apparent power that does no work.

        Example:
            >>> round(float(nf.example_measurement('kettle').reactive_power()), 1)
            87.8
            >>> round(float(nf.example_measurement('fridge').reactive_power()), 1)
            171.2
        """
        s, p = self.apparent_power(), self.active_power()
        return (s.pow(2) - p.pow(2)).clamp_min(0).sqrt()

    def power_factor(self) -> Tensor:
        """Active over apparent power: 1 for a resistive load, less for a reactive one.

        Example:
            >>> round(float(nf.example_measurement('kettle').power_factor()), 3)
            1.0
            >>> round(float(nf.example_measurement('fridge').power_factor()), 3)
            0.811
        """
        return self.active_power() / self.apparent_power().clamp_min(1e-12)

    # -- reshaping ---------------------------------------------------------- #

    def window(self, start: int = 0, samples: int | None = None) -> Measurement:
        """A sub-window, in samples.

        Example:
            >>> m = nf.example_measurement(seconds=0.5)
            >>> m.window(0, 600).n_samples
            600
            >>> round(m.window(600, 600).t0, 3)
            0.1
        """
        stop = self.n_samples if samples is None else start + samples
        if self.aligned_:
            raise ValueError("slice before aligning; cycle indices are not samples")
        cut = slice(start, stop)
        if self.is_waveform:
            return replace(self, _v=self._v[cut], _i=self._i[:, cut], t0=self.t0 + start / self.fs)
        return replace(self, _p=self._p[:, cut], t0=self.t0 + start / self.fs)

    def seconds(self, start: float = 0.0, duration: float | None = None) -> Measurement:
        """A sub-window, in seconds.

        Example:
            >>> m = nf.example_measurement(seconds=1.0)
            >>> round(m.seconds(0.2, 0.3).duration, 3)
            0.3
        """
        return self.window(
            int(start * self.fs), None if duration is None else int(duration * self.fs)
        )

    def aligned(
        self,
        cycle_size: int = 128,
        n_cycles: int | None = None,
        *,
        f0: float | None = None,
        tol: float = 0.2,
    ) -> Measurement:
        """Cycle-align: resample every mains cycle onto a fixed grid.

        Returns a measurement whose signals are ``(n_cycles, cycle_size)``.

        Example:
            >>> m = nf.example_measurement(seconds=0.5)
            >>> a = m.aligned(cycle_size=128)
            >>> a.aligned_, a.n_cycles, a.cycle_size
            (True, 23, 128)
            >>> tuple(a.i.shape)
            (23, 128)
        """
        from nilmframe.nn.align import cycle_align

        if not self.is_waveform:
            raise TypeError("cycle alignment needs a waveform, not a power series")
        if self.aligned_:
            return self

        vc, ic, mask = cycle_align(
            self._v.unsqueeze(0),
            self._i.unsqueeze(0) if self._i.ndim == 2 else self._i,
            fs=self.fs,
            cycle_size=cycle_size,
            n_cycles=n_cycles,
            f0=f0 if f0 is not None else self.f0,
            tol=tol,
        )
        keep = mask[0]
        return replace(
            self,
            _v=vc[0][keep],
            _i=ic[0][..., keep, :] if ic.ndim == 4 else ic[0][keep].unsqueeze(0),
            meta={**self.meta, "rejected_cycles": int((~keep).sum())},
        )

    def lowpass(self, n_harmonics: int = 8) -> Measurement:
        """Keep only the first ``n_harmonics`` harmonics of each cycle.

        Example:
            >>> a = nf.example_measurement('laptop').aligned(cycle_size=128)
            >>> [round(float(x), 3) for x in a.harmonics(5)]
            [0.0, 1.0, 0.55, 0.399, 0.0]
            >>> [round(float(x), 3) for x in a.lowpass(3).harmonics(5)]
            [0.0, 1.0, 0.55, 0.0, 0.0]
        """
        from nilmframe.nn.repr import HarmonicLowpass

        if not self.aligned_:
            raise ValueError("harmonics are only meaningful on aligned cycles; call .aligned()")
        filt = HarmonicLowpass(n_harmonics)
        return replace(self, _v=filt(self._v), _i=filt(self._i))

    def resample(self, cycle_size: int) -> Measurement:
        """Change the number of points per cycle.

        Example:
            >>> a = nf.example_measurement().aligned(cycle_size=128)
            >>> a.resample(32).cycle_size
            32
        """
        from nilmframe.data.views import resample_to

        if not self.aligned_:
            raise ValueError("resample changes points per cycle; call .aligned() first")
        return replace(
            self, _v=resample_to(self._v, cycle_size), _i=resample_to(self._i, cycle_size)
        )

    # -- representations ---------------------------------------------------- #

    def fryze(self) -> Tensor:
        """Fryze decomposition: ``(..., 3, T)`` of voltage, active and non-active current.

        Example:
            >>> a = nf.example_measurement('fridge').aligned(cycle_size=64)
            >>> tuple(a.fryze().shape)
            (23, 3, 64)
        """
        from nilmframe.nn.repr import Fryze

        return Fryze()(self.v, self.i)

    def vi_image(self, size: int = 64) -> Tensor:
        """VI trajectory image, ``(..., 3, size, size)``.

        Example:
            >>> a = nf.example_measurement().aligned(cycle_size=64)
            >>> tuple(a.vi_image(size=32).shape)
            (23, 3, 32, 32)
        """
        from nilmframe.nn.repr import VITrajectory

        return VITrajectory(image_size=size)(self.v, self.i)

    def spectrogram(self, window_size: int = 256, hop_size: int = 64) -> Tensor:
        """Log-power STFT of the current, ``(freq, time)``.

        Example:
            >>> tuple(nf.example_measurement().spectrogram(window_size=64, hop_size=32).shape)
            (33, 94)
        """
        from nilmframe.nn.repr import Spectrogram

        return Spectrogram(window_size=window_size, hop_size=hop_size)(self.i.flatten())

    def harmonics(self, n: int = 16, normalize: bool = True) -> Tensor:
        """Harmonic amplitudes of the current, averaged over cycles.

        Example:
            >>> a = nf.example_measurement('laptop').aligned(cycle_size=128)
            >>> [round(float(x), 3) for x in a.harmonics(6)]
            [0.0, 1.0, 0.55, 0.399, 0.0, 0.0]
        """
        if not self.aligned_:
            raise ValueError("harmonics need aligned cycles; call .aligned()")
        spectrum = torch.fft.rfft(self.i, dim=-1).abs().mean(0)[:n]
        return spectrum / spectrum[1].clamp_min(1e-12) if normalize else spectrum

    # -- events ------------------------------------------------------------- #

    def events(self, detector=None, **kwargs) -> Tensor:
        """Boolean event mask over the power envelope.

        Example:
            >>> watts = torch.cat([torch.zeros(40), torch.full((40,), 2000.), torch.zeros(40)])
            >>> m = nf.Measurement.from_power(watts, fs=1.0)
            >>> int(m.events(window=8, threshold=2.5, min_delta=100.).sum())
            2
        """
        from nilmframe.nn.segment import ZScoreDetector

        detector = detector or ZScoreDetector(**kwargs)
        envelope = self.p.mean(-1) if self.p.ndim > 1 else self.p
        return detector(envelope)

    def segments(self, detector=None, min_length: int = 1, **kwargs) -> list[Measurement]:
        """Split at detected events. Only meaningful before alignment.

        Example:
            >>> watts = torch.cat([torch.zeros(40), torch.full((40,), 2000.), torch.zeros(40)])
            >>> m = nf.Measurement.from_power(watts, fs=1.0)
            >>> [(s.n_samples) for s in m.segments(window=8, threshold=2.5, min_delta=100.)]
            [40, 40, 40]
        """
        from nilmframe.nn.segment import segments_from_mask

        spans = segments_from_mask(self.events(detector, **kwargs), min_length=min_length)
        return [self.window(a, b - a) for a, b in spans]

    # -- composition -------------------------------------------------------- #

    def __add__(self, other: Measurement | int) -> Measurement:
        """Superpose two measurements -- what a meter would see if both ran at once.

        The predecessor's version raised on every call: it forgot to pass ``fs`` to
        the constructor it built, and compared an ``f0`` attribute that did not
        exist.
        """
        if isinstance(other, int) and other == 0:
            return self  # so sum() works
        if not isinstance(other, Measurement):
            return NotImplemented
        # Kind first: mixing a waveform with a power series is a more fundamental
        # mistake than a rate mismatch, and saying so is more useful than
        # complaining about the rates that necessarily differ between them.
        if self.is_waveform != other.is_waveform:
            raise ValueError("cannot superpose a waveform with a power series")
        if abs(self.fs - other.fs) > 1e-9:
            raise ValueError(f"sampling rates differ: {self.fs} != {other.fs}")

        if self.is_waveform:
            if self._v.shape != other._v.shape:
                raise ValueError(
                    f"shapes differ: {tuple(self._v.shape)} != {tuple(other._v.shape)}"
                )
            return replace(
                self,
                _i=torch.cat([self._i, other._i], dim=0),
                appliances=self.appliances + other.appliances,
                source=f"{self.source}+{other.source}",
            )
        if self._p.shape[1:] != other._p.shape[1:]:
            raise ValueError(
                f"lengths differ: {self._p.shape[-1]} != {other._p.shape[-1]} samples. "
                "Real channels rarely run for exactly the same time; take a common "
                "window first, e.g. n = min(a.n_samples, b.n_samples); "
                "a.window(0, n) + b.window(0, n)"
            )
        return replace(
            self,
            _p=torch.cat([self._p, other._p], dim=0),
            appliances=self.appliances + other.appliances,
            source=f"{self.source}+{other.source}",
        )

    __radd__ = __add__

    # -- interop ------------------------------------------------------------ #

    def to(self, device) -> Measurement:
        """Move the signals to a device, as on a tensor.

        Example:
            >>> nf.example_measurement().to('cpu').v.device.type
            'cpu'
        """
        move = lambda t: None if t is None else t.to(device)  # noqa: E731
        return replace(self, _v=move(self._v), _i=move(self._i), _p=move(self._p))

    def numpy(self) -> dict[str, Any]:
        """Plain numpy arrays, for anything that does not speak torch.

        Example:
            >>> arrays = nf.example_measurement(seconds=0.1).numpy()
            >>> sorted(arrays)
            ['f0', 'fs', 'i', 't0', 'v']
            >>> type(arrays['v']).__name__, arrays['v'].shape
            ('ndarray', (600,))
        """
        out = {"fs": self.fs, "f0": self.f0, "t0": self.t0}
        if self.is_waveform:
            out["v"], out["i"] = self.v.cpu().numpy(), self.i.cpu().numpy()
        else:
            out["p"] = self.p.cpu().numpy()
        return out

    def batch(self) -> dict[str, Tensor]:
        """The same window in the shape a model expects, with a batch axis.

        Example:
            >>> batch = nf.example_measurement().aligned(cycle_size=64).batch()
            >>> {k: tuple(v.shape) for k, v in batch.items()}
            {'v': (1, 23, 64), 'i': (1, 23, 64), 'cycle_mask': (1, 23), 'p_total': (1,)}
        """
        if self.is_waveform:
            item = {"v": self.v.unsqueeze(0), "i": self.i.unsqueeze(0)}
            if self.aligned_:
                item["cycle_mask"] = torch.ones(
                    1, self.n_cycles, dtype=torch.bool, device=self.v.device
                )
        else:
            item = {"p": self.p.unsqueeze(0)}
        item["p_total"] = self.active_power().reshape(1)
        return item

    # -- looking at it ------------------------------------------------------ #

    def __repr__(self) -> str:
        """A one-line summary that fits on a line.

        Example:
            >>> nf.example_measurement('kettle')
            Measurement(waveform raw, 6000Hz, 0.500s, kettle, 2926W)
            >>> nf.example_measurement('fridge').aligned(cycle_size=64)
            Measurement(waveform 23x64, 6000Hz, 0.245s, fridge, 237W)
            >>> nf.example_measurement('kettle') + nf.example_measurement('fridge')
            Measurement(waveform raw, 6000Hz, 0.500s, 2 components, 3164W)
        """
        if self.is_waveform:
            what = (
                f"waveform {self.n_cycles}x{self.cycle_size}" if self.aligned_ else "waveform raw"
            )
        else:
            what = "power"

        bits = [what, f"{self.fs:g}Hz", f"{self.duration:.3f}s"]
        named = [a for a in self.appliances if a]
        if len(named) == 1:
            bits.append(named[0])
        elif named:
            bits.append(f"{len(named)} components")
        elif self.n_components > 1:
            bits.append(f"{self.n_components} components")
        with contextlib.suppress(Exception):  # odd shapes should not break a repr
            bits.append(f"{float(self.active_power()):.0f}W")
        return f"Measurement({', '.join(bits)})"

    def plot(self, ax=None, seconds: float | None = None, **kwargs):
        """Draw it. Returns the matplotlib axes.

        Plotting is the one thing here that needs a library the core does not,
        so matplotlib is an extra rather than a dependency -- a training run on a
        cluster has no use for it.

        Example:
            >>> import matplotlib; matplotlib.use('Agg')
            >>> ax = nf.example_measurement(seconds=0.05).plot(title='kettle')
            >>> ax.get_title()
            'kettle'
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:  # pragma: no cover - exercised by the extras matrix
            raise ImportError(
                "Measurement.plot needs matplotlib: pip install 'nilmframe[plot]'"
            ) from exc

        # Pull out everything that is ours before the rest is forwarded to the
        # line: matplotlib rejects unknown artist properties, so a stray `title=`
        # reaching `ax.plot` raises rather than being ignored.
        figsize = kwargs.pop("figsize", (11, 2.8))
        title = kwargs.pop("title", self.source or self.kind)

        if ax is None:
            _, ax = plt.subplots(figsize=figsize)

        if not self.is_waveform:
            series = self.p.cpu().numpy()
            ax.step(torch.arange(len(series)).numpy() / self.fs, series, where="post", **kwargs)
            ax.set_xlabel("s")
            ax.set_ylabel("W")
        elif self.aligned_:
            grid = torch.linspace(0, 1, self.cycle_size).numpy()
            current = self.i.cpu().numpy()
            for row in current[: min(len(current), 80)]:
                ax.plot(grid, row, lw=0.5, alpha=0.35, **kwargs)
            ax.plot(grid, current.mean(0), lw=1.6, color="black", label="mean cycle")
            ax.set_xlabel("phase within cycle")
            ax.set_ylabel("A")
            ax.legend()
        else:
            n = self.n_samples if seconds is None else int(seconds * self.fs)
            t = torch.arange(n).numpy() / self.fs * 1000
            ax.plot(t, self.v[:n].cpu().numpy(), lw=0.9, label="V")
            twin = ax.twinx()
            twin.grid(False)
            twin.plot(t, self.i[:n].cpu().numpy(), lw=0.9, color="#c1553b", label="A")
            ax.set_xlabel("ms")
            ax.set_ylabel("V")
            twin.set_ylabel("A")
        ax.set_title(title)
        return ax
