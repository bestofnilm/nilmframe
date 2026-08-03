"""Canonical store schema.

The store is the single place data lives. Two properties drive its shape:

**Metadata is tabular, signals are memory-mapped.** Everything the predecessor's
``HighFreqDataset`` did with thirty eager methods -- ``submetered``, ``drop_rare``,
``filter``, ``groupby``, ``devices``, ``brands`` -- is a dataframe query here, and
none of it touches a waveform. Signals are plain ``.npy`` arrays opened with
``mmap_mode='r'``, so a window read costs one page fault rather than a dataset
load, and ``num_workers > 0`` needs no per-worker file-handle juggling.

**Time is explicit.** Every channel carries an absolute start time and a sampling
rate, so channels recorded on the same clock (a house's mains and its submeters)
can be intersected. That is what lets targets for an aggregate window be assembled
from sibling submeters, and it is what makes the same store serve both a
high-frequency and a low-frequency view of one recording.

Tables
------
``channels.parquet``
    One row per measured signal. ``instance_id`` identifies a *physical unit* --
    this specific kettle, not the class "kettle" -- and is what leakage-safe
    splits partition on.
``activations.parquet``
    One row per (channel, appliance, on, off) interval, in samples relative to the
    channel start. Present for aggregate recordings whose per-appliance power is
    unknown but whose on/off times are annotated.
``appliances.parquet``
    One row per appliance class: the power threshold above which it counts as on,
    a coarse category, and whether it is part of the known label space (the
    open-set experiments hold classes out by flipping ``is_known``).
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = [
    "ACTIVATION_COLUMNS",
    "APPLIANCE_COLUMNS",
    "CHANNEL_COLUMNS",
    "DEFAULT_ON_THRESHOLD_W",
    "STORE_FORMAT_VERSION",
    "Activation",
    "ChannelKind",
    "Quantity",
    "Recording",
]

# Bumped when the on-disk layout changes incompatibly. `Store` refuses to open a
# store written by a newer format than it understands.
STORE_FORMAT_VERSION = 1

# Legacy `BaseModel.thresh` was a bare 10 W applied to every appliance. Keeping a
# per-appliance table makes the assumption visible and overridable instead of
# hard-coded in the model.
DEFAULT_ON_THRESHOLD_W = 10.0


class ChannelKind(str, enum.Enum):
    """What a channel measures.

    The distinction drives target construction: a submeter's window knows its own
    power exactly, whereas a mains window's per-appliance truth has to be
    assembled from sibling submeters or from annotations -- and may not exist at
    all, which is what ``power_mask`` records.

    Attributes:
        MAINS: an aggregate measurement, everything downstream of one meter.
        SUBMETER: a single appliance, measured in isolation.

    Example:
        >>> from nilmframe.store import ChannelKind
        >>> ChannelKind.MAINS.value, ChannelKind.SUBMETER.value
        ('mains', 'submeter')
        >>> ChannelKind('submeter') is ChannelKind.SUBMETER
        True
    """

    MAINS = "mains"
    """An aggregate measurement: everything downstream of one meter."""

    SUBMETER = "submeter"
    """A single appliance, measured in isolation."""


class Quantity(str, enum.Enum):
    """A signal stored for a channel.

    A waveform channel carries ``v`` and ``i``; a meter channel carries ``p``.
    Which of them a channel has decides which views can render it -- see
    :meth:`HighFreqView.supports <nilmframe.data.HighFreqView.supports>`.

    Attributes:
        VOLTAGE: ``v``, in volts.
        CURRENT: ``i``, in amperes.
        ACTIVE_POWER: ``p``, in watts.

    Example:
        >>> from nilmframe.store import Quantity
        >>> [(q.value, q.unit) for q in Quantity]
        [('v', 'V'), ('i', 'A'), ('p', 'W')]
    """

    VOLTAGE = "v"
    CURRENT = "i"
    ACTIVE_POWER = "p"

    @property
    def unit(self) -> str:
        return {"v": "V", "i": "A", "p": "W"}[self.value]


CHANNEL_COLUMNS: dict[str, str] = {
    "channel_id": "string",
    "dataset": "string",
    "house": "string",
    "session": "string",
    "kind": "string",
    "appliance": "string",
    "brand": "string",
    "instance_id": "string",
    "fs": "float64",
    "f0": "float64",
    "t0": "float64",
    "n_samples": "int64",
    "quantities": "string",
    "sha256": "string",
}

ACTIVATION_COLUMNS: dict[str, str] = {
    "channel_id": "string",
    "appliance": "string",
    "on": "int64",
    "off": "int64",
}

APPLIANCE_COLUMNS: dict[str, str] = {
    "appliance": "string",
    "on_threshold_w": "float64",
    "category": "string",
    "is_known": "bool",
}


@dataclass(frozen=True, slots=True)
class Activation:
    """An appliance on/off interval, in samples relative to the channel start.

    Attributes:
        appliance: which appliance ran.
        on: first sample of the interval.
        off: one past the last sample.

    Note:
        Intervals are half-open: ``on`` is the first sample and ``off`` is one past
        the last. A window counts an appliance present when it overlaps the interval
        by more than half its own length -- a sliver at the edge does not.

    Example:
        >>> from nilmframe.store import Activation
        >>> Activation('kettle', 100, 250)
        Activation(appliance='kettle', on=100, off=250)
    """

    appliance: str
    on: int
    off: int

    def __post_init__(self) -> None:
        if self.off < self.on:
            raise ValueError(f"activation ends before it starts: {self.on} > {self.off}")


@dataclass(slots=True)
class Recording:
    """One measured channel, as produced by a dataset reader.

    This is the readers' only output type. The predecessor's readers returned bare
    tuples of differing arity -- PLAID six, WHITED five, with PLAID's aggregate
    branch leaving `brands` unbound -- and the dataset layer patched them up
    positionally with ``args[3] = [args[3]]``. A named record makes arity errors
    impossible and lets a reader omit what it does not know.

    Args:
        dataset: source dataset name, e.g. ``"plaid"``.
        house: site identifier. For datasets of isolated recordings this is the
            recording group; for UK-DALE it is the house.
        session: contiguous recording within a house. Channels sharing a session
            share a clock, so their windows can be intersected.
        kind: mains or submeter.
        signals: quantity name to 1-D array. High-frequency channels carry
            ``{"v", "i"}``; low-frequency channels carry ``{"p"}``.
        fs: sampling rate in Hz.
        appliance: appliance class, for submeter channels.
        brand: manufacturer, where known. Used by brand-disjoint splits.
        instance_id: identifies the physical unit. Defaults to ``brand`` when
            absent, and to the channel itself when neither is known -- the
            conservative choice, since it prevents two recordings of one unit from
            landing on both sides of a split.
        f0: mains frequency. Estimated from the voltage when omitted.
        t0: absolute start time in seconds since the epoch, for cross-channel
            alignment. Zero when a dataset gives no absolute clock.
        activations: annotated on/off intervals, for aggregate recordings.
        meta: anything else worth keeping; written to the manifest, not the tables.

    Example:
        >>> from nilmframe.store import ChannelKind, Recording
        >>> rec = Recording(dataset='d', house='h', session='s',
        ...                 kind=ChannelKind.SUBMETER, appliance='kettle',
        ...                 brand='acme', signals={'p': np.ones(600, np.float32)},
        ...                 fs=1.0)
        >>> rec.n_samples, rec.duration_s, rec.quantities
        (600, 600.0, ['p'])
        >>> rec.resolved_instance_id()
        'kettle:acme'
    """

    dataset: str
    house: str
    session: str
    kind: ChannelKind
    signals: dict[str, np.ndarray]
    fs: float
    appliance: str | None = None
    brand: str | None = None
    instance_id: str | None = None
    f0: float | None = None
    t0: float = 0.0
    activations: list[Activation] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.signals:
            raise ValueError("a recording must carry at least one signal")

        lengths = {name: np.asarray(a).shape for name, a in self.signals.items()}
        for name, shape in lengths.items():
            if len(shape) != 1:
                raise ValueError(f"signal {name!r} must be 1-D, got shape {shape}")
        if len({s[0] for s in lengths.values()}) != 1:
            raise ValueError(f"signals have mismatched lengths: {lengths}")

        known = {q.value for q in Quantity}
        unknown = set(self.signals) - known
        if unknown:
            raise ValueError(f"unknown quantities {sorted(unknown)}; expected a subset of {known}")

        if self.fs <= 0:
            raise ValueError(f"fs must be positive, got {self.fs}")

        self.kind = ChannelKind(self.kind)
        if self.kind is ChannelKind.SUBMETER and not self.appliance:
            raise ValueError("a submeter channel must name its appliance")

        for act in self.activations:
            if act.off > self.n_samples:
                raise ValueError(
                    f"activation {act} runs past the end of a {self.n_samples}-sample recording"
                )

        self.signals = {
            k: np.ascontiguousarray(v, dtype=np.float32) for k, v in self.signals.items()
        }

    @property
    def n_samples(self) -> int:
        """Length of every signal in this recording.

        Example:
            >>> from nilmframe.store import ChannelKind, Recording
            >>> rec = Recording(dataset='d', house='h', session='s',
            ...                 kind=ChannelKind.MAINS,
            ...                 signals={'p': np.zeros(42, np.float32)}, fs=1.0)
            >>> rec.n_samples
            42
        """
        return int(next(iter(self.signals.values())).shape[0])

    @property
    def duration_s(self) -> float:
        """Seconds of signal.

        Example:
            >>> from nilmframe.store import ChannelKind, Recording
            >>> rec = Recording(dataset='d', house='h', session='s',
            ...                 kind=ChannelKind.MAINS,
            ...                 signals={'p': np.zeros(120, np.float32)}, fs=2.0)
            >>> rec.duration_s
            60.0
        """
        return self.n_samples / self.fs

    @property
    def quantities(self) -> list[str]:
        """Which signals this recording carries, sorted.

        Example:
            >>> from nilmframe.store import ChannelKind, Recording
            >>> wave = {'i': np.zeros(4, np.float32), 'v': np.zeros(4, np.float32)}
            >>> rec = Recording(dataset='d', house='h', session='s',
            ...                 kind=ChannelKind.MAINS, signals=wave, fs=1.0)
            >>> rec.quantities
            ['i', 'v']
        """
        return sorted(self.signals)

    def resolved_instance_id(self) -> str:
        """The identity a leakage-safe split must not straddle.

        Example:
            >>> from nilmframe.store import ChannelKind, Recording
            >>> base = {'dataset': 'd', 'house': 'h', 'session': 's',
            ...         'kind': ChannelKind.SUBMETER, 'appliance': 'kettle',
            ...         'signals': {'p': np.zeros(4, np.float32)}, 'fs': 1.0}
            >>> Recording(**base, instance_id='unit-7').resolved_instance_id()
            'unit-7'
            >>> Recording(**base, brand='acme').resolved_instance_id()
            'kettle:acme'
            >>> Recording(**base).resolved_instance_id()
            'd:h:s'
        """
        if self.instance_id:
            return str(self.instance_id)
        if self.brand:
            return f"{self.appliance or self.kind.value}:{self.brand}"
        return f"{self.dataset}:{self.house}:{self.session}"
