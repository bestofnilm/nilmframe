"""Can these datasets be combined, and under what rules?

Merging corpora is the normal case in NILM, not an exotic one: PLAID is 6 kHz at
60 Hz mains and 120 V, WHITED is 44.1 kHz at 50 Hz and 230 V, UK-DALE is 16 kHz
plus a 1/6 Hz meter. Putting them in one store is easy; putting them in one store
*and having the result mean something* is the part that needs stating.

This module answers two questions:

**What varies?** :func:`compatibility` enumerates the axes along which a set of
channels disagrees -- sampling rate, mains frequency, voltage level, available
quantities, appliance vocabulary -- without loading a single waveform for the
metadata ones.

**Does it matter?** That depends entirely on the view, and the interesting answer
is that it often does not. Cycle alignment resamples every mains cycle onto a
fixed grid, so after it a 6 kHz 60 Hz recording and a 44.1 kHz 50 Hz one are the
same shape and the same phase. Sampling rate and mains frequency stop being
blocking axes -- which is the strongest practical argument for the representation.
They stay blocking for a raw-waveform view, because there a fixed-length slice of
one is not comparable with a fixed-length slice of the other.

What alignment does *not* fix is voltage level and vocabulary. A kettle on 120 V
draws twice the current it draws on 230 V, and "fridge" in one corpus and
"refrigerator" in another are the same appliance under two names. Both are
handled at merge time -- see :func:`~nilmframe.store.merge.merge_stores`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from nilmframe.store import Store

__all__ = ["Axis", "CompatibilityReport", "compatibility"]


@dataclass(frozen=True)
class Axis:
    """One dimension along which a set of channels may disagree.

    Attributes:
        name: the axis.
        values: the distinct values present.
        blocking_for: views this difference actually breaks.
        note: why, in one line.

    Note:
        ``blocking_for`` and ``partial_for`` are different claims. A blocking axis
        makes a view impossible; a partial one merely narrows it to the channels
        that can serve it, and the dataset skips the rest rather than failing.

    Example:
        >>> axis = nf.compatibility(store).axis('appliance vocabulary')
        >>> axis.name, axis.varies
        ('appliance vocabulary', True)
    """

    name: str
    values: list
    blocking_for: tuple[str, ...]
    note: str
    #: Views this axis merely *narrows* -- some channels can serve them, others
    #: cannot, and the dataset skips the ones that cannot rather than failing.
    partial_for: tuple[str, ...] = ()

    @property
    def varies(self) -> bool:
        return len(self.values) > 1

    def __repr__(self) -> str:
        shown = self.values if len(self.values) <= 4 else [*self.values[:4], "..."]
        return f"Axis({self.name}, {len(self.values)} distinct: {shown})"


@dataclass
class CompatibilityReport:
    """What varies across a set of channels, and what it breaks.

    Example:
        >>> report = nf.compatibility(store)
        >>> [a.name for a in report.varying()]
        ['appliance vocabulary']
        >>> report.usable('lowfreq'), report.usable('highfreq_aligned')
        (8, 8)
    """

    axes: list[Axis]
    channels: pd.DataFrame
    datasets: list[str] = field(default_factory=list)

    def __repr__(self) -> str:
        varying = [a.name for a in self.axes if a.varies]
        return (
            f"CompatibilityReport(datasets={self.datasets}, channels={len(self.channels)}, "
            f"varying={varying or 'nothing'})"
        )

    def axis(self, name: str) -> Axis:
        """One axis by name.

        Raises:
            KeyError: when there is no such axis.

        Example:
            >>> nf.compatibility(store).axis('fs')
            Axis(fs, 1 distinct: [6000.0])
        """
        for a in self.axes:
            if a.name == name:
                return a
        raise KeyError(f"no axis {name!r}; have {[a.name for a in self.axes]}")

    def varying(self) -> list[Axis]:
        """Only the axes that actually differ across the inputs.

        Example:
            >>> [a.name for a in nf.compatibility(store).varying()]
            ['appliance vocabulary']
        """
        return [a for a in self.axes if a.varies]

    def partial(self, view: str = "highfreq_aligned") -> list[Axis]:
        """Axes that narrow a view to a subset of channels rather than breaking it.

        Example:
            >>> [a.name for a in nf.compatibility(store).partial('highfreq_aligned')]
            []
        """
        return [a for a in self.axes if a.varies and _view_name(view) in a.partial_for]

    def usable(self, view: str = "highfreq_aligned") -> int:
        """How many channels a view can actually read.

        Example:
            >>> report = nf.compatibility(store)
            >>> report.usable('highfreq_aligned'), report.usable('lowfreq')
            (8, 8)
        """
        name = _view_name(view)
        has_waveform = self.channels["quantities"].str.contains("v") & self.channels[
            "quantities"
        ].str.contains("i")
        if name == "lowfreq":
            return int((has_waveform | self.channels["quantities"].str.contains("p")).sum())
        return int(has_waveform.sum())

    def blocking(self, view: str = "highfreq_aligned") -> list[Axis]:
        """Axes that actually break a given view.

        Args:
            view: ``"highfreq_aligned"``, ``"highfreq_raw"`` or ``"lowfreq"``. A
                :class:`~nilmframe.data.views.HighFreqView` /
                :class:`~nilmframe.data.views.LowFreqView` instance is also accepted.

        Example:
            >>> [a.name for a in nf.compatibility(store).blocking('highfreq_raw')]
            []
        """
        return [a for a in self.axes if a.varies and _view_name(view) in a.blocking_for]

    def is_compatible(self, view: str = "highfreq_aligned") -> bool:
        """No blocking axis, and at least one channel the view can read.

        Example:
            >>> report = nf.compatibility(store)
            >>> report.is_compatible('highfreq_aligned'), report.is_compatible('lowfreq')
            (True, True)
        """
        return not self.blocking(view) and self.usable(view) > 0

    def to_frame(self) -> pd.DataFrame:
        """The report as a table, one row per axis.

        Example:
            >>> nf.compatibility(store, deep=False).to_frame()[['axis', 'distinct', 'blocks']]
                               axis  distinct        blocks
            0                    fs         1  highfreq_raw
            1                    f0         1  highfreq_raw
            2            quantities         1             -
            3  appliance vocabulary         4             -
            4               dataset         1             -
        """
        return pd.DataFrame(
            [
                {
                    "axis": a.name,
                    "distinct": len(a.values),
                    "values": a.values if len(a.values) <= 6 else [*a.values[:6], "..."],
                    "blocks": ", ".join(a.blocking_for) or "-",
                    "narrows": ", ".join(a.partial_for) or "-",
                    "note": a.note,
                }
                for a in self.axes
            ]
        )

    def summary(self) -> str:
        """A human-readable verdict per view.

        Example:
            >>> print(nf.compatibility(store, deep=False).summary())
            CompatibilityReport(datasets=['example'], channels=8, varying=['appliance vocabulary'])
            <BLANKLINE>
              highfreq_aligned   ok on all 8 channels
              highfreq_raw       ok on all 8 channels
              lowfreq            ok on all 8 channels
            <BLANKLINE>
              varying axes:
                appliance vocabulary 4 distinct: ['fridge', 'kettle', 'laptop', 'microwave'])
        """
        lines = [repr(self), ""]
        total = len(self.channels)
        for view in ("highfreq_aligned", "highfreq_raw", "lowfreq"):
            blockers = self.blocking(view)
            usable = self.usable(view)
            if blockers:
                verdict = "blocked by " + ", ".join(a.name for a in blockers)
            elif usable == 0:
                verdict = "no channel can serve it"
            elif usable < total:
                verdict = f"ok on {usable}/{total} channels (the rest are skipped)"
            else:
                verdict = f"ok on all {total} channels"
            lines.append(f"  {view:<18} {verdict}")
        varying = self.varying()
        if varying:
            lines += ["", "  varying axes:"]
            lines += [f"    {a.name:<12} {a!r}".replace(f"Axis({a.name}, ", "") for a in varying]
        return "\n".join(lines)


def _view_name(view) -> str:
    """Normalise a view object or string to one of the three regimes."""
    if isinstance(view, str):
        return view
    kind = getattr(view, "name", "")
    if kind == "lowfreq":
        return "lowfreq"
    return "highfreq_aligned" if getattr(view, "align", None) else "highfreq_raw"


def _sample_vrms(store: Store, channel_id: str, samples: int = 4096) -> float | None:
    """RMS voltage from the head of a channel. ``None`` when it stores no voltage."""
    try:
        chunk = store.read_window(channel_id, "v", 0, samples)
    except KeyError:
        return None
    return float(np.sqrt(np.mean(np.square(chunk.astype(np.float64)))))


def compatibility(
    *stores: Store,
    deep: bool = True,
    voltage_tolerance: float = 0.15,
) -> CompatibilityReport:
    """Enumerate what varies across one or more stores.

    Args:
        stores: the stores to compare. One store is fine -- a single corpus can be
            internally inconsistent, and usually is.
        deep: sample each channel's voltage to find the supply level. Reads a few
            thousand samples per channel; set ``False`` for metadata only.
        voltage_tolerance: fractional spread within which supply levels count as
            the same. 120 V and 230 V are different; 228 V and 231 V are not.

    Returns:
        A :class:`CompatibilityReport`.

    Example:
        >>> report = nf.compatibility(store)
        >>> report
        CompatibilityReport(datasets=['example'], channels=8, varying=['appliance vocabulary'])
        >>> report.is_compatible('highfreq_aligned')
        True
    """
    if not stores:
        raise ValueError("compatibility() needs at least one store")

    frames = []
    for store in stores:
        frame = store.channels.copy()
        frame["_store"] = str(store.path)
        frames.append(frame)
    channels = pd.concat(frames, ignore_index=True)

    axes: list[Axis] = [
        Axis(
            "fs",
            sorted(channels["fs"].round(4).unique().tolist()),
            ("highfreq_raw",),
            "cycle alignment resamples to a fixed grid, so this stops mattering once aligned",
        ),
        Axis(
            "f0",
            sorted(round(float(x), 2) for x in channels["f0"].dropna().unique()),
            ("highfreq_raw",),
            "50 vs 60 Hz mains; alignment normalises the period away",
        ),
        Axis(
            "quantities",
            sorted(channels["quantities"].unique().tolist()),
            (),
            "a waveform view needs v and i; channels holding only p are skipped, "
            "not fatal -- unless none are left",
            partial_for=("highfreq_aligned", "highfreq_raw"),
        ),
        Axis(
            "appliance vocabulary",
            sorted(channels["appliance"].dropna().unique().tolist()),
            (),
            "the merged label space is the union; names that mean the same thing must match",
        ),
        Axis(
            "dataset",
            sorted(channels["dataset"].unique().tolist()),
            (),
            "not blocking, but worth a CrossDataset split rather than a random one",
        ),
    ]

    if deep:
        levels = []
        for store in stores:
            for cid in store.channels["channel_id"]:
                vrms = _sample_vrms(store, cid)
                if vrms and vrms > 1.0:
                    levels.append(vrms)
        if levels:
            clusters = _cluster(levels, voltage_tolerance)
            axes.append(
                Axis(
                    "supply voltage",
                    [round(c, 1) for c in clusters],
                    ("highfreq_aligned", "highfreq_raw", "lowfreq"),
                    "the same appliance draws different current on 120 V and on 230 V; "
                    "alignment does not fix this -- normalise at merge time",
                )
            )

    return CompatibilityReport(
        axes=axes,
        channels=channels,
        datasets=sorted(channels["dataset"].unique().tolist()),
    )


def _cluster(values: list[float], tolerance: float) -> list[float]:
    """Group values whose spread is within ``tolerance``; return the cluster means."""
    out: list[list[float]] = []
    for value in sorted(values):
        if out and value <= out[-1][0] * (1 + tolerance):
            out[-1].append(value)
        else:
            out.append([value])
    return [float(np.mean(group)) for group in out]
