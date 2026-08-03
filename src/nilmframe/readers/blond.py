"""BLOND reader.

BLOND is the largest thing this package reads: 213 days of a three-phase office
circuit at 50 kHz with every appliance separately metered, and a second release
at 250 kHz. Uncompressed it is roughly 8.9 TB, which makes it the dataset that
justifies fetching a subset rather than a corpus -- see
:class:`~nilmframe.sources.BLONDSource`.

On-disk layout::

    BLOND-50/
      2016-09-30/
        clear/       clear-<date>T<time>T+0200-<seq>.hdf5     3-phase mains, 50 kHz
        medal-1/     medal-1-<date>T<time>T+0200-<seq>.hdf5   6 sockets, 6.4 kHz
        ...
        medal-15/
    appliance_log.json

Two units, two shapes. A **CLEAR** file holds ``voltage1..3`` and ``current1..3``
-- one pair per phase of the building's mains, five minutes at 50 kHz. A
**MEDAL** file holds one ``voltage`` and six ``current1..6``, one per socket,
fifteen minutes at 6.4 kHz. Samples are ``int16`` and become volts and amperes by
multiplying by the ``calibration_factor`` attribute each dataset carries; nothing
else is needed, and the resulting mains RMS lands where it should.

**The labels are time-versioned.** ``appliance_log.json`` records, per MEDAL, a
list of socket configurations each stamped with the moment it took effect --
because over seven months people unplugged things. Labelling a recording means
taking the newest configuration at or before the recording's own clock, not the
first one in the file. Getting this wrong silently mislabels whole weeks.

**A phase ties a submeter to its aggregate.** Each MEDAL sits on one circuit,
``L1`` to ``L3``, which is the CLEAR phase whose current includes it. That is
recorded on the channel so a mains window can be matched to the submeters that
actually explain it, rather than to all fifteen units.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from nilmframe.store.schema import ChannelKind, Recording

if TYPE_CHECKING:  # `nilmframe.sources` is imported only when data is fetched
    from nilmframe.sources import Plan

__all__ = ["BLOND", "parse_blond_name"]

logger = logging.getLogger(__name__)

#: ``<unit>-<date>T<time>.<microseconds>T<offset>-<sequence>.hdf5``. The unit name
#: itself contains a dash for the MEDALs, so the timestamp is matched from the
#: right rather than by splitting on dashes.
_NAME = re.compile(
    r"^(?P<unit>.+?)-(?P<date>\d{4}-\d{2}-\d{2})T(?P<hour>\d{2})-(?P<minute>\d{2})"
    r"-(?P<second>\d{2})\.(?P<micro>\d+)T(?P<offset>[+-]\d{4})-(?P<sequence>\d+)\.hdf5$"
)

_SOCKETS = tuple(f"socket_{i}" for i in range(1, 7))


def parse_blond_name(name: str) -> dict | None:
    """Unit, local time and absolute start from a BLOND filename.

    The filename is the clock, so a time range can be turned into a file list
    without opening anything -- which is what makes planning a fetch cheap.

    Args:
        name: the file's basename.

    Returns:
        ``{"unit", "sequence", "t0", "local"}``, or ``None`` if the name does not
        parse. ``t0`` is unix seconds; ``local`` is the wall-clock time the log's
        timestamps are written in.

    Example:
        >>> from nilmframe.readers.blond import parse_blond_name
        >>> info = parse_blond_name('medal-1-2016-09-30T11-02-08.736487T+0200-0000000.hdf5')
        >>> info['unit'], info['sequence']
        ('medal-1', 0)
        >>> info['local'].isoformat()
        '2016-09-30T11:02:08.736487'
    """
    match = _NAME.match(name)
    if match is None:
        return None
    fields = match.groupdict()
    local = datetime(
        *(int(part) for part in fields["date"].split("-")),
        int(fields["hour"]),
        int(fields["minute"]),
        int(fields["second"]),
        int(fields["micro"][:6].ljust(6, "0")),
    )
    sign = 1 if fields["offset"][0] == "+" else -1
    offset = timedelta(hours=int(fields["offset"][1:3]), minutes=int(fields["offset"][3:5]))
    return {
        "unit": fields["unit"],
        "sequence": int(fields["sequence"]),
        "local": local,
        "t0": local.replace(tzinfo=timezone(sign * offset)).timestamp(),
    }


class BLOND:
    """Iterate BLOND recordings as :class:`Recording` objects.

    Args:
        dirpath: a ``BLOND-50`` or ``BLOND-250`` directory, holding ``YYYY-MM-DD``
            day directories.
        appliance_log: path to ``appliance_log.json``. Without it the MEDAL
            sockets have no appliance names and only the mains can be read.
        units: unit names to read, e.g. ``["clear", "medal-1"]``. ``None`` reads
            every unit present.
        days: ``YYYY-MM-DD`` strings to read. ``None`` reads every day present.
        sockets: MEDAL socket numbers, 1 to 6. ``None`` reads all six.
        phases: CLEAR phase numbers, 1 to 3. ``None`` reads all three.
        max_files: files to read per unit per day. Each CLEAR file is five
            minutes at 50 kHz -- 15 million samples per phase -- so this is the
            knob that decides whether a conversion finishes.
        max_seconds: seconds to take from the start of each file. Independent of
            ``max_files``: one bounds how many recordings, the other how long.
        remove_offset: subtract each mains cycle's mean, as the dataset's authors
            prescribe. The offset removed during recording was the converter's
            nominal midpoint and the true zero drifts from it; on the metered
            sockets the residual is as large as the signal. Turn it off only to
            see the samples exactly as stored.
        f0: mains frequency, which sets the cycle the offset is removed over.
        time_range: ``(start, stop)`` unix seconds, matched against the start
            time in each filename.

    Example:
        One five-minute file of the three-phase mains is enough to see the shape
        of things, and already 15 million samples per phase::

            >>> from nilmframe.readers import BLOND
            >>> from nilmframe.store import StoreWriter
            >>> reader = BLOND("blond/BLOND-50",              # doctest: +SKIP
            ...                appliance_log="blond/appliance_log.json",
            ...                units=["clear"], max_files=1, max_seconds=10)
            >>> with StoreWriter("stores/blond") as w:        # doctest: +SKIP
            ...     for rec in reader:
            ...         w.add(rec)

        A MEDAL unit gives the appliances on one circuit, labelled from the log
        as it stood on the day::

            >>> reader = BLOND("blond/BLOND-50",              # doctest: +SKIP
            ...                appliance_log="blond/appliance_log.json",
            ...                units=["medal-1"], max_seconds=60)
    """

    dataset = "blond"

    def __init__(
        self,
        dirpath: str | Path,
        *,
        appliance_log: str | Path | dict | None = None,
        units: list[str] | None = None,
        days: list[str] | None = None,
        sockets: list[int] | None = None,
        phases: list[int] | None = None,
        max_files: int | None = 1,
        max_seconds: float | None = None,
        time_range: tuple[float, float] | None = None,
        remove_offset: bool = True,
        f0: float = 50.0,
    ) -> None:
        self.dirpath = Path(dirpath).expanduser()
        if not self.dirpath.is_dir():
            raise NotADirectoryError(f"{self.dirpath} is not a directory")

        self.units = units
        self.days = days
        self.sockets = sockets or list(range(1, 7))
        self.phases = phases or [1, 2, 3]
        self.max_files = max_files
        self.max_seconds = max_seconds
        self.time_range = time_range
        self.remove_offset = remove_offset
        self.f0 = f0

        if isinstance(appliance_log, dict):
            self.log = appliance_log
        elif appliance_log is not None:
            self.log = json.loads(Path(appliance_log).expanduser().read_text())
        else:
            self.log = {}
            logger.warning("BLOND: no appliance log given; MEDAL sockets will be skipped")

    def __iter__(self) -> Iterator[Recording]:
        for day in self._days():
            for unit in self._units(day):
                yield from self._read_unit(day, unit)

    def __len__(self) -> int:
        return sum(len(self._units(day)) for day in self._days())

    # -- getting the data --------------------------------------------------- #

    @classmethod
    def plan(
        cls,
        *,
        resolution: str = "BLOND-50",
        units: list[str] | tuple[str, ...] = ("clear",),
        days: list[str] | None = None,
        time_range: tuple[float, float] | None = None,
        max_files: int | None = 1,
        index_cache: str | Path | None = None,
    ) -> Plan:
        """Which files this configuration would need, without fetching any.

        Args:
            resolution: ``"BLOND-50"`` or ``"BLOND-250"``.
            units: ``"clear"`` and/or ``"medal-1"`` .. ``"medal-15"``.
            days: ``YYYY-MM-DD`` strings. ``None`` takes what ``time_range`` says.
            time_range: ``(start, stop)`` unix seconds.
            max_files: files per unit per day.
            index_cache: directory to keep the FTP listings in.

        Returns:
            A :class:`~nilmframe.sources.Plan`.

        Example:
            >>> from nilmframe.readers import BLOND
            >>> print(BLOND.plan(units=['clear'],           # doctest: +SKIP
            ...                  days=['2016-09-30']).summary())
        """
        from nilmframe.sources import BLONDSource

        return BLONDSource(index_cache=index_cache).plan(
            resolution=resolution,
            units=units,
            days=days,
            time_range=time_range,
            max_files=max_files,
        )

    @classmethod
    def download(
        cls,
        cache: str | Path,
        *,
        resolution: str = "BLOND-50",
        units: list[str] | tuple[str, ...] = ("clear",),
        days: list[str] | None = None,
        time_range: tuple[float, float] | None = None,
        max_files: int | None = 1,
        workers: int = 2,
        max_bytes: int | None = None,
        progress: bool = True,
        **kwargs,
    ) -> BLOND:
        """Fetch a subset into ``cache`` and return a reader over it.

        Args:
            cache: directory to fetch into.
            resolution: ``"BLOND-50"`` or ``"BLOND-250"``.
            units: which units to fetch.
            days: ``YYYY-MM-DD`` strings.
            time_range: ``(start, stop)`` unix seconds.
            max_files: files per unit per day. A CLEAR file is 118 MB and a MEDAL
                file 31 MB, and a day of the whole rig is 42 GB, so this matters.
            workers: concurrent transfers. The delivery server is shared
                infrastructure and the default is deliberately gentle.
            max_bytes: refuse a plan larger than this before transferring.
            progress: print one line per file as it lands.
            **kwargs: passed to the constructor, e.g. ``max_seconds``.

        Returns:
            A reader over the fetched subset.

        Example:
            >>> from nilmframe.readers import BLOND
            >>> reader = BLOND.download("~/.cache/nilmframe/blond",  # doctest: +SKIP
            ...                         units=["clear", "medal-1"],
            ...                         days=["2016-09-30"], max_seconds=10)
        """
        from nilmframe.sources import materialize

        root = Path(cache).expanduser()
        plan = cls.plan(
            resolution=resolution,
            units=units,
            days=days,
            time_range=time_range,
            max_files=max_files,
            index_cache=root / ".index",
        )
        paths, _ = materialize(
            plan,
            root,
            workers=workers,
            max_bytes=max_bytes,
            progress=print if progress else None,
        )
        return cls(
            paths["dirpath"],
            appliance_log=paths["appliance_log"],
            units=list(units),
            days=days,
            max_files=max_files,
            time_range=time_range,
            **kwargs,
        )

    # -- internals ---------------------------------------------------------- #

    def _days(self) -> list[str]:
        found = sorted(p.name for p in self.dirpath.iterdir() if p.is_dir())
        return [d for d in found if self.days is None or d in self.days]

    def _units(self, day: str) -> list[str]:
        found = sorted(p.name for p in (self.dirpath / day).iterdir() if p.is_dir())
        return [u for u in found if self.units is None or u in self.units]

    def _files(self, day: str, unit: str) -> list[tuple[Path, dict]]:
        out = []
        for path in sorted((self.dirpath / day / unit).glob("*.hdf5")):
            if path.name.startswith("summary"):
                continue  # the 1 Hz per-day roll-up, not a waveform recording
            info = parse_blond_name(path.name)
            if info is None:
                logger.warning("BLOND: cannot parse %s", path.name)
                continue
            if self.time_range and not (self.time_range[0] <= info["t0"] <= self.time_range[1]):
                continue
            out.append((path, info))
        out.sort(key=lambda item: item[1]["t0"])
        return out[: self.max_files] if self.max_files else out

    def _read_unit(self, day: str, unit: str) -> Iterator[Recording]:
        try:
            import h5py
        except ImportError as exc:  # pragma: no cover - exercised by the extras matrix
            raise ImportError("reading BLOND needs h5py: pip install 'nilmframe[readers]'") from exc

        for path, info in self._files(day, unit):
            with h5py.File(path, "r") as handle:
                fs = float(handle.attrs.get("frequency", 0)) or None
                if fs is None:
                    logger.warning("BLOND: %s has no frequency attribute", path.name)
                    continue
                take = int(self.max_seconds * fs) if self.max_seconds else None
                if unit == "clear":
                    yield from self._clear(handle, info, day, fs, take, path)
                else:
                    yield from self._medal(handle, info, day, unit, fs, take, path)

    def _calibrated(self, dataset, take: int | None, fs: float) -> np.ndarray:
        """Raw ADC counts to volts or amperes, with the residual offset removed.

        Two steps, and the second is not optional. The ``calibration_factor``
        attribute scales counts to physical units. But the offset subtracted
        during recording was the converter's nominal midpoint, and the true zero
        drifts away from it -- on a MEDAL socket the leftover is of the same order
        as the signal, so ``current1`` arrives with a mean of -0.21 A against an
        RMS of 0.21 A. Read it as-is and the channel is mostly its own offset:
        every RMS, every power, every V-I trajectory wrong.

        The dataset's authors prescribe the remedy -- normalise the mean over each
        mains cycle -- and doing it per cycle rather than once for the file is
        what tracks the drift.
        """
        raw = dataset[:take] if take else dataset[:]
        values = raw.astype(np.float32) * np.float32(dataset.attrs["calibration_factor"])
        if not self.remove_offset:
            return np.ascontiguousarray(values)

        period = max(1, round(fs / self.f0))
        whole = (values.size // period) * period
        if whole:
            block = values[:whole].reshape(-1, period)
            block -= block.mean(axis=1, keepdims=True)
        if whole < values.size:
            tail = values[whole:]
            tail -= tail.mean()
        return np.ascontiguousarray(values)

    def _clear(self, handle, info, day, fs, take, path) -> Iterator[Recording]:
        for phase in self.phases:
            volts, amps = f"voltage{phase}", f"current{phase}"
            if volts not in handle or amps not in handle:
                continue
            yield Recording(
                dataset=self.dataset,
                house="blond",
                # Every unit recorded on a day shares one clock, so one session per
                # day is what lets a mains window find the submeters explaining it.
                session=day,
                kind=ChannelKind.MAINS,
                signals={
                    "v": self._calibrated(handle[volts], take, fs),
                    "i": self._calibrated(handle[amps], take, fs),
                },
                fs=fs,
                t0=info["t0"],
                meta={
                    "unit": "clear",
                    "phase": f"L{phase}",
                    "sequence": info["sequence"],
                    "source": path.name,
                },
            )

    def _medal(self, handle, info, day, unit, fs, take, path) -> Iterator[Recording]:
        config = self._config_at(unit, info["local"])
        if config is None:
            logger.warning("BLOND: no appliance log entry for %s at %s", unit, info["local"])
            return
        circuit = self.log.get(unit.upper(), {}).get("circuit_id")
        voltage = self._calibrated(handle["voltage"], take, fs) if "voltage" in handle else None

        for socket in self.sockets:
            amps = f"current{socket}"
            entry = config.get(f"socket_{socket}") or {}
            label = str(entry.get("class_name") or "").strip()
            if amps not in handle or not label:
                continue
            appliance = "_".join(label.lower().split())
            model = str(entry.get("appliance_name") or "").strip()
            signals = {"i": self._calibrated(handle[amps], take, fs)}
            if voltage is not None:
                signals["v"] = voltage
            yield Recording(
                dataset=self.dataset,
                house="blond",
                session=day,
                kind=ChannelKind.SUBMETER,
                appliance=appliance,
                brand="_".join(model.lower().split()) or None,
                # One socket held one physical unit for the span the log entry
                # covers; the model name is what must not straddle a split.
                instance_id=f"{appliance}:{model}" if model else f"{unit}:socket_{socket}",
                signals=signals,
                fs=fs,
                t0=info["t0"],
                meta={
                    "unit": unit,
                    "socket": socket,
                    "phase": circuit,
                    "rated_power": entry.get("power"),
                    "sequence": info["sequence"],
                    "source": path.name,
                },
            )

    def _config_at(self, unit: str, when: datetime) -> dict | None:
        """The socket configuration in force at ``when``.

        The log is a history, not a snapshot: sockets were re-used over seven
        months, so the newest entry at or before the recording is the one that
        describes it.
        """
        record = self.log.get(unit.upper())
        if not record:
            return None
        best = None
        for entry in record.get("entries", []):
            stamp = str(entry.get("timestamp", ""))
            try:
                moment = datetime.strptime(stamp, "%Y-%m-%dT%H-%M-%S")
            except ValueError:
                continue
            if moment <= when and (best is None or moment > best[0]):
                best = (moment, entry)
        # Before the first entry there is no configuration; fall back to the
        # earliest rather than dropping the recording.
        if best is None:
            entries = record.get("entries") or []
            return entries[0] if entries else None
        return best[1]
