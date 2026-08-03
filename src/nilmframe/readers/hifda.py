"""HIFDA reader.

HIFDA is 100 kSPS steady-state voltage and current for fourteen household
appliances, plus the empty grid measured under the same conditions -- which is
the part worth noticing, since it gives a background class most submetered
corpora do not have.

On-disk layout::

    Full_time_window_dataset/
      Current/Air_conditioner/AirConditioner10Current.txt
      Voltage/Air_conditioner/AirConditioner10Voltage.txt

One plain-text column of samples per file, one appliance class per directory, and
current and voltage in parallel trees whose filenames differ only in the trailing
word. So a recording is a *pair*, and the reader's job is mostly to match them up
and refuse the ones that are missing a half.

The release ships the same measurements windowed four ways -- ``10.24ms``,
``163.84ms``, ``1310.72ms`` and ``Full_time`` -- which is 770,581 files in total.
That is a choice you make when fetching rather than when reading; see
:class:`~nilmframe.sources.HIFDASource`.

Two things about the measurement are easy to miss and change what the data means.

**The files hold ADC volts, not volts and amperes.** Samples span 0 to 3.3 V, the
converter's output range, and the release documents the affine conversion to
physical units. The reader applies it, so what reaches the store is amperes and
volts like every other corpus.

**The grid fundamental is not in the data.** The voltage channel is band-limited
to 300 Hz -- 50 kHz and the current to roughly 30 Hz -- 50 kHz, so the 50 Hz
component is filtered out of both. That is a deliberate choice by the authors,
who were after the high-frequency signature, but it means this corpus does not
support anything that needs the fundamental: cycle alignment has no zero
crossings to find, and instantaneous power computed from ``v * i`` is not the
appliance's power. Use it for high-frequency representations, not for load
disaggregation against an aggregate.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from nilmframe.store.schema import ChannelKind, Recording

if TYPE_CHECKING:  # `nilmframe.sources` is imported only when data is fetched
    from nilmframe.sources import Plan

__all__ = ["HIFDA"]

logger = logging.getLogger(__name__)

#: The empty grid is measured like an appliance but is the absence of one, so it
#: becomes a mains channel rather than a submeter with a nonsense label.
BACKGROUND = "emptygrid"

#: ``(offset, scale)`` per quantity, from the release's own Readme: the stored
#: samples are the converter's 0--3.3 V output, and ``(sample - offset) * scale``
#: is the ampere or volt it represents. Without this the store would hold ADC
#: volts labelled as amperes, which nothing downstream could detect.
CALIBRATION: dict[str, tuple[float, float]] = {
    "i": (1.6462, 12.55),
    "v": (1.5992, 33.64),
}


def _appliance_of(directory: str) -> str:
    """``Air_conditioner`` to ``air_conditioner``."""
    return "_".join(directory.lower().replace("-", "_").split())


class HIFDA:
    """Iterate HIFDA recordings as :class:`Recording` objects.

    Args:
        dirpath: a window directory holding ``Current/`` and ``Voltage/``.
        fs: sampling rate. The release is 100 kSPS throughout.
        appliances: appliance directories to read. ``None`` reads all present.
        max_seconds: seconds to take from the start of each recording.
        limit: stop after this many recordings.
        calibrate: convert the stored ADC volts to amperes and volts using the
            release's documented equations. Turning this off gives the numbers
            exactly as published, which is what you want when comparing against
            work that used them raw.

    Example:
        >>> from nilmframe.readers import HIFDA
        >>> from nilmframe.store import StoreWriter
        >>> reader = HIFDA("hifda/Full_time_window_dataset")   # doctest: +SKIP
        >>> with StoreWriter("stores/hifda") as w:             # doctest: +SKIP
        ...     for rec in reader:
        ...         w.add(rec)
    """

    dataset = "hifda"

    def __init__(
        self,
        dirpath: str | Path,
        *,
        fs: float = 100_000.0,
        appliances: list[str] | None = None,
        max_seconds: float | None = None,
        limit: int | None = None,
        calibrate: bool = True,
    ) -> None:
        self.dirpath = Path(dirpath).expanduser()
        if not self.dirpath.is_dir():
            raise NotADirectoryError(f"{self.dirpath} is not a directory")
        self.fs = fs
        self.appliances = {_appliance_of(a) for a in appliances} if appliances else None
        self.max_seconds = max_seconds
        self.limit = limit
        self.calibrate = calibrate
        self.pairs = self._pairs()

    def __len__(self) -> int:
        return len(self.pairs)

    def __iter__(self) -> Iterator[Recording]:
        take = int(self.max_seconds * self.fs) if self.max_seconds else None
        for appliance, name, current, voltage in self.pairs:
            amps = self._read(current, take, "i" if self.calibrate else None)
            volts = self._read(voltage, take, "v" if self.calibrate else None)
            if amps is None or volts is None:
                continue
            n = min(amps.size, volts.size)
            if n < 2:
                logger.warning("HIFDA: %s is too short to use", name)
                continue

            is_background = appliance.replace("_", "") == BACKGROUND
            yield Recording(
                dataset=self.dataset,
                house="hifda",
                session=name,
                kind=ChannelKind.MAINS if is_background else ChannelKind.SUBMETER,
                appliance=None if is_background else appliance,
                # Each class is one physical appliance measured repeatedly, so
                # every recording of it is the same unit and must not straddle a
                # split.
                instance_id=None if is_background else f"hifda:{appliance}",
                signals={"i": amps[:n], "v": volts[:n]},
                fs=self.fs,
                meta={
                    "source": current.name,
                    "calibrated": self.calibrate,
                    # Recorded so nothing downstream mistakes this for a corpus
                    # whose voltage carries the fundamental.
                    "bandwidth_hz": "300-50000 (v), 30-50000 (i)",
                },
            )

    # -- getting the data --------------------------------------------------- #

    @classmethod
    def plan(
        cls,
        *,
        window: str = "Full_time",
        appliances: list[str] | None = None,
        limit: int | None = None,
        index_cache: str | Path | None = None,
    ) -> Plan:
        """Which files this configuration would need, without fetching any.

        Args:
            window: ``"Full_time"``, ``"1310.72ms"``, ``"163.84ms"`` or
                ``"10.24ms"``. The short windows hold hundreds of thousands of
                slices; the full-time one holds 750 whole recordings.
            appliances: appliance names to take. ``None`` takes all fifteen.
            limit: take at most this many recordings.
            index_cache: directory for the archive's parsed directory, which for
                this record has 770,612 entries and is worth keeping.

        Returns:
            A :class:`~nilmframe.sources.Plan`.
        """
        from nilmframe.sources import HIFDASource

        return HIFDASource(index_cache=index_cache).plan(
            window=window, appliances=appliances, limit=limit
        )

    @classmethod
    def download(
        cls,
        cache: str | Path,
        *,
        window: str = "Full_time",
        appliances: list[str] | None = None,
        limit: int | None = None,
        workers: int = 4,
        max_bytes: int | None = None,
        progress: bool = True,
        **kwargs,
    ) -> HIFDA:
        """Fetch a subset into ``cache`` and return a reader over it.

        Args:
            cache: directory to fetch into.
            window: which windowing of the release to take.
            appliances: appliance names to take.
            limit: take at most this many recordings.
            workers: concurrent extractions.
            max_bytes: refuse a plan larger than this before transferring.
            progress: print one line per file as it lands.
            **kwargs: passed to the constructor, e.g. ``max_seconds``.

        Returns:
            A reader over the fetched subset.

        Example:
            >>> from nilmframe.readers import HIFDA
            >>> reader = HIFDA.download("~/.cache/nilmframe/hifda",  # doctest: +SKIP
            ...                         appliances=["Microwave"], limit=4)
        """
        from nilmframe.sources import materialize

        root = Path(cache).expanduser()
        plan = cls.plan(
            window=window, appliances=appliances, limit=limit, index_cache=root / ".index"
        )
        paths, _ = materialize(
            plan,
            root,
            workers=workers,
            max_bytes=max_bytes,
            progress=print if progress else None,
        )
        # `limit` bounds the reader too, not just the fetch: a cache filled by an
        # earlier, wider run would otherwise hand back more than was asked for.
        return cls(paths["dirpath"], appliances=appliances, limit=limit, **kwargs)

    # -- internals ---------------------------------------------------------- #

    def _pairs(self) -> list[tuple[str, str, Path, Path]]:
        """Matched ``(appliance, name, current, voltage)`` recordings."""
        current_root = self.dirpath / "Current"
        voltage_root = self.dirpath / "Voltage"
        if not current_root.is_dir():
            raise NotADirectoryError(
                f"{current_root} is missing; expected a HIFDA window directory"
            )

        out = []
        for path in sorted(current_root.rglob("*Current.txt")):
            appliance = _appliance_of(path.parent.name)
            if self.appliances is not None and appliance not in self.appliances:
                continue
            partner = voltage_root / path.parent.name / path.name.replace("Current", "Voltage")
            if not partner.exists():
                logger.warning("HIFDA: %s has no matching voltage file", path.name)
                continue
            out.append((appliance, path.stem[: -len("Current")], path, partner))
        return out[: self.limit] if self.limit else out

    @staticmethod
    def _read(path: Path, take: int | None, quantity: str | None) -> np.ndarray | None:
        try:
            values = np.loadtxt(path, dtype=np.float32, max_rows=take)
        except ValueError as exc:
            logger.warning("HIFDA: cannot parse %s: %s", path.name, exc)
            return None
        values = np.atleast_1d(values)
        if quantity is not None:
            offset, scale = CALIBRATION[quantity]
            values = (values - np.float32(offset)) * np.float32(scale)
        return np.ascontiguousarray(values, dtype=np.float32)
