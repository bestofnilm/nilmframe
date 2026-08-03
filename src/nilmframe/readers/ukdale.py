"""UK-DALE reader.

UK-DALE is the dataset the LF/HF comparison needs, because it is the one that
carries both: per-appliance submeters at ~1/6 Hz *and*, for house 1, a 16 kHz
stereo recording of the mains. Same house, same clock, two rates. That is exactly
the structure ``edframe_concept_note.tex`` asks for -- "one view of the data in the
low-frequency domain and another in the high-frequency domain" -- and it only
works if both land in one store with a shared time base.

On-disk layout::

    house_1/
      labels.dat          channel number -> appliance name
      channel_1.dat       "<unix timestamp> <watts>" per line; channel 1 is mains
      channel_2.dat       ...
      mains.flac          optional 16 kHz stereo: voltage, current

Two things this reader does that a naive one would not:

**It puts the samples on a uniform grid.** The store's contract is a constant
sampling rate, and UK-DALE's timestamps drift. Power meters report a step
function, so the resampling is a forward fill onto a regular grid rather than an
interpolation -- interpolating would invent ramps between readings that the
appliance never made.

**It splits on gaps instead of filling them.** A monitor that dropped out for
three hours leaves a hole, and forward-filling it would manufacture three hours of
an appliance being on. Anything longer than ``max_gap_s`` starts a new session,
which is also what keeps the mains and the submeters honestly aligned.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from nilmframe.store.schema import ChannelKind, Recording

if TYPE_CHECKING:  # `nilmframe.sources` is imported only when data is fetched
    from nilmframe.sources import Plan

__all__ = ["UKDALE", "forward_fill_to_grid", "split_on_gaps"]

logger = logging.getLogger(__name__)


def split_on_gaps(timestamps: np.ndarray, max_gap_s: float) -> list[tuple[int, int]]:
    """Contiguous ``[start, stop)`` runs separated by gaps longer than ``max_gap_s``.

    Example:
        >>> from nilmframe.readers.ukdale import split_on_gaps
        >>> stamps = np.array([0., 6., 12., 5000., 5006., 5012.])
        >>> split_on_gaps(stamps, max_gap_s=300.0)
        [(0, 3), (3, 6)]
    """
    if timestamps.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(timestamps) > max_gap_s) + 1
    edges = [0, *breaks.tolist(), timestamps.size]
    return [(a, b) for a, b in pairwise(edges) if b - a > 1]


def forward_fill_to_grid(
    timestamps: np.ndarray, values: np.ndarray, fs: float
) -> tuple[np.ndarray, float]:
    """Resample an irregular step signal onto a uniform grid.

    Args:
        timestamps: strictly increasing seconds.
        values: one reading per timestamp.
        fs: target rate in Hz.

    Returns:
        ``(resampled, t0)``.

    Example:
        >>> from nilmframe.readers.ukdale import forward_fill_to_grid
        >>> stamps = np.array([0., 6., 12.])
        >>> watts = np.array([0., 100., 0.])
        >>> series, t0 = forward_fill_to_grid(stamps, watts, fs=1.0)
        >>> series.shape, t0
        ((13,), 0.0)
        >>> sorted(set(series.tolist()))
        [0.0, 100.0]
    """
    t0, t1 = float(timestamps[0]), float(timestamps[-1])
    n = max(1, int(np.floor((t1 - t0) * fs)) + 1)
    grid = t0 + np.arange(n) / fs
    # searchsorted with "right" then step back: each grid point takes the most
    # recent reading at or before it, which is what a step function means.
    idx = np.searchsorted(timestamps, grid, side="right") - 1
    return values[np.clip(idx, 0, values.size - 1)].astype(np.float32), t0


class UKDALE:
    """Iterate UK-DALE channels as :class:`Recording` objects.

    Args:
        dirpath: dataset root, containing ``house_*`` directories.
        houses: house numbers to read. ``None`` reads all found.
        rate_hz: uniform rate for the low-frequency channels. UK-DALE's native
            cadence is about 1/6 Hz.
        max_gap_s: gaps longer than this split a channel into separate sessions
            rather than being filled.
        high_freq: also read the waveform mains where present.
        high_freq_root: separate root for the waveform files. The public
            distribution ships them apart from the meter readings, as
            ``high_freq/house_N/YYYY/wkNN/vi-<unix_ts>_<fraction>.flac``, so this
            points at that ``high_freq`` directory while ``dirpath`` points at
            ``low_freq``. When ``None``, a ``house_N/mains.flac`` beside the meter
            readings is used instead.
        hf_fs: sampling rate of the waveform recording.
        hf_calibration: ``(volt, amp)`` scaling from normalised PCM. The defaults
            were derived from house 1 by scaling channel 0 to 230 V RMS and then
            choosing the current scale so that the waveform's active power matches
            what the low-frequency meter reported over the same hour -- see
            ``notebooks/`` for the derivation. UK-DALE's own
            ``calibration_house_N.cfg`` gives ``volts_per_adc_step`` for the ADC
            input, upstream of the divider, so it does not convert the stored
            samples directly.
        max_seconds: seconds to read from each *waveform* file. This is the knob
            that matters for cost -- one file is an hour at 16 kHz -- and it
            deliberately does not touch the meter channels: a second of a 1/6 Hz
            series is not a sample, and truncating one to satisfy a waveform
            budget would collapse the run span that waveform files are matched
            against. Use ``time_range`` to bound the meter extent.
        max_hf_files: waveform files to take per house. Each is an hour at 16 kHz
            and roughly 200 MB.
        time_range: ``(start, stop)`` unix seconds. Meter files span years at 6 s
            and reading one in full costs minutes; this bounds ingestion to the
            span actually wanted, and stops early because the files are sorted.

    Example:
        The public distribution keeps the meter readings and the waveform files in
        separate trees, so both roots are given. One waveform file per house is
        plenty to see the shape of things -- each is an hour at 16 kHz::

            >>> from nilmframe.readers import UKDALE
            >>> from nilmframe.store import StoreWriter
            >>> reader = UKDALE(                              # doctest: +SKIP
            ...     "ukdale/low_freq",
            ...     high_freq_root="ukdale/high_freq",
            ...     houses=[1],
            ...     max_hf_files=1,
            ...     max_seconds=60,
            ... )
            >>> with StoreWriter("stores/ukdale") as w:       # doctest: +SKIP
            ...     for rec in reader:
            ...         w.add(rec)

        Meter files span years at 6 s. Bound the ingest to the span you want rather
        than reading them whole::

            >>> reader = UKDALE("ukdale/low_freq",            # doctest: +SKIP
            ...                 time_range=(1362268800, 1362355200),
            ...                 high_freq=False)
    """

    dataset = "ukdale"

    def __init__(
        self,
        dirpath: str | Path,
        *,
        houses: list[int] | None = None,
        rate_hz: float = 1.0 / 6.0,
        max_gap_s: float = 300.0,
        high_freq: bool = True,
        high_freq_root: str | Path | None = None,
        hf_fs: float = 16000.0,
        hf_calibration: tuple[float, float] = (388.45, 251.51),
        max_seconds: float | None = None,
        max_hf_files: int | None = 1,
        time_range: tuple[float, float] | None = None,
    ) -> None:
        self.dirpath = Path(dirpath).expanduser()
        if not self.dirpath.is_dir():
            raise NotADirectoryError(f"{self.dirpath} is not a directory")

        self.rate_hz = rate_hz
        self.max_gap_s = max_gap_s
        self.high_freq = high_freq
        self.high_freq_root = Path(high_freq_root).expanduser() if high_freq_root else None
        self.hf_fs = hf_fs
        self.hf_calibration = hf_calibration
        self.max_seconds = max_seconds
        self.max_hf_files = max_hf_files
        self.time_range = time_range
        # Time spans of the low-frequency runs, filled as they are read, so a
        # waveform file can be assigned the session of the run that contains it.
        # Sharing a session is what lets its windows draw targets from the
        # submeters -- see WindowDataset's sibling lookup.
        self._runs: dict[str, list[tuple[str, float, float]]] = {}

        found = sorted(
            int(p.name.split("_")[1])
            for p in self.dirpath.glob("house_*")
            if p.is_dir() and p.name.split("_")[-1].isdigit()
        )
        self.houses = [h for h in found if houses is None or h in houses]
        if houses:
            missing = set(houses) - set(found)
            if missing:
                raise ValueError(f"houses not found under {self.dirpath}: {sorted(missing)}")

    def __len__(self) -> int:
        return len(self.houses)

    def __iter__(self) -> Iterator[Recording]:
        for house in self.houses:
            yield from self._read_house(house)

    # -- getting the data --------------------------------------------------- #

    @classmethod
    def plan(
        cls,
        *,
        houses: list[int] | tuple[int, ...] = (1,),
        channels: list[int] | None = None,
        time_range: tuple[float, float] | None = None,
        low_freq: bool = True,
        high_freq: bool = True,
        max_hf_files: int | None = 1,
        index_cache: str | Path | None = None,
    ) -> Plan:
        """Which files this configuration would need, without fetching any.

        Costs a handful of directory listings. Print the result before spending
        anything -- UK-DALE is measured in terabytes and one waveform file is
        200 MB, so the difference between a sensible request and a mistyped one
        is worth seeing first.

        Args:
            houses: house numbers. Only house 1 carries the 16 kHz waveforms.
            channels: meter channels to take; ``None`` takes all of them.
            time_range: ``(start, stop)`` unix seconds, bounding both halves.
            low_freq: include the meter channels.
            high_freq: include the waveform files.
            max_hf_files: waveform files per house. ``None`` takes the whole range.
            index_cache: directory to keep the listings in, so planning the same
                subset again costs nothing.

        Returns:
            A :class:`~nilmframe.sources.Plan`.

        Example:
            >>> from nilmframe.readers import UKDALE
            >>> plan = UKDALE.plan(houses=[1], channels=[1],   # doctest: +SKIP
            ...                    time_range=(1421784000, 1421870400))
            >>> print(plan.summary())                          # doctest: +SKIP
        """
        from nilmframe.sources import UKDALESource

        return UKDALESource(index_cache=index_cache).plan(
            houses=houses,
            channels=channels,
            time_range=time_range,
            low_freq=low_freq,
            high_freq=high_freq,
            max_hf_files=max_hf_files,
        )

    @classmethod
    def download(
        cls,
        cache: str | Path,
        *,
        houses: list[int] | tuple[int, ...] = (1,),
        channels: list[int] | None = None,
        time_range: tuple[float, float] | None = None,
        low_freq: bool = True,
        high_freq: bool = True,
        max_hf_files: int | None = 1,
        workers: int = 4,
        max_bytes: int | None = None,
        progress: bool = True,
        **kwargs,
    ) -> UKDALE:
        """Fetch a subset into ``cache`` and return a reader over it.

        The cache is an ordinary directory in the layout this reader documents,
        so it can be inspected, reused, and pointed at directly afterwards.
        Re-running skips whatever is already there.

        Args:
            cache: directory to fetch into.
            houses: house numbers.
            channels: meter channels to take; ``None`` takes all of them.
            time_range: ``(start, stop)`` unix seconds.
            low_freq: include the meter channels. Leaving them out costs the
                waveform windows their submeter targets -- see the module note.
            high_freq: include the waveform files.
            max_hf_files: waveform files per house, each an hour and about 200 MB.
            workers: concurrent downloads.
            max_bytes: refuse a plan larger than this before transferring.
            progress: print one line per file as it lands.
            **kwargs: passed to the constructor, e.g. ``rate_hz``.

        Returns:
            A reader over the fetched subset.

        Example:
            >>> from nilmframe.readers import UKDALE
            >>> reader = UKDALE.download(                      # doctest: +SKIP
            ...     "~/.cache/nilmframe/ukdale",
            ...     houses=[1], channels=[1, 5],
            ...     time_range=(1421784000, 1421870400), max_hf_files=2)
        """
        from nilmframe.sources import materialize

        root = Path(cache).expanduser()
        plan = cls.plan(
            houses=houses,
            channels=channels,
            time_range=time_range,
            low_freq=low_freq,
            high_freq=high_freq,
            max_hf_files=max_hf_files,
            index_cache=root / ".index",
        )
        paths, _ = materialize(
            plan,
            root,
            workers=workers,
            max_bytes=max_bytes,
            progress=print if progress else None,
        )
        window = time_range
        if window is not None:
            # A waveform takes the session of the meter run that *contains* its
            # start. Clipping the meter channels to exactly the window asked for
            # makes their first run begin at the first reading inside it -- a few
            # seconds after the waveform that was recorded on the hour -- so every
            # waveform window would land in `hf_only` with no submeter targets.
            # Reaching back one gap is enough for the run to enclose it.
            window = (window[0] - kwargs.get("max_gap_s", 300.0), window[1])

        return cls(
            paths["dirpath"],
            high_freq_root=paths["high_freq_root"] if high_freq else None,
            houses=list(houses),
            high_freq=high_freq,
            max_hf_files=max_hf_files,
            time_range=window,
            **kwargs,
        )

    # -- internals ---------------------------------------------------------- #

    def _read_house(self, house: int) -> Iterator[Recording]:
        root = self.dirpath / f"house_{house}"
        labels = self._read_labels(root)

        for channel, appliance in sorted(labels.items()):
            path = root / f"channel_{channel}.dat"
            if not path.exists():
                logger.warning("UK-DALE: %s listed in labels.dat but missing", path.name)
                continue
            yield from self._read_channel(house, channel, appliance, path)

        if self.high_freq:
            yield from self._read_high_freq(house, root)

    @staticmethod
    def _read_labels(root: Path) -> dict[int, str]:
        path = root / "labels.dat"
        if not path.exists():
            raise FileNotFoundError(f"{path} is missing; UK-DALE houses must carry labels.dat")
        labels: dict[int, str] = {}
        for line in path.read_text().splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[0].isdigit():
                labels[int(parts[0])] = "_".join(parts[1:]).lower()
        return labels

    def _load_dat(self, path: Path) -> tuple[np.ndarray, np.ndarray]:
        """Read a ``<timestamp> <watts>`` file, bounded by ``time_range``.

        Meter files span years at 6 s -- house 1's aggregate is about 11 million
        rows -- so this uses pandas' C parser rather than ``np.loadtxt``, and
        stops reading as soon as a chunk passes the end of the window, which the
        sorted timestamps make safe.
        """
        import pandas as pd

        if self.time_range is None:
            frame = pd.read_csv(path, sep=r"\s+", header=None, names=["t", "w"], engine="c")
            return frame["t"].to_numpy(np.float64), frame["w"].to_numpy(np.float64)

        start, stop = self.time_range
        keep: list[pd.DataFrame] = []
        reader = pd.read_csv(
            path, sep=r"\s+", header=None, names=["t", "w"], engine="c", chunksize=1_000_000
        )
        for chunk in reader:
            if chunk["t"].iloc[0] > stop:
                break
            keep.append(chunk[(chunk["t"] >= start) & (chunk["t"] <= stop)])
        if not keep:
            return np.empty(0), np.empty(0)
        frame = pd.concat(keep, ignore_index=True)
        return frame["t"].to_numpy(np.float64), frame["w"].to_numpy(np.float64)

    def _read_channel(
        self, house: int, channel: int, appliance: str, path: Path
    ) -> Iterator[Recording]:
        timestamps, watts = self._load_dat(path)
        if timestamps.size == 0:
            return
        order = np.argsort(timestamps)
        timestamps, watts = timestamps[order], watts[order]

        # UK-DALE names channel 1 "aggregate"; it is the house mains, and its
        # per-appliance truth comes from the other channels rather than from
        # itself.
        is_mains = appliance in {"aggregate", "mains"} or channel == 1

        for run, (a, b) in enumerate(split_on_gaps(timestamps, self.max_gap_s)):
            series, t0 = forward_fill_to_grid(timestamps[a:b], watts[a:b], self.rate_hz)
            if series.size < 2:
                continue
            # Remember the run's span so a waveform file recorded inside it can be
            # given the same session, and therefore reach these submeters.
            self._runs.setdefault(f"house_{house}", []).append(
                (f"run_{run}", t0, t0 + series.size / self.rate_hz)
            )
            yield Recording(
                dataset=self.dataset,
                house=f"house_{house}",
                # The session is what ties a mains window to the submeter windows
                # that explain it, so every channel of one contiguous run must
                # share it.
                session=f"run_{run}",
                kind=ChannelKind.MAINS if is_mains else ChannelKind.SUBMETER,
                appliance=None if is_mains else appliance,
                instance_id=None if is_mains else f"house_{house}:{appliance}",
                signals={"p": series},
                fs=self.rate_hz,
                t0=t0,
                meta={"channel": channel, "source": path.name},
            )

    def high_freq_files(self, house: int) -> list[Path]:
        """Waveform files for a house, oldest first.

        Handles both layouts: the public distribution's
        ``high_freq/house_N/YYYY/wkNN/vi-<ts>_<frac>.flac``, and a single
        ``house_N/mains.flac`` beside the meter readings.
        """
        if self.high_freq_root is not None:
            root = self.high_freq_root / f"house_{house}"
            files = sorted(root.rglob("vi-*.flac"), key=lambda p: self._timestamp_of(p) or 0.0)
        else:
            single = self.dirpath / f"house_{house}" / "mains.flac"
            files = [single] if single.exists() else []
        if self.time_range is not None:
            start, stop = self.time_range
            files = [
                p for p in files if (ts := self._timestamp_of(p)) is None or start <= ts <= stop
            ]
        return files

    @staticmethod
    def _timestamp_of(path: Path) -> float | None:
        """Recording start from a ``vi-<seconds>_<fraction>.flac`` name.

        The filename *is* the clock. It is what puts a waveform file and the meter
        readings on the same time base without a separate index, and therefore what
        lets a waveform window take its targets from the submeters.
        """
        stem = path.stem
        if not stem.startswith("vi-"):
            return None
        try:
            seconds, _, fraction = stem[3:].partition("_")
            return float(f"{int(seconds)}.{fraction}") if fraction else float(seconds)
        except ValueError:
            return None

    def _session_for(self, house: int, timestamp: float) -> str:
        """The low-frequency run containing ``timestamp``, so siblings can be found."""
        for name, start, stop in self._runs.get(f"house_{house}", []):
            if start <= timestamp <= stop:
                return name
        return "hf_only"

    def _read_high_freq(self, house: int, root: Path) -> Iterator[Recording]:
        files = self.high_freq_files(house)
        if not files:
            return
        try:
            import soundfile as sf
        except ImportError as exc:  # pragma: no cover - exercised by the extras matrix
            raise ImportError(
                "reading UK-DALE high-frequency mains needs soundfile: "
                "pip install 'nilmframe[readers]'"
            ) from exc

        volt, amp = self.hf_calibration
        for path in files[: self.max_hf_files]:
            frames = int(self.max_seconds * self.hf_fs) if self.max_seconds else -1
            data, fs = sf.read(path, dtype="float32", always_2d=True, frames=frames)
            if data.shape[1] < 2:
                logger.warning("UK-DALE: %s has %d channels, expected 2", path.name, data.shape[1])
                continue

            t0 = self._timestamp_of(path)
            if t0 is None:
                t0 = self._read_hf_start(root)
            yield Recording(
                dataset=self.dataset,
                house=f"house_{house}",
                session=self._session_for(house, t0),
                kind=ChannelKind.MAINS,
                signals={
                    "v": np.ascontiguousarray(volt * data[:, 0]),
                    "i": np.ascontiguousarray(amp * data[:, 1]),
                },
                fs=float(fs),
                t0=t0,
                meta={"source": path.name, "resolution": "high"},
            )

    @staticmethod
    def _read_hf_start(root: Path) -> float:
        """Absolute start of the waveform recording, so it shares a clock with the submeters."""
        path = root / "mains.dat"
        if not path.exists():
            return 0.0
        with open(path) as fh:
            first = fh.readline().split()
        return float(first[0]) if first else 0.0
