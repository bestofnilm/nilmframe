"""FIRED reader.

Fifty-two days of a three-room German apartment, fully labelled: a three-phase
smart meter at 8 kHz and twenty-one plug-level meters at 2 kHz, plus 50 Hz and
1 Hz power summaries of the same measurements. That is the same shape as BLOND --
a measured aggregate with every appliance behind it separately metered -- in a
home rather than an office, and with the raw waveform for both sides.

**It is stored as audio.** Every file is a Matroska container holding WavPack
streams, which is a genuinely good choice -- WavPack is lossless and compresses
these signals well -- but it means decoding needs ``ffmpeg`` on ``PATH`` rather
than a Python library. The containers carry their own metadata:

* ``TIMESTAMP`` -- the recording's start in unix seconds, so a file needs no
  index to be placed on the clock;
* ``CHANNEL_TAGS`` -- ``v,i`` for waveforms and ``p,q,s`` for summaries, which is
  the store's own vocabulary already;
* one stream per phase for the smart meter, titled ``smartmeter001 L1`` and so on.

Two conversions are needed and neither is announced in the data.

**The high-frequency current is in milliamperes.** Read it as amperes and the
apartment draws a hundred kilowatts. The check is in the dataset itself: scaling
by a thousand makes the waveform's mean ``v * i`` agree with the 1 Hz summary's
own active power to well under a percent -- 58.1 W against 57.8 W on the first
ten minutes of phase L1.

**Some plug meters were installed backwards, and the publishers already fixed
it.** ``deviceMapping.json`` flags those with ``flip``. It is tempting to negate
their current on read -- the flag exists, so surely it is for something -- and
doing so is wrong: the published waveforms already have the correction applied,
so a second negation turns the appliance into a generator.

Measured against the dataset's own 1 Hz summary, on 2020-06-14:

===================  ======  ==============  ==============
meter                flip    read as-is      negated
===================  ======  ==============  ==============
powermeter08 (lamp)  true    **+598 W**      -598 W
powermeter09 (fridge)  --    **+61 W**       -61 W
===================  ======  ==============  ==============

The summary reports +599 W and +62 W for the same windows. So the flag is a
record of the physical installation, not an instruction to the reader, and the
current is passed through with the sign it was published with.
"""

from __future__ import annotations

import json
import logging
import subprocess
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from nilmframe.store.schema import ChannelKind, Recording

if TYPE_CHECKING:  # `nilmframe.sources` is imported only when data is fetched
    from nilmframe.sources import Plan

__all__ = ["FIRED", "MILLIAMPS_PER_AMP", "RESOLUTIONS"]

logger = logging.getLogger(__name__)

#: The published resolutions. ``highFreq`` is the waveform; the rest are power.
RESOLUTIONS = ("1Hz", "50Hz", "highFreq")

#: The waveform current is stored in milliamperes -- see the module note.
MILLIAMPS_PER_AMP = 1000.0

_AGGREGATE_PREFIX = "smartmeter"


def _probe(path: Path) -> list[dict]:
    """Every audio stream in a container, with its tags."""
    try:
        out = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "a",
                "-show_entries",
                "stream=index,sample_rate,channels:stream_tags",
                "-of",
                "json",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    except FileNotFoundError as exc:  # pragma: no cover - exercised by the extras matrix
        raise RuntimeError(
            "reading FIRED needs ffmpeg on PATH: it is stored as WavPack in Matroska. "
            "Install ffmpeg (`apt install ffmpeg`, `brew install ffmpeg`)."
        ) from exc
    if out.returncode != 0:
        logger.warning("FIRED: ffprobe failed on %s: %s", path.name, out.stderr.strip()[:200])
        return []
    return json.loads(out.stdout or "{}").get("streams", [])


def _decode(path: Path, index: int, channels: int, limit: int | None) -> np.ndarray | None:
    """One stream as ``(n, channels)`` float32."""
    out = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(path),
            "-map",
            f"0:a:{index}",
            "-f",
            "f32le",
            "-acodec",
            "pcm_f32le",
            "-",
        ],
        capture_output=True,
        timeout=1800,
        check=False,
    )
    if out.returncode != 0:
        logger.warning("FIRED: ffmpeg failed on %s: %s", path.name, out.stderr[:200])
        return None
    flat = np.frombuffer(out.stdout, dtype=np.float32)
    usable = (flat.size // channels) * channels
    block = flat[:usable].reshape(-1, channels)
    return block[:limit] if limit else block


class FIRED:
    """Iterate FIRED recordings as :class:`Recording` objects.

    Args:
        dirpath: the FIRED root, holding ``summary/``, ``highFreq/`` and ``info/``.
        resolution: one of :data:`RESOLUTIONS`.
        meters: meter names, e.g. ``["smartmeter001", "powermeter08"]``. ``None``
            reads every meter present.
        days: ``YYYY_MM_DD`` strings as they appear in the filenames.
        device_mapping: path to ``deviceMapping.json``, or the parsed dict.
            Without it the plug meters have no appliance names and are skipped.
        max_files: files to read per meter. A ten-minute smart-meter file is
            78 MB and 4.8 million samples per phase.
        max_seconds: seconds to take from the start of each file.

    Example:
        >>> from nilmframe.readers import FIRED
        >>> from nilmframe.store import StoreWriter
        >>> reader = FIRED("fired", resolution="1Hz")        # doctest: +SKIP
        >>> with StoreWriter("stores/fired") as w:           # doctest: +SKIP
        ...     for rec in reader:
        ...         w.add(rec)
    """

    dataset = "fired"

    def __init__(
        self,
        dirpath: str | Path,
        *,
        resolution: str = "1Hz",
        meters: list[str] | None = None,
        days: list[str] | None = None,
        device_mapping: str | Path | dict | None = None,
        max_files: int | None = 1,
        max_seconds: float | None = None,
    ) -> None:
        self.dirpath = Path(dirpath).expanduser()
        if not self.dirpath.is_dir():
            raise NotADirectoryError(f"{self.dirpath} is not a directory")
        if resolution not in RESOLUTIONS:
            raise ValueError(f"resolution must be one of {RESOLUTIONS}, got {resolution!r}")

        self.resolution = resolution
        self.meters = meters
        self.days = days
        self.max_files = max_files
        self.max_seconds = max_seconds

        if isinstance(device_mapping, dict):
            self.mapping = device_mapping
        else:
            path = (
                Path(device_mapping).expanduser()
                if device_mapping
                else (self.dirpath / "info" / "deviceMapping.json")
            )
            self.mapping = json.loads(path.read_text()) if path.exists() else {}
            if not self.mapping:
                logger.warning("FIRED: no deviceMapping.json; plug meters will be skipped")

    @property
    def root(self) -> Path:
        """Where this resolution's meters live."""
        return (
            self.dirpath / "highFreq"
            if self.resolution == "highFreq"
            else self.dirpath / "summary" / self.resolution
        )

    def __len__(self) -> int:
        return sum(len(files) for _, files in self._targets())

    def __iter__(self) -> Iterator[Recording]:
        for meter, files in self._targets():
            for path in files:
                yield from self._read(meter, path)

    # -- getting the data --------------------------------------------------- #

    @classmethod
    def plan(
        cls,
        *,
        resolution: str = "1Hz",
        meters: list[str] | None = None,
        days: list[str] | None = None,
        max_files: int | None = 1,
        index_cache: str | Path | None = None,
    ) -> Plan:
        """Which files this configuration would need, without fetching any.

        Args:
            resolution: ``"1Hz"``, ``"50Hz"`` or ``"highFreq"``.
            meters: meter names. ``None`` takes the smart meter and every plug.
            days: ``YYYY_MM_DD`` strings.
            max_files: files per meter.
            index_cache: directory for the rsync listings.

        Returns:
            A :class:`~nilmframe.sources.Plan`.
        """
        from nilmframe.sources import FIREDSource

        return FIREDSource(index_cache=index_cache).plan(
            resolution=resolution, meters=meters, days=days, max_files=max_files
        )

    @classmethod
    def download(
        cls,
        cache: str | Path,
        *,
        resolution: str = "1Hz",
        meters: list[str] | None = None,
        days: list[str] | None = None,
        max_files: int | None = 1,
        workers: int = 2,
        max_bytes: int | None = None,
        progress: bool = True,
        **kwargs,
    ) -> FIRED:
        """Fetch a subset into ``cache`` and return a reader over it.

        Args:
            cache: directory to fetch into.
            resolution: which tier to take. The whole 8 kHz tier is 3.2 TB; the
                1 Hz summaries of everything are 1.7 GB.
            meters: meter names.
            days: ``YYYY_MM_DD`` strings.
            max_files: files per meter.
            workers: concurrent transfers.
            max_bytes: refuse a plan larger than this before transferring.
            progress: print one line per file as it lands.
            **kwargs: passed to the constructor, e.g. ``max_seconds``.

        Returns:
            A reader over the fetched subset.
        """
        from nilmframe.sources import materialize

        root = Path(cache).expanduser()
        plan = cls.plan(
            resolution=resolution,
            meters=meters,
            days=days,
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
            resolution=resolution,
            meters=meters,
            days=days,
            max_files=max_files,
            **kwargs,
        )

    # -- internals ---------------------------------------------------------- #

    def _targets(self) -> list[tuple[str, list[Path]]]:
        if not self.root.is_dir():
            return []
        out = []
        for directory in sorted(p for p in self.root.iterdir() if p.is_dir()):
            if self.meters is not None and directory.name not in self.meters:
                continue
            files = sorted(directory.glob("*.mkv"))
            if self.days is not None:
                files = [f for f in files if any(d in f.name for d in self.days)]
            if files:
                out.append((directory.name, files[: self.max_files] if self.max_files else files))
        return out

    def _read(self, meter: str, path: Path) -> Iterator[Recording]:
        is_aggregate = meter.startswith(_AGGREGATE_PREFIX)
        entry = self.mapping.get(meter, {})
        appliance = None
        if not is_aggregate:
            names = entry.get("appliances") or []
            if not names:
                logger.warning("FIRED: %s has no appliance in the mapping, skipping", meter)
                return
            appliance = "_".join(str(names[0]).lower().split())

        for position, stream in enumerate(_probe(path)):
            tags = {k.upper(): v for k, v in (stream.get("tags") or {}).items()}
            quantities = [q.strip() for q in str(tags.get("CHANNEL_TAGS", "")).split(",") if q]
            fs = float(stream.get("sample_rate") or 0)
            channels = int(stream.get("channels") or 0)
            if not quantities or not fs or channels != len(quantities):
                logger.warning("FIRED: %s stream %d has no usable tags", path.name, position)
                continue

            limit = int(self.max_seconds * fs) if self.max_seconds else None
            block = _decode(path, position, channels, limit)
            if block is None or block.shape[0] < 2:
                continue

            signals = {}
            for column, quantity in enumerate(quantities):
                if quantity not in ("v", "i", "p"):
                    continue  # `q` and `s` have no home in the store's schema
                values = block[:, column].astype(np.float32)
                if quantity == "i":
                    values = values / np.float32(MILLIAMPS_PER_AMP)
                    # No negation for entry["flip"] -- see the module docstring.
                    # The published signal is already the right way round, and
                    # flipping it again makes the appliance generate power.
                signals[quantity] = np.ascontiguousarray(values)
            if not signals:
                continue

            title = str(tags.get("TITLE", meter))
            phase = title.rsplit(" ", 1)[-1] if title.startswith(_AGGREGATE_PREFIX) else None
            yield Recording(
                dataset=self.dataset,
                house="apartment",
                # Every meter recorded on a day shares one clock, so the day ties
                # an aggregate window to the submeters that explain it.
                session=self._day_of(path),
                kind=ChannelKind.MAINS if is_aggregate else ChannelKind.SUBMETER,
                appliance=appliance,
                instance_id=None if is_aggregate else f"apartment:{appliance}",
                signals=signals,
                fs=fs,
                t0=float(tags.get("TIMESTAMP", 0.0) or 0.0),
                meta={
                    "meter": meter,
                    "phase": phase or entry.get("phase"),
                    "resolution": self.resolution,
                    "source": path.name,
                },
            )

    @staticmethod
    def _day_of(path: Path) -> str:
        """``powermeter08_2020_06_14__00_10_00.mkv`` to ``2020_06_14``."""
        parts = path.stem.split("_")
        return "_".join(parts[1:4]) if len(parts) >= 4 else path.stem
