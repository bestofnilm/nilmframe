"""Planning a FIRED subset.

FIRED is 3.2 TB in full and its authors publish it over an rsync daemon, with
the password printed in the dataset's own README. Their download instructions are
three ``rsync`` commands that differ only in what they exclude, which is a fine
interface for "give me everything under 80 GB" and no help at all for "give me
one hour of one meter".

This does the same thing per file. The tree is ``<resolution>/<meter>/<name>.mkv``
with the date in every filename, so a day or a meter resolves from directory
listings, and each listing reports its files' sizes -- which is what lets a plan
be costed before anything moves.

The tiers are worth knowing before choosing one:

* ``1Hz`` -- power summaries of everything, about 800 KB per meter per day;
* ``50Hz`` -- the same, at fifty times the rate;
* ``highFreq`` -- the waveforms: 7.6 MB per plug meter per ten minutes, and
  78 MB per ten minutes of the three-phase smart meter.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from nilmframe.readers.fired import RESOLUTIONS
from nilmframe.sources import registry
from nilmframe.sources._rsync import RsyncLocation, rsync_listdir
from nilmframe.sources.base import Artifact, Plan

__all__ = ["FIREDSource"]

logger = logging.getLogger(__name__)

#: ``powermeter08_2020_06_14__00_10_00.mkv``
_DAY = re.compile(r"_(\d{4}_\d{2}_\d{2})__")


class FIREDSource:
    """Plan a FIRED download.

    Args:
        index_cache: directory for the rsync listings, so planning the same
            subset twice costs nothing.

    Example:
        >>> from nilmframe.sources import FIREDSource
        >>> plan = FIREDSource().plan(resolution="1Hz")   # doctest: +SKIP
        >>> print(plan.summary())                          # doctest: +SKIP
    """

    name = "fired"
    dataset = "fired"

    def __init__(self, index_cache: str | Path | None = None) -> None:
        self.index_cache = Path(index_cache).expanduser() if index_cache else None

    # -- listings ----------------------------------------------------------- #

    def _location(self, *parts: str) -> RsyncLocation:
        path = "/".join(p.strip("/") for p in parts if p)
        base = registry.FIRED["url"].rstrip("/")
        return RsyncLocation(
            url=f"{base}/{path}" if path else base + "/",
            password=registry.FIRED["password"],
        )

    def _listdir(self, *parts: str) -> list[tuple[str, int | None]]:
        key = "_".join(p.replace("/", "_") for p in parts) or "root"
        cached = self.index_cache / f"fired_{key}.json" if self.index_cache else None
        if cached is not None and cached.exists():
            return [(n, s) for n, s in json.loads(cached.read_text())]

        entries = rsync_listdir(self._location(*parts))
        if cached is not None:
            cached.parent.mkdir(parents=True, exist_ok=True)
            cached.write_text(json.dumps(entries))
        return entries

    def _subtree(self, resolution: str) -> str:
        return "highFreq" if resolution == "highFreq" else f"summary/{resolution}"

    def meters(self, resolution: str = "1Hz") -> list[str]:
        """Meter names published at this resolution."""
        return sorted(
            name
            for name, size in self._listdir(self._subtree(resolution))
            if size is None and not name.startswith(".")
        )

    # -- the plan ----------------------------------------------------------- #

    def plan(
        self,
        *,
        resolution: str = "1Hz",
        meters: list[str] | None = None,
        days: list[str] | None = None,
        max_files: int | None = 1,
        info: bool = True,
    ) -> Plan:
        """Work out exactly which files this configuration needs.

        Args:
            resolution: ``"1Hz"``, ``"50Hz"`` or ``"highFreq"``.
            meters: meter names, e.g. ``["smartmeter001", "powermeter08"]``.
                ``None`` takes every meter at that resolution.
            days: ``YYYY_MM_DD`` strings as they appear in the filenames.
            max_files: files per meter. Ten minutes of the smart meter's waveform
                is 78 MB, so this is the knob that decides the bill.
            info: include ``deviceMapping.json`` and ``deviceInfo.json``. Without
                the mapping the plug meters have no appliance names, and two of
                them have reversed polarity that only the mapping records.

        Returns:
            A :class:`~nilmframe.sources.Plan` whose ``reader_kwargs`` point
            :class:`~nilmframe.readers.FIRED` at the cache.
        """
        if resolution not in RESOLUTIONS:
            raise ValueError(f"resolution must be one of {RESOLUTIONS}, got {resolution!r}")

        subtree = self._subtree(resolution)
        available = self.meters(resolution)
        wanted = [m for m in (meters or available) if m in available]
        missing = sorted(set(meters or []) - set(available))
        if missing:
            logger.warning("FIRED: %s has no meters %s", resolution, missing)
        if not wanted:
            raise ValueError(f"no FIRED meters matched {meters}; {resolution} has {available}")

        artifacts: list[Artifact] = []
        if info:
            for name, size in self._listdir("info"):
                if name.endswith(".json"):
                    artifacts.append(
                        Artifact(
                            url=self._location("info", name).url,
                            relpath=f"info/{name}",
                            size=size,
                        )
                    )

        for meter in wanted:
            rows = []
            for name, size in self._listdir(subtree, meter):
                if size is None or not name.endswith(".mkv"):
                    continue
                match = _DAY.search(name)
                if days is not None and (match is None or match.group(1) not in days):
                    continue
                rows.append((name, size))
            rows.sort()
            if max_files:
                rows = rows[:max_files]
            artifacts += [
                Artifact(
                    url=self._location(subtree, meter, name).url,
                    relpath=f"{subtree}/{meter}/{name}",
                    size=size,
                )
                for name, size in rows
            ]

        return Plan(
            dataset=self.dataset,
            artifacts=tuple(artifacts),
            reader_kwargs={"dirpath": "."},
            notes=(
                f"{registry.FIRED['home']} -- cite the BuildSys '20 paper",
                "stored as WavPack in Matroska: reading it needs ffmpeg on PATH",
                "waveform current is in milliamperes; the reader scales it",
            ),
        )
