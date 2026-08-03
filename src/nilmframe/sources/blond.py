"""Planning a BLOND subset.

BLOND is where fetching a subset stops being a convenience. Uncompressed it is
about 8.9 TB: one day of the full rig is 42 GB, a single five-minute mains file
is 118 MB, and there are 213 days of it. Nobody's first experiment needs any of
that, and the naive path -- download, then look -- costs a week.

It is also structured perfectly for it, in the same way UK-DALE is: every file's
start time is in its name, so a day, an hour or one unit resolves to a file list
from directory listings alone.

The transport is FTP rather than HTTP. mediaTUM's web front end is rate-limited
against crawlers; what it publishes for bulk access is a delivery server, with a
per-dataset account whose credentials are the dataset's own node id. That is the
channel this uses -- see :mod:`nilmframe.sources._ftp` -- and ``MLSD`` gives each
file's size in the listing, so a plan is costed without touching one.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

from nilmframe.readers.blond import parse_blond_name
from nilmframe.sources import registry
from nilmframe.sources._ftp import ftp_listdir
from nilmframe.sources.base import Artifact, Plan

__all__ = ["BLONDSource"]

logger = logging.getLogger(__name__)

RESOLUTIONS = ("BLOND-50", "BLOND-250")


def _days_in(time_range: tuple[float, float]) -> set[str]:
    """The ``YYYY-MM-DD`` directories a time range can touch.

    Widened by a day at each end: the directories are named for local dates while
    the range is in unix seconds, and one listing is cheaper than a missed hour.

    Example:
        >>> from nilmframe.sources.blond import _days_in
        >>> sorted(_days_in((1475233536, 1475237136)))
        ['2016-09-29', '2016-09-30', '2016-10-01']
    """
    start = datetime.fromtimestamp(time_range[0], tz=timezone.utc).date() - timedelta(days=1)
    stop = datetime.fromtimestamp(time_range[1], tz=timezone.utc).date() + timedelta(days=1)
    days, day = set(), start
    while day <= stop:
        days.add(day.isoformat())
        day += timedelta(days=1)
    return days


class BLONDSource:
    """Plan a BLOND download.

    Args:
        index_cache: directory for the FTP listings, so planning the same subset
            twice costs nothing.

    Example:
        >>> from nilmframe.sources import BLONDSource
        >>> plan = BLONDSource().plan(units=['clear'],        # doctest: +SKIP
        ...                           days=['2016-09-30'], max_files=1)
        >>> print(plan.summary())                             # doctest: +SKIP
    """

    name = "blond"
    dataset = "blond"

    def __init__(self, index_cache: str | Path | None = None) -> None:
        self.index_cache = Path(index_cache).expanduser() if index_cache else None

    # -- listings ----------------------------------------------------------- #

    def _url(self, *parts: str) -> str:
        creds = f"{registry.BLOND['user']}:{registry.BLOND['password']}"
        path = "/".join([registry.BLOND["root"].strip("/"), *(p.strip("/") for p in parts if p)])
        return f"ftp://{creds}@{registry.BLOND['host']}/{path}"

    def _listdir(self, *parts: str) -> list[tuple[str, int | None]]:
        key = "_".join(p.replace("/", "_") for p in parts) or "root"
        cached = self.index_cache / f"blond_{key}.json" if self.index_cache else None
        if cached is not None and cached.exists():
            return [(n, s) for n, s in json.loads(cached.read_text())]

        entries = ftp_listdir(self._url(*parts))
        if cached is not None:
            cached.parent.mkdir(parents=True, exist_ok=True)
            cached.write_text(json.dumps(entries))
        return entries

    def days(self, resolution: str = "BLOND-50") -> list[str]:
        """Every day directory the release carries."""
        return sorted(n for n, size in self._listdir(resolution) if size is None and n[:2] == "20")

    # -- the plan ----------------------------------------------------------- #

    def plan(
        self,
        *,
        resolution: str = "BLOND-50",
        units: list[str] | tuple[str, ...] = ("clear",),
        days: list[str] | None = None,
        time_range: tuple[float, float] | None = None,
        max_files: int | None = 1,
    ) -> Plan:
        """Work out exactly which files this configuration needs.

        Args:
            resolution: ``"BLOND-50"`` (213 days, 50 kHz mains) or
                ``"BLOND-250"`` (50 days, 250 kHz).
            units: ``"clear"`` for the three-phase mains, ``"medal-1"`` ..
                ``"medal-15"`` for the metered sockets. Fetching a MEDAL without
                ``clear`` gives appliances with no aggregate to disaggregate.
            days: ``YYYY-MM-DD`` strings. ``None`` uses ``time_range``; if that is
                also ``None``, the first day of the release.
            time_range: ``(start, stop)`` unix seconds, matched against the start
                time in each filename.
            max_files: files per unit per day. A CLEAR file is five minutes and
                118 MB; a whole day of one unit is 18 GB.

        Returns:
            A :class:`~nilmframe.sources.Plan` whose ``reader_kwargs`` point
            :class:`~nilmframe.readers.BLOND` at the cache.
        """
        if resolution not in RESOLUTIONS:
            raise ValueError(f"resolution must be one of {RESOLUTIONS}, got {resolution!r}")

        available = self.days(resolution)
        if days is not None:
            wanted = [d for d in days if d in available]
            missing = sorted(set(days) - set(available))
            if missing:
                logger.warning("BLOND: %s has no days %s", resolution, missing)
        elif time_range is not None:
            wanted = sorted(_days_in(time_range) & set(available))
        else:
            wanted = available[:1]
        if not wanted:
            raise ValueError(f"no BLOND days matched under {resolution}")

        artifacts: list[Artifact] = []
        # The log is 170 KB and the MEDAL sockets are unlabelled without it, so
        # it comes along whatever else was asked for.
        log_name = registry.BLOND["appliance_log"]
        artifacts.append(
            Artifact(url=self._url(log_name), relpath=log_name, size=self._log_size(log_name))
        )

        for day in wanted:
            present = {n for n, size in self._listdir(resolution, day) if size is None}
            for unit in units:
                if unit not in present:
                    logger.warning("BLOND: %s/%s has no %s", resolution, day, unit)
                    continue
                artifacts += self._unit_artifacts(resolution, day, unit, time_range, max_files)

        return Plan(
            dataset=self.dataset,
            artifacts=tuple(artifacts),
            reader_kwargs={"dirpath": resolution, "appliance_log": log_name},
            notes=(
                f"{registry.BLOND['home']} -- cite the Scientific Data paper",
                "one CLEAR file is 5 minutes of 3-phase mains at 50 kHz: "
                "15 million samples per phase",
            ),
        )

    def _log_size(self, name: str) -> int | None:
        for entry, size in self._listdir():
            if entry == name:
                return size
        return None

    def _unit_artifacts(
        self,
        resolution: str,
        day: str,
        unit: str,
        time_range: tuple[float, float] | None,
        max_files: int | None,
    ) -> list[Artifact]:
        rows = []
        for name, size in self._listdir(resolution, day, unit):
            if size is None or not name.endswith(".hdf5") or name.startswith("summary"):
                continue  # `summary-*.hdf5` is the 1 Hz roll-up, not a waveform
            info = parse_blond_name(name)
            if info is None:
                continue
            if time_range and not (time_range[0] <= info["t0"] <= time_range[1]):
                continue
            rows.append((info["t0"], name, size))

        rows.sort()
        if max_files:
            rows = rows[:max_files]
        return [
            Artifact(
                url=self._url(resolution, day, unit, name),
                relpath=f"{resolution}/{day}/{unit}/{name}",
                size=size,
            )
            for _, name, size in rows
        ]
