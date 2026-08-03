"""Planning a UK-DALE subset.

UK-DALE is the dataset that makes a planner worth writing. Its 16 kHz mains
recording is published as one file per hour, roughly 200 MB each, several years
deep -- and the recording's start time is *in the filename*, which is the same
fact :meth:`UKDALE._timestamp_of <nilmframe.readers.UKDALE>` already relies on to
put a waveform and the meter readings on one clock. So "which hours do I need"
is answerable from names alone, and a day of waveform costs a day of waveform.

CEDA helps more than it has to: every directory answers ``?json`` with a listing
that carries each file's size *and* MD5. The plan therefore knows the bill before
a byte moves, and every download is checksummed on arrival.

The meter readings are the harder half, because they are published only as one
3.6 GB archive -- there are no per-channel files to link to. They are reached by
reading the archive's directory over a range request and pulling single members
out of it: one channel is about 50 MB compressed against 3.6 GB for the archive.
Bound the ingest with ``time_range`` and it drops again, because the extraction
stops decompressing once the readings pass the end of the window.

One coupling is easy to get wrong and expensive to discover later. A waveform
recording takes its session -- and therefore its access to the submeters that
label it -- from the low-frequency run that contains it. Fetch waveforms without
the meter channels covering the same hours and every window lands in ``hf_only``
with nothing to learn from. So asking for high-frequency data here implies the
meter channels over the same range unless you say otherwise.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from pathlib import Path

from nilmframe.readers.ukdale import UKDALE as _Reader
from nilmframe.sources import registry
from nilmframe.sources._http import FetchError, read_json
from nilmframe.sources._zip import RemoteZip
from nilmframe.sources.base import Artifact, Plan

__all__ = ["UKDALESource"]

logger = logging.getLogger(__name__)

_CHANNEL = re.compile(r"channel_(\d+)\.dat$")
_DAP = "https://dap.ceda.ac.uk"


def _iso_weeks(start: float, stop: float) -> set[tuple[int, int]]:
    """``(year, week)`` directories that can hold recordings in ``[start, stop]``.

    UK-DALE files waveforms under ``YYYY/wkNN`` where ``NN`` is the ISO week. A
    recording that starts minutes before a week boundary can be filed either side
    of it, so the range is widened by a week at each end -- listing one extra
    directory is cheaper than missing an hour.

    Example:
        >>> from nilmframe.sources.ukdale import _iso_weeks
        >>> sorted(_iso_weeks(1421784000, 1421870400))
        [(2015, 3), (2015, 4), (2015, 5)]
    """
    day = datetime.fromtimestamp(start, tz=timezone.utc).date() - timedelta(days=7)
    end = datetime.fromtimestamp(stop, tz=timezone.utc).date() + timedelta(days=7)
    weeks: set[tuple[int, int]] = set()
    while day <= end:
        iso = day.isocalendar()
        # Around New Year the calendar year and the ISO year disagree, and the
        # directory is named for one of them; take both rather than guess.
        weeks.add((day.year, iso[1]))
        weeks.add((iso[0], iso[1]))
        day += timedelta(days=7)
    return weeks


class UKDALESource:
    """Plan a UK-DALE download.

    Args:
        index_cache: directory for the CEDA directory listings. Planning the same
            subset twice then costs nothing, which is what makes ``--dry-run``
            worth iterating on.

    Example:
        >>> from nilmframe.sources import UKDALESource
        >>> source = UKDALESource()
        >>> plan = source.plan(houses=[1], channels=[1, 5],       # doctest: +SKIP
        ...                    time_range=(1421784000, 1421870400),
        ...                    max_hf_files=2)
        >>> print(plan.summary())                                 # doctest: +SKIP
    """

    name = "ukdale"
    dataset = "ukdale"

    def __init__(self, index_cache: str | Path | None = None) -> None:
        self.index_cache = Path(index_cache).expanduser() if index_cache else None
        self._archive: RemoteZip | None = None

    # -- listings ----------------------------------------------------------- #

    def _listing(self, url: str) -> list[dict]:
        """One CEDA directory, cached on disk when a cache was given."""
        cached: Path | None = None
        if self.index_cache is not None:
            key = url.replace(registry.UKDALE["high_freq"], "").strip("/") or "root"
            cached = self.index_cache / f"{key.replace('/', '_')}.json"
            if cached.exists():
                return json.loads(cached.read_text())["items"]

        payload = read_json(url + registry.UKDALE["listing_suffix"])
        if cached is not None:
            cached.parent.mkdir(parents=True, exist_ok=True)
            cached.write_text(json.dumps(payload))
        return payload["items"]

    def _subdirs(self, url: str) -> list[str]:
        return sorted(i["name"] for i in self._listing(url) if i["type"] == "dir")

    @property
    def archive(self) -> RemoteZip:
        """The meter-reading archive, with its directory read once."""
        if self._archive is None:
            self._archive = RemoteZip(registry.UKDALE["low_freq_zip"])
        return self._archive

    # -- the two halves ----------------------------------------------------- #

    def low_freq_artifacts(
        self,
        houses: list[int],
        *,
        channels: list[int] | None = None,
        time_range: tuple[float, float] | None = None,
        labels_only: bool = False,
    ) -> list[Artifact]:
        """Meter channels, as members of the published archive.

        Args:
            houses: house numbers.
            channels: channel numbers to take. ``None`` takes every channel the
                house has.
            time_range: bound extraction; readings past the end are never
                decompressed.
            labels_only: fetch just ``labels.dat``, which the reader requires
                even when no meter channel is wanted.
        """
        entries = self.archive.entries
        artifacts: list[Artifact] = []
        until = time_range[1] if time_range else None

        for house in houses:
            prefix = f"house_{house}/"
            names = [n for n in entries if n.startswith(prefix)]
            if not names:
                raise ValueError(
                    f"house_{house} is not in the UK-DALE archive; it has "
                    f"{sorted({n.split('/')[0] for n in entries})}"
                )

            wanted = [f"{prefix}labels.dat"]
            if not labels_only:
                for name in sorted(names):
                    match = _CHANNEL.search(name)
                    # `mains.dat` is the 1 Hz aggregate: 4.3 GB, and the reader
                    # touches it only as a fallback for a waveform start time that
                    # the published filenames already carry. Never worth fetching.
                    if match and (channels is None or int(match.group(1)) in channels):
                        wanted.append(name)

            for name in wanted:
                entry = entries.get(name)
                if entry is None:
                    logger.warning("UK-DALE: %s is not in the archive", name)
                    continue
                # Only the readings are time-ordered; truncating the label table
                # would throw away the appliance names.
                bounded = until if _CHANNEL.search(name) else None
                artifacts.append(
                    Artifact(
                        url=self.archive.url,
                        relpath=f"low_freq/{name}",
                        # A bounded extraction stops somewhere inside the member,
                        # and where cannot be known without reading it -- so the
                        # bill promises an upper bound and nothing tighter.
                        size=None if bounded else entry.size,
                        size_max=entry.size if bounded else None,
                        member=name,
                        archive_size=self.archive.size,
                        stop_after_timestamp=bounded,
                    )
                )
        return artifacts

    def high_freq_files(
        self,
        house: int,
        *,
        time_range: tuple[float, float] | None = None,
        limit: int | None = None,
    ) -> Iterator[dict]:
        """Waveform files for a house, oldest first, listing only what it must.

        With no ``time_range`` this walks years and weeks in order and stops as
        soon as ``limit`` files are in hand -- three listing requests for one hour
        of audio, rather than a crawl of the tree.
        """
        root = f"{registry.UKDALE['high_freq']}/house_{house}"
        weeks = _iso_weeks(*time_range) if time_range else None
        found = 0

        try:
            years = self._subdirs(root)
        except FetchError:
            # Only house 1 was recorded at 16 kHz. Asking for several houses at
            # once is the natural thing to do, and the ones without a waveform
            # tree should cost a line of warning rather than the whole plan.
            logger.warning("UK-DALE: house %d has no 16 kHz recordings", house)
            return

        for year in years:
            if not year.isdigit():
                continue
            if weeks is not None and not any(y == int(year) for y, _ in weeks):
                continue

            available = self._subdirs(f"{root}/{year}")
            candidates = available
            if weeks is not None:
                candidates = [
                    w for w in available if (int(year), _week_number(w)) in weeks
                ] or available  # the ISO guess found nothing: fall back to the year

            for week in candidates:
                items = [i for i in self._listing(f"{root}/{year}/{week}") if i["type"] == "file"]
                files = sorted(
                    (i for i in items if _timestamp_of(i["name"]) is not None),
                    key=lambda i: _timestamp_of(i["name"]),
                )
                for item in files:
                    stamp = _timestamp_of(item["name"])
                    if time_range and not (time_range[0] <= stamp <= time_range[1]):
                        continue
                    yield {
                        "url": item.get("download") or f"{_DAP}{item['path']}?download=1",
                        "relpath": f"high_freq/house_{house}/{year}/{week}/{item['name']}",
                        "size": item.get("size"),
                        "md5": item.get("md5"),
                    }
                    found += 1
                    if limit is not None and found >= limit:
                        return

    # -- the plan ----------------------------------------------------------- #

    def plan(
        self,
        *,
        houses: list[int] | tuple[int, ...] = (1,),
        channels: list[int] | None = None,
        time_range: tuple[float, float] | None = None,
        low_freq: bool = True,
        high_freq: bool = True,
        max_hf_files: int | None = 1,
    ) -> Plan:
        """Work out exactly which files this configuration needs.

        Args:
            houses: house numbers. Only house 1 carries the 16 kHz waveforms.
            channels: meter channel numbers. ``None`` takes all of them, which is
                about 2 GB for house 1; naming a few is usually what you want.
            time_range: ``(start, stop)`` unix seconds. Bounds both halves.
            low_freq: fetch the meter channels.
            high_freq: fetch the 16 kHz waveform files.
            max_hf_files: how many waveform files to take per house. Each is an
                hour and roughly 200 MB, so this is the knob that decides the
                bill. ``None`` takes every file in the range.

        Returns:
            A :class:`~nilmframe.sources.Plan` whose ``reader_kwargs`` point
            :class:`~nilmframe.readers.UKDALE` at the cache.
        """
        houses = list(houses)
        artifacts: list[Artifact] = []
        notes: list[str] = []

        if low_freq or high_freq:
            # `labels.dat` maps channel numbers to appliance names and the reader
            # refuses a house without it, so it comes along even in a waveform-only
            # fetch.
            artifacts += self.low_freq_artifacts(
                houses, channels=channels, time_range=time_range, labels_only=not low_freq
            )
        if not low_freq and high_freq:
            notes.append(
                "no meter channels requested: waveform windows will land in the "
                "'hf_only' session with no submeter targets, and the reader will "
                "warn once per channel named in labels.dat"
            )

        if high_freq:
            for house in houses:
                files = list(self.high_freq_files(house, time_range=time_range, limit=max_hf_files))
                if not files:
                    logger.warning("UK-DALE: no waveform files for house %d in range", house)
                artifacts += [
                    Artifact(url=f["url"], relpath=f["relpath"], size=f["size"], md5=f["md5"])
                    for f in files
                ]

        if time_range is not None and low_freq:
            notes.append(
                "meter channels are truncated at the end of the time range; "
                "widening it later re-fetches them"
            )

        return Plan(
            dataset=self.dataset,
            artifacts=tuple(artifacts),
            reader_kwargs={"dirpath": "low_freq", "high_freq_root": "high_freq"},
            notes=tuple(notes),
        )


def _week_number(name: str) -> int:
    """``wk04`` to ``4``; anything unrecognised sorts out of every range."""
    digits = "".join(c for c in name if c.isdigit())
    return int(digits) if digits else -1


def _timestamp_of(name: str) -> float | None:
    """Recording start encoded in a ``vi-<seconds>_<fraction>.flac`` name."""
    return _Reader._timestamp_of(Path(name))
