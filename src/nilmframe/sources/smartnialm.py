"""Planning a SmartNIALMeter subset.

Two Zenodo files, ``raw.7z`` at 10.15 GB and ``preprocessed.7z`` at 16.98 GB,
holding 120 HDF5 files between twenty buildings. Everything good about Zenodo
applies -- DOI, versions, checksums, range requests -- and then the archive
format takes most of it back.

7z compresses members in solid blocks. This release is twelve of them, holding
4 to 26 files each, so the unit of laziness is a block and not a file: one
building's aggregate costs the roughly 0.85 GB block it sits in. That is still
twelve times better than the whole release, and it is the reason to ask for a
whole building at once rather than a file at a time -- the second file in a block
is free once the first has paid for it.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from nilmframe.readers.smartnialm import AGGREGATE_FILE
from nilmframe.sources import registry
from nilmframe.sources._sevenzip import RemoteSevenZip
from nilmframe.sources.base import Artifact, Plan
from nilmframe.sources.zenodo import ZenodoRecord

__all__ = ["VERSIONS", "SmartNIALMSource"]

logger = logging.getLogger(__name__)

#: The two published curations of the same measurements.
VERSIONS = ("raw", "preprocessed")

_BUILDING = re.compile(r"building_(\d+)")


class SmartNIALMSource:
    """Plan a SmartNIALMeter download.

    Args:
        record: Zenodo record id.
        index_cache: unused for now; accepted so every source takes it.

    Example:
        >>> from nilmframe.sources import SmartNIALMSource
        >>> plan = SmartNIALMSource().plan(buildings=[1])   # doctest: +SKIP
        >>> print(plan.summary())                           # doctest: +SKIP
    """

    name = "smartnialm"
    dataset = "smartnialm"

    def __init__(self, record: int | None = None, index_cache: str | Path | None = None) -> None:
        self.record = ZenodoRecord(record or registry.SMARTNIALM["record"])
        self.index_cache = Path(index_cache).expanduser() if index_cache else None
        self._archives: dict[str, RemoteSevenZip] = {}

    def archive(self, version: str) -> RemoteSevenZip:
        """The published ``.7z`` for one curation."""
        if version not in VERSIONS:
            raise ValueError(f"version must be one of {VERSIONS}, got {version!r}")
        if version not in self._archives:
            row = self.record.file(f"{version}.7z")
            self._archives[version] = RemoteSevenZip(row.url, size=row.size)
        return self._archives[version]

    def plan(
        self,
        *,
        version: str = "raw",
        buildings: list[int] | None = None,
        appliances: list[str] | None = None,
        aggregate: bool = True,
    ) -> Plan:
        """Work out which files this configuration needs.

        Args:
            version: ``"raw"`` or ``"preprocessed"``.
            buildings: building numbers. ``None`` takes building 1 -- the whole
                release is 45 GB unpacked, so there is no useful "everything".
            appliances: appliance file stems, e.g. ``["freezer"]``. ``None``
                takes every appliance in the chosen buildings.
            aggregate: include each building's smart meter. Leaving it out gives
                appliances with nothing to disaggregate from.

        Returns:
            A :class:`~nilmframe.sources.Plan` whose ``reader_kwargs`` point
            :class:`~nilmframe.readers.SmartNIALM` at the cache.
        """
        archive = self.archive(version)
        entries = archive.entries
        wanted_buildings = set(buildings) if buildings else {1}
        wanted_appliances = {a.lower() for a in appliances} if appliances else None

        artifacts: list[Artifact] = []
        for name, size in sorted(entries.items()):
            match = _BUILDING.search(name)
            if match is None or int(match.group(1)) not in wanted_buildings:
                continue
            stem = name.rsplit("/", 1)[-1].removesuffix(".h5").lower()
            if stem == AGGREGATE_FILE:
                if not aggregate:
                    continue
            elif wanted_appliances is not None and stem not in wanted_appliances:
                continue
            artifacts.append(
                Artifact(
                    url=archive.url,
                    # Drop the leading `raw/` or `preprocessed/`: the cache is
                    # the version directory the reader is pointed at.
                    relpath=name.split("/", 1)[1] if "/" in name else name,
                    size=size,
                    member=name,
                    archive_format="7z",
                    archive_size=archive.size,
                )
            )

        if not artifacts:
            found = sorted({int(m.group(1)) for m in map(_BUILDING.search, entries) if m})
            raise ValueError(
                f"no SmartNIALMeter files matched buildings={sorted(wanted_buildings)}"
                f" appliances={appliances}; the release has buildings {found}"
            )

        return Plan(
            dataset=self.dataset,
            artifacts=tuple(artifacts),
            reader_kwargs={"dirpath": "."},
            notes=(
                f"zenodo record {self.record.record}, DOI {self.record.doi}",
                "5-second power, not waveforms: no cycle alignment, no V-I trajectory",
                "extraction decompresses whole solid blocks, so asking for one "
                "building at once costs far less per file than one file at a time",
            ),
        )
