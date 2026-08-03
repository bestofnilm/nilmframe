"""Planning a HIFDA subset.

HIFDA publishes 770,612 files inside one 4.38 GB archive on Zenodo: the same
recordings windowed four ways, from 10.24 ms slices to whole activations. Ninety
per cent of that is the shortest window, which almost nobody wants.

Because Zenodo honours range requests, choosing is cheap. Reading the archive's
directory is the one real cost -- 770,612 entries is 85 MB of footer -- so it is
kept on disk, and every plan after the first is free. Four recordings of a
microwave then cost about 45 MB rather than 4.38 GB.
"""

from __future__ import annotations

import logging
from pathlib import Path

from nilmframe.sources import registry
from nilmframe.sources.base import Artifact, Plan
from nilmframe.sources.zenodo import ZenodoRecord

__all__ = ["WINDOWS", "HIFDASource"]

logger = logging.getLogger(__name__)

#: The four windowings the release ships, shortest first.
WINDOWS = ("10.24ms", "163.84ms", "1310.72ms", "Full_time")


class HIFDASource:
    """Plan a HIFDA download.

    Args:
        record: Zenodo record id. Pinned by default to the release the reader was
            written against.
        index_cache: directory for the archive's parsed directory.

    Example:
        >>> from nilmframe.sources import HIFDASource
        >>> plan = HIFDASource().plan(appliances=['Microwave'],  # doctest: +SKIP
        ...                           limit=4)
        >>> print(plan.summary())                                # doctest: +SKIP
    """

    name = "hifda"
    dataset = "hifda"

    def __init__(self, record: int | None = None, index_cache: str | Path | None = None) -> None:
        self.record = ZenodoRecord(record or registry.HIFDA["record"], index_cache=index_cache)

    def plan(
        self,
        *,
        window: str = "Full_time",
        appliances: list[str] | None = None,
        limit: int | None = None,
    ) -> Plan:
        """Work out which recordings to fetch.

        Args:
            window: one of :data:`WINDOWS`. ``"Full_time"`` is 750 whole
                recordings; ``"10.24ms"`` is 358,740 slices of them.
            appliances: appliance directory names, e.g. ``["Microwave"]``.
                ``None`` takes all fifteen classes.
            limit: take at most this many recordings, in name order.

        Returns:
            A :class:`~nilmframe.sources.Plan` whose ``reader_kwargs`` point
            :class:`~nilmframe.readers.HIFDA` at the cache.
        """
        if window not in WINDOWS:
            raise ValueError(f"window must be one of {WINDOWS}, got {window!r}")

        archive = self.record.archive(registry.HIFDA["archive"])
        entries = archive.entries
        group = f"/{window}_window_dataset/"
        wanted = {a.lower() for a in appliances} if appliances else None

        # Current and voltage live in parallel trees under the same name, and a
        # recording is only usable as a pair -- so selection walks the current
        # side and requires the matching voltage file to exist.
        pairs: list[tuple[str, str, str]] = []
        for name in entries:
            if group not in name or "/Current/" not in name or not name.endswith("Current.txt"):
                continue
            appliance = name.split("/")[-2]
            if wanted is not None and appliance.lower() not in wanted:
                continue
            partner = name.replace("/Current/", "/Voltage/")
            partner = partner[: -len("Current.txt")] + "Voltage.txt"
            if partner not in entries:
                logger.warning("HIFDA: %s has no matching voltage member", name)
                continue
            pairs.append((name.split("/")[-1], name, partner))

        pairs.sort()
        if limit:
            pairs = pairs[:limit]
        if not pairs and appliances:
            classes = sorted({n.split("/")[-2] for n in entries if group in n and "/Current/" in n})
            raise ValueError(f"no HIFDA recordings matched {appliances}; it has {classes}")

        artifacts: list[Artifact] = []
        for _, current, voltage in pairs:
            for member in (current, voltage):
                # Drop the archive's own top directory and the window group, so
                # the cache is the window directory the reader expects.
                relpath = f"{window}_window_dataset/" + member.split(group, 1)[1]
                artifacts.append(
                    Artifact(
                        url=archive.url,
                        relpath=relpath,
                        size=entries[member].size,
                        member=member,
                        archive_size=archive.size,
                    )
                )

        return Plan(
            dataset=self.dataset,
            artifacts=tuple(artifacts),
            reader_kwargs={"dirpath": f"{window}_window_dataset"},
            notes=(
                f"zenodo record {self.record.record}, DOI {self.record.doi}",
                f"{len(pairs)} recordings at 100 kSPS, current and voltage as separate files",
            ),
        )
