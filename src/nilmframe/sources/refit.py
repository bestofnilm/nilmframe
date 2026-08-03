"""Planning a REFIT download.

REFIT publishes one CSV per house on Zenodo, which makes this the simplest source
here: the unit of laziness is a house, because a house is a file. There is no
finer slice -- the mains and all nine appliances share the file -- so the only
question is which homes.

The record carries six of the twenty houses. The full release lives in the
authors' institutional repository, which is figshare-backed and refuses some
hosts outright; the Zenodo copy is the one that fetches reliably.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from nilmframe.sources import registry
from nilmframe.sources.base import Artifact, Plan
from nilmframe.sources.zenodo import ZenodoRecord

__all__ = ["REFITSource"]

logger = logging.getLogger(__name__)

_HOUSE = re.compile(r"House(\d+)", re.I)


class REFITSource:
    """Plan a REFIT download.

    Args:
        record: Zenodo record id.
        index_cache: accepted so every source takes it; unused here, since the
            record's file list is one small API call.

    Example:
        >>> from nilmframe.sources import REFITSource
        >>> print(REFITSource().plan(houses=[1]).summary())   # doctest: +SKIP
    """

    name = "refit"
    dataset = "refit"

    def __init__(self, record: int | None = None, index_cache: str | Path | None = None) -> None:
        self.record = ZenodoRecord(record or registry.REFIT["record"])
        self.index_cache = Path(index_cache).expanduser() if index_cache else None

    def houses(self) -> dict[int, object]:
        """House number to the published file for it."""
        out = {}
        for row in self.record.files:
            match = _HOUSE.search(row.name)
            if match and row.name.lower().endswith(".csv"):
                out[int(match.group(1))] = row
        return out

    def plan(self, *, houses: list[int] | None = None) -> Plan:
        """Work out which houses to fetch.

        Args:
            houses: house numbers. ``None`` takes every house in the record,
                which is about 2.2 GB.

        Returns:
            A :class:`~nilmframe.sources.Plan` whose ``reader_kwargs`` point
            :class:`~nilmframe.readers.REFIT` at the cache.
        """
        available = self.houses()
        wanted = sorted(available if houses is None else [h for h in houses if h in available])
        missing = sorted(set(houses or []) - set(available))
        if missing:
            logger.warning(
                "REFIT: this record has no house %s; it publishes %s",
                missing,
                sorted(available),
            )
        if not wanted:
            raise ValueError(
                f"no REFIT houses matched {houses}; the record has {sorted(available)}"
            )

        return Plan(
            dataset=self.dataset,
            artifacts=tuple(
                Artifact(
                    url=available[house].url,
                    relpath=available[house].name,
                    size=available[house].size,
                    md5=available[house].md5,
                )
                for house in wanted
            ),
            reader_kwargs={"dirpath": "."},
            notes=(
                f"zenodo record {self.record.record}, DOI {self.record.doi}",
                f"houses {wanted}; one CSV holds a house's mains and all its appliances",
                "columns are named Appliance1..9; the reader maps them to real "
                "appliances from the dataset's own per-house metadata",
            ),
        )
