"""Planning the UCI household download.

The whole corpus is one 20 MB zip holding one 133 MB text file, so there is
nothing to plan: the only choice is whether to fetch it. It is here for the same
reason the others are -- so that getting it, verifying it and recording where it
came from work the same way as everything else, rather than being a wget in a
README.

The archive is fetched whole rather than read in place, and that is forced: the
UCI server answers with chunked transfer encoding, no ``Content-Length`` and no
``Accept-Ranges``, so there is no footer to seek to. At 20 MB it does not matter
-- the reader opens the member straight out of the zip, so nothing is unpacked to
disk either way.
"""

from __future__ import annotations

import logging
from pathlib import Path

from nilmframe.sources import registry
from nilmframe.sources.base import Artifact, Plan

__all__ = ["UCISource"]

logger = logging.getLogger(__name__)


class UCISource:
    """Plan the UCI household download.

    Args:
        url: the published archive, if not the pinned one.
        index_cache: accepted so every source takes it; unused here.

    Example:
        >>> from nilmframe.sources import UCISource
        >>> print(UCISource().plan().summary())      # doctest: +SKIP
    """

    name = "uci_household"
    dataset = "uci_household"

    def __init__(self, url: str | None = None, index_cache: str | Path | None = None) -> None:
        self.url = url or registry.UCI["url"]
        self.index_cache = Path(index_cache).expanduser() if index_cache else None

    def plan(self) -> Plan:
        """Work out what to fetch, which is the one data file.

        Returns:
            A :class:`~nilmframe.sources.Plan` whose ``reader_kwargs`` point
            :class:`~nilmframe.readers.UCIHousehold` at the extracted file.
        """
        name = registry.UCI["archive"]
        return Plan(
            dataset=self.dataset,
            artifacts=(Artifact(url=self.url, relpath=name, size=registry.UCI["size"]),),
            reader_kwargs={"path": name},
            notes=(
                registry.UCI["home"],
                "one house, one reading a minute, four years",
                "aggregate is in kilowatts and the submeters in watt-hours per "
                "minute; the reader converts both to watts",
            ),
        )
