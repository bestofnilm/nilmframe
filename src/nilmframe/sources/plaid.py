"""Planning a PLAID download.

PLAID is the well-behaved case and worth holding up as one: it lives on figshare
under a DOI, the article is versioned, and the API reports the archive's MD5. A
fetch that pins the article id is a fetch somebody else can reproduce.

It is also, unexpectedly, the most granular. The published archive stores its
members *uncompressed*, which means the waveform archive nested inside it is a
contiguous zip at a known offset -- so its 1793 per-recording CSVs are individually
addressable through two levels of range request. Reading ten recordings costs a
couple of megabytes instead of the 695 MB article, and the metadata JSON that says
what those recordings contain costs 580 KB on its own.
"""

from __future__ import annotations

import logging

from nilmframe.sources import registry
from nilmframe.sources._http import FetchError, read_json
from nilmframe.sources._zip import RemoteZip
from nilmframe.sources.base import Artifact, Plan

__all__ = ["PLAIDSource"]

logger = logging.getLogger(__name__)


class PLAIDSource:
    """Plan a PLAID download.

    Args:
        article: figshare article id. Pinned by default to the release the
            reader was written against; override it to take a different version.

    Example:
        >>> from nilmframe.sources import PLAIDSource
        >>> plan = PLAIDSource().plan(limit=5)          # doctest: +SKIP
        >>> print(plan.summary())                       # doctest: +SKIP
    """

    name = "plaid"
    dataset = "plaid"

    def __init__(self, article: int | None = None) -> None:
        self.article = article or registry.PLAID["article"]
        self._outer: RemoteZip | None = None
        self._inner: RemoteZip | None = None
        self._url: str | None = None

    def _open(self) -> tuple[RemoteZip, RemoteZip]:
        """The published archive and the waveform archive nested inside it."""
        if self._outer is None:
            record = self._record()
            self._url = record["download_url"]
            self._outer = RemoteZip(self._url, size=record.get("size"))
            self._inner = self._outer.nested(registry.PLAID["inner_archive"])
        return self._outer, self._inner

    def _record(self) -> dict:
        """The published file, from the API or from what the API used to say.

        figshare's metadata API refuses whole networks -- a datacenter address
        gets a bare 403 no matter what it asks -- while the download host answers
        the same request normally. Falling back to the pinned record keeps the
        fetch working there rather than failing on a metadata lookup.
        """
        try:
            meta = read_json(registry.PLAID["api"].format(article=self.article))
        except FetchError as exc:
            if self.article != registry.PLAID["article"]:
                raise  # a different article has no recorded answer to fall back to
            logger.warning(
                "PLAID: figshare's API is unreachable (%s); using the pinned record for article %s",
                exc,
                self.article,
            )
            return dict(registry.PLAID["fallback"])

        files = meta.get("files") or []
        if not files:
            raise FetchError(f"figshare article {self.article} publishes no files")
        return files[0]

    def plan(
        self,
        *,
        ids: list[int] | list[str] | None = None,
        limit: int | None = None,
    ) -> Plan:
        """Work out which recordings to fetch.

        Args:
            ids: recording ids to take. ``None`` takes every recording, which is
                the whole 693 MB of waveforms.
            limit: take at most this many, lowest id first. Useful for a smoke
                test that costs a few megabytes.

        Returns:
            A :class:`~nilmframe.sources.Plan` whose ``reader_kwargs`` point
            :class:`~nilmframe.readers.PLAID` at the cache.
        """
        outer, inner = self._open()
        url = self._url or ""
        metadata_members = list(registry.PLAID["metadata_members"])

        artifacts = [
            Artifact(
                url=url,
                relpath=member,
                size=outer.entries[member].size,
                member=member,
                archive_size=outer.size,
            )
            for member in metadata_members
            if member in outer.entries
        ]

        wanted = {str(i) for i in ids} if ids is not None else None
        recordings = []
        for name, entry in inner.entries.items():
            if not name.endswith(".csv"):
                continue
            recording_id = name.rsplit("/", 1)[-1][: -len(".csv")]
            if wanted is not None and recording_id not in wanted:
                continue
            recordings.append((recording_id, name, entry))
        recordings.sort(key=lambda item: (len(item[0]), item[0]))

        if wanted is not None:
            missing = wanted - {rid for rid, _, _ in recordings}
            if missing:
                logger.warning("PLAID: no waveform for ids %s", sorted(missing))
        if limit is not None:
            recordings = recordings[:limit]

        artifacts += [
            Artifact(
                url=url,
                relpath=f"csv/{recording_id}.csv",
                size=entry.size,
                member=name,
                archive_offset=inner.offset,
                archive_size=inner.size,
            )
            for recording_id, name, entry in recordings
        ]

        notes = [f"figshare article {self.article}, DOI {registry.PLAID['doi']}"]
        if limit is not None or ids is not None:
            notes.append(
                f"{len(recordings)} of {sum(1 for n in inner.entries if n.endswith('.csv'))} "
                "recordings; the metadata describes all of them, and the reader "
                "warns about the ones that are not on disk"
            )

        return Plan(
            dataset=self.dataset,
            artifacts=tuple(artifacts),
            reader_kwargs={
                "dirpath": "csv",
                # Both halves: they partition the release, so passing one reads
                # 60% of the recordings and warns about the rest.
                "metadata": [a.relpath for a in artifacts if a.relpath.endswith(".json")],
            },
            notes=tuple(notes),
        )
