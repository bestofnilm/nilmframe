"""Zenodo records, which are the case everything else should look like.

A Zenodo record has a DOI, immutable versions, an API that reports each file's
size and MD5, and storage that honours range requests. That combination is what
makes a fetch reproducible: pinning a record id pins the bytes, and a checksum
mismatch is caught on arrival rather than in a training curve six hours later.

This module is the shared half. A dataset hosted here needs only to say which
record it is and how to pick members out of the archive inside it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from nilmframe.sources._http import FetchError, read_json
from nilmframe.sources._zip import RemoteZip

__all__ = ["ZenodoFile", "ZenodoRecord"]

logger = logging.getLogger(__name__)

API = "https://zenodo.org/api/records/{record}"


@dataclass(frozen=True, slots=True)
class ZenodoFile:
    """One file published in a record.

    Attributes:
        name: the filename as published.
        url: where to fetch it.
        size: bytes.
        md5: checksum, without Zenodo's ``md5:`` prefix.
    """

    name: str
    url: str
    size: int
    md5: str | None


class ZenodoRecord:
    """The files of one Zenodo record, and the archives inside them.

    Args:
        record: the numeric record id. Pin it: Zenodo mints a new one per
            version, so a fixed id is a fixed set of bytes.
        index_cache: directory to keep parsed archive directories in.

    Example:
        >>> from nilmframe.sources.zenodo import ZenodoRecord
        >>> record = ZenodoRecord(14886758)            # doctest: +SKIP
        >>> [f.name for f in record.files]             # doctest: +SKIP
        ['HIFDA_HF_electrical_signals_dataset.zip']
    """

    def __init__(self, record: int, index_cache: str | Path | None = None) -> None:
        self.record = int(record)
        self.index_cache = Path(index_cache).expanduser() if index_cache else None
        self._files: tuple[ZenodoFile, ...] | None = None

    @property
    def doi(self) -> str:
        return f"10.5281/zenodo.{self.record}"

    @property
    def files(self) -> tuple[ZenodoFile, ...]:
        """Every file in the record, from the API."""
        if self._files is None:
            payload = read_json(API.format(record=self.record))
            rows = payload.get("files") or []
            if not rows:
                raise FetchError(f"zenodo record {self.record} publishes no files")
            self._files = tuple(
                ZenodoFile(
                    name=row["key"],
                    url=row["links"]["self"],
                    size=int(row["size"]),
                    md5=str(row.get("checksum", "")).removeprefix("md5:") or None,
                )
                for row in rows
            )
        return self._files

    def file(self, name: str) -> ZenodoFile:
        """One file by name, or the only file when the record has just one."""
        rows = self.files
        for row in rows:
            if row.name == name:
                return row
        if len(rows) == 1:
            return rows[0]
        raise FetchError(
            f"zenodo record {self.record} has no file {name!r}; it has {[r.name for r in rows]}"
        )

    def archive(self, name: str) -> RemoteZip:
        """A published zip, read over range requests rather than downloaded."""
        row = self.file(name)
        return RemoteZip(row.url, size=row.size, index_cache=self.index_cache)
