"""Reading members out of a remote ``.7z``.

A zip is a directory of independently compressed members, which is why any one of
them is reachable with two range requests. 7z is not: it groups members into
*solid blocks* and compresses each block as a single stream, so the first byte of
the fourth file in a block can only be reached by decompressing the three before
it. That is what makes 7z compress better and what makes it a poor fetch target.

It is not hopeless, only coarse. SmartNIALMeter's 10 GB archive is twelve blocks
rather than one, so a building costs its block -- roughly 0.85 GB transferred and
3.8 GB inflated -- instead of the whole release. The two rules that follow are
what this module exists to enforce:

* **Ask for everything at once.** ``py7zr`` decompresses each block it needs once
  per call, so one call with twenty targets is one pass; twenty calls with one
  target each is twenty passes over the same block. The fetcher therefore batches
  7z members by archive instead of running them through its thread pool.
* **Read the archive over ranges, not to disk.** The header is at the end and the
  block data is contiguous, so a seekable file backed by range requests lets
  ``py7zr`` pull only the blocks it needs.
"""

from __future__ import annotations

import io
import logging
from collections.abc import Callable
from pathlib import Path

from nilmframe.sources._http import FetchError, content_length, read_range

__all__ = ["RemoteSevenZip"]

logger = logging.getLogger(__name__)

BLOCK = 1 << 22


class _RangeFile(io.RawIOBase):
    """A seekable read-only file backed by HTTP range requests."""

    def __init__(self, url: str, size: int | None = None) -> None:
        self.url = url
        self._size = size if size is not None else content_length(url)
        self._pos = 0
        self.bytes_read = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            self._pos = offset
        elif whence == io.SEEK_CUR:
            self._pos += offset
        else:
            self._pos = self._size + offset
        return self._pos

    def readinto(self, buffer) -> int:  # type: ignore[override]
        want = min(len(buffer), self._size - self._pos)
        if want <= 0:
            return 0
        chunk = read_range(self.url, self._pos, self._pos + want - 1)
        buffer[: len(chunk)] = chunk
        self._pos += len(chunk)
        self.bytes_read += len(chunk)
        return len(chunk)


class RemoteSevenZip:
    """A ``.7z`` archive read over range requests.

    Args:
        url: the archive.
        size: its length, when already known.

    Example:
        >>> from nilmframe.sources import RemoteSevenZip
        >>> archive = RemoteSevenZip('https://example.org/raw.7z')   # doctest: +SKIP
        >>> archive.entries['raw/building_01/freezer.h5']            # doctest: +SKIP
        491874515
    """

    def __init__(self, url: str, *, size: int | None = None) -> None:
        self.url = url
        self.size = size
        self._entries: dict[str, int] | None = None

    @staticmethod
    def _py7zr():
        try:
            import py7zr
        except ImportError as exc:  # pragma: no cover - exercised by the extras matrix
            raise ImportError(
                "reading a .7z archive needs py7zr: pip install 'nilmframe[readers]'"
            ) from exc
        return py7zr

    def _open(self):
        handle = _RangeFile(self.url, self.size)
        self.size = handle._size
        return handle, self._py7zr().SevenZipFile(io.BufferedReader(handle, buffer_size=BLOCK))

    @property
    def entries(self) -> dict[str, int]:
        """Member name to uncompressed size, for the files only.

        Costs about a megabyte: 7z keeps its header at the end, and reading it
        does not touch any block.
        """
        if self._entries is None:
            handle, archive = self._open()
            with archive:
                self._entries = {
                    row.filename: int(row.uncompressed or 0)
                    for row in archive.list()
                    if not row.is_directory
                }
            logger.debug("7z header for %s cost %d bytes", self.url, handle.bytes_read)
        return self._entries

    def extract(
        self,
        members: list[str],
        dest: Path,
        *,
        on_file: Callable[[str], None] | None = None,
    ) -> dict[str, Path]:
        """Extract members in one pass, decompressing each block at most once.

        Args:
            members: member names to extract.
            dest: directory to extract into. Members keep their archive paths.
            on_file: called with each member name once it has landed.

        Returns:
            Member name to the path it was written to.

        Raises:
            FetchError: if a requested member is not in the archive.
        """
        known = self.entries
        missing = [m for m in members if m not in known]
        if missing:
            raise FetchError(f"{self.url} has no members {missing}")

        dest.mkdir(parents=True, exist_ok=True)
        _, archive = self._open()
        with archive:
            archive.extract(path=str(dest), targets=list(members))

        written = {}
        for name in members:
            path = dest / name
            if not path.exists():
                raise FetchError(f"{name} did not land under {dest}")
            written[name] = path
            if on_file is not None:
                on_file(name)
        return written
