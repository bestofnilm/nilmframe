"""Reading one member out of a remote zip, without downloading the zip.

Three of the four things this package fetches are published as a single large
archive. Nobody wants the whole of any of them: UK-DALE's meter archive is 3.6 GB
to reach one 50 MB channel, and WHITED is 2.1 GB to reach one appliance.

A zip is built for exactly this, though almost nothing uses it that way. Its
directory lives at the *end* of the file and records, for every member, where in
the file that member starts and how long it is. So two range requests -- one for
the footer, one for the member -- reach any file inside an archive of any size.
The hosts here all answer ``206 Partial Content``, which is what turns that from a
property of the format into a property of the download.

The awkward parts, all of which the real archives exercise:

* **ZIP64.** UK-DALE's archive holds a 4.29 GB member, and a classic zip header
  cannot express a size that large; the real value hides in an extra field. Read
  only the 32-bit header and you get ``0xFFFFFFFF`` bytes of nonsense.
* **The local header is not the directory entry.** A member's name and extra
  field are stored twice, at different lengths, so where its bytes begin can only
  be learned by reading the local header first.
* **Nesting.** PLAID publishes an archive whose members are *stored* rather than
  deflated -- so the waveform archive inside it is itself a contiguous zip at a
  known offset, and the same two range requests reach into it one level deeper.
"""

from __future__ import annotations

import hashlib
import json
import logging
import struct
import zlib
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from nilmframe.sources._http import FetchError, content_length, read_range, stream

__all__ = ["RemoteZip", "ZipEntry"]

logger = logging.getLogger(__name__)

_EOCD = b"PK\x05\x06"
_EOCD64 = b"PK\x06\x06"
_EOCD64_LOCATOR = b"PK\x06\x07"
_CENTRAL = b"PK\x01\x02"

# A zip comment can be 64 KiB, and the footer that follows it is 22 bytes; add
# the ZIP64 records that may sit in front of the classic one.
_FOOTER_SEARCH = (1 << 16) + 22 + 20 + 56

_SENTINEL32 = 0xFFFFFFFF


@dataclass(frozen=True, slots=True)
class ZipEntry:
    """One member of an archive, as its directory describes it.

    Attributes:
        name: path inside the archive.
        method: 0 for stored, 8 for deflate. Nothing else is supported.
        compressed_size: bytes occupied inside the archive.
        size: bytes once decompressed -- what it will occupy on disk.
        header_offset: where the member's local header begins.
        encrypted: whether the member needs a password.

    Example:
        >>> from nilmframe.sources._zip import ZipEntry
        >>> ZipEntry('house_1/labels.dat', 8, 200, 541, 1941527905).size
        541
    """

    name: str
    method: int
    compressed_size: int
    size: int
    header_offset: int
    encrypted: bool = False


def _zip64_fields(extra: bytes, wanted: int) -> list[int]:
    """The 8-byte values in a ZIP64 extended-information extra field.

    A directory entry stores ``0xFFFFFFFF`` in place of any 32-bit field that has
    overflowed, and the real values appear here in a fixed order -- uncompressed
    size, compressed size, local header offset -- but *only* for the fields that
    actually overflowed. So the caller says how many it is expecting.

    Example:
        >>> from nilmframe.sources._zip import _zip64_fields
        >>> import struct
        >>> blob = struct.pack('<HHQ', 1, 8, 4294967296)
        >>> _zip64_fields(blob, 1)
        [4294967296]
    """
    pos = 0
    while pos + 4 <= len(extra):
        header_id, length = struct.unpack("<HH", extra[pos : pos + 4])
        body = extra[pos + 4 : pos + 4 + length]
        if header_id == 0x0001:
            n = min(wanted, len(body) // 8)
            return list(struct.unpack(f"<{n}Q", body[: n * 8]))
        pos += 4 + length
    return []


class RemoteZip:
    """A zip archive read over HTTP range requests.

    Args:
        url: the archive, or the resource containing it.
        size: length of the archive. Fetched from the host when omitted.
        offset: where the archive starts inside the resource at ``url``, for an
            archive nested inside another one.

    Example:
        >>> from nilmframe.sources import RemoteZip
        >>> archive = RemoteZip('https://example.org/ukdale.zip')  # doctest: +SKIP
        >>> archive.entries['house_1/labels.dat'].size              # doctest: +SKIP
        541
    """

    def __init__(
        self,
        url: str,
        *,
        size: int | None = None,
        offset: int = 0,
        index_cache: str | Path | None = None,
    ) -> None:
        self.url = url
        self.offset = offset
        self._size = size
        self._entries: dict[str, ZipEntry] | None = None
        self.index_cache = Path(index_cache).expanduser() if index_cache else None

    @property
    def size(self) -> int:
        """Length of the archive in bytes."""
        if self._size is None:
            self._size = content_length(self.url) - self.offset
        return self._size

    @property
    def entries(self) -> dict[str, ZipEntry]:
        """Every member, keyed by name, in the order the directory lists them.

        Reading this costs two range requests and is cached for the archive's
        lifetime, so a planner can ask about thousands of members for the price of
        a footer.
        """
        if self._entries is None:
            self._entries = self._cached_directory()
        return self._entries

    def _cached_directory(self) -> dict[str, ZipEntry]:
        """Read the directory, keeping it on disk when a cache was given.

        Most archives here have a few hundred members and a footer costs nothing.
        One has 770,612, and its directory alone is 85 MB -- enough that
        re-reading it on every plan would undo the point of planning cheaply.
        """
        cached: Path | None = None
        if self.index_cache is not None:
            digest = hashlib.sha256(f"{self.url}|{self.offset}".encode()).hexdigest()[:16]
            cached = self.index_cache / f"zipindex-{digest}.json"
            if cached.exists():
                try:
                    rows = json.loads(cached.read_text())
                    return {
                        name: ZipEntry(name, *fields) for name, fields in rows["entries"].items()
                    }
                except (OSError, ValueError, TypeError):
                    logger.warning("zip index at %s is unreadable, re-reading", cached)

        entries = self._read_directory()
        if cached is not None:
            cached.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "url": self.url,
                "entries": {
                    e.name: [e.method, e.compressed_size, e.size, e.header_offset, e.encrypted]
                    for e in entries.values()
                },
            }
            cached.write_text(json.dumps(payload))
        return entries

    # -- internals ---------------------------------------------------------- #

    def _read(self, start: int, end: int) -> bytes:
        """Read ``[start, end]`` inclusive, in archive-relative coordinates."""
        return read_range(self.url, self.offset + start, self.offset + end)

    def _read_directory(self) -> dict[str, ZipEntry]:
        total = self.size
        tail_len = min(total, _FOOTER_SEARCH)
        tail = self._read(total - tail_len, total - 1)

        idx = tail.rfind(_EOCD)
        if idx < 0:
            raise FetchError(f"{self.url} is not a zip archive (no end-of-directory record)")
        count, cd_size, cd_offset = struct.unpack("<HII", tail[idx + 10 : idx + 20])

        # A ZIP64 archive keeps the classic record for compatibility and parks the
        # real numbers in its own, found through a locator just before it.
        locator = tail.rfind(_EOCD64_LOCATOR, 0, idx)
        if locator >= 0 and (count == 0xFFFF or cd_size == _SENTINEL32 or cd_offset == _SENTINEL32):
            (eocd64_offset,) = struct.unpack("<Q", tail[locator + 8 : locator + 16])
            head = tail.rfind(_EOCD64, 0, locator)
            record = (
                tail[head : head + 56]
                if head >= 0
                else self._read(eocd64_offset, eocd64_offset + 55)
            )
            # Layout: signature, record size, two versions, two disk numbers,
            # entries-on-this-disk, entries-total, directory size, directory
            # offset. The three that matter are the last three.
            count, cd_size, cd_offset = struct.unpack("<QQQ", record[32:56])

        directory = self._read(cd_offset, cd_offset + cd_size - 1)
        return self._parse_directory(directory, count)

    def _parse_directory(self, blob: bytes, count: int) -> dict[str, ZipEntry]:
        entries: dict[str, ZipEntry] = {}
        pos = 0
        while pos + 46 <= len(blob) and blob[pos : pos + 4] == _CENTRAL:
            flags, method = struct.unpack("<HH", blob[pos + 8 : pos + 12])
            compressed, uncompressed = struct.unpack("<II", blob[pos + 20 : pos + 28])
            name_len, extra_len, comment_len = struct.unpack("<HHH", blob[pos + 28 : pos + 34])
            (header_offset,) = struct.unpack("<I", blob[pos + 42 : pos + 46])
            name = blob[pos + 46 : pos + 46 + name_len].decode("utf-8", "replace")
            extra = blob[pos + 46 + name_len : pos + 46 + name_len + extra_len]

            # Order matters: only overflowed fields are present, in this sequence.
            overflowed = [
                uncompressed == _SENTINEL32,
                compressed == _SENTINEL32,
                header_offset == _SENTINEL32,
            ]
            if any(overflowed):
                values = iter(_zip64_fields(extra, sum(overflowed)))
                if overflowed[0]:
                    uncompressed = next(values, uncompressed)
                if overflowed[1]:
                    compressed = next(values, compressed)
                if overflowed[2]:
                    header_offset = next(values, header_offset)

            if not name.endswith("/"):
                entries[name] = ZipEntry(
                    name=name,
                    method=method,
                    compressed_size=compressed,
                    size=uncompressed,
                    header_offset=header_offset,
                    encrypted=bool(flags & 0x1),
                )
            pos += 46 + name_len + extra_len + comment_len
        if count and len(entries) > count:
            raise FetchError(f"{self.url}: directory says {count} members, parsed {len(entries)}")
        return entries

    def data_offset(self, name: str) -> int:
        """Where a member's bytes begin, archive-relative.

        The local header repeats the member's name and extra field at lengths that
        need not match the directory's, so this costs one small range request.
        """
        entry = self.entries[name]
        header = self._read(entry.header_offset, entry.header_offset + 29)
        name_len, extra_len = struct.unpack("<HH", header[26:30])
        return entry.header_offset + 30 + name_len + extra_len

    def stream(self, name: str) -> Iterator[bytes]:
        """Yield a member's decompressed bytes in chunks.

        Args:
            name: member to read.

        Raises:
            KeyError: if the archive has no such member.
            FetchError: if the member is encrypted, or uses a compression method
                other than stored or deflate.
        """
        entry = self.entries[name]
        if entry.encrypted:
            raise FetchError(
                f"{name} is encrypted; nilmframe cannot decrypt archive members. "
                "Download and unpack the archive yourself, then point the reader at it."
            )
        if entry.method not in (0, 8):
            raise FetchError(f"{name} uses unsupported compression method {entry.method}")

        start = self.offset + self.data_offset(name)
        raw = stream(self.url, start=start, length=entry.compressed_size)
        if entry.method == 0:
            yield from raw
            return

        decompressor = zlib.decompressobj(-zlib.MAX_WBITS)
        for block in raw:
            out = decompressor.decompress(block)
            if out:
                yield out
        out = decompressor.flush()
        if out:
            yield out

    def nested(self, name: str) -> RemoteZip:
        """An archive stored uncompressed inside this one, read in place.

        Args:
            name: the member holding the inner archive.

        Raises:
            FetchError: if that member is deflated, in which case its bytes are
                not contiguous plaintext and it has to be extracted first.
        """
        entry = self.entries[name]
        if entry.method != 0:
            raise FetchError(f"{name} is compressed, so it cannot be read in place")
        return RemoteZip(self.url, size=entry.size, offset=self.offset + self.data_offset(name))
