"""FTP, for the hosts that publish that way.

BLOND is 8.9 TB and mediaTUM does not serve it over the web interface -- that
front end is rate-limited against crawlers. What it does run is a delivery
server: an FTP host, with a per-dataset account whose credentials are the
dataset's own node id, published alongside the data precisely so that bulk
download does not go through the web interface. This module speaks to that.

FTP is a worse protocol than HTTP in most ways, but it has the two things that
matter here. ``MLSD`` lists a directory *with sizes*, so a plan can be costed
without touching a file. And ``REST`` resumes a transfer at a byte offset, which
on a 117 MB file over a research link is the difference between a retry and a
restart.

What it lacks is HTTP's connection model: every operation needs a live session,
and servers drop idle ones. So connections are pooled per thread rather than
per request, and a dropped one is reopened rather than raised.
"""

from __future__ import annotations

import ftplib
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from nilmframe.sources._http import FetchError

__all__ = ["FTPLocation", "ftp_download", "ftp_listdir"]

logger = logging.getLogger(__name__)

CHUNK = 1 << 20
TIMEOUT = 60.0
RETRIES = 3

_local = threading.local()


@dataclass(frozen=True, slots=True)
class FTPLocation:
    """Where an FTP resource lives, parsed from a URL.

    Attributes:
        host: server name or address.
        path: absolute path on the server.
        user: account name.
        password: account password.

    Example:
        >>> from nilmframe.sources._ftp import FTPLocation
        >>> loc = FTPLocation.parse('ftp://m1:pw@example.org/data/a.hdf5')
        >>> loc.host, loc.path, loc.user
        ('example.org', '/data/a.hdf5', 'm1')
    """

    host: str
    path: str
    user: str = "anonymous"
    password: str = "anonymous@"

    @classmethod
    def parse(cls, url: str) -> FTPLocation:
        parts = urlparse(url)
        if parts.scheme != "ftp":
            raise ValueError(f"not an ftp url: {url}")
        return cls(
            host=parts.hostname or "",
            path=parts.path,
            user=parts.username or "anonymous",
            password=parts.password or "anonymous@",
        )

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.host, self.user, self.password)


def _connect(location: FTPLocation) -> ftplib.FTP:
    session = ftplib.FTP()
    session.connect(location.host, 21, timeout=TIMEOUT)
    session.login(user=location.user, passwd=location.password)
    # SIZE and REST are refused in ASCII mode, and every file here is binary.
    session.voidcmd("TYPE I")
    return session


def _session(location: FTPLocation) -> ftplib.FTP:
    """A live session for this thread, reconnecting if the server dropped it.

    ``ftplib.FTP`` is not safe to share between threads -- a second command on
    one control channel interleaves with the first one's response -- so each
    worker keeps its own.
    """
    pool: dict = getattr(_local, "ftp", None) or {}
    _local.ftp = pool
    session = pool.get(location.key)
    if session is not None:
        try:
            session.voidcmd("NOOP")
            return session
        except (OSError, ftplib.Error):
            pool.pop(location.key, None)
    session = _connect(location)
    pool[location.key] = session
    return session


def _retrying(location: FTPLocation, action: Callable[[ftplib.FTP], object]) -> object:
    last: Exception | None = None
    for attempt in range(RETRIES):
        try:
            return action(_session(location))
        except ftplib.error_perm as exc:
            # A permanent error is the server saying no; asking again will not help.
            raise FetchError(f"{location.host}{location.path}: {exc}") from exc
        except (OSError, ftplib.Error) as exc:
            last = exc
            getattr(_local, "ftp", {}).pop(location.key, None)
            delay = 2.0**attempt
            logger.warning(
                "ftp: %s (attempt %d/%d), retrying in %.0fs", exc, attempt + 1, RETRIES, delay
            )
            time.sleep(delay)
    raise FetchError(f"giving up on ftp://{location.host}{location.path}: {last}")


def ftp_listdir(url: str) -> list[tuple[str, int | None]]:
    """List a directory as ``(name, size)``, size ``None`` for subdirectories.

    Uses ``MLSD``, which reports type and size in the listing itself, so costing
    a plan takes one command per directory rather than one per file.

    Args:
        url: ``ftp://user:password@host/path`` of the directory.

    Returns:
        Entries sorted by name.
    """
    location = FTPLocation.parse(url)

    def run(session: ftplib.FTP) -> list[tuple[str, int | None]]:
        session.cwd(location.path)
        out: list[tuple[str, int | None]] = []
        for name, facts in session.mlsd(facts=["type", "size"]):
            kind = facts.get("type")
            if kind == "file":
                size = facts.get("size")
                out.append((name, int(size) if size and size.isdigit() else None))
            elif kind == "dir" and name not in (".", ".."):
                out.append((name, None))
        return sorted(out)

    return _retrying(location, run)  # type: ignore[return-value]


def ftp_download(
    url: str,
    dest: Path,
    *,
    size: int | None = None,
    resume: bool = True,
    on_progress: Callable[[int], None] | None = None,
) -> int:
    """Fetch a file, resuming a previous attempt.

    Args:
        url: ``ftp://user:password@host/path`` of the file.
        dest: where it lands.
        size: expected length, checked after writing when given.
        resume: continue an existing ``.part`` with ``REST``.
        on_progress: called with the byte count of each block written.

    Returns:
        Bytes written to ``dest``.

    Raises:
        FetchError: on a size mismatch, or if the transfer keeps failing.
    """
    location = FTPLocation.parse(url)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")

    have = part.stat().st_size if resume and part.exists() else 0
    if size is not None and have >= size:
        have = 0  # a leftover at or past the full length is not progress

    def run(session: ftplib.FTP) -> int:
        nonlocal have
        with open(part, "ab" if have else "wb") as fh:

            def write(block: bytes) -> None:
                fh.write(block)
                if on_progress is not None:
                    on_progress(len(block))

            session.retrbinary(f"RETR {location.path}", write, blocksize=CHUNK, rest=have or None)
        return part.stat().st_size

    written = _retrying(location, run)
    if size is not None and written != size:
        raise FetchError(f"{url}: expected {size} bytes, got {written}")
    part.replace(dest)
    return int(written)  # type: ignore[arg-type]
