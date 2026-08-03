"""HTTP, kept in one place.

This is the only module in the package that opens a socket. Everything above it
-- planners, the fetcher, the readers -- deals in :class:`~nilmframe.sources.Artifact`
objects and local paths, which is what makes the rest of the machinery testable
against a directory instead of the internet.

Two behaviours here are load-bearing rather than incidental:

**A range request that is not honoured is an error, not a slow path.** Asking for
64 KiB of a 3.6 GB archive and silently receiving all 3.6 GB is the exact failure
this package exists to avoid, so a response that is not ``206 Partial Content`` is
refused rather than read.

**Downloads resume.** A 200 MB waveform file over a research-council link fails
often enough that restarting from zero is a real cost, so a partial download is
kept and continued.
"""

from __future__ import annotations

import hashlib
import http.cookiejar
import json
import logging
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from nilmframe import __version__

__all__ = ["FetchError", "content_length", "download", "read_json", "read_range", "resolve"]

logger = logging.getLogger(__name__)

CHUNK = 1 << 20
TIMEOUT = 60.0
RETRIES = 4

_USER_AGENT = f"nilmframe/{__version__} (+https://github.com/arx7ti/nilmframe)"

# One opener for the process. Google Drive's download gate sets a cookie on the
# interstitial and expects it back on the download, so the jar has to outlive a
# single request. `http.cookiejar.CookieJar` takes its own lock, which is what
# makes this safe to share across the fetcher's worker threads.
_opener = urllib.request.build_opener(
    urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar())
)
_opener.addheaders = [("User-Agent", _USER_AGENT)]

_resolved: dict[str, str] = {}


class FetchError(RuntimeError):
    """A remote read failed, or succeeded in a way that cannot be used."""


def _open(url: str, headers: dict[str, str] | None = None, *, retries: int = RETRIES):
    """Open ``url``, retrying the failures that are worth retrying.

    A 4xx other than 429 means the request is wrong and will stay wrong, so it is
    raised immediately; everything else gets exponential backoff.
    """
    request = urllib.request.Request(url, headers=headers or {})
    last: Exception | None = None
    for attempt in range(retries):
        try:
            return _opener.open(request, timeout=TIMEOUT)
        except urllib.error.HTTPError as exc:
            if exc.code < 500 and exc.code != 429:
                raise FetchError(f"{exc.code} {exc.reason} for {url}") from exc
            last = exc
        except (urllib.error.URLError, ConnectionError, TimeoutError) as exc:
            last = exc
        delay = 2.0**attempt
        logger.warning(
            "fetch: %s (attempt %d/%d), retrying in %.0fs", last, attempt + 1, retries, delay
        )
        time.sleep(delay)
    raise FetchError(f"giving up on {url} after {retries} attempts: {last}")


def resolve(url: str) -> str:
    """The URL that actually serves bytes, following any download gate.

    Most hosts serve the URL you were given. Google Drive, where WHITED lives,
    answers large files with a virus-scan interstitial carrying a one-shot token
    in a hidden form; the real download is that form's target. The result is
    memoised, since the token is good for the rest of the session and the
    interstitial costs a round trip.

    Args:
        url: the published URL.

    Returns:
        A URL that responds with the file itself.
    """
    if "drive.google.com" not in url:
        return url
    if url in _resolved:
        return _resolved[url]

    with _open(url) as response:
        if "text/html" not in response.headers.get("content-type", ""):
            _resolved[url] = response.geturl()
            return _resolved[url]
        page = response.read().decode("utf-8", "replace")

    fields = dict(re.findall(r'name="([^"]+)"\s+value="([^"]*)"', page))
    if "uuid" not in fields or "id" not in fields:
        # Drive reports quota exhaustion and sharing changes as an ordinary page,
        # so surface what it said rather than a parse error against the HTML.
        message = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", page)).strip()[:300]
        raise FetchError(f"Google Drive did not offer a download for {url}: {message}")

    query = urllib.parse.urlencode(
        {"id": fields["id"], "export": "download", "confirm": "t", "uuid": fields["uuid"]}
    )
    _resolved[url] = f"https://drive.usercontent.google.com/download?{query}"
    return _resolved[url]


def content_length(url: str) -> int:
    """Size of the resource in bytes.

    Asks for one byte rather than issuing a HEAD: some hosts answer HEAD without
    a length, and a one-byte range answers the size and proves range support in
    the same round trip.
    """
    with _open(resolve(url), {"Range": "bytes=0-0"}) as response:
        if response.status == 206:
            total = response.headers.get("content-range", "").rpartition("/")[2]
            if total.isdigit():
                return int(total)
        length = response.headers.get("content-length")
        if response.status == 200 and length and length.isdigit():
            return int(length)
    raise FetchError(f"{url} did not report a size")


def read_range(url: str, start: int, end: int) -> bytes:
    """Bytes ``[start, end]`` inclusive, or an error.

    Args:
        url: resource to read from.
        start: first byte offset.
        end: last byte offset, inclusive, as HTTP counts them.

    Raises:
        FetchError: if the host ignored the range and offered the whole resource.
    """
    if end < start:
        return b""
    with _open(resolve(url), {"Range": f"bytes={start}-{end}"}) as response:
        if response.status != 206:
            raise FetchError(
                f"{url} does not support range requests (answered {response.status}); "
                "the whole archive would have to be downloaded"
            )
        return response.read()


def read_json(url: str) -> Any:
    """Fetch and parse a JSON document."""
    with _open(resolve(url)) as response:
        return json.loads(response.read().decode("utf-8"))


def stream(url: str, *, start: int = 0, length: int | None = None) -> Iterator[bytes]:
    """Yield a resource, or a byte span of it, in chunks.

    Args:
        url: resource to read.
        start: first byte to read.
        length: how many bytes to read; ``None`` reads to the end.
    """
    headers = {}
    if start or length is not None:
        end = "" if length is None else str(start + length - 1)
        headers["Range"] = f"bytes={start}-{end}"
    with _open(resolve(url), headers) as response:
        if headers and response.status != 206:
            raise FetchError(f"{url} does not support range requests (answered {response.status})")
        remaining = length
        while True:
            want = CHUNK if remaining is None else min(CHUNK, remaining)
            if want <= 0:
                return
            block = response.read(want)
            if not block:
                return
            if remaining is not None:
                remaining -= len(block)
            yield block


def download(
    url: str,
    dest: Path,
    *,
    size: int | None = None,
    md5: str | None = None,
    resume: bool = True,
    on_progress: Callable[[int], None] | None = None,
) -> int:
    """Fetch a whole resource to ``dest``, resuming a previous attempt.

    The download goes to ``dest.part`` and is renamed only once it is complete and
    verified, so an interrupted run never leaves a truncated file that looks
    finished to the next one.

    Args:
        url: what to fetch.
        dest: where it lands.
        size: expected length, checked after writing when given.
        md5: expected checksum, checked after writing when given.
        resume: continue an existing ``.part`` rather than restarting.
        on_progress: called with the byte count of each block written.

    Returns:
        Bytes written to ``dest``.

    Raises:
        FetchError: on a size or checksum mismatch.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    target = resolve(url)

    have = part.stat().st_size if resume and part.exists() else 0
    if size is not None and have >= size:
        have = 0  # a `.part` at or past the full length is a leftover, not progress
    digest = hashlib.md5() if md5 else None

    if have and digest is not None:
        with open(part, "rb") as fh:
            for block in iter(lambda: fh.read(CHUNK), b""):
                digest.update(block)

    headers = {"Range": f"bytes={have}-"} if have else {}
    with _open(target, headers) as response:
        if have and response.status != 206:
            # The host ignored the resume request and is sending from the top.
            have, digest = 0, hashlib.md5() if md5 else None
        mode = "ab" if have else "wb"
        with open(part, mode) as fh:
            while True:
                block = response.read(CHUNK)
                if not block:
                    break
                fh.write(block)
                if digest is not None:
                    digest.update(block)
                if on_progress is not None:
                    on_progress(len(block))

    written = part.stat().st_size
    if size is not None and written != size:
        raise FetchError(f"{url}: expected {size} bytes, got {written}")
    if digest is not None and digest.hexdigest() != md5:
        part.unlink(missing_ok=True)
        raise FetchError(f"{url}: MD5 mismatch, expected {md5}, got {digest.hexdigest()}")
    part.replace(dest)
    return written
