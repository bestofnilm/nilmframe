"""rsync, for the hosts that publish that way.

FIRED is 3.2 TB and its authors publish it over an rsync daemon, with the
password printed in the dataset's own README. That is the documented way in, so
it is the way this takes.

Unlike the HTTP and FTP transports here, this one shells out: the rsync daemon
protocol is not something to reimplement, and every system that would run this
already has the client or can install it. The cost is one hard requirement --
an ``rsync`` binary on ``PATH`` -- which is checked for up front so the failure
is a sentence rather than a traceback from ``subprocess``.

The password goes in the ``RSYNC_PASSWORD`` environment variable of the child
process rather than into the URL, so it never reaches a process listing.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from nilmframe.sources._http import FetchError

__all__ = ["RsyncLocation", "rsync_download", "rsync_listdir"]

logger = logging.getLogger(__name__)

TIMEOUT = 60

#: ``-rw-r--r--   7,461,916 2020/06/13 22:10:00 name.mkv``
_LINE = re.compile(
    r"^(?P<mode>[d\-l][rwx\-]{9})\s+(?P<size>[\d,]+)\s+"
    r"\d{4}/\d{2}/\d{2}\s+\d{2}:\d{2}:\d{2}\s+(?P<name>.+)$"
)


@dataclass(frozen=True, slots=True)
class RsyncLocation:
    """An ``rsync://user@host/module/path`` location and its password."""

    url: str
    password: str | None = None

    @property
    def env(self) -> dict[str, str]:
        env = dict(os.environ)
        if self.password:
            env["RSYNC_PASSWORD"] = self.password
        return env


def _require_binary() -> str:
    binary = shutil.which("rsync")
    if binary is None:
        raise FetchError(
            "this dataset is published over rsync and no `rsync` binary is on PATH. "
            "Install it (`apt install rsync`, `brew install rsync`) and try again."
        )
    return binary


def _run(location: RsyncLocation, args: list[str], *, timeout: int = TIMEOUT) -> str:
    binary = _require_binary()
    try:
        done = subprocess.run(
            [binary, f"--contimeout={TIMEOUT}", *args],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=location.env,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise FetchError(f"rsync timed out on {location.url}") from exc
    if done.returncode != 0:
        raise FetchError(f"rsync failed on {location.url}: {done.stderr.strip()[:300]}")
    return done.stdout


def rsync_listdir(location: RsyncLocation) -> list[tuple[str, int | None]]:
    """List a remote directory as ``(name, size)``, size ``None`` for directories.

    Args:
        location: the directory, with a trailing slash.

    Returns:
        Entries sorted by name, excluding ``.`` and ``..``.
    """
    out = _run(location, [location.url if location.url.endswith("/") else location.url + "/"])
    rows: list[tuple[str, int | None]] = []
    for line in out.splitlines():
        match = _LINE.match(line.strip())
        if match is None:
            continue
        name = match.group("name")
        if name in (".", ".."):
            continue
        is_dir = match.group("mode").startswith("d")
        rows.append((name, None if is_dir else int(match.group("size").replace(",", ""))))
    return sorted(rows)


def rsync_download(
    location: RsyncLocation,
    dest: Path,
    *,
    size: int | None = None,
    on_progress: Callable[[int], None] | None = None,
) -> int:
    """Fetch one file.

    rsync resumes by itself with ``--partial``, so an interrupted transfer costs
    only what it had not yet moved.

    Args:
        location: the remote file.
        dest: where it lands.
        size: expected length, checked afterwards when given.
        on_progress: called once with the byte count, after the transfer.

    Returns:
        Bytes written.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    _run(
        location,
        ["-a", "--partial", location.url, str(dest)],
        timeout=max(TIMEOUT, 3600),
    )
    if not dest.exists():
        raise FetchError(f"rsync reported success but {dest} is missing")
    written = dest.stat().st_size
    if size is not None and written != size:
        raise FetchError(f"{location.url}: expected {size} bytes, got {written}")
    if on_progress is not None:
        on_progress(written)
    return written
