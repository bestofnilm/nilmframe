"""Turning a plan into a directory.

The cache is not an opaque blob store. It is laid out exactly as the reader's
documented on-disk layout, so ``<cache>/low_freq`` is a directory a reader will
accept, and somebody who already has UK-DALE by other means can point a reader at
their own copy and never call any of this. That constraint is deliberate: a cache
you cannot inspect with ``ls`` is a cache you cannot debug.

Alongside it sits ``nilmframe-cache.json``, recording for each file where it came
from and, where the host published one, its checksum. That is what makes a second
run cheap -- it skips what is already correct -- and it is what makes a store
built from the cache reproducible, since the provenance survives the download.
"""

from __future__ import annotations

import json
import logging
import shutil
import threading
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from nilmframe.sources._ftp import ftp_download
from nilmframe.sources._http import FetchError, download
from nilmframe.sources._rsync import RsyncLocation, rsync_download
from nilmframe.sources._zip import RemoteZip
from nilmframe.sources.base import Artifact, Plan, human_bytes

__all__ = ["FetchReport", "fetch", "materialize"]

logger = logging.getLogger(__name__)

MANIFEST = "nilmframe-cache.json"
MANIFEST_VERSION = 1


def _public_url(url: str) -> str:
    """A URL without its userinfo.

    FTP artifacts carry an account in the URL. Those credentials are published
    with the dataset rather than secret, but a cache manifest is a file in
    somebody's home directory and is the wrong place to copy them to.

    Example:
        >>> from nilmframe.sources.fetch import _public_url
        >>> _public_url('ftp://m1375836:m1375836@example.org/BLOND/a.hdf5')
        'ftp://example.org/BLOND/a.hdf5'
        >>> _public_url('https://example.org/a.zip')
        'https://example.org/a.zip'
    """
    scheme, _, rest = url.partition("://")
    if not rest or "@" not in rest.split("/", 1)[0]:
        return url
    return f"{scheme}://{rest.split('@', 1)[1]}"


def _leading_timestamp(line: bytes) -> float | None:
    """The first whitespace-separated field of a line, as a number."""
    fields = line.split(maxsplit=1)
    if not fields:
        return None
    try:
        return float(fields[0].decode("ascii"))
    except (ValueError, UnicodeDecodeError):
        return None


def _bounded(chunks: Iterator[bytes], until: float) -> Iterator[bytes]:
    """Stop a stream of meter readings once it passes ``until``.

    Meter files are sorted by time and span years, so a request for one day should
    not have to decompress the four years after it. Only whole lines are emitted:
    a half-written reading is worse than a missing one.

    Example:
        >>> from nilmframe.sources.fetch import _bounded
        >>> lines = [b'100 5.0\\n200 6.0\\n', b'300 7.0\\n400 8.0\\n']
        >>> b''.join(_bounded(iter(lines), until=250))
        b'100 5.0\\n200 6.0\\n'
    """
    tail = b""
    for chunk in chunks:
        buffer = tail + chunk
        cut = buffer.rfind(b"\n")
        if cut < 0:
            tail = buffer
            continue
        body, tail = buffer[: cut + 1], buffer[cut + 1 :]

        # Check the last complete line first: while the stream is still inside the
        # window -- which is most of it -- that is one parse per megabyte.
        last = body[body.rfind(b"\n", 0, cut) + 1 : cut]
        stamp = _leading_timestamp(last)
        if stamp is not None and stamp > until:
            keep = []
            for line in body.splitlines(keepends=True):
                value = _leading_timestamp(line)
                if value is not None and value > until:
                    break
                keep.append(line)
            if keep:
                yield b"".join(keep)
            return
        yield body
    if tail:
        yield tail


@dataclass(frozen=True, slots=True)
class FetchReport:
    """What a fetch did.

    Attributes:
        root: the cache directory the reader should be pointed at.
        fetched: paths written this run, relative to ``root``.
        skipped: paths already present and current.
        failed: ``(relpath, reason)`` for each artifact that did not land.
        nbytes: bytes actually transferred.
    """

    root: Path
    fetched: tuple[str, ...] = ()
    skipped: tuple[str, ...] = ()
    failed: tuple[tuple[str, str], ...] = ()
    nbytes: int = 0

    def __str__(self) -> str:
        parts = [f"{len(self.fetched)} fetched ({human_bytes(self.nbytes)})"]
        if self.skipped:
            parts.append(f"{len(self.skipped)} already present")
        if self.failed:
            parts.append(f"{len(self.failed)} failed")
        return f"{self.root}: " + ", ".join(parts)


class _Ledger:
    """The cache manifest: what is on disk, and where it came from."""

    def __init__(self, root: Path) -> None:
        self.path = root / MANIFEST
        self._lock = threading.Lock()
        self.entries: dict[str, dict] = {}
        if self.path.exists():
            try:
                blob = json.loads(self.path.read_text())
                self.entries = dict(blob.get("entries", {}))
            except (OSError, ValueError):
                logger.warning("cache manifest at %s is unreadable, rebuilding", self.path)

    def record(self, artifact: Artifact, size: int) -> None:
        with self._lock:
            self.entries[artifact.relpath] = {
                "url": _public_url(artifact.url),
                "member": artifact.member,
                "size": size,
                "md5": artifact.md5,
                "bounded_until": artifact.stop_after_timestamp,
            }

    def is_current(self, root: Path, artifact: Artifact) -> bool:
        """Whether the cached copy already satisfies this artifact."""
        record = self.entries.get(artifact.relpath)
        path = root / artifact.relpath
        if record is None or not path.exists():
            return False
        if record.get("url") != _public_url(artifact.url):
            return False
        if record.get("member") != artifact.member:
            return False
        if record.get("size") != path.stat().st_size:
            return False  # touched since it was written; re-fetch rather than guess
        if artifact.md5 is not None and record.get("md5") != artifact.md5:
            return False

        # A copy truncated at some earlier bound covers a narrower request, not a
        # wider one, and an untruncated copy covers everything.
        bound = record.get("bounded_until")
        if bound is None:
            return artifact.size is None or artifact.size == record["size"]
        if artifact.stop_after_timestamp is None:
            return False
        return bound >= artifact.stop_after_timestamp

    def flush(self) -> None:
        with self._lock:
            payload = {"version": MANIFEST_VERSION, "entries": self.entries}
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".json.part")
            tmp.write_text(json.dumps(payload, indent=1, sort_keys=True))
            tmp.replace(self.path)


def _write_stream(chunks: Iterator[bytes], dest: Path, *, until: float | None) -> int:
    """Write a member to ``dest`` atomically, optionally bounded in time."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    written = 0
    with open(part, "wb") as fh:
        for block in _bounded(chunks, until) if until is not None else chunks:
            fh.write(block)
            written += len(block)
    part.replace(dest)
    return written


def fetch(
    plan: Plan,
    cache: str | Path,
    *,
    dry_run: bool = False,
    workers: int = 4,
    force: bool = False,
    max_bytes: int | None = None,
    progress: Callable[[str], None] | None = None,
) -> FetchReport:
    """Materialise a plan into a cache directory.

    Args:
        plan: what to fetch, from a source's ``plan()``.
        cache: directory to fetch into. Created if absent.
        dry_run: report what would happen and touch nothing.
        workers: concurrent downloads. The hosts here are shared research
            infrastructure; the default is deliberately polite.
        force: re-fetch even what the manifest says is already current.
        max_bytes: refuse a plan larger than this, before transferring anything.
            A guard against a mistyped time range costing a terabyte.
        progress: called with one line per artifact as it completes.

    Returns:
        A :class:`FetchReport`.

    Raises:
        FetchError: if the plan exceeds ``max_bytes``, or if any artifact failed.

    Example:
        >>> from nilmframe.sources import UKDALESource, fetch
        >>> plan = UKDALESource().plan(houses=[1], max_hf_files=1)   # doctest: +SKIP
        >>> fetch(plan, "~/.cache/nilmframe/ukdale", dry_run=True)   # doctest: +SKIP
    """
    root = Path(cache).expanduser()
    if max_bytes is not None and plan.nbytes_max > max_bytes:
        raise FetchError(
            f"plan is up to {human_bytes(plan.nbytes_max)}, over the {human_bytes(max_bytes)} limit"
        )

    ledger = _Ledger(root)
    todo: list[Artifact] = []
    skipped: list[str] = []
    for artifact in plan.artifacts:
        if force or not ledger.is_current(root, artifact):
            todo.append(artifact)
        else:
            skipped.append(artifact.relpath)

    if dry_run:
        # What a real run would transfer, which is the number worth printing: the
        # plan's total counts files the cache already has.
        return FetchReport(
            root=root,
            fetched=tuple(a.relpath for a in todo),
            skipped=tuple(skipped),
            nbytes=sum(a.size or 0 for a in todo),
        )

    root.mkdir(parents=True, exist_ok=True)

    # One RemoteZip per archive: its directory is read once and shared by every
    # member pulled out of it, which is the difference between one footer request
    # and one per file.
    archives: dict[tuple[str, int], RemoteZip] = {}
    archive_lock = threading.Lock()
    emit_lock = threading.Lock()
    fetched: list[str] = []
    failed: list[tuple[str, str]] = []
    total = [0]

    def archive_for(artifact: Artifact) -> RemoteZip:
        key = (artifact.url, artifact.archive_offset)
        with archive_lock:
            if key not in archives:
                archive = RemoteZip(
                    artifact.url, size=artifact.archive_size, offset=artifact.archive_offset
                )
                # Read the directory while still holding the lock. WHITED's is
                # 1.5 MB; letting every worker discover it is missing at once
                # would fetch it once per worker.
                _ = archive.entries
                archives[key] = archive
            return archives[key]

    def run_sevenzip(group: list[Artifact]) -> None:
        """Every member of one 7z in a single pass.

        Solid blocks mean the cost is per block, not per file, and ``py7zr``
        decompresses each block it needs once per call -- so running these
        through the thread pool one at a time would inflate the same 3.8 GB block
        once for every file in it.
        """
        from nilmframe.sources._sevenzip import RemoteSevenZip

        archive = RemoteSevenZip(group[0].url, size=group[0].archive_size)
        staging = root / ".7z-staging"
        try:
            written = archive.extract([a.member for a in group], staging)
        except (FetchError, OSError, ImportError, ValueError) as exc:
            with emit_lock:
                failed.extend((a.relpath, str(exc)) for a in group)
                logger.error("7z extraction failed for %s: %s", group[0].url, exc)
            return

        for position, artifact in enumerate(group, 1):
            dest = root / artifact.relpath
            dest.parent.mkdir(parents=True, exist_ok=True)
            source = written[artifact.member]
            source.replace(dest)
            size = dest.stat().st_size
            ledger.record(artifact, size)
            with emit_lock:
                fetched.append(artifact.relpath)
                total[0] += size
                if progress is not None:
                    progress(
                        f"[{position:>4}/{len(group)}] {human_bytes(size):>9}  {artifact.relpath}"
                    )
        shutil.rmtree(staging, ignore_errors=True)

    def run(index: int, artifact: Artifact) -> None:
        dest = root / artifact.relpath
        try:
            if artifact.is_member:
                chunks = archive_for(artifact).stream(artifact.member)
                written = _write_stream(chunks, dest, until=artifact.stop_after_timestamp)
            elif artifact.url.startswith("ftp://"):
                written = ftp_download(artifact.url, dest, size=artifact.size)
            elif artifact.url.startswith("rsync://"):
                from nilmframe.sources import registry

                written = rsync_download(
                    RsyncLocation(artifact.url, registry.FIRED["password"]),
                    dest,
                    size=artifact.size,
                )
            else:
                written = download(artifact.url, dest, size=artifact.size, md5=artifact.md5)
        except (FetchError, OSError, KeyError, ValueError) as exc:
            with emit_lock:
                failed.append((artifact.relpath, str(exc)))
                logger.error("fetch failed for %s: %s", artifact.relpath, exc)
            return
        ledger.record(artifact, written)
        with emit_lock:
            fetched.append(artifact.relpath)
            total[0] += written
            if progress is not None:
                progress(
                    f"[{index:>4}/{len(independent)}] {human_bytes(written):>9}  {artifact.relpath}"
                )

    # 7z members are batched per archive; everything else is independent.
    batched: dict[str, list[Artifact]] = {}
    independent: list[Artifact] = []
    for artifact in todo:
        if artifact.is_member and artifact.archive_format == "7z":
            batched.setdefault(artifact.url, []).append(artifact)
        else:
            independent.append(artifact)

    try:
        for group in batched.values():
            run_sevenzip(group)
        if workers > 1 and len(independent) > 1:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                list(pool.map(lambda pair: run(*pair), enumerate(independent, 1)))
        else:
            for pair in enumerate(independent, 1):
                run(*pair)
    finally:
        ledger.flush()

    report = FetchReport(
        root=root,
        fetched=tuple(fetched),
        skipped=tuple(skipped),
        failed=tuple(failed),
        nbytes=total[0],
    )
    if failed:
        detail = "; ".join(f"{name}: {why}" for name, why in failed[:5])
        raise FetchError(f"{len(failed)} of {len(todo)} artifacts failed -- {detail}")
    return report


def materialize(
    plan: Plan, cache: str | Path, **kwargs
) -> tuple[dict[str, str | list[str]], FetchReport]:
    """Fetch a plan and resolve its reader arguments against the cache.

    Args:
        plan: what to fetch.
        cache: where to fetch it.
        **kwargs: forwarded to :func:`fetch`.

    Returns:
        ``(reader_kwargs, report)`` -- the first is ready to hand to the reader's
        constructor, with every cache-relative path made absolute.
    """
    report = fetch(plan, cache, **kwargs)
    resolved: dict[str, str | list[str]] = {}
    for key, value in plan.reader_kwargs.items():
        if isinstance(value, list):
            resolved[key] = [str(report.root / item) for item in value]
        else:
            resolved[key] = str(report.root / value)
    return resolved, report
