"""What a fetch plan is made of.

A :class:`Plan` is the answer to one question: *given this reader configuration,
which remote bytes do I actually need?* It is computed without downloading any
payload, which is the whole point -- a 4 TB corpus punishes you for finding out
the bill after the fact, so planning and fetching are separate steps and only the
second one costs anything.

An :class:`Artifact` is one remote thing landing at one local path. It comes in
two shapes, because the datasets do:

* a **whole file** at a URL, which is how UK-DALE publishes its hour-long 16 kHz
  waveform recordings;
* a **member of a remote archive**, which is how everything else is published --
  one multi-gigabyte zip that nobody wants in full. Since these hosts serve HTTP
  range requests, a member can be pulled out of the middle of an archive without
  the archive ever being downloaded, so this is not a lesser case of the first.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import PurePosixPath

__all__ = ["Artifact", "Plan", "human_bytes"]


def human_bytes(n: int) -> str:
    """Format a byte count the way a download prompt should read.

    Example:
        >>> from nilmframe.sources import human_bytes
        >>> human_bytes(0), human_bytes(1536), human_bytes(3_585_155_959)
        ('0 B', '1.5 KiB', '3.3 GiB')
    """
    units = ("B", "KiB", "MiB", "GiB", "TiB", "PiB")
    value, order = float(n), 0
    while value >= 1024 and order < len(units) - 1:
        value /= 1024
        order += 1
    return f"{n} B" if order == 0 else f"{value:.1f} {units[order]}"


@dataclass(frozen=True, slots=True)
class Artifact:
    """One remote thing that lands at one path under the cache.

    Args:
        url: where it lives. For an archive member this is the *archive's* URL.
        relpath: destination, relative to the cache root. Readers are pointed at
            the cache, so this is also the on-disk layout the reader expects.
        size: bytes it occupies once written -- for a member, its *uncompressed*
            size, which the archive's directory reports without decompressing it.
            ``None`` when the host does not say, which only makes the plan's total
            a lower bound.
        size_max: an upper bound, when the exact size is not knowable in advance.
            A time-bounded meter channel is the case: the archive says how large
            the whole channel is, and the extraction will stop somewhere inside
            it, so the plan can promise "no more than this" and nothing tighter.
        md5: checksum of the bytes as they land, when the host publishes one.
            Members carry ``None``: archives record a CRC of the member, not an
            MD5, so there is nothing to compare against.
        member: name inside the archive at ``url``. ``None`` fetches the whole
            resource.
        archive_format: ``"zip"`` or ``"7z"``. The distinction is not cosmetic: a
            zip member is reachable on its own, while a 7z member shares a
            compressed block with its neighbours and can only be reached by
            inflating all of them -- so the fetcher extracts 7z members in one
            batched pass rather than independently.
        archive_offset: where the archive begins inside the resource at ``url``.
            Non-zero for a zip nested inside another zip, which is how PLAID
            ships -- its outer archive stores the inner one uncompressed, so the
            inner archive is directly addressable at an offset.
        archive_size: length of that nested archive, needed to find its footer.
        stop_after_timestamp: for whitespace-separated ``<unix_time> <value>``
            meter files, stop extracting once a line's timestamp passes this.
            The whole low-frequency NILM family -- UK-DALE, REDD, ECO -- writes
            this format sorted by time, so a bounded ingest need not decompress
            the years it is going to discard.

    Example:
        >>> from nilmframe.sources import Artifact
        >>> art = Artifact(url='https://example.org/d.zip',
        ...                relpath='low_freq/house_1/labels.dat',
        ...                size=541, member='house_1/labels.dat')
        >>> art.is_member, art.size
        (True, 541)
    """

    url: str
    relpath: str
    size: int | None = None
    size_max: int | None = None
    md5: str | None = None
    member: str | None = None
    archive_format: str = "zip"
    archive_offset: int = 0
    archive_size: int | None = None
    stop_after_timestamp: float | None = None

    def __post_init__(self) -> None:
        # `relpath` is often derived from an archive member name, and an archive
        # is untrusted input: a member called `../../.ssh/authorized_keys` would
        # otherwise write outside the cache. Reject the escape here, once, rather
        # than trusting every planner to sanitise its own names.
        parts = PurePosixPath(self.relpath).parts
        if not parts or PurePosixPath(self.relpath).is_absolute() or ".." in parts:
            raise ValueError(f"relpath must stay inside the cache, got {self.relpath!r}")

    @property
    def is_member(self) -> bool:
        """Whether this is extracted from an archive rather than fetched whole."""
        return self.member is not None


@dataclass(frozen=True, slots=True)
class Plan:
    """Everything a dataset subset needs, and where the reader should look.

    Args:
        dataset: the name the reader and the CLI know this dataset by.
        artifacts: what to fetch.
        reader_kwargs: reader arguments naming *cache-relative* paths, resolved
            against the cache root once it is materialised. This is what lets a
            plan hand back a working reader without the reader learning anything
            about HTTP. A value may be a list where the reader takes several
            paths, as PLAID's annotations do.
        notes: things worth telling the user before they spend the bandwidth.

    Example:
        >>> from nilmframe.sources import Artifact, Plan
        >>> plan = Plan(dataset='demo', artifacts=(
        ...     Artifact('https://example.org/a.flac', 'high_freq/a.flac', size=2048),
        ...     Artifact('https://example.org/b.flac', 'high_freq/b.flac', size=1024),
        ... ))
        >>> len(plan), plan.nbytes
        (2, 3072)
        >>> print(plan.summary())
        demo: 2 files, 3.0 KiB
             2.0 KiB  high_freq/a.flac
             1.0 KiB  high_freq/b.flac
    """

    dataset: str
    artifacts: tuple[Artifact, ...] = ()
    reader_kwargs: dict[str, str | list[str]] = field(default_factory=dict)
    notes: tuple[str, ...] = ()

    def __len__(self) -> int:
        return len(self.artifacts)

    def __iter__(self):
        return iter(self.artifacts)

    @property
    def nbytes(self) -> int:
        """What the plan is known to transfer, counting an unknown size as zero.

        Sizes come from the host, so this is a lower bound; :attr:`nbytes_max` is
        the other end.
        """
        return sum(a.size or 0 for a in self.artifacts)

    @property
    def nbytes_max(self) -> int:
        """The most the plan can transfer -- the number worth confirming against."""
        return sum(a.size if a.size is not None else (a.size_max or 0) for a in self.artifacts)

    @property
    def n_unsized(self) -> int:
        """How many artifacts have no size at all, not even a bound."""
        return sum(1 for a in self.artifacts if a.size is None and a.size_max is None)

    def summary(self, limit: int = 12) -> str:
        """A printable bill: what would be downloaded, and how much of it.

        Args:
            limit: list at most this many artifacts before eliding the rest.
        """
        head = f"{self.dataset}: {len(self)} files, {human_bytes(self.nbytes)}"
        if self.nbytes_max > self.nbytes:
            head += f" (up to {human_bytes(self.nbytes_max)})"
        if self.n_unsized:
            head += f" (+{self.n_unsized} of unknown size)"
        lines = [head]
        for art in self.artifacts[:limit]:
            if art.size is not None:
                size = human_bytes(art.size)
            elif art.size_max is not None:
                size = f"≤{human_bytes(art.size_max)}"
            else:
                size = "?"
            lines.append(f"  {size:>10}  {art.relpath}")
        if len(self.artifacts) > limit:
            lines.append(f"  ... and {len(self.artifacts) - limit} more")
        lines.extend(f"  note: {note}" for note in self.notes)
        return "\n".join(lines)
