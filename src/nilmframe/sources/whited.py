"""Planning a WHITED download.

WHITED is one 2.1 GB archive on Google Drive, which sounds like the case where
laziness is impossible -- and would be, except that Drive honours range requests.
So the archive's directory is readable without the archive, and since every
recording's appliance, brand, region and measurement kit are encoded in its
filename, a subset can be chosen from names alone: one appliance is a few
megabytes.

Two things are worth knowing before using this.

**Drive gates large files.** It answers with a virus-scan interstitial carrying a
one-shot token rather than the file; :func:`~nilmframe.sources._http.resolve`
answers it. Drive also enforces its own download quotas, and when it refuses,
it refuses with a web page -- so the error you get here is Drive's own words.

**The archive is not encrypted, despite the password on the distribution page.**
The published archive's members carry no encryption flag. If that changes,
nothing here will decrypt them: the fetch fails with a pointer to the page where
the authors publish both the password and the citation they ask for, which is
what that gate is there to make you read.
"""

from __future__ import annotations

import logging

from nilmframe.readers.whited import CALIBRATION
from nilmframe.sources import registry
from nilmframe.sources._http import FetchError
from nilmframe.sources._zip import RemoteZip
from nilmframe.sources.base import Artifact, Plan

__all__ = ["WHITEDSource"]

logger = logging.getLogger(__name__)


class WHITEDSource:
    """Plan a WHITED download.

    Example:
        >>> from nilmframe.sources import WHITEDSource
        >>> plan = WHITEDSource().plan(appliances=['Kettle'])   # doctest: +SKIP
        >>> print(plan.summary())                               # doctest: +SKIP
    """

    name = "whited"
    dataset = "whited"

    def __init__(self, url: str | None = None) -> None:
        self.url = url or registry.WHITED["url"]
        self._archive: RemoteZip | None = None

    @property
    def archive(self) -> RemoteZip:
        if self._archive is None:
            self._archive = RemoteZip(self.url)
        return self._archive

    def plan(
        self,
        *,
        appliances: list[str] | None = None,
        kits: list[str] | None = None,
        regions: list[str] | None = None,
        limit: int | None = None,
    ) -> Plan:
        """Work out which recordings to fetch.

        Filters match the fields WHITED encodes in each filename,
        ``{appliance}_{model}_{region}_{kit}_{timestamp}.flac``, and are compared
        case-insensitively.

        Args:
            appliances: appliance names to take, e.g. ``["Kettle", "Fan"]``.
            regions: region codes, e.g. ``["r1"]``.
            kits: measurement kits. Defaults to the kits the reader has
                calibration factors for -- there is no point spending bandwidth
                on recordings the reader will skip for want of a scale factor.
            limit: take at most this many, in filename order.

        Returns:
            A :class:`~nilmframe.sources.Plan` whose ``reader_kwargs`` point
            :class:`~nilmframe.readers.WHITED` at the cache.
        """
        prefix = registry.WHITED["member_prefix"]
        known_kits = {k.lower() for k in (kits if kits is not None else CALIBRATION)}
        want_appliances = {a.lower() for a in appliances} if appliances else None
        want_regions = {r.lower() for r in regions} if regions else None

        selected: list[tuple[str, int]] = []
        elsewhere = 0
        unparsed = 0
        for name, entry in self.archive.entries.items():
            if not name.startswith(prefix) or not name.lower().endswith(".flac"):
                continue
            relative = name[len(prefix) :]
            if "/" in relative:
                # The corpus is the flat files directly under DATEN/. The archive
                # also carries `Experiments/` (including simultaneous two-appliance
                # runs), `notUsed/` and `MIXED/`, which are neither single-appliance
                # nor labelled as their filenames suggest -- and which the reader's
                # non-recursive glob would not see anyway. 527 MiB of the archive is
                # in those subtrees; fetching them is bandwidth for files that are
                # then ignored.
                elsewhere += 1
                continue
            fields = relative.rsplit(".", 1)[0].split("_")
            if len(fields) < 4:
                unparsed += 1
                continue
            appliance, _model, region, kit = fields[:4]
            if kit.lower() not in known_kits:
                # No calibration factor means the reader cannot scale it, so
                # fetching it would be wasted bandwidth.
                unparsed += 1
                continue
            if want_appliances is not None and appliance.lower() not in want_appliances:
                continue
            if want_regions is not None and region.lower() not in want_regions:
                continue
            if entry.encrypted:
                raise FetchError(
                    f"{name} is encrypted. Download the archive by hand from "
                    f"{registry.WHITED['home']}, which publishes the password and "
                    "the citation the authors ask for, then point WHITED at the "
                    "unpacked directory."
                )
            selected.append((name, entry.size))

        selected.sort()
        if limit is not None:
            selected = selected[:limit]

        notes = [f"terms and citation: {registry.WHITED['home']}"]
        if elsewhere:
            notes.append(
                f"{elsewhere} files under DATEN/Experiments, DATEN/notUsed and "
                "DATEN/MIXED are not part of the single-appliance corpus and are "
                "not fetched"
            )
        if unparsed:
            notes.append(
                f"{unparsed} recordings skipped: their filenames name no measurement "
                "kit the reader can calibrate"
            )

        return Plan(
            dataset=self.dataset,
            artifacts=tuple(
                Artifact(
                    url=self.url,
                    relpath=f"flac/{name[len(prefix) :]}",
                    size=size,
                    member=name,
                    archive_size=self.archive.size,
                )
                for name, size in selected
            ),
            reader_kwargs={"dirpath": "flac"},
            notes=tuple(notes),
        )
