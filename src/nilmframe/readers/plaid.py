"""PLAID reader.

PLAID ships one CSV per recording (``current,voltage`` columns, no header) plus
metadata describing each one. Recordings are either *submetered* -- one appliance,
described under ``appliance`` -- or *aggregate*, described under ``appliances``
with per-appliance on/off sample indices.

Two things about the distribution are easy to get wrong, and both cost you data
rather than raising:

* **The annotations come as a list, in two files.** The 2017 release splits its
  1793 recordings across ``meta_2017.json`` (719) and ``meta_2014.json`` (1074),
  which partition the corpus with no overlap. Passing one reads 40% of it.
* **The house is ``location``, not the collection date.** Every record names its
  site -- 55 of them -- while the collection campaign has two values, so deriving
  the house from the date leaves house-disjoint splitting nothing to partition.

Differences from ``legacy/data/highfreq/_plaid.py``:

* It returned bare tuples, six long in one branch and with ``brands`` unbound in
  the other, so the aggregate path raised ``UnboundLocalError`` on first use. This
  yields :class:`~nilmframe.store.Recording` objects, and aggregate recordings keep
  their appliance intervals as activations instead of being flattened into a
  parallel list that the dataset layer then re-wrapped positionally.
* Labels are normalised once, here, rather than by a ``format_label`` method that
  callers had to remember to apply.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from nilmframe.store.schema import Activation, ChannelKind, Recording

if TYPE_CHECKING:  # `nilmframe.sources` is imported only when data is fetched
    from nilmframe.sources import Plan

__all__ = ["PLAID"]

logger = logging.getLogger(__name__)

_INT = re.compile(r"\d+")


def _house_of(meta: dict) -> str:
    """The site a recording was made at.

    PLAID names it outright: every record in the 2017 release carries a
    ``location``, and there are 55 of them. That is the identity
    :class:`~nilmframe.data.LeaveHouseOut` should partition on.

    The collection campaign is only a fallback for a distribution that omits
    ``location``. It used to be the primary, and it collapsed all 1793 recordings
    into two groups named after a month -- which is not a house, and left
    house-disjoint splitting with nothing to work with.

    Example:
        >>> from nilmframe.readers.plaid import _house_of
        >>> _house_of({'location': 'house12'})
        'house12'
        >>> _house_of({'header': {'collection_time': 'July, 2013'}})
        'july_2013'
        >>> _house_of({})
        'unknown'
    """
    location = str(meta.get("location", "")).strip()
    if location:
        return format_label(location)
    campaign = str(meta.get("header", {}).get("collection_time", ""))[:10]
    return format_label(campaign.replace(",", "")) or "unknown"


def _by_id(item: tuple[str, dict]) -> tuple[int, int, str]:
    """Sort recordings numerically, tolerating an id that is not a number."""
    key = item[0]
    return (0, int(key), "") if key.isdigit() else (1, 0, key)


def format_label(label: str) -> str:
    """Normalise an appliance label to ``lower_snake_case``.

    Example:
        >>> from nilmframe.readers.plaid import format_label
        >>> format_label('Air Conditioner')
        'air_conditioner'
    """
    return "_".join(str(label).lower().split())


class PLAID:
    """Iterate PLAID recordings as :class:`Recording` objects.

    Args:
        dirpath: directory of ``{id}.csv`` waveform files.
        metadata: the annotations, in any of the shapes PLAID is distributed in
            -- a ``{id: meta}`` mapping, the published list of ``{"id", "meta"}``
            records, a path to either, or several paths. The 2017 release splits
            its 1793 recordings across ``meta_2017.json`` and ``meta_2014.json``
            with no overlap, so reading all of it means passing both.
        fs: override the sampling rate when the metadata header lacks one.

    Example:
        >>> for rec in PLAID("PLAID/CSV", "PLAID/meta.json"):   # doctest: +SKIP
        ...     writer.add(rec)

        The published release needs both halves of its annotations::

            >>> reader = PLAID("plaid/csv",                     # doctest: +SKIP
            ...                ["plaid/meta_2017.json", "plaid/meta_2014.json"])
    """

    dataset = "plaid"

    def __init__(
        self,
        dirpath: str | Path,
        metadata: dict | list | str | Path,
        *,
        fs: float | None = None,
    ) -> None:
        self.dirpath = Path(dirpath).expanduser()
        if not self.dirpath.is_dir():
            raise NotADirectoryError(f"{self.dirpath} is not a directory")

        records = self._as_records(metadata)
        self.metadata = sorted(records.items(), key=_by_id)
        self._fs_override = fs

    @staticmethod
    def _as_records(metadata: dict | list | str | Path) -> dict[str, dict]:
        """Normalise however the annotations arrived into ``{id: meta}``.

        PLAID has been distributed both as a mapping and as a list of
        ``{"id", "meta"}`` records, and the current release uses the list -- so a
        reader that accepted only the mapping could not read the published file.
        Several sources merge, which is what reading a whole release takes.
        """
        if isinstance(metadata, (str, Path)):
            metadata = json.loads(Path(metadata).expanduser().read_text())
        if isinstance(metadata, dict):
            return {str(key): value for key, value in metadata.items()}
        if isinstance(metadata, (list, tuple)):
            merged: dict[str, dict] = {}
            for item in metadata:
                if isinstance(item, (str, Path)):
                    merged.update(PLAID._as_records(item))
                elif isinstance(item, dict) and "id" in item and "meta" in item:
                    merged[str(item["id"])] = item["meta"]
                else:
                    raise TypeError(
                        "a metadata list must hold paths or {'id', 'meta'} records, "
                        f"got {type(item).__name__}"
                    )
            return merged
        raise TypeError("metadata must be a mapping, a list of records, or a path")

    def __len__(self) -> int:
        return len(self.metadata)

    def __iter__(self) -> Iterator[Recording]:
        for idx, meta in self.metadata:
            rec = self._read(idx, meta)
            if rec is not None:
                yield rec

    # -- getting the data --------------------------------------------------- #

    @classmethod
    def plan(
        cls,
        *,
        ids: list[int] | list[str] | None = None,
        limit: int | None = None,
        article: int | None = None,
    ) -> Plan:
        """Which files this configuration would need, without fetching any.

        Args:
            ids: recording ids to take; ``None`` takes all of them.
            limit: take at most this many, lowest id first.
            article: figshare article id, if not the pinned release.

        Returns:
            A :class:`~nilmframe.sources.Plan`.

        Example:
            >>> from nilmframe.readers import PLAID
            >>> print(PLAID.plan(limit=5).summary())   # doctest: +SKIP
        """
        from nilmframe.sources import PLAIDSource

        return PLAIDSource(article=article).plan(ids=ids, limit=limit)

    @classmethod
    def download(
        cls,
        cache: str | Path,
        *,
        ids: list[int] | list[str] | None = None,
        limit: int | None = None,
        article: int | None = None,
        workers: int = 4,
        max_bytes: int | None = None,
        progress: bool = True,
        **kwargs,
    ) -> PLAID:
        """Fetch a subset into ``cache`` and return a reader over it.

        The metadata describes every recording in the release, so a partial fetch
        leaves the reader warning about the waveforms that are not on disk. That
        is the honest behaviour -- the annotations really do describe more than
        you asked for -- but it means ``limit`` is for smoke tests rather than for
        building a store.

        Args:
            cache: directory to fetch into.
            ids: recording ids to take; ``None`` takes all of them.
            limit: take at most this many, lowest id first.
            article: figshare article id, if not the pinned release.
            workers: concurrent downloads.
            max_bytes: refuse a plan larger than this before transferring.
            progress: print one line per file as it lands.
            **kwargs: passed to the constructor, e.g. ``fs``.

        Returns:
            A reader over the fetched subset.

        Example:
            >>> from nilmframe.readers import PLAID
            >>> reader = PLAID.download("~/.cache/nilmframe/plaid",  # doctest: +SKIP
            ...                         limit=20)
        """
        from nilmframe.sources import materialize

        plan = cls.plan(ids=ids, limit=limit, article=article)
        paths, _ = materialize(
            plan,
            Path(cache).expanduser(),
            workers=workers,
            max_bytes=max_bytes,
            progress=print if progress else None,
        )
        return cls(paths["dirpath"], paths["metadata"], **kwargs)

    # -- internals ---------------------------------------------------------- #

    def _read(self, idx: str, meta: dict) -> Recording | None:
        path = self.dirpath / f"{idx}.csv"
        if not path.exists():
            logger.warning("PLAID: %s referenced in metadata but missing on disk", path.name)
            return None

        fs = self._fs_override or self._parse_fs(meta)
        raw = np.loadtxt(path, delimiter=",", dtype=np.float32, ndmin=2)
        if raw.shape[1] < 2:
            logger.warning("PLAID: %s has %d columns, expected 2", path.name, raw.shape[1])
            return None
        current, voltage = raw[:, 0], raw[:, 1]

        signals = {"i": current, "v": voltage}
        common = {
            "dataset": self.dataset,
            "house": _house_of(meta),
            "session": str(idx),
            "signals": signals,
            "fs": float(fs),
            "meta": {"plaid_id": str(idx)},
        }

        if "appliance" in meta:
            appliance = meta["appliance"]
            label = format_label(appliance.get("type", "unknown"))
            brand = appliance.get("brand") or None
            # PLAID's instance identity is brand plus model where present; two
            # recordings of the same physical unit must not straddle a split.
            instance = ":".join(
                str(p) for p in (label, brand, appliance.get("model")) if p and p != "unknown"
            )
            return Recording(
                kind=ChannelKind.SUBMETER,
                appliance=label,
                brand=format_label(brand) if brand else None,
                instance_id=instance or None,
                **common,
            )

        if "appliances" in meta:
            activations = self._parse_activations(meta["appliances"], len(current))
            return Recording(kind=ChannelKind.MAINS, activations=activations, **common)

        logger.warning("PLAID: recording %s has neither 'appliance' nor 'appliances'", idx)
        return None

    @staticmethod
    def _parse_fs(meta: dict) -> float:
        header = meta.get("header", {})
        raw = str(header.get("sampling_frequency", "")).lower().replace("hz", "").strip()
        if not raw:
            raise ValueError("PLAID metadata has no sampling_frequency and no fs= override given")
        return float(raw)

    @staticmethod
    def _parse_activations(entries: list[dict], n_samples: int) -> list[Activation]:
        """Turn per-appliance on/off annotations into intervals.

        An appliance may switch several times in one recording, so a single
        appliance can produce several activations.
        """
        out: list[Activation] = []
        for entry in entries:
            label = format_label(entry.get("type", "unknown"))
            ons = [int(x) for x in _INT.findall(str(entry.get("on", "")))]
            offs = [int(x) for x in _INT.findall(str(entry.get("off", "")))]
            # An appliance still running when the recording ends has no off time.
            offs.extend([n_samples] * max(0, len(ons) - len(offs)))
            if len(ons) != len(offs):
                logger.warning("PLAID: %s has %d on and %d off marks", label, len(ons), len(offs))
                continue
            for on, off in zip(ons, offs, strict=True):
                on, off = max(0, min(on, n_samples)), max(0, min(off, n_samples))
                if off > on:
                    out.append(Activation(appliance=label, on=on, off=off))
        return sorted(out, key=lambda a: (a.appliance, a.on))
