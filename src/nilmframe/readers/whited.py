"""WHITED reader.

WHITED ships stereo FLAC files named
``{appliance}_{model}_{region}_{kit}_{timestamp}.flac``, channel 0 voltage and
channel 1 current, scaled by per-kit calibration factors.

Differences from ``legacy/data/highfreq/_whited.py``:

* It read FLAC through ``audioread`` by concatenating raw byte buffers and
  reinterpreting them as int16 -- fragile, and an extra dependency. ``soundfile``
  reads FLAC directly and is already needed for the readers extra.
* It returned a five-tuple where PLAID returned six, which the dataset layer then
  patched positionally. Both now yield :class:`Recording`.
* ``model`` was carried as the "brand". It is kept as the instance identity too,
  since two recordings of the same model are the same physical unit here and must
  not straddle a split.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from nilmframe.store.schema import ChannelKind, Recording

if TYPE_CHECKING:  # `nilmframe.sources` is imported only when data is fetched
    from nilmframe.sources import Plan

__all__ = ["CALIBRATION", "WHITED"]

logger = logging.getLogger(__name__)

#: Per-measurement-kit scaling from normalised PCM to volts and amperes.
CALIBRATION: dict[str, dict[str, float]] = {
    "MK1": {"volt": 1033.64, "amp": 61.4835},
    "MK2": {"volt": 861.15, "amp": 60.200},
    "MK3": {"volt": 988.926, "amp": 60.9562},
}


def format_label(label: str) -> str:
    """Normalise an appliance label to ``lower_snake_case``."""
    return "_".join(str(label).lower().split())


class WHITED:
    """Iterate WHITED recordings as :class:`Recording` objects.

    Args:
        dirpath: directory containing the ``.flac`` files.
        calibration: override the per-kit scaling factors.
        strict: raise on an unknown measurement kit instead of skipping the file.

    Example:
        WHITED ships one stereo FLAC per activation, with the measurement kit named
        in the filename; the reader picks the right scaling from that name::

            >>> from nilmframe.readers import WHITED
            >>> from nilmframe.store import StoreWriter
            >>> with StoreWriter("stores/whited") as w:       # doctest: +SKIP
            ...     for rec in WHITED("WHITED"):
            ...         w.add(rec)

        Pass ``strict=True`` while checking a new download, so an unrecognised kit
        is reported rather than silently skipped::

            >>> recs = list(WHITED("WHITED", strict=True))    # doctest: +SKIP
    """

    dataset = "whited"

    def __init__(
        self,
        dirpath: str | Path,
        *,
        calibration: dict[str, dict[str, float]] | None = None,
        strict: bool = False,
    ) -> None:
        self.dirpath = Path(dirpath).expanduser()
        if not self.dirpath.is_dir():
            raise NotADirectoryError(f"{self.dirpath} is not a directory")
        self.calibration = dict(calibration or CALIBRATION)
        self.strict = strict
        self.files = sorted(self.dirpath.glob("*.flac"))

    def __len__(self) -> int:
        return len(self.files)

    @classmethod
    def plan(
        cls,
        *,
        appliances: list[str] | None = None,
        kits: list[str] | None = None,
        regions: list[str] | None = None,
        limit: int | None = None,
    ) -> Plan:
        """Which files this configuration would need, without fetching any.

        WHITED is one 2.1 GB archive, but its host serves range requests and every
        recording's appliance, region and measurement kit are in its filename, so
        a subset can be chosen from the archive's directory alone.

        Args:
            appliances: appliance names to take, e.g. ``["Kettle"]``.
            kits: measurement kits. Defaults to those with calibration factors.
            regions: region codes, e.g. ``["r1"]``.
            limit: take at most this many.

        Returns:
            A :class:`~nilmframe.sources.Plan`.

        Example:
            >>> from nilmframe.readers import WHITED
            >>> print(WHITED.plan(appliances=['Kettle']).summary())   # doctest: +SKIP
        """
        from nilmframe.sources import WHITEDSource

        return WHITEDSource().plan(appliances=appliances, kits=kits, regions=regions, limit=limit)

    @classmethod
    def download(
        cls,
        cache: str | Path,
        *,
        appliances: list[str] | None = None,
        kits: list[str] | None = None,
        regions: list[str] | None = None,
        limit: int | None = None,
        workers: int = 4,
        max_bytes: int | None = None,
        progress: bool = True,
        **kwargs,
    ) -> WHITED:
        """Fetch a subset into ``cache`` and return a reader over it.

        Args:
            cache: directory to fetch into.
            appliances: appliance names to take.
            kits: measurement kits. Defaults to those with calibration factors.
            regions: region codes.
            limit: take at most this many.
            workers: concurrent downloads.
            max_bytes: refuse a plan larger than this before transferring.
            progress: print one line per file as it lands.
            **kwargs: passed to the constructor, e.g. ``strict``.

        Returns:
            A reader over the fetched subset.

        Example:
            >>> from nilmframe.readers import WHITED
            >>> reader = WHITED.download("~/.cache/nilmframe/whited",  # doctest: +SKIP
            ...                          appliances=['Kettle'], limit=10)
        """
        from nilmframe.sources import materialize

        plan = cls.plan(appliances=appliances, kits=kits, regions=regions, limit=limit)
        paths, _ = materialize(
            plan,
            Path(cache).expanduser(),
            workers=workers,
            max_bytes=max_bytes,
            progress=print if progress else None,
        )
        return cls(paths["dirpath"], **kwargs)

    def __iter__(self) -> Iterator[Recording]:
        try:
            import soundfile as sf
        except ImportError as exc:  # pragma: no cover - exercised by the extras matrix
            raise ImportError(
                "reading WHITED needs soundfile: pip install 'nilmframe[readers]'"
            ) from exc

        for path in self.files:
            parts = path.stem.split("_")
            if len(parts) < 4:
                logger.warning("WHITED: cannot parse %s, expected 4+ underscore fields", path.name)
                continue
            appliance, model, region, kit = parts[:4]

            factors = self.calibration.get(kit)
            if factors is None:
                if self.strict:
                    raise ValueError(f"unknown measurement kit {kit!r} in {path.name}")
                logger.warning("WHITED: unknown kit %s in %s, skipping", kit, path.name)
                continue

            data, fs = sf.read(path, dtype="float32", always_2d=True)
            if data.shape[1] < 2:
                logger.warning("WHITED: %s has %d channels, expected 2", path.name, data.shape[1])
                continue

            label = format_label(appliance)
            yield Recording(
                dataset=self.dataset,
                house=region,
                session=path.stem,
                kind=ChannelKind.SUBMETER,
                appliance=label,
                brand=format_label(model),
                instance_id=f"{label}:{format_label(model)}",
                signals={
                    "v": np.ascontiguousarray(factors["volt"] * data[:, 0]),
                    "i": np.ascontiguousarray(factors["amp"] * data[:, 1]),
                },
                fs=float(fs),
                meta={"kit": kit, "region": region, "model": model, "file": path.name},
            )
