"""REFIT reader.

Twenty UK households measured for about two years at eight-second resolution:
the mains and up to nine individual appliances per home. Its value is the number
of *houses* -- most fully submetered corpora are one building, and a model that
transfers across twenty homes is a different claim from one that fits a single
one.

On-disk layout is one CSV per house::

    CLEAN_House1.csv
    Time,Unix,Aggregate,Appliance1,...,Appliance9,Issues
    2013-10-09 13:06:17,1381323977,523,74,0,69,0,0,0,0,0,1,0

Everything is watts, which is the pleasant part. Two things are not.

**The columns do not say what they measure.** ``Appliance4`` is a washer dryer in
house 1 and something else in house 2; the identities live in the dataset's own
documentation rather than in the file. :data:`APPLIANCES` carries the mapping for
all twenty houses, so the store gets real labels instead of ``appliance4``.

**``Issues`` flags rows the cleaning could not reconcile.** They are kept by
default -- dropping data silently is worse than carrying a flag -- but
``drop_issues=True`` removes them, and the count is logged either way.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from nilmframe.readers.ukdale import forward_fill_to_grid, split_on_gaps
from nilmframe.store.schema import ChannelKind, Recording

if TYPE_CHECKING:  # `nilmframe.sources` is imported only when data is fetched
    from nilmframe.sources import Plan

__all__ = ["APPLIANCES", "REFIT"]

logger = logging.getLogger(__name__)

#: Which appliance each generic column is, per house. Taken from the dataset's
#: published per-building metadata; without it every submeter would be labelled
#: ``appliance3`` and the label space would be meaningless across houses.
APPLIANCES: dict[int, dict[str, str]] = {
    1: {
        "Appliance1": "Fridge",
        "Appliance2": "Freezer(1)",
        "Appliance3": "Freezer(2)",
        "Appliance4": "Washer Dryer",
        "Appliance5": "Washing Machine",
        "Appliance6": "Dishwaser",
        "Appliance7": "Computer",
        "Appliance8": "Television Site",
        "Appliance9": "Electric Heater",
    },
    2: {
        "Appliance1": "Fridge-Freezer",
        "Appliance2": "Washing Machine",
        "Appliance3": "Dishwaser",
        "Appliance4": "Television Site",
        "Appliance5": "Microwave",
        "Appliance6": "Toaster",
        "Appliance7": "Hi-Fi",
        "Appliance8": "Kettle",
        "Appliance9": "Overhead Fan",
    },
    3: {
        "Appliance1": "Toaster",
        "Appliance2": "Fridge-Freezer",
        "Appliance3": "Freezer",
        "Appliance4": "Tumble Dryer",
        "Appliance5": "Dishwasher",
        "Appliance6": "Washing Machine",
        "Appliance7": "Television Site",
        "Appliance8": "Microwave",
        "Appliance9": "Kettle",
    },
    4: {
        "Appliance1": "Fridge",
        "Appliance2": "Freezer",
        "Appliance3": "Fridge-Freezer",
        "Appliance4": "Tumble Dryer",
        "Appliance5": "Washing Machine(1)",
        "Appliance6": "Washing Machine(2)",
        "Appliance7": "Television Site",
        "Appliance8": "Microwave",
        "Appliance9": "Kettle",
    },
    5: {
        "Appliance1": "Fridge-Freezer",
        "Appliance2": "Tumble Dryer",
        "Appliance3": "Washing Machine",
        "Appliance4": "Dishwasher",
        "Appliance5": "Desktop Computer",
        "Appliance6": "Television Site",
        "Appliance7": "Microwave",
        "Appliance8": "Kettle",
        "Appliance9": "Toaster",
    },
    6: {
        "Appliance1": "Freezer",
        "Appliance2": "Washing Machine",
        "Appliance3": "Dishwasher",
        "Appliance4": "MJY Computer",
        "Appliance5": "TV/Satellite",
        "Appliance6": "Microwave",
        "Appliance7": "Kettle",
        "Appliance8": "Toaster",
        "Appliance9": "PGM Computer",
    },
    7: {
        "Appliance1": "Fridge",
        "Appliance2": "Freezer(1)",
        "Appliance3": "Freezer(2)",
        "Appliance4": "Tumble Dryer",
        "Appliance5": "Washing Machine",
        "Appliance6": "Dishwasher",
        "Appliance7": "Television Site",
        "Appliance8": "Toaster",
        "Appliance9": "Kettle",
    },
    8: {
        "Appliance1": "Fridge",
        "Appliance2": "Freezer",
        "Appliance3": "Washer Dryer",
        "Appliance4": "Washing Machine",
        "Appliance5": "Toaster",
        "Appliance6": "Computer",
        "Appliance7": "Television Site",
        "Appliance8": "Microwave",
        "Appliance9": "Kettle",
    },
    9: {
        "Appliance1": "Fridge-Freezer",
        "Appliance2": "Washer Dryer",
        "Appliance3": "Washing Machine",
        "Appliance4": "Dishwasher",
        "Appliance5": "Television Site",
        "Appliance6": "Microwave",
        "Appliance7": "Kettle",
        "Appliance8": "Hi-Fi",
        "Appliance9": "Electric Heater",
    },
    10: {
        "Appliance1": "Magimix(Blender)",
        "Appliance2": "Toaster",
        "Appliance3": "Chest Freezer",
        "Appliance4": "Fridge-Freezer",
        "Appliance5": "Washing Machine",
        "Appliance6": "Dishwasher",
        "Appliance7": "Television Site",
        "Appliance8": "Microwave",
        "Appliance9": "K Mix",
    },
    11: {
        "Appliance1": "Firdge",
        "Appliance2": "Fridge-Freezer",
        "Appliance3": "Washing Machine",
        "Appliance4": "Dishwasher",
        "Appliance5": "Computer Site",
        "Appliance6": "Microwave",
        "Appliance7": "Kettle",
        "Appliance8": "Router",
        "Appliance9": "Hi-Fi",
    },
    12: {
        "Appliance1": "Fridge-Freezer",
        "Appliance2": "???",
        "Appliance3": "???",
        "Appliance4": "Computer Site",
        "Appliance5": "Microwave",
        "Appliance6": "Kettle",
        "Appliance7": "Toaster",
        "Appliance8": "Television",
        "Appliance9": "???",
    },
    13: {
        "Appliance1": "Television Site",
        "Appliance2": "Freezer",
        "Appliance3": "Washing Machine",
        "Appliance4": "Dishwasher",
        "Appliance5": "???",
        "Appliance6": "Network Site",
        "Appliance7": "Microwave",
        "Appliance8": "Microwave",
        "Appliance9": "Kettle",
    },
    14: {
        "Appliance1": "Fridge-Freezer",
        "Appliance2": "Tumble Dryer",
        "Appliance3": "Washing Machine",
        "Appliance4": "Dishwasher",
        "Appliance5": "Computer Site",
        "Appliance6": "Television Site",
        "Appliance7": "Microwave",
        "Appliance8": "Hi-Fi",
        "Appliance9": "Toaster",
    },
    15: {
        "Appliance1": "Fridge-Freezer(1)",
        "Appliance2": "Fridge-Freezer(2)",
        "Appliance3": "Electric Heater(1)",
        "Appliance4": "Electric Heater(2)",
        "Appliance5": "Washing Machine",
        "Appliance6": "Dishwasher",
        "Appliance7": "Computer Site",
        "Appliance8": "Television Site",
        "Appliance9": "Dehumidifier",
    },
    16: {
        "Appliance1": "Freezer",
        "Appliance2": "Fridge-Freezer",
        "Appliance3": "Tumble Dryer",
        "Appliance4": "Washing Machine",
        "Appliance5": "Computer Site",
        "Appliance6": "Television Site",
        "Appliance7": "Microwave",
        "Appliance8": "Kettle",
        "Appliance9": "TV Site(Bedroom)",
    },
    17: {
        "Appliance1": "Fridge(garage)",
        "Appliance2": "Freezer(garage)",
        "Appliance3": "Fridge-Freezer",
        "Appliance4": "Washer Dryer(garage)",
        "Appliance5": "Washing Machine",
        "Appliance6": "Dishwasher",
        "Appliance7": "Desktop Computer",
        "Appliance8": "Television Site",
        "Appliance9": "Microwave",
    },
    18: {
        "Appliance1": "Fridge Freezer",
        "Appliance2": "Washing Machine",
        "Appliance3": "Television Site",
        "Appliance4": "Microwave",
        "Appliance5": "Kettle",
        "Appliance6": "Toaster",
        "Appliance7": "Bread-maker",
        "Appliance8": "Games Console",
        "Appliance9": "Hi-Fi",
    },
    19: {
        "Appliance1": "Fridge",
        "Appliance2": "Freezer",
        "Appliance3": "Tumble Dryer",
        "Appliance4": "Washing Machine",
        "Appliance5": "Dishwasher",
        "Appliance6": "Computer Site",
        "Appliance7": "Television Site",
        "Appliance8": "Microwave",
        "Appliance9": "Kettle",
    },
    20: {
        "Appliance1": "Fridge-Freezer",
        "Appliance2": "Tumble Dryer",
        "Appliance3": "Washing Machine",
        "Appliance4": "Dishwasher",
        "Appliance5": "Food Mixer",
        "Appliance6": "Television",
        "Appliance7": "???",
        "Appliance8": "Vivarium",
        "Appliance9": "Pond Pump",
    },
}

_HOUSE = re.compile(r"House(\d+)", re.I)

#: Spellings the published metadata uses for one thing. Left alone, `Dishwaser`
#: and `Dishwasher` become two appliance classes and a model trained across
#: houses sees two rare labels instead of one common one.
_SPELLINGS = {
    "dishwaser": "dishwasher",
    "firdge": "fridge",
    "fridge-freezer": "fridge_freezer",
}

#: A parenthetical says *which* one or *where*, not *what*: `Freezer(1)` and
#: `Freezer(garage)` are both freezers. The qualifier is dropped from the label
#: and kept on the channel's metadata.
_QUALIFIER = re.compile(r"\(([^)]*)\)")


def _label(name: str) -> str:
    """The appliance class, with the dataset's own spelling variants reconciled.

    Example:
        >>> from nilmframe.readers.refit import _label
        >>> _label('Freezer(1)'), _label('Freezer(garage)'), _label('Dishwaser')
        ('freezer', 'freezer', 'dishwasher')
        >>> _label('Television Site'), _label('Fridge-Freezer')
        ('television_site', 'fridge_freezer')
    """
    bare = _QUALIFIER.sub("", name).strip()
    slug = "_".join(bare.lower().replace("-", "_").split())
    return _SPELLINGS.get(slug, _SPELLINGS.get(bare.lower(), slug))


def _qualifier(name: str) -> str | None:
    """The bracketed part, if any -- which unit, or which room."""
    match = _QUALIFIER.search(name)
    return match.group(1).strip() or None if match else None


class REFIT:
    """Iterate REFIT recordings as :class:`Recording` objects.

    Args:
        dirpath: directory of ``CLEAN_HouseN.csv`` files.
        houses: house numbers to read. ``None`` reads every file present.
        appliances: appliance labels to keep, after mapping. ``None`` keeps all.
        aggregate: include each house's mains.
        rate_hz: uniform rate to resample onto. REFIT's nominal cadence is one
            reading every eight seconds.
        max_gap_s: gaps longer than this split a channel rather than being
            forward-filled across.
        drop_issues: discard rows the dataset flags in its ``Issues`` column.
        max_seconds: seconds to take from the start of each channel.
        time_range: ``(start, stop)`` unix seconds.

    Example:
        >>> from nilmframe.readers import REFIT
        >>> from nilmframe.store import StoreWriter
        >>> reader = REFIT("refit", houses=[1])           # doctest: +SKIP
        >>> with StoreWriter("stores/refit") as w:        # doctest: +SKIP
        ...     for rec in reader:
        ...         w.add(rec)
    """

    dataset = "refit"

    def __init__(
        self,
        dirpath: str | Path,
        *,
        houses: list[int] | None = None,
        appliances: list[str] | None = None,
        aggregate: bool = True,
        rate_hz: float = 1.0 / 8.0,
        max_gap_s: float = 300.0,
        drop_issues: bool = False,
        max_seconds: float | None = None,
        time_range: tuple[float, float] | None = None,
    ) -> None:
        self.dirpath = Path(dirpath).expanduser()
        if not self.dirpath.is_dir():
            raise NotADirectoryError(f"{self.dirpath} is not a directory")
        self.houses = houses
        self.appliances = {_label(a) for a in appliances} if appliances else None
        self.aggregate = aggregate
        self.rate_hz = rate_hz
        self.max_gap_s = max_gap_s
        self.drop_issues = drop_issues
        self.max_seconds = max_seconds
        self.time_range = time_range

    def __len__(self) -> int:
        return len(self._files())

    def __iter__(self) -> Iterator[Recording]:
        for house, path in self._files():
            yield from self._read(house, path)

    # -- getting the data --------------------------------------------------- #

    @classmethod
    def plan(cls, *, houses: list[int] | None = None) -> Plan:
        """Which files this configuration would need, without fetching any.

        Args:
            houses: house numbers. ``None`` takes every house published.

        Returns:
            A :class:`~nilmframe.sources.Plan`.
        """
        from nilmframe.sources import REFITSource

        return REFITSource().plan(houses=houses)

    @classmethod
    def download(
        cls,
        cache: str | Path,
        *,
        houses: list[int] | None = None,
        workers: int = 3,
        max_bytes: int | None = None,
        progress: bool = True,
        **kwargs,
    ) -> REFIT:
        """Fetch whole houses into ``cache`` and return a reader over them.

        The unit here is a house: one CSV holds its mains and every appliance,
        about 400 MB, so there is no cheaper slice than one home.

        Args:
            cache: directory to fetch into.
            houses: house numbers.
            workers: concurrent downloads.
            max_bytes: refuse a plan larger than this before transferring.
            progress: print one line per file as it lands.
            **kwargs: passed to the constructor, e.g. ``max_seconds``.

        Returns:
            A reader over the fetched houses.
        """
        from nilmframe.sources import materialize

        plan = cls.plan(houses=houses)
        paths, _ = materialize(
            plan,
            Path(cache).expanduser(),
            workers=workers,
            max_bytes=max_bytes,
            progress=print if progress else None,
        )
        return cls(paths["dirpath"], houses=houses, **kwargs)

    # -- internals ---------------------------------------------------------- #

    def _files(self) -> list[tuple[int, Path]]:
        out = []
        for path in sorted(self.dirpath.glob("*House*.csv")):
            match = _HOUSE.search(path.stem)
            if match is None:
                continue
            house = int(match.group(1))
            if self.houses is None or house in self.houses:
                out.append((house, path))
        return sorted(out)

    def _read(self, house: int, path: Path) -> Iterator[Recording]:
        import pandas as pd

        frame = pd.read_csv(path, engine="c")
        if "Unix" not in frame.columns:
            logger.warning("REFIT: %s has no Unix column", path.name)
            return

        if self.drop_issues and "Issues" in frame.columns:
            flagged = int((frame["Issues"] != 0).sum())
            if flagged:
                logger.info("REFIT: dropping %d flagged rows from %s", flagged, path.name)
            frame = frame[frame["Issues"] == 0]

        stamps = frame["Unix"].to_numpy(np.float64)
        if self.time_range is not None:
            keep = (stamps >= self.time_range[0]) & (stamps <= self.time_range[1])
            frame, stamps = frame[keep], stamps[keep]
        if stamps.size < 2:
            return

        names = APPLIANCES.get(house, {})
        columns: list[tuple[str, str | None]] = []
        if self.aggregate and "Aggregate" in frame.columns:
            columns.append(("Aggregate", None))
        for column in frame.columns:
            if not column.startswith("Appliance"):
                continue
            label = _label(names.get(column, column))
            if self.appliances is not None and label not in self.appliances:
                continue
            columns.append((column, label))

        for column, appliance in columns:
            watts = frame[column].to_numpy(np.float64)
            for run, (a, b) in enumerate(split_on_gaps(stamps, self.max_gap_s)):
                series, t0 = forward_fill_to_grid(stamps[a:b], watts[a:b], self.rate_hz)
                if self.max_seconds:
                    series = series[: max(2, int(self.max_seconds * self.rate_hz))]
                if series.size < 2:
                    continue
                yield Recording(
                    dataset=self.dataset,
                    house=f"house_{house}",
                    # Every column of one file shares a clock and is cut on the
                    # same gaps, so a run number ties the mains to its submeters.
                    session=f"run_{run}",
                    kind=ChannelKind.MAINS if appliance is None else ChannelKind.SUBMETER,
                    appliance=appliance,
                    instance_id=None if appliance is None else f"house_{house}:{appliance}",
                    signals={"p": series},
                    fs=self.rate_hz,
                    t0=t0,
                    meta={
                        "source": path.name,
                        "column": column,
                        "house": house,
                        "original_name": names.get(column),
                        "qualifier": _qualifier(names.get(column, "")),
                    },
                )
