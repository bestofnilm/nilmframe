"""UCI Individual Household Electric Power Consumption reader.

One French house, four years, one reading a minute. It is the smallest thing here
-- 20 MB compressed -- and the most used: a great deal of published load
forecasting and disaggregation work benchmarks on it, so it is worth having even
though it is a single home.

One file, semicolon-separated::

    Date;Time;Global_active_power;Global_reactive_power;Voltage;Global_intensity;Sub_metering_1;Sub_metering_2;Sub_metering_3
    16/12/2006;17:24:00;4.216;0.418;234.840;18.400;0.000;1.000;17.000

Three things about it are easy to get wrong, and all three change the numbers.

**The units are not watts.** ``Global_active_power`` is in *kilowatts*, and the
three sub-meterings are in *watt-hours of active energy per minute*. Reading
either at face value puts the aggregate and its submeters three orders of
magnitude apart. The reader converts both to watts -- multiply the aggregate by
1000, the sub-meterings by 60.

**The submeters do not add up to the aggregate, by design.** They cover three
circuits -- kitchen, laundry, and water heater with air conditioning -- and
everything else in the house is unmetered. The documented remainder is
``Global_active_power * 1000 / 60 - sub_1 - sub_2 - sub_3`` watt-hours, and it is
usually the larger part. This is a corpus with a *partially* submetered
aggregate, which is a different problem from a fully submetered one.

**Missing values are ``?``, not blanks**, and there are about 26,000 of them --
1.25% of rows. They become gaps rather than zeros, because a zero here would read
as an appliance that was off.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from nilmframe.readers.ukdale import forward_fill_to_grid, split_on_gaps
from nilmframe.store.schema import ChannelKind, Recording

if TYPE_CHECKING:  # `nilmframe.sources` is imported only when data is fetched
    from nilmframe.sources import Plan

__all__ = ["SUBMETERS", "UCIHousehold"]

logger = logging.getLogger(__name__)

#: What each sub-metering circuit covers, from the dataset's documentation. They
#: are circuits rather than appliances, which is why the labels are rooms.
SUBMETERS: dict[str, str] = {
    "Sub_metering_1": "kitchen",
    "Sub_metering_2": "laundry_room",
    "Sub_metering_3": "water_heater_air_conditioner",
}

#: ``Global_active_power`` is kilowatts; the store speaks watts.
KW_TO_W = 1000.0

#: The sub-meterings are watt-hours of active energy per minute, so a reading of
#: 17 is a circuit averaging 17 * 60 = 1020 W over that minute.
WH_PER_MINUTE_TO_W = 60.0


class UCIHousehold:
    """Iterate the UCI household recordings as :class:`Recording` objects.

    Args:
        path: the ``household_power_consumption.txt`` file, the ``.zip`` holding
            it, or a directory containing either. Reading straight out of the zip
            avoids unpacking 133 MB to get at 20.
        aggregate: include ``Global_active_power`` as the mains channel.
        submeters: which sub-metering circuits to read. ``None`` reads all three.
        rate_hz: uniform rate to resample onto. The corpus is one reading a
            minute.
        max_gap_s: gaps longer than this split a channel rather than being
            forward-filled across. The default is one hour, which keeps the
            documented multi-day outages from being invented.
        max_seconds: seconds to take from the start of each channel.
        time_range: ``(start, stop)`` unix seconds.

    Example:
        >>> from nilmframe.readers import UCIHousehold
        >>> from nilmframe.store import StoreWriter
        >>> reader = UCIHousehold("uci")                   # doctest: +SKIP
        >>> with StoreWriter("stores/uci") as w:           # doctest: +SKIP
        ...     for rec in reader:
        ...         w.add(rec)
    """

    dataset = "uci_household"

    def __init__(
        self,
        path: str | Path,
        *,
        aggregate: bool = True,
        submeters: list[str] | None = None,
        rate_hz: float = 1.0 / 60.0,
        max_gap_s: float = 3600.0,
        max_seconds: float | None = None,
        time_range: tuple[float, float] | None = None,
    ) -> None:
        path = Path(path).expanduser()
        if path.is_dir():
            found = sorted(path.glob("household_power_consumption.*"))
            if not found:
                raise FileNotFoundError(f"no household_power_consumption file under {path}")
            path = found[0]
        if not path.exists():
            raise FileNotFoundError(f"{path} is missing")
        self.path = path
        self.aggregate = aggregate
        self.submeters = set(submeters) if submeters else set(SUBMETERS)
        self.rate_hz = rate_hz
        self.max_gap_s = max_gap_s
        self.max_seconds = max_seconds
        self.time_range = time_range

    def _open(self):
        """The data file, out of the zip when that is what we were given."""
        if self.path.suffix.lower() != ".zip":
            return self.path.open("rb")
        import zipfile

        archive = zipfile.ZipFile(self.path)
        names = [n for n in archive.namelist() if n.endswith(".txt")]
        if not names:
            raise FileNotFoundError(f"{self.path} holds no .txt member")
        return archive.open(names[0])

    def __iter__(self) -> Iterator[Recording]:
        import pandas as pd

        with self._open() as handle:
            frame = pd.read_csv(
                handle,
                sep=";",
                na_values=["?"],
                engine="c",
                dtype={"Date": "string", "Time": "string"},
            )
        # Combined after the read rather than through `parse_dates`, which pandas
        # is removing for multi-column specs.
        when = pd.to_datetime(
            frame["Date"] + " " + frame["Time"], format="%d/%m/%Y %H:%M:%S", errors="coerce"
        )
        stamps = when.to_numpy("datetime64[s]").astype(np.float64)
        if self.time_range is not None:
            keep = (stamps >= self.time_range[0]) & (stamps <= self.time_range[1])
            frame, stamps = frame[keep], stamps[keep]

        columns: list[tuple[str, str | None, float]] = []
        if self.aggregate:
            columns.append(("Global_active_power", None, KW_TO_W))
        for column, label in SUBMETERS.items():
            if column in self.submeters or label in self.submeters:
                columns.append((column, label, WH_PER_MINUTE_TO_W))

        for column, appliance, scale in columns:
            if column not in frame.columns:
                logger.warning("UCI: %s has no column %s", self.path.name, column)
                continue
            values = frame[column].to_numpy(np.float64) * scale
            # A missing reading is not a zero: drop it and let the gap logic
            # decide whether it becomes a new run.
            usable = np.isfinite(values)
            missing = int((~usable).sum())
            if missing:
                logger.info("UCI: %s has %d missing readings", column, missing)
            times, watts = stamps[usable], values[usable]
            if times.size < 2:
                continue

            for run, (a, b) in enumerate(split_on_gaps(times, self.max_gap_s)):
                series, t0 = forward_fill_to_grid(times[a:b], watts[a:b], self.rate_hz)
                if self.max_seconds:
                    series = series[: max(2, int(self.max_seconds * self.rate_hz))]
                if series.size < 2:
                    continue
                yield Recording(
                    dataset=self.dataset,
                    house="house_1",
                    session=f"run_{run}",
                    kind=ChannelKind.MAINS if appliance is None else ChannelKind.SUBMETER,
                    appliance=appliance,
                    instance_id=None if appliance is None else f"house_1:{appliance}",
                    signals={"p": series},
                    fs=self.rate_hz,
                    t0=t0,
                    meta={"source": self.path.name, "column": column},
                )

    # -- getting the data --------------------------------------------------- #

    @classmethod
    def plan(cls) -> Plan:
        """Which files this needs, without fetching any.

        Returns:
            A :class:`~nilmframe.sources.Plan`. There is only one file.
        """
        from nilmframe.sources import UCISource

        return UCISource().plan()

    @classmethod
    def download(
        cls,
        cache: str | Path,
        *,
        max_bytes: int | None = None,
        progress: bool = True,
        **kwargs,
    ) -> UCIHousehold:
        """Fetch the dataset into ``cache`` and return a reader over it.

        Args:
            cache: directory to fetch into.
            max_bytes: refuse a plan larger than this before transferring.
            progress: print one line per file as it lands.
            **kwargs: passed to the constructor, e.g. ``max_seconds``.

        Returns:
            A reader over the fetched file.
        """
        from nilmframe.sources import materialize

        paths, _ = materialize(
            cls.plan(),
            Path(cache).expanduser(),
            workers=1,
            max_bytes=max_bytes,
            progress=print if progress else None,
        )
        return cls(paths["path"], **kwargs)
