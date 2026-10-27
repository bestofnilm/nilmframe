"""AMPds2 reader.

One house near Vancouver, metered once a minute from April 2012 to March 2014:
the whole-house feed and 21 breaker-level sub-meters, with no missing readings.
Two years of one building is a narrower claim than REFIT's twenty homes, and it
buys something none of the others offer -- every meter reports **reactive** power
next to active.

That second axis is the reason to read this dataset. A low-frequency aggregate is
one number per minute, and one number cannot tell a 1500 W wall oven from a
1500 W heat pump. With ``q`` alongside ``p`` the two separate on power factor
alone: the oven is resistive and sits near ``q = 0``, the heat pump is inductive
and does not. Hart's 1992 signature space is exactly this plane, and it is
reachable here without a waveform.

The deposit publishes the same numbers twice, in two layouts that share a
directory and a filename prefix. One file per **meter**, carrying every quantity
that meter measured::

    Electricity_WHE.csv
    unix_ts,V,I,f,DPF,APF,P,Pt,Q,Qt,S,St
    1333263600,242.1,4.4,60.0,0.98,0.94,1123,15,224,3,1145,15

and one file per **measurement**, carrying that one quantity for every meter::

    Electricity_P.csv
    UNIX_TS,WHE,RSE,GRE,MHE,B1E,BME,CWE,DWE,EQE,FRE,HPE,OFE,UTE,WOE,B2E,CDE,DNE,EBE,FGE,HTE,OUE,TVE,UNE
    1333263600,1123,0,0,1123,0,32,0,0,58,3,4,58,0,0,3,0,0,29,120,0,0,42,691

Both are read, and which one is on disk is worked out from the suffix rather
than configured -- meter codes and measurement names are disjoint, so
``Electricity_P.csv`` and ``Electricity_HPE.csv`` cannot be confused. The
measurement layout is the cheaper fetch by a factor of ten when only ``P`` and
``Q`` are wanted, which is why the source asks for it first.

Everything is watts and vars on a clean one-minute grid, which is the pleasant
part. Two columns are not meters. ``MHE`` and ``UNE`` are computed rather than
measured -- the main house's share, and the whole-house reading minus everything
that was metered. They are read, because a residual is worth being able to look
at, and excluded from the appliance label space, because "the part nobody
measured" is not a class a model can be asked to predict.
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

__all__ = ["AMPDS_METERS", "AMPds2"]

logger = logging.getLogger(__name__)

#: What each four-letter meter code measures, and a coarse physical category.
#:
#: The codes are the dataset's own and appear nowhere in the CSV but the header
#: row, so without this table every channel would be labelled ``fge`` and the
#: label space would say nothing. The category is by *electrical behaviour*
#: rather than by room, because that is what the measurement can actually
#: support: a resistive load and a motor differ in the data, a bedroom and a
#: dining room do not.
AMPDS_METERS: dict[str, tuple[str, str]] = {
    "WHE": ("whole_house", "aggregate"),
    "RSE": ("rental_suite", "aggregate"),
    "GRE": ("garage", "mixed"),
    "MHE": ("main_house", "aggregate"),
    "B1E": ("north_bedroom", "mixed"),
    "B2E": ("south_bedroom", "mixed"),
    "BME": ("basement", "mixed"),
    "CDE": ("clothes_dryer", "resistive"),
    "CWE": ("clothes_washer", "motor"),
    "DNE": ("dining_room", "mixed"),
    "DWE": ("dishwasher", "motor"),
    "EBE": ("electronics_workbench", "electronic"),
    "EQE": ("security_network_equipment", "electronic"),
    "FGE": ("fridge", "motor"),
    "FRE": ("furnace_fan", "motor"),
    "HPE": ("heat_pump", "motor"),
    "HTE": ("instant_hot_water", "resistive"),
    "OFE": ("home_office", "electronic"),
    "OUE": ("outside_plug", "mixed"),
    "TVE": ("entertainment", "electronic"),
    "UTE": ("utility_room_plug", "mixed"),
    "WOE": ("wall_oven", "resistive"),
    "UNE": ("unmetered_residual", "residual"),
}

#: Codes that are not a single load. ``WHE`` is the aggregate; ``RSE`` and
#: ``MHE`` are sub-aggregates of it; ``UNE`` is what was left over. They are all
#: read, and none of them is an appliance.
AGGREGATE_CODES = frozenset({"WHE", "RSE", "MHE", "UNE"})

#: Measurement names, as they appear in the one-file-per-measurement layout.
#: Disjoint from the meter codes, which is what makes the two layouts
#: distinguishable from a filename alone.
MEASUREMENTS = frozenset({"V", "I", "F", "DPF", "APF", "P", "PT", "Q", "QT", "S", "ST"})

#: Loads that turn electricity into heat with a resistive element. This is a
#: statement about physics, not about rooms: each one draws its power at a power
#: factor near unity, which puts it in a corner of the P-Q plane that a motor or
#: a switched-mode supply cannot reach. That separability is what makes the class
#: recoverable without labels.
HEATING_METERS: tuple[str, ...] = ("CDE", "WOE", "HTE", "DWE")

#: Heats the house, but by moving heat rather than dissipating it -- a compressor,
#: so inductive and nowhere near unity power factor. Kept apart from
#: :data:`HEATING_METERS` because lumping it in would blur the one physical
#: property the unsupervised step relies on.
HEAT_PUMP_METERS: tuple[str, ...] = ("HPE",)

#: On-power in watts, per appliance, where the default 10 W is clearly wrong.
#: A dryer idling at 12 W is off; a fridge at 12 W is not.
ON_THRESHOLDS: dict[str, float] = {
    "clothes_dryer": 100.0,
    "wall_oven": 100.0,
    "heat_pump": 100.0,
    "instant_hot_water": 100.0,
    "clothes_washer": 20.0,
    "dishwasher": 20.0,
    "furnace_fan": 20.0,
    "fridge": 20.0,
}


class AMPds2:
    """Iterate AMPds2 meters as :class:`Recording` objects.

    Args:
        dirpath: directory holding the dataset's CSVs, in either published
            layout.
        meters: meter codes to read. ``None`` reads every one found.
        rate_hz: uniform rate for the output. AMPds2's native cadence is one
            reading a minute, which is ``1/60``.
        max_gap_s: gaps longer than this split a meter into separate sessions
            rather than being filled. AMPds2 claims none, so the default is
            generous enough to be a check rather than a policy.
        time_range: ``(start, stop)`` unix seconds, to read a slice of the two
            years rather than all of it.

    Example:
        >>> from nilmframe.readers import AMPds2
        >>> reader = AMPds2('~/datasets/ampds2')                # doctest: +SKIP
        >>> rec = next(iter(reader))                            # doctest: +SKIP
        >>> rec.appliance, rec.quantities, rec.fs               # doctest: +SKIP
        ('whole_house', ['p', 'q'], 0.016666666666666666)
    """

    dataset = "ampds2"

    def __init__(
        self,
        dirpath: str | Path,
        *,
        meters: list[str] | None = None,
        rate_hz: float = 1.0 / 60.0,
        max_gap_s: float = 3600.0,
        time_range: tuple[float, float] | None = None,
    ) -> None:
        self.dirpath = Path(dirpath).expanduser()
        self.meters = [m.upper() for m in meters] if meters else None
        self.rate_hz = float(rate_hz)
        self.max_gap_s = float(max_gap_s)
        self.time_range = time_range

    # -- reading ------------------------------------------------------------ #

    def _read_csv(self, path: Path) -> tuple[list[str], np.ndarray]:
        """Header names and a float table, with the timestamp column first.

        Read with pandas rather than :func:`numpy.genfromtxt`: a measurement file
        is 1,051,200 rows by 24 columns and genfromtxt parses it in Python, which
        turns a two-second read into a two-minute one.
        """
        import pandas as pd

        frame = pd.read_csv(path, dtype=np.float64)
        header = [str(c).strip().strip('"') for c in frame.columns]
        return header, frame.to_numpy(dtype=np.float64)

    @staticmethod
    def _suffix(path: Path) -> str:
        """The part of the filename that names either a meter or a measurement."""
        stem = path.stem
        return (stem.split("_", 1)[1] if stem.startswith("Electricity_") else stem).upper()

    def _pivot_files(self) -> dict[str, Path]:
        """``{measurement: path}`` for the one-file-per-measurement layout."""
        return {
            self._suffix(p): p
            for p in sorted(self.dirpath.glob("*.csv"))
            if self._suffix(p) in MEASUREMENTS
        }

    def _per_meter_files(self) -> dict[str, Path]:
        """``{meter_code: path}`` for the one-file-per-meter layout."""
        return {
            self._suffix(p): p
            for p in sorted(self.dirpath.glob("*.csv"))
            if self._suffix(p) in AMPDS_METERS
        }

    def _series(self) -> dict[str, tuple[np.ndarray, dict[str, np.ndarray]]]:
        """``{meter_code: (timestamps, {quantity: values})}`` from whichever layout is on disk."""
        out: dict[str, tuple[np.ndarray, dict[str, np.ndarray]]] = {}

        pivots = self._pivot_files()
        if "P" in pivots:
            stamps = None
            columns: dict[str, dict[str, np.ndarray]] = {}
            for quantity, path in pivots.items():
                if quantity not in ("P", "Q"):
                    continue
                header, table = self._read_csv(path)
                ts = table[:, 0]
                if stamps is None:
                    stamps = ts
                elif not np.array_equal(stamps, ts):
                    raise ValueError(
                        f"{path.name} is not on the same clock as the other quantity files"
                    )
                for col, name in enumerate(header):
                    code = name.upper()
                    if code in AMPDS_METERS:
                        columns.setdefault(code, {})[quantity.lower()] = table[:, col]
            assert stamps is not None
            for code, signals in columns.items():
                out[code] = (stamps, signals)
            return out

        for code, path in self._per_meter_files().items():
            header, table = self._read_csv(path)
            index = {name.upper(): i for i, name in enumerate(header)}
            if "P" not in index:
                logger.warning("%s has no P column; skipping", path.name)
                continue
            signals = {"p": table[:, index["P"]]}
            if "Q" in index:
                signals["q"] = table[:, index["Q"]]
            out[code] = (table[:, 0], signals)
        return out

    def __iter__(self) -> Iterator[Recording]:
        series = self._series()
        if not series:
            raise FileNotFoundError(
                f"no AMPds2 CSVs under {self.dirpath}; expected either "
                f"Electricity_P.csv / Electricity_Q.csv or one file per meter code"
            )
        wanted = self.meters or sorted(series)
        for code in wanted:
            if code not in series:
                logger.warning("meter %s not present under %s", code, self.dirpath)
                continue
            stamps, signals = series[code]

            if self.time_range is not None:
                start, stop = self.time_range
                keep = (stamps >= start) & (stamps <= stop)
                if not keep.any():
                    logger.warning("meter %s has nothing in %s", code, self.time_range)
                    continue
                stamps = stamps[keep]
                signals = {q: v[keep] for q, v in signals.items()}

            appliance, category = AMPDS_METERS[code]
            is_aggregate = code in AGGREGATE_CODES

            for run, (lo, hi) in enumerate(split_on_gaps(stamps, self.max_gap_s)):
                gridded = {}
                t0 = 0.0
                for quantity, values in sorted(signals.items()):
                    gridded[quantity], t0 = forward_fill_to_grid(
                        stamps[lo:hi], values[lo:hi], self.rate_hz
                    )
                yield Recording(
                    dataset=self.dataset,
                    house="house_1",
                    session=f"run_{run}",
                    kind=ChannelKind.MAINS if is_aggregate else ChannelKind.SUBMETER,
                    appliance=None if is_aggregate else appliance,
                    signals=gridded,
                    fs=self.rate_hz,
                    instance_id=f"ampds2-{code.lower()}",
                    t0=t0,
                    meta={"meter_code": code, "category": category, "label": appliance},
                )

    # -- fetching ----------------------------------------------------------- #

    @classmethod
    def plan(cls, **kwargs) -> Plan:
        """What the download would cost, without spending it."""
        from nilmframe.sources import AMPds2Source

        return AMPds2Source(**kwargs).plan()

    @classmethod
    def download(
        cls,
        cache: str | Path,
        *,
        max_bytes: int | None = None,
        progress: bool = True,
        source_kwargs: dict | None = None,
        **kwargs,
    ) -> AMPds2:
        """Fetch the dataset into ``cache`` and return a reader over it.

        Args:
            cache: directory to fetch into.
            max_bytes: refuse a plan larger than this before transferring.
            progress: print one line per file as it lands.
            source_kwargs: passed to :class:`~nilmframe.sources.AMPds2Source`,
                e.g. ``quantities``.
            **kwargs: passed to the constructor, e.g. ``time_range``.

        Returns:
            A reader over the fetched files.
        """
        from nilmframe.sources import materialize

        paths, _ = materialize(
            cls.plan(**(source_kwargs or {})),
            Path(cache).expanduser(),
            workers=1,
            max_bytes=max_bytes,
            progress=print if progress else None,
        )
        return cls(paths["dirpath"], **kwargs)
