"""SmartNIALMeter reader.

Twenty buildings, up to two years each, five-second resolution: an aggregate from
the building's smart meter plus a dedicated sensor on every appliance behind it.
That combination -- long, fully submetered, and *many* buildings -- is what makes
it useful, and it is a different kind of usefulness from the waveform corpora.

At 0.2 Hz there is no waveform here. Nothing that needs a mains cycle applies: no
V-I trajectory, no cycle alignment, no harmonics. What it does support is the
problem most NILM papers actually pose -- recover appliance power from an
aggregate power series -- with enough buildings to hold some out.

On-disk layout::

    raw/
      building_01/
        cii-adapter.h5        the smart meter: the aggregate
        freezer.h5            one sensor per appliance
        washing_machine.h5
      building_02/
        ...

**The aggregate is called ``cii-adapter``.** It is named for the interface it was
read through -- the smart meter's Consumer Information Interface -- rather than
for what it measures, and it is the one file present in all twenty buildings.
Treating it as an appliance would put the whole building's consumption in the
label space as a device called "cii adapter".

Each file is a pandas ``frame_table`` written through PyTables: a timestamp index
and a block of columns. It is read here with ``h5py`` directly rather than through
``pandas.read_hdf``, which would pull in PyTables for a layout that is two arrays
and an attribute.

**The columns are not the same from file to file**, and this is the thing that
quietly loses you most of the corpus. A single-phase appliance publishes
``Active Power``; a three-phase one publishes ``Active Power L1..L3``, which are
the halves of one machine and must be summed; and the smart meter publishes no
power column at all -- just ``Voltage``, ``Current`` and ``Power Factor`` per
phase, whose product summed over phases *is* the aggregate. See
:func:`active_power`.
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

__all__ = ["AGGREGATE_FILE", "SmartNIALM", "active_power"]

logger = logging.getLogger(__name__)

#: The smart-meter file. Present in every building; not an appliance.
AGGREGATE_FILE = "cii-adapter"

#: Pickle protocol 0 stores each string as ``V<text>\n``. The column names live in
#: an attribute in that form, and matching the text is safer than unpickling a
#: value that came out of a downloaded file.
_PICKLED_STRING = re.compile(rb"V([^\n]+)")


def active_power(columns: list[str], block: np.ndarray) -> np.ndarray | None:
    """Total active power in watts, however this file happens to report it.

    The release is not one schema but three, and the difference is not cosmetic:

    * a single-phase appliance publishes ``Active Power``;
    * a three-phase appliance publishes ``Active Power L1..L3``, which are the
      per-phase halves of one machine and have to be summed;
    * the smart meter publishes no power column at all -- it gives ``Voltage``,
      ``Current`` and ``Power Factor`` per phase, and the aggregate everything
      else is compared against is their product, summed over the phases.

    Read only the literal ``Active Power`` column and you get the single-phase
    appliances and nothing else -- no aggregate, and none of the largest loads.

    Args:
        columns: the block's column names, in order.
        block: samples, ``(n, len(columns))``.

    Returns:
        Watts per sample, or ``None`` if this file reports neither form.

    Example:
        >>> import numpy as np
        >>> from nilmframe.readers.smartnialm import active_power
        >>> block = np.array([[230.0, 2.0, 0.5], [230.0, 4.0, 1.0]])
        >>> active_power(['Voltage L1', 'Current L1', 'Power Factor L1'], block)
        array([230., 920.])
        >>> active_power(['Active Power L1', 'Active Power L2'],
        ...              np.array([[10.0, 5.0]]))
        array([15.])
    """
    index = {name: n for n, name in enumerate(columns)}

    watts = [n for name, n in index.items() if name.startswith("Active Power")]
    if watts:
        return block[:, sorted(watts)].sum(axis=1)

    phases = sorted(name.removeprefix("Voltage ") for name in index if name.startswith("Voltage "))
    total = None
    for phase in phases:
        try:
            volts = block[:, index[f"Voltage {phase}"]]
            amps = block[:, index[f"Current {phase}"]]
            factor = block[:, index[f"Power Factor {phase}"]]
        except KeyError:
            continue
        term = volts * amps * factor
        total = term if total is None else total + term
    return total


def _column_names(dataset) -> list[str]:
    """Column names for a pandas ``frame_table`` value block.

    Example:
        >>> from nilmframe.readers.smartnialm import _column_names
        >>> blob = b'(lp0\\nVActive Power\\np1\\naVPower Factor\\np2\\na.'
        >>> class Fake:
        ...     attrs = {'values_block_0_kind': blob}
        >>> _column_names(Fake())
        ['Active Power', 'Power Factor']
    """
    raw = dataset.attrs.get("values_block_0_kind", b"")
    if isinstance(raw, str):
        raw = raw.encode()
    return [m.decode("utf-8", "replace") for m in _PICKLED_STRING.findall(raw)]


class SmartNIALM:
    """Iterate SmartNIALMeter recordings as :class:`Recording` objects.

    Args:
        dirpath: a ``raw`` or ``preprocessed`` directory holding ``building_NN``.
        buildings: building numbers to read. ``None`` reads every one present.
        appliances: appliance file stems to read, e.g. ``["freezer"]``. The
            aggregate is always read unless ``aggregate=False``.
        aggregate: include each building's smart meter.
        rate_hz: uniform rate to resample onto. The corpus is nominally one
            reading every five seconds.
        max_gap_s: gaps longer than this split a channel into separate sessions
            rather than being forward-filled across. Two years of a real
            installation contains outages, and filling one manufactures days of
            an appliance sitting at its last known power.
        max_seconds: seconds to take from the start of each channel.
        time_range: ``(start, stop)`` unix seconds.

    Example:
        >>> from nilmframe.readers import SmartNIALM
        >>> from nilmframe.store import StoreWriter
        >>> reader = SmartNIALM("snm/raw", buildings=[1])       # doctest: +SKIP
        >>> with StoreWriter("stores/snm") as w:                # doctest: +SKIP
        ...     for rec in reader:
        ...         w.add(rec)
    """

    dataset = "smartnialm"

    def __init__(
        self,
        dirpath: str | Path,
        *,
        buildings: list[int] | None = None,
        appliances: list[str] | None = None,
        aggregate: bool = True,
        rate_hz: float = 0.2,
        max_gap_s: float = 900.0,
        max_seconds: float | None = None,
        time_range: tuple[float, float] | None = None,
    ) -> None:
        self.dirpath = Path(dirpath).expanduser()
        if not self.dirpath.is_dir():
            raise NotADirectoryError(f"{self.dirpath} is not a directory")
        self.buildings = buildings
        self.appliances = {a.lower() for a in appliances} if appliances else None
        self.aggregate = aggregate
        self.rate_hz = rate_hz
        self.max_gap_s = max_gap_s
        self.max_seconds = max_seconds
        self.time_range = time_range

    def __len__(self) -> int:
        return sum(len(files) for _, files in self._targets())

    def __iter__(self) -> Iterator[Recording]:
        for building, files in self._targets():
            for path in files:
                yield from self._read(building, path)

    # -- getting the data --------------------------------------------------- #

    @classmethod
    def plan(
        cls,
        *,
        version: str = "raw",
        buildings: list[int] | None = None,
        appliances: list[str] | None = None,
        aggregate: bool = True,
        index_cache: str | Path | None = None,
    ) -> Plan:
        """Which files this configuration would need, without fetching any.

        Args:
            version: ``"raw"`` or ``"preprocessed"``.
            buildings: building numbers. ``None`` takes building 1.
            appliances: appliance file stems. ``None`` takes every appliance in
                the chosen buildings.
            aggregate: include the smart meter.
            index_cache: directory for the archive's parsed listing.

        Returns:
            A :class:`~nilmframe.sources.Plan`.
        """
        from nilmframe.sources import SmartNIALMSource

        return SmartNIALMSource(index_cache=index_cache).plan(
            version=version, buildings=buildings, appliances=appliances, aggregate=aggregate
        )

    @classmethod
    def download(
        cls,
        cache: str | Path,
        *,
        version: str = "raw",
        buildings: list[int] | None = None,
        appliances: list[str] | None = None,
        aggregate: bool = True,
        max_bytes: int | None = None,
        progress: bool = True,
        **kwargs,
    ) -> SmartNIALM:
        """Fetch a subset into ``cache`` and return a reader over it.

        Extraction is coarse here in a way the other corpora are not: the release
        is a ``.7z`` whose members are grouped into twelve solid blocks, so
        reaching one file means decompressing the roughly 3.8 GB block holding
        it. Asking for several files from one building is therefore much cheaper
        per file than asking for one.

        Args:
            cache: directory to fetch into.
            version: ``"raw"`` or ``"preprocessed"``.
            buildings: building numbers.
            appliances: appliance file stems.
            aggregate: include the smart meter.
            max_bytes: refuse a plan larger than this before transferring.
            progress: print one line per file as it lands.
            **kwargs: passed to the constructor, e.g. ``max_seconds``.

        Returns:
            A reader over the fetched subset.
        """
        from nilmframe.sources import materialize

        root = Path(cache).expanduser()
        plan = cls.plan(
            version=version,
            buildings=buildings,
            appliances=appliances,
            aggregate=aggregate,
            index_cache=root / ".index",
        )
        paths, _ = materialize(
            plan, root, workers=1, max_bytes=max_bytes, progress=print if progress else None
        )
        return cls(
            paths["dirpath"],
            buildings=buildings,
            appliances=appliances,
            aggregate=aggregate,
            **kwargs,
        )

    # -- internals ---------------------------------------------------------- #

    def _targets(self) -> list[tuple[int, list[Path]]]:
        out = []
        for directory in sorted(self.dirpath.glob("building_*")):
            digits = "".join(c for c in directory.name if c.isdigit())
            if not digits:
                continue
            number = int(digits)
            if self.buildings is not None and number not in self.buildings:
                continue
            files = []
            for path in sorted(directory.glob("*.h5")):
                stem = path.stem.lower()
                if stem == AGGREGATE_FILE:
                    if self.aggregate:
                        files.append(path)
                elif self.appliances is None or stem in self.appliances:
                    files.append(path)
            if files:
                out.append((number, files))
        return out

    def _read(self, building: int, path: Path) -> Iterator[Recording]:
        try:
            import h5py
        except ImportError as exc:  # pragma: no cover - exercised by the extras matrix
            raise ImportError(
                "reading SmartNIALMeter needs h5py: pip install 'nilmframe[readers]'"
            ) from exc

        with h5py.File(path, "r") as handle:
            table = handle.get("data/table")
            if table is None:
                logger.warning("SmartNIALMeter: %s has no data/table", path.name)
                return
            columns = _column_names(table)
            # The index is datetime64 nanoseconds; the store's clock is seconds.
            stamps = table["index"][:].astype(np.float64) / 1e9
            watts = active_power(columns, table["values_block_0"][:].astype(np.float64))
            if watts is None:
                logger.warning(
                    "SmartNIALMeter: %s reports neither active power nor V/I/PF, only %s",
                    path.name,
                    columns,
                )
                return

        if self.time_range is not None:
            keep = (stamps >= self.time_range[0]) & (stamps <= self.time_range[1])
            stamps, watts = stamps[keep], watts[keep]
        if stamps.size == 0:
            return

        order = np.argsort(stamps)
        stamps, watts = stamps[order], watts[order]
        is_aggregate = path.stem.lower() == AGGREGATE_FILE
        appliance = None if is_aggregate else "_".join(path.stem.lower().split())
        house = f"building_{building:02d}"

        for run, (a, b) in enumerate(split_on_gaps(stamps, self.max_gap_s)):
            series, t0 = forward_fill_to_grid(stamps[a:b], watts[a:b], self.rate_hz)
            if self.max_seconds:
                series = series[: max(2, int(self.max_seconds * self.rate_hz))]
            if series.size < 2:
                continue
            yield Recording(
                dataset=self.dataset,
                house=house,
                # Every channel of one building shares a clock, and the runs are
                # cut on the same rule, so a run number ties an aggregate window
                # to the submeter windows that explain it.
                session=f"run_{run}",
                kind=ChannelKind.MAINS if is_aggregate else ChannelKind.SUBMETER,
                appliance=appliance,
                instance_id=None if is_aggregate else f"{house}:{appliance}",
                signals={"p": series},
                fs=self.rate_hz,
                t0=t0,
                meta={"source": path.name, "building": building},
            )
