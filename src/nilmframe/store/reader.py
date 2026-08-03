"""Reading a canonical store.

``Store`` is a thin, lazy handle: it loads three small parquet tables and nothing
else. Signals are opened on demand as read-only memory maps and cached by handle,
so a ``DataLoader`` worker touching a thousand windows opens each file once and the
operating system decides what stays resident.

The query surface is deliberately small. The predecessor had thirty methods on its
dataset object -- ``submetered``, ``aggregated``, ``drop_rare``, ``filter``,
``groupby``, ``devices``, ``brands``, ``count_components`` -- several of which were
broken by a single wrong property. Here they are one-line pandas expressions on
:attr:`Store.channels`, so there is nothing to get wrong and nothing to maintain.
"""

from __future__ import annotations

import json
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nilmframe.store.schema import STORE_FORMAT_VERSION, ChannelKind

__all__ = ["Store"]


class Store:
    """Read-only handle on a canonical store.

    Args:
        path: store directory, as written by :class:`~nilmframe.store.StoreWriter`.

    Example:
        >>> store = nf.example_store()
        >>> store
        Store('nilmframe-example-store', channels=8, appliances=4, datasets=['example'])
        >>> len(store), store.n_appliances
        (8, 4)
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser()
        if not (self.path / "manifest.json").exists():
            raise FileNotFoundError(f"no store at {self.path} (manifest.json is missing)")

        self.manifest: dict[str, Any] = json.loads((self.path / "manifest.json").read_text())
        version = int(self.manifest.get("format_version", 0))
        if version > STORE_FORMAT_VERSION:
            raise RuntimeError(
                f"store at {self.path} is format v{version}; this build understands "
                f"v{STORE_FORMAT_VERSION}. Upgrade nilmframe."
            )
        self._mmaps: dict[tuple[str, str], np.memmap] = {}

    def __repr__(self) -> str:
        """Identify the store without printing an absolute path.

        The full path is machine-specific, so putting it in the repr makes every
        doctest that prints a store pass on one machine and fail on the next.
        ``store.path`` still has it.
        """
        return (
            f"Store({self.path.name!r}, channels={len(self.channels)}, "
            f"appliances={len(self.appliances)}, datasets={self.manifest.get('datasets')})"
        )

    def __len__(self) -> int:
        """Number of channels in the store -- the row count of :attr:`channels`.

        Example:
            >>> len(store) == len(store.channels)
            True
        """
        return len(self.channels)

    # -- tables ------------------------------------------------------------- #

    @cached_property
    def channels(self) -> pd.DataFrame:
        """One row per measured signal.

        Example:
            >>> store.channels[['channel_id', 'house', 'kind', 'appliance', 'fs']].head(3)
                   channel_id    house      kind appliance      fs
            0  house_1-kettle  house_1  submeter    kettle  6000.0
            1  house_1-fridge  house_1  submeter    fridge  6000.0
            2  house_1-laptop  house_1  submeter    laptop  6000.0
        """
        return pd.read_parquet(self.path / "channels.parquet")

    @cached_property
    def activations(self) -> pd.DataFrame:
        """Annotated appliance on/off intervals, in samples from the channel start.

        Example:
            >>> list(store.activations.columns)
            ['channel_id', 'appliance', 'on', 'off']
        """
        return pd.read_parquet(self.path / "activations.parquet")

    @cached_property
    def appliance_table(self) -> pd.DataFrame:
        """Appliance classes with their on-thresholds and known/unknown status.

        Example:
            >>> store.appliance_table[['appliance', 'on_threshold_w', 'is_known']]
               appliance  on_threshold_w  is_known
            0     fridge            10.0      True
            1     kettle            10.0      True
            2     laptop            10.0      True
            3  microwave            10.0      True
        """
        return pd.read_parquet(self.path / "appliances.parquet")

    # -- derived views ------------------------------------------------------ #

    @cached_property
    def appliances(self) -> list[str]:
        """The label space, in a fixed order. Target vectors index into this.

        Example:
            >>> store.appliances
            ['fridge', 'kettle', 'laptop', 'microwave']
        """
        return sorted(self.appliance_table["appliance"].tolist())

    @property
    def n_appliances(self) -> int:
        return len(self.appliances)

    @cached_property
    def appliance_index(self) -> dict[str, int]:
        """Appliance name to its column in a target vector.

        Example:
            >>> store.appliance_index
            {'fridge': 0, 'kettle': 1, 'laptop': 2, 'microwave': 3}
        """
        return {name: k for k, name in enumerate(self.appliances)}

    @cached_property
    def on_thresholds(self) -> np.ndarray:
        """Per-appliance on-power in watts, ordered like :attr:`appliances`.

        Example:
            >>> store.on_thresholds
            array([10., 10., 10., 10.], dtype=float32)
        """
        table = self.appliance_table.set_index("appliance")["on_threshold_w"]
        return np.asarray([float(table[a]) for a in self.appliances], dtype=np.float32)

    @cached_property
    def known_mask(self) -> np.ndarray:
        """Boolean mask over :attr:`appliances`: is each class in the known label space.

        Example:
            >>> store.known_mask
            array([ True,  True,  True,  True])
        """
        table = self.appliance_table.set_index("appliance")["is_known"]
        return np.asarray([bool(table[a]) for a in self.appliances])

    @property
    def brands(self) -> list[str]:
        """Distinct manufacturers, for brand-disjoint splits.

        Example:
            >>> store.brands
            ['acme', 'globex']
        """
        return sorted(self.channels["brand"].dropna().unique().tolist())

    @property
    def houses(self) -> list[str]:
        """Distinct sites, for house-disjoint splits.

        Example:
            >>> store.houses
            ['house_1', 'house_2']
        """
        return sorted(self.channels["house"].unique().tolist())

    @property
    def datasets(self) -> list[str]:
        """Source corpora present in this store.

        Example:
            >>> store.datasets
            ['example']
        """
        return sorted(self.channels["dataset"].unique().tolist())

    def submeters(self) -> pd.DataFrame:
        """Single-appliance channels.

        The predecessor's equivalent returned an empty dataset for every input,
        because `n_components` counted time samples. Here it is a column lookup.

        Example:
            >>> len(store.submeters())
            6
            >>> sorted(store.submeters().appliance.unique())
            ['fridge', 'kettle', 'laptop', 'microwave']
        """
        return self.channels[self.channels["kind"] == ChannelKind.SUBMETER.value]

    def mains(self) -> pd.DataFrame:
        """Aggregate channels.

        Example:
            >>> list(store.mains().channel_id)
            ['house_1-mains', 'house_2-mains']
        """
        return self.channels[self.channels["kind"] == ChannelKind.MAINS.value]

    def channel(self, channel_id: str) -> pd.Series:
        """One channel's metadata row.

        Raises:
            KeyError: when the channel is not in this store.

        Example:
            >>> row = store.channel('house_1-kettle')
            >>> row.appliance, row.fs, row.n_samples
            ('kettle', np.float64(6000.0), np.int64(12000))
        """
        rows = self.channels[self.channels["channel_id"] == channel_id]
        if rows.empty:
            raise KeyError(f"no channel {channel_id!r} in {self.path}")
        return rows.iloc[0]

    def siblings(self, channel_id: str) -> pd.DataFrame:
        """Other channels recorded on the same clock as ``channel_id``.

        Windows of a mains channel take their per-appliance targets from these.

        Example:
            >>> sorted(store.siblings('house_1-mains').channel_id)
            ['house_1-fridge', 'house_1-kettle', 'house_1-laptop']
        """
        row = self.channel(channel_id)
        same = self.channels[
            (self.channels["house"] == row["house"])
            & (self.channels["session"] == row["session"])
            & (self.channels["dataset"] == row["dataset"])
        ]
        return same[same["channel_id"] != channel_id]

    def activations_for(self, channel_id: str) -> pd.DataFrame:
        """Annotated on/off intervals for one channel.

        Example:
            >>> len(store.activations_for('house_1-mains'))
            0
        """
        return self.activations[self.activations["channel_id"] == channel_id]

    # -- signals ------------------------------------------------------------ #

    def signal_path(self, channel_id: str, quantity: str) -> Path:
        """Where one signal lives on disk.

        Example:
            >>> store.signal_path('house_1-kettle', 'i').name
            'house_1-kettle.i.npy'
        """
        return self.path / "signals" / f"{channel_id}.{quantity}.npy"

    def signal(self, channel_id: str, quantity: str) -> np.memmap:
        """Memory-mapped read-only view of one signal.

        The array is *not* copied: slicing it reads only the pages touched, which
        is what makes windows into a house-year affordable.

        Example:
            >>> i = store.signal('house_1-kettle', 'i')
            >>> type(i).__name__, i.shape, i.dtype
            ('memmap', (12000,), dtype('float32'))
        """
        key = (channel_id, quantity)
        cached = self._mmaps.get(key)
        if cached is not None:
            return cached
        path = self.signal_path(channel_id, quantity)
        if not path.exists():
            raise KeyError(f"channel {channel_id!r} has no quantity {quantity!r} ({path} missing)")
        array = np.load(path, mmap_mode="r")
        self._mmaps[key] = array
        return array

    def has_quantity(self, channel_id: str, quantity: str) -> bool:
        """Does a channel carry this quantity.

        Example:
            >>> store.has_quantity('house_1-kettle', 'i'), store.has_quantity('house_1-kettle', 'p')
            (True, False)
        """
        return quantity in str(self.channel(channel_id)["quantities"]).split(",")

    def read_window(self, channel_id: str, quantity: str, start: int, length: int) -> np.ndarray:
        """Copy ``length`` samples starting at ``start``, zero-padded at the end.

        The copy is deliberate. Slicing the memory map returns a read-only view, and
        ``torch.from_numpy`` on a read-only array yields a tensor whose in-place
        writes are undefined behaviour -- which augmentation would hit immediately.
        Only the window is copied, so this stays proportional to the window and not
        to the channel.

        Example:
            >>> chunk = store.read_window('house_1-kettle', 'i', 0, 128)
            >>> chunk.shape, chunk.flags.writeable
            ((128,), True)
        """
        array = self.signal(channel_id, quantity)
        stop = min(start + length, array.shape[0])
        chunk = np.array(array[start:stop], dtype=np.float32, copy=True)
        if chunk.shape[0] < length:
            chunk = np.pad(chunk, (0, length - chunk.shape[0]))
        return chunk

    def measurement(
        self,
        channel_id: str,
        *,
        start: int = 0,
        seconds: float | None = None,
        samples: int | None = None,
    ):
        """A :class:`~nilmframe.measurement.Measurement` over a window of a channel.

        This is the interactive entry point: an object to dot into and chain on,
        as opposed to the dict of tensors a ``WindowDataset`` yields for training.
        Both read the same memory-mapped signals; only the wrapper differs.

        Args:
            channel_id: which channel.
            start: first sample.
            seconds: window length in seconds. Defaults to the whole channel.
            samples: window length in samples, if you would rather say it that way.

        Example:
            >>> store.measurement('house_1-kettle', seconds=0.5)
            Measurement(waveform raw, 6000Hz, 0.500s, kettle, 2926W)
            >>> store.measurement('house_1-mains', samples=600).n_samples
            600
        """
        from nilmframe.measurement import Measurement

        row = self.channel(channel_id)
        fs = float(row["fs"])
        length = int(row["n_samples"]) - start
        if seconds is not None:
            length = min(length, int(seconds * fs))
        if samples is not None:
            length = min(length, int(samples))
        if length <= 0:
            raise ValueError(f"empty window: start={start} is past the end of {channel_id!r}")

        quantities = str(row["quantities"]).split(",")
        f0 = float(row["f0"])
        common = {
            "t0": float(row["t0"]) + start / fs,
            "source": f"{channel_id}@{start}",
            # A mains channel has no appliance, and pandas spells that NaN --
            # which is truthy, so a bare `if` would carry `nan` through as a name.
            "appliances": (row["appliance"],) if pd.notna(row["appliance"]) else (),
        }
        if "v" in quantities and "i" in quantities:
            return Measurement.from_vi(
                self.read_window(channel_id, "v", start, length),
                self.read_window(channel_id, "i", start, length),
                fs,
                f0=f0 if np.isfinite(f0) else None,
                **common,
            )
        if "p" in quantities:
            return Measurement.from_power(
                self.read_window(channel_id, "p", start, length), fs, **common
            )
        raise KeyError(f"channel {channel_id!r} carries {quantities}, which is not a measurement")

    # -- integrity ---------------------------------------------------------- #

    def verify(self, *, deep: bool = False) -> list[str]:
        """Check the store is intact. Returns a list of problems; empty is good.

        With ``deep=False`` only presence and declared lengths are checked. With
        ``deep=True`` every signal is rehashed and compared against the table --
        slow, but it is what a released benchmark should be able to prove.

        Example:
            >>> store.verify()
            []
            >>> store.verify(deep=True)
            []
        """
        problems: list[str] = []
        import hashlib

        from nilmframe.store.writer import sha256_file

        for _, row in self.channels.iterrows():
            digests = []
            for quantity in str(row["quantities"]).split(","):
                path = self.signal_path(row["channel_id"], quantity)
                if not path.exists():
                    problems.append(f"{row['channel_id']}: missing signal {quantity}")
                    continue
                array = np.load(path, mmap_mode="r")
                if array.shape[0] != row["n_samples"]:
                    problems.append(
                        f"{row['channel_id']}.{quantity}: {array.shape[0]} samples on disk, "
                        f"{row['n_samples']} in the table"
                    )
                if deep:
                    digests.append(sha256_file(path))
            if deep and digests:
                combined = hashlib.sha256("".join(digests).encode()).hexdigest()
                if combined != row["sha256"]:
                    problems.append(f"{row['channel_id']}: content hash mismatch")

        missing = set(self.channels["appliance"].dropna()) - set(self.appliances)
        if missing:
            problems.append(f"appliances present on channels but absent from the table: {missing}")
        return problems

    def describe(self) -> pd.DataFrame:
        """Per-appliance summary: channel count, total duration, instance count.

        Example:
            >>> store.describe()
               appliance  channels  instances  brands     hours
            0     fridge         2          2       2  0.001111
            1     kettle         2          2       2  0.001111
            2     laptop         1          1       1  0.000556
            3  microwave         1          1       1  0.000556
        """
        rows = []
        for appliance in self.appliances:
            sel = self.channels[self.channels["appliance"] == appliance]
            rows.append(
                {
                    "appliance": appliance,
                    "channels": len(sel),
                    "instances": sel["instance_id"].nunique(),
                    "brands": sel["brand"].nunique(),
                    "hours": (
                        float((sel["n_samples"] / sel["fs"]).sum() / 3600) if len(sel) else 0.0
                    ),
                }
            )
        return pd.DataFrame(rows)
