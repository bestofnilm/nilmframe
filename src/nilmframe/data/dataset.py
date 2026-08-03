"""``torch.utils.data.Dataset`` over a canonical store.

One item is one window: the view's tensors plus targets plus provenance. Indexing
touches only the pages of the memory-mapped signals it needs, so a dataset over a
house-year costs the same as one over a minute.

Target construction, and why it is shaped this way
--------------------------------------------------
Three quantities come out, and they are deliberately not one vector:

``presence`` ``(K,)``
    Is appliance *k* on during this window, by its own threshold from the store's
    appliance table. Trained with ``BCEWithLogits``.
``power`` ``(K,)``
    Watts attributable to appliance *k*. Trained with a masked regression loss.
``power_mask`` ``(K,)``
    Whether ``power[k]`` is *known*. An aggregate recording annotated only with
    on/off times knows presence but not per-appliance power, and inventing a
    number there would train the model on fiction.

``p_total`` comes from the view -- computed from the input signal -- and never
from the labels. The predecessor divided predictions by ``Y_true.sum(1)`` at
scoring time, which handed the model a ground-truth total it could not have at
inference and inflated every published number.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from nilmframe.data.views import View
from nilmframe.data.windows import WindowIndex
from nilmframe.store import DEFAULT_ON_THRESHOLD_W, ChannelKind, Store

__all__ = ["WindowDataset", "collate_windows"]

_TARGETS = ("presence", "power")


class WindowDataset(Dataset):
    """Windows over a set of channels, rendered through a view.

    Args:
        store: the store to read.
        channels: channel ids to draw from, typically one fold of a
            :class:`~nilmframe.data.splits.Split`.
        view: how a window becomes model input.
        targets: which targets to build. ``()`` for inference.
        stride: window hop as a fraction of the window length.
        augment: callable applied to the item dict before it is returned. Receives
            ``(item, rng)`` so augmentation is seeded per item rather than from
            global state.
        seed: base seed for augmentation.
        drop_last: discard a trailing partial window.
        max_windows_per_channel: cap windows per channel, evenly spaced.
        appliances: override the label space. Defaults to the store's.

    Example:
        >>> from nilmframe.data import HighFreqView, LowFreqView, WindowDataset, collate_windows
        >>> ids = store.submeters().channel_id.tolist()
        >>> ds = WindowDataset(store, ids, view=HighFreqView(n_cycles=5, cycle_size=64))
        >>> len(ds), ds.n_appliances
        (72, 4)
        >>> sorted(ds[0])[:5]
        ['channel', 'cycle_mask', 'i', 'p_total', 'power']
        >>> ds.describe()['view']
        'highfreq'
    """

    def __init__(
        self,
        store: Store,
        channels: list[str],
        *,
        view: View,
        targets: tuple[str, ...] = _TARGETS,
        stride: float = 1.0,
        augment=None,
        seed: int = 0,
        drop_last: bool = True,
        max_windows_per_channel: int | None = None,
        appliances: list[str] | None = None,
        unknown_appliances: Sequence[str] = (),
    ) -> None:
        unknown = set(targets) - set(_TARGETS)
        if unknown:
            raise ValueError(f"unknown targets {sorted(unknown)}; expected a subset of {_TARGETS}")

        self.store = store
        self.view = view
        self.targets = tuple(targets)
        self.seed = int(seed)
        self.augment = augment
        self.epoch = 0

        # Open set: held-out classes get no column in the label space at all, so
        # the model has nowhere to put them and must learn to say "not mine".
        # Giving them a column and then ignoring it at evaluation would leak the
        # existence of the class into training.
        self.unknown_appliances = set(unknown_appliances)
        base = list(appliances) if appliances else list(store.appliances)
        missing = self.unknown_appliances - set(base)
        if missing:
            raise ValueError(f"unknown_appliances not in the label space: {sorted(missing)}")
        self.appliances = [a for a in base if a not in self.unknown_appliances]
        self._appliance_index = {name: k for k, name in enumerate(self.appliances)}
        thresholds = store.appliance_table.set_index("appliance")["on_threshold_w"]
        self.on_thresholds = np.asarray(
            [float(thresholds.get(a, 10.0)) for a in self.appliances], dtype=np.float32
        )

        # Drop channels this view cannot render. A UK-DALE split contains a
        # house's waveform mains and its low-rate submeters together; the
        # high-frequency arm wants the former and the low-frequency arm either.
        rows = store.channels.set_index("channel_id")
        usable = [
            cid
            for cid in channels
            if view.supports(
                set(str(rows.loc[cid, "quantities"]).split(",")),
                float(rows.loc[cid, "fs"]),
                float(rows.loc[cid, "f0"]) if np.isfinite(rows.loc[cid, "f0"]) else 50.0,
            )
        ]
        self.skipped_channels = [cid for cid in channels if cid not in set(usable)]

        self.index = WindowIndex.build(
            store,
            usable,
            window_samples=view.window_samples,
            stride=stride,
            drop_last=drop_last,
            max_windows_per_channel=max_windows_per_channel,
        )

        # Channel metadata is hit once per item; a dict lookup beats a dataframe
        # filter by orders of magnitude inside a DataLoader worker.
        self._meta: dict[str, dict[str, Any]] = {
            cid: rows.loc[cid].to_dict() for cid in self.index.channel_ids
        }
        self._activations: dict[str, list[tuple[str, int, int]]] = {}
        for cid in self.index.channel_ids:
            acts = store.activations_for(cid)
            self._activations[cid] = list(
                zip(acts["appliance"], acts["on"], acts["off"], strict=True)
            )

        # Submeters sharing a clock with each mains channel. This is what turns an
        # aggregate window's targets from "which appliances were annotated as on"
        # into "how many watts each one actually drew" -- and it is the reason the
        # store records an absolute t0 per channel.
        self._siblings: dict[str, list[dict[str, Any]]] = {}
        for cid in self.index.channel_ids:
            if str(self._meta[cid]["kind"]) != ChannelKind.MAINS.value:
                continue
            found = []
            for _, row in store.siblings(cid).iterrows():
                appliance = row["appliance"]
                if str(row["kind"]) != ChannelKind.SUBMETER.value or "p" not in str(
                    row["quantities"]
                ).split(","):
                    continue
                held_out = appliance in self.unknown_appliances
                if not held_out and appliance not in self._appliance_index:
                    continue
                found.append(
                    {
                        "channel_id": row["channel_id"],
                        "appliance": appliance,
                        # Held-out classes are kept in the sibling list rather than
                        # dropped, so a window can still be *flagged* as containing
                        # one. Dropping them would make an open-set evaluation
                        # silently score every unknown window as known-and-quiet.
                        "index": -1 if held_out else self._appliance_index[appliance],
                        "unknown": held_out,
                        "fs": float(row["fs"]),
                        "t0": float(row["t0"]),
                        "n_samples": int(row["n_samples"]),
                    }
                )
            self._siblings[cid] = found

    def __len__(self) -> int:
        """Number of windows across every channel in the split fold.

        Example:
            >>> ids = store.submeters().channel_id.tolist()
            >>> ds = nf.WindowDataset(store, ids,
            ...                       view=nf.HighFreqView(n_cycles=4, cycle_size=64))
            >>> len(ds) == len(ds.index)
            True
        """
        return len(self.index)

    def __repr__(self) -> str:
        return (
            f"WindowDataset(windows={len(self)}, channels={len(self.index.channel_ids)}, "
            f"view={self.view.describe()}, targets={self.targets})"
        )

    @property
    def n_appliances(self) -> int:
        return len(self.appliances)

    def set_epoch(self, epoch: int) -> None:
        """Vary the augmentation stream between epochs, reproducibly.

        Seeding is ``(seed, epoch, index)``, so a rerun of epoch 7 sees exactly the
        mixtures epoch 7 saw the first time.

        Example:
            >>> from nilmframe.data import HighFreqView, LowFreqView, WindowDataset, collate_windows
            >>> ids = store.submeters().channel_id.tolist()
            >>> ds = WindowDataset(store, ids, view=HighFreqView(n_cycles=5, cycle_size=64))
            >>> ds.epoch
            0
            >>> ds.set_epoch(3)
            >>> ds.epoch
            3
        """
        self.epoch = int(epoch)

    def raw_item(self, idx: int) -> dict[str, Tensor]:
        """One window with no augmentation applied.

        Augmentations that draw partner windows -- :class:`MixAggregate` -- go
        through this. Calling ``__getitem__`` instead would re-enter the
        augmentation and recurse.

        Example:
            >>> from nilmframe.data import HighFreqView, LowFreqView, WindowDataset, collate_windows
            >>> ids = store.submeters().channel_id.tolist()
            >>> ds = WindowDataset(store, ids, view=HighFreqView(n_cycles=5, cycle_size=64))
            >>> sorted(ds.raw_item(0))[:4]
            ['channel', 'cycle_mask', 'i', 'p_total']
            >>> sorted(ds.raw_item(0))[4:]
            ['power', 'power_mask', 'presence', 'start', 'v']
        """
        channel_id, start, length = self.index.locate(idx)
        meta = self._meta[channel_id]
        fs, f0 = float(meta["fs"]), float(meta["f0"])

        available = set(str(meta["quantities"]).split(","))
        wanted = self.view.required_quantities(available)
        raw = {
            q: torch.from_numpy(self.store.read_window(channel_id, q, start, length))
            for q in wanted
        }

        item: dict[str, Any] = dict(self.view(raw, fs=fs, f0=f0))
        item["channel"] = channel_id
        item["start"] = start

        if self.targets:
            presence, power, power_mask, is_unknown = self._build_targets(
                channel_id, meta, item, start, length
            )
            if "presence" in self.targets:
                item["presence"] = presence
            if "power" in self.targets:
                item["power"] = power
                item["power_mask"] = power_mask
            if self.unknown_appliances:
                item["is_unknown"] = is_unknown
        return item

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        """One window as a dict of tensors, ready for :func:`collate_windows`.

        Signals come first (``v``, ``i`` or ``p``, plus ``p_total``), then the
        targets the fold's annotations support (``presence``, ``power``,
        ``power_mask``). Transforms run last, so a mixing transform sees the same
        keys the model will.

        Args:
            idx: window position, ``0 <= idx < len(self)``.

        Returns:
            A dict of tensors. The exact keys depend on the view and on what the
            store knows about the channel -- ask a real item rather than assuming.

        Example:
            >>> ids = store.submeters().channel_id.tolist()
            >>> ds = nf.WindowDataset(store, ids,
            ...                       view=nf.HighFreqView(n_cycles=4, cycle_size=64))
            >>> item = ds[0]
            >>> tuple(item['v'].shape)
            (4, 64)
            >>> item['presence'].dtype
            torch.float32
        """
        item = self.raw_item(idx)
        if self.augment is not None:
            rng = np.random.default_rng((self.seed, self.epoch, idx))
            item = self.augment(item, rng, self)
        return item

    # -- targets ------------------------------------------------------------ #

    def _build_targets(
        self,
        channel_id: str,
        meta: dict,
        rendered: dict[str, Tensor],
        start: int,
        length: int,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        k = self.n_appliances
        presence = torch.zeros(k)
        power = torch.zeros(k)
        power_mask = torch.zeros(k, dtype=torch.bool)
        is_unknown = torch.zeros(())

        if str(meta["kind"]) == ChannelKind.SUBMETER.value:
            appliance = meta["appliance"]
            if appliance in self.unknown_appliances:
                # Out of vocabulary: no column to put it in, and the power it draws
                # is unattributable, so nothing is known about this window except
                # that it contains something the label space does not cover.
                return presence, power, power_mask, torch.ones(())

            j = self._appliance_index.get(appliance)
            if j is not None:
                # The view's own measurement of the window it rendered, so the
                # target and p_total describe the same signal exactly.
                watts = rendered["p_total"]
                power[j] = watts
                presence[j] = float(watts > self.on_thresholds[j])
                # Every appliance's power is known here, not just this one: the
                # recording contains a single appliance, so every *other* class
                # draws exactly zero watts from it. Masking those out would throw
                # away the only negative examples a submetered dataset provides.
                power_mask[:] = True
            return presence, power, power_mask, is_unknown

        # Aggregate channel, two sources of truth in order of strength.
        #
        # 1. Sibling submeters on the same clock (UK-DALE): the window's absolute
        #    time span is intersected with each submeter, giving real watts. This
        #    is the only way an aggregate window gets *power* targets.
        siblings = self._siblings.get(channel_id, ())
        if siblings:
            fs, t0 = float(meta["fs"]), float(meta["t0"])
            begin, end = t0 + start / fs, t0 + (start + length) / fs
            for sibling in siblings:
                watts = self._sibling_power(sibling, begin, end)
                if watts is None:
                    continue
                if sibling["unknown"]:
                    # A held-out appliance drawing real power makes this window
                    # out-of-vocabulary, whatever the known classes are doing.
                    if float(watts) > DEFAULT_ON_THRESHOLD_W:
                        is_unknown = torch.ones(())
                    continue
                j = sibling["index"]
                power[j] = watts
                power_mask[j] = True
                presence[j] = float(watts > self.on_thresholds[j])

        # 2. Annotated on/off intervals (PLAID aggregates): presence only. Power
        #    stays masked rather than invented -- an annotation says an appliance
        #    was running, not how hard.
        stop = start + length
        for appliance, on, off in self._activations[channel_id]:
            overlap = min(stop, int(off)) - max(start, int(on))
            if overlap <= 0.5 * length:  # a sliver, not a majority of the window
                continue
            if appliance in self.unknown_appliances:
                is_unknown = torch.ones(())
                continue
            j = self._appliance_index.get(appliance)
            if j is not None:
                presence[j] = 1.0
        return presence, power, power_mask, is_unknown

    def _sibling_power(self, sibling: dict, begin: float, end: float) -> Tensor | None:
        """Mean watts a sibling submeter drew over an absolute time span."""
        fs, t0 = sibling["fs"], sibling["t0"]
        first = int(np.floor((begin - t0) * fs))
        last = int(np.ceil((end - t0) * fs))
        first, last = max(0, first), min(sibling["n_samples"], last)
        if last - first < 1:
            return None  # the submeter was not recording over this span
        series = self.store.read_window(sibling["channel_id"], "p", first, last - first)
        return torch.as_tensor(float(series.mean()))

    # -- convenience -------------------------------------------------------- #

    def measurement(self, idx: int):
        """Window ``idx`` as a :class:`~nilmframe.measurement.Measurement`.

        The same window ``__getitem__`` returns, wrapped for inspection rather
        than for a ``DataLoader``.

        Example:
            >>> from nilmframe.data import HighFreqView, LowFreqView, WindowDataset, collate_windows
            >>> ids = store.submeters().channel_id.tolist()
            >>> ds = WindowDataset(store, ids, view=HighFreqView(n_cycles=5, cycle_size=64))
            >>> ds.measurement(0)
            Measurement(waveform 5x64, 6000Hz, 0.053s, 2925W)
        """
        from nilmframe.measurement import Measurement

        channel_id, _, _ = self.index.locate(idx)
        meta = self._meta[channel_id]
        item = self.raw_item(idx)
        return Measurement.from_item(item, fs=float(meta["fs"]), f0=float(meta["f0"]))

    def window_counts(self):
        """Windows per channel, as a pandas Series.

        Example:
            >>> from nilmframe.data import HighFreqView, LowFreqView, WindowDataset, collate_windows
            >>> ids = store.submeters().channel_id.tolist()
            >>> ds = WindowDataset(store, ids, view=HighFreqView(n_cycles=5, cycle_size=64))
            >>> ds.window_counts().head(3)
            house_1-kettle    12
            house_1-fridge    12
            house_1-laptop    12
            dtype: int64
        """
        return self.index.counts()

    def describe(self) -> dict:
        """A summary of the dataset and the view behind it.

        Example:
            >>> from nilmframe.data import HighFreqView, LowFreqView, WindowDataset, collate_windows
            >>> ids = store.submeters().channel_id.tolist()
            >>> ds = WindowDataset(store, ids, view=HighFreqView(n_cycles=5, cycle_size=64))
            >>> ds.describe()['windows'], ds.describe()['view']
            (72, 'highfreq')
        """
        return {
            "windows": len(self),
            "channels": len(self.index.channel_ids),
            "appliances": self.appliances,
            "targets": list(self.targets),
            "unknown_appliances": sorted(self.unknown_appliances),
            **self.view.describe(),
        }


def collate_windows(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Stack a list of items, keeping string provenance as a plain list.

    ``torch.utils.data.default_collate`` turns a list of strings into a list, which
    is fine, but it chokes on mixed types elsewhere; being explicit here also keeps
    ``channel`` and ``start`` usable for per-window reporting.

    Args:
        batch: items from a :class:`WindowDataset`. Every item must carry the same
            keys -- an augmentation that adds one conditionally is a bug, and this
            says so rather than failing obscurely inside torch.

    Returns:
        The same keys, tensors stacked along a new leading batch axis, strings left
        as a list.

    Raises:
        ValueError: when the items disagree about which keys they have.

    Example:
        >>> from nilmframe.data import HighFreqView, LowFreqView, WindowDataset, collate_windows
        >>> ids = store.submeters().channel_id.tolist()
        >>> ds = WindowDataset(store, ids, view=HighFreqView(n_cycles=5, cycle_size=64))
        >>> batch = collate_windows([ds[0], ds[1]])
        >>> tuple(batch['i'].shape)
        (2, 5, 64)
        >>> tuple(batch['presence'].shape)
        (2, 4)
    """
    keys = set(batch[0])
    for item in batch[1:]:
        if set(item) != keys:
            difference = keys.symmetric_difference(item)
            raise ValueError(
                f"items in a batch must carry the same keys; {sorted(difference)} "
                "is present on some items and not others. An augmentation that adds "
                "a key conditionally must add it unconditionally."
            )

    out: dict[str, Any] = {}
    for key in batch[0]:
        values = [item[key] for item in batch]
        if isinstance(values[0], torch.Tensor):
            out[key] = torch.stack(values)
        elif isinstance(values[0], (int, float, np.integer, np.floating)):
            out[key] = torch.as_tensor(values)
        else:
            out[key] = values
    return out
