"""The window index.

A dataset is a list of ``(channel, start)`` pairs. Building that list once, as two
numpy arrays, is what makes ``__len__`` and ``__getitem__`` O(1), keeps the dataset
picklable for ``num_workers > 0`` under spawn (macOS and Windows), and keeps memory
flat regardless of how much signal the store holds.

Window lengths are resolved *per channel* from that channel's own ``fs`` and
``f0``, because a store can hold 6 kHz PLAID and 44.1 kHz WHITED side by side and a
view is expressed in physical units (cycles, seconds) rather than samples.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from nilmframe.store import Store

__all__ = ["WindowIndex"]


@dataclass(frozen=True)
class WindowIndex:
    """Flat index of windows over a set of channels.

    Attributes:
        channel_ids: unique channel ids, in index order.
        channel_of: per-window index into ``channel_ids``.
        start: per-window start sample.
        length: per-window length in samples (varies with the channel's rate).

    Example:
        >>> from nilmframe.data import HighFreqView, LowFreqView, WindowDataset, collate_windows
        >>> from nilmframe.data import WindowIndex
        >>> view = HighFreqView(n_cycles=5, cycle_size=64)
        >>> width = view.window_samples
        >>> index = WindowIndex.build(store, ['house_1-kettle'], window_samples=width)
        >>> index
        WindowIndex(windows=12, channels=1)
        >>> len(index)
        12
    """

    channel_ids: list[str]
    channel_of: np.ndarray
    start: np.ndarray
    length: np.ndarray

    def __len__(self) -> int:
        """Number of windows in the index.

        Example:
            >>> from nilmframe.data.windows import WindowIndex
            >>> view = nf.HighFreqView(n_cycles=4, cycle_size=64)
            >>> index = WindowIndex.build(store, ['house_1-kettle'],
            ...                           window_samples=view.window_samples)
            >>> len(index) == len(index.start) == len(index.channel_of)
            True
        """
        return int(self.channel_of.shape[0])

    def __repr__(self) -> str:
        return f"WindowIndex(windows={len(self)}, channels={len(self.channel_ids)})"

    def locate(self, idx: int) -> tuple[str, int, int]:
        """Return ``(channel_id, start, length)`` for window ``idx``.

        Example:
            >>> from nilmframe.data import HighFreqView, LowFreqView, WindowDataset, collate_windows
            >>> from nilmframe.data import WindowIndex
            >>> view = HighFreqView(n_cycles=5, cycle_size=64)
            >>> width = view.window_samples
            >>> index = WindowIndex.build(store, ['house_1-kettle'], window_samples=width)
            >>> index.locate(0)
            ('house_1-kettle', 0, 937)
        """
        return (
            self.channel_ids[int(self.channel_of[idx])],
            int(self.start[idx]),
            int(self.length[idx]),
        )

    def counts(self) -> pd.Series:
        """Windows per channel.

        Example:
            >>> from nilmframe.data import HighFreqView, LowFreqView, WindowDataset, collate_windows
            >>> from nilmframe.data import WindowIndex
            >>> view = HighFreqView(n_cycles=5, cycle_size=64)
            >>> width = view.window_samples
            >>> index = WindowIndex.build(store, ['house_1-kettle'], window_samples=width)
            >>> index.counts()
            house_1-kettle    12
            dtype: int64
        """
        return pd.Series(
            np.bincount(self.channel_of, minlength=len(self.channel_ids)),
            index=self.channel_ids,
        )

    @classmethod
    def build(
        cls,
        store: Store,
        channel_ids: list[str],
        *,
        window_samples,
        stride: float = 1.0,
        drop_last: bool = True,
        max_windows_per_channel: int | None = None,
    ) -> WindowIndex:
        """Enumerate windows.

        Args:
            store: the store to index.
            channel_ids: channels to draw windows from.
            window_samples: callable ``(fs, f0) -> int`` giving the window length
                for a channel. A view supplies this, so the same index code serves
                a 20-cycle high-frequency window and a 60-second low-frequency one.
            stride: hop as a fraction of the window length. ``1.0`` is
                non-overlapping; ``0.5`` overlaps by half.
            drop_last: discard a trailing window that would run past the channel.
                Keeping it is useful for inference over a whole recording, where
                the zero padding is harmless and skipping the tail is not.
            max_windows_per_channel: cap per channel, applied by taking an evenly
                spaced subset rather than the first N, so a cap does not bias
                towards the start of every recording.

        Example:
            >>> from nilmframe.data import HighFreqView, LowFreqView, WindowDataset, collate_windows
            >>> from nilmframe.data import WindowIndex
            >>> view = HighFreqView(n_cycles=5, cycle_size=64)
            >>> ids = store.submeters().channel_id.tolist()
            >>> width = view.window_samples
            >>> dense = WindowIndex.build(store, ids, window_samples=width, stride=0.5)
            >>> sparse = WindowIndex.build(store, ids, window_samples=width, stride=1.0)
            >>> len(dense) > len(sparse)
            True
        """
        if not 0 < stride <= 1.0:
            raise ValueError(f"stride is a fraction of the window, got {stride}")

        rows = store.channels.set_index("channel_id")
        missing = [cid for cid in channel_ids if cid not in rows.index]
        if missing:
            raise KeyError(f"channels not in the store: {missing[:5]}")

        ids: list[str] = []
        chan_of: list[np.ndarray] = []
        starts: list[np.ndarray] = []
        lengths: list[np.ndarray] = []

        for cid in channel_ids:
            row = rows.loc[cid]
            fs = float(row["fs"])
            f0 = float(row["f0"])
            width = int(window_samples(fs, f0 if np.isfinite(f0) else 50.0))
            if width <= 0:
                raise ValueError(f"window_samples returned {width} for channel {cid!r}")

            n = int(row["n_samples"])
            hop = max(1, round(width * stride))
            last = n - width if drop_last else n - 1
            if last < 0:
                # Channel shorter than one window. Keeping a single zero-padded
                # window would fabricate data, so skip it.
                continue

            offsets = np.arange(0, last + 1, hop, dtype=np.int64)
            if max_windows_per_channel and offsets.size > max_windows_per_channel:
                take = np.linspace(0, offsets.size - 1, max_windows_per_channel).round().astype(int)
                offsets = offsets[np.unique(take)]

            ids.append(cid)
            chan_of.append(np.full(offsets.shape, len(ids) - 1, dtype=np.int64))
            starts.append(offsets)
            lengths.append(np.full(offsets.shape, width, dtype=np.int64))

        if not ids:
            return cls([], np.empty(0, np.int64), np.empty(0, np.int64), np.empty(0, np.int64))

        return cls(
            channel_ids=ids,
            channel_of=np.concatenate(chan_of),
            start=np.concatenate(starts),
            length=np.concatenate(lengths),
        )

    def subset(self, keep: np.ndarray) -> WindowIndex:
        """A new index over the windows selected by a boolean or integer array.

        Example:
            >>> from nilmframe.data import HighFreqView, LowFreqView, WindowDataset, collate_windows
            >>> from nilmframe.data import WindowIndex
            >>> view = HighFreqView(n_cycles=5, cycle_size=64)
            >>> width = view.window_samples
            >>> index = WindowIndex.build(store, ['house_1-kettle'], window_samples=width)
            >>> len(index), len(index.subset(np.arange(3)))
            (12, 3)
        """
        return WindowIndex(
            channel_ids=self.channel_ids,
            channel_of=self.channel_of[keep],
            start=self.start[keep],
            length=self.length[keep],
        )
