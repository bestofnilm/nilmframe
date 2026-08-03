"""Writing a canonical store.

``StoreWriter`` is deliberately streaming: a reader yields :class:`Recording`
objects one at a time and each is written and released. The predecessor's
generator held the whole synthetic dataset in a Python list, wrote HDF5 one row at
a time with ``resize(+1)`` per sample, and swept ``gc.get_objects()`` looking for
file handles it had leaked.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nilmframe.store.schema import (
    ACTIVATION_COLUMNS,
    APPLIANCE_COLUMNS,
    CHANNEL_COLUMNS,
    DEFAULT_ON_THRESHOLD_W,
    STORE_FORMAT_VERSION,
    ChannelKind,
    Quantity,
    Recording,
)

__all__ = ["StoreWriter", "sha256_file"]

_CHUNK = 1 << 22  # 4 MiB


def sha256_file(path: Path) -> str:
    """Content hash of a file, streamed so large signals do not enter memory.

    Read in 4 MiB chunks, so hashing a multi-gigabyte channel costs a constant
    amount of memory. This backs
    :meth:`Store.verify(deep=True) <nilmframe.store.Store.verify>` and the store
    fingerprint in the manifest.

    Args:
        path: the file to hash.

    Returns:
        The SHA-256 digest as a hex string.

    Example:
        >>> from nilmframe.store import sha256_file
        >>> sha256_file(store.signal_path('house_1-kettle', 'i'))[:16]
        '83c6cf5a61599ac9'
    """
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def _empty(columns: dict[str, str]) -> pd.DataFrame:
    return pd.DataFrame({name: pd.Series(dtype=dtype) for name, dtype in columns.items()})


class StoreWriter:
    """Build a store on disk.

    Args:
        path: destination directory. Created if absent.
        source: free-form provenance for the manifest (a path, a URL, a DOI).
        appliance_thresholds: per-appliance on-power in watts. Appliances not
            listed get :data:`DEFAULT_ON_THRESHOLD_W`.
        appliance_categories: optional coarse grouping per appliance.
        overwrite: replace an existing store at ``path``.

    Example:
        >>> import tempfile, pathlib
        >>> from nilmframe.store import ChannelKind, Recording, Store, StoreWriter
        >>> dst = pathlib.Path(tempfile.mkdtemp()) / 'demo'
        >>> watts = np.full(100, 2000.0, np.float32)
        >>> rec = Recording(dataset='d', house='h', session='s',
        ...                 kind=ChannelKind.SUBMETER, appliance='kettle',
        ...                 signals={'p': watts}, fs=1.0)
        >>> with StoreWriter(dst, source='demo') as w:
        ...     cid = w.add(rec)
        ...
        >>> cid
        'd-h-s-kettle'
        >>> len(Store(dst)), Store(dst).appliances
        (1, ['kettle'])
    """

    def __init__(
        self,
        path: str | Path,
        *,
        source: str = "",
        appliance_thresholds: dict[str, float] | None = None,
        appliance_categories: dict[str, str] | None = None,
        overwrite: bool = False,
    ) -> None:
        self.path = Path(path).expanduser()
        self.signals_dir = self.path / "signals"
        if self.path.exists() and any(self.path.iterdir()) and not overwrite:
            raise FileExistsError(
                f"{self.path} is not empty; pass overwrite=True to replace the store"
            )
        self.signals_dir.mkdir(parents=True, exist_ok=True)

        self.source = source
        self.appliance_thresholds = dict(appliance_thresholds or {})
        self.appliance_categories = dict(appliance_categories or {})

        self._channels: list[dict[str, Any]] = []
        self._activations: list[dict[str, Any]] = []
        self._appliances: set[str] = set()
        self._seen_ids: set[str] = set()
        self._closed = False

    # -- context manager ---------------------------------------------------- #

    def __enter__(self) -> StoreWriter:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None:
            self.close()

    # -- writing ------------------------------------------------------------ #

    def add(self, recording: Recording, channel_id: str | None = None) -> str:
        """Write one channel. Returns its assigned ``channel_id``.

        Example:
            >>> import tempfile, pathlib
            >>> from nilmframe.store import ChannelKind, Recording, StoreWriter
            >>> w = StoreWriter(pathlib.Path(tempfile.mkdtemp()) / 's')
            >>> rec = Recording(dataset='d', house='h', session='s',
            ...                 kind=ChannelKind.SUBMETER, appliance='kettle',
            ...                 signals={'p': np.ones(10, np.float32)}, fs=1.0)
            >>> w.add(rec)
            'd-h-s-kettle'
            >>> w.add(rec, channel_id='explicit')
            'explicit'
        """
        if self._closed:
            raise RuntimeError("writer is closed")

        channel_id = channel_id or self._make_id(recording)
        if channel_id in self._seen_ids:
            raise ValueError(f"duplicate channel_id {channel_id!r}")
        self._seen_ids.add(channel_id)

        f0 = recording.f0
        if f0 is None and Quantity.VOLTAGE.value in recording.signals:
            import torch

            from nilmframe.nn.align import estimate_f0

            v = torch.from_numpy(recording.signals[Quantity.VOLTAGE.value])
            f0 = float(estimate_f0(v, recording.fs))

        digests = []
        for quantity, array in sorted(recording.signals.items()):
            dest = self.signals_dir / f"{channel_id}.{quantity}.npy"
            np.save(dest, array, allow_pickle=False)
            digests.append(sha256_file(dest))

        self._channels.append(
            {
                "channel_id": channel_id,
                "dataset": recording.dataset,
                "house": str(recording.house),
                "session": str(recording.session),
                "kind": ChannelKind(recording.kind).value,
                "appliance": recording.appliance,
                "brand": recording.brand,
                "instance_id": recording.resolved_instance_id(),
                "fs": float(recording.fs),
                "f0": float(f0) if f0 is not None else float("nan"),
                "t0": float(recording.t0),
                "n_samples": int(recording.n_samples),
                "quantities": ",".join(recording.quantities),
                "sha256": hashlib.sha256("".join(digests).encode()).hexdigest(),
            }
        )

        if recording.appliance:
            self._appliances.add(recording.appliance)
        for act in recording.activations:
            self._appliances.add(act.appliance)
            self._activations.append(
                {
                    "channel_id": channel_id,
                    "appliance": act.appliance,
                    "on": int(act.on),
                    "off": int(act.off),
                }
            )
        return channel_id

    def extend(self, recordings) -> list[str]:
        """Write an iterable of recordings.

        Example:
            >>> import tempfile, pathlib
            >>> from nilmframe.store import ChannelKind, Recording, StoreWriter
            >>> w = StoreWriter(pathlib.Path(tempfile.mkdtemp()) / 's')
            >>> def make(k):
            ...     return Recording(dataset='d', house='h', session=str(k),
            ...                      kind=ChannelKind.SUBMETER, appliance='kettle',
            ...                      signals={'p': np.ones(10, np.float32)}, fs=1.0)
            ...
            >>> w.extend(make(k) for k in range(3))
            ['d-h-0-kettle', 'd-h-1-kettle', 'd-h-2-kettle']
        """
        return [self.add(rec) for rec in recordings]

    def close(self) -> None:
        """Write the metadata tables and the manifest."""
        if self._closed:
            return

        channels = (
            pd.DataFrame(self._channels) if self._channels else _empty(CHANNEL_COLUMNS)
        ).astype({k: v for k, v in CHANNEL_COLUMNS.items() if k not in ("appliance", "brand")})
        activations = (
            pd.DataFrame(self._activations) if self._activations else _empty(ACTIVATION_COLUMNS)
        ).astype(ACTIVATION_COLUMNS)

        appliances = pd.DataFrame(
            {
                "appliance": sorted(self._appliances),
                "on_threshold_w": [
                    float(self.appliance_thresholds.get(a, DEFAULT_ON_THRESHOLD_W))
                    for a in sorted(self._appliances)
                ],
                "category": [
                    self.appliance_categories.get(a, "unknown") for a in sorted(self._appliances)
                ],
                "is_known": [True] * len(self._appliances),
            }
        )
        if appliances.empty:
            appliances = _empty(APPLIANCE_COLUMNS)
        appliances = appliances.astype(APPLIANCE_COLUMNS)

        channels.to_parquet(self.path / "channels.parquet", index=False)
        activations.to_parquet(self.path / "activations.parquet", index=False)
        appliances.to_parquet(self.path / "appliances.parquet", index=False)

        manifest = {
            "format_version": STORE_FORMAT_VERSION,
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "source": self.source,
            "datasets": sorted(channels["dataset"].unique().tolist()) if len(channels) else [],
            "n_channels": len(channels),
            "n_appliances": len(appliances),
            "total_samples": int(channels["n_samples"].sum()) if len(channels) else 0,
            # One hash over every channel's content hash: a store fingerprint that
            # can be quoted in a paper without rehashing terabytes of signal.
            "content_sha256": hashlib.sha256(
                "".join(sorted(channels["sha256"].tolist())).encode()
            ).hexdigest(),
        }
        (self.path / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        self._closed = True

    # -- internals ---------------------------------------------------------- #

    def _make_id(self, rec: Recording) -> str:
        base = "-".join(
            str(p)
            for p in (rec.dataset, rec.house, rec.session, rec.appliance or rec.kind.value)
            if p
        )
        base = "".join(c if c.isalnum() or c in "-_." else "_" for c in base)
        candidate, n = base, 1
        while candidate in self._seen_ids:
            n += 1
            candidate = f"{base}#{n}"
        return candidate
