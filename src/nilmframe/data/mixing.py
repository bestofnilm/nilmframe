"""Synthetic aggregation as augmentation.

Submetered recordings are building blocks: add the currents of several appliances
measured under the same mains and you have an aggregate window whose per-appliance
truth you know exactly. That idea is the predecessor's ``FLEPGen`` and it is worth
keeping. What changes is *when* it happens.

``FLEPGen`` was a batch job. It materialised a fixed set of mixtures into an HDF5
file one row at a time with ``resize(+1)`` per sample, stored currents as float16
and then repaired the resulting rounding with an ``adj16`` fixup, scanned a Python
list to reject duplicate combinations (O(n) per sample, O(n^2) overall), and swept
``gc.get_objects()`` looking for file handles it had leaked.

Mixing in the dataset instead makes all of that go away, and buys something the
batch job could not offer: a different mixture every epoch. Seeding is per
``(seed, epoch, index)``, so the stream is reproducible without being fixed.

Exactness matters here and is cheap to get. Every component's power target is
recomputed under the *base window's voltage*, the same voltage the mixed current
is measured against, so ``sum(power) == p_total`` holds to floating-point rather
than approximately. ``tests/test_mixing.py`` pins the relative error below 1e-6 --
float32 epsilon. The bound is relative because it cannot be absolute: these are
kilowatt loads, and 7 kW times 2^-23 is already about 1e-3 W.

A frozen benchmark still wants a fixed set of mixtures on disk; that is what
:func:`materialize` is for.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from torch import Tensor

__all__ = ["Compose", "GainJitter", "MixAggregate", "VoltageJitter", "materialize"]


class Compose:
    """Chain augmentations, each receiving the same rng.

    Args:
        transforms: callables taking ``(item, rng, dataset)`` and returning an item.

    Args:
        transforms: callables taking ``(item, rng, dataset)`` and returning an item.
            They share one generator, so the whole chain is reproducible from the
            dataset's ``(seed, epoch, index)``.

    Example:
        >>> pipeline = nf.Compose([nf.VoltageJitter(0.02), nf.GainJitter(0.05)])
        >>> pipeline
        Compose(VoltageJitter(sigma=0.02), GainJitter(sigma=0.05))
    """

    def __init__(self, transforms: Sequence) -> None:
        self.transforms = list(transforms)

    def __call__(self, item: dict, rng: np.random.Generator, dataset) -> dict:
        for transform in self.transforms:
            item = transform(item, rng, dataset)
        return item

    def __repr__(self) -> str:
        inner = ", ".join(repr(t) for t in self.transforms)
        return f"Compose({inner})"


def _active_power(v: Tensor, i: Tensor) -> Tensor:
    return (v * i).mean()


class MixAggregate:
    """Superpose several submetered windows into one synthetic aggregate.

    Args:
        k: inclusive range of components per mixture. ``(1, 5)`` leaves some
            windows unmixed, which matters -- a model trained only on mixtures of
            three or more never learns what a single appliance looks like.
        p: probability of mixing a given window at all.
        same_rate_only: only superpose windows from channels sharing a sampling
            rate. Superposition assumes a common voltage; two recordings made on
            different rigs at different rates do not have one, and mixing them
            produces a signature no meter would ever see.
        max_tries: attempts to find a compatible partner before giving up and
            returning the window unmixed.

    Note:
        Presence targets combine by max and power targets by sum. ``power_mask``
        combines by AND: a mixture is only as knowable as its least-known part.

    Example:
        >>> from nilmframe.data import HighFreqView, LowFreqView, WindowDataset, collate_windows
        >>> ids = store.submeters().channel_id.tolist()
        >>> ds = WindowDataset(store, ids, view=HighFreqView(n_cycles=5, cycle_size=64))
        >>> mixed = WindowDataset(store, ids, view=HighFreqView(n_cycles=5, cycle_size=64),
        ...                       augment=nf.MixAggregate(k=(1, 3), p=1.0), seed=0)
        >>> item = mixed[0]
        >>> item['n_components']
        2
        >>> bool(torch.allclose(item['power'].sum(), item['p_total'], rtol=1e-5))
        True
    """

    def __init__(
        self,
        k: tuple[int, int] = (1, 4),
        p: float = 1.0,
        *,
        same_rate_only: bool = True,
        max_tries: int = 8,
    ) -> None:
        if k[0] < 1 or k[1] < k[0]:
            raise ValueError(f"k must be an increasing range with k[0] >= 1, got {k}")
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p must be a probability, got {p}")
        self.k = (int(k[0]), int(k[1]))
        self.p = float(p)
        self.same_rate_only = same_rate_only
        self.max_tries = max_tries
        self._pools: dict[Any, np.ndarray] | None = None

    def __repr__(self) -> str:
        return f"MixAggregate(k={self.k}, p={self.p})"

    def _pool_for(self, dataset, rate: float) -> np.ndarray:
        """Window indices eligible as mixture partners, grouped by sampling rate."""
        if self._pools is None:
            pools: dict[Any, list[int]] = {}
            meta = dataset._meta
            for idx in range(len(dataset)):
                channel_id = dataset.index.channel_ids[int(dataset.index.channel_of[idx])]
                row = meta[channel_id]
                if str(row["kind"]) != "submeter":
                    continue  # only isolated appliances superpose meaningfully
                key = round(float(row["fs"]), 3) if self.same_rate_only else 0
                pools.setdefault(key, []).append(idx)
            self._pools = {k: np.asarray(v, dtype=np.int64) for k, v in pools.items()}
        return self._pools.get(round(rate, 3) if self.same_rate_only else 0, np.empty(0, np.int64))

    def __call__(self, item: dict, rng: np.random.Generator, dataset) -> dict:
        # `n_components` is stamped on every item, mixed or not: a batch that
        # mixes some windows and not others would otherwise have items with
        # different keys, and collation fails on the first one that disagrees.
        unmixed = {**item, "n_components": 1}
        if rng.random() > self.p:
            return unmixed

        n_components = int(rng.integers(self.k[0], self.k[1] + 1))
        if n_components <= 1:
            return unmixed

        channel = item["channel"]
        rate = float(dataset._meta[channel]["fs"])
        pool = self._pool_for(dataset, rate)
        if pool.size < 2:
            return unmixed

        item = dict(item)
        mixed_from = [channel]

        for _ in range(n_components - 1):
            partner = self._draw(dataset, pool, rng, mixed_from)
            if partner is None:
                break
            item = self._superpose(item, partner)
            mixed_from.append(partner["channel"])

        item["n_components"] = len(mixed_from)
        return item

    def _draw(self, dataset, pool: np.ndarray, rng: np.random.Generator, used: list[str]):
        for _ in range(self.max_tries):
            other = dataset.raw_item(int(pool[rng.integers(pool.size)]))
            # Two windows of the same channel are the same appliance; adding them
            # would fabricate a two-kettle household from one kettle.
            if other["channel"] not in used:
                return other
        return None

    @staticmethod
    def _superpose(base: dict, other: dict) -> dict:
        out = dict(base)

        if "i" in base and "v" in base:
            voltage = base["v"]
            current = base["i"] + other["i"]
            out["i"] = current
            # Recompute the aggregate from the constructed signal, so p_total is
            # what a meter would measure rather than a running sum of estimates.
            out["p_total"] = _active_power(voltage, current)
            # And recompute the partner's contribution under *this* voltage, so
            # the parts sum to the whole exactly.
            contribution = _active_power(voltage, other["i"])
        else:
            out["p"] = base["p"] + other["p"]
            out["p_total"] = out["p"].mean()
            contribution = other["p"].mean()

        if "presence" in base and "presence" in other:
            out["presence"] = torch.maximum(base["presence"], other["presence"])

        if "power" in base and "power" in other:
            share = other["power"].sum().clamp_min(1e-12)
            # Rescale the partner's per-appliance powers so they add up to what it
            # actually contributes under this voltage.
            out["power"] = base["power"] + other["power"] * (contribution / share)
            if "power_mask" in base and "power_mask" in other:
                out["power_mask"] = base["power_mask"] & other["power_mask"]

        if "is_unknown" in base or "is_unknown" in other:
            out["is_unknown"] = torch.maximum(
                base.get("is_unknown", torch.zeros(())),
                other.get("is_unknown", torch.zeros(())),
            )
        return out


class VoltageJitter:
    """Scale the mains voltage, simulating supply variation between homes.

    Current is scaled inversely so the load's power is unchanged: a kettle draws
    the same watts on a slightly high supply, it just draws less current. Scaling
    voltage alone would silently relabel every power target.

    Args:
        sigma: standard deviation of the multiplicative jitter. ``0.02`` is about
            the spread of real mains voltage between homes.

    Example:
        >>> from nilmframe.data import HighFreqView, LowFreqView, WindowDataset, collate_windows
        >>> ids = store.submeters().channel_id.tolist()
        >>> ds = WindowDataset(store, ids, view=HighFreqView(n_cycles=5, cycle_size=64))
        >>> jittered = WindowDataset(store, ids, view=HighFreqView(n_cycles=5, cycle_size=64),
        ...                          augment=nf.VoltageJitter(sigma=0.1), seed=1)
        >>> plain = WindowDataset(store, ids, view=HighFreqView(n_cycles=5, cycle_size=64))
        >>> a, b = plain[0], jittered[0]
        >>> bool(torch.allclose(a['v'], b['v']))
        False
        >>> bool(torch.allclose(a['p_total'], b['p_total'], rtol=1e-3))
        True
    """

    def __init__(self, sigma: float = 0.02) -> None:
        self.sigma = sigma

    def __repr__(self) -> str:
        return f"VoltageJitter(sigma={self.sigma})"

    def __call__(self, item: dict, rng: np.random.Generator, dataset) -> dict:
        if "v" not in item or self.sigma <= 0:
            return item
        scale = float(1.0 + rng.normal(0.0, self.sigma))
        out = dict(item)
        out["v"] = item["v"] * scale
        out["i"] = item["i"] / scale
        return out


class GainJitter:
    """Scale the whole window, simulating measurement-chain calibration error.

    Power targets scale with the square of the gain, because both voltage and
    current are scaled. Keeping the targets consistent is the entire point: an
    augmentation that changes the signal without changing the label is a bug, not
    a regulariser.

    Args:
        sigma: standard deviation of the log-normal gain. ``0.05`` is roughly the
            calibration spread between measurement rigs.

    Example:
        >>> nf.GainJitter(sigma=0.05)
        GainJitter(sigma=0.05)
    """

    def __init__(self, sigma: float = 0.05) -> None:
        self.sigma = sigma

    def __repr__(self) -> str:
        return f"GainJitter(sigma={self.sigma})"

    def __call__(self, item: dict, rng: np.random.Generator, dataset) -> dict:
        if self.sigma <= 0:
            return item
        gain = float(np.exp(rng.normal(0.0, self.sigma)))
        out = dict(item)
        if "v" in item and "i" in item:
            out["v"] = item["v"] * gain
            out["i"] = item["i"] * gain
            factor = gain * gain
        elif "p" in item:
            out["p"] = item["p"] * gain
            factor = gain
        else:
            return item

        for key in ("p_total", "power"):
            if key in item:
                out[key] = item[key] * factor
        return out


def materialize(dataset, path, *, n_samples: int | None = None, seed: int = 0):
    """Freeze an augmented dataset into a store, for a released benchmark.

    On-the-fly mixing is right for training -- unbounded diversity, no disk. A
    published benchmark is the one case where the fixed artefact is the point,
    because a number nobody else can reproduce is not a result.

    Args:
        dataset: a :class:`~nilmframe.data.WindowDataset` with an augmentation.
        path: destination store directory.
        n_samples: how many windows to write. Defaults to the whole dataset.
        seed: augmentation seed, recorded in the manifest.

    Returns:
        The path written.

    Example:
        >>> from nilmframe.data import HighFreqView, LowFreqView, WindowDataset, collate_windows
        >>> ids = store.submeters().channel_id.tolist()
        >>> ds = WindowDataset(store, ids, view=HighFreqView(n_cycles=5, cycle_size=64))
        >>> import tempfile, pathlib
        >>> mixed = WindowDataset(store, ids, view=HighFreqView(n_cycles=5, cycle_size=64),
        ...                       augment=nf.MixAggregate(k=(2, 3), p=1.0), seed=0)
        >>> from nilmframe.data.mixing import materialize
        >>> dst = materialize(mixed, pathlib.Path(tempfile.mkdtemp()) / 'frozen', n_samples=4)
        >>> len(nf.Store(dst))
        4
    """
    from nilmframe.store import ChannelKind, Recording, StoreWriter

    n_samples = n_samples or len(dataset)
    with StoreWriter(path, source=f"materialized:{type(dataset).__name__}:seed={seed}") as writer:
        for idx in range(n_samples):
            item = dataset[idx]
            channel = dataset._meta[item["channel"]]
            signals = (
                {"v": item["v"].flatten().numpy(), "i": item["i"].flatten().numpy()}
                if "v" in item
                else {"p": item["p"].numpy()}
            )
            present = [
                dataset.appliances[k] for k, on in enumerate(item.get("presence", [])) if on > 0.5
            ]
            writer.add(
                Recording(
                    dataset="synthetic",
                    house="mixed",
                    session=f"{idx:07d}",
                    kind=ChannelKind.MAINS if len(present) != 1 else ChannelKind.SUBMETER,
                    appliance=present[0] if len(present) == 1 else None,
                    signals=signals,
                    fs=float(channel["fs"]),
                    f0=float(channel["f0"]),
                    meta={"components": present, "source_channel": item["channel"]},
                ),
                channel_id=f"mix-{idx:07d}",
            )
    return path
