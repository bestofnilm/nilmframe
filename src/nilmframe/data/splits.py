"""Evaluation protocols as objects.

A split is not a helper, it is an experimental claim: "this model generalises to a
brand it has never seen", "to a house it has never seen", "to an appliance class
absent from training". Making each protocol a named object with a manifest means
the claim is recorded alongside the number.

Every protocol partitions on ``instance_id`` at minimum, so two recordings of the
same physical appliance can never land on both sides. :func:`check_leakage` proves
it, and ``tests/test_splits.py`` runs that check against every protocol -- the
thing a reviewer cannot verify for themselves.

The predecessor's ``train_test`` sorted brands by descending sample count and put
the *smallest* brands in validation, ignored its own ``random_state`` (it built an
``rng`` and never used it), and filtered with ``sample.devices == [device]`` so
every multi-appliance recording was silently dropped from both sides.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from nilmframe.store import Store

__all__ = [
    "CrossDataset",
    "LeaveBrandOut",
    "LeaveHouseOut",
    "RandomSplit",
    "Split",
    "SplitProtocol",
    "UnseenAppliance",
    "check_leakage",
]


@dataclass(frozen=True)
class Split:
    """Channel ids per fold, plus the protocol that produced them.

    Attributes:
        train: channel ids in the training fold.
        val: channel ids in the validation fold.
        test: channel ids in the optional third fold, empty unless
            ``holdout_size`` was set.
        manifest: what produced this split -- protocol, seed, sizes, and which
            groups landed where. Recorded so a reported number can be traced back
            to the claim it supports.

    Example:
        >>> split = nf.RandomSplit(seed=0).apply(store)
        >>> sorted(split.folds)
        ['test', 'train', 'val']
        >>> split.manifest['protocol']
        'RandomSplit'
    """

    train: list[str]
    val: list[str]
    test: list[str] = field(default_factory=list)
    manifest: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"Split(train={len(self.train)}, val={len(self.val)}, test={len(self.test)}, "
            f"protocol={self.manifest.get('protocol')})"
        )

    @property
    def folds(self) -> dict[str, list[str]]:
        """Fold name to its channel ids.

        Example:
            >>> sorted(nf.RandomSplit(seed=0).apply(store).folds)
            ['test', 'train', 'val']
        """
        return {"train": self.train, "val": self.val, "test": self.test}

    def summary(self, store: Store) -> pd.DataFrame:
        """Per-fold channel, instance, appliance and hour counts.

        Example:
            >>> nf.LeaveHouseOut(test_size=0.5, seed=0).apply(store).summary(store)
                fold  channels  instances  appliances     hours
            0  train         4          4           3  0.002222
            1    val         4          4           3  0.002222
            2   test         0          0           0  0.000000
        """
        rows = []
        for name, ids in self.folds.items():
            sel = store.channels[store.channels["channel_id"].isin(ids)]
            rows.append(
                {
                    "fold": name,
                    "channels": len(sel),
                    "instances": sel["instance_id"].nunique(),
                    "appliances": sel["appliance"].nunique(),
                    "hours": (
                        float((sel["n_samples"] / sel["fs"]).sum() / 3600) if len(sel) else 0.0
                    ),
                }
            )
        return pd.DataFrame(rows)


def check_leakage(
    split: Split, store: Store, keys: tuple[str, ...] = ("instance_id",)
) -> list[str]:
    """Return the identities that appear in more than one fold. Empty is good.

    Args:
        split: the split to check.
        store: the store the channel ids refer to.
        keys: channel columns that must not straddle folds.

    Example:
        >>> from nilmframe.data import check_leakage
        >>> split = nf.LeaveHouseOut(test_size=0.5, seed=0).apply(store)
        >>> check_leakage(split, store)
        []
        >>> check_leakage(split, store, keys=('house', 'instance_id'))
        []
    """
    channels = store.channels.set_index("channel_id")
    problems: list[str] = []

    for key in keys:
        seen: dict[Any, str] = {}
        for fold, ids in split.folds.items():
            if not ids:
                continue
            for value in channels.loc[ids, key].dropna().unique():
                previous = seen.setdefault(value, fold)
                if previous != fold:
                    problems.append(f"{key}={value!r} appears in both {previous!r} and {fold!r}")

    overlap = set(split.train) & set(split.val) | set(split.train) & set(split.test)
    overlap |= set(split.val) & set(split.test)
    problems.extend(f"channel {cid!r} appears in more than one fold" for cid in sorted(overlap))
    return problems


class SplitProtocol:
    """Base class: partition channels by a grouping column.

    Args:
        test_size: fraction of *groups* (not channels) held out for validation.
        seed: controls the group shuffle. Unlike the predecessor, this is used.
        holdout_size: additional fraction of groups reserved as a test fold.

    Example:
        >>> protocol = nf.RandomSplit(test_size=0.25, seed=1)
        >>> protocol
        RandomSplit(test_size=0.25, holdout_size=0.0, seed=1)
        >>> protocol.apply(store)
        Split(train=6, val=2, test=0, protocol=RandomSplit)
    """

    #: Channel column whose distinct values are partitioned.
    group_by: str = "instance_id"
    #: Columns that must not straddle folds. Checked by tests, not enforced here.
    leakage_keys: tuple[str, ...] = ("instance_id",)

    def __init__(self, test_size: float = 0.2, seed: int = 0, holdout_size: float = 0.0) -> None:
        if not 0.0 < test_size < 1.0:
            raise ValueError(f"test_size must be in (0, 1), got {test_size}")
        if not 0.0 <= holdout_size < 1.0:
            raise ValueError(f"holdout_size must be in [0, 1), got {holdout_size}")
        if test_size + holdout_size >= 1.0:
            raise ValueError("test_size + holdout_size must leave something for training")
        self.test_size = test_size
        self.holdout_size = holdout_size
        self.seed = seed

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(test_size={self.test_size}, "
            f"holdout_size={self.holdout_size}, seed={self.seed})"
        )

    def candidates(self, store: Store) -> pd.DataFrame:
        """Channels eligible for splitting. Subclasses narrow this.

        Example:
            >>> len(nf.RandomSplit().candidates(store))
            8
            >>> len(nf.LeaveBrandOut().candidates(store))
            6
        """
        return store.channels

    def apply(self, store: Store) -> Split:
        """Partition the store's channels into folds.

        Raises:
            ValueError: when there are too few distinct groups to split.

        Example:
            >>> nf.LeaveHouseOut(seed=0).apply(store)
            Split(train=4, val=4, test=0, protocol=LeaveHouseOut)
        """
        channels = self.candidates(store)
        if channels.empty:
            raise ValueError(f"{type(self).__name__}: no eligible channels in {store.path}")
        if self.group_by not in channels.columns:
            raise KeyError(f"{self.group_by!r} is not a channel column")

        groups = channels[self.group_by].fillna("__missing__")
        unique = np.asarray(sorted(groups.unique()))
        if len(unique) < 2:
            raise ValueError(
                f"{type(self).__name__} needs at least 2 distinct {self.group_by} values, "
                f"found {len(unique)}: {unique.tolist()}. Use RandomSplit for a store this small."
            )

        rng = np.random.default_rng(self.seed)
        order = rng.permutation(len(unique))
        shuffled = unique[order]

        # At least one group per non-empty fold, whatever the rounding says.
        n_val = max(1, round(len(unique) * self.test_size))
        n_test = max(1, round(len(unique) * self.holdout_size)) if self.holdout_size else 0
        n_val = min(n_val, len(unique) - 1 - n_test)

        test_groups = set(shuffled[:n_test])
        val_groups = set(shuffled[n_test : n_test + n_val])
        train_groups = set(shuffled[n_test + n_val :])

        def pick(wanted: set) -> list[str]:
            return channels.loc[groups.isin(wanted), "channel_id"].tolist()

        return Split(
            train=pick(train_groups),
            val=pick(val_groups),
            test=pick(test_groups),
            manifest={
                "protocol": type(self).__name__,
                "group_by": self.group_by,
                "seed": self.seed,
                "test_size": self.test_size,
                "holdout_size": self.holdout_size,
                "n_groups": len(unique),
                "groups": {
                    "train": sorted(map(str, train_groups)),
                    "val": sorted(map(str, val_groups)),
                    "test": sorted(map(str, test_groups)),
                },
            },
        )


class RandomSplit(SplitProtocol):
    """Split by physical appliance instance.

    The weakest defensible protocol: a model may see the same *class* and the same
    *brand* in training, just not the same unit. Use it as a baseline that upper-bounds
    what the stricter protocols report.

    Args:
        test_size: fraction of *groups* -- not channels -- held out for validation.
        seed: controls the group shuffle.
        holdout_size: additional fraction of groups reserved as a test fold.

    Example:
        >>> split = nf.RandomSplit(test_size=0.3, seed=0).apply(store)
        >>> split
        Split(train=6, val=2, test=0, protocol=RandomSplit)
        >>> len(split.train), len(split.val)
        (6, 2)
    """

    group_by = "instance_id"

    def __init__(self, test_size: float = 0.2, seed: int = 0, holdout_size: float = 0.0) -> None:
        super().__init__(test_size=test_size, seed=seed, holdout_size=holdout_size)


class LeaveBrandOut(SplitProtocol):
    """Hold out whole brands.

    Tests whether a model has learned an appliance class or memorised one
    manufacturer's implementation of it.

    Args:
        test_size: fraction of *groups* -- not channels -- held out for validation.
        seed: controls the group shuffle.
        holdout_size: additional fraction of groups reserved as a test fold.

    Example:
        >>> split = nf.LeaveBrandOut(test_size=0.5, seed=0).apply(store)
        >>> split.manifest['groups']['val']
        ['acme']
    """

    group_by = "brand"
    leakage_keys = ("brand", "instance_id")

    def __init__(self, test_size: float = 0.2, seed: int = 0, holdout_size: float = 0.0) -> None:
        super().__init__(test_size=test_size, seed=seed, holdout_size=holdout_size)

    def candidates(self, store: Store) -> pd.DataFrame:
        # A channel with no brand cannot be assigned to a brand fold without
        # guessing, and guessing is how leakage happens.
        return store.channels[store.channels["brand"].notna()]


class LeaveHouseOut(SplitProtocol):
    """Hold out whole houses -- the standard NILM generalisation protocol.

    Differences in wiring, mains impedance and appliance population all covary with
    the house, so this is the closest proxy for deployment in a new home.

    Args:
        test_size: fraction of *groups* -- not channels -- held out for validation.
        seed: controls the group shuffle.
        holdout_size: additional fraction of groups reserved as a test fold.

    Example:
        >>> split = nf.LeaveHouseOut(test_size=0.5, seed=0).apply(store)
        >>> split.manifest['groups']['val']
        ['house_1']
    """

    group_by = "house"
    leakage_keys = ("house", "instance_id")

    def __init__(self, test_size: float = 0.2, seed: int = 0, holdout_size: float = 0.0) -> None:
        super().__init__(test_size=test_size, seed=seed, holdout_size=holdout_size)


class CrossDataset(SplitProtocol):
    """Train on some datasets, evaluate on others.

    Args:
        train_on: dataset names for training.
        test_on: dataset names for validation.

    Example:
        >>> protocol = nf.CrossDataset(train_on=['example'], test_on=[])
        >>> protocol
        CrossDataset(train_on=['example'], test_on=[])
    """

    group_by = "dataset"
    leakage_keys = ("dataset", "instance_id")

    def __init__(self, train_on: list[str], test_on: list[str], seed: int = 0) -> None:
        super().__init__(test_size=0.5, seed=seed)
        if set(train_on) & set(test_on):
            raise ValueError(f"datasets appear on both sides: {set(train_on) & set(test_on)}")
        self.train_on = list(train_on)
        self.test_on = list(test_on)

    def __repr__(self) -> str:
        return f"CrossDataset(train_on={self.train_on}, test_on={self.test_on})"

    def apply(self, store: Store) -> Split:
        channels = store.channels
        available = set(channels["dataset"])
        missing = (set(self.train_on) | set(self.test_on)) - available
        if missing:
            raise ValueError(
                f"datasets not in the store: {sorted(missing)}; have {sorted(available)}"
            )

        return Split(
            train=channels.loc[channels["dataset"].isin(self.train_on), "channel_id"].tolist(),
            val=channels.loc[channels["dataset"].isin(self.test_on), "channel_id"].tolist(),
            manifest={
                "protocol": "CrossDataset",
                "group_by": "dataset",
                "train_on": self.train_on,
                "test_on": self.test_on,
                "seed": self.seed,
            },
        )


class UnseenAppliance(SplitProtocol):
    """Hold out whole appliance classes -- the open-set protocol.

    Held-out classes are absent from training entirely, so at evaluation the model
    meets appliances it has no label for. This is what
    ``open_set_note.tex`` asks for, and it is a property of the *split*, not of a
    post-hoc rejection rule bolted on later.

    Args:
        unknown: appliance classes to hold out. When ``None``, ``n_unknown``
            classes are drawn at random with the given seed.
        n_unknown: how many classes to hold out when ``unknown`` is not given.
        seed: controls that draw.

    Note:
        :meth:`apply` returns a split whose ``val`` fold contains *only* unknown
        classes. Pair it with a second protocol for the known-class validation set
        if you need both, or read ``manifest['unknown']`` and mark those classes
        unknown in the store's appliance table.

    Example:
        >>> split = nf.UnseenAppliance(unknown=['kettle']).apply(store)
        >>> split.manifest['unknown']
        ['kettle']
        >>> split.manifest['known']
        ['fridge', 'laptop', 'microwave']
    """

    group_by = "appliance"
    leakage_keys = ("appliance", "instance_id")

    def __init__(
        self,
        unknown: list[str] | None = None,
        n_unknown: int = 1,
        seed: int = 0,
    ) -> None:
        super().__init__(test_size=0.5, seed=seed)
        self.unknown = list(unknown) if unknown else None
        self.n_unknown = int(n_unknown)

    def __repr__(self) -> str:
        return (
            f"UnseenAppliance(unknown={self.unknown}, n_unknown={self.n_unknown}, seed={self.seed})"
        )

    def apply(self, store: Store) -> Split:
        channels = store.channels[store.channels["appliance"].notna()]
        classes = sorted(channels["appliance"].unique())

        if self.unknown is not None:
            missing = set(self.unknown) - set(classes)
            if missing:
                raise ValueError(f"appliances not in the store: {sorted(missing)}")
            unknown = list(self.unknown)
        else:
            if self.n_unknown >= len(classes):
                raise ValueError(
                    f"cannot hold out {self.n_unknown} of {len(classes)} appliance classes"
                )
            rng = np.random.default_rng(self.seed)
            unknown = sorted(
                np.asarray(classes)[rng.permutation(len(classes))[: self.n_unknown]].tolist()
            )

        held = channels["appliance"].isin(unknown)
        return Split(
            train=channels.loc[~held, "channel_id"].tolist(),
            val=channels.loc[held, "channel_id"].tolist(),
            manifest={
                "protocol": "UnseenAppliance",
                "group_by": "appliance",
                "seed": self.seed,
                "unknown": unknown,
                "known": [c for c in classes if c not in unknown],
            },
        )
