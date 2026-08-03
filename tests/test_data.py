"""Phase 3 acceptance: dataset, views, splits, and the leakage guarantee.

plan.html's criterion: one training script runs with and without alignment under an
identical harness, and a leakage test is green.
"""

from __future__ import annotations

import pickle

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from nilmframe.data import (
    CrossDataset,
    HighFreqView,
    LeaveBrandOut,
    LeaveHouseOut,
    LowFreqView,
    RandomSplit,
    UnseenAppliance,
    WindowDataset,
    WindowIndex,
    check_leakage,
    collate_windows,
)

# --------------------------------------------------------------------------- #
# Splits -- the leakage guarantee
# --------------------------------------------------------------------------- #

PROTOCOLS = [
    RandomSplit(test_size=0.3, seed=0),
    LeaveBrandOut(test_size=0.4, seed=0),
    LeaveHouseOut(test_size=0.5, seed=0),
]


@pytest.mark.parametrize("protocol", PROTOCOLS, ids=lambda p: type(p).__name__)
def test_no_protocol_leaks_an_instance_across_folds(protocol, plaid_store):
    """The claim a reviewer cannot check for themselves, checked here."""
    split = protocol.apply(plaid_store)
    assert check_leakage(split, plaid_store, keys=protocol.leakage_keys) == []
    assert split.train and split.val


def test_leave_brand_out_holds_whole_brands(plaid_store):
    split = LeaveBrandOut(test_size=0.4, seed=0).apply(plaid_store)
    channels = plaid_store.channels.set_index("channel_id")
    train_brands = set(channels.loc[split.train, "brand"])
    val_brands = set(channels.loc[split.val, "brand"])
    assert train_brands and val_brands
    assert not (train_brands & val_brands)


def test_unseen_appliance_removes_the_class_from_training(plaid_store):
    split = UnseenAppliance(unknown=["fridge"]).apply(plaid_store)
    channels = plaid_store.channels.set_index("channel_id")
    assert "fridge" not in set(channels.loc[split.train, "appliance"])
    assert set(channels.loc[split.val, "appliance"]) == {"fridge"}
    assert split.manifest["unknown"] == ["fridge"]
    assert "fridge" not in split.manifest["known"]


def test_unseen_appliance_can_draw_at_random(plaid_store):
    split = UnseenAppliance(n_unknown=2, seed=3).apply(plaid_store)
    assert len(split.manifest["unknown"]) == 2
    assert check_leakage(split, plaid_store, keys=("appliance",)) == []


def test_unseen_appliance_rejects_impossible_requests(plaid_store):
    with pytest.raises(ValueError, match="not in the store"):
        UnseenAppliance(unknown=["teleporter"]).apply(plaid_store)
    with pytest.raises(ValueError, match="cannot hold out"):
        UnseenAppliance(n_unknown=99).apply(plaid_store)


def test_seed_changes_the_partition_and_is_actually_used(plaid_store):
    """The predecessor built an rng and never used it, so seeds did nothing."""
    a = LeaveBrandOut(test_size=0.4, seed=0).apply(plaid_store)
    b = LeaveBrandOut(test_size=0.4, seed=7).apply(plaid_store)
    partitions = {tuple(sorted(s.manifest["groups"]["val"])) for s in (a, b)}
    assert len(partitions) == 2, "different seeds must be able to give different folds"


def test_split_is_reproducible(plaid_store):
    a = LeaveHouseOut(seed=5).apply(plaid_store)
    b = LeaveHouseOut(seed=5).apply(plaid_store)
    assert a.train == b.train and a.val == b.val


def test_holdout_creates_a_third_fold(plaid_store):
    split = RandomSplit(test_size=0.2, holdout_size=0.2, seed=1).apply(plaid_store)
    assert split.train and split.val and split.test
    assert check_leakage(split, plaid_store) == []


def test_multi_appliance_recordings_are_not_silently_dropped(plaid_store):
    """The predecessor's train_test dropped every aggregate recording from both folds."""
    split = RandomSplit(test_size=0.3, seed=0).apply(plaid_store)
    assigned = set(split.train) | set(split.val) | set(split.test)
    assert set(plaid_store.channels["channel_id"]) == assigned
    assert set(plaid_store.mains()["channel_id"]) <= assigned


def test_leave_brand_out_skips_channels_without_a_brand(plaid_store):
    """Aggregate recordings have no brand; assigning them would mean guessing."""
    split = LeaveBrandOut(test_size=0.4, seed=0).apply(plaid_store)
    assigned = set(split.train) | set(split.val)
    assert not (assigned & set(plaid_store.mains()["channel_id"]))


def test_cross_dataset(combined_store):
    split = CrossDataset(train_on=["plaid"], test_on=["whited"]).apply(combined_store)
    channels = combined_store.channels.set_index("channel_id")
    assert set(channels.loc[split.train, "dataset"]) == {"plaid"}
    assert set(channels.loc[split.val, "dataset"]) == {"whited"}
    assert check_leakage(split, combined_store, keys=("dataset",)) == []


def test_cross_dataset_rejects_datasets_not_in_the_store(combined_store):
    with pytest.raises(ValueError, match="not in the store"):
        CrossDataset(train_on=["plaid"], test_on=["redd"]).apply(combined_store)


def test_cross_dataset_rejects_overlap_and_missing():
    with pytest.raises(ValueError, match="both sides"):
        CrossDataset(train_on=["plaid"], test_on=["plaid"])


def test_split_rejects_bad_sizes():
    with pytest.raises(ValueError, match="test_size"):
        RandomSplit(test_size=0.0)
    with pytest.raises(ValueError, match="leave something for training"):
        RandomSplit(test_size=0.7, holdout_size=0.4)


def test_split_summary(plaid_store):
    split = RandomSplit(seed=0).apply(plaid_store)
    summary = split.summary(plaid_store).set_index("fold")
    assert summary.loc["train", "channels"] > 0
    assert summary.loc["train", "hours"] > 0


# --------------------------------------------------------------------------- #
# Window index
# --------------------------------------------------------------------------- #


def test_window_index_covers_the_channel(plaid_store):
    view = HighFreqView(n_cycles=5, cycle_size=32)
    index = WindowIndex.build(
        plaid_store,
        plaid_store.submeters()["channel_id"].tolist(),
        window_samples=view.window_samples,
        stride=1.0,
    )
    assert len(index) > 0
    cid, start, length = index.locate(0)
    assert start == 0
    n = int(plaid_store.channel(cid)["n_samples"])
    assert all(s + length <= n for s in index.start[index.channel_of == 0])


def test_window_index_stride_controls_overlap(plaid_store):
    view = HighFreqView(n_cycles=5, cycle_size=32)
    ids = plaid_store.submeters()["channel_id"].tolist()
    full = WindowIndex.build(plaid_store, ids, window_samples=view.window_samples, stride=1.0)
    half = WindowIndex.build(plaid_store, ids, window_samples=view.window_samples, stride=0.5)
    assert len(half) > len(full)


def test_window_index_skips_channels_shorter_than_a_window(plaid_store):
    """A short channel must be skipped, not zero-padded into a fabricated window."""
    huge = HighFreqView(n_cycles=100_000, cycle_size=32)
    index = WindowIndex.build(
        plaid_store,
        plaid_store.channels["channel_id"].tolist(),
        window_samples=huge.window_samples,
    )
    assert len(index) == 0


def test_window_index_cap_is_evenly_spaced(plaid_store):
    view = HighFreqView(n_cycles=2, cycle_size=32)
    ids = plaid_store.submeters()["channel_id"].tolist()[:1]
    capped = WindowIndex.build(
        plaid_store, ids, window_samples=view.window_samples, max_windows_per_channel=3
    )
    assert len(capped) <= 3
    # Spread across the recording rather than bunched at the start.
    assert capped.start[-1] > capped.start[0]


def test_window_index_rejects_unknown_channels(plaid_store):
    with pytest.raises(KeyError, match="not in the store"):
        WindowIndex.build(plaid_store, ["nope"], window_samples=lambda fs, f0: 10)


# --------------------------------------------------------------------------- #
# Views
# --------------------------------------------------------------------------- #


def test_aligned_and_unaligned_views_have_identical_shapes(plaid_store):
    """The condition for a fair comparison: one backbone consumes both arms."""
    ids = plaid_store.submeters()["channel_id"].tolist()
    shapes = []
    for align in ("fitps", None):
        view = HighFreqView(n_cycles=6, cycle_size=64, align=align)
        ds = WindowDataset(plaid_store, ids, view=view)
        item = ds[0]
        shapes.append((tuple(item["v"].shape), tuple(item["i"].shape)))
    assert shapes[0] == shapes[1] == ((6, 64), (6, 64))


def test_both_high_frequency_arms_see_identical_windows(plaid_store):
    """The control must consume the same signal, or the comparison is confounded.

    Alignment needs a slack margin to guarantee n_cycles survive. If the control
    took exactly n_cycles nominal periods it would get *more* windows out of the
    same channel, and "aligned versus not" would silently also be "fewer training
    windows versus more".
    """
    ids = plaid_store.submeters()["channel_id"].tolist()
    aligned = WindowDataset(
        plaid_store, ids, view=HighFreqView(n_cycles=6, cycle_size=64, align="fitps")
    )
    control = WindowDataset(
        plaid_store, ids, view=HighFreqView(n_cycles=6, cycle_size=64, align=None)
    )

    assert len(aligned) == len(control)
    assert np.array_equal(aligned.index.start, control.index.start)
    assert np.array_equal(aligned.index.length, control.index.length)


def test_alignment_reduces_cycle_to_cycle_variance(plaid_store):
    """Alignment must actually do something: aligned cycles should agree better.

    The fixture mains runs at 50.1 Hz, so a fixed-length chunk drifts in phase
    while an aligned cycle does not.
    """
    ids = plaid_store.submeters()["channel_id"].tolist()[:1]
    spreads = {}
    for align in ("fitps", None):
        view = HighFreqView(n_cycles=10, cycle_size=64, align=align, f0=50.0)
        item = WindowDataset(plaid_store, ids, view=view)[0]
        cycles = item["i"]
        spreads[align] = (cycles - cycles.mean(0, keepdim=True)).abs().mean().item()
    assert spreads["fitps"] < spreads[None]


def test_lowfreq_view_derives_power_from_the_same_waveform(plaid_store):
    """The LF arm must be a reduction of the HF arm, not a separate pipeline."""
    ids = plaid_store.submeters()["channel_id"].tolist()[:1]
    lf = LowFreqView(rate_hz=10.0, n_steps=5)
    ds = WindowDataset(plaid_store, ids, view=lf)
    item = ds[0]

    assert item["p"].shape == (5,)
    # The window mean of the derived series equals the window's active power.
    assert torch.allclose(item["p"].mean(), item["p_total"], rtol=1e-4)
    assert item["p_total"] > 0


def test_lowfreq_and_highfreq_agree_on_total_power(plaid_store):
    """Both views measure the same physical window, so totals must match.

    They do not see the same number of samples (the HF view oversizes its window
    so alignment can discard partial cycles), so this compares the power of a
    steady signal, which is what the fixture provides.
    """
    ids = plaid_store.submeters()["channel_id"].tolist()[:1]
    hf = WindowDataset(plaid_store, ids, view=HighFreqView(n_cycles=10, cycle_size=64))[0]
    lf = WindowDataset(plaid_store, ids, view=LowFreqView(rate_hz=5.0, n_steps=5))[0]
    assert torch.allclose(hf["p_total"], lf["p_total"], rtol=0.05)


def test_lowfreq_view_reads_stored_power_when_there_is_no_waveform(tmp_path):
    from nilmframe.store import ChannelKind, Recording, Store, StoreWriter

    watts = np.linspace(100, 200, 600, dtype=np.float32)
    with StoreWriter(tmp_path / "lf") as w:
        cid = w.add(
            Recording(
                dataset="synthetic",
                house="h",
                session="s",
                kind=ChannelKind.SUBMETER,
                appliance="kettle",
                signals={"p": watts},
                fs=1.0,
            )
        )
    store = Store(tmp_path / "lf")
    ds = WindowDataset(store, [cid], view=LowFreqView(rate_hz=1.0, n_steps=60))
    item = ds[0]
    assert item["p"].shape == (60,)
    assert torch.allclose(item["p"].mean(), torch.tensor(watts[:60].mean()), rtol=1e-4)


def test_view_skips_channels_it_cannot_render(tmp_path):
    """A store holds several rates at once; each arm selects what it can read.

    A UK-DALE split contains a house's 16 kHz mains and its 1/6 Hz submeters
    together. The high-frequency arm must quietly take the former rather than
    failing on the first low-rate channel it meets, which is what lets one split
    serve every arm of a sweep.
    """
    from nilmframe.store import ChannelKind, Recording, Store, StoreWriter

    with StoreWriter(tmp_path / "lf") as w:
        low = w.add(
            Recording(
                dataset="s",
                house="h",
                session="s",
                kind=ChannelKind.SUBMETER,
                appliance="kettle",
                signals={"p": np.ones(600, np.float32)},
                fs=1.0,
            )
        )
    store = Store(tmp_path / "lf")

    ds = WindowDataset(store, [low], view=HighFreqView(n_cycles=2, cycle_size=8))
    assert len(ds) == 0
    assert ds.skipped_channels == [low]

    # ...and the low-frequency view does read it.
    assert len(WindowDataset(store, [low], view=LowFreqView(rate_hz=1.0, n_steps=60))) > 0


def test_supports_is_explicit_about_what_a_view_can_read():
    hf = HighFreqView(n_cycles=4, cycle_size=32, f0=50.0)
    assert hf.supports({"v", "i"}, fs=6000.0, f0=50.0)
    assert not hf.supports({"p"}, fs=6000.0, f0=50.0), "no waveform"
    assert not hf.supports({"v", "i"}, fs=60.0, f0=50.0), "below two samples per cycle"

    lf = LowFreqView(rate_hz=1.0)
    assert lf.supports({"p"}, fs=1.0, f0=50.0)
    assert lf.supports({"v", "i"}, fs=6000.0, f0=50.0)
    assert not lf.supports({"p"}, fs=0.1, f0=50.0), "cannot upsample to 1 Hz"


def test_view_validation():
    with pytest.raises(ValueError, match="align must be"):
        HighFreqView(align="wavelet")
    with pytest.raises(ValueError, match="quantity must be"):
        LowFreqView(quantity="reactive")


# --------------------------------------------------------------------------- #
# Dataset and targets
# --------------------------------------------------------------------------- #


def test_submeter_targets_are_presence_and_power(plaid_store):
    ids = plaid_store.submeters()["channel_id"].tolist()
    ds = WindowDataset(plaid_store, ids, view=HighFreqView(n_cycles=5, cycle_size=32))
    item = ds[0]

    k = plaid_store.n_appliances
    assert item["presence"].shape == (k,)
    assert item["power"].shape == (k,)
    assert item["power_mask"].shape == (k,)
    # A submetered recording contains one appliance, so every class's power is
    # known: one positive value and K-1 exact zeros.
    assert item["power_mask"].all()
    assert item["presence"].sum() == 1
    assert (item["power"] > 0).sum() == 1


def test_power_target_matches_the_measured_window_power(plaid_store):
    ids = plaid_store.submeters()["channel_id"].tolist()
    ds = WindowDataset(plaid_store, ids, view=HighFreqView(n_cycles=5, cycle_size=32))
    item = ds[0]
    known = item["power"][item["power_mask"]]
    assert torch.allclose(known.sum(), item["p_total"], rtol=1e-4)
    assert torch.allclose(item["power"].sum(), item["p_total"], rtol=1e-4)


def test_aggregate_targets_mask_out_unknown_power(plaid_store):
    """Presence is annotated; per-appliance power is not, so it must stay masked."""
    ids = plaid_store.mains()["channel_id"].tolist()
    ds = WindowDataset(plaid_store, ids, view=HighFreqView(n_cycles=5, cycle_size=32), stride=0.5)
    seen_presence = False
    for k in range(len(ds)):
        item = ds[k]
        assert not item["power_mask"].any(), "power must never be invented for an aggregate"
        seen_presence |= bool(item["presence"].any())
    assert seen_presence, "some window should overlap an annotated activation"


def test_presence_uses_the_appliance_threshold(tmp_path):
    from nilmframe.store import ChannelKind, Recording, Store, StoreWriter

    fs = 6000.0
    t = np.arange(6000) / fs
    v = (230 * np.sqrt(2) * np.sin(2 * np.pi * 50 * t)).astype(np.float32)
    i = (0.01 * np.sin(2 * np.pi * 50 * t)).astype(np.float32)  # ~1.6 W, below 10 W

    with StoreWriter(tmp_path / "s") as w:
        cid = w.add(
            Recording(
                dataset="s",
                house="h",
                session="s",
                kind=ChannelKind.SUBMETER,
                appliance="kettle",
                signals={"v": v, "i": i},
                fs=fs,
            )
        )
    store = Store(tmp_path / "s")
    item = WindowDataset(store, [cid], view=HighFreqView(n_cycles=5, cycle_size=32))[0]
    assert item["presence"].sum() == 0, "below threshold must not count as on"
    assert item["power_mask"].any(), "power is still known, it is just small"


def test_no_label_derived_quantity_reaches_the_input(plaid_store):
    """p_total must come from the signal, not from summing the targets.

    Scaling the labels alone must not change the measured aggregate. The
    predecessor's `scores()` used `Y_true.sum(1)` and would fail this.
    """
    ids = plaid_store.submeters()["channel_id"].tolist()[:1]
    ds = WindowDataset(plaid_store, ids, view=HighFreqView(n_cycles=5, cycle_size=32))
    item = ds[0]

    from nilmframe.data.views import active_power

    recomputed = active_power(item["v"].flatten(), item["i"].flatten())
    assert torch.allclose(recomputed, item["p_total"], rtol=0.05)


def test_targets_can_be_switched_off_for_inference(plaid_store):
    ids = plaid_store.submeters()["channel_id"].tolist()
    item = WindowDataset(
        plaid_store, ids, view=HighFreqView(n_cycles=3, cycle_size=16), targets=()
    )[0]
    assert "presence" not in item and "power" not in item
    assert "i" in item and "p_total" in item


def test_unknown_target_is_rejected(plaid_store):
    with pytest.raises(ValueError, match="unknown targets"):
        WindowDataset(plaid_store, [], view=HighFreqView(), targets=("magic",))


def test_item_carries_provenance(plaid_store):
    ids = plaid_store.submeters()["channel_id"].tolist()
    item = WindowDataset(plaid_store, ids, view=HighFreqView(n_cycles=3, cycle_size=16))[0]
    assert item["channel"] in ids
    assert isinstance(item["start"], int)


# --------------------------------------------------------------------------- #
# DataLoader integration -- "works as PyTorch"
# --------------------------------------------------------------------------- #


def test_dataloader_batches(plaid_store):
    ids = plaid_store.submeters()["channel_id"].tolist()
    ds = WindowDataset(plaid_store, ids, view=HighFreqView(n_cycles=4, cycle_size=32))
    loader = DataLoader(ds, batch_size=4, collate_fn=collate_windows, shuffle=True)
    batch = next(iter(loader))

    assert batch["i"].shape == (4, 4, 32)
    assert batch["presence"].shape == (4, plaid_store.n_appliances)
    assert batch["p_total"].shape == (4,)
    assert len(batch["channel"]) == 4


@pytest.mark.parametrize("workers", [0, 2])
def test_dataloader_with_workers(plaid_store, workers):
    """Memory-mapped signals and a numpy index must survive process spawn."""
    ids = plaid_store.submeters()["channel_id"].tolist()
    ds = WindowDataset(plaid_store, ids, view=HighFreqView(n_cycles=4, cycle_size=32))
    loader = DataLoader(ds, batch_size=2, num_workers=workers, collate_fn=collate_windows)
    total = sum(b["i"].shape[0] for b in loader)
    assert total == len(ds)


def test_dataset_is_picklable(plaid_store):
    ids = plaid_store.submeters()["channel_id"].tolist()
    ds = WindowDataset(plaid_store, ids, view=HighFreqView(n_cycles=4, cycle_size=32))
    revived = pickle.loads(pickle.dumps(ds))
    assert len(revived) == len(ds)
    assert torch.equal(revived[0]["i"], ds[0]["i"])


def test_two_views_share_one_harness(plaid_store):
    """plan.html's phase-3 acceptance: one script, both arms, identical harness."""
    ids = plaid_store.submeters()["channel_id"].tolist()
    for view in (
        HighFreqView(n_cycles=4, cycle_size=32, align="fitps"),
        HighFreqView(n_cycles=4, cycle_size=32, align=None),
        LowFreqView(rate_hz=8.0, n_steps=4),
    ):
        ds = WindowDataset(plaid_store, ids, view=view)
        loader = DataLoader(ds, batch_size=3, collate_fn=collate_windows)
        batch = next(iter(loader))
        assert batch["presence"].shape[0] == 3
        assert batch["p_total"].shape == (3,)
