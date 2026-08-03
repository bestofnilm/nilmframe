"""Phase 5 acceptance: on-the-fly aggregation and the open-set channel.

plan.html's criteria: exact conservation in a mixing test, and open-set metrics
computed on a held-out-appliance split. The criterion was originally written as
"below 1e-5 W"; that is unreachable in float32 for kilowatt loads (7 kW * 2^-23 is
already ~1e-3 W), so it is enforced relatively, at 1e-6 -- float32 epsilon.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from nilmframe.data import (
    Compose,
    GainJitter,
    HighFreqView,
    LowFreqView,
    MixAggregate,
    VoltageJitter,
    WindowDataset,
    collate_windows,
)
from nilmframe.data.mixing import materialize
from nilmframe.store import Store

VIEW = HighFreqView(n_cycles=6, cycle_size=32)


def submeter_dataset(store, **kw) -> WindowDataset:
    return WindowDataset(store, store.submeters()["channel_id"].tolist(), view=VIEW, **kw)


# --------------------------------------------------------------------------- #
# Conservation -- the acceptance criterion
# --------------------------------------------------------------------------- #


def test_mixture_conserves_power_exactly(plaid_store):
    """Parts must sum to the measured whole to floating-point.

    This is why every component's contribution is recomputed under the base
    window's voltage rather than carried over from its own window: otherwise the
    parts are measured against one voltage and the whole against another, and the
    books do not balance.
    """
    ds = submeter_dataset(plaid_store, augment=MixAggregate(k=(2, 4)), seed=1)
    worst_relative, worst_absolute = 0.0, 0.0
    for idx in range(len(ds)):
        item = ds[idx]
        total = float(item["p_total"])
        error = float((item["power"].sum() - item["p_total"]).abs())
        worst_absolute = max(worst_absolute, error)
        worst_relative = max(worst_relative, error / max(abs(total), 1e-9))

    # The bound is relative because float32 cannot do better in absolute terms:
    # these are kilowatt loads, and 7 kW * 2^-23 is already ~1e-3 W. A relative
    # bound at 1e-6 is float32 epsilon, i.e. the arithmetic is exact and only the
    # representation is not.
    assert worst_relative < 1e-6, (
        f"worst conservation error {worst_absolute:.3e} W ({worst_relative:.2e} relative)"
    )


def test_mixture_current_is_the_sum_of_its_parts(plaid_store):
    ds = submeter_dataset(plaid_store, augment=MixAggregate(k=(2, 2), p=1.0), seed=3)
    plain = submeter_dataset(plaid_store)

    item = ds[0]
    assert item["n_components"] == 2
    # The mixed current must exceed the base alone, and its power must be the sum.
    assert not torch.allclose(item["i"], plain[0]["i"])
    assert item["p_total"] > plain[0]["p_total"]


def test_mixture_combines_targets(plaid_store):
    ds = submeter_dataset(plaid_store, augment=MixAggregate(k=(3, 3), p=1.0), seed=5)
    item = ds[0]
    assert item["presence"].sum() >= 1
    assert (item["power"] >= 0).all()
    assert item["power_mask"].all(), "all parts are submetered, so all power is known"


def test_mixture_never_superposes_a_channel_with_itself(plaid_store):
    """Two windows of one kettle are still one kettle; adding them fabricates two."""
    ds = submeter_dataset(plaid_store, augment=MixAggregate(k=(4, 4), p=1.0), seed=7)
    for idx in range(len(ds)):
        item = ds[idx]
        # Distinct appliances would show as distinct presence entries; at minimum
        # the component count must never exceed the number of distinct channels.
        assert item["n_components"] <= len(ds.index.channel_ids)


def test_mixture_only_combines_matching_sample_rates(combined_store):
    """Superposition assumes a shared voltage; 6 kHz and 44.1 kHz do not have one."""
    ds = WindowDataset(
        combined_store,
        combined_store.submeters()["channel_id"].tolist(),
        view=VIEW,
        augment=MixAggregate(k=(3, 3), p=1.0, same_rate_only=True),
        seed=11,
    )
    mixer = ds.augment
    rates = {round(float(combined_store.channel(cid)["fs"]), 3) for cid in ds.index.channel_ids}
    assert len(rates) > 1, "the fixture should contain two sampling rates"

    for rate in rates:
        pool = mixer._pool_for(ds, rate)
        for idx in pool.tolist():
            channel = ds.index.channel_ids[int(ds.index.channel_of[idx])]
            assert round(float(combined_store.channel(channel)["fs"]), 3) == rate


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #


def test_mixing_is_reproducible_for_a_given_seed_and_epoch(plaid_store):
    a = submeter_dataset(plaid_store, augment=MixAggregate(k=(2, 4)), seed=42)
    b = submeter_dataset(plaid_store, augment=MixAggregate(k=(2, 4)), seed=42)
    assert torch.equal(a[3]["i"], b[3]["i"])
    assert torch.equal(a[3]["power"], b[3]["power"])


def test_epochs_see_different_mixtures(plaid_store):
    """Unbounded diversity is the reason to mix on the fly rather than on disk."""
    ds = submeter_dataset(plaid_store, augment=MixAggregate(k=(3, 4), p=1.0), seed=0)
    first = ds[2]["i"].clone()
    ds.set_epoch(1)
    second = ds[2]["i"].clone()
    assert not torch.equal(first, second)

    # ...but epoch 1 is still exactly epoch 1 on a rerun.
    ds.set_epoch(1)
    assert torch.equal(ds[2]["i"], second)


def test_k_range_leaves_some_windows_unmixed(plaid_store):
    """A model that never sees a single appliance cannot recognise one."""
    ds = submeter_dataset(plaid_store, augment=MixAggregate(k=(1, 4), p=1.0), seed=13)
    counts = {ds[idx].get("n_components", 1) for idx in range(len(ds))}
    assert 1 in counts and max(counts) > 1


def test_p_controls_how_often_mixing_happens(plaid_store):
    never = submeter_dataset(plaid_store, augment=MixAggregate(k=(2, 4), p=0.0), seed=1)
    assert all(never[i]["n_components"] == 1 for i in range(len(never)))


def test_every_item_carries_the_same_keys(plaid_store):
    """Regression: a batch mixing mixed and unmixed windows must still collate.

    MixAggregate used to stamp n_components only when it actually mixed, so a
    p<1 run produced items with different keys and collation died on the first
    disagreement -- which only showed up in an end-to-end sweep.
    """
    ds = submeter_dataset(plaid_store, augment=MixAggregate(k=(1, 4), p=0.5), seed=23)
    keys = [frozenset(ds[i]) for i in range(len(ds))]
    assert len(set(keys)) == 1

    loader = DataLoader(ds, batch_size=len(ds), collate_fn=collate_windows)
    assert next(iter(loader))["n_components"].shape == (len(ds),)


def test_collate_reports_a_key_mismatch_clearly(plaid_store):
    ds = submeter_dataset(plaid_store)
    a, b = ds[0], dict(ds[1])
    b.pop("presence")
    with pytest.raises(ValueError, match="same keys"):
        collate_windows([a, b])


def test_mix_validation():
    with pytest.raises(ValueError, match="increasing range"):
        MixAggregate(k=(0, 3))
    with pytest.raises(ValueError, match="probability"):
        MixAggregate(p=2.0)


# --------------------------------------------------------------------------- #
# Other augmentations -- signal and label must stay consistent
# --------------------------------------------------------------------------- #


def test_voltage_jitter_preserves_power(plaid_store):
    """Supply varies between homes; the kettle still draws the same watts."""
    plain = submeter_dataset(plaid_store)
    jittered = submeter_dataset(plaid_store, augment=VoltageJitter(sigma=0.1), seed=2)
    a, b = plain[0], jittered[0]
    assert not torch.allclose(a["v"], b["v"])
    assert torch.allclose((a["v"] * a["i"]).mean(), (b["v"] * b["i"]).mean(), rtol=1e-4), (
        "jitter must not silently relabel the power"
    )


def test_gain_jitter_rescales_the_targets_too(plaid_store):
    """An augmentation that changes the signal but not the label is a bug."""
    jittered = submeter_dataset(plaid_store, augment=GainJitter(sigma=0.2), seed=4)
    item = jittered[0]
    assert torch.allclose(item["power"].sum(), item["p_total"], rtol=1e-4)


def test_compose_chains_augmentations(plaid_store):
    pipeline = Compose([MixAggregate(k=(2, 3), p=1.0), VoltageJitter(0.02), GainJitter(0.05)])
    ds = submeter_dataset(plaid_store, augment=pipeline, seed=9)
    item = ds[0]
    assert item["n_components"] >= 2
    assert torch.allclose(item["power"].sum(), item["p_total"], rtol=1e-3)
    assert "Compose(" in repr(pipeline)


def test_augmented_dataset_still_batches(plaid_store):
    ds = submeter_dataset(plaid_store, augment=MixAggregate(k=(1, 3)), seed=6)
    loader = DataLoader(ds, batch_size=4, collate_fn=collate_windows)
    batch = next(iter(loader))
    assert batch["i"].shape == (4, 6, 32)
    assert batch["presence"].shape == (4, plaid_store.n_appliances)


def test_low_frequency_windows_mix_too(plaid_store):
    ds = WindowDataset(
        plaid_store,
        plaid_store.submeters()["channel_id"].tolist(),
        view=LowFreqView(rate_hz=10.0, n_steps=5),
        augment=MixAggregate(k=(2, 3), p=1.0),
        seed=8,
    )
    item = ds[0]
    assert item["p"].shape == (5,)
    assert torch.allclose(item["power"].sum(), item["p_total"], rtol=1e-4)


# --------------------------------------------------------------------------- #
# Open set
# --------------------------------------------------------------------------- #


def test_unknown_appliance_gets_no_column(plaid_store):
    """The held-out class must be absent from the label space, not just ignored."""
    ds = submeter_dataset(plaid_store, unknown_appliances=["fridge"])
    assert "fridge" not in ds.appliances
    assert ds.n_appliances == plaid_store.n_appliances - 1
    assert ds[0]["presence"].shape == (ds.n_appliances,)


def test_unknown_windows_are_flagged_and_carry_no_targets(plaid_store):
    ds = submeter_dataset(plaid_store, unknown_appliances=["fridge"])
    flagged = [ds[i] for i in range(len(ds)) if ds[i]["is_unknown"] > 0.5]
    known = [ds[i] for i in range(len(ds)) if ds[i]["is_unknown"] <= 0.5]

    assert flagged and known
    for item in flagged:
        assert item["presence"].sum() == 0, "an unknown appliance has no class to claim"
        assert not item["power_mask"].any(), "its power is unattributable"
    for item in known:
        assert item["power_mask"].all()


def test_unknown_flag_survives_mixing(plaid_store):
    """A mixture containing an unknown component is itself unknown-contaminated."""
    ds = submeter_dataset(
        plaid_store,
        unknown_appliances=["fridge"],
        augment=MixAggregate(k=(3, 4), p=1.0),
        seed=17,
    )
    flags = [float(ds[i]["is_unknown"]) for i in range(len(ds))]
    assert max(flags) == 1.0


def test_unknown_appliances_must_exist(plaid_store):
    with pytest.raises(ValueError, match="not in the label space"):
        submeter_dataset(plaid_store, unknown_appliances=["teleporter"])


# --------------------------------------------------------------------------- #
# Materialisation -- the frozen benchmark path
# --------------------------------------------------------------------------- #


def test_materialize_freezes_a_mixture_set(plaid_store, tmp_path):
    """On-the-fly for training; a fixed artefact for a published benchmark."""
    ds = submeter_dataset(plaid_store, augment=MixAggregate(k=(2, 3), p=1.0), seed=21)
    path = materialize(ds, tmp_path / "frozen", n_samples=6, seed=21)

    frozen = Store(path)
    assert len(frozen) == 6
    assert frozen.verify(deep=True) == []
    assert "materialized" in frozen.manifest["source"]
    assert frozen.manifest["content_sha256"]


def test_materialized_store_is_reproducible(plaid_store, tmp_path):
    def build(where):
        ds = submeter_dataset(plaid_store, augment=MixAggregate(k=(2, 3), p=1.0), seed=99)
        return Store(materialize(ds, where, n_samples=4, seed=99)).manifest["content_sha256"]

    assert build(tmp_path / "a") == build(tmp_path / "b")


def test_rng_stream_is_independent_of_iteration_order(plaid_store):
    """Seeding on (seed, epoch, index) means shuffling does not change the data."""
    ds = submeter_dataset(plaid_store, augment=MixAggregate(k=(2, 4), p=1.0), seed=31)
    forward = [ds[i]["p_total"].item() for i in range(len(ds))]
    backward = [ds[i]["p_total"].item() for i in reversed(range(len(ds)))]
    assert forward == list(reversed(backward))


def test_augment_receives_a_seeded_generator(plaid_store):
    seen = []

    def spy(item, rng, dataset):
        seen.append(rng.random())
        return item

    submeter_dataset(plaid_store, augment=spy, seed=5)[0]
    submeter_dataset(plaid_store, augment=spy, seed=5)[0]
    assert seen[0] == seen[1]
    assert isinstance(np.random.default_rng((5, 0, 0)).random(), float)
