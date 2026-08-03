"""Combining corpora: what varies, what breaks, and what merging harmonises."""

from __future__ import annotations

import numpy as np
import pytest

from nilmframe import compatibility, merge_stores
from nilmframe.data import HighFreqView, LowFreqView, WindowDataset
from nilmframe.store import ChannelKind, Recording, Store, StoreWriter

# --------------------------------------------------------------------------- #
# What varies
# --------------------------------------------------------------------------- #


def test_report_finds_the_axes_that_differ(plaid_store, whited_store):
    report = compatibility(plaid_store, whited_store)

    assert set(report.datasets) == {"plaid", "whited"}
    assert report.axis("fs").varies, "6 kHz PLAID vs 44.1 kHz WHITED"
    assert report.axis("dataset").varies
    assert len(report.axis("appliance vocabulary").values) > 3


def test_alignment_is_what_makes_rates_compatible(plaid_store, whited_store):
    """The strongest practical argument for the representation, stated as a check.

    Sampling rate and mains frequency block a raw-waveform view and stop blocking
    once cycles are aligned, because alignment resamples every cycle onto the same
    grid regardless of how many samples it took to record.
    """
    report = compatibility(plaid_store, whited_store)

    blocked_raw = {a.name for a in report.blocking("highfreq_raw")}
    blocked_aligned = {a.name for a in report.blocking("highfreq_aligned")}

    assert "fs" in blocked_raw
    assert "fs" not in blocked_aligned
    assert report.is_compatible("highfreq_aligned")
    assert not report.is_compatible("highfreq_raw")


def test_report_accepts_a_view_object(plaid_store, whited_store):
    report = compatibility(plaid_store, whited_store)
    assert report.is_compatible(HighFreqView(align="fitps"))
    assert not report.is_compatible(HighFreqView(align=None))


def test_quantities_narrow_a_waveform_view_rather_than_breaking_it(ukdale_store):
    """UK-DALE mixes a 16 kHz waveform with 1/6 Hz meter channels in one store.

    A waveform view cannot read the meter channels, but the dataset *skips* them
    rather than failing -- so this narrows the view, it does not block it. Saying
    "blocked" would send you looking for a problem that is not there.
    """
    report = compatibility(ukdale_store)
    assert report.axis("quantities").varies

    assert "quantities" not in {a.name for a in report.blocking("highfreq_aligned")}
    assert "quantities" in {a.name for a in report.partial("highfreq_aligned")}

    total = len(report.channels)
    assert 0 < report.usable("highfreq_aligned") < total
    assert report.usable("lowfreq") == total, "a power view reads both kinds"
    assert report.is_compatible("highfreq_aligned")
    assert "skipped" in report.summary()


def test_a_view_no_channel_can_serve_is_not_compatible(tmp_path):
    with StoreWriter(tmp_path / "lf") as w:
        w.add(
            Recording(
                dataset="d",
                house="h",
                session="s",
                kind=ChannelKind.SUBMETER,
                appliance="kettle",
                signals={"p": np.ones(500, np.float32)},
                fs=1.0,
            )
        )
    report = compatibility(Store(tmp_path / "lf"))
    assert report.usable("highfreq_aligned") == 0
    assert not report.is_compatible("highfreq_aligned")
    assert report.is_compatible("lowfreq")


def test_supply_voltage_is_sampled_and_clustered(tmp_path):
    """120 V and 230 V are different supplies; 228 V and 231 V are one."""
    fs, n = 6000.0, 6000
    t = np.arange(n) / fs

    def channel(vrms):
        return Recording(
            dataset="d",
            house="h",
            session="s",
            kind=ChannelKind.SUBMETER,
            appliance="kettle",
            signals={
                "v": (vrms * np.sqrt(2) * np.sin(2 * np.pi * 50 * t)).astype(np.float32),
                "i": (5 * np.sin(2 * np.pi * 50 * t)).astype(np.float32),
            },
            fs=fs,
        )

    with StoreWriter(tmp_path / "eu") as w:
        w.add(channel(230.0))
        w.add(channel(231.0))
    with StoreWriter(tmp_path / "us") as w:
        w.add(channel(120.0))

    same = compatibility(Store(tmp_path / "eu"))
    assert not same.axis("supply voltage").varies, "228-231 V is one supply"

    mixed = compatibility(Store(tmp_path / "eu"), Store(tmp_path / "us"))
    assert mixed.axis("supply voltage").varies
    assert "supply voltage" in {a.name for a in mixed.blocking("highfreq_aligned")}, (
        "alignment does not fix voltage level"
    )


def test_deep_false_skips_reading_signals(plaid_store):
    report = compatibility(plaid_store, deep=False)
    assert "supply voltage" not in [a.name for a in report.axes]


def test_summary_and_frame_render(plaid_store, whited_store):
    report = compatibility(plaid_store, whited_store)
    text = report.summary()
    assert "highfreq_aligned" in text and "highfreq_raw" in text
    assert set(report.to_frame().columns) >= {"axis", "distinct", "blocks", "note"}


def test_compatibility_needs_a_store():
    with pytest.raises(ValueError, match="at least one store"):
        compatibility()


# --------------------------------------------------------------------------- #
# Merging
# --------------------------------------------------------------------------- #


def test_merge_combines_two_corpora(plaid_store, whited_store, tmp_path):
    merged = merge_stores([plaid_store, whited_store], tmp_path / "merged")

    assert len(merged) == len(plaid_store) + len(whited_store)
    assert set(merged.datasets) == {"plaid", "whited"}
    assert set(merged.appliances) == set(plaid_store.appliances) | set(whited_store.appliances)
    assert merged.verify(deep=True) == []
    assert merged.manifest["merged_from"]


def test_merge_prefixes_channel_ids_to_avoid_collisions(tmp_path):
    """Two corpora may both call a channel `house_1-mains`."""
    fs, n = 100.0, 500

    def build(where, dataset):
        with StoreWriter(where) as w:
            w.add(
                Recording(
                    dataset=dataset,
                    house="house_1",
                    session="s",
                    kind=ChannelKind.SUBMETER,
                    appliance="kettle",
                    signals={"p": np.ones(n, np.float32)},
                    fs=fs,
                ),
                channel_id="house_1-mains",
            )
        return Store(where)

    merged = merge_stores(
        [build(tmp_path / "a", "alpha"), build(tmp_path / "b", "beta")], tmp_path / "m"
    )
    assert len(merged) == 2
    assert sorted(merged.channels["channel_id"]) == ["alpha-house_1-mains", "beta-house_1-mains"]


def test_rename_harmonises_the_label_space(tmp_path):
    """Two names for one appliance become two classes unless mapped together."""
    fs, n = 100.0, 500

    def build(where, dataset, appliance):
        with StoreWriter(where) as w:
            w.add(
                Recording(
                    dataset=dataset,
                    house="h",
                    session="s",
                    kind=ChannelKind.SUBMETER,
                    appliance=appliance,
                    signals={"p": np.full(n, 50.0, np.float32)},
                    fs=fs,
                )
            )
        return Store(where)

    a = build(tmp_path / "a", "alpha", "refrigerator")
    b = build(tmp_path / "b", "beta", "fridge")

    naive = merge_stores([a, b], tmp_path / "naive")
    assert set(naive.appliances) == {"refrigerator", "fridge"}, "two classes for one appliance"

    harmonised = merge_stores([a, b], tmp_path / "harmonised", rename={"refrigerator": "fridge"})
    assert harmonised.appliances == ["fridge"]
    assert set(harmonised.channels["appliance"]) == {"fridge"}


def test_rename_also_applies_to_activations(plaid_store, tmp_path):
    merged = merge_stores([plaid_store], tmp_path / "m", rename={"kettle": "water_heater"})
    assert "water_heater" in merged.appliances
    assert "kettle" not in set(merged.activations["appliance"])


def test_normalize_voltage_preserves_power(tmp_path):
    """The load's power is a property of the load, not of the supply it met."""
    fs, n = 6000.0, 6000
    t = np.arange(n) / fs

    def build(where, vrms, dataset):
        with StoreWriter(where) as w:
            w.add(
                Recording(
                    dataset=dataset,
                    house="h",
                    session="s",
                    kind=ChannelKind.SUBMETER,
                    appliance="kettle",
                    signals={
                        "v": (vrms * np.sqrt(2) * np.sin(2 * np.pi * 50 * t)).astype(np.float32),
                        # 2300 W on either supply: 10 A at 230 V, 19.17 A at 120 V.
                        "i": ((2300 / vrms) * np.sqrt(2) * np.sin(2 * np.pi * 50 * t)).astype(
                            np.float32
                        ),
                    },
                    fs=fs,
                )
            )
        return Store(where)

    eu = build(tmp_path / "eu", 230.0, "eu")
    us = build(tmp_path / "us", 120.0, "us")

    merged = merge_stores([eu, us], tmp_path / "m", normalize_voltage=230.0)

    for cid in merged.channels["channel_id"]:
        v = np.asarray(merged.signal(cid, "v"), dtype=np.float64)
        i = np.asarray(merged.signal(cid, "i"), dtype=np.float64)
        assert np.sqrt((v**2).mean()) == pytest.approx(230.0, rel=1e-3)
        assert (v * i).mean() == pytest.approx(2300.0, rel=1e-2), "power must be preserved"

    assert not compatibility(merged).axis("supply voltage").varies


def test_require_refuses_a_merge_that_breaks_a_rule(plaid_store, whited_store, tmp_path):
    with pytest.raises(ValueError, match="fs differs across the inputs"):
        merge_stores([plaid_store, whited_store], tmp_path / "m", require=["fs"])

    # ...and permits it when the rule is satisfied.
    merged = merge_stores([plaid_store, whited_store], tmp_path / "ok", require=["quantities"])
    assert len(merged) > 0


def test_require_voltage_is_satisfied_by_normalising(tmp_path):
    fs, n = 6000.0, 6000
    t = np.arange(n) / fs

    def build(where, vrms, dataset):
        with StoreWriter(where) as w:
            w.add(
                Recording(
                    dataset=dataset,
                    house="h",
                    session="s",
                    kind=ChannelKind.SUBMETER,
                    appliance="kettle",
                    signals={
                        "v": (vrms * np.sqrt(2) * np.sin(2 * np.pi * 50 * t)).astype(np.float32),
                        "i": np.sin(2 * np.pi * 50 * t).astype(np.float32),
                    },
                    fs=fs,
                )
            )
        return Store(where)

    eu, us = build(tmp_path / "eu", 230.0, "eu"), build(tmp_path / "us", 120.0, "us")

    with pytest.raises(ValueError, match="voltage differs"):
        merge_stores([eu, us], tmp_path / "bad", require=["voltage"])

    merged = merge_stores([eu, us], tmp_path / "good", require=["voltage"], normalize_voltage=230.0)
    assert len(merged) == 2


def test_require_rejects_an_unknown_axis(plaid_store, tmp_path):
    with pytest.raises(ValueError, match="unknown axis"):
        merge_stores([plaid_store], tmp_path / "m", require=["colour"])


def test_merge_records_its_rules(plaid_store, whited_store, tmp_path):
    merged = merge_stores(
        [plaid_store, whited_store],
        tmp_path / "m",
        rename={"fan": "ventilator"},
        normalize_voltage=230.0,
    )
    rules = merged.manifest["merge_rules"]
    assert rules["rename"] == {"fan": "ventilator"}
    assert rules["normalize_voltage"] == 230.0
    assert len(merged.manifest["merged_from"]) == 2


def test_merge_needs_a_source(tmp_path):
    with pytest.raises(ValueError, match="at least one source"):
        merge_stores([], tmp_path / "m")


# --------------------------------------------------------------------------- #
# The merged store is a normal store
# --------------------------------------------------------------------------- #


def test_a_merged_store_trains_across_both_corpora(plaid_store, whited_store, tmp_path):
    """The payoff: one dataset, one label space, cycle alignment doing the work."""
    from nilmframe.data import CrossDataset

    merged = merge_stores([plaid_store, whited_store], tmp_path / "m", normalize_voltage=230.0)
    view = HighFreqView(n_cycles=4, cycle_size=32, align="fitps")

    ds = WindowDataset(merged, merged.submeters()["channel_id"].tolist(), view=view)
    assert len(ds) > 0

    rates = merged.channels["fs"].nunique()
    assert rates > 1, "the point is that two different rates coexist"
    item = ds[0]
    assert item["i"].shape == (4, 32), "alignment gives one shape regardless of rate"

    split = CrossDataset(train_on=["plaid"], test_on=["whited"]).apply(merged)
    assert split.train and split.val


def test_merged_windows_have_one_shape_across_rates(plaid_store, whited_store, tmp_path):
    merged = merge_stores([plaid_store, whited_store], tmp_path / "m")
    ds = WindowDataset(
        merged,
        merged.submeters()["channel_id"].tolist(),
        view=HighFreqView(n_cycles=3, cycle_size=64, align="fitps"),
    )
    shapes = {tuple(ds[k]["i"].shape) for k in range(len(ds))}
    assert shapes == {(3, 64)}

    channels = merged.channels.set_index("channel_id")
    seen = {float(channels.loc[ds[k]["channel"], "fs"]) for k in range(len(ds))}
    assert len(seen) > 1, "windows really do come from both rates"


def test_lowfreq_view_spans_a_merged_store(ukdale_store, plaid_store, tmp_path):
    """A power view reads both a waveform corpus and a meter corpus."""
    merged = merge_stores([ukdale_store, plaid_store], tmp_path / "m")
    ds = WindowDataset(
        merged,
        merged.submeters()["channel_id"].tolist(),
        view=LowFreqView(rate_hz=1 / 6, n_steps=8),
    )
    assert len(ds) > 0
    assert ds[0]["p"].shape == (8,)


def test_voltage_scale_is_measured_over_whole_cycles(tmp_path):
    """RMS over a partial mains cycle is biased, and the bias lands in the scale.

    Regression: the head was a flat 4096 samples, which at 6 kHz/50 Hz is 34.1
    cycles -- the trailing 0.1 of a cycle skewed the RMS enough to leave the
    merged channel 0.12% off its target.
    """
    fs, f0 = 6000.0, 50.0
    n = 6000
    t = np.arange(n) / fs
    with StoreWriter(tmp_path / "src") as w:
        w.add(
            Recording(
                dataset="d",
                house="h",
                session="s",
                kind=ChannelKind.SUBMETER,
                appliance="kettle",
                signals={
                    "v": (240.0 * np.sqrt(2) * np.sin(2 * np.pi * f0 * t)).astype(np.float32),
                    "i": np.sin(2 * np.pi * f0 * t).astype(np.float32),
                },
                fs=fs,
                f0=f0,
            )
        )

    merged = merge_stores([Store(tmp_path / "src")], tmp_path / "m", normalize_voltage=230.0)
    cid = merged.channels["channel_id"].iloc[0]
    v = np.asarray(merged.signal(cid, "v"), dtype=np.float64)
    assert np.sqrt((v**2).mean()) == pytest.approx(230.0, rel=1e-5)
