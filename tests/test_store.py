"""Phase 2 acceptance: the store round-trips, is verifiable, and readers agree.

plan.html's criterion: `nilmframe convert` produces a store, row counts match the
source, checksums are recorded.
"""

from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pytest

from nilmframe.readers import PLAID, WHITED
from nilmframe.store import Activation, ChannelKind, Recording, Store, StoreWriter

# --------------------------------------------------------------------------- #
# Recording validation -- the invariants the old sample class never enforced
# --------------------------------------------------------------------------- #


def _recording(**kw) -> Recording:
    base = {
        "dataset": "t",
        "house": "h",
        "session": "s",
        "kind": ChannelKind.SUBMETER,
        "appliance": "kettle",
        "signals": {"v": np.zeros(100, np.float32), "i": np.ones(100, np.float32)},
        "fs": 1000.0,
    }
    return Recording(**{**base, **kw})


def test_recording_rejects_mismatched_signal_lengths():
    with pytest.raises(ValueError, match="mismatched lengths"):
        _recording(signals={"v": np.zeros(100, np.float32), "i": np.zeros(90, np.float32)})


def test_recording_rejects_non_1d_signals():
    with pytest.raises(ValueError, match="must be 1-D"):
        _recording(signals={"v": np.zeros((2, 100), np.float32)})


def test_recording_rejects_unknown_quantity():
    with pytest.raises(ValueError, match="unknown quantities"):
        _recording(signals={"reactive": np.zeros(100, np.float32)})


def test_recording_rejects_nonpositive_fs():
    with pytest.raises(ValueError, match="fs must be positive"):
        _recording(fs=0.0)


def test_submeter_must_name_its_appliance():
    with pytest.raises(ValueError, match="must name its appliance"):
        _recording(appliance=None)


def test_activation_cannot_run_past_the_recording():
    with pytest.raises(ValueError, match="past the end"):
        _recording(activations=[Activation("kettle", 0, 500)])


def test_activation_cannot_end_before_it_starts():
    with pytest.raises(ValueError, match="ends before it starts"):
        Activation("kettle", 50, 10)


def test_instance_id_falls_back_conservatively():
    """Unknown identity must not let two recordings of one unit split apart."""
    assert _recording(instance_id="abc").resolved_instance_id() == "abc"
    assert _recording(brand="acme").resolved_instance_id() == "kettle:acme"
    assert _recording().resolved_instance_id() == "t:h:s"


def test_signals_are_cast_to_float32():
    rec = _recording(signals={"v": np.zeros(100, np.float64)})
    assert rec.signals["v"].dtype == np.float32


# --------------------------------------------------------------------------- #
# Store round-trip
# --------------------------------------------------------------------------- #


def test_write_read_round_trip(tmp_path):
    rng = np.random.default_rng(0)
    v = rng.normal(size=512).astype(np.float32)
    i = rng.normal(size=512).astype(np.float32)

    with StoreWriter(tmp_path / "s", source="unit-test") as w:
        cid = w.add(_recording(signals={"v": v, "i": i}, fs=6000.0))

    store = Store(tmp_path / "s")
    assert len(store) == 1
    assert store.appliances == ["kettle"]
    assert np.array_equal(np.asarray(store.signal(cid, "v")), v)
    assert np.array_equal(np.asarray(store.signal(cid, "i")), i)
    assert store.verify(deep=True) == []


def test_manifest_records_a_content_hash(tmp_path):
    with StoreWriter(tmp_path / "s") as w:
        w.add(_recording())
    manifest = json.loads((tmp_path / "s" / "manifest.json").read_text())
    assert len(manifest["content_sha256"]) == 64
    assert manifest["n_channels"] == 1
    assert manifest["total_samples"] == 100
    assert manifest["format_version"] >= 1


def test_content_hash_is_stable_and_content_sensitive(tmp_path):
    def build(where, value):
        with StoreWriter(where) as w:
            w.add(_recording(signals={"i": np.full(100, value, np.float32)}))
        return json.loads((where / "manifest.json").read_text())["content_sha256"]

    assert build(tmp_path / "a", 1.0) == build(tmp_path / "b", 1.0)
    assert build(tmp_path / "a2", 1.0) != build(tmp_path / "c", 2.0)


def test_verify_detects_a_corrupted_signal(tmp_path):
    with StoreWriter(tmp_path / "s") as w:
        cid = w.add(_recording())
    store = Store(tmp_path / "s")
    assert store.verify(deep=True) == []

    np.save(store.signal_path(cid, "i"), np.zeros(100, np.float32) + 7)
    assert any("hash mismatch" in p for p in Store(tmp_path / "s").verify(deep=True))


def test_verify_detects_a_truncated_signal(tmp_path):
    with StoreWriter(tmp_path / "s") as w:
        cid = w.add(_recording())
    np.save(Store(tmp_path / "s").signal_path(cid, "i"), np.zeros(7, np.float32))
    assert any("samples on disk" in p for p in Store(tmp_path / "s").verify())


def test_signals_are_memory_mapped_not_loaded():
    """The whole point of the layout: a window read must not load the channel."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        with StoreWriter(tmp, overwrite=True) as w:
            cid = w.add(_recording(signals={"i": np.arange(100_000, dtype=np.float32)}))
        store = Store(tmp)
        array = store.signal(cid, "i")
        assert isinstance(array, np.memmap)
        assert np.array_equal(store.read_window(cid, "i", 500, 4), [500, 501, 502, 503])


def test_read_window_returns_a_writable_copy(tmp_path):
    """Augmentation writes in place; a read-only memmap view would be UB in torch."""
    import torch

    with StoreWriter(tmp_path / "s") as w:
        cid = w.add(_recording(signals={"i": np.ones(100, np.float32)}))
    store = Store(tmp_path / "s")
    chunk = store.read_window(cid, "i", 0, 10)
    assert chunk.flags.writeable

    tensor = torch.from_numpy(chunk)
    tensor += 1.0
    assert np.array_equal(np.asarray(store.signal(cid, "i"))[:10], np.ones(10, np.float32))


def test_read_window_pads_past_the_end(tmp_path):
    with StoreWriter(tmp_path / "s") as w:
        cid = w.add(_recording(signals={"i": np.ones(10, np.float32)}))
    chunk = Store(tmp_path / "s").read_window(cid, "i", 8, 5)
    assert chunk.shape == (5,)
    assert np.array_equal(chunk, [1, 1, 0, 0, 0])


def test_duplicate_channel_ids_are_rejected(tmp_path):
    with StoreWriter(tmp_path / "s") as w:
        w.add(_recording(), channel_id="x")
        with pytest.raises(ValueError, match="duplicate channel_id"):
            w.add(_recording(), channel_id="x")


def test_auto_channel_ids_are_unique(tmp_path):
    with StoreWriter(tmp_path / "s") as w:
        ids = [w.add(_recording()) for _ in range(5)]
    assert len(set(ids)) == 5


def test_refuses_to_clobber_an_existing_store(tmp_path):
    with StoreWriter(tmp_path / "s") as w:
        w.add(_recording())
    with pytest.raises(FileExistsError, match="overwrite=True"):
        StoreWriter(tmp_path / "s")
    StoreWriter(tmp_path / "s", overwrite=True).close()  # allowed when asked


def test_rejects_a_store_from_a_newer_format(tmp_path):
    with StoreWriter(tmp_path / "s") as w:
        w.add(_recording())
    manifest = json.loads((tmp_path / "s" / "manifest.json").read_text())
    manifest["format_version"] = 999
    (tmp_path / "s" / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match="Upgrade nilmframe"):
        Store(tmp_path / "s")


def test_missing_store_raises_clearly(tmp_path):
    with pytest.raises(FileNotFoundError, match=r"manifest\.json is missing"):
        Store(tmp_path / "nope")


def test_f0_is_estimated_from_the_voltage(tmp_path):
    fs, f0 = 6000.0, 50.0
    t = np.arange(6000) / fs
    v = np.sin(2 * np.pi * f0 * t).astype(np.float32)
    with StoreWriter(tmp_path / "s") as w:
        cid = w.add(_recording(signals={"v": v, "i": v.copy()}, fs=fs))
    assert abs(Store(tmp_path / "s").channel(cid)["f0"] - f0) < 1.0


# --------------------------------------------------------------------------- #
# PLAID
# --------------------------------------------------------------------------- #


def test_plaid_reader_row_count_matches_the_source(plaid_source):
    csv_dir, metadata = plaid_source
    reader = PLAID(csv_dir, metadata)
    assert len(list(reader)) == len(reader) == len(json.loads(metadata.read_text()))


def test_plaid_separates_submetered_from_aggregate(plaid_store):
    assert len(plaid_store.submeters()) == 8
    assert len(plaid_store.mains()) == 2
    assert len(plaid_store) == 10


def test_plaid_aggregate_recordings_carry_activations(plaid_store):
    mains = plaid_store.mains()
    assert len(mains) > 0
    for cid in mains["channel_id"]:
        acts = plaid_store.activations_for(cid)
        assert len(acts) >= 2
        assert (acts["off"] > acts["on"]).all()


def test_plaid_column_order_is_current_then_voltage(plaid_store):
    """PLAID's CSV is current-first. Swapping them is silent, so pin it."""
    cid = plaid_store.submeters()["channel_id"].iloc[0]
    v = np.asarray(plaid_store.signal(cid, "v"))
    i = np.asarray(plaid_store.signal(cid, "i"))
    assert v.std() > 100, "voltage should be hundreds of volts"
    assert i.std() < 100, "current should be single-digit amperes"


def test_plaid_labels_are_normalised(plaid_store):
    assert set(plaid_store.appliances) <= {"kettle", "fridge", "laptop", "fan"}
    assert all(a == a.lower() and " " not in a for a in plaid_store.appliances)


def test_plaid_skips_a_recording_whose_csv_is_missing(plaid_source, tmp_path):
    csv_dir, metadata = plaid_source
    meta = json.loads(metadata.read_text())
    meta["9999"] = {"header": {"sampling_frequency": "6000Hz"}, "appliance": {"type": "Ghost"}}
    path = tmp_path / "meta_with_ghost.json"
    path.write_text(json.dumps(meta))
    assert len(list(PLAID(csv_dir, path))) == len(meta) - 1


def test_plaid_activation_without_an_off_mark_runs_to_the_end(tmp_path):
    from synthetic import make_plaid

    csv_dir, metadata = make_plaid(tmp_path / "src", n_aggregate=0)
    meta = json.loads(metadata.read_text())
    n = len(np.loadtxt(csv_dir / "1.csv", delimiter=","))
    meta["1"] = {
        "header": {"sampling_frequency": "6000Hz"},
        "appliances": [{"type": "Kettle", "on": "[10]", "off": ""}],
    }
    metadata.write_text(json.dumps(meta))
    rec = next(r for r in PLAID(csv_dir, metadata) if r.session == "1")
    assert rec.activations == [Activation("kettle", 10, n)]


# --------------------------------------------------------------------------- #
# WHITED
# --------------------------------------------------------------------------- #


def test_whited_applies_kit_calibration(whited_store):
    """Voltage must come back in volts, not normalised PCM."""
    for cid in whited_store.channels["channel_id"]:
        v = np.asarray(whited_store.signal(cid, "v"))
        assert 200 < float(np.sqrt(np.mean(v**2))) < 260


def test_whited_parses_filename_fields(whited_store):
    channels = whited_store.channels.set_index("channel_id")
    assert set(channels["appliance"]) == {"microwave", "vacuum", "fan"}
    assert set(channels["house"]) == {"DE", "AT"}
    assert channels["instance_id"].nunique() == 4


def test_whited_skips_unknown_kits_but_can_be_strict(whited_source, tmp_path):
    import shutil

    dest = tmp_path / "mixed"
    shutil.copytree(whited_source, dest)
    shutil.copy(next(dest.glob("*.flac")), dest / "Toaster_tA_DE_MK9_20150101.flac")

    assert len(list(WHITED(dest))) == 4  # the MK9 file is skipped
    with pytest.raises(ValueError, match="unknown measurement kit"):
        list(WHITED(dest, strict=True))


def test_whited_skips_unparseable_filenames(whited_source, tmp_path):
    import shutil

    dest = tmp_path / "odd"
    shutil.copytree(whited_source, dest)
    shutil.copy(next(dest.glob("*.flac")), dest / "nonsense.flac")
    assert len(list(WHITED(dest))) == 4


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _cli(*args):
    return subprocess.run(
        [sys.executable, "-m", "nilmframe.cli", *args], capture_output=True, text=True
    )


def test_cli_convert_and_describe(plaid_source, tmp_path):
    csv_dir, metadata = plaid_source
    dst = tmp_path / "cli_store"

    out = _cli(
        "convert",
        "plaid",
        "--src",
        str(csv_dir),
        "--metadata",
        str(metadata),
        "--dst",
        str(dst),
        "--verify",
    )
    assert out.returncode == 0, out.stderr
    assert "wrote 10 channels" in out.stdout
    assert "content hash" in out.stdout

    described = _cli("describe", str(dst), "--verify")
    assert described.returncode == 0, described.stderr
    assert "integrity: ok" in described.stdout
    assert "kettle" in described.stdout


def test_cli_convert_requires_metadata_for_plaid(plaid_source, tmp_path):
    csv_dir, _ = plaid_source
    out = _cli("convert", "plaid", "--src", str(csv_dir), "--dst", str(tmp_path / "x"))
    assert out.returncode == 2
    assert "--metadata" in out.stderr


def test_cli_convert_respects_limit(plaid_source, tmp_path):
    csv_dir, metadata = plaid_source
    out = _cli(
        "convert",
        "plaid",
        "--src",
        str(csv_dir),
        "--metadata",
        str(metadata),
        "--dst",
        str(tmp_path / "lim"),
        "--limit",
        "3",
    )
    assert out.returncode == 0, out.stderr
    assert Store(tmp_path / "lim").manifest["n_channels"] == 3


# --------------------------------------------------------------------------- #
# Query surface -- what replaced thirty eager dataset methods
# --------------------------------------------------------------------------- #


def test_queries_are_dataframe_expressions(plaid_store):
    channels = plaid_store.channels

    kettles = channels[channels["appliance"] == "kettle"]
    assert len(kettles) == 3

    counts = channels["appliance"].value_counts()
    common = counts[counts >= 2].index.tolist()
    assert "kettle" in common and "fan" not in common  # drop_rare, as a query

    assert plaid_store.brands == ["acme", "globex", "initech"]
    assert plaid_store.datasets == ["plaid"]


def test_describe_summarises_per_appliance(plaid_store):
    described = plaid_store.describe().set_index("appliance")
    assert described.loc["kettle", "channels"] == 3
    assert described.loc["kettle", "instances"] == 2  # two acme:k100 share an identity
    assert described.loc["kettle", "hours"] > 0


def test_on_thresholds_are_explicit_and_ordered(plaid_store):
    assert plaid_store.on_thresholds.shape == (plaid_store.n_appliances,)
    assert (plaid_store.on_thresholds == 10.0).all()
    assert plaid_store.known_mask.all()
    assert list(plaid_store.appliance_index) == plaid_store.appliances


def test_per_appliance_thresholds_can_be_overridden(tmp_path):
    with StoreWriter(tmp_path / "s", appliance_thresholds={"kettle": 2000.0}) as w:
        w.add(_recording())
    assert Store(tmp_path / "s").on_thresholds[0] == 2000.0


def test_siblings_are_same_house_session_and_dataset(tmp_path):
    with StoreWriter(tmp_path / "s") as w:
        a = w.add(_recording(house="h1", session="s1", appliance="kettle"))
        b = w.add(_recording(house="h1", session="s1", appliance="fridge"))
        w.add(_recording(house="h2", session="s1", appliance="fan"))
    store = Store(tmp_path / "s")
    assert store.siblings(a)["channel_id"].tolist() == [b]
