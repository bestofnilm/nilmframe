"""``Measurement`` -- the interactive object, and the bugs its predecessor had.

`HighFreqSample` was deleted because it *was* the storage and carried mutable
pending state. The ergonomics it offered were worth keeping, so they came back as
a lens over the lazy store. These tests pin the ergonomics, and pin each of the
three bugs that made the original untrustworthy.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest
import torch

from nilmframe import Measurement

FS = 6000.0


def sine(n: int = 6000, f: float = 50.0, amp: float = 1.0, phase: float = 0.0) -> torch.Tensor:
    t = torch.arange(n, dtype=torch.float32) / FS
    return amp * torch.sin(2 * math.pi * f * t + phase)


def kettle(n: int = 6000) -> Measurement:
    """A resistive load: current in phase with the voltage."""
    return Measurement.from_vi(
        230 * math.sqrt(2) * sine(n), sine(n, amp=9.0), FS, f0=50.0, appliances=["kettle"]
    )


def fridge(n: int = 6000) -> Measurement:
    """An inductive load: current lagging."""
    return Measurement.from_vi(
        230 * math.sqrt(2) * sine(n),
        sine(n, amp=0.9, phase=-0.6),
        FS,
        f0=50.0,
        appliances=["fridge"],
    )


# --------------------------------------------------------------------------- #
# Regressions against the predecessor
# --------------------------------------------------------------------------- #


def test_n_components_counts_components_not_samples():
    """The bug that made every recording look aggregated.

    `HighFreqSample.i` summed over components, then `n_components` read
    `self.i.shape[0]` -- which after the sum is the *time* axis. So a
    1000-sample single-appliance recording reported 1000 components, and
    `dataset.submetered()` returned an empty dataset for every input.
    """
    one = kettle()
    assert one.n_components == 1
    assert one.n_samples == 6000

    two = kettle() + fridge()
    assert two.n_components == 2
    assert two.n_samples == 6000


def test_superposition_actually_works():
    """`__superpose__` raised on every call: it never passed `fs` to the
    constructor it built, and compared an `f0` attribute that did not exist."""
    mixture = kettle() + fridge()

    assert mixture.n_components == 2
    assert mixture.appliances == ("kettle", "fridge")
    assert torch.allclose(mixture.i, kettle().i + fridge().i)
    # The aggregate power is the sum of the parts.
    assert torch.allclose(
        mixture.active_power(), kettle().active_power() + fridge().active_power(), rtol=1e-4
    )


def test_components_survive_every_operation():
    """`copy()` collapsed multi-component samples down to a single component."""
    mixture = kettle() + fridge()
    for derived in (
        mixture.window(0, 3000),
        mixture.seconds(0.0, 0.5),
        mixture.aligned(cycle_size=64),
        mixture.aligned(cycle_size=64).lowpass(8),
        mixture.aligned(cycle_size=64).resample(32),
    ):
        assert derived.n_components == 2, derived
        assert derived.appliances == ("kettle", "fridge")


def test_measurements_are_immutable():
    """No `.save()`, no `__v_modified__`: every operation returns a new object.

    The original's methods behaved differently depending on whether invisible
    pending edits existed, which is most of why its bugs were hard to see.
    """
    original = kettle()
    derived = original.aligned(cycle_size=64)

    assert derived is not original
    assert not original.aligned_ and derived.aligned_
    with pytest.raises(dataclasses.FrozenInstanceError):
        original.fs = 999.0


# --------------------------------------------------------------------------- #
# The ergonomics this exists for
# --------------------------------------------------------------------------- #


def test_dot_and_chain():
    result = kettle().seconds(0.0, 0.5).aligned(cycle_size=128).lowpass(6)
    assert result.aligned_
    assert result.cycle_size == 128
    assert 20 <= result.n_cycles <= 26  # 0.5 s of 50 Hz
    assert float(result.active_power()) > 0


def test_signals_are_plain_tensors():
    """`.v` and `.i` are torch tensors, so everything torch still works."""
    m = kettle()
    assert isinstance(m.v, torch.Tensor) and isinstance(m.i, torch.Tensor)
    assert torch.fft.rfft(m.i).shape[0] == m.n_samples // 2 + 1
    assert (m.i * 2).abs().max() == pytest.approx(float(m.i.abs().max()) * 2)


def test_electrical_quantities_are_physical():
    m = kettle()
    assert float(m.vrms) == pytest.approx(230.0, rel=1e-3)
    assert float(m.irms) == pytest.approx(9.0 / math.sqrt(2), rel=1e-3)
    # In phase -> all real power, power factor 1.
    assert float(m.active_power()) == pytest.approx(float(m.apparent_power()), rel=1e-3)
    assert float(m.power_factor()) == pytest.approx(1.0, rel=1e-3)
    assert float(m.reactive_power()) == pytest.approx(0.0, abs=1.0)


def test_reactive_load_has_a_power_factor_below_one():
    assert float(fridge().power_factor()) < 0.9
    assert float(fridge().reactive_power()) > 0


def test_power_breaks_down_per_component():
    """The original's `active_power(multicomponent=True)` silently returned the
    aggregate, because it read an already-summed current."""
    mixture = kettle() + fridge()
    per = mixture.active_power(per_component=True)

    assert per.shape == (2,)
    assert float(per[0]) == pytest.approx(float(kettle().active_power()), rel=1e-4)
    assert float(per[1]) == pytest.approx(float(fridge().active_power()), rel=1e-4)
    assert float(per.sum()) == pytest.approx(float(mixture.active_power()), rel=1e-4)


def test_components_can_be_taken_apart_again():
    mixture = kettle() + fridge()
    parts = mixture.components

    assert [p.appliances[0] for p in parts] == ["kettle", "fridge"]
    assert all(p.n_components == 1 for p in parts)
    assert torch.allclose(sum(parts).i, mixture.i)


def test_repr_says_what_it_is_and_stays_short():
    """A repr people read in a notebook has to fit on a line."""
    raw = repr(kettle())
    aligned = repr(kettle().aligned(cycle_size=64))
    mixed = repr(kettle() + fridge())

    assert "waveform raw" in raw and "kettle" in raw and "W)" in raw
    assert "waveform 48x64" in aligned, aligned
    assert "2 components" in mixed, mixed
    for text in (raw, aligned, mixed):
        assert len(text) <= 80, f"{len(text)} chars: {text}"


# --------------------------------------------------------------------------- #
# Alignment and representations
# --------------------------------------------------------------------------- #


def test_alignment_normalises_off_nominal_mains():
    """50.7 Hz: cycles are not an integer number of samples."""
    drifting = Measurement.from_vi(sine(6000, f=50.7), sine(6000, f=50.7, amp=2.0), FS)
    aligned = drifting.aligned(cycle_size=128)

    assert aligned.aligned_ and aligned.cycle_size == 128
    cycles = aligned.i
    spread = (cycles - cycles.mean(0)).abs().mean()
    assert float(spread) < 1e-3, "aligned cycles of a pure tone should superpose"


def test_harmonics_need_alignment():
    m = kettle()
    with pytest.raises(ValueError, match="aligned"):
        m.harmonics()
    with pytest.raises(ValueError, match="aligned"):
        m.lowpass(4)

    spectrum = m.aligned(cycle_size=128).harmonics(n=8)
    assert spectrum.shape == (8,)
    assert float(spectrum[1]) == pytest.approx(1.0)  # normalised to the fundamental
    assert float(spectrum[3]) < 0.05  # a pure tone has no 3rd harmonic


def test_representations_come_back_as_tensors():
    aligned = kettle().aligned(cycle_size=64)
    assert aligned.fryze().shape[-2:] == (3, 64)
    assert aligned.vi_image(size=32).shape[-3:] == (3, 32, 32)
    assert kettle().spectrogram(window_size=64, hop_size=32).ndim == 2


def test_slicing_before_alignment_only():
    m = kettle().aligned(cycle_size=64)
    with pytest.raises(ValueError, match="cycle indices are not samples"):
        m.window(0, 10)


# --------------------------------------------------------------------------- #
# Power-series measurements
# --------------------------------------------------------------------------- #


def test_power_series_measurement():
    watts = torch.tensor([0.0, 0.0, 2000.0, 2000.0, 0.0, 0.0])
    m = Measurement.from_power(watts, fs=1 / 6, appliances=["kettle"])

    assert m.kind == "power" and not m.is_waveform
    assert m.n_samples == 6
    assert m.duration == pytest.approx(36.0)
    assert float(m.active_power()) == pytest.approx(watts.mean())
    with pytest.raises(AttributeError, match="no voltage"):
        _ = m.v


def test_power_series_superposes():
    a = Measurement.from_power(torch.full((10,), 100.0), fs=1.0, appliances=["a"])
    b = Measurement.from_power(torch.full((10,), 50.0), fs=1.0, appliances=["b"])
    total = a + b
    assert total.n_components == 2
    assert float(total.active_power()) == pytest.approx(150.0)
    assert float(total.active_power(per_component=True).sum()) == pytest.approx(150.0)


def test_cannot_mix_a_waveform_with_a_power_series():
    with pytest.raises(ValueError, match="cannot superpose"):
        kettle() + Measurement.from_power(torch.ones(10), fs=1.0)


def test_superposition_checks_rate_and_shape():
    with pytest.raises(ValueError, match="sampling rates differ"):
        kettle() + Measurement.from_vi(sine(6000), sine(6000), 1000.0)
    with pytest.raises(ValueError, match="shapes differ"):
        kettle() + kettle(3000)


def test_sum_builtin_works():
    """`sum()` starts from 0, so __radd__ has to tolerate it."""
    total = sum([kettle(), fridge()])
    assert isinstance(total, Measurement) and total.n_components == 2


# --------------------------------------------------------------------------- #
# Interop and the store
# --------------------------------------------------------------------------- #


def test_numpy_escape_hatch():
    arrays = kettle().numpy()
    assert isinstance(arrays["v"], np.ndarray) and arrays["fs"] == FS


def test_from_vi_validates_shapes():
    with pytest.raises(ValueError, match="not compatible"):
        Measurement.from_vi(sine(100), torch.zeros(2, 3, 100), FS)
    with pytest.raises(ValueError, match="must match the voltage"):
        Measurement.from_vi(sine(100), torch.zeros(2, 50), FS)


def test_store_hands_back_measurements(plaid_store):
    cid = plaid_store.submeters()["channel_id"].iloc[0]
    m = plaid_store.measurement(cid, seconds=0.5)

    assert m.is_waveform
    assert m.fs == float(plaid_store.channel(cid)["fs"])
    assert m.duration == pytest.approx(0.5, rel=0.01)
    assert m.appliances and m.appliances[0] in plaid_store.appliances
    assert cid in m.source
    assert float(m.vrms) > 100

    aligned = m.aligned(cycle_size=64)
    assert aligned.n_cycles > 10


def test_store_measurement_window_bounds(plaid_store):
    cid = plaid_store.submeters()["channel_id"].iloc[0]
    n = int(plaid_store.channel(cid)["n_samples"])
    assert plaid_store.measurement(cid, samples=100).n_samples == 100
    assert plaid_store.measurement(cid, start=10).n_samples == n - 10
    with pytest.raises(ValueError, match="empty window"):
        plaid_store.measurement(cid, start=n + 5)


def test_dataset_hands_back_measurements(plaid_store):
    from nilmframe.data import HighFreqView, WindowDataset

    ds = WindowDataset(
        plaid_store,
        plaid_store.submeters()["channel_id"].tolist(),
        view=HighFreqView(n_cycles=6, cycle_size=32),
    )
    m = ds.measurement(0)
    assert m.aligned_ and m.n_cycles == 6 and m.cycle_size == 32
    assert torch.allclose(m.i, ds[0]["i"])


def test_plot_returns_axes(plaid_store):
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    cid = plaid_store.submeters()["channel_id"].iloc[0]
    m = plaid_store.measurement(cid, seconds=0.2)
    assert m.plot() is not None
    assert m.aligned(cycle_size=64).plot() is not None


def test_a_mains_channel_has_no_appliance_name(ukdale_store):
    """pandas spells "no appliance" as NaN, which is truthy.

    A bare `if row["appliance"]` therefore carried the float nan through as an
    appliance name, and `repr` showed `appliances=[nan]`.
    """
    mains = ukdale_store.mains()["channel_id"].iloc[0]
    submeter = ukdale_store.submeters()["channel_id"].iloc[0]

    assert ukdale_store.measurement(mains, samples=600).appliances == ()
    assert "nan" not in repr(ukdale_store.measurement(mains, samples=600))

    named = ukdale_store.measurement(submeter, samples=600)
    assert named.appliances and isinstance(named.appliances[0], str)


def test_plot_accepts_a_title_without_forwarding_it_to_the_line(plaid_store):
    """matplotlib rejects unknown artist properties rather than ignoring them.

    Regression: `title=` was popped only after `**kwargs` had already been
    forwarded to `ax.plot`, so `m.plot(title=...)` raised
    "Line2D.set() got an unexpected keyword argument 'title'".
    """
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cid = plaid_store.submeters()["channel_id"].iloc[0]
    m = plaid_store.measurement(cid, seconds=0.2)

    _, ax = plt.subplots()
    assert m.plot(ax=ax, title="custom").get_title() == "custom"

    _, ax2 = plt.subplots()
    assert m.aligned(cycle_size=64).plot(ax=ax2, title="aligned").get_title() == "aligned"

    _, ax3 = plt.subplots()
    lf = Measurement.from_power(torch.arange(20.0), fs=1.0)
    assert lf.plot(ax=ax3, title="power").get_title() == "power"


def test_superposing_power_series_of_different_lengths_says_so():
    """Real channels rarely run for exactly the same time.

    The waveform branch checked shapes; the power branch fell straight through to
    `torch.cat` and produced "Sizes of tensors must match except in dimension 0",
    which does not tell you what to do about it.
    """
    a = Measurement.from_power(torch.ones(620), fs=1 / 6, appliances=["fridge"])
    b = Measurement.from_power(torch.ones(342), fs=1 / 6, appliances=["lights"])

    with pytest.raises(ValueError, match="lengths differ"):
        a + b

    n = min(a.n_samples, b.n_samples)
    combined = a.window(0, n) + b.window(0, n)
    assert combined.n_components == 2 and combined.n_samples == n
