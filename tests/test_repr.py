"""Representation transforms: the batched-nn.Module contract.

Every transform must accept arbitrary leading dimensions, so the same object works
per-sample in a worker and per-batch on a GPU. These tests pin that contract, since
breaking it is silent until a shape error appears three layers away.
"""

from __future__ import annotations

import math

import pytest
import torch

from nilmframe.nn.repr import (
    DFIA,
    PAA,
    DistanceMatrix,
    Downsample,
    Fryze,
    HarmonicLowpass,
    Patchify,
    ReIm,
    Spectrogram,
    StandardScale,
    VITrajectory,
)


def _irfft_backward_works() -> bool:
    """Some oneMKL builds cannot differentiate `irfft` at all.

    Observed on torch 2.5.1+cpu on Linux: even the trivial
    `irfft(rfft(x)).sum().backward()` raises "Intel oneMKL DFTI ERROR:
    Inconsistent configuration parameters". The forward pass is fine. This is a
    property of the installed FFT backend, not of the transform, so the check is
    scoped to exactly that primitive rather than skipping the whole test.
    """
    probe = torch.randn(2, 16, requires_grad=True)
    try:
        torch.fft.irfft(torch.fft.rfft(probe, dim=-1), n=16, dim=-1).sum().backward()
    except RuntimeError:
        return False
    return True


IRFFT_BACKWARD = _irfft_backward_works()


def sine(n: int = 128, f: float = 1.0, phase: float = 0.0) -> torch.Tensor:
    m = torch.arange(n, dtype=torch.float32) / n
    return torch.sin(2 * math.pi * f * m + phase)


# --------------------------------------------------------------------------- #
# The contract: leading dimensions are free
# --------------------------------------------------------------------------- #

ONE_ARG = [
    HarmonicLowpass(4),
    ReIm(8),
    PAA(16),
    Downsample(16),
    Patchify(16, 8),
    StandardScale(0.0, 1.0),
    DistanceMatrix(),
    Spectrogram(window_size=32, hop_size=16),
]
TWO_ARG = [Fryze(), VITrajectory(image_size=16), DFIA(n_fft=(4, 4))]


@pytest.mark.parametrize("transform", ONE_ARG, ids=lambda t: type(t).__name__)
@pytest.mark.parametrize("lead", [(), (3,), (2, 5)], ids=str)
def test_one_arg_transforms_accept_any_leading_dims(transform, lead):
    x = torch.randn(*lead, 128)
    out = transform(x)
    assert out.shape[: len(lead)] == lead
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("transform", TWO_ARG, ids=lambda t: type(t).__name__)
@pytest.mark.parametrize("lead", [(), (3,), (2, 5)], ids=str)
def test_two_arg_transforms_accept_any_leading_dims(transform, lead):
    v, i = torch.randn(*lead, 64), torch.randn(*lead, 64)
    out = transform(v, i)
    assert out.shape[: len(lead)] == lead
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("transform", ONE_ARG, ids=lambda t: type(t).__name__)
def test_batching_is_equivalent_to_looping(transform):
    """A batch must not be able to leak across items."""
    items = [torch.randn(128) for _ in range(4)]
    batched = transform(torch.stack(items))
    for k, item in enumerate(items):
        assert torch.allclose(batched[k], transform(item), atol=1e-5)


@pytest.mark.parametrize("transform", TWO_ARG, ids=lambda t: type(t).__name__)
def test_two_arg_batching_is_equivalent_to_looping(transform):
    vs = [torch.randn(64) for _ in range(4)]
    is_ = [torch.randn(64) for _ in range(4)]
    batched = transform(torch.stack(vs), torch.stack(is_))
    for k in range(4):
        assert torch.allclose(batched[k], transform(vs[k], is_[k]), atol=1e-5)


@pytest.mark.parametrize("transform", ONE_ARG + TWO_ARG, ids=lambda t: type(t).__name__)
def test_every_transform_is_torchscript_scriptable(transform):
    """plan.html's contract test.

    Scriptability is what lets a representation be exported *with* the model
    rather than reimplemented at the serving boundary, which is where training
    and inference silently diverge. It fails on `reshape(*shape[:-1], ...)`, so
    every transform builds its output shape as an explicit list.
    """
    scripted = torch.jit.script(transform)
    args = [torch.randn(2, 64) for _ in range(2 if transform in TWO_ARG else 1)]
    assert torch.allclose(scripted(*args), transform(*args), atol=1e-5)


@pytest.mark.parametrize("transform", ONE_ARG + TWO_ARG, ids=lambda t: type(t).__name__)
def test_output_shape_depends_only_on_input_shape(transform):
    """Shape must not depend on the input's *contents*, or nothing can be traced."""
    n = 2 if transform in TWO_ARG else 1
    zeros = transform(*[torch.zeros(3, 64) for _ in range(n)])
    noise = transform(*[torch.randn(3, 64) * 1000 for _ in range(n)])
    assert zeros.shape == noise.shape


@pytest.mark.parametrize("transform", ONE_ARG + TWO_ARG, ids=lambda t: type(t).__name__)
def test_gradients_flow(transform):
    if isinstance(transform, HarmonicLowpass) and not IRFFT_BACKWARD:
        pytest.skip("this torch build cannot differentiate irfft; see _irfft_backward_works")
    args = [torch.randn(2, 64, requires_grad=True) for _ in range(2 if transform in TWO_ARG else 1)]
    out = transform(*args)
    out.sum().backward()
    assert all(a.grad is not None and torch.isfinite(a.grad).all() for a in args)


# --------------------------------------------------------------------------- #
# Behaviour
# --------------------------------------------------------------------------- #


def test_fryze_splits_current_into_orthogonal_parts():
    """The non-active part must carry no power: that is the definition."""
    v = 230 * sine(256)
    i = 2 * sine(256, phase=0.7)  # phase-shifted, so partly non-active
    out = Fryze()(v, i)
    v_out, active, non_active = out[0], out[1], out[2]

    assert torch.allclose(v_out, v)
    assert torch.allclose(active + non_active, i, atol=1e-5)
    assert (v * non_active).mean().abs() < 1e-3 * (v * i).mean().abs()
    assert torch.allclose((v * active).mean(), (v * i).mean(), rtol=1e-4)


def test_fryze_decomposes_each_item_on_its_own_terms():
    """Legacy reduced over every axis, so a batch shared one scalar power."""
    v = torch.stack([230 * sine(128), 230 * sine(128)])
    i = torch.stack([1.0 * sine(128), 50.0 * sine(128)])
    out = Fryze()(v, i)
    assert not torch.allclose(out[0, 1], out[1, 1])
    # Compare the signals directly; an elementwise ratio is 0/0 at every crossing.
    assert torch.allclose(out[1, 1], 50.0 * out[0, 1], rtol=1e-3, atol=1e-4)


def test_fryze_is_a_pure_resistive_identity():
    v = 230 * sine(128)
    i = 3 * sine(128)  # in phase: entirely active
    out = Fryze()(v, i)
    assert out[2].abs().max() < 1e-4


def test_harmonic_lowpass_removes_high_harmonics():
    x = sine(128, f=1.0) + 0.5 * sine(128, f=9.0)
    out = HarmonicLowpass(n_harmonics=4)(x)
    assert torch.allclose(out, sine(128, f=1.0), atol=1e-4)


def test_reim_width_is_fixed():
    assert ReIm(8)(torch.randn(3, 128)).shape == (3, 16)
    assert ReIm(8)(torch.randn(3, 6)).shape == (3, 16), "short input must be padded, not truncated"


def test_paa_handles_indivisible_lengths():
    """The legacy loop silently dropped the remainder."""
    out = PAA(7)(torch.ones(100))
    assert out.shape == (7,)
    assert torch.allclose(out, torch.ones(7))


def test_patchify_shape():
    assert Patchify(16, 8)(torch.randn(2, 64)).shape == (2, 7, 16)


def test_distance_matrix_is_symmetric_with_zero_diagonal():
    d = DistanceMatrix()(torch.randn(2, 16))
    assert torch.allclose(d, d.transpose(-1, -2))
    assert torch.allclose(torch.diagonal(d, dim1=-2, dim2=-1), torch.zeros(2, 16))


def test_vi_trajectory_shape_and_range():
    v, i = 230 * sine(512), 2 * sine(512, phase=0.4)
    img = VITrajectory(image_size=32)(v, i)
    assert img.shape == (3, 32, 32)
    assert img.min() >= 0 and img.max() <= 1
    assert img[0].sum() > 0, "the occupancy channel must mark the orbit"


def test_vi_trajectory_distinguishes_resistive_from_reactive():
    """A resistive load traces a line; a reactive one traces an ellipse."""
    v = 230 * sine(512)
    resistive = VITrajectory(image_size=32)(v, 2 * sine(512))
    reactive = VITrajectory(image_size=32)(v, 2 * sine(512, phase=1.2))
    assert reactive[0].sum() > 1.5 * resistive[0].sum()


def test_vi_trajectory_is_normalised_per_item():
    """Scaling a load must not change its trajectory shape."""
    v = 230 * sine(512)
    small = VITrajectory(image_size=32)(v, 0.1 * sine(512, phase=0.5))
    large = VITrajectory(image_size=32)(v, 90.0 * sine(512, phase=0.5))
    assert torch.equal(small[0], large[0])


def test_spectrogram_finds_the_tone():
    fs, n = 1000, 1024
    t = torch.arange(n) / fs
    x = torch.sin(2 * math.pi * 125 * t)
    spec = Spectrogram(window_size=128, hop_size=64, power=False)(x)
    peak_bin = spec.mean(-1).argmax().item()
    assert abs(peak_bin * fs / 128 - 125) < 12


def test_dfia_shape():
    out = DFIA(n_fft=(4, 6))(torch.randn(2, 3, 32), torch.randn(2, 3, 32))
    assert out.shape == (2, 3, 2, 4, 6)


def test_standard_scale_buffers_move_with_the_module():
    scale = StandardScale(mean=5.0, std=2.0)
    assert "mean" in dict(scale.named_buffers())
    assert torch.allclose(scale(torch.full((4,), 7.0)), torch.ones(4))


def test_standard_scale_survives_a_state_dict_round_trip():
    """Legacy held plain floats, which never reached the checkpoint."""
    original = StandardScale(mean=3.0, std=4.0)
    revived = StandardScale()
    revived.load_state_dict(original.state_dict())
    assert torch.allclose(revived.mean, torch.tensor(3.0))


def test_harmonic_lowpass_rejects_bad_argument():
    with pytest.raises(ValueError, match="n_harmonics"):
        HarmonicLowpass(0)


def test_transforms_compose_into_a_sequential_pipeline():
    """`nn.Sequential` composition is the point of making these modules."""
    pipeline = torch.nn.Sequential(HarmonicLowpass(8), PAA(32), StandardScale(0.0, 1.0))
    out = pipeline(torch.randn(4, 10, 128))
    assert out.shape == (4, 10, 32)
