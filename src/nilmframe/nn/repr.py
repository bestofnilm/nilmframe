"""Representations: waveform to model input.

Every transform here is a batched ``nn.Module`` accepting ``(..., T)`` with
arbitrary leading dimensions, so the *same object* runs per-sample inside a
DataLoader worker and per-batch on the GPU. That is what lets training and
deployment share one code path rather than two implementations that drift.

Two consequences of that contract, both deliberate:

* **No numba.** ``legacy/features/_images.py`` built VI trajectory images with an
  ``@njit`` loop over samples, which cannot batch, cannot run on a GPU and cannot
  carry gradients. :class:`VITrajectory` here is ``scatter_``-based: batched,
  differentiable in the values it writes, and roughly two orders of magnitude
  faster on a batch.
* **Shape-stable.** A transform's output shape depends only on its constructor
  arguments and the input shape, never on the input's contents, so a model built
  around one can be traced and compiled.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

__all__ = [
    "DFIA",
    "PAA",
    "DistanceMatrix",
    "Downsample",
    "Fryze",
    "HarmonicLowpass",
    "Patchify",
    "ReIm",
    "Spectrogram",
    "StandardScale",
    "VITrajectory",
]


class Fryze(nn.Module):
    """Fryze decomposition: split current into active and non-active parts.

    The active part is the component collinear with the voltage,
    ``i_a = P / V_rms^2 * v``; the remainder ``i - i_a`` carries the reactive and
    distortion content. Feeding a model both parts separates "how much power" from
    "what shape", which is most of what distinguishes appliance classes.

    Ports ``legacy/features/_decomposition.py``, which reduced over *every* axis so
    a batch shared one scalar power. Here the reduction is over the last axis, so
    each item -- and each cycle, if cycles are the last axis -- is decomposed on
    its own terms.

    Args:
        eps: floor on the mean-square voltage, so a dead channel divides safely.

    Example:
        >>> m = nf.example_measurement('fridge').aligned(cycle_size=64)
        >>> out = nf.nn.Fryze()(m.v, m.i)
        >>> tuple(out.shape)
        (23, 3, 64)
        >>> float((m.v * out[:, 2]).mean().abs()) < 1e-3
        True
    """

    def __init__(self, eps: float = 1e-9) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, v: Tensor, i: Tensor) -> Tensor:
        """Decompose a current against its voltage.

        Args:
            v: ``(..., T)`` voltage.
            i: ``(..., T)`` current, the same shape.

        Returns:
            ``(..., 3, T)`` stacking voltage, active current and non-active
            current. The active part carries all the real power; the non-active
            part carries none, by construction.
        """
        p = (v * i).mean(-1, keepdim=True)
        v2 = v.pow(2).mean(-1, keepdim=True)
        i_active = p / (v2 + self.eps) * v
        return torch.stack((v, i_active, i - i_active), dim=-2)


class HarmonicLowpass(nn.Module):
    """Keep the first ``n_harmonics`` harmonics of each cycle.

    Only meaningful on cycle-aligned input, where bin *k* of the per-cycle FFT is
    the *k*-th harmonic of the mains. On unaligned input the bins smear across
    harmonics, which is one concrete reason alignment is worth its cost.

    Args:
        n_harmonics: harmonics to retain, counting the DC term as the first.

    Example:
        >>> m = nf.example_measurement('laptop').aligned(cycle_size=128)
        >>> filtered = nf.nn.HarmonicLowpass(n_harmonics=3)(m.i)
        >>> tuple(filtered.shape)
        (23, 128)
    """

    def __init__(self, n_harmonics: int = 16) -> None:
        super().__init__()
        if n_harmonics < 1:
            raise ValueError(f"n_harmonics must be >= 1, got {n_harmonics}")
        self.n_harmonics = n_harmonics

    def forward(self, x: Tensor) -> Tensor:
        """Keep the low harmonics of every cycle.

        Args:
            x: ``(..., cycle_size)``. Meaningful on cycle-aligned input, where bin
                *k* of the per-cycle FFT is the *k*-th harmonic of the mains.

        Returns:
            ``(..., cycle_size)`` -- the same shape, high harmonics removed.
        """
        # Slice rather than zero in place. `irfft` zero-pads back to `n`, so the
        # two are numerically identical, but writing into the spectrum mutates a
        # tensor autograd is tracking -- which raises on some torch versions and
        # silently works on others. Out-of-place is correct on all of them.
        spectrum = torch.fft.rfft(x, dim=-1)[..., : self.n_harmonics]
        return torch.fft.irfft(spectrum, n=x.shape[-1], dim=-1)


class ReIm(nn.Module):
    """Concatenated real and imaginary Fourier coefficients of each cycle.

    Args:
        n_components: coefficients kept per part, so the output width is
            ``2 * n_components``.

    Example:
        >>> tuple(nf.nn.ReIm(n_components=8)(torch.randn(4, 128)).shape)
        (4, 16)
    """

    def __init__(self, n_components: int = 16) -> None:
        super().__init__()
        self.n_components = n_components

    def forward(self, x: Tensor) -> Tensor:
        """Fourier coefficients as real numbers a network can consume.

        Args:
            x: ``(..., T)``.

        Returns:
            ``(..., 2 * n_components)`` -- real parts followed by imaginary ones.
            Short input is zero-padded rather than truncated, so the width is
            fixed whatever the input length.
        """
        z = torch.fft.rfft(x, norm="forward", dim=-1)[..., : self.n_components]
        if z.shape[-1] < self.n_components:
            pad = self.n_components - z.shape[-1]
            z = F.pad(z, (0, pad))
        return torch.cat((z.real, z.imag), dim=-1)


class Spectrogram(nn.Module):
    """Log-power STFT magnitude.

    Args:
        window_size: STFT window length.
        hop_size: hop between windows.
        n_fft: frequency bins kept.
        power: return decibels rather than magnitude.

    Example:
        >>> tuple(nf.nn.Spectrogram(window_size=64, hop_size=16)(torch.randn(2, 512)).shape)
        (2, 33, 33)
    """

    def __init__(
        self,
        window_size: int = 256,
        hop_size: int = 64,
        n_fft: int | None = None,
        power: bool = True,
        eps: float = 1e-9,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.hop_size = hop_size
        self.n_fft = n_fft
        self.power = power
        self.eps = eps
        self.register_buffer("window", torch.hann_window(window_size), persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        """Short-time Fourier magnitude.

        Args:
            x: ``(..., T)`` with arbitrary leading dimensions.

        Returns:
            ``(..., freq, time)``. ``freq`` is ``window_size // 2 + 1``, or
            ``n_fft`` when that was given; ``time`` follows from ``hop_size``.
        """
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        spec = torch.stft(
            flat,
            self.window_size,
            hop_length=self.hop_size,
            window=self.window.to(x.dtype),
            return_complex=True,
            normalized=True,
            center=True,
        ).abs()
        if self.power:
            spec = 10 * torch.log10(spec.clamp_min(self.eps))
        if self.n_fft is not None:
            spec = spec[:, : self.n_fft]
        # Build the target shape as a list rather than starring a slice:
        # TorchScript cannot statically infer the size of `*shape[:-1]`, and these
        # transforms are required to be scriptable so they can be exported with
        # the model rather than reimplemented at the serving boundary.
        out_shape: list[int] = list(shape[:-1]) + [spec.shape[-2], spec.shape[-1]]
        return spec.reshape(out_shape)


class DFIA(nn.Module):
    """Double Fourier Integral Analysis of the instantaneous power matrix.

    Builds the outer product ``v_m i_n`` per frame and takes its 2-D FFT, giving a
    joint voltage-current spectral signature.

    Args:
        n_fft: optional ``(rows, cols)`` crop of the 2-D spectrum.

    Example:
        >>> tuple(nf.nn.DFIA(n_fft=(4, 4))(torch.randn(2, 32), torch.randn(2, 32)).shape)
        (2, 2, 4, 4)
    """

    def __init__(self, n_fft: tuple[int, int] | None = None) -> None:
        super().__init__()
        self.n_fft = n_fft

    def forward(self, v: Tensor, i: Tensor) -> Tensor:
        """Two-dimensional spectrum of the instantaneous power matrix.

        Args:
            v: ``(..., T)`` voltage.
            i: ``(..., T)`` current.

        Returns:
            ``(..., 2, rows, cols)`` -- magnitude and phase of the 2-D FFT of the
            outer product ``v_m i_n``. ``rows`` and ``cols`` are ``T`` unless
            ``n_fft`` crops them.
        """
        power = v.unsqueeze(-1) * i.unsqueeze(-2)
        z = torch.fft.fft2(power, norm="forward")
        if self.n_fft is not None:
            z = z[..., : self.n_fft[0], : self.n_fft[1]]
        return torch.stack((z.abs(), z.angle()), dim=-3)


class VITrajectory(nn.Module):
    """VI trajectory image: the current-voltage orbit rasterised to a grid.

    Three channels, following ``legacy/features/_images.py``: occupancy, local
    trajectory slope, and instantaneous power.

    The legacy implementation was an ``@njit`` Python loop over individual samples
    that could only ever process one recording at a time on a CPU. This version
    scatters into a flattened canvas, so it batches, runs on the GPU and stays
    inside the autograd graph.

    Args:
        image_size: output resolution.
        normalize: scale each item to its own peak before rasterising, so shape
            rather than magnitude drives the image.

    Returns:
        ``(..., 3, image_size, image_size)`` in ``[0, 1]``.

    Example:
        >>> m = nf.example_measurement().aligned(cycle_size=64)
        >>> tuple(nf.nn.VITrajectory(image_size=32)(m.v, m.i).shape)
        (23, 3, 32, 32)
    """

    def __init__(self, image_size: int = 64, normalize: bool = True, eps: float = 1e-9) -> None:
        super().__init__()
        self.image_size = image_size
        self.normalize = normalize
        self.eps = eps

    def forward(self, v: Tensor, i: Tensor) -> Tensor:
        """Rasterise the current-voltage orbit.

        Args:
            v: ``(..., T)`` voltage.
            i: ``(..., T)`` current.

        Returns:
            ``(..., 3, image_size, image_size)`` in ``[0, 1]``: occupancy, local
            trajectory slope, and instantaneous power. Where the orbit revisits a
            pixel the strongest value wins, not whichever sample came last.
        """
        shape = v.shape
        v_flat = v.reshape(-1, shape[-1])
        i_flat = i.reshape(-1, shape[-1])
        b = v_flat.shape[0]
        n = self.image_size

        if self.normalize:
            v_flat = v_flat / (v_flat.abs().amax(-1, keepdim=True) + self.eps)
            i_flat = i_flat / (i_flat.abs().amax(-1, keepdim=True) + self.eps)

        x = (((v_flat + 1) / 2) * (n - 1)).round().clamp(0, n - 1).long()
        y = ((1 - (i_flat + 1) / 2) * (n - 1)).round().clamp(0, n - 1).long()
        cell = y * n + x  # (b, t) flat pixel index

        # Slope of the trajectory, mapped to [0, 1]; the final sample repeats the
        # previous slope so the channel is the same length as the orbit.
        dv = v_flat[:, 1:] - v_flat[:, :-1]
        di = i_flat[:, 1:] - i_flat[:, :-1]
        slope = torch.where(
            dv.abs() > self.eps,
            torch.atan(di / torch.where(dv.abs() > self.eps, dv, torch.ones_like(dv))) / torch.pi
            + 0.5,
            torch.full_like(dv, 0.5),
        )
        slope = torch.cat((slope, slope[:, -1:]), dim=1)

        power = v_flat * i_flat
        lo = power.amin(-1, keepdim=True)
        hi = power.amax(-1, keepdim=True)
        power = (power - lo) / (hi - lo + self.eps)

        # Out-of-place scatters, built separately and stacked. Scattering in place
        # into slices of one canvas bumps the version counter of a tensor autograd
        # still needs, so the backward pass raises -- and silently having no
        # gradient here would defeat the point of leaving numba behind.
        blank = v_flat.new_zeros(b, n * n)
        occupancy = blank.scatter(1, cell, torch.ones_like(v_flat))
        # `amax` rather than overwrite: where the orbit revisits a pixel, keep the
        # strongest value instead of whichever sample happened to come last.
        slope_map = blank.scatter_reduce(1, cell, slope, reduce="amax", include_self=True)
        power_map = blank.scatter_reduce(1, cell, power, reduce="amax", include_self=True)
        canvas = torch.stack((occupancy, slope_map, power_map), dim=1)
        out_shape: list[int] = list(shape[:-1]) + [3, n, n]
        return canvas.reshape(out_shape)


class DistanceMatrix(nn.Module):
    """Pairwise absolute-difference matrix of a signal.

    A self-similarity view: entry ``(m, n)`` is ``|x_m - x_n|``. Periodic structure
    shows up as diagonal banding, which is what makes it a useful 2-D input to a
    convolutional encoder.

    The matrix is ``T x T``, so cost grows with the square of the input length --
    reach for :class:`PAA` first if ``T`` is large.

    Example:
        >>> tuple(nf.nn.DistanceMatrix()(torch.randn(2, 16)).shape)
        (2, 16, 16)
    """

    def __init__(self) -> None:
        # Explicit even though it takes nothing: without it autodoc renders the
        # class signature as nn.Module's `(*args, **kwargs)`, which tells a reader
        # nothing at all.
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        """Build the self-similarity matrix.

        Args:
            x: ``(..., T)``.

        Returns:
            ``(..., T, T)``, symmetric, with a zero diagonal.
        """
        return (x.unsqueeze(-1) - x.unsqueeze(-2)).abs()


class PAA(nn.Module):
    """Piecewise aggregate approximation: mean-pool the last axis to ``size``.

    Ports ``legacy/features/_preprocessing.py``'s Python loop to a single pooling
    call, which also fixes its silent truncation when the length was not divisible
    by the target.

    Args:
        size: output length.

    Example:
        >>> tuple(nf.nn.PAA(8)(torch.ones(3, 100)).shape)
        (3, 8)
    """

    def __init__(self, size: int) -> None:
        super().__init__()
        self.size = size

    def forward(self, x: Tensor) -> Tensor:
        """Mean-pool the last axis to a fixed length.

        Args:
            x: ``(..., T)``. ``T`` need not divide ``size``.

        Returns:
            ``(..., size)``.
        """
        shape = x.shape
        flat = x.reshape(-1, 1, shape[-1])
        out_shape: list[int] = list(shape[:-1]) + [self.size]
        return F.adaptive_avg_pool1d(flat, self.size).reshape(out_shape)


class Downsample(nn.Module):
    """Adaptive average pooling of the last axis.

    Identical in effect to :class:`PAA`. Both names exist because the NILM
    literature says "piecewise aggregate approximation" for the representation and
    "downsampling" for the plumbing, and a pipeline reads better when it says which
    one it means.

    Args:
        size: output length.

    Example:
        >>> tuple(nf.nn.Downsample(16)(torch.randn(2, 5, 128)).shape)
        (2, 5, 16)
    """

    def __init__(self, size: int) -> None:
        super().__init__()
        self.size = size

    def forward(self, x: Tensor) -> Tensor:
        """Mean-pool the last axis to a fixed length.

        Args:
            x: ``(..., T)``. ``T`` need not divide ``size``.

        Returns:
            ``(..., size)``.
        """
        shape = x.shape
        flat = x.reshape(-1, 1, shape[-1])
        out_shape: list[int] = list(shape[:-1]) + [self.size]
        return F.adaptive_avg_pool1d(flat, self.size).reshape(out_shape)


class Patchify(nn.Module):
    """Cut a signal into sliding patches, for a sequence model.

    Args:
        patch_size: samples per patch.
        stride: hop between patches. Defaults to ``patch_size``, which gives
            non-overlapping patches.

    Example:
        >>> tuple(nf.nn.Patchify(patch_size=16, stride=8)(torch.randn(2, 64)).shape)
        (2, 7, 16)
    """

    def __init__(self, patch_size: int, stride: int | None = None) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride or patch_size

    def forward(self, x: Tensor) -> Tensor:
        """Extract the patches.

        Args:
            x: ``(..., T)``.

        Returns:
            ``(..., n_patches, patch_size)`` where ``n_patches`` is
            ``(T - patch_size) // stride + 1``. A trailing remainder shorter than
            one patch is dropped.
        """
        return x.unfold(-1, self.patch_size, self.stride)


class StandardScale(nn.Module):
    """Affine normalisation by fixed statistics.

    Statistics are registered as buffers, so they move with ``.to(device)`` and are
    saved in the checkpoint -- the legacy version held plain Python floats or bare
    tensors, which silently stayed on the CPU and vanished from ``state_dict``.

    Example:
        >>> scale = nf.nn.StandardScale(mean=5.0, std=2.0)
        >>> scale(torch.full((3,), 7.0))
        tensor([1., 1., 1.])
    """

    def __init__(self, mean: float | Tensor = 0.0, std: float | Tensor = 1.0) -> None:
        super().__init__()
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.as_tensor(std, dtype=torch.float32))

    def forward(self, x: Tensor) -> Tensor:
        """Normalise by the stored statistics.

        Args:
            x: any shape broadcastable against ``mean`` and ``std``.

        Returns:
            The same shape as ``x``.
        """
        return (x - self.mean.to(x.dtype)) / self.std.to(x.dtype).clamp_min(1e-12)
