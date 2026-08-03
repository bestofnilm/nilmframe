"""The architectures that are not a convolutional stem with a head on it.

A U-Net, a dilated causal stack, a temporal-pooling encoder-decoder, an attention
model, and two transformers. They share the contract in
:mod:`nilmframe.nn.models.base` and nothing else, which is the point of having the
contract.

Every one of these is a reimplementation from the paper, not a port of the
authors' code, and the hyper-parameters are the papers' defaults where the paper
states them. Where a paper leaves something unstated -- and several do -- the
choice is marked in the class it appears in rather than buried here.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from nilmframe.nn.models.base import NILMModel, Standardiser

__all__ = ["BERT4NILM", "TPNILM", "AttentionNILM", "ELECTRIcity", "UNetNILM", "WaveNILM"]


# --------------------------------------------------------------------------- #
# U-Net
# --------------------------------------------------------------------------- #


class _Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel: int = 3) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel, padding="same"),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel, padding="same"),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class UNetNILM(NILMModel):
    """Faustine et al. (BuildSys 2020). A 1D U-Net.

    Encoder, decoder, and skip connections carrying the encoder's activations
    across to the decoder at the same resolution. The skips are what the paper is
    about: pooling down to a coarse representation is what lets the model see a
    long context, and it is also what destroys the sharp edge of a switching
    event -- the skip puts the edge back.

    Args:
        n_appliances: size of the label space.
        window: input and output length. Must be divisible by ``2 ** depth``.
        in_channels: measured quantities per step.
        width: channels at the first level, doubling each level down.
        depth: number of pooling steps.
        standardiser: input/output scaling.

    Example:
        >>> model = nf.nn.models.UNetNILM(3, window=64, depth=2, width=8)
        >>> tuple(model(torch.rand(2, 1, 64) * 500).shape)
        (2, 3, 64)
    """

    kind = "seq2seq"

    def __init__(
        self,
        n_appliances: int,
        *,
        window: int = 512,
        in_channels: int = 1,
        width: int = 16,
        depth: int = 4,
        standardiser: Standardiser | None = None,
    ) -> None:
        super().__init__(
            n_appliances, window=window, in_channels=in_channels, standardiser=standardiser
        )
        if window % (2**depth):
            raise ValueError(f"window {window} is not divisible by 2 ** depth = {2**depth}")
        self.depth = depth

        self.downs = nn.ModuleList()
        channels = in_channels
        widths = []
        for level in range(depth):
            out = width * 2**level
            self.downs.append(_Down(channels, out))
            widths.append(out)
            channels = out
        self.bottom = _Down(channels, channels * 2)
        channels *= 2

        self.ups = nn.ModuleList()
        self.merges = nn.ModuleList()
        for out in reversed(widths):
            self.ups.append(nn.ConvTranspose1d(channels, out, 2, stride=2))
            self.merges.append(_Down(out * 2, out))
            channels = out
        self.out = nn.Conv1d(channels, n_appliances, 1)

    def encode(self, x: Tensor) -> Tensor:
        skips = []
        h = x
        for down in self.downs:
            h = down(h)
            skips.append(h)
            h = nn.functional.max_pool1d(h, 2)
        h = self.bottom(h)
        for up, merge, skip in zip(self.ups, self.merges, reversed(skips), strict=True):
            h = merge(torch.cat([up(h), skip], dim=1))
        return self.out(h)


# --------------------------------------------------------------------------- #
# WaveNet-style causal dilations
# --------------------------------------------------------------------------- #


class WaveNILM(NILMModel):
    """Harell, Makonin & Bajić (ICASSP 2019). Causal dilated convolutions.

    A stack whose dilation doubles each layer, so the receptive field grows
    exponentially in depth while every output still depends only on the past. The
    causality is not decoration -- the paper's case is online disaggregation, where
    a model that peeks forward cannot be deployed.

    It is also the one model here that takes the *complex* power seriously.
    ``in_channels=2`` feeds it active and reactive power together, which is what
    separates loads that draw the same watts at different power factors.

    Args:
        n_appliances: size of the label space.
        window: input and output length.
        in_channels: measured quantities per step. 2 for (P, Q).
        width: channels per residual block.
        layers: number of blocks; the receptive field is ``2 ** layers``.
        standardiser: input/output scaling.

    Example:
        >>> model = nf.nn.models.WaveNILM(3, window=64, in_channels=2, layers=4)
        >>> tuple(model(torch.rand(2, 2, 64) * 500).shape)
        (2, 3, 64)
    """

    kind = "seq2seq"

    def __init__(
        self,
        n_appliances: int,
        *,
        window: int = 512,
        in_channels: int = 1,
        width: int = 64,
        layers: int = 6,
        kernel: int = 3,
        standardiser: Standardiser | None = None,
    ) -> None:
        super().__init__(
            n_appliances, window=window, in_channels=in_channels, standardiser=standardiser
        )
        self.kernel = kernel
        self.dilations = [2**i for i in range(layers)]
        self.entry = nn.Conv1d(in_channels, width, 1)
        self.filters = nn.ModuleList(
            nn.Conv1d(width, width, kernel, dilation=d) for d in self.dilations
        )
        self.gates = nn.ModuleList(
            nn.Conv1d(width, width, kernel, dilation=d) for d in self.dilations
        )
        self.residuals = nn.ModuleList(nn.Conv1d(width, width, 1) for _ in self.dilations)
        self.out = nn.Sequential(nn.ReLU(inplace=True), nn.Conv1d(width, n_appliances, 1))

    def encode(self, x: Tensor) -> Tensor:
        h = self.entry(x)
        skip = torch.zeros_like(h)
        for dilation, filt, gate, residual in zip(
            self.dilations, self.filters, self.gates, self.residuals, strict=True
        ):
            # Pad on the left only: that is what makes the convolution causal.
            padded = nn.functional.pad(h, (dilation * (self.kernel - 1), 0))
            activated = torch.tanh(filt(padded)) * torch.sigmoid(gate(padded))
            h = h + residual(activated)
            skip = skip + activated
        return self.out(skip)


# --------------------------------------------------------------------------- #
# Temporal pooling
# --------------------------------------------------------------------------- #


class TPNILM(NILMModel):
    """Massidda, Marrocco & Manca (Applied Sciences 2020). Temporal pooling.

    An encoder of strided convolutions, then a pyramid that pools the encoded
    sequence at several scales and concatenates the results back. The pyramid is
    the idea: a fridge cycles on a scale of minutes and a kettle lasts seconds, and
    a single receptive field has to pick one of them to be good at.

    Args:
        n_appliances: size of the label space.
        window: input and output length.
        in_channels: measured quantities per step.
        width: encoder channels.
        scales: pooling factors in the pyramid.
        standardiser: input/output scaling.

    Example:
        >>> model = nf.nn.models.TPNILM(3, window=64, width=16)
        >>> tuple(model(torch.rand(2, 1, 64) * 500).shape)
        (2, 3, 64)
    """

    kind = "seq2seq"

    def __init__(
        self,
        n_appliances: int,
        *,
        window: int = 512,
        in_channels: int = 1,
        width: int = 32,
        scales: tuple[int, ...] = (2, 4, 8),
        standardiser: Standardiser | None = None,
    ) -> None:
        super().__init__(
            n_appliances, window=window, in_channels=in_channels, standardiser=standardiser
        )
        self.scales = scales
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, width, 9, padding="same"),
            nn.BatchNorm1d(width),
            nn.ReLU(inplace=True),
            nn.Conv1d(width, width, 5, padding="same"),
            nn.BatchNorm1d(width),
            nn.ReLU(inplace=True),
        )
        self.branches = nn.ModuleList(
            nn.Sequential(nn.Conv1d(width, width, 1), nn.ReLU(inplace=True)) for _ in scales
        )
        self.decoder = nn.Sequential(
            nn.Conv1d(width * (1 + len(scales)), width, 3, padding="same"),
            nn.ReLU(inplace=True),
            nn.Conv1d(width, n_appliances, 1),
        )

    def encode(self, x: Tensor) -> Tensor:
        h = self.encoder(x)
        pooled = [h]
        for scale, branch in zip(self.scales, self.branches, strict=True):
            # Pool to a coarser view, transform, then put it back on the fine grid
            # so the branches can be concatenated channel-wise.
            coarse = nn.functional.adaptive_avg_pool1d(h, max(1, h.shape[-1] // scale))
            up = nn.functional.interpolate(branch(coarse), size=h.shape[-1], mode="linear")
            pooled.append(up)
        return self.decoder(torch.cat(pooled, dim=1))


# --------------------------------------------------------------------------- #
# Attention
# --------------------------------------------------------------------------- #


class _PositionalEncoding(nn.Module):
    """Sinusoidal positions. Fixed, not learned, so a window length can change."""

    def __init__(self, width: int, max_len: int = 8192) -> None:
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        step = torch.exp(torch.arange(0, width, 2) * (-math.log(10000.0) / width))
        table = torch.zeros(max_len, width)
        table[:, 0::2] = torch.sin(position * step)
        table[:, 1::2] = torch.cos(position * step)
        self.register_buffer("table", table)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.table[: x.shape[1]].unsqueeze(0)


class AttentionNILM(NILMModel):
    """Piccialli & Sudoso (Energies 2021). Convolutions, then attention.

    A convolutional encoder for local shape, then self-attention over the encoded
    sequence for everything else. The argument against a pure CNN is that its
    receptive field is a hyper-parameter you have to guess, and guessing it wrong
    caps what the model can represent; attention lets any step read any other.

    Args:
        n_appliances: size of the label space.
        window: input and output length.
        in_channels: measured quantities per step.
        width: model dimension.
        heads: attention heads.
        layers: encoder layers.
        standardiser: input/output scaling.

    Example:
        >>> model = nf.nn.models.AttentionNILM(3, window=64, width=32, heads=2, layers=1)
        >>> tuple(model(torch.rand(2, 1, 64) * 500).shape)
        (2, 3, 64)
    """

    kind = "seq2seq"

    def __init__(
        self,
        n_appliances: int,
        *,
        window: int = 512,
        in_channels: int = 1,
        width: int = 128,
        heads: int = 4,
        layers: int = 2,
        standardiser: Standardiser | None = None,
    ) -> None:
        super().__init__(
            n_appliances, window=window, in_channels=in_channels, standardiser=standardiser
        )
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, width, 5, padding="same"), nn.ReLU(inplace=True)
        )
        self.positions = _PositionalEncoding(width)
        layer = nn.TransformerEncoderLayer(
            width, heads, dim_feedforward=width * 4, batch_first=True, dropout=0.1
        )
        self.attention = nn.TransformerEncoder(layer, layers)
        self.out = nn.Conv1d(width, n_appliances, 1)

    def encode(self, x: Tensor) -> Tensor:
        h = self.stem(x).transpose(1, 2)
        h = self.attention(self.positions(h)).transpose(1, 2)
        return self.out(h)


# --------------------------------------------------------------------------- #
# Transformers
# --------------------------------------------------------------------------- #


class BERT4NILM(NILMModel):
    """Yue et al. (BuildSys 2020). A bidirectional transformer.

    The framing is borrowed wholesale from language modelling: mask part of the
    aggregate sequence, ask the model to fill it in, and use the representation
    that falls out. :meth:`mask` produces the corrupted input, so the pretraining
    objective is reproducible rather than described.

    Args:
        n_appliances: size of the label space.
        window: input and output length.
        in_channels: measured quantities per step.
        width: model dimension.
        heads: attention heads.
        layers: transformer layers.
        mask_ratio: fraction of steps hidden during pretraining.
        standardiser: input/output scaling.

    Example:
        >>> model = nf.nn.models.BERT4NILM(3, window=64, width=32, heads=2, layers=1)
        >>> tuple(model(torch.rand(2, 1, 64) * 500).shape)
        (2, 3, 64)
        >>> corrupted, hidden = model.mask(torch.rand(2, 1, 64), generator=None)
        >>> bool(hidden.any())
        True
    """

    kind = "seq2seq"

    def __init__(
        self,
        n_appliances: int,
        *,
        window: int = 512,
        in_channels: int = 1,
        width: int = 256,
        heads: int = 2,
        layers: int = 2,
        mask_ratio: float = 0.25,
        standardiser: Standardiser | None = None,
    ) -> None:
        super().__init__(
            n_appliances, window=window, in_channels=in_channels, standardiser=standardiser
        )
        if not 0.0 < mask_ratio < 1.0:
            raise ValueError(f"mask_ratio must be in (0, 1), got {mask_ratio}")
        self.mask_ratio = mask_ratio
        self.embed = nn.Sequential(
            nn.Conv1d(in_channels, width, 5, padding="same"), nn.LeakyReLU(0.1, inplace=True)
        )
        self.positions = _PositionalEncoding(width)
        layer = nn.TransformerEncoderLayer(
            width, heads, dim_feedforward=width * 4, batch_first=True, dropout=0.1
        )
        self.encoder = nn.TransformerEncoder(layer, layers)
        self.out = nn.Sequential(
            nn.Conv1d(width, width // 2, 1),
            nn.Tanh(),
            nn.Conv1d(width // 2, n_appliances, 1),
        )

    def mask(self, x: Tensor, generator: torch.Generator | None = None) -> tuple[Tensor, Tensor]:
        """Hide a fraction of the steps.

        Args:
            x: ``(B, C, L)`` input.
            generator: for a reproducible mask.

        Returns:
            ``(corrupted, hidden)`` -- the input with masked steps zeroed, and the
            ``(B, L)`` boolean mask saying which those were, so a pretraining loss
            can be computed on exactly the hidden positions.
        """
        shape = (x.shape[0], x.shape[-1])
        draw = torch.rand(shape, device=x.device, generator=generator)
        hidden = draw < self.mask_ratio
        return x * (~hidden).unsqueeze(1).to(x.dtype), hidden

    def encode(self, x: Tensor) -> Tensor:
        h = self.embed(x).transpose(1, 2)
        h = self.encoder(self.positions(h)).transpose(1, 2)
        return self.out(h)


class ELECTRIcity(NILMModel):
    """Sykiotis et al. (Sensors 2022). A transformer with a corruption task.

    Where BERT4NILM masks and reconstructs, this pretrains a small generator to
    *replace* steps and a larger discriminator to spot which were replaced -- the
    ELECTRA objective. The claimed benefit is sample efficiency: every step gives
    the discriminator a training signal, not only the masked quarter.

    :meth:`corrupt` implements the replacement so the objective is runnable;
    ``forward`` is the discriminator body, which is what you keep afterwards.

    Args:
        n_appliances: size of the label space.
        window: input and output length.
        in_channels: measured quantities per step.
        width: discriminator dimension.
        heads: attention heads.
        layers: discriminator layers.
        replace_ratio: fraction of steps the generator replaces.
        standardiser: input/output scaling.

    Example:
        >>> model = nf.nn.models.ELECTRIcity(3, window=64, width=32, heads=2, layers=1)
        >>> tuple(model(torch.rand(2, 1, 64) * 500).shape)
        (2, 3, 64)
        >>> corrupted, replaced = model.corrupt(torch.rand(2, 1, 64))
        >>> tuple(replaced.shape)
        (2, 64)
    """

    kind = "seq2seq"

    def __init__(
        self,
        n_appliances: int,
        *,
        window: int = 512,
        in_channels: int = 1,
        width: int = 256,
        heads: int = 2,
        layers: int = 3,
        replace_ratio: float = 0.15,
        standardiser: Standardiser | None = None,
    ) -> None:
        super().__init__(
            n_appliances, window=window, in_channels=in_channels, standardiser=standardiser
        )
        if not 0.0 < replace_ratio < 1.0:
            raise ValueError(f"replace_ratio must be in (0, 1), got {replace_ratio}")
        self.replace_ratio = replace_ratio
        self.embed = nn.Sequential(nn.Conv1d(in_channels, width, 5, padding="same"), nn.GELU())
        self.positions = _PositionalEncoding(width)
        layer = nn.TransformerEncoderLayer(
            width,
            heads,
            dim_feedforward=width * 4,
            batch_first=True,
            dropout=0.1,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, layers)
        self.out = nn.Conv1d(width, n_appliances, 1)
        # The generator is deliberately small -- a strong one produces replacements
        # the discriminator cannot detect, and the task stops teaching anything.
        self.generator = nn.Sequential(
            nn.Conv1d(in_channels, width // 4, 5, padding="same"),
            nn.GELU(),
            nn.Conv1d(width // 4, in_channels, 5, padding="same"),
        )

    def corrupt(self, x: Tensor, generator: torch.Generator | None = None) -> tuple[Tensor, Tensor]:
        """Replace a fraction of the steps with the generator's own guesses.

        Returns:
            ``(corrupted, replaced)`` -- the sequence with some steps swapped, and
            the ``(B, L)`` boolean mask of which, which is the discriminator's
            target.
        """
        shape = (x.shape[0], x.shape[-1])
        draw = torch.rand(shape, device=x.device, generator=generator)
        replaced = draw < self.replace_ratio
        fake = self.generator(x)
        keep = (~replaced).unsqueeze(1).to(x.dtype)
        return x * keep + fake * (1 - keep), replaced

    def encode(self, x: Tensor) -> Tensor:
        h = self.embed(x).transpose(1, 2)
        h = self.encoder(self.positions(h)).transpose(1, 2)
        return self.out(h)
