"""Input adapters and encoder backbones -- the high-frequency path.

The models in :mod:`nilmframe.nn.models` read a low-rate power series: ``(B, C, L)``
watts, one step per meter reading. A waveform corpus does not arrive in that shape.
It arrives as voltage and current at kilohertz, aligned into cycles, and getting
from one to the other is what this module is for.

An *input adapter* turns a view's item dict into a tensor. A *backbone* turns that
tensor into a fixed-width embedding. Splitting them is what lets one backbone serve
both arms of a low-frequency/high-frequency comparison: the adapter absorbs the
difference in what the store hands over, and the tensor reaching the encoder has
the same rank either way.

Nothing here decides what an appliance is drawing. These are transforms and
encoders; a model is what puts a number on the end, and
:class:`~nilmframe.nn.models.Faustine` and :class:`~nilmframe.nn.models.Schirmer`
are the two that do it for this path.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from nilmframe.nn.repr import Fryze

__all__ = [
    "AvgSeqAdaptivePool",
    "ConvNet1d",
    "ConvNet2d",
    "CycleInput",
    "CycleTransformer",
    "SequenceInput",
    "faustine_cnn",
    "schirmer_cnn",
]


# --------------------------------------------------------------------------- #
# Input adapters
# --------------------------------------------------------------------------- #


class CycleInput(nn.Module):
    """Item dict to ``(B, C, n_cycles, cycle_size)`` for a high-frequency view.

    Args:
        mode:
            ``"vi"`` -- two channels, voltage and current.
            ``"i"`` -- current only, the cheapest useful input.
            ``"fryze"`` -- three channels: voltage, active current, non-active
            current. Separating the collinear part from the rest hands the model
            "how much power" and "what shape" as different channels.
        normalize: divide the current by its per-item peak, so the encoder sees
            shape and the power head sees magnitude. Scale information is not lost
            -- it reaches the head through ``p_total``.
        mask_padding: zero cycles that alignment could not fill.

    Example:
        >>> adapter = nf.nn.CycleInput('fryze')
        >>> adapter.channels
        3
        >>> batch = nf.example_measurement().aligned(cycle_size=64).batch()
        >>> tuple(adapter(batch).shape)
        (1, 3, 23, 64)
    """

    def __init__(self, mode: str = "vi", normalize: bool = True, mask_padding: bool = True) -> None:
        super().__init__()
        if mode not in ("vi", "i", "fryze"):
            raise ValueError(f"mode must be 'vi', 'i' or 'fryze', got {mode!r}")
        self.mode = mode
        self.normalize = normalize
        self.mask_padding = mask_padding
        self.fryze = Fryze() if mode == "fryze" else None

    @property
    def channels(self) -> int:
        return {"vi": 2, "i": 1, "fryze": 3}[self.mode]

    def forward(self, batch: dict) -> Tensor:
        """Turn a high-frequency item into an encoder input.

        Args:
            batch: an item or batch carrying ``v`` and ``i``, and optionally
                ``cycle_mask``.

        Returns:
            ``(B, C, n_cycles, cycle_size)`` where ``C`` is :attr:`channels`.
        """
        v, i = batch["v"], batch["i"]
        if v.ndim == 2:  # a single item; add the batch axis
            v, i = v.unsqueeze(0), i.unsqueeze(0)

        if self.normalize:
            scale = i.abs().amax(dim=(-2, -1), keepdim=True).clamp_min(1e-9)
            i = i / scale
            v = v / v.abs().amax(dim=(-2, -1), keepdim=True).clamp_min(1e-9)

        if self.mode == "i":
            x = i.unsqueeze(1)
        elif self.mode == "vi":
            x = torch.stack((v, i), dim=1)
        else:
            x = self.fryze(v, i).movedim(-2, 1)

        if self.mask_padding and "cycle_mask" in batch:
            mask = batch["cycle_mask"]
            if mask.ndim == 1:
                mask = mask.unsqueeze(0)
            x = x * mask[:, None, :, None].to(x.dtype)
        return x


class SequenceInput(nn.Module):
    """Item dict to ``(B, 1, n_steps)`` for a low-frequency view.

    Args:
        normalize: divide by the window's own mean power, so the encoder sees the
            load's *shape* over time rather than its scale -- the same convention
            as :class:`CycleInput`, which keeps the two arms comparable.

    Example:
        >>> adapter = nf.nn.SequenceInput()
        >>> adapter.channels
        1
        >>> tuple(adapter({'p': torch.rand(2, 60) * 500}).shape)
        (2, 1, 60)
    """

    def __init__(self, normalize: bool = True) -> None:
        super().__init__()
        self.normalize = normalize

    @property
    def channels(self) -> int:
        return 1

    def forward(self, batch: dict) -> Tensor:
        """Turn a low-frequency item into an encoder input.

        Args:
            batch: an item or batch carrying ``p``.

        Returns:
            ``(B, 1, n_steps)``.
        """
        p = batch["p"]
        if p.ndim == 1:
            p = p.unsqueeze(0)
        if self.normalize:
            p = p / p.abs().mean(-1, keepdim=True).clamp_min(1e-9)
        return p.unsqueeze(1)


# --------------------------------------------------------------------------- #
# Backbones
# --------------------------------------------------------------------------- #


class ConvNet2d(nn.Module):
    """Strided 2-D convolutional encoder over ``(B, C, n_cycles, cycle_size)``.

    Generalises the two published CNNs; :func:`faustine_cnn` and
    :func:`schirmer_cnn` are presets.

    Args:
        in_channels: input channels, from the adapter.
        widths: channel width per block.
        kernel: convolution kernel size.
        stride: stride per block.
        out_features: embedding width.
        dropout: dropout before the projection.

    Example:
        >>> backbone = nf.nn.ConvNet2d(in_channels=2, out_features=32)
        >>> tuple(backbone(torch.randn(2, 2, 10, 64)).shape)
        (2, 32)
    """

    def __init__(
        self,
        in_channels: int = 2,
        widths: tuple[int, ...] = (16, 32, 64, 128),
        kernel: int = 3,
        stride: int = 2,
        out_features: int = 256,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        blocks: list[nn.Module] = []
        c_in = in_channels
        for width in widths:
            blocks += [
                nn.Conv2d(c_in, width, kernel_size=kernel, stride=stride, padding=kernel // 2),
                nn.BatchNorm2d(width),
                nn.ReLU(inplace=True),
            ]
            c_in = width
        self.features = nn.Sequential(*blocks, nn.AdaptiveAvgPool2d(1), nn.Flatten(1))
        self.project = nn.Sequential(
            nn.Linear(c_in, out_features),
            nn.LayerNorm(out_features),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.out_features = out_features

    def forward(self, x: Tensor) -> Tensor:
        """Encode an input into a fixed-width embedding.

        Args:
            x: ``(B, C, H, W)`` for the 2-D encoders, ``(B, C, T)`` for the 1-D one.
                The spatial extent is free -- the encoder pools adaptively.

        Returns:
            ``(B, out_features)``.
        """
        return self.project(self.features(x))


def faustine_cnn(in_channels: int = 2, out_features: int = 256) -> ConvNet2d:
    """The Faustine architecture: four strided blocks, 16 to 128 channels.

    A wide 5x5 kernel and stride-2 downsampling at every block, so the receptive
    field covers a whole cycle within three blocks. The heavier of the two ported
    CNNs; start here when the signature is carried by the shape of the waveform.

    Args:
        in_channels: channels from the adapter -- 2 for ``vi``, 3 for ``fryze``.
        out_features: embedding width handed to the head.

    Returns:
        A configured :class:`ConvNet2d`.

    Example:
        >>> backbone = nf.nn.faustine_cnn(in_channels=2, out_features=16)
        >>> tuple(backbone(torch.randn(2, 2, 10, 64)).shape)
        (2, 16)
    """
    return ConvNet2d(in_channels, widths=(16, 32, 64, 128), kernel=5, out_features=out_features)


def schirmer_cnn(in_channels: int = 2, out_features: int = 256) -> ConvNet2d:
    """The Schirmer architecture: three narrow same-padded blocks.

    Eight channels throughout and no downsampling, so it keeps full resolution and
    stays small -- roughly a tenth of :func:`faustine_cnn`'s parameters. Useful as
    a baseline, and when the training set is small enough that the larger model
    only overfits faster.

    Args:
        in_channels: channels from the adapter -- 2 for ``vi``, 3 for ``fryze``.
        out_features: embedding width handed to the head.

    Returns:
        A configured :class:`ConvNet2d`.

    Example:
        >>> backbone = nf.nn.schirmer_cnn(in_channels=2, out_features=16)
        >>> tuple(backbone(torch.randn(2, 2, 10, 64)).shape)
        (2, 16)
    """
    return ConvNet2d(in_channels, widths=(8, 8, 8), kernel=3, stride=1, out_features=out_features)


class AvgSeqAdaptivePool(nn.Module):
    """Pool a sequence to a fixed length by averaging over the sequence axis.

    ``legacy/nilm/models/_cold.py`` referenced this class but it existed nowhere in
    the tree, so ``COLD`` could not be instantiated. Reconstructed from its use:
    it takes ``(B, S, D)`` and returns ``(B, pool_size, D)``.

    Example:
        >>> tuple(nf.nn.AvgSeqAdaptivePool(3)(torch.randn(2, 12, 8)).shape)
        (2, 3, 8)
    """

    def __init__(self, pool_size: int) -> None:
        super().__init__()
        self.pool_size = pool_size

    def forward(self, x: Tensor) -> Tensor:
        """Shorten a sequence by averaging over its length.

        Args:
            x: ``(B, S, D)``.

        Returns:
            ``(B, pool_size, D)``.
        """
        return F.adaptive_avg_pool1d(x.transpose(1, 2), self.pool_size).transpose(1, 2)

    def extra_repr(self) -> str:
        return f"pool_size={self.pool_size}"


class CycleTransformer(nn.Module):
    """Transformer encoder over the cycle sequence -- the COLD architecture.

    Each mains cycle is a token. Progressive pooling between blocks shortens the
    sequence, so attention cost falls as depth grows.

    Args:
        in_channels: channels from the adapter; they are flattened into the token.
        cycle_size: samples per cycle, so the token width is
            ``in_channels * cycle_size``.
        hidden: model width.
        n_head: attention heads.
        pools: sequence length after each block.
        dropout: dropout inside the encoder layers.
        out_features: embedding width.

    Example:
        >>> backbone = nf.nn.CycleTransformer(in_channels=2, cycle_size=64, hidden=32,
        ...                                   n_head=4, pools=(2, 1), out_features=32)
        >>> tuple(backbone(torch.randn(2, 2, 8, 64)).shape)
        (2, 32)
    """

    def __init__(
        self,
        in_channels: int = 2,
        cycle_size: int = 128,
        hidden: int = 256,
        n_head: int = 8,
        pools: tuple[int, ...] = (8, 4, 1),
        dropout: float = 0.1,
        out_features: int = 256,
    ) -> None:
        super().__init__()
        self.embed = nn.Linear(in_channels * cycle_size, hidden)
        layers: list[nn.Module] = []
        for pool in pools:
            layers.append(
                nn.TransformerEncoderLayer(
                    d_model=hidden,
                    nhead=n_head,
                    dim_feedforward=4 * hidden,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
            )
            layers.append(AvgSeqAdaptivePool(pool))
        self.blocks = nn.Sequential(*layers)
        self.norm = nn.LayerNorm(hidden)
        self.project = nn.Linear(hidden * pools[-1], out_features)
        self.out_features = out_features

    def forward(self, x: Tensor) -> Tensor:
        """Encode a cycle sequence with attention.

        Args:
            x: ``(B, C, n_cycles, cycle_size)``. Each cycle becomes one token of
                width ``C * cycle_size``.

        Returns:
            ``(B, out_features)``.
        """
        b, c, n_cycles, cycle_size = x.shape
        tokens = x.permute(0, 2, 1, 3).reshape(b, n_cycles, c * cycle_size)
        h = self.blocks(self.embed(tokens))
        return self.project(self.norm(h).flatten(1))


class ConvNet1d(nn.Module):
    """Dilated 1-D convolutional encoder for low-frequency power sequences.

    Dilation rather than striding, so a 60-second window is covered by a receptive
    field wide enough to see a whole appliance cycle without throwing away
    resolution at the switching edges, which is where the label information is.

    Example:
        >>> backbone = nf.nn.ConvNet1d(in_channels=1, out_features=32)
        >>> tuple(backbone(torch.randn(2, 1, 60)).shape)
        (2, 32)
    """

    def __init__(
        self,
        in_channels: int = 1,
        widths: tuple[int, ...] = (32, 64, 128),
        kernel: int = 5,
        out_features: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        blocks: list[nn.Module] = []
        c_in = in_channels
        for depth, width in enumerate(widths):
            dilation = 2**depth
            blocks += [
                nn.Conv1d(
                    c_in,
                    width,
                    kernel_size=kernel,
                    padding=dilation * (kernel // 2),
                    dilation=dilation,
                ),
                nn.BatchNorm1d(width),
                nn.GELU(),
            ]
            c_in = width
        self.features = nn.Sequential(*blocks, nn.AdaptiveAvgPool1d(1), nn.Flatten(1))
        self.project = nn.Sequential(
            nn.Linear(c_in, out_features),
            nn.LayerNorm(out_features),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.out_features = out_features

    def forward(self, x: Tensor) -> Tensor:
        """Encode an input into a fixed-width embedding.

        Args:
            x: ``(B, C, H, W)`` for the 2-D encoders, ``(B, C, T)`` for the 1-D one.
                The spatial extent is free -- the encoder pools adaptively.

        Returns:
            ``(B, out_features)``.
        """
        return self.project(self.features(x))
