"""The high-frequency path: a model over aligned cycles.

Every other model in this package reads a low-rate power series -- one step per
meter reading, watts. A waveform corpus does not arrive that way. It arrives as
voltage and current at kilohertz, and once :func:`~nilmframe.nn.cycle_align` has
put each mains cycle on a common grid it is a *two-dimensional* thing: cycles by
samples-within-a-cycle. Flattening that to a series would throw away the axis that
makes it worth having.

So this model takes the aligned tensor as it is. The contract still holds --
watts per appliance out, the shape declared rather than implied -- but the input
rank is four instead of three, and :attr:`~NILMModel.input_rank` says so.

It is a point model by construction. An aligned window is one observation of a
load, so there is one power vector to predict, not one per step; ``L_out`` is 1
and the extra axis is kept only so the output shape matches everything else.

The encoder is whatever you hand it. :class:`~nilmframe.nn.ConvNet2d` treats the
cycle grid as an image, :class:`~nilmframe.nn.CycleTransformer` treats each cycle
as a token, and :func:`~nilmframe.nn.faustine_cnn` and
:func:`~nilmframe.nn.schirmer_cnn` are the two published presets.
"""

from __future__ import annotations

from torch import Tensor, nn

from nilmframe.nn.encoders import ConvNet2d, faustine_cnn, schirmer_cnn
from nilmframe.nn.models.base import NILMModel, Standardiser

__all__ = ["CycleCNN", "faustine", "schirmer"]


class CycleCNN(NILMModel):
    """Per-appliance watts from a window of aligned mains cycles.

    Args:
        n_appliances: size of the label space, ``K``.
        cycles: cycles per window.
        cycle_size: samples per cycle, after alignment.
        in_channels: channels the adapter produces -- 1 for ``"i"``, 2 for
            ``"vi"``, 3 for ``"fryze"``.
        encoder: a module mapping ``(B, C, cycles, cycle_size)`` to
            ``(B, out_features)``. Defaults to a :class:`~nilmframe.nn.ConvNet2d`.
        out_features: encoder width, when the default encoder is used.
        standardiser: input/output scaling.

    Example:
        >>> model = nf.nn.models.CycleCNN(3, cycles=8, cycle_size=64, out_features=32)
        >>> model.kind, model.input_rank
        ('seq2point', 4)
        >>> tuple(model(torch.rand(2, 2, 8, 64)).shape)
        (2, 3, 1)

    Note:
        The adapter is separate on purpose. :class:`~nilmframe.nn.CycleInput` turns
        a store's item dict into the tensor this expects, and keeping it outside
        the model is what lets the same encoder read ``vi`` on one corpus and
        ``fryze`` on another without the model knowing.

        >>> adapter = nf.nn.CycleInput("fryze")
        >>> batch = nf.example_measurement().aligned(cycle_size=64).batch()
        >>> x = adapter(batch)
        >>> model = nf.nn.models.CycleCNN(4, cycles=x.shape[2], cycle_size=64,
        ...                               in_channels=adapter.channels, out_features=16)
        >>> tuple(model(x).shape)
        (1, 4, 1)
    """

    kind = "seq2point"
    input_rank = 4

    def __init__(
        self,
        n_appliances: int,
        *,
        cycles: int = 20,
        cycle_size: int = 128,
        in_channels: int = 2,
        encoder: nn.Module | None = None,
        out_features: int = 256,
        standardiser: Standardiser | None = None,
    ) -> None:
        # `window` in the base is the last axis; for this path that is the cycle.
        super().__init__(
            n_appliances,
            window=cycle_size,
            in_channels=in_channels,
            standardiser=standardiser,
        )
        if cycles < 1:
            raise ValueError(f"cycles must be >= 1, got {cycles}")
        self.cycles = cycles
        self.cycle_size = cycle_size
        self.encoder = encoder or ConvNet2d(in_channels, out_features=out_features)
        width = getattr(self.encoder, "out_features", out_features)
        self.project = nn.Linear(width, n_appliances)

    def encode(self, x: Tensor) -> Tensor:
        return self.project(self.encoder(x)).unsqueeze(-1)

    def extra_repr(self) -> str:
        return (
            f"n_appliances={self.n_appliances}, cycles={self.cycles}, "
            f"cycle_size={self.cycle_size}, in_channels={self.in_channels}"
        )


def faustine(n_appliances: int, *, in_channels: int = 2, out_features: int = 256, **kwargs):
    """:class:`CycleCNN` with the Faustine encoder: four strided blocks, 16 to 128.

    Example:
        >>> model = nf.nn.models.faustine(3, out_features=16, cycles=8, cycle_size=64)
        >>> tuple(model(torch.rand(1, 2, 8, 64)).shape)
        (1, 3, 1)
    """
    return CycleCNN(
        n_appliances,
        in_channels=in_channels,
        encoder=faustine_cnn(in_channels, out_features),
        out_features=out_features,
        **kwargs,
    )


def schirmer(n_appliances: int, *, in_channels: int = 2, out_features: int = 256, **kwargs):
    """:class:`CycleCNN` with the Schirmer encoder: three narrow same-padded blocks.

    Example:
        >>> model = nf.nn.models.schirmer(3, out_features=16, cycles=8, cycle_size=64)
        >>> tuple(model(torch.rand(1, 2, 8, 64)).shape)
        (1, 3, 1)
    """
    return CycleCNN(
        n_appliances,
        in_channels=in_channels,
        encoder=schirmer_cnn(in_channels, out_features),
        out_features=out_features,
        **kwargs,
    )
