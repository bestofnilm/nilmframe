"""The two published architectures that read aligned mains cycles.

Every other model in this package reads a low-rate power series. These two read
the waveform: once :func:`~nilmframe.nn.cycle_align` has put each mains cycle on a
common grid, a window is cycles by samples-within-a-cycle, and both papers treat
that grid as an image.

They differ in the encoder and in nothing else. Faustine's is four strided blocks
widening 16 to 128; Schirmer's is three narrow same-padded blocks. Both are in
:mod:`nilmframe.nn.encoders` as :func:`~nilmframe.nn.faustine_cnn` and
:func:`~nilmframe.nn.schirmer_cnn`, so they can also be used on their own.

The contract holds, with one declared exception: the input is rank 4 rather than
rank 3, because the aligned window genuinely has a second axis and flattening it
would throw away the reason for measuring at kilohertz. Both are point models --
one aligned window is one observation of a load, so there is one power vector to
predict.
"""

from __future__ import annotations

from torch import Tensor, nn

from nilmframe.nn.encoders import faustine_cnn, schirmer_cnn
from nilmframe.nn.models.base import NILMModel, Standardiser

__all__ = ["Faustine", "Schirmer"]


class _CycleModel(NILMModel):
    """Shared plumbing: an encoder over the cycle grid, projected onto appliances.

    Not a public architecture -- the two subclasses are. This exists so the shape
    checking and the projection are written once rather than twice.
    """

    kind = "seq2point"
    input_rank = 4

    def __init__(
        self,
        n_appliances: int,
        encoder: nn.Module,
        *,
        cycles: int,
        cycle_size: int,
        in_channels: int,
        out_features: int,
        standardiser: Standardiser | None,
    ) -> None:
        # `window` in the base is the last axis; here that is the cycle.
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
        self.encoder = encoder
        self.project = nn.Linear(out_features, n_appliances)

    def encode(self, x: Tensor) -> Tensor:
        return self.project(self.encoder(x)).unsqueeze(-1)

    def extra_repr(self) -> str:
        return (
            f"n_appliances={self.n_appliances}, cycles={self.cycles}, "
            f"cycle_size={self.cycle_size}, in_channels={self.in_channels}"
        )


class Faustine(_CycleModel):
    """Faustine et al. Four strided blocks over the cycle grid, 16 to 128 channels.

    Args:
        n_appliances: size of the label space, ``K``.
        cycles: cycles per window.
        cycle_size: samples per cycle, after alignment.
        in_channels: channels the adapter produces -- 1 for ``"i"``, 2 for
            ``"vi"``, 3 for ``"fryze"``.
        out_features: encoder width.
        standardiser: input/output scaling.

    Example:
        >>> model = nf.nn.models.Faustine(3, cycles=8, cycle_size=64, out_features=16)
        >>> model.kind, model.input_rank
        ('seq2point', 4)
        >>> tuple(model(torch.rand(2, 2, 8, 64)).shape)
        (2, 3, 1)

    Note:
        The adapter stays outside the model, which is what lets one encoder read
        ``vi`` on one corpus and ``fryze`` on another without knowing it has.

        >>> adapter = nf.nn.CycleInput("fryze")
        >>> x = adapter(nf.example_measurement().aligned(cycle_size=64).batch())
        >>> model = nf.nn.models.Faustine(4, cycles=23, cycle_size=64,
        ...                               in_channels=3, out_features=16)
        >>> tuple(model(x).shape)
        (1, 4, 1)
    """

    def __init__(
        self,
        n_appliances: int,
        *,
        cycles: int = 20,
        cycle_size: int = 128,
        in_channels: int = 2,
        out_features: int = 256,
        standardiser: Standardiser | None = None,
    ) -> None:
        super().__init__(
            n_appliances,
            faustine_cnn(in_channels, out_features),
            cycles=cycles,
            cycle_size=cycle_size,
            in_channels=in_channels,
            out_features=out_features,
            standardiser=standardiser,
        )


class Schirmer(_CycleModel):
    """Schirmer et al. Three narrow same-padded blocks over the cycle grid.

    Args:
        n_appliances: size of the label space, ``K``.
        cycles: cycles per window.
        cycle_size: samples per cycle, after alignment.
        in_channels: channels the adapter produces.
        out_features: encoder width.
        standardiser: input/output scaling.

    Example:
        >>> model = nf.nn.models.Schirmer(3, cycles=8, cycle_size=64, out_features=16)
        >>> tuple(model(torch.rand(2, 2, 8, 64)).shape)
        (2, 3, 1)
    """

    def __init__(
        self,
        n_appliances: int,
        *,
        cycles: int = 20,
        cycle_size: int = 128,
        in_channels: int = 2,
        out_features: int = 256,
        standardiser: Standardiser | None = None,
    ) -> None:
        super().__init__(
            n_appliances,
            schirmer_cnn(in_channels, out_features),
            cycles=cycles,
            cycle_size=cycle_size,
            in_channels=in_channels,
            out_features=out_features,
            standardiser=standardiser,
        )
