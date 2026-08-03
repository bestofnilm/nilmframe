"""The convolutional architectures.

Four papers, one lineage. Kelly & Knottenbelt put a denoising autoencoder on the
problem; Zhang et al. observed that predicting the *midpoint* of the window rather
than all of it removes the averaging that blurs a sequence model's edges; Shin et
al. noticed the regression head was being trained on windows where the appliance
was off and gated it; D'Incecco et al. took the same network and asked what
transfers between appliances and between houses.

They are grouped here because they share a stem -- five convolutions with the same
widths, straight out of the seq2point paper -- and differ in what sits on top. That
is not a simplification for tidiness: the papers really do reuse each other's
feature extractor, and pretending otherwise would put four near-identical stems in
four files and make the actual differences hard to see.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from nilmframe.nn.models.base import NILMModel, Standardiser

__all__ = ["DAE", "SGN", "Seq2Point", "Seq2Seq", "TransferNILM"]

#: (out_channels, kernel) per layer. Zhang et al., Table 1 -- carried by every
#: descendant in this file.
STEM = ((30, 10), (30, 8), (40, 6), (50, 5), (50, 5))


def _stem(in_channels: int, spec=STEM) -> nn.Sequential:
    """The shared feature extractor: same-padded convolutions, ReLU between."""
    layers: list[nn.Module] = []
    channels = in_channels
    for out_channels, kernel in spec:
        layers += [
            # "same" rather than kernel // 2: an even kernel with half-padding
            # returns L + 1, which silently breaks the flattened head's width.
            nn.Conv1d(channels, out_channels, kernel, padding="same"),
            nn.ReLU(inplace=True),
        ]
        channels = out_channels
    return nn.Sequential(*layers)


def _stem_width(spec=STEM) -> int:
    return spec[-1][0]


class Seq2Point(NILMModel):
    """Zhang et al. (AAAI 2018). A window of aggregate to its midpoint.

    The most-benchmarked model in the field, and the reason the point framing
    caught on: a sequence model has to commit to a value for every step of the
    window, including the edges where it has seen almost no context, and averaging
    those overlapping commitments is what smears its switching edges. Predicting
    only the centre means every prediction is made with the full window on both
    sides of it.

    The cost is arithmetic. A series of length ``T`` needs ``T`` forward passes,
    one per position, where a sequence model needs ``T / L``.

    Args:
        n_appliances: size of the label space.
        window: input length. 599 in the paper, and odd on purpose -- an even
            window has no midpoint.
        in_channels: measured quantities per step.
        hidden: width of the dense layer over the flattened features.
        standardiser: input/output scaling.

    Example:
        >>> model = nf.nn.models.Seq2Point(3, window=99)
        >>> tuple(model(torch.rand(2, 1, 99) * 500).shape)
        (2, 3, 1)
        >>> model.kind, model.output_length
        ('seq2point', 1)
    """

    kind = "seq2point"

    def __init__(
        self,
        n_appliances: int,
        *,
        window: int = 599,
        in_channels: int = 1,
        hidden: int = 1024,
        standardiser: Standardiser | None = None,
    ) -> None:
        super().__init__(
            n_appliances, window=window, in_channels=in_channels, standardiser=standardiser
        )
        self.features = _stem(in_channels)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(_stem_width() * window, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, n_appliances),
        )

    def encode(self, x: Tensor) -> Tensor:
        return self.head(self.features(x)).unsqueeze(-1)


class Seq2Seq(NILMModel):
    """The sequence counterpart of :class:`Seq2Point`, same stem.

    Zhang et al. present both; this one reconstructs the whole window, so a series
    of length ``T`` costs ``T / L`` passes instead of ``T``. Use it when you want a
    reconstruction to plot, or when the inference budget rules the point model out.

    Args:
        n_appliances: size of the label space.
        window: input and output length.
        in_channels: measured quantities per step.
        hidden: width of the dense bottleneck.
        standardiser: input/output scaling.

    Example:
        >>> model = nf.nn.models.Seq2Seq(3, window=64)
        >>> tuple(model(torch.rand(2, 1, 64) * 500).shape)
        (2, 3, 64)
    """

    kind = "seq2seq"

    def __init__(
        self,
        n_appliances: int,
        *,
        window: int = 599,
        in_channels: int = 1,
        hidden: int = 1024,
        standardiser: Standardiser | None = None,
    ) -> None:
        super().__init__(
            n_appliances, window=window, in_channels=in_channels, standardiser=standardiser
        )
        self.features = _stem(in_channels)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(_stem_width() * window, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, n_appliances * window),
        )

    def encode(self, x: Tensor) -> Tensor:
        flat = self.head(self.features(x))
        return flat.view(x.shape[0], self.n_appliances, self.window)


class DAE(NILMModel):
    """Kelly & Knottenbelt (BuildSys 2015). The denoising autoencoder.

    The first deep NILM model. The framing is that the aggregate *is* the appliance
    signal with every other appliance as noise, so recovering one appliance is
    denoising -- which is why it is an autoencoder rather than a regressor, and why
    it reconstructs the window rather than a point.

    It is a weak baseline by modern numbers and still worth carrying: it is the
    comparison every later paper reports, and its failure mode (smeared edges,
    plausible-looking averages) is the one the point models were invented to fix.

    Args:
        n_appliances: size of the label space.
        window: input and output length.
        in_channels: measured quantities per step.
        hidden: width of the dense bottleneck.
        standardiser: input/output scaling.

    Example:
        >>> model = nf.nn.models.DAE(2, window=64)
        >>> tuple(model(torch.rand(1, 1, 64) * 500).shape)
        (1, 2, 64)
    """

    kind = "seq2seq"

    def __init__(
        self,
        n_appliances: int,
        *,
        window: int = 599,
        in_channels: int = 1,
        hidden: int = 128,
        standardiser: Standardiser | None = None,
    ) -> None:
        super().__init__(
            n_appliances, window=window, in_channels=in_channels, standardiser=standardiser
        )
        self.encoder_conv = nn.Sequential(
            nn.Conv1d(in_channels, 8, 4, padding="same"), nn.ReLU(inplace=True)
        )
        flat = 8 * window
        self.bottleneck = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, flat),
            nn.ReLU(inplace=True),
        )
        self.decoder_conv = nn.Conv1d(8, n_appliances, 4, padding="same")

    def encode(self, x: Tensor) -> Tensor:
        h = self.encoder_conv(x)
        h = self.bottleneck(h).view(h.shape)
        return self.decoder_conv(h)


class SGN(NILMModel):
    """Shin et al. (AAAI 2019). Subtask gated networks.

    Two copies of the seq2point stem. One regresses power, the other classifies
    on/off, and the regression is multiplied by the gate before it leaves the
    model. The point is not ensembling -- it is that a regression head trained on
    windows where the appliance is off spends most of its capacity learning to
    output zero, and learns the actual power curve from whatever is left.

    The gate is returned alongside the power, because a gate that has collapsed to
    always-on is the failure mode here and is invisible in the power alone.

    Args:
        n_appliances: size of the label space.
        window: input length.
        in_channels: measured quantities per step.
        hidden: width of each branch's dense layer.
        standardiser: input/output scaling.

    Example:
        >>> model = nf.nn.models.SGN(3, window=99)
        >>> power = model(torch.rand(2, 1, 99) * 500)
        >>> tuple(power.shape)
        (2, 3, 1)
        >>> gate = model.gate(torch.rand(2, 1, 99) * 500)
        >>> bool(((gate >= 0) & (gate <= 1)).all())
        True
    """

    kind = "seq2point"

    def __init__(
        self,
        n_appliances: int,
        *,
        window: int = 599,
        in_channels: int = 1,
        hidden: int = 1024,
        standardiser: Standardiser | None = None,
    ) -> None:
        super().__init__(
            n_appliances, window=window, in_channels=in_channels, standardiser=standardiser
        )

        def branch() -> nn.Module:
            return nn.Sequential(
                _stem(in_channels),
                nn.Flatten(),
                nn.Linear(_stem_width() * window, hidden),
                nn.ReLU(inplace=True),
                nn.Linear(hidden, n_appliances),
            )

        self.regression = branch()
        self.classification = branch()

    def encode(self, x: Tensor) -> Tensor:
        power = self.regression(x)
        gate = torch.sigmoid(self.classification(x))
        return (power * gate).unsqueeze(-1)

    @torch.no_grad()
    def gate(self, x: Tensor) -> Tensor:
        """``(B, K, 1)`` on/off probability, for inspecting the classifier alone."""
        if x.ndim == 2:
            x = x.unsqueeze(1)
        if self.standardiser is not None:
            x = self.standardiser.encode(x)
        return torch.sigmoid(self.classification(x)).unsqueeze(-1)


class TransferNILM(Seq2Point):
    """D'Incecco, Squartini & Zhong (IEEE TSG 2020). seq2point, moved.

    Architecturally this *is* :class:`Seq2Point` -- the contribution is the
    training protocol, not the network. It is here as a distinct name because the
    protocol is what you reproduce: train on one appliance or one house, then
    either freeze the convolutional stem and refit the dense head, or fine-tune
    everything at a lower rate.

    :meth:`freeze_features` is the first of those. There is no method for the
    second because it is an optimiser argument, not a model property.

    Args:
        Same as :class:`Seq2Point`.

    Example:
        >>> model = nf.nn.models.TransferNILM(3, window=99)
        >>> model.freeze_features()
        >>> any(p.requires_grad for p in model.features.parameters())
        False
        >>> all(p.requires_grad for p in model.head.parameters())
        True
    """

    def freeze_features(self, frozen: bool = True) -> None:
        """Stop the shared stem training, leaving the head to refit."""
        for parameter in self.features.parameters():
            parameter.requires_grad_(not frozen)
