"""One contract for every reference architecture.

The models in this package come from ten different papers and disagree about
almost everything internally -- convolution against attention, a window against a
single point, one appliance per model against all of them at once. What they must
not disagree about is how you call them, or comparing two of them means rewriting
the code around them, and a comparison that needs rewriting is a comparison nobody
runs twice.

So the contract is fixed and small:

**In:** ``(B, C, L)`` -- a batch of aggregate windows. ``C`` is the number of
measured quantities the model reads, almost always 1 (active power); WaveNILM is
the one that wants more. ``L`` is the model's window, which it declares.

**Out:** ``(B, K, L_out)`` -- watts per appliance. ``L_out`` is ``L`` for a model
that reconstructs the whole window and ``1`` for one that predicts its midpoint.
Which of the two a model is, it declares as :attr:`~NILMModel.kind`; nothing
downstream has to know the architecture to know the shape.

The asymmetry between those two is real and is not smoothed over. A sequence model
returns a reconstruction you can plot against the truth; a point model returns one
number and gets a full series only by sliding. Pretending otherwise -- by padding
the point output to ``L``, say -- would invent values the model never produced and
quietly inflate any metric computed over them.

Normalisation is part of the contract too. Every one of these papers standardises
its input, and the constants differ per dataset; a model that silently assumes the
authors' UK-DALE statistics will look broken on REFIT for a reason that takes a day
to find. So it is an explicit argument, applied by the base class, and reported in
the repr.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor, nn

__all__ = ["NILMModel", "Standardiser"]

Kind = Literal["seq2seq", "seq2point"]


class Standardiser(nn.Module):
    """Centre and scale the input, and undo it on the output.

    Held as buffers rather than constants so the values travel with a checkpoint.
    A model trained on one corpus and evaluated on another with the wrong constants
    is the most common way these reimplementations go quietly wrong.

    Args:
        mean: input mean, in watts.
        std: input standard deviation, in watts.
        power_mean: output mean, in watts. Defaults to ``mean``.
        power_std: output scale, in watts. Defaults to ``std``.

    Example:
        >>> s = nf.nn.models.Standardiser(mean=500.0, std=800.0)
        >>> x = torch.full((1, 1, 4), 500.0)
        >>> float(s.encode(x).abs().max())
        0.0
        >>> float(s.decode(s.encode(x))[0, 0, 0])
        500.0
    """

    def __init__(
        self,
        mean: float = 0.0,
        std: float = 1.0,
        power_mean: float | None = None,
        power_std: float | None = None,
    ) -> None:
        super().__init__()
        if std <= 0:
            raise ValueError(f"std must be positive, got {std}")
        self.register_buffer("mean", torch.tensor(float(mean)))
        self.register_buffer("std", torch.tensor(float(std)))
        self.register_buffer(
            "power_mean", torch.tensor(float(mean if power_mean is None else power_mean))
        )
        self.register_buffer(
            "power_std", torch.tensor(float(std if power_std is None else power_std))
        )

    def encode(self, x: Tensor) -> Tensor:
        return (x - self.mean) / self.std

    def decode(self, y: Tensor) -> Tensor:
        return y * self.power_std + self.power_mean

    def extra_repr(self) -> str:
        return f"mean={float(self.mean):g}, std={float(self.std):g}"


class NILMModel(nn.Module):
    """Base for every architecture here.

    A subclass implements :meth:`encode` -- normalised ``(B, C, L)`` in, normalised
    ``(B, K, L_out)`` out -- and the base handles standardisation, shape checking
    and the batch-dict entry point. That split is why ten papers end up with one
    call signature.

    Args:
        n_appliances: size of the output label space, ``K``.
        window: input length the architecture requires, ``L``.
        in_channels: measured quantities read per step, ``C``.
        standardiser: input/output scaling. ``None`` means the model is fed raw
            watts, which is what you want only if you have already scaled them.

    Attributes:
        kind: ``"seq2seq"`` if the model reconstructs the window, ``"seq2point"``
            if it predicts the midpoint.
        input_rank: 3 for ``(B, C, L)``, the low-rate series every model here reads
            except one. The high-frequency path aligns a window into cycles, which
            is two-dimensional, so :class:`~nilmframe.nn.models.CycleCNN` declares
            4 and is checked against ``(B, C, cycles, cycle_size)`` instead.
    """

    kind: Kind = "seq2seq"
    input_rank: int = 3

    def __init__(
        self,
        n_appliances: int,
        *,
        window: int,
        in_channels: int = 1,
        standardiser: Standardiser | None = None,
    ) -> None:
        super().__init__()
        if n_appliances < 1:
            raise ValueError(f"n_appliances must be >= 1, got {n_appliances}")
        if window < 2:
            raise ValueError(f"window must be >= 2, got {window}")
        self.n_appliances = n_appliances
        self.window = window
        self.in_channels = in_channels
        self.standardiser = standardiser

    @property
    def output_length(self) -> int:
        """``L`` for a sequence model, ``1`` for a point model."""
        return self.window if self.kind == "seq2seq" else 1

    def encode(self, x: Tensor) -> Tensor:  # pragma: no cover - abstract
        """Normalised aggregate in, normalised per-appliance power out."""
        raise NotImplementedError

    def forward(self, x: Tensor) -> Tensor:
        """Watts per appliance from a batch of aggregate windows.

        Args:
            x: ``(B, C, L)``, or ``(B, L)`` when the model reads one quantity.

        Returns:
            ``(B, K, L_out)`` in watts, clamped non-negative -- an appliance
            cannot draw less than nothing, and letting a model say otherwise turns
            a small regression error into a physically impossible one.
        """
        if x.ndim == self.input_rank - 1:
            x = x.unsqueeze(1)
        if x.ndim != self.input_rank:
            shape = "(B, C, L)" if self.input_rank == 3 else "(B, C, cycles, cycle_size)"
            raise ValueError(f"expected {shape}, got {tuple(x.shape)}")
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"{type(self).__name__} reads {self.in_channels} channel(s), got {x.shape[1]}"
            )
        if x.shape[-1] != self.window:
            raise ValueError(
                f"{type(self).__name__} needs a window of {self.window}, got {x.shape[-1]}"
            )

        if self.standardiser is not None:
            x = self.standardiser.encode(x)
        y = self.encode(x)
        if self.standardiser is not None:
            y = self.standardiser.decode(y)
        return y.clamp_min(0.0)

    @torch.no_grad()
    def predict(self, batch: dict | Tensor) -> dict[str, Tensor]:
        """Predict from a dataset batch, reading signals only.

        Args:
            batch: a batch dict, or the aggregate tensor directly. From a dict it
                reads ``p_total`` if present and ``p`` otherwise -- never ``power``
                or ``presence``, which are labels.

        Returns:
            ``power`` ``(B, K, L_out)`` and ``total`` ``(B, L_out)``.
        """
        was_training = self.training
        self.eval()
        try:
            power = self.forward(_aggregate(batch, self.window))
        finally:
            self.train(was_training)
        return {"power": power, "total": power.sum(1)}

    def extra_repr(self) -> str:
        return (
            f"n_appliances={self.n_appliances}, window={self.window}, "
            f"in_channels={self.in_channels}, kind={self.kind!r}"
        )


def _aggregate(batch: dict | Tensor, window: int) -> Tensor:
    """The aggregate series out of a batch, as ``(B, C, L)``."""
    if isinstance(batch, Tensor):
        return batch
    for key in ("p_total", "p"):
        if key in batch:
            x = batch[key]
            break
    else:
        raise KeyError("batch carries no aggregate: expected 'p_total' or 'p'")
    if x.ndim == 1:
        x = x.unsqueeze(0)
    if x.ndim == 2:
        x = x.unsqueeze(1)
    if x.shape[-1] != window:
        raise ValueError(f"batch window is {x.shape[-1]}, model needs {window}")
    return x
