"""Reference NILM architectures under one call signature.

Eleven models from ten papers. They are here to be compared, not to be improved
on: each is a reimplementation from its paper, with the paper's defaults where the
paper states them, so that a number you get out of one is a number the literature
would recognise.

The whole point is the shared contract. Every model takes ``(B, C, L)`` watts and
returns ``(B, K, L_out)`` watts, so swapping one for another is a constructor call
and nothing else:

    >>> for name in ("seq2point", "unet", "bert4nilm"):
    ...     model = nf.nn.models.build(name, n_appliances=3, window=64)
    ...     print(f"{name:<10} {model.kind:<9} {tuple(model(torch.rand(1, 1, 64)).shape)}")
    seq2point  seq2point (1, 3, 1)
    unet       seq2seq   (1, 3, 64)
    bert4nilm  seq2seq   (1, 3, 64)

What differs between them is declared rather than implied.
:attr:`~nilmframe.nn.models.NILMModel.kind` says whether a model reconstructs the
window or predicts its midpoint, and ``output_length`` follows from it. Nothing
downstream has to know which architecture it is holding.

Two models want more than active power: :class:`WaveNILM` is built for ``(P, Q)``
and takes ``in_channels=2``. Two carry a pretraining objective as a method rather
than a description -- :meth:`BERT4NILM.mask` and :meth:`ELECTRIcity.corrupt`.

The papers are listed in each class. None of this is trained: constructing a model
gives you random weights, and the library ships no checkpoints.
"""

from __future__ import annotations

from nilmframe.nn.models.base import NILMModel, Standardiser
from nilmframe.nn.models.conv import DAE, SGN, Seq2Point, Seq2Seq, TransferNILM
from nilmframe.nn.models.deep import (
    BERT4NILM,
    TPNILM,
    AttentionNILM,
    ELECTRIcity,
    UNetNILM,
    WaveNILM,
)

__all__ = [
    "BERT4NILM",
    "DAE",
    "MODELS",
    "SGN",
    "TPNILM",
    "AttentionNILM",
    "ELECTRIcity",
    "NILMModel",
    "Seq2Point",
    "Seq2Seq",
    "Standardiser",
    "TransferNILM",
    "UNetNILM",
    "WaveNILM",
    "build",
]

#: Short name to class. The names are what a config file or a sweep writes, so
#: they are lowercase and stable; the classes can be renamed, these should not be.
MODELS: dict[str, type[NILMModel]] = {
    "dae": DAE,
    "seq2seq": Seq2Seq,
    "seq2point": Seq2Point,
    "sgn": SGN,
    "transfer": TransferNILM,
    "unet": UNetNILM,
    "wavenilm": WaveNILM,
    "tpnilm": TPNILM,
    "attention": AttentionNILM,
    "bert4nilm": BERT4NILM,
    "electricity": ELECTRIcity,
}


def build(name: str, n_appliances: int, **kwargs) -> NILMModel:
    """Construct a model by name.

    The indirection exists so an experiment can name its architecture in a config
    file instead of importing it, which is what makes a sweep over architectures a
    list of strings rather than a chain of imports.

    Args:
        name: a key of :data:`MODELS`.
        n_appliances: size of the label space.
        **kwargs: passed to the class. Everything is keyword-only there, so a
            misspelled argument is an error rather than a silent default.

    Returns:
        The model, with random weights.

    Raises:
        KeyError: naming the available models, because a typo here is otherwise
            a stack trace three frames deep.

    Example:
        >>> model = nf.nn.models.build("sgn", n_appliances=4, window=99)
        >>> type(model).__name__, model.kind
        ('SGN', 'seq2point')
        >>> nf.nn.models.build("seq2pont", 4)
        Traceback (most recent call last):
            ...
        KeyError: "unknown model 'seq2pont'; available: attention, bert4nilm, dae,
        electricity, seq2point, seq2seq, sgn, tpnilm, transfer, unet, wavenilm"
    """
    try:
        cls = MODELS[name]
    except KeyError:
        raise KeyError(f"unknown model {name!r}; available: {', '.join(sorted(MODELS))}") from None
    return cls(n_appliances, **kwargs)
