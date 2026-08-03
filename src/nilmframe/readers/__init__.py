"""Dataset readers.

A reader is an iterable of :class:`~nilmframe.store.Recording` objects and nothing
more. It does no windowing, no alignment and no target construction; those belong
to the view and the dataset. Adding a dataset means writing one iterator.

Each of them also knows where its data is published. ``Reader.plan()`` works out
which remote files a configuration needs without fetching them, and
``Reader.download()`` fetches a subset and returns a reader over it -- see
:mod:`nilmframe.sources`. Nothing in this package imports that one at module
level, so constructing a reader never touches the network.
"""

from __future__ import annotations

from nilmframe.readers.blond import BLOND
from nilmframe.readers.fired import FIRED
from nilmframe.readers.hifda import HIFDA
from nilmframe.readers.plaid import PLAID
from nilmframe.readers.refit import REFIT
from nilmframe.readers.smartnialm import SmartNIALM
from nilmframe.readers.uci import UCIHousehold
from nilmframe.readers.ukdale import UKDALE
from nilmframe.readers.whited import WHITED

#: Reader classes by the name the CLI knows them under.
REGISTRY: dict[str, type] = {
    "blond": BLOND,
    "fired": FIRED,
    "hifda": HIFDA,
    "plaid": PLAID,
    "refit": REFIT,
    "smartnialm": SmartNIALM,
    "uci": UCIHousehold,
    "ukdale": UKDALE,
    "whited": WHITED,
}

__all__ = [
    "BLOND",
    "FIRED",
    "HIFDA",
    "PLAID",
    "REFIT",
    "REGISTRY",
    "UKDALE",
    "WHITED",
    "SmartNIALM",
    "UCIHousehold",
]
