"""Getting the data.

A reader reads a directory. This package fills that directory, and the split is
deliberate: nothing here is imported by a reader at module level, so constructing
a reader never touches the network and a corpus you already have on disk needs
none of this.

The work happens in two steps, and only the second one costs anything.

**Planning** answers *which remote bytes does this configuration need?* using
directory listings and archive footers -- kilobytes, not gigabytes. The result is
a :class:`Plan` you can print. On a corpus measured in terabytes, being able to
see the bill before agreeing to it is the feature.

**Fetching** takes a plan and a cache directory and materialises it, verifying
checksums where the host publishes them, resuming interrupted transfers, and
skipping what is already there. The cache is laid out exactly as each reader's
documented on-disk layout, so it is an ordinary directory that an ordinary reader
accepts::

    from nilmframe.readers import UKDALE

    reader = UKDALE.download("~/.cache/nilmframe/ukdale",
                             houses=[1], channels=[1, 5],
                             time_range=(1421784000, 1421870400),
                             max_hf_files=2)

Every dataset here turns out to support partial fetches, though none of them
advertise it. UK-DALE publishes its 16 kHz mains an hour per file with the start
time in the filename. The other three archives -- UK-DALE's meter readings,
PLAID's waveforms, WHITED's recordings -- are single multi-gigabyte zips served
by hosts that honour HTTP range requests, and a zip keeps its directory at the
end, so any member can be read without the archive. See :mod:`nilmframe.sources._zip`.
"""

from __future__ import annotations

from typing import Any

from nilmframe.sources._http import FetchError
from nilmframe.sources._sevenzip import RemoteSevenZip
from nilmframe.sources._zip import RemoteZip, ZipEntry
from nilmframe.sources.base import Artifact, Plan, human_bytes
from nilmframe.sources.blond import BLONDSource
from nilmframe.sources.fetch import FetchReport, fetch, materialize
from nilmframe.sources.fired import FIREDSource
from nilmframe.sources.hifda import HIFDASource
from nilmframe.sources.plaid import PLAIDSource
from nilmframe.sources.refit import REFITSource
from nilmframe.sources.smartnialm import SmartNIALMSource
from nilmframe.sources.uci import UCISource
from nilmframe.sources.ukdale import UKDALESource
from nilmframe.sources.whited import WHITEDSource

__all__ = [
    "SOURCES",
    "Artifact",
    "BLONDSource",
    "FIREDSource",
    "FetchError",
    "FetchReport",
    "HIFDASource",
    "PLAIDSource",
    "Plan",
    "REFITSource",
    "RemoteSevenZip",
    "RemoteZip",
    "SmartNIALMSource",
    "UCISource",
    "UKDALESource",
    "WHITEDSource",
    "ZipEntry",
    "fetch",
    "human_bytes",
    "materialize",
    "plan_for",
]

#: Source classes by the name the CLI and the readers know them under. Mirrors
#: :data:`nilmframe.readers.REGISTRY`, one source per reader.
SOURCES: dict[str, type] = {
    "blond": BLONDSource,
    "fired": FIREDSource,
    "hifda": HIFDASource,
    "plaid": PLAIDSource,
    "refit": REFITSource,
    "smartnialm": SmartNIALMSource,
    "uci": UCISource,
    "ukdale": UKDALESource,
    "whited": WHITEDSource,
}


def plan_for(dataset: str, **kwargs: Any) -> Plan:
    """Plan a download for a named dataset.

    Named ``plan_for`` rather than ``plan`` so that it and :class:`Plan` do not
    collide as filenames when the documentation generates a page per object.

    Args:
        dataset: one of :data:`SOURCES`.
        **kwargs: forwarded to that source's ``plan()``.

    Returns:
        The :class:`Plan`, which you can print before spending anything.

    Example:
        >>> from nilmframe.sources import SOURCES
        >>> sorted(SOURCES)
        ['blond', 'fired', 'hifda', 'plaid', 'refit', 'smartnialm', 'uci', 'ukdale', 'whited']
    """
    if dataset not in SOURCES:
        raise KeyError(f"unknown dataset {dataset!r}; expected one of {sorted(SOURCES)}")
    return SOURCES[dataset]().plan(**kwargs)
