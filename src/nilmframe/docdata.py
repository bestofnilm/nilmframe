"""A tiny built-in corpus, so examples run.

Documentation whose examples cannot be executed rots silently. Everything in this
package's docstrings runs against the store built here, which means ``make
doctest`` and ``pytest --doctest-modules`` both verify that the documentation
still describes the code.

It is also the fastest way to try the library without downloading anything::

    >>> import nilmframe as nf
    >>> store = nf.example_store()
    >>> len(store.submeters())
    6

The signals are synthetic but physically shaped: a 50 Hz mains at 230 V, appliance
currents with characteristic harmonic content and phase lag, and an aggregate
channel that is the exact sum of its submeters plus a standing load. Being exact
is what lets examples assert things -- a doctest that prints an unpredictable
number is not a doctest.
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy as np

__all__ = ["APPLIANCES", "example_measurement", "example_store", "example_store_path"]

#: name -> (rms amperes, harmonic amplitudes relative to the fundamental, phase lag)
APPLIANCES: dict[str, tuple[float, tuple[float, ...], float]] = {
    "kettle": (9.0, (1.0, 0.02, 0.01), 0.02),  # near-resistive
    "fridge": (0.9, (1.0, 0.12, 0.30), 0.55),  # inductive motor
    "laptop": (0.4, (1.0, 0.55, 0.40), 0.35),  # switched-mode supply
    "microwave": (6.0, (1.0, 0.30, 0.18), 0.30),
}

_FS = 6000.0
_F0 = 50.0
_SECONDS = 2.0
_CACHE: dict[str, Path] = {}


def _voltage(n: int) -> np.ndarray:
    t = np.arange(n) / _FS
    return (230.0 * math.sqrt(2) * np.sin(2 * math.pi * _F0 * t)).astype(np.float32)


def _current(appliance: str, n: int, scale: float = 1.0) -> np.ndarray:
    rms, harmonics, lag = APPLIANCES[appliance]
    t = np.arange(n) / _FS
    wave = np.zeros(n)
    for order, amplitude in enumerate(harmonics, start=1):
        wave += amplitude * np.sin(2 * math.pi * _F0 * order * t - lag * order)
    wave /= np.sqrt(np.mean(wave**2))
    return (rms * math.sqrt(2) * scale * wave).astype(np.float32)


def example_store_path(rebuild: bool = False) -> Path:
    """Path to the built-in example store, building it on first use.

    Args:
        rebuild: rebuild even if it already exists.

    Returns:
        The store directory.

    Example:
        >>> from nilmframe.docdata import example_store_path
        >>> example_store_path().name
        'nilmframe-example-store'
        >>> example_store_path().is_dir()
        True
    """
    cached = _CACHE.get("path")
    if cached is not None and cached.exists() and not rebuild:
        return cached

    root = Path(tempfile.gettempdir()) / "nilmframe-example-store"
    if root.exists() and not rebuild:
        _CACHE["path"] = root
        return root

    from nilmframe.store import ChannelKind, Recording, StoreWriter

    if root.exists():
        import shutil

        shutil.rmtree(root)

    n = int(_FS * _SECONDS)
    voltage = _voltage(n)

    with StoreWriter(root, source="nilmframe built-in example", overwrite=True) as writer:
        # Two houses, so leave-house-out has something to hold out; two brands per
        # appliance where possible, so leave-brand-out does too.
        houses = [
            (["kettle", "fridge", "laptop"], "acme"),
            (["kettle", "fridge", "microwave"], "globex"),
        ]
        for house, (names, brand) in enumerate(houses, start=1):
            for appliance in names:
                writer.add(
                    Recording(
                        dataset="example",
                        house=f"house_{house}",
                        session="run_0",
                        kind=ChannelKind.SUBMETER,
                        appliance=appliance,
                        brand=brand,
                        instance_id=f"{appliance}:{brand}",
                        signals={"v": voltage, "i": _current(appliance, n)},
                        fs=_FS,
                        f0=_F0,
                        t0=0.0,
                    ),
                    channel_id=f"house_{house}-{appliance}",
                )

            # The aggregate is the exact sum of this house's submeters plus a
            # small standing load, so conservation examples can assert on it.
            total = sum(_current(a, n) for a in names) + _current("laptop", n, scale=0.25)
            writer.add(
                Recording(
                    dataset="example",
                    house=f"house_{house}",
                    session="run_0",
                    kind=ChannelKind.MAINS,
                    signals={"v": voltage, "i": total.astype(np.float32)},
                    fs=_FS,
                    f0=_F0,
                    t0=0.0,
                ),
                channel_id=f"house_{house}-mains",
            )

    _CACHE["path"] = root
    return root


def example_store(rebuild: bool = False):
    """A small built-in :class:`~nilmframe.store.Store`, for examples and experiments.

    Two houses, three appliances each, at 6 kHz. Nothing is downloaded; the store is
    generated into the system temporary directory on first use and reused after.

    Args:
        rebuild: regenerate even if it already exists.

    Returns:
        :class:`~nilmframe.store.Store`

    Example:
        >>> import nilmframe as nf
        >>> store = nf.example_store()
        >>> store.appliances
        ['fridge', 'kettle', 'laptop', 'microwave']
        >>> store.houses
        ['house_1', 'house_2']
        >>> len(store.mains()), len(store.submeters())
        (2, 6)
    """
    from nilmframe.store import Store

    return Store(example_store_path(rebuild=rebuild))


def example_measurement(appliance: str = "kettle", seconds: float = 0.5):
    """One appliance from the example store, as a :class:`~nilmframe.Measurement`.

    Args:
        appliance: which appliance; one of :data:`APPLIANCES`.
        seconds: window length.

    Returns:
        :class:`~nilmframe.measurement.Measurement`

    Example:
        >>> import nilmframe as nf
        >>> m = nf.example_measurement("kettle")
        >>> m.n_components, round(m.duration, 2)
        (1, 0.5)
        >>> round(float(m.active_power()))
        2926
    """
    store = example_store()
    matches = store.channels.query("appliance == @appliance")
    if matches.empty:
        raise KeyError(f"no {appliance!r} in the example store; have {store.appliances}")
    return store.measurement(matches["channel_id"].iloc[0], seconds=seconds)
