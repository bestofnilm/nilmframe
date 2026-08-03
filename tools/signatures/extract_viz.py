"""Reduce a signature store to what a page can hold.

Two resolutions of the same recording. The *envelope* is one RMS reading per mains
cycle, binned down to something a screen can draw; the *cycles* are the waveform
itself, FitPS-aligned onto a common grid so a period from the first second and one
from the fiftieth minute sit in the same phase and can be drawn on top of each
other.

Everything is quantised to integers -- milliamps and decivolts -- because the page
carries the data inline and a float64 in JSON costs four times what the precision
is worth at screen resolution.

Usage::

    python tools/signatures/extract_viz.py ~/nf-work/stores/signatures-all out.json
"""

from __future__ import annotations

import argparse
import json
import logging
import pathlib
import sys

import numpy as np

from nilmframe.store import Store

logger = logging.getLogger("extract-viz")

ENVELOPE = 700      #: envelope points per signature
CYCLES = 120        #: aligned cycles kept per signature
CYCLE_SIZE = 64     #: samples per aligned cycle


def _cycles(measurement, f0: float):
    """Cycles as ``(n, CYCLE_SIZE)`` for current and voltage, and whether FitPS ran.

    FitPS locates each period from the voltage's rising zero crossing. That needs a
    voltage worth crossing zero, and HIFDA's does not have one -- its channel spans
    13 mV of mostly noise, so alignment finds no periods at all. Rather than drop
    the corpus, fall back to cutting at the nominal period and say so, because a
    fixed cut and a phase-locked one are not the same thing and a page that mixed
    them silently would be lying about the second.
    """
    try:
        a = measurement.aligned(cycle_size=CYCLE_SIZE, f0=f0)
        i = np.asarray(a.i.cpu().numpy(), np.float64)
        v = np.asarray(a.v.cpu().numpy(), np.float64)
        if i.ndim == 2 and i.shape[0] >= 2:
            return i, v, True
    except Exception:
        pass

    raw_i = np.asarray(measurement.i.cpu().numpy(), np.float64).reshape(-1)
    raw_v = np.asarray(measurement.v.cpu().numpy(), np.float64).reshape(-1)
    period = measurement.fs / f0
    n = int(min(raw_i.size, raw_v.size) // period)
    if n < 2:
        return None, None, False
    grid = np.linspace(0, period, CYCLE_SIZE, endpoint=False)
    cut = lambda x: np.stack([  # noqa: E731
        np.interp(k * period + grid, np.arange(x.size), x) for k in range(n)
    ])
    return cut(raw_i), cut(raw_v), False


def summarise(store: Store, channel_id: str) -> dict | None:
    row = store.channel(channel_id)
    measurement = store.measurement(channel_id)
    f0 = float(row.f0) if np.isfinite(row.f0) else 50.0
    if not 45.0 <= f0 <= 65.0:
        # A mains frequency outside this band is not a mains frequency. HIFDA
        # reaches here: its voltage channel spans 13 mV, so the fundamental sits
        # at 15% of a spectrum led by noise and the estimator returns tens of
        # kilohertz. The current is clean, and the grid it was recorded on is 50 Hz.
        logger.info("%s: f0 was %.0f Hz, using 50", channel_id, f0)
        f0 = 50.0
    i, v, aligned_ok = _cycles(measurement, f0)
    if i is None:
        logger.info("%s: no usable cycles", channel_id)
        return None

    n = i.shape[0]
    rms = np.sqrt((i**2).mean(axis=1))
    watts = (v * i).mean(axis=1)

    edges = np.linspace(0, n, min(ENVELOPE, n) + 1).astype(int)
    take = lambda a: [  # noqa: E731
        round(float(a[x:y].mean()), 4) if y > x else 0.0
        for x, y in zip(edges[:-1], edges[1:])
    ]

    picks = np.unique(np.linspace(0, n - 1, min(CYCLES, n)).astype(int))
    return {
        "id": channel_id,
        "fitps": aligned_ok,
        "dataset": row.dataset,
        "name": row.appliance,
        "fs": float(row.fs),
        "f0": f0,
        "n_cycles": int(n),
        "seconds": float(measurement.duration),
        "irms": round(float(np.sqrt((i**2).mean())), 4),
        "watts": round(float(watts.mean()), 1),
        "peak_w": round(float(watts.max()), 1),
        "env_rms": take(rms),
        "env_w": take(watts),
        "cycle_at": [int(k) for k in picks],
        "cycles_i": [[int(round(x * 1000)) for x in i[k]] for k in picks],
        "cycles_v": [[int(round(x * 10)) for x in v[k]] for k in picks],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("store", type=pathlib.Path)
    parser.add_argument("out", type=pathlib.Path)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    store = Store(args.store.expanduser())
    rows = []
    for channel_id in store.channels.channel_id:
        row = summarise(store, channel_id)
        if row is not None:
            rows.append(row)
    # Heaviest current first: roughly most to least interesting to look at.
    rows.sort(key=lambda r: (r["dataset"], -r["irms"]))

    payload = {"cycle_size": CYCLE_SIZE, "appliances": rows}
    args.out.expanduser().write_text(json.dumps(payload, separators=(",", ":")))
    logger.info("%d signatures, %.1f MB", len(rows),
                args.out.expanduser().stat().st_size / 1e6)
    return 0


if __name__ == "__main__":
    sys.exit(main())
