"""Collect one high-frequency signature per appliance, from every cached corpus.

Each corpus needs its own picker, because each records a different thing. FIRED
and BLOND meter an appliance for weeks, so a signature is a run cut out of that;
WHITED, PLAID and HIFDA record a few seconds around one switch-on, so the whole
recording is the signature. Pretending those are the same would mean either
padding an activation to look like a run or truncating a run to look like an
activation, and both would put a number on the page that nothing measured.

What every picker must return is the same: a :class:`Recording` with ``v`` and
``i`` at the corpus's own rate, labelled with the appliance and the corpus it came
from. Everything downstream reads the store and does not care.

Usage::

    python tools/signatures/build_store.py ~/nf-work/stores/signatures [--cache ~/nf-work/cache]
"""

from __future__ import annotations

import argparse
import logging
import pathlib
import sys

import numpy as np

from nilmframe.store import ChannelKind, Recording, StoreWriter

logger = logging.getLogger("signatures")

#: Longest signature kept, in seconds. A fridge run and a kettle run differ by two
#: orders of magnitude; past an hour the extra costs disk and shows nothing new.
MAX_SECONDS = 3600.0

#: Shortest worth keeping. Below this there are not enough mains cycles to align.
MIN_SECONDS = 0.15


def _clean(x: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.nan_to_num(np.asarray(x, dtype=np.float32)))


def _emit(writer: StoreWriter, dataset: str, appliance: str, v, i, fs: float,
          f0: float | None = 50.0, session: str = "signature", **meta) -> bool:
    v, i = _clean(v), _clean(i)
    if min(v.size, i.size) < MIN_SECONDS * fs:
        logger.info("%s/%s: too short, skipped", dataset, appliance)
        return False
    n = min(v.size, i.size, int(MAX_SECONDS * fs))
    writer.add(
        Recording(
            dataset=dataset,
            house="signature",
            session=session,
            kind=ChannelKind.SUBMETER,
            appliance=appliance,
            instance_id=f"{dataset}:{appliance}",
            signals={"v": v[:n], "i": i[:n]},
            fs=float(fs),
            f0=f0,
            meta=meta,
        )
    )
    return True


# --------------------------------------------------------------------------- #
# activation corpora: the recording is the signature
# --------------------------------------------------------------------------- #


def from_activations(writer, reader, dataset: str, limit_per_class: int = 1) -> int:
    """One recording per appliance class, the longest one on offer.

    The longest rather than the first: activation corpora vary from under a second
    to several, and a longer example shows more of the load's steady state.
    """
    best: dict[str, Recording] = {}
    for rec in reader:
        if not rec.appliance or "v" not in rec.signals or "i" not in rec.signals:
            continue
        have = best.get(rec.appliance)
        if have is None or rec.signals["i"].size > have.signals["i"].size:
            best[rec.appliance] = rec

    written = 0
    for appliance, rec in sorted(best.items()):
        seconds = rec.signals["i"].size / rec.fs
        if _emit(writer, dataset, appliance, rec.signals["v"], rec.signals["i"],
                 rec.fs, rec.f0, session=rec.session or "activation",
                 seconds=round(seconds, 3), source=rec.meta.get("source")):
            written += 1
    return written


# --------------------------------------------------------------------------- #
# continuous corpora: a signature is a run, found in the low-rate summary
# --------------------------------------------------------------------------- #


def find_run(power: np.ndarray, fs: float) -> tuple[int, int] | None:
    """The longest on-to-off run that does not touch either end of the series.

    The threshold comes from the appliance's own baseline and peak rather than a
    fixed wattage: a phone charger and an oven do not become interesting at the
    same power, and one number cannot serve both.
    """
    p = np.where(np.isfinite(power), power, 0.0)
    base, peak = float(np.percentile(p, 20)), float(np.percentile(p, 99.9))
    if peak - base < 5.0:
        return None
    on = p > base + 0.25 * (peak - base)
    edges = np.diff(on.astype(np.int8))
    starts = list(np.flatnonzero(edges == 1) + 1)
    stops = list(np.flatnonzero(edges == -1) + 1)
    runs = [
        (a, b) for a, b in zip(starts, stops)
        # A run against a file edge may be cut off, and a cut-off run is not a run.
        if a > 0 and b < len(p) - 1 and (b - a) / fs >= 20.0
    ]
    return max(runs, key=lambda r: r[1] - r[0]) if runs else None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dst", type=pathlib.Path)
    parser.add_argument("--cache", type=pathlib.Path, default=pathlib.Path("~/nf-work/cache"))
    parser.add_argument("--only", nargs="*", help="corpora to include; default all cached")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cache = args.cache.expanduser()
    wanted = set(args.only) if args.only else None

    from nilmframe.readers import HIFDA, PLAID, WHITED

    def hifda_windows(root: pathlib.Path):
        """HIFDA publishes several window lengths; take the longest available."""
        dirs = sorted(
            (d for d in root.iterdir() if d.is_dir() and (d / "Current").is_dir()),
            key=lambda d: float(d.name.split("ms")[0]) if "ms" in d.name else 1e9,
        )
        return dirs[-1] if dirs else root

    def plaid_metadata(root: pathlib.Path):
        return [p for p in sorted(root.glob("meta*.json"))]

    plans: list[tuple[str, callable]] = [
        # WHITED puts its recordings under flac/; handed the cache root the reader
        # finds nothing and says so only at DEBUG.
        ("whited", lambda w: from_activations(
            w, WHITED(cache / "whited" / "flac"), "whited")),
        ("plaid", lambda w: from_activations(
            w, PLAID(cache / "plaid" / "csv", plaid_metadata(cache / "plaid")), "plaid")),
        ("hifda", lambda w: from_activations(
            w, HIFDA(hifda_windows(cache / "hifda")), "hifda")),
    ]

    args.dst.expanduser().parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with StoreWriter(args.dst.expanduser(), source="one signature per appliance",
                     overwrite=True) as writer:
        for name, run in plans:
            if wanted and name not in wanted:
                continue
            if not (cache / name).exists():
                logger.info("%s: not cached, skipped", name)
                continue
            try:
                n = run(writer)
            except Exception as exc:  # a corpus that will not read is not fatal
                logger.warning("%s: %s: %s", name, type(exc).__name__, exc)
                continue
            logger.info("%-8s %d signatures", name, n)
            total += n
    logger.info("total %d", total)
    return 0


if __name__ == "__main__":
    sys.exit(main())
