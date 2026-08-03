"""Fetch exactly one high-frequency recording per appliance class.

The signature database needs one example of each class, not each corpus. Fetching
a corpus whole to keep 1% of it costs gigabytes and hours; the planners already
list every remote file with its size, and for these three corpora the class is
recoverable from the listing alone -- WHITED puts it in the filename, HIFDA in the
directory, PLAID in a metadata sidecar small enough to fetch first.

So the plan is filtered before a byte of signal is downloaded, and what arrives is
one file per class.

Usage::

    python tools/signatures/fetch_one_per_class.py --cache ~/nf-work/cache whited plaid hifda
"""

from __future__ import annotations

import argparse
import json
import logging
import pathlib
import sys

from nilmframe.sources import fetch, human_bytes, plan_for
from nilmframe.sources.base import Plan

logger = logging.getLogger("fetch-one")


def whited_class(relpath: str) -> str | None:
    name = relpath.rsplit("/", 1)[-1]
    return name.split("_")[0].lower() if name.lower().endswith(".flac") else None


def hifda_class(relpath: str) -> str | None:
    parts = relpath.strip("/").split("/")
    # .../<window>/<Current|Voltage>/<Appliance>/<file>.txt
    return parts[-2].lower() if len(parts) >= 2 and relpath.endswith(".txt") else None


def plaid_class_map(cache: pathlib.Path) -> dict[str, str]:
    """``csv/<id>.csv`` to appliance, from whichever metadata files are on disk."""
    mapping: dict[str, str] = {}
    for meta in sorted((cache / "plaid").glob("meta*.json")):
        payload = json.loads(meta.read_text())
        records = payload if isinstance(payload, list) else list(payload.values())
        for rec in records:
            if not isinstance(rec, dict):
                continue
            ident = str(rec.get("id") or rec.get("file_name") or "").split(".")[0]
            # The type is nested: {"id": ..., "meta": {"appliance": {"type": ...}}}
            appliance = ((rec.get("meta") or {}).get("appliance")
                         or rec.get("appliance") or {})
            name = appliance.get("type") if isinstance(appliance, dict) else appliance
            if ident and name:
                mapping[f"csv/{ident}.csv"] = str(name).strip().lower()
    return mapping


def pick(dataset: str, cache: pathlib.Path, per_class: int) -> Plan:
    plan = plan_for(dataset)
    if dataset == "plaid":
        table = plaid_class_map(cache)
        key = lambda a: table.get(a.relpath)  # noqa: E731
    elif dataset == "whited":
        key = lambda a: whited_class(a.relpath)  # noqa: E731
    elif dataset == "hifda":
        key = lambda a: hifda_class(a.relpath)  # noqa: E731
    else:
        raise SystemExit(f"no class rule for {dataset!r}")

    seen: dict[str, int] = {}
    keep = []
    for artifact in plan.artifacts:
        cls = key(artifact)
        if cls is None:
            # Metadata and index files carry no class and are always needed.
            if artifact.relpath.endswith((".json", ".csv")) and "/" not in artifact.relpath:
                keep.append(artifact)
            continue
        # HIFDA stores current and voltage as separate files; both must come.
        slot = f"{cls}|{'V' if 'Voltage' in artifact.relpath else 'I'}"
        if seen.get(slot, 0) >= per_class:
            continue
        seen[slot] = seen.get(slot, 0) + 1
        keep.append(artifact)

    classes = {s.split("|")[0] for s in seen}
    logger.info("%-8s %d classes, %d files, %s", dataset, len(classes), len(keep),
                human_bytes(sum(a.size or a.size_max or 0 for a in keep)))
    return Plan(dataset, tuple(keep))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("datasets", nargs="+")
    parser.add_argument("--cache", type=pathlib.Path, default=pathlib.Path("~/nf-work/cache"))
    parser.add_argument("--per-class", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cache = args.cache.expanduser()

    for dataset in args.datasets:
        plan = pick(dataset, cache, args.per_class)
        if args.dry_run:
            continue
        report = fetch(plan, cache / dataset, workers=4)
        logger.info("   fetched %d, skipped %d", len(report.fetched), len(report.skipped))
    return 0


if __name__ == "__main__":
    sys.exit(main())
