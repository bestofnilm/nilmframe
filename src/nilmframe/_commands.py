"""Subcommand registration for the CLI.

Each ``register_*`` adds one subparser and imports its implementation lazily, so a
missing optional dependency degrades to one broken command rather than an unusable
CLI.
"""

from __future__ import annotations

import argparse
import sys

_REGISTRARS: list = []


def register(fn):
    _REGISTRARS.append(fn)
    return fn


def register_all(sub: argparse._SubParsersAction) -> None:
    for fn in _REGISTRARS:
        fn(sub)


# --------------------------------------------------------------------------- #
# fetch
# --------------------------------------------------------------------------- #

#: Above this, a fetch asks before spending the bandwidth. One UK-DALE waveform
#: file is 200 MB, so a plan past a gigabyte is usually either deliberate or a
#: mistyped time range, and the two are worth telling apart.
CONFIRM_ABOVE = 1 << 30


@register
def _register_fetch(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "fetch",
        help="download part of a public dataset into a cache directory",
        description=(
            "Work out which files a subset needs, then fetch them. Planning is "
            "separate from fetching and costs a few directory listings, so "
            "--dry-run prints the bill before anything is spent. The cache is an "
            "ordinary directory in the layout each reader expects."
        ),
    )
    p.add_argument("dataset", choices=sorted(_source_names()), help="source dataset")
    p.add_argument("--cache", required=True, help="directory to fetch into")
    p.add_argument("--dry-run", action="store_true", help="print the plan and stop")
    p.add_argument("-y", "--yes", action="store_true", help="do not ask before a large fetch")
    p.add_argument("--workers", type=int, default=4, help="concurrent downloads")
    p.add_argument("--force", action="store_true", help="re-fetch what is already cached")
    p.add_argument("--max-bytes", help="refuse a plan larger than this, e.g. 20G")
    p.add_argument("--limit", type=int, help="take at most N recordings (PLAID, WHITED)")

    ukdale = p.add_argument_group("UK-DALE")
    ukdale.add_argument("--houses", type=int, nargs="*", default=[1], help="house numbers")
    ukdale.add_argument("--channels", type=int, nargs="*", help="meter channels; default all")
    ukdale.add_argument("--from", dest="start", help="start time: unix seconds or ISO date (UTC)")
    ukdale.add_argument("--to", dest="stop", help="end time: unix seconds or ISO date (UTC)")
    ukdale.add_argument("--max-hf-files", type=int, default=1, help="waveform files per house")
    ukdale.add_argument("--all-hf-files", action="store_true", help="every waveform in the range")
    ukdale.add_argument("--no-high-freq", action="store_true", help="skip the 16 kHz waveforms")
    ukdale.add_argument("--no-low-freq", action="store_true", help="skip the meter channels")

    blond = p.add_argument_group("BLOND")
    # Shared between BLOND (BLOND-50/250) and FIRED (1Hz/50Hz/highFreq); each
    # planner validates its own values, so the choices are not fixed here.
    p.add_argument("--resolution", help="which release/tier (BLOND, FIRED)")
    blond.add_argument(
        "--units", nargs="*", default=["clear"], help="clear and/or medal-1 .. medal-15"
    )
    blond.add_argument("--days", nargs="*", help="YYYY-MM-DD day directories")
    blond.add_argument("--max-files", type=int, default=1, help="files per unit per day")

    fired = p.add_argument_group("FIRED")
    fired.add_argument("--meters", nargs="*", help="smartmeter001, powermeterNN")

    hifda = p.add_argument_group("HIFDA")
    hifda.add_argument(
        "--window",
        default="Full_time",
        choices=["10.24ms", "163.84ms", "1310.72ms", "Full_time"],
        help="which windowing of the release",
    )

    snm = p.add_argument_group("SmartNIALMeter")
    snm.add_argument(
        "--version", default="raw", choices=["raw", "preprocessed"], help="which curation"
    )
    snm.add_argument("--buildings", type=int, nargs="*", help="building numbers; default 1")
    snm.add_argument("--no-aggregate", action="store_true", help="skip the smart meter")

    plaid = p.add_argument_group("PLAID")
    plaid.add_argument("--ids", nargs="*", help="recording ids; default all")

    whited = p.add_argument_group("WHITED")
    whited.add_argument("--appliances", nargs="*", help="appliance names, e.g. Kettle")
    whited.add_argument("--kits", nargs="*", help="measurement kits; default the calibrated ones")
    whited.add_argument("--regions", nargs="*", help="region codes, e.g. r1")

    p.set_defaults(func=_fetch)


def _source_names() -> list[str]:
    from nilmframe.sources import SOURCES

    return list(SOURCES)


def _parse_time(text: str | None) -> float | None:
    """Unix seconds from either unix seconds or an ISO date, read as UTC."""
    if text is None:
        return None
    try:
        return float(text)
    except ValueError:
        pass
    from datetime import datetime, timezone

    moment = datetime.fromisoformat(text)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.timestamp()


def _parse_bytes(text: str | None) -> int | None:
    """``20G`` to bytes. Plain digits are bytes."""
    if text is None:
        return None
    scale = {"k": 10**3, "m": 10**6, "g": 10**9, "t": 10**12}
    suffix = text[-1].lower()
    if suffix in scale:
        return int(float(text[:-1]) * scale[suffix])
    return int(text)


def _plan_kwargs(args: argparse.Namespace) -> dict:
    """The planner arguments this dataset understands, from a shared parser."""
    if args.dataset == "ukdale":
        window = (_parse_time(args.start), _parse_time(args.stop))
        if (window[0] is None) != (window[1] is None):
            raise ValueError("--from and --to have to be given together")
        return {
            "houses": args.houses,
            "channels": args.channels,
            "time_range": None if window[0] is None else window,
            "low_freq": not args.no_low_freq,
            "high_freq": not args.no_high_freq,
            "max_hf_files": None if args.all_hf_files else args.max_hf_files,
        }
    if args.dataset == "blond":
        window = (_parse_time(args.start), _parse_time(args.stop))
        if (window[0] is None) != (window[1] is None):
            raise ValueError("--from and --to have to be given together")
        return {
            "resolution": args.resolution or "BLOND-50",
            "units": args.units,
            "days": args.days,
            "time_range": None if window[0] is None else window,
            "max_files": args.max_files,
        }
    if args.dataset == "fired":
        return {
            "resolution": args.resolution or "1Hz",
            "meters": args.meters,
            "days": args.days,
            "max_files": args.max_files,
        }
    if args.dataset == "refit":
        return {"houses": args.houses if args.houses != [1] or "--houses" in sys.argv else None}
    if args.dataset == "uci":
        return {}
    if args.dataset == "hifda":
        return {"window": args.window, "appliances": args.appliances, "limit": args.limit}
    if args.dataset == "smartnialm":
        return {
            "version": args.version,
            "buildings": args.buildings,
            "appliances": args.appliances,
            "aggregate": not args.no_aggregate,
        }
    if args.dataset == "plaid":
        return {"ids": args.ids, "limit": args.limit}
    return {
        "appliances": args.appliances,
        "kits": args.kits,
        "regions": args.regions,
        "limit": args.limit,
    }


def _fetch(args: argparse.Namespace) -> int:
    from pathlib import Path

    from nilmframe.sources import SOURCES, FetchError, fetch, human_bytes

    cache = Path(args.cache).expanduser()
    source = SOURCES[args.dataset]
    kwargs = {}
    if args.dataset in ("ukdale", "blond", "hifda", "fired"):
        kwargs["index_cache"] = cache / ".index"

    try:
        plan = source(**kwargs).plan(**_plan_kwargs(args))
    except (FetchError, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 2

    print(plan.summary())
    if not plan:
        print("nothing to fetch", file=sys.stderr)
        return 1

    if args.dry_run:
        report = fetch(plan, cache, dry_run=True, force=args.force)
        print(
            f"\ndry run: would fetch {len(report.fetched)} files "
            f"({human_bytes(report.nbytes)}); {len(report.skipped)} already cached"
        )
        return 0

    if plan.nbytes_max > CONFIRM_ABOVE and not args.yes:
        if not sys.stdin.isatty():
            print(
                f"\nthis plan is up to {human_bytes(plan.nbytes_max)}; pass --yes to go ahead",
                file=sys.stderr,
            )
            return 1
        answer = input(f"\nfetch up to {human_bytes(plan.nbytes_max)} into {cache}? [y/N] ")
        if answer.strip().lower() not in ("y", "yes"):
            return 1

    try:
        report = fetch(
            plan,
            cache,
            workers=args.workers,
            force=args.force,
            max_bytes=_parse_bytes(args.max_bytes),
            progress=print,
        )
    except FetchError as exc:
        print(exc, file=sys.stderr)
        return 1

    print(f"\n{report}")
    print(f"\nnext: {_convert_hint(args.dataset, plan, report.root)}")
    return 0


def _convert_hint(dataset: str, plan, root) -> str:
    """The ``convert`` command that turns this cache into a store.

    The fetch knows where every piece landed; making the user reconstruct that
    from the directory listing would be a small, avoidable cruelty.
    """
    flags = {
        "dirpath": "--src",
        "metadata": "--metadata",
        "high_freq_root": "--high-freq-root",
        "appliance_log": "--appliance-log",
    }
    parts = [f"nilmframe convert {dataset}"]
    for key, value in plan.reader_kwargs.items():
        if key not in flags:
            continue
        paths = value if isinstance(value, list) else [value]
        parts.append(f"{flags[key]} " + " ".join(str(root / path) for path in paths))
    parts.append("--dst <store>")
    return " ".join(parts)


# --------------------------------------------------------------------------- #
# convert
# --------------------------------------------------------------------------- #


@register
def _register_convert(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "convert",
        help="convert a source dataset into a canonical store",
        description=(
            "Read a source dataset and write a canonical store: parquet metadata "
            "plus memory-mapped signal arrays. Run once per dataset."
        ),
    )
    p.add_argument("dataset", choices=sorted(_reader_names()), help="source dataset")
    p.add_argument("--src", required=True, help="source directory")
    p.add_argument("--dst", required=True, help="destination store directory")
    p.add_argument(
        "--metadata",
        nargs="+",
        help="metadata JSON (PLAID). The 2017 release splits its annotations "
        "across two files that partition the corpus; pass both.",
    )
    p.add_argument("--fs", type=float, help="override the sampling rate")
    p.add_argument("--rate-hz", type=float, help="uniform low-frequency rate (UK-DALE)")
    p.add_argument("--houses", type=int, nargs="*", help="house numbers to read (UK-DALE)")
    p.add_argument("--max-seconds", type=float, help="truncate each channel (UK-DALE)")
    p.add_argument(
        "--high-freq-root",
        help="waveform tree, when it is kept apart from the meter readings (UK-DALE)",
    )
    p.add_argument("--max-hf-files", type=int, help="waveform files per house (UK-DALE)")
    p.add_argument("--no-high-freq", action="store_true", help="skip the waveforms (UK-DALE)")
    p.add_argument("--appliance-log", help="appliance_log.json (BLOND)")
    p.add_argument("--appliances", nargs="*", help="appliance names (HIFDA, SmartNIALMeter)")
    p.add_argument("--buildings", type=int, nargs="*", help="building numbers (SmartNIALMeter)")
    p.add_argument("--meters", nargs="*", help="meter names (FIRED)")
    p.add_argument("--resolution", help="release/tier (BLOND, FIRED)")
    p.add_argument("--units", nargs="*", help="clear, medal-N (BLOND)")
    p.add_argument("--days", nargs="*", help="YYYY-MM-DD directories (BLOND)")
    p.add_argument("--max-files", type=int, help="files per unit per day (BLOND)")
    p.add_argument("--limit", type=int, help="stop after N recordings (for smoke tests)")
    p.add_argument("--overwrite", action="store_true", help="replace an existing store")
    p.add_argument("--verify", action="store_true", help="rehash every signal after writing")
    p.set_defaults(func=_convert)


def _reader_names() -> list[str]:
    from nilmframe.readers import REGISTRY

    return list(REGISTRY)


def _convert(args: argparse.Namespace) -> int:
    from itertools import islice

    from nilmframe.readers import REGISTRY
    from nilmframe.store import Store, StoreWriter

    cls = REGISTRY[args.dataset]
    kwargs: dict = {}
    if args.dataset == "plaid":
        if not args.metadata:
            print("plaid needs --metadata pointing at its metadata JSON", file=sys.stderr)
            return 2
        # One path stays one path: the reader takes either, and a bare string
        # keeps the single-file case reading the way it always did.
        kwargs["metadata"] = args.metadata if len(args.metadata) > 1 else args.metadata[0]
        if args.fs is not None:
            kwargs["fs"] = args.fs
    if args.dataset == "ukdale":
        if args.rate_hz is not None:
            kwargs["rate_hz"] = args.rate_hz
        if args.houses:
            kwargs["houses"] = args.houses
        if args.max_seconds is not None:
            kwargs["max_seconds"] = args.max_seconds
        if args.high_freq_root:
            kwargs["high_freq_root"] = args.high_freq_root
        if args.max_hf_files is not None:
            kwargs["max_hf_files"] = args.max_hf_files
        kwargs["high_freq"] = not args.no_high_freq
    if args.dataset == "fired":
        if args.resolution:
            kwargs["resolution"] = args.resolution
        for name in ("meters", "days", "max_files", "max_seconds"):
            value = getattr(args, name, None)
            if value:
                kwargs[name] = value
    if args.dataset == "refit":
        for name in ("houses", "appliances", "max_seconds"):
            value = getattr(args, name, None)
            if value:
                kwargs[name] = value
    if args.dataset == "uci" and args.max_seconds is not None:
        kwargs["max_seconds"] = args.max_seconds
    if args.dataset == "smartnialm":
        for name in ("buildings", "appliances", "max_seconds"):
            value = getattr(args, name, None)
            if value:
                kwargs[name] = value
    if args.dataset == "hifda":
        for name in ("appliances", "max_seconds"):
            value = getattr(args, name)
            if value:
                kwargs[name] = value
        if args.fs is not None:
            kwargs["fs"] = args.fs
    if args.dataset == "blond":
        if not args.appliance_log:
            print(
                "blond needs --appliance-log pointing at appliance_log.json; "
                "without it the MEDAL sockets have no appliance names",
                file=sys.stderr,
            )
            return 2
        kwargs["appliance_log"] = args.appliance_log
        for name in ("units", "days", "max_files", "max_seconds"):
            value = getattr(args, name)
            if value:
                kwargs[name] = value

    reader = cls(args.src, **kwargs)
    records = iter(reader)
    if args.limit:
        records = islice(records, args.limit)

    with StoreWriter(args.dst, source=f"{args.dataset}:{args.src}", overwrite=args.overwrite) as w:
        n = len(w.extend(records))

    store = Store(args.dst)
    print(f"wrote {n} channels to {args.dst}")
    print(f"  appliances    {store.n_appliances}")
    print(f"  total samples {store.manifest['total_samples']:,}")
    print(f"  content hash  {store.manifest['content_sha256'][:16]}")

    problems = store.verify(deep=args.verify)
    if problems:
        print(f"\n{len(problems)} integrity problem(s):", file=sys.stderr)
        for problem in problems[:20]:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    return 0


# --------------------------------------------------------------------------- #
# describe
# --------------------------------------------------------------------------- #


@register
def _register_describe(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("describe", help="summarise a store")
    p.add_argument("store", help="store directory")
    p.add_argument("--verify", action="store_true", help="rehash every signal")
    p.set_defaults(func=_describe)


def _describe(args: argparse.Namespace) -> int:
    import pandas as pd

    from nilmframe.store import Store

    store = Store(args.store)
    print(store)
    with pd.option_context("display.max_rows", 200, "display.width", 120):
        print(f"\n{store.describe().to_string(index=False)}")

    problems = store.verify(deep=args.verify)
    if problems:
        print(f"\n{len(problems)} problem(s):", file=sys.stderr)
        for problem in problems[:20]:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    print("\nintegrity: ok")
    return 0


# --------------------------------------------------------------------------- #
# sweep
# --------------------------------------------------------------------------- #


@register
def _register_compat(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "compat",
        help="report what varies across stores, and what it breaks",
        description=(
            "Enumerate the axes along which a set of stores disagrees -- sampling "
            "rate, mains frequency, supply voltage, quantities, vocabulary -- and "
            "say which of them actually block each view."
        ),
    )
    p.add_argument("stores", nargs="+", help="store directories")
    p.add_argument("--shallow", action="store_true", help="metadata only; do not sample voltages")
    p.set_defaults(func=_compat)


def _compat(args: argparse.Namespace) -> int:
    import pandas as pd

    from nilmframe.compat import compatibility
    from nilmframe.store import Store

    report = compatibility(*[Store(p) for p in args.stores], deep=not args.shallow)
    print(report.summary())
    with pd.option_context("display.width", 200, "display.max_colwidth", 60):
        print(f"\n{report.to_frame().to_string(index=False)}")
    return 0


@register
def _register_merge(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "merge",
        help="combine stores under explicit rules",
        description=(
            "Merge several stores into one. Rules are arguments rather than "
            "assumptions: --require refuses the merge when an axis disagrees, "
            "--rename harmonises the label space, --normalize-voltage brings every "
            "channel to one supply level without changing any load's power."
        ),
    )
    p.add_argument("stores", nargs="+", help="source store directories")
    p.add_argument("--dst", required=True, help="destination store directory")
    p.add_argument(
        "--require",
        nargs="*",
        default=[],
        metavar="AXIS",
        help="axes that must agree: fs, f0, quantities, dataset, voltage",
    )
    p.add_argument(
        "--rename",
        nargs="*",
        default=[],
        metavar="FROM=TO",
        help="appliance aliases, e.g. --rename refrigerator=fridge",
    )
    p.add_argument(
        "--normalize-voltage",
        type=float,
        metavar="VRMS",
        help="rescale every channel to this supply level",
    )
    p.add_argument(
        "--no-prefix", action="store_true", help="do not prefix channel ids with their dataset"
    )
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(func=_merge)


def _merge(args: argparse.Namespace) -> int:
    from nilmframe.store import merge_stores

    aliases = {}
    for pair in args.rename:
        if "=" not in pair:
            print(f"--rename wants FROM=TO, got {pair!r}", file=sys.stderr)
            return 2
        old, new = pair.split("=", 1)
        aliases[old] = new

    try:
        merged = merge_stores(
            args.stores,
            args.dst,
            require=args.require,
            rename=aliases,
            normalize_voltage=args.normalize_voltage,
            prefix_with_dataset=not args.no_prefix,
            overwrite=args.overwrite,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(f"merged {len(args.stores)} stores into {args.dst}")
    print(f"  channels   {len(merged)}")
    print(f"  datasets   {merged.datasets}")
    print(f"  appliances {merged.n_appliances}")
    print(f"  hash       {merged.manifest['content_sha256'][:16]}")
    problems = merged.verify()
    print(f"  integrity  {problems or 'ok'}")
    return 0 if not problems else 1
