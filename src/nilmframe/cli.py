"""Command line entry point: ``nilmframe <command>``."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from nilmframe import __version__


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="nilmframe", description=__doc__)
    parser.add_argument("--version", action="version", version=f"nilmframe {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="<command>")

    # Subcommands register themselves so that an optional dependency missing for
    # one command never breaks the others.
    from nilmframe import _commands

    _commands.register_all(sub)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "command", None) is None:
        parser.print_help()
        return 1
    return int(args.func(args) or 0)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
