"""Command-line interface for cves-autotriager."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from cves_autotriager import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cves-autotriager",
        description="Automated triage tooling for CVEs.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    parser.parse_args(argv)
    return 0
