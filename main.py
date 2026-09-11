"""Compatibility entry point for the repository root."""

from __future__ import annotations

from collections.abc import Sequence

from cves_autotriager.cli import main as cli_main


def main(argv: Sequence[str] | None = None) -> int:
    return cli_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
