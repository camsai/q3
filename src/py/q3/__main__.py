"""q3/__main__.py — CLI entry point.

Allows running ``python -m q3 input.yml`` as a thin wrapper around the
``q3`` binary, printing ``results.json`` to stdout on success.

Usage::

    python -m q3 input.yml [--binary ./build/q3] [--no-capture]
"""

from __future__ import annotations

import argparse
import json
import sys

from q3.binary import run_q3
from q3.errors import Q3Error


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m q3",
        description="Run a Q3 DFT calculation from a YAML input file.",
    )
    parser.add_argument("input", help="Path to the Q3 YAML input file.")
    parser.add_argument(
        "--binary",
        default=None,
        metavar="PATH",
        help="Path to the q3 binary (default: auto-detect).",
    )
    parser.add_argument(
        "--no-capture",
        action="store_true",
        default=False,
        help="Stream binary stdout/stderr to terminal instead of capturing.",
    )
    return parser


def main() -> int:
    parser = _build_argument_parser()
    args = parser.parse_args()

    try:
        results = run_q3(
            args.input,
            q3_binary=args.binary,
            capture_output=not args.no_capture,
        )
        print(json.dumps(results, indent=4))
        return 0
    except Q3Error as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
