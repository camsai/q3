#!/usr/bin/env python3
"""
Format floating-point numbers in k-point lists within YAML files to a
consistent fixed-point precision (default: 9 decimal places, matching
C++ printf("%.9f") format).

Only reformats numbers inside `kpoints_list:` blocks. All other values
(cutoffs, mixing parameters, thresholds, etc.) are left untouched.
Scientific notation values (e.g. 1.0e-08) are always preserved as-is.

Usage:
    python scripts/repo/format_yaml_floats.py [--decimals N] [--check] file1.yml [file2.yml ...]

Options:
    --decimals N   Number of decimal places (default: 9)
    --check        Check mode: report files that would change, exit 1 if any

Examples:
    # Format all example YAML files
    python scripts/repo/format_yaml_floats.py examples/**/*.yml

    # Check without modifying
    python scripts/repo/format_yaml_floats.py --check examples/**/*.yml
"""

import argparse
import re
import sys
from pathlib import Path

# Matches bare floating-point numbers (digits.digits, no scientific notation).
FLOAT_PATTERN = re.compile(
    r"""
    (?<=[\s:\-])        # preceded by whitespace, colon, or dash (YAML context)
    (-?                 # optional minus sign
     \d+                # integer part
     \.                 # decimal point
     \d+                # fractional part
    )
    (?![eE\d])          # NOT followed by e/E (scientific) or more digits
    """,
    re.VERBOSE,
)


def format_float(match: re.Match, decimals: int) -> str:
    """Reformat a matched float to exactly the specified number of decimal places."""
    value = float(match.group(1))
    return f"{value:.{decimals}f}"


def format_yaml_floats(content: str, decimals: int = 9) -> str:
    """
    Reformat floating-point numbers in kpoints_list blocks to fixed-point
    notation with exactly `decimals` decimal places.

    Only lines inside a `kpoints_list:` block are modified. A kpoints_list
    block starts at a line matching `kpoints_list:` and continues while
    subsequent lines are indented deeper (list items starting with `- -` or `  -`).
    """
    lines = content.split("\n")
    formatted_lines = []
    in_kpoints_block = False
    kpoints_base_indent = 0

    for line in lines:
        stripped = line.lstrip()
        current_indent = len(line) - len(stripped)

        # Detect start of kpoints_list block
        if stripped.startswith("kpoints_list:"):
            in_kpoints_block = True
            kpoints_base_indent = current_indent
            formatted_lines.append(line)
            continue

        # Detect end of kpoints_list block: a line at same or lesser indent
        # that is not a list continuation
        if in_kpoints_block and stripped and not stripped.startswith("#"):
            if current_indent <= kpoints_base_indent and not stripped.startswith("-"):
                in_kpoints_block = False

        if in_kpoints_block and stripped.startswith("-"):
            # Format floats in this kpoints_list line
            formatted_line = FLOAT_PATTERN.sub(lambda m: format_float(m, decimals), line)
            formatted_lines.append(formatted_line)
        else:
            formatted_lines.append(line)

    return "\n".join(formatted_lines)


def main():
    parser = argparse.ArgumentParser(
        description="Format k-point floating-point numbers in YAML files to consistent precision."
    )
    parser.add_argument(
        "files",
        nargs="+",
        type=Path,
        help="YAML files to format",
    )
    parser.add_argument(
        "--decimals",
        type=int,
        default=9,
        help="Number of decimal places (default: 9)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check mode: report files that would change, exit 1 if any",
    )
    arguments = parser.parse_args()

    files_changed = []

    for file_path in arguments.files:
        if not file_path.exists():
            print(f"  WARNING: {file_path} does not exist, skipping", file=sys.stderr)
            continue

        original = file_path.read_text()
        formatted = format_yaml_floats(original, arguments.decimals)

        if original != formatted:
            files_changed.append(file_path)
            if arguments.check:
                print(f"  NEEDS FORMAT: {file_path}")
            else:
                file_path.write_text(formatted)
                print(f"  FORMATTED: {file_path}")
        else:
            if not arguments.check:
                print(f"  OK: {file_path}")

    if arguments.check:
        if files_changed:
            print(f"\n{len(files_changed)} file(s) need formatting.", file=sys.stderr)
            return 1
        else:
            print("All files OK.")
            return 0

    print(f"\n{len(files_changed)} file(s) formatted.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
