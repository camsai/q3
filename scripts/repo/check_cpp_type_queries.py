#!/usr/bin/env python3
"""Ratchet the number of dynamic type queries in ``src/`` (AGENTS.md § Calculator Design).

AGENTS.md asks calculators not to branch on the physical approximation they
are handed (``if (quantum_system.has_nonlinear_core_correction())``) but to
decompose into one class per physical context. The tree does not meet that
rule yet: it still asks objects what they are -- through ``dynamic_cast``,
``typeid``, downcasts to the ultrasoft/PAW pseudopotential types, the
``as_ultrasoft()``/``as_paw()`` accessors, and ``has_*()`` capability flags
used as branch conditions. Retiring those is a long refactor; this script
makes sure the count can only go down while it happens.

Every source file under ``src/`` is scanned (comments removed, string
literals blanked except where a category matches on one) and each occurrence
is counted per file and per category:

``dynamic_cast``
    ``dynamic_cast<`` and ``dynamic_pointer_cast<``.
``typeid``
    ``typeid(``.
``pseudopotential_downcast``
    ``static_cast<...>``/``static_pointer_cast<...>`` whose target type names
    an Ultrasoft or PAW type.
``family_accessor``
    ``as_ultrasoft(`` / ``as_paw(`` -- the accessors that hide a downcast.
``family_flag``
    ``has_ultrasoft(``, ``has_augmentation_charges(``,
    ``has_nonlinear_core_correction(``: the pseudopotential-family queries the
    Calculator Design rule names, counted wherever they are called.
``capability_branch``
    any other ``has_*(`` call (standard-library presence checks such as
    ``has_value()`` excepted) on a line that branches (``if``/``while``/
    ``?:``) -- the general shape of flag-based dispatch.
``string_switch``
    a comparison against a string literal naming a physics variant
    (``== "ortho-atomic"`` and friends, listed in ``STRING_SWITCH_LITERALS``).

Declarations of the queries themselves (``bool has_x() const``,
``... as_paw() const``) are not uses and are not counted.

The committed baseline, ``scripts/repo/check_cpp_type_queries_baseline.json``,
holds the per-file, per-category counts measured on the tree. The check
fails when

* any (file, category) count rises above its baseline (a file absent from the
  baseline has a baseline of zero), or
* any count has fallen below its baseline -- the ratchet must be tightened in
  the same change that earned it, otherwise a later change could spend the
  slack. ``--write-baseline`` rewrites the file from the tree.

``--write-baseline`` refuses to raise a count unless ``--allow-increase`` is
also given, so the only way to grow the baseline is a visible, reviewed flag.

Usage::

    python scripts/repo/check_cpp_type_queries.py                  # check
    python scripts/repo/check_cpp_type_queries.py --write-baseline # tighten
    python scripts/repo/check_cpp_type_queries.py --report         # print counts
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
BASELINE_PATH = Path(__file__).resolve().parent / "check_cpp_type_queries_baseline.json"

SOURCE_SUFFIXES = {".hpp", ".cpp", ".h", ".cc", ".cu", ".mm"}

# Auto-generated ESSE wrappers are rewritten by a generator, not by hand.
EXCLUDED_DIRECTORIES = ("generated",)

FAMILY_FLAG_NAMES = ("has_ultrasoft", "has_augmentation_charges", "has_nonlinear_core_correction")

# Standard-library and yaml-cpp members spelled has_*: optional/filesystem
# presence checks, not capability queries on a Q3 object.
STANDARD_HAS_METHODS = frozenset(
    {
        "has_value",
        "has_filename",
        "has_extension",
        "has_stem",
        "has_parent_path",
        "has_root_path",
        "has_root_name",
        "has_root_directory",
        "has_relative_path",
    }
)

STRING_SWITCH_LITERALS = ("ortho-atomic", "atomic", "paw", "us", "nc")

PATTERNS: dict[str, re.Pattern[str]] = {
    "dynamic_cast": re.compile(r"\bdynamic(?:_pointer)?_cast\s*<"),
    "typeid": re.compile(r"\btypeid\s*\("),
    "pseudopotential_downcast": re.compile(
        r"\bstatic(?:_pointer)?_cast\s*<[^;()]*?(?:Ultrasoft|Paw|PAW)[^;()]*?>"
    ),
    "family_accessor": re.compile(r"\bas_(?:ultrasoft|paw)\s*\("),
    "family_flag": re.compile(r"\b(?:" + "|".join(FAMILY_FLAG_NAMES) + r")\s*\("),
}

GENERIC_HAS_CALL_RE = re.compile(r"\bhas_\w+\s*\(")
BRANCH_LINE_RE = re.compile(r"\b(?:if|while)\s*\(|\?")
STRING_SWITCH_RE = re.compile(
    r'[!=]=\s*"(?:' + "|".join(re.escape(s) for s in STRING_SWITCH_LITERALS) + r')"'
    r'|"(?:' + "|".join(re.escape(s) for s in STRING_SWITCH_LITERALS) + r')"\s*[!=]='
)

# A declaration (not a use) of a query: a return type, then the query name,
# then a parameter list, then const/override/noexcept or a body/semicolon.
QUERY_DECLARATION_RE = re.compile(
    r"^\s*(?:virtual\s+|static\s+|inline\s+|constexpr\s+|\[\[nodiscard\]\]\s+)*"
    r"[\w:<>,\s\*&]+?\s+[\*&]*(?:has_\w+|as_ultrasoft|as_paw)\s*\([^)]*\)\s*"
    r"(?:const\s*)?(?:noexcept\s*)?(?:override\s*)?(?:final\s*)?(?:[{;=]|$)"
)

CATEGORIES = (*PATTERNS, "capability_branch", "string_switch")


def strip_comments(text: str) -> str:
    """Remove // and /* */ comments, keeping newlines and string literals intact."""
    result: list[str] = []
    index = 0
    length = len(text)
    while index < length:
        char = text[index]
        if char == '"' or char == "'":
            end = index + 1
            while end < length and text[end] != char and text[end] != "\n":
                end += 2 if text[end] == "\\" else 1
            result.append(text[index : end + 1])
            index = end + 1
        elif text.startswith("//", index):
            end = text.find("\n", index)
            index = length if end == -1 else end
        elif text.startswith("/*", index):
            end = text.find("*/", index + 2)
            block = text[index : length if end == -1 else end + 2]
            result.append("\n" * block.count("\n"))
            index = length if end == -1 else end + 2
        else:
            result.append(char)
            index += 1
    return "".join(result)


STRING_LITERAL_RE = re.compile(r'"(?:\\.|[^"\\\n])*"')


def count_type_queries(text: str) -> Counter[str]:
    """Count type-query occurrences per category in one C++ source text."""
    counts: Counter[str] = Counter()
    for raw_line in strip_comments(text).splitlines():
        if QUERY_DECLARATION_RE.match(raw_line):
            continue
        counts["string_switch"] += len(STRING_SWITCH_RE.findall(raw_line))
        line = STRING_LITERAL_RE.sub('""', raw_line)
        for category, pattern in PATTERNS.items():
            counts[category] += len(pattern.findall(line))
        if BRANCH_LINE_RE.search(line):
            for match in GENERIC_HAS_CALL_RE.finditer(line):
                name = match.group(0).split("(")[0].strip()
                if name not in FAMILY_FLAG_NAMES and name not in STANDARD_HAS_METHODS:
                    counts["capability_branch"] += 1
    return +counts


def iter_source_files(root: Path):
    """Yield every C++/CUDA/ObjC++ source under root, skipping generated code."""
    for path in sorted(root.rglob("*")):
        if path.suffix not in SOURCE_SUFFIXES or not path.is_file():
            continue
        if any(part in EXCLUDED_DIRECTORIES for part in path.relative_to(root).parts):
            continue
        yield path


def measure(root: Path = None) -> dict[str, dict[str, int]]:
    """Return {repo-relative path: {category: count}} for every file with a nonzero count."""
    root = root or SOURCE_ROOT
    measured: dict[str, dict[str, int]] = {}
    for path in iter_source_files(root):
        counts = count_type_queries(path.read_text(encoding="utf-8", errors="replace"))
        if counts:
            key = path.relative_to(root.parent).as_posix()
            measured[key] = dict(sorted(counts.items()))
    return measured


def addon_baseline_paths() -> list[Path]:
    """Every add-on's own baseline: ``src/addons/<name>/<this baseline's file name>``."""
    return sorted((SOURCE_ROOT / "addons").glob(f"*/{BASELINE_PATH.name}"))


def addon_prefix(file: str) -> str | None:
    """Return the ``src/addons/<name>/`` prefix *file* sits under, or None for a core file."""
    parts = file.split("/")
    if len(parts) > 3 and parts[0] == "src" and parts[1] == "addons":
        return "/".join(parts[:3]) + "/"
    return None


def load_baseline(path: Path = None) -> dict[str, dict[str, int]]:
    """Load the committed baseline: core's, plus every add-on's own when no *path* is given.

    A build-time add-on keeps its files' entries in a baseline beside its code
    (``src/addons/<name>/``), so the core baseline names no add-on file and a
    tree without ``src/addons/`` carries no keys for files it does not have. A
    missing file is an empty baseline.
    """
    if path is not None:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    baseline = load_baseline(BASELINE_PATH)
    for addon_baseline_path in addon_baseline_paths():
        baseline.update(load_baseline(addon_baseline_path))
    return baseline


def compare(measured, baseline) -> tuple[list[str], list[str]]:
    """Return (increases, decreases) as human-readable lines."""
    increases: list[str] = []
    decreases: list[str] = []
    for file in sorted(set(measured) | set(baseline)):
        now = measured.get(file, {})
        then = baseline.get(file, {})
        for category in sorted(set(now) | set(then)):
            current, allowed = now.get(category, 0), then.get(category, 0)
            line = f"{file}: {category} {allowed} -> {current}"
            if current > allowed:
                increases.append(line)
            elif current < allowed:
                decreases.append(line)
    return increases, decreases


def write_baseline(measured, path: Path = None) -> None:
    """Write the baseline JSON (sorted, 4-space indent, trailing newline).

    Without *path*, each add-on's entries go to the baseline beside its code
    and the rest to the core baseline (see load_baseline).
    """
    if path is not None:
        path.write_text(json.dumps(measured, indent=4, sort_keys=True) + "\n", encoding="utf-8")
        return
    write_baseline(
        {file: value for file, value in measured.items() if addon_prefix(file) is None},
        BASELINE_PATH,
    )
    addon_directories = sorted(
        directory for directory in (SOURCE_ROOT / "addons").glob("*") if directory.is_dir()
    )
    for addon_directory in addon_directories:
        prefix = f"src/addons/{addon_directory.name}/"
        entries = {file: value for file, value in measured.items() if file.startswith(prefix)}
        target = addon_directory / BASELINE_PATH.name
        if entries or target.exists():
            write_baseline(entries, target)


def total(counts: dict[str, dict[str, int]]) -> int:
    """Sum every count in a {file: {category: count}} mapping."""
    return sum(sum(per_file.values()) for per_file in counts.values())


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point; returns the process exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--write-baseline", action="store_true", help="rewrite the baseline")
    parser.add_argument(
        "--allow-increase",
        action="store_true",
        help="with --write-baseline, permit raising counts (needs review justification)",
    )
    parser.add_argument("--report", action="store_true", help="print per-category totals")
    arguments = parser.parse_args(argv)

    measured = measure()
    baseline = load_baseline()
    increases, decreases = compare(measured, baseline)

    if arguments.report:
        per_category: Counter[str] = Counter()
        for per_file in measured.values():
            per_category.update(per_file)
        for category in CATEGORIES:
            print(f"{category:26s} {per_category.get(category, 0)}")
        print(f"{'total':26s} {total(measured)} (baseline {total(baseline)})")

    if arguments.write_baseline:
        if increases and not arguments.allow_increase:
            print("Refusing to raise the type-query baseline without --allow-increase:")
            for line in increases:
                print(f"  {line}")
            return 1
        write_baseline(measured)
        print(f"Wrote {BASELINE_PATH.name}: {total(measured)} type queries.")
        return 0

    if increases:
        print("New dynamic type queries (AGENTS.md § Calculator Design) -- use polymorphism:")
        for line in increases:
            print(f"  {line}")
    if decreases:
        print(
            "Type queries were removed -- tighten the ratchet with "
            "`python scripts/repo/check_cpp_type_queries.py --write-baseline`:"
        )
        for line in decreases:
            print(f"  {line}")
    return 1 if increases or decreases else 0


if __name__ == "__main__":
    sys.exit(main())
