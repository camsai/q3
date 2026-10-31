#!/usr/bin/env python3
"""Enforce the ``q3`` Python package import layering (AGENTS.md § Python Package Layering).

The C++ side has a hard downward-only rule: ``layer_N`` must never include
``layer_(N+k)``.  The Python package needs the same discipline for the same
reason — without it, a lower-level module reaches up into a higher-level one,
the two form an import cycle, and the cycle gets "fixed" by burying the import
inside a function.  That hides the design defect instead of resolving it and
is what this checker exists to prevent.

Tiers (a module may import from its own tier or any tier below it):

    3  q3.workflows, q3.jax        orchestration; may use everything below
    2  q3.calculators, q3.flavors  one run of the engine, driven by an API
    1  q3.binary, q3.io, q3.structure, q3.properties
                                   process/format/structure adapters and
                                   fits of properties from computed values
    0  q3.errors                   leaf: imports nothing from q3

Two violations are reported:

  * upward import   — a tier-N module importing a tier-(N+k) module.
  * same-tier cycle — two modules in the same tier importing each other
                      (an upward import in disguise, since one of them is
                      really the lower of the pair).

Usage::

    python scripts/repo/check_python_import_layering.py [PATH ...]

With no arguments the whole ``src/py/q3`` tree is checked.  Exits non-zero and
prints one line per violation when the layering is broken.
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "src" / "py" / "q3"

# Subpackage/module name (first component under ``q3.``) → tier.
TIERS: dict[str, int] = {
    "constants": 0,
    "errors": 0,
    "binary": 1,
    "io": 1,
    "generated": 1,
    "structure": 1,
    "properties": 1,
    "calculators": 2,
    "flavors": 2,
    "workflows": 3,
    "jax": 3,
}

TIER_DESCRIPTIONS: dict[int, str] = {
    0: "leaf (imports nothing from q3)",
    1: "process/format/structure adapters, property fits",
    2: "single-run drivers",
    3: "orchestration",
}


class Violation:
    """One layering violation, ready to be printed as a compiler-style diagnostic."""

    def __init__(self, path: Path, line: int, message: str) -> None:
        self.path = path
        self.line = line
        self.message = message

    def __str__(self) -> str:
        return f"{self.path}:{self.line}: {self.message}"


def module_name_for(path: Path) -> str | None:
    """Return the dotted ``q3.*`` module name for *path*, or None if outside the package."""
    try:
        relative = path.resolve().relative_to(PACKAGE_ROOT)
    except ValueError:
        return None
    parts = list(relative.parts)
    if parts[-1] == "__init__.py":
        parts.pop()
    elif parts[-1].endswith(".py"):
        parts[-1] = parts[-1][: -len(".py")]
    else:
        return None
    return ".".join(["q3", *parts])


def tier_of(module_name: str) -> int | None:
    """Return the tier of a ``q3.*`` module, or None when it has no assigned tier.

    ``q3`` itself and any top-level module not listed in ``TIERS`` (for example
    ``q3.__main__``) are untiered: they are entry points that sit above
    everything and are never imported by package code.
    """
    parts = module_name.split(".")
    if len(parts) < 2:
        return None
    return TIERS.get(parts[1])


def imported_q3_modules(tree: ast.AST) -> list[tuple[str, int]]:
    """Collect every ``q3.*`` module imported by *tree*, with its line number."""
    found: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "q3" or alias.name.startswith("q3."):
                    found.append((alias.name, node.lineno))
        elif isinstance(node, ast.ImportFrom):
            # ``level > 0`` is a relative import; the package uses absolute
            # imports throughout, so treat only absolute ``q3.*`` here.
            if (
                node.level == 0
                and node.module
                and (node.module == "q3" or node.module.startswith("q3."))
            ):
                found.append((node.module, node.lineno))
    return found


def check_file(path: Path) -> list[Violation]:
    """Return the layering violations in a single source file."""
    module_name = module_name_for(path)
    if module_name is None:
        return []
    own_tier = tier_of(module_name)
    if own_tier is None:
        return []

    tree = ast.parse(path.read_text(), filename=str(path))
    violations: list[Violation] = []
    for imported, line in imported_q3_modules(tree):
        imported_tier = tier_of(imported)
        if imported_tier is None or imported_tier <= own_tier:
            continue
        violations.append(
            Violation(
                path,
                line,
                f"upward import: {module_name} (tier {own_tier}, "
                f"{TIER_DESCRIPTIONS[own_tier]}) imports {imported} "
                f"(tier {imported_tier}, {TIER_DESCRIPTIONS[imported_tier]}). "
                "Move the shared code down to a tier both sides can import, "
                "or invert the dependency — do not defer the import into a "
                "function (see AGENTS.md § Python Package Layering).",
            )
        )
    return violations


def check_same_tier_cycles(paths: list[Path]) -> list[Violation]:
    """Return violations for pairs of same-tier modules that import each other."""
    edges: dict[str, set[str]] = {}
    locations: dict[tuple[str, str], tuple[Path, int]] = {}

    for path in paths:
        module_name = module_name_for(path)
        if module_name is None or tier_of(module_name) is None:
            continue
        tree = ast.parse(path.read_text(), filename=str(path))
        for imported, line in imported_q3_modules(tree):
            if tier_of(imported) != tier_of(module_name):
                continue
            edges.setdefault(module_name, set()).add(imported)
            locations.setdefault((module_name, imported), (path, line))

    violations: list[Violation] = []
    reported: set[frozenset[str]] = set()
    for source, targets in edges.items():
        for target in targets:
            # ``q3.io.output_writer`` importing ``q3.io`` is not a cycle; the
            # package __init__ importing its own submodules is how a package
            # is assembled. Only flag genuine two-way edges between distinct
            # modules that are not in an ancestor relationship.
            if target.startswith(source + ".") or source.startswith(target + "."):
                continue
            if source not in edges.get(target, set()):
                continue
            pair = frozenset({source, target})
            if pair in reported:
                continue
            reported.add(pair)
            path, line = locations[(source, target)]
            violations.append(
                Violation(
                    path,
                    line,
                    f"same-tier import cycle: {source} <-> {target}. "
                    "Extract the shared code into a lower tier (see "
                    "AGENTS.md § Python Package Layering).",
                )
            )
    return violations


def collect_python_files(arguments: list[str]) -> list[Path]:
    """Expand command-line arguments into the list of files to check."""
    if not arguments:
        return sorted(PACKAGE_ROOT.rglob("*.py"))
    paths: list[Path] = []
    for argument in arguments:
        path = Path(argument)
        if path.is_dir():
            paths.extend(sorted(path.rglob("*.py")))
        elif path.suffix == ".py":
            paths.append(path)
    return paths


def main() -> int:
    """Check the layering and report violations; return the process exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="*", help="files or directories to check")
    arguments = parser.parse_args()

    # Cycle detection needs the whole package in view, not just the files a
    # pre-commit run happens to pass in, or a cycle whose two halves land in
    # different commits would slip through.
    files_to_check = collect_python_files(arguments.paths)
    all_package_files = sorted(PACKAGE_ROOT.rglob("*.py"))

    violations: list[Violation] = []
    for path in files_to_check:
        violations.extend(check_file(path))
    violations.extend(check_same_tier_cycles(all_package_files))

    for violation in sorted(violations, key=lambda v: (str(v.path), v.line)):
        print(violation, file=sys.stderr)

    if violations:
        print(
            f"\n{len(violations)} import-layering violation(s). "
            "Tiers are defined in scripts/repo/check_python_import_layering.py.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
