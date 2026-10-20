#!/usr/bin/env python3
"""Ratchet docstring coverage of the public C++ and Python APIs (AGENTS-generic.md § Comment Style).

Roughly half of the public functions declared in ``src/`` headers carry no
doc comment. Documenting them all in one change is neither reviewable nor
useful; instead this script measures the undocumented public declarations
per file, and the committed baseline
``scripts/repo/check_docstring_coverage_baseline.json`` caps each file at what it
has today. A new undocumented public API fails the check; documenting one
requires tightening the baseline (``--write-baseline``) in the same change,
so the slack can never be spent later. This is the same ratchet shape as
``check_cpp_type_queries.py``.

What counts as a public declaration
-----------------------------------

C++ (``src/**/*.hpp``, ``src/generated/`` excluded): a class/struct
definition, or a function/method declaration, that is reachable from outside
-- at namespace scope, or under ``public:`` in a class (``struct`` defaults to
public, ``class`` to private). Skipped as self-describing: constructors and
destructors, ``= default``/``= delete`` members, operators, and anything in
an ``internal``/``detail`` namespace. Function bodies are not entered.

A declaration is documented when the nearest non-blank line above it (after
skipping ``template <...>`` lines and attributes) ends a ``/** ... */`` block
or is a ``///`` comment -- the two doc-comment forms AGENTS-generic.md
allows.

Python (``src/py/q3/**/*.py``): module-level functions and classes, and
methods of public classes, whose names do not start with ``_``, checked with
``ast.get_docstring``.

The scanner is a line-oriented heuristic, not a C++ parser. It is
deterministic, which is all a ratchet needs: a miscount moves the baseline
and the measurement together.

Usage::

    python scripts/repo/check_docstring_coverage.py                  # check
    python scripts/repo/check_docstring_coverage.py --write-baseline # tighten
    python scripts/repo/check_docstring_coverage.py --report         # coverage figures
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
PYTHON_PACKAGE_ROOT = SOURCE_ROOT / "py" / "q3"
BASELINE_PATH = Path(__file__).resolve().parent / "check_docstring_coverage_baseline.json"

CPP_HEADER_SUFFIXES = {".hpp", ".h"}
EXCLUDED_DIRECTORIES = ("generated",)
PRIVATE_NAMESPACES = ("internal", "detail")

TEMPLATE_LINE_RE = re.compile(r"^\s*template\s*<.*>\s*$")
ATTRIBUTE_LINE_RE = re.compile(r"^\s*\[\[.*\]\]\s*$")
ACCESS_RE = re.compile(r"^\s*(public|private|protected)\s*:")
NAMESPACE_RE = re.compile(r"^\s*(?:inline\s+)?namespace\b\s*([\w:]*)")
CLASS_RE = re.compile(
    r"^\s*(?:template\s*<.*>\s*)?(class|struct)\s+(?:\[\[.*?\]\]\s*)?(?:alignas\([^)]*\)\s*)?(\w+)"
    r"(?:\s+final)?\s*(?::[^;{]*)?(?:\{|$)"
)
ENUM_RE = re.compile(r"^\s*enum\b")
FUNCTION_RE = re.compile(
    r"^\s*(?:template\s*<.*>\s*)?"
    r"(?:(?:virtual|static|inline|constexpr|explicit|friend|extern|\[\[\w+\]\])\s+)*"
    r"(?P<return>[\w:<>,\s\*&~]+?)\s*(?<![\w:])(?P<name>~?\w+)\s*\("
)
NON_DECLARATION_STARTS = (
    "return",
    "if",
    "else",
    "for",
    "while",
    "switch",
    "case",
    "do",
    "using",
    "typedef",
    "static_assert",
    "throw",
    "delete",
    "new",
    "goto",
    "co_return",
)
SPECIAL_MEMBER_RE = re.compile(r"=\s*(?:default|delete)\s*;")


def strip_line_comments_and_strings(line: str) -> str:
    """Blank string literals and drop a trailing // comment (for brace counting)."""
    line = re.sub(r'"(?:\\.|[^"\\])*"', '""', line)
    line = re.sub(r"'(?:\\.|[^'\\])*'", "''", line)
    index = line.find("//")
    return line if index == -1 else line[:index]


def is_doc_comment_end(line: str) -> bool:
    """Whether a source line closes a /** */ block or is a /// comment."""
    stripped = line.strip()
    return stripped.startswith("///") or stripped.endswith("*/")


class _Scope:
    """One brace-delimited scope while scanning a header."""

    def __init__(self, kind: str, public: bool, name: str = "") -> None:
        self.kind = kind  # namespace | class | body | other
        self.public = public
        self.name = name


def _code_lines(text: str) -> list[tuple[str, str]]:
    """Return (raw line, code-only line) pairs, with /* */ comment text blanked in the code view."""
    pairs: list[tuple[str, str]] = []
    in_block = False
    for raw in text.splitlines():
        code_chars: list[str] = []
        index = 0
        while index < len(raw):
            if in_block:
                end = raw.find("*/", index)
                if end == -1:
                    index = len(raw)
                else:
                    in_block = False
                    index = end + 2
            else:
                start = raw.find("/*", index)
                line_comment = raw.find("//", index)
                if start == -1 or (line_comment != -1 and line_comment < start):
                    code_chars.append(raw[index:])
                    break
                code_chars.append(raw[index:start])
                in_block = True
                index = start + 2
        pairs.append((raw, strip_line_comments_and_strings("".join(code_chars))))
    return pairs


def scan_cpp_header(text: str) -> tuple[int, int]:
    """Return (public declarations, undocumented public declarations) for one header."""
    pairs = _code_lines(text)
    scopes: list[_Scope] = [_Scope("namespace", True)]
    total = 0
    undocumented = 0
    statement_start = True
    pending_scope: _Scope | None = None
    paren_depth = 0

    for index, (raw, code) in enumerate(pairs):
        stripped = code.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            continue
        scope = scopes[-1]
        in_body = any(s.kind == "body" for s in scopes)
        exposed = (
            not in_body
            and all(s.public for s in scopes)
            and not any(s.name in PRIVATE_NAMESPACES for s in scopes)
        )

        access = ACCESS_RE.match(code)
        if access and scope.kind == "class":
            scope.public = access.group(1) == "public"
            statement_start = True
            continue

        if statement_start and paren_depth == 0 and not in_body:
            namespace = NAMESPACE_RE.match(code)
            class_match = None if namespace else CLASS_RE.match(code)
            if namespace:
                pending_scope = _Scope("namespace", True, namespace.group(1).split("::")[-1])
            elif class_match and not stripped.endswith(";"):
                pending_scope = _Scope(
                    "class", class_match.group(1) == "struct", class_match.group(2)
                )
                if exposed:
                    total += 1
                    undocumented += not _documented(pairs, index)
            elif ENUM_RE.match(code):
                pending_scope = _Scope("other", True)
            elif scope.kind in ("namespace", "class") and not TEMPLATE_LINE_RE.match(code):
                function = FUNCTION_RE.match(code)
                if function and _is_counted_function(function, code, scope):
                    pending_scope = _Scope("body", True)
                    if exposed:
                        total += 1
                        undocumented += not _documented(pairs, index)
                elif function is None:
                    pending_scope = _Scope("other", True)

        for char in code:
            if char == "(":
                paren_depth += 1
            elif char == ")":
                paren_depth = max(0, paren_depth - 1)
            elif char == "{":
                scopes.append(pending_scope or _Scope("other", True))
                pending_scope = None
            elif char == "}":
                if len(scopes) > 1:
                    scopes.pop()
            elif char == ";" and paren_depth == 0:
                pending_scope = None

        last = stripped[-1]
        statement_start = (
            paren_depth == 0
            and (
                last in ";{}" or TEMPLATE_LINE_RE.match(code) is not None or stripped.endswith(":")
            )
            or ATTRIBUTE_LINE_RE.match(code) is not None
        )
    return total, undocumented


def _is_counted_function(function: re.Match[str], code: str, scope: _Scope) -> bool:
    """Whether a FUNCTION_RE match is a documentable function declaration."""
    return_part = function.group("return").strip()
    name = function.group("name")
    first_word = re.match(r"\w+", code.strip())
    if first_word and first_word.group(0) in NON_DECLARATION_STARTS:
        return False
    if not return_part or return_part in ("return", "else", "new", "delete", "throw"):
        return False  # a call statement or a constructor without a return type
    if name.startswith("~") or name == scope.name or "operator" in code.split("(")[0]:
        return False
    if SPECIAL_MEMBER_RE.search(code):
        return False
    return not return_part.endswith(("=", ".", "->"))


def _documented(pairs: list[tuple[str, str]], index: int) -> bool:
    """Whether the declaration at pairs[index] is preceded by a doc comment."""
    cursor = index - 1
    while cursor >= 0:
        raw = pairs[cursor][0]
        if not raw.strip():
            return False
        if TEMPLATE_LINE_RE.match(raw) or ATTRIBUTE_LINE_RE.match(raw):
            cursor -= 1
            continue
        return is_doc_comment_end(raw)
    return False


def scan_python_module(text: str) -> tuple[int, int]:
    """Return (public definitions, undocumented public definitions) for one Python module."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return 0, 0
    total = 0
    undocumented = 0
    definitions = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
    for node in tree.body:
        if not isinstance(node, definitions) or node.name.startswith("_"):
            continue
        total += 1
        undocumented += ast.get_docstring(node) is None
        if isinstance(node, ast.ClassDef):
            for member in node.body:
                if isinstance(member, definitions) and not member.name.startswith("_"):
                    total += 1
                    undocumented += ast.get_docstring(member) is None
    return total, undocumented


def _excluded(path: Path, root: Path) -> bool:
    return any(part in EXCLUDED_DIRECTORIES for part in path.relative_to(root).parts)


def measure_all(source_root: Path = None) -> dict[str, tuple[int, int]]:
    """Return {repo-relative path: (public, undocumented)} for every scanned file."""
    source_root = source_root or SOURCE_ROOT
    python_root = source_root / "py" / "q3"
    results: dict[str, tuple[int, int]] = {}
    for path in sorted(source_root.rglob("*")):
        if not path.is_file() or _excluded(path, source_root):
            continue
        text = None
        if path.suffix in CPP_HEADER_SUFFIXES:
            text = path.read_text(encoding="utf-8", errors="replace")
            figures = scan_cpp_header(text)
        elif path.suffix == ".py" and python_root in path.parents:
            text = path.read_text(encoding="utf-8", errors="replace")
            figures = scan_python_module(text)
        if text is not None and figures[0]:
            results[path.relative_to(source_root.parent).as_posix()] = figures
    return results


def undocumented_counts(measured: dict[str, tuple[int, int]]) -> dict[str, int]:
    """Project the measurement onto the baseline shape {file: undocumented count}."""
    return {file: figures[1] for file, figures in measured.items() if figures[1]}


def addon_baseline_paths() -> list[Path]:
    """Every add-on's own baseline: ``src/addons/<name>/<this baseline's file name>``."""
    return sorted((SOURCE_ROOT / "addons").glob(f"*/{BASELINE_PATH.name}"))


def addon_prefix(file: str) -> str | None:
    """Return the ``src/addons/<name>/`` prefix *file* sits under, or None for a core file."""
    parts = file.split("/")
    if len(parts) > 3 and parts[0] == "src" and parts[1] == "addons":
        return "/".join(parts[:3]) + "/"
    return None


def load_baseline(path: Path = None) -> dict[str, int]:
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


def compare(current: dict[str, int], baseline: dict[str, int]) -> tuple[list[str], list[str]]:
    """Return (increases, decreases) as human-readable lines."""
    increases: list[str] = []
    decreases: list[str] = []
    for file in sorted(set(current) | set(baseline)):
        now, allowed = current.get(file, 0), baseline.get(file, 0)
        line = f"{file}: undocumented public declarations {allowed} -> {now}"
        if now > allowed:
            increases.append(line)
        elif now < allowed:
            decreases.append(line)
    return increases, decreases


def write_baseline(current, path: Path = None) -> None:
    """Write the baseline JSON (sorted, 4-space indent, trailing newline).

    Without *path*, each add-on's entries go to the baseline beside its code
    and the rest to the core baseline (see load_baseline).
    """
    if path is not None:
        path.write_text(json.dumps(current, indent=4, sort_keys=True) + "\n", encoding="utf-8")
        return
    write_baseline(
        {file: value for file, value in current.items() if addon_prefix(file) is None},
        BASELINE_PATH,
    )
    addon_directories = sorted(
        directory for directory in (SOURCE_ROOT / "addons").glob("*") if directory.is_dir()
    )
    for addon_directory in addon_directories:
        prefix = f"src/addons/{addon_directory.name}/"
        entries = {file: value for file, value in current.items() if file.startswith(prefix)}
        target = addon_directory / BASELINE_PATH.name
        if entries or target.exists():
            write_baseline(entries, target)


def _report(measured: dict[str, tuple[int, int]]) -> None:
    for label, suffixes in (("C++ headers", (".hpp", ".h")), ("Python", (".py",))):
        rows = [f for file, f in measured.items() if file.endswith(suffixes)]
        public = sum(f[0] for f in rows)
        missing = sum(f[1] for f in rows)
        percent = 100.0 * (public - missing) / public if public else 100.0
        print(
            f"{label:12s} {public - missing}/{public} public declarations documented ({percent:.1f}%)"
        )


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point; returns the process exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--write-baseline", action="store_true", help="rewrite the baseline")
    parser.add_argument(
        "--allow-increase",
        action="store_true",
        help="with --write-baseline, permit raising counts (needs review justification)",
    )
    parser.add_argument("--report", action="store_true", help="print coverage figures")
    arguments = parser.parse_args(argv)

    measured = measure_all()
    current = undocumented_counts(measured)
    baseline = load_baseline()
    increases, decreases = compare(current, baseline)

    if arguments.report:
        _report(measured)

    if arguments.write_baseline:
        if increases and not arguments.allow_increase:
            print("Refusing to raise the docstring baseline without --allow-increase:")
            for line in increases:
                print(f"  {line}")
            return 1
        write_baseline(current)
        print(f"Wrote {BASELINE_PATH.name}: {sum(current.values())} undocumented declarations.")
        return 0

    if increases:
        print(
            "New undocumented public APIs (AGENTS-generic.md § Comment Style) -- add a doc comment:"
        )
        for line in increases:
            print(f"  {line}")
    if decreases:
        print(
            "Docstring coverage improved -- tighten the ratchet with "
            "`python scripts/repo/check_docstring_coverage.py --write-baseline`:"
        )
        for line in decreases:
            print(f"  {line}")
    return 1 if increases or decreases else 0


if __name__ == "__main__":
    sys.exit(main())
