#!/usr/bin/env python3
"""
Q3's JSON schemas as files: ESSE's written from mat3ra-esse, any of them resolved.

ESSE's Python package carries every schema resolved: each ``$ref`` inlined and
each ``allOf`` merged (``mat3ra.esse.data.schemas.SCHEMAS``). Q3's schemas
``$ref`` them by relative path into schemas/esse, and the code
generators, the schema tests and CMake read them as files. This script writes
one file per schema into schemas/esse/, at the path ESSE derives from its
``$id`` (dashes become underscores), with a stamp naming the package version.
The directory is not committed: whoever reads it calls ``ensure_esse()``,
which rewrites it only when it is missing or holds another version's schemas.

``resolved(path)`` gives any schema under schemas/q3 with every ``$ref``
inlined, ESSE's included: one self-contained document a validator needs no
resolver for. ``--out`` writes every Q3 schema that way.

The ESSE version is the ``mat3ra-esse`` pin in pyproject.toml, which also
supplies the Python models (``mat3ra.esse.models``) the generated Q3 models use.

Usage:
    python scripts/repo/schemas.py [--check] [--out DIRECTORY]
"""

import argparse
import importlib.metadata
import json
import shutil
import sys
from pathlib import Path

from mat3ra.esse.data.schemas import SCHEMAS

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ESSE_DIRECTORY = PROJECT_ROOT / "schemas" / "esse"
Q3_DIRECTORY = PROJECT_ROOT / "schemas" / "q3"
STAMP_NAME = ".mat3ra-esse-version"
PACKAGE = "mat3ra-esse"


def installed_version():
    """The installed mat3ra-esse version, which the written schemas are stamped with."""
    return importlib.metadata.version(PACKAGE)


def is_current(directory=ESSE_DIRECTORY):
    """True when *directory* holds the installed version's schemas."""
    stamp = directory / STAMP_NAME
    return stamp.is_file() and stamp.read_text().strip() == installed_version()


def schema_path(schema_id):
    """Where ESSE puts the schema with *schema_id*, relative to the schema root."""
    return Path(schema_id.replace("-", "_") + ".json")


def write(directory=ESSE_DIRECTORY):
    """Replace *directory* with one file per schema of the installed package."""
    unmerged = [schema["$id"] for schema in SCHEMAS if '"allOf"' in json.dumps(schema)]
    if unmerged:
        raise RuntimeError(
            f"{PACKAGE} {installed_version()} ships schemas with unmerged allOf "
            f"({len(unmerged)}, e.g. {unmerged[0]}); install the version pyproject.toml pins"
        )
    if directory.exists():
        shutil.rmtree(directory)
    for schema in SCHEMAS:
        destination = directory / schema_path(schema["$id"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(schema, indent=4, ensure_ascii=False))
    (directory / STAMP_NAME).write_text(installed_version() + "\n")


def ensure_esse(directory=ESSE_DIRECTORY):
    """Make *directory* hold the installed version's schemas; write them only if it does not."""
    if is_current(directory):
        return False
    print(f"Writing ESSE schemas from {PACKAGE} {installed_version()}", file=sys.stderr)
    write(directory)
    return True


def _inline(node, base_directory, root):
    """*node* with every ``$ref`` inlined: local (``#/...``, into *root*) or a file beside it."""
    if isinstance(node, list):
        return [_inline(item, base_directory, root) for item in node]
    if not isinstance(node, dict):
        return node
    reference = node.get("$ref")
    if reference is None:
        return {key: _inline(value, base_directory, root) for key, value in node.items()}
    location, _, fragment = reference.partition("#")
    if location:
        target_path = (base_directory / location).resolve()
        if not target_path.is_file():
            raise FileNotFoundError(f"$ref points at no file: {reference} (from {base_directory})")
        target = json.loads(target_path.read_text())
        root = target
        base_directory = target_path.parent
    else:
        target = root
    for step in (part for part in fragment.split("/") if part):
        target = target[step]
    return _inline(target, base_directory, root)


def resolved(path):
    """The schema at *path* (relative to schemas/q3) with every ``$ref`` inlined."""
    ensure_esse()
    schema_file = (Q3_DIRECTORY / path).resolve()
    schema = json.loads(schema_file.read_text())
    return _inline(schema, schema_file.parent, schema)


def q3_schema_paths():
    """Every Q3 schema, relative to schemas/q3."""
    return sorted(path.relative_to(Q3_DIRECTORY) for path in Q3_DIRECTORY.rglob("*.json"))


def write_resolved(directory):
    """Write every Q3 schema, resolved, under *directory* at its path in schemas/q3."""
    for path in q3_schema_paths():
        destination = directory / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(resolved(path), indent=4, ensure_ascii=False) + "\n")


def main():
    """Write ESSE's schemas, or check them, or write every Q3 schema resolved."""
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument(
        "--check", action="store_true", help="fail, without writing, if the schemas are not current"
    )
    parser.add_argument(
        "--out", type=Path, metavar="DIRECTORY", help="also write every Q3 schema, resolved, here"
    )
    arguments = parser.parse_args()
    if arguments.check:
        if not is_current():
            print(
                f"{ESSE_DIRECTORY.relative_to(PROJECT_ROOT)} does not hold {PACKAGE} {installed_version()}."
            )
            sys.exit(1)
        print(f"ESSE schemas are {PACKAGE} {installed_version()}.")
        return
    ensure_esse()
    if arguments.out:
        write_resolved(arguments.out)


if __name__ == "__main__":
    main()
