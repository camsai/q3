#!/usr/bin/env python3
"""
JSON Schema -> Python (pydantic) model generator.

The Python side of schemas/manifest.yaml: every Q3 schema entry with a
``python`` block is generated into src/py/q3/generated/ by
datamodel-code-generator, the tool and the flags ESSE generates
``mat3ra.esse.models`` with. Before generating, each schema is prepared:

* Each Q3 schema file that is an object with properties is a module of its
  own, at the schema's path (``schemas/q3/results/properties.json`` is
  ``q3.generated.results.properties``), and a ``$ref`` to it is its class.
  Any other Q3 ``$ref`` (an array, an open map) is inlined into the module
  that refers to it.
* A ``$ref`` into ESSE becomes ESSE's own model (``customTypePath``), so a
  record is ``mat3ra.esse.models``' class and is never generated twice. A
  reference into an untitled part of an ESSE schema (band_gaps' per-k-point
  entry) is matched to the ESSE class carrying exactly its keys.
* The ``bases`` the manifest names, by schema ``$id``, become the base class
  of that block's model (``customBasePath``): the reading helpers a block
  needs (records by name, eigenvalue arrays, read/write) are written once
  there, by hand, and the data itself is all generated.
* Every field is optional and unknown keys are kept, so a reader takes a
  subset or a newer file as it comes: ``required`` and
  ``additionalProperties: false`` are the writer's guarantee, checked by the
  schema tests against real runs, not the reader's precondition.

Usage:
    python scripts/repo/generate_python_from_schemas.py [--check]
"""

import argparse
import importlib
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml
from pydantic import BaseModel

import schemas

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MANIFEST_PATH = PROJECT_ROOT / "schemas" / "manifest.yaml"
PYTHON_ROOT = PROJECT_ROOT / "src" / "py"
ESSE_ROOT = schemas.ESSE_DIRECTORY.resolve()

# ESSE's own flags (schemas/esse/.pre-commit-config.yaml), plus what a reader
# of Q3's files needs: optional fields, unknown keys kept, Python 3.12 syntax.
CODEGEN_FLAGS = [
    "--input-file-type",
    "jsonschema",
    "--output-model-type",
    "pydantic_v2.BaseModel",
    "--use-field-description",
    "--use-double-quotes",
    "--enable-version-header",
    "--use-title-as-name",
    "--disable-timestamp",
    "--use-default",
    "--remove-special-field-name-prefix",
    "--force-optional",
    "--reuse-model",
    "--use-union-operator",
    "--allow-extra-fields",
    "--target-python-version",
    "3.12",
]


def class_name(title):
    """The class name datamodel-code-generator gives a title (``--use-title-as-name``)."""
    return "".join(
        word[:1].upper() + word[1:] for word in re.split(r"[^0-9A-Za-z]+", title) if word
    )


def esse_model(path, fragment):
    """The ``mat3ra.esse.models`` class an ESSE schema (or a part of one) is, or None."""
    module_name = "mat3ra.esse.models." + ".".join(
        path.relative_to(ESSE_ROOT).with_suffix("").parts
    )
    module = importlib.import_module(module_name)
    node = json.loads(path.read_text())
    for step in (part for part in fragment.split("/") if part):
        node = node[step]
    title = node.get("title")
    if title and hasattr(module, class_name(title)):
        return f"{module_name}.{class_name(title)}", node
    if node.get("type") == "object" and "properties" in node:
        keys = set(node["properties"])
        matches = [
            name
            for name, candidate in vars(module).items()
            if isinstance(candidate, type)
            and issubclass(candidate, BaseModel)
            and candidate.__module__ == module_name
            and set(candidate.model_fields) == keys
        ]
        if len(matches) == 1:
            return f"{module_name}.{matches[0]}", node
    return None, node


class Preparer:
    """Prepares one module's schema for the code generator (see the module docstring)."""

    def __init__(self, bases, modules):
        self.bases = bases
        # Every other generated module, by its schema's $id
        self.modules = modules
        # The modules this one refers to: (schema file, its property path here)
        self.references = []

    def prepare(self, node, directory, path=()):
        """*node* (read from *directory*, at property *path*) ready for the generator."""
        if isinstance(node, list):
            return [self.prepare(item, directory, path) for item in node]
        if not isinstance(node, dict):
            return node
        all_of = node.get("allOf")
        if (
            isinstance(all_of, list)
            and len(all_of) == 1
            and "$ref" in all_of[0]
            and "properties" not in node
        ):
            node = {**{key: value for key, value in node.items() if key != "allOf"}, **all_of[0]}
        if "$ref" in node:
            return self.resolve(node, directory, path)
        prepared = {
            key: (
                {name: self.prepare(value[name], directory, path + (name,)) for name in value}
                if key == "properties" and isinstance(value, dict)
                else self.prepare(value, directory, path)
            )
            for key, value in node.items()
            # `contains` constraints say which records must be present: the
            # writer's guarantee, which a model of the shape cannot hold.
            if not (key == "allOf" and all("contains" in part for part in value))
        }
        if prepared.get("additionalProperties") is False:
            del prepared["additionalProperties"]
        if prepared.get("type") == "object" and "properties" not in prepared:
            prepared.setdefault("additionalProperties", True)
        base = self.bases.get(prepared.get("$id"))
        if base:
            prepared["customBasePath"] = base
        return prepared

    def resolve(self, node, directory, path):
        """A ``$ref`` node: ESSE's model, another module's class, or the inlined target."""
        location, _, fragment = node["$ref"].partition("#")
        target = (directory / location).resolve() if location else None
        rest = {key: value for key, value in node.items() if key != "$ref"}
        if target is not None and target.is_relative_to(ESSE_ROOT):
            model, inner = esse_model(target, fragment)
            if model:
                return {**rest, "customTypePath": model}
            return {**self.prepare(inner, target.parent, path), **rest}
        document = json.loads(target.read_text())
        for step in (part for part in fragment.split("/") if part):
            document = document[step]
        module = self.modules.get(document.get("$id"))
        if module:
            self.references.append((target, path))
            return {**rest, "customTypePath": f"{module}.{class_name(document['title'])}"}
        return {**self.prepare(document, target.parent, path), **rest}


def make_opaque(schema, names):
    """*schema* with the property at the path *names* read as a plain JSON object.

    The manifest's ``opaque`` paths are built as JSON where they are made
    (the material, the provenance stamp) and validated by the schema, not
    modelled: the C++ writer carries them as JSON, and so does the reader.
    """
    properties = dict(schema["properties"])
    first, rest = names[0], names[1:]
    if rest:
        properties[first] = make_opaque(properties[first], rest)
    else:
        description = properties[first].get("description")
        properties[first] = {"type": "object", "additionalProperties": True}
        if description:
            properties[first]["description"] = description
    return {**schema, "properties": properties}


def is_module(schema):
    """True when a Q3 schema file is a module of its own: an object with properties."""
    return schema.get("type") == "object" and bool(schema.get("properties"))


def generated_source(schema_path, bases, modules, opaque=()):
    """The module datamodel-code-generator writes for one schema file.

    Returns the source and the schema files it refers to as modules, each with
    the ``opaque`` paths that fall inside it.
    """
    root = json.loads(schema_path.read_text())
    others = {key: value for key, value in modules.items() if key != root["$id"]}
    preparer = Preparer(bases, others)
    prepared = preparer.prepare(root, schema_path.parent)
    children = []
    for target, path in preparer.references:
        inside = [names[len(path) :] for names in opaque if names[: len(path)] == path]
        children.append((target, inside))
    for names in opaque:
        if not any(names[: len(path)] == path for _, path in preparer.references):
            prepared = make_opaque(prepared, list(names))
    with tempfile.TemporaryDirectory() as scratch:
        input_path = Path(scratch) / schema_path.name
        output_path = Path(scratch) / "model.py"
        input_path.write_text(json.dumps(prepared, indent=2))
        subprocess.run(
            [sys.executable, "-m", "datamodel_code_generator"]
            + ["--input", str(input_path), "--output", str(output_path)]
            + CODEGEN_FLAGS,
            check=True,
        )
        source = output_path.read_text()
    # The header names the prepared copy's temporary path; name the schema.
    relative = schema_path.relative_to(PROJECT_ROOT)
    source = re.sub(r"#   filename:  .*", f"#   filename:  {relative}", source, count=1)
    return source, children


def module_path(module, packages):
    """Where *module* is written: a package's ``__init__.py`` when it has modules below it."""
    relative = Path(*module.split("."))
    if module in packages:
        return PYTHON_ROOT / relative / "__init__.py"
    return PYTHON_ROOT / relative.with_suffix(".py")


def main():
    """Generate every module, or with ``--check`` fail on one out of date."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--check", action="store_true", help="fail if a generated file is stale")
    arguments = parser.parse_args()

    schemas.ensure_esse()
    manifest = yaml.safe_load(MANIFEST_PATH.read_text())
    group = manifest["q3"]
    base_path = PROJECT_ROOT / group["base_path"]
    entries = [entry for entry in group["schemas"] if "python" in entry]

    # Every schema file below a document is a module below the document's
    # module, at the same path: results/properties.json, beside results.json
    # (q3.generated.results), is q3.generated.results.properties.
    modules = {}
    for entry in entries:
        document = base_path / entry["schema"]
        package = entry["python"]["module"].rsplit(".", len(Path(entry["schema"]).parts))[0]
        for path in [document, *sorted(document.with_suffix("").rglob("*.json"))]:
            schema = json.loads(path.read_text())
            if path == document or is_module(schema):
                relative = path.relative_to(base_path).with_suffix("")
                modules[schema["$id"]] = ".".join([package, *relative.parts])
    packages = {module.rsplit(".", 1)[0] for module in modules.values()}

    stale = []
    written = []
    work = [
        (base_path / entry["schema"], entry["python"].get("bases", {}), entry.get("opaque", ()))
        for entry in entries
    ]
    while work:
        schema_path, bases, opaque = work.pop(0)
        module = modules[json.loads(schema_path.read_text())["$id"]]
        output_path = module_path(module, packages)
        if output_path in written:
            continue
        source, children = generated_source(
            schema_path, bases, modules, [tuple(path.split(".")) for path in opaque]
        )
        work.extend(
            (child, bases, [".".join(names) for names in inside]) for child, inside in children
        )
        written.append(output_path)
        if arguments.check:
            if not output_path.exists() or output_path.read_text() != source:
                stale.append(str(output_path.relative_to(PROJECT_ROOT)))
            continue
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(source)
        print(f"  Generated: {output_path.relative_to(PROJECT_ROOT)}")

    # A module no schema generates any more (one that became a package, say)
    leftovers = {
        path
        for path in (PYTHON_ROOT / "q3" / "generated").rglob("*.py")
        if path.name != "__init__.py"
    } - set(written)
    for path in sorted(leftovers):
        if arguments.check:
            stale.append(str(path.relative_to(PROJECT_ROOT)))
        else:
            path.unlink()
            print(f"  Removed: {path.relative_to(PROJECT_ROOT)}")

    if stale:
        print("Stale generated Python models (run scripts/repo/generate_python_from_schemas.py):")
        for path in stale:
            print(f"  {path}")
        sys.exit(1)
    if arguments.check:
        print("All generated Python models are up-to-date.")


if __name__ == "__main__":
    main()
