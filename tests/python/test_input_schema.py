"""tests/python/test_input_schema.py — the input schema mirrors the input files.

``schemas/q3/input.json`` describes the YAML manifest format, and its per-object
key allow-lists are code-generated into ``src/generated/q3/input.hpp``, which is
what ``src/io/input/parser/yaml.hpp`` rejects unknown keys against. That makes
the schema the one place a manifest key is declared — but only as long as the
schema keeps up with the manifests people actually write.

These tests close that loop from the other end: every committed example manifest
is checked against the schema, so a key or enum value that examples rely on and
the schema never learned about fails here rather than at a user's parse. The
schema's own structural invariants (the closed objects the parser looks up, the
shared ``scf``/``self_consistent_field`` definition) are checked too, since
losing one of those silently un-validates a whole manifest level.

Deliberately not a full JSON Schema validation: this checks keys and enumerated
values, which is the drift that actually happens, without pulling in a
validation dependency the project does not otherwise carry.
"""

import json
from pathlib import Path

import pytest
import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent.parent
SCHEMA_PATH = REPOSITORY_ROOT / "schemas" / "q3" / "input.json"
EXAMPLES_ROOT = REPOSITORY_ROOT / "examples"


def _load_schema():
    """Load the input schema with its local ``#/definitions/...`` refs resolved."""
    schema = json.loads(SCHEMA_PATH.read_text())

    def resolve(node):
        if isinstance(node, dict):
            reference = node.get("$ref")
            if reference is not None:
                assert reference.startswith("#/"), f"non-local $ref: {reference}"
                target = schema
                for step in reference[2:].split("/"):
                    target = target[step]
                return resolve(target)
            return {key: resolve(value) for key, value in node.items()}
        if isinstance(node, list):
            return [resolve(item) for item in node]
        return node

    return resolve(schema)


SCHEMA = _load_schema()


def _manifest_paths():
    """Every committed YAML manifest: a mapping with a top-level ``material``."""
    candidates = sorted(EXAMPLES_ROOT.rglob("*.yml")) + sorted(EXAMPLES_ROOT.rglob("*.yaml"))
    manifests = []
    for path in candidates:
        document = yaml.safe_load(path.read_text())
        # examples/data/ also holds non-manifest YAML (pseudopotential metadata).
        if isinstance(document, dict) and "material" in document:
            manifests.append(path)
    return manifests


MANIFEST_PATHS = _manifest_paths()


def _check(document, schema, path, problems):
    """Collect every key/enum disagreement between a document and its schema."""
    branches = schema.get("oneOf") or schema.get("anyOf")
    if branches:
        # A document satisfying any branch satisfies the location; report the
        # branch complaints only when none of them accepts it.
        for branch in branches:
            branch_problems = []
            _check(document, branch, path, branch_problems)
            if not branch_problems:
                return
        for branch in branches:
            _check(document, branch, path, problems)
        return

    if isinstance(document, dict):
        properties = schema.get("properties", {})
        if schema.get("additionalProperties") is False:
            for key in document:
                if key not in properties:
                    problems.append(f"{path or 'manifest root'}: unknown key '{key}'")
        for key, value in document.items():
            if key in properties:
                _check(value, properties[key], f"{path}.{key}" if path else key, problems)
        return

    if isinstance(document, list):
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for item in document:
                _check(item, item_schema, path + "[]", problems)
        return

    allowed_values = schema.get("enum")
    if allowed_values is not None and document not in allowed_values:
        problems.append(f"{path}: value {document!r} is not one of {allowed_values}")


# ── Committed manifests match the schema ─────────────────────────────────────


def test_examples_directory_has_manifests():
    """Guard the guard: an empty file list would make every check below vacuous."""
    assert len(MANIFEST_PATHS) > 10


@pytest.mark.parametrize("manifest_path", MANIFEST_PATHS, ids=lambda path: path.name)
def test_example_manifest_matches_schema(manifest_path):
    """Every key and enumerated value an example uses is declared in the schema."""
    document = yaml.safe_load(manifest_path.read_text())
    problems = []
    _check(document, SCHEMA, "", problems)
    relative_path = manifest_path.relative_to(REPOSITORY_ROOT)
    assert not problems, f"{relative_path} disagrees with schemas/q3/input.json:\n" + "\n".join(
        f"  {problem}" for problem in problems
    )


def test_a_misspelled_key_is_caught():
    """The checker itself must fail on the drift it exists to catch."""
    problems = []
    _check({"material": "x.json", "controll": {}}, SCHEMA, "", problems)
    assert problems


# ── Manifests Q3 itself writes ───────────────────────────────────────────────


def test_written_scf_and_nscf_inputs_match_schema(tmp_path):
    """``q3.io.input_writer`` is the other producer of manifests; its output must fit too.

    A key the writer emits and the schema never learned about would be
    rejected by the C++ parser on the very next run of a Python workflow.
    """
    input_writer = pytest.importorskip("q3.io.input_writer")

    scf_path = input_writer.write_scf_input(
        tmp_path / "scf.yml",
        material_path="material.json",
        pseudopotentials={"Si": "Si.upf"},
        smearing="gaussian",
        smearing_width=0.02,
    )
    nscf_path = input_writer.write_nscf_input(
        tmp_path / "nscf.yml",
        scf_input_path=scf_path,
        kpoints_list=[(0.0, 0.0, 0.0, 1.0), (0.5, 0.0, 0.0, 1.0)],
        number_of_bands=8,
        compute_projections=True,
    )

    for written_path in (scf_path, nscf_path):
        problems = []
        _check(yaml.safe_load(Path(written_path).read_text()), SCHEMA, "", problems)
        assert not problems, f"{written_path.name} disagrees with the schema:\n" + "\n".join(
            f"  {problem}" for problem in problems
        )


# ── Schema invariants the parser depends on ──────────────────────────────────

# Kept in step with validated_schema_paths in tests/unit/io/input/parser/test_yaml.cpp.
PARSER_VALIDATED_PATHS = [
    "",
    "model",
    "method",
    "method.cutoffs",
    "method.scf",
    "method.scf.hubbard[]",
    "method.scf.hubbard_starting_eigenvalues[]",
    "method.scf.mixing",
    "method.scf.smearing",
    "method.scf.kgrid",
    "method.band_structure",
    "control",
    "control.band_path[]",
]


def _closed_object_paths(node, path, found):
    """Dotted paths of every object the schema closes, as the generator sees them."""
    if not isinstance(node, dict):
        return
    for keyword in ("oneOf", "anyOf", "allOf"):
        for branch in node.get(keyword, []):
            _closed_object_paths(branch, path, found)
    if node.get("additionalProperties") is False:
        found[path] = sorted(node.get("properties", {}))
    for name, subschema in node.get("properties", {}).items():
        _closed_object_paths(subschema, f"{path}.{name}" if path else name, found)
    items = node.get("items")
    if isinstance(items, dict):
        _closed_object_paths(items, path + "[]", found)


def test_every_level_the_parser_validates_is_closed():
    """Losing ``additionalProperties: false`` anywhere here breaks parsing outright.

    yaml.hpp looks each of these paths up in the generated allow-lists; a path
    the schema stops closing throws for every manifest carrying that block.
    """
    closed = {}
    _closed_object_paths(SCHEMA, "", closed)
    missing = [path for path in PARSER_VALIDATED_PATHS if path not in closed]
    assert not missing, f"schemas/q3/input.json no longer closes: {missing}"


def test_scf_spellings_share_one_definition():
    """``scf`` and ``self_consistent_field`` must stay the same block, not two copies."""
    closed = {}
    _closed_object_paths(SCHEMA, "", closed)
    assert closed["method.scf"] == closed["method.self_consistent_field"]


def test_material_stays_open():
    """An ESSE material carries fields Q3 does not consume, so nothing under it is closed."""
    closed = {}
    _closed_object_paths(SCHEMA, "", closed)
    assert not [path for path in closed if path == "material" or path.startswith("material.")]
