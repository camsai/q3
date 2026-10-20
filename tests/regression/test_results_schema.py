"""tests/regression/test_results_schema.py — results.json matches its published schema.

``schemas/q3/results.json`` is what a reader outside this repository -- a
results ledger, a platform importer, anyone consuming a finished run -- is
entitled to rely on. A schema nothing validates against is a document, not a
contract: it drifts from the writer the first time a key is added, and the
readers it was written for go back to guessing.

These tests close that loop from both ends:

* **From the writer**: every key ``FlavorResult::results_json()`` can emit is
  declared in the schema. The schema closes its root, so an undeclared key
  would make a real run's output invalid -- caught here, statically, without
  running anything.
* **From a real run**: the file a converged silicon SCF and a converged
  spin-polarized SCF actually leave on disk is validated against the schema,
  with its ``$ref``\\ s into ESSE resolved. Two cases rather than one because
  the optional blocks differ: magnetization and the DFT+U occupations exist
  only on the second.
* **On the schema itself**: that it rejects an unknown key and a missing
  required one, since a schema permissive enough to accept anything would pass
  both tests above while checking nothing.
"""

import json
import re
import subprocess
from pathlib import Path

import jsonschema
import pytest

from helpers import EXAMPLES_DIRECTORY, Q3_ROOT, apply_config_overlay, load_results_json

SCHEMA_PATH = Q3_ROOT / "schemas" / "q3" / "results.json"
ESSE_SCHEMA_ROOT = Q3_ROOT / "schemas" / "esse" / "dist" / "js" / "schema"
RESULTS_WRITER_PATH = Q3_ROOT / "src" / "layer_4_computation" / "flavors" / "result.hpp"

pytestmark = pytest.mark.skipif(
    not ESSE_SCHEMA_ROOT.is_dir(),
    reason=(
        "the ESSE schemas the results schema references are not checked out "
        "(run `git submodule update --init schemas/esse`)"
    ),
)


# ── Schema loading ───────────────────────────────────────────────────


def _resolve_references(node, base_directory):
    """Inline every ``$ref``, local (``#/...``) or relative to a sibling file.

    Resolving here rather than handing a resolver to ``jsonschema`` keeps the
    test off that library's evolving reference API (``RefResolver`` is
    deprecated, ``referencing`` is its replacement) for a schema whose refs are
    all static files on disk -- and it means a broken path fails with the path
    in the message rather than as a validation error deep inside a document.
    """
    if isinstance(node, list):
        return [_resolve_references(item, base_directory) for item in node]
    if not isinstance(node, dict):
        return node

    reference = node.get("$ref")
    if reference is None:
        return {key: _resolve_references(value, base_directory) for key, value in node.items()}

    location, _, fragment = reference.partition("#")
    if location:
        target_path = (base_directory / location).resolve()
        assert target_path.is_file(), f"$ref points at no file: {reference}"
        target = json.loads(target_path.read_text())
        target_directory = target_path.parent
    else:
        target = _ROOT_SCHEMA
        target_directory = base_directory
    for step in (part for part in fragment.split("/") if part):
        target = target[step]
    return _resolve_references(target, target_directory)


_ROOT_SCHEMA = json.loads(SCHEMA_PATH.read_text())
SCHEMA = _resolve_references(_ROOT_SCHEMA, SCHEMA_PATH.parent)


# ── Runs ─────────────────────────────────────────────────────────────

SILICON_INPUT = EXAMPLES_DIRECTORY / "total_energy--semiconductor--silicon" / "nc.yml"
IRON_INPUT = EXAMPLES_DIRECTORY / "total_energy--metal--magnetic--iron" / "nc.yml"

# Coarse enough to stay inside a regression tier's time budget. Nothing here is
# checked for its value -- the shape of the output is the subject -- so the
# grids are the smallest that still reach every block the schema describes.
SILICON_OVERLAY = {
    "method": {"scf": {"kgrid": {"dimensions": [2, 2, 2], "shifts": [1, 1, 1]}}},
    "control": {"prefix": "si_schema"},
}
IRON_OVERLAY = {
    "method": {"scf": {"kgrid": {"dimensions": [4, 4, 4], "shifts": [1, 1, 1]}}},
    "control": {"prefix": "fe_schema"},
}


def _run_and_load(q3_command, base_input, overlay, prefix, run_directory, timeout=900):
    """Run q3 on an example plus an overlay; return the parsed results.json."""
    merged_input = apply_config_overlay(base_input, overlay, run_directory)
    completed = subprocess.run(
        list(q3_command) + [str(merged_input)],
        cwd=str(run_directory),
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    assert completed.returncode == 0, (
        f"q3 exited {completed.returncode}\n{completed.stdout[-3000:]}"
    )
    return load_results_json(Path(run_directory) / "output", prefix)


@pytest.fixture(scope="module")
def silicon_results(q3_command, tmp_path_factory):
    directory = tmp_path_factory.mktemp("results_schema_silicon")
    return _run_and_load(q3_command, SILICON_INPUT, SILICON_OVERLAY, "si_schema", directory)


@pytest.fixture(scope="module")
def iron_results(q3_command, tmp_path_factory):
    directory = tmp_path_factory.mktemp("results_schema_iron")
    return _run_and_load(q3_command, IRON_INPUT, IRON_OVERLAY, "fe_schema", directory)


# ── Tests ────────────────────────────────────────────────────────────


def test_schema_declares_every_key_the_writer_can_emit():
    """The closed root must name every key results_json() assigns.

    Every one of them goes through ``json_output["<key>"]`` in the one writer,
    so the writer's own source is the list to check against -- no run needed,
    and a key added without a schema entry fails before it can reach a file.
    """
    emitted = set(re.findall(r'json_output\["([^"]+)"\]', RESULTS_WRITER_PATH.read_text()))
    assert emitted, "found no emitted keys in the results writer; has it moved?"
    declared = set(SCHEMA["properties"])
    assert emitted <= declared, (
        "results.json keys the writer emits but schemas/q3/results.json does not "
        f"declare: {sorted(emitted - declared)}"
    )


def test_silicon_results_validate_against_the_schema(silicon_results):
    jsonschema.validate(instance=silicon_results, schema=SCHEMA)


def test_silicon_results_carry_the_derived_blocks(silicon_results):
    """The keys a reader of a semiconductor run needs, present and self-consistent."""
    assert silicon_results["converged"] is True
    assert "pressure_kbar" in silicon_results
    assert set(silicon_results["force_contributions_ry_per_bohr"]) >= {
        "local",
        "ewald",
        "nonlocal",
    }
    gap = silicon_results["band_gap_ry"]
    valence = silicon_results["valence_band_maximum_ry"]
    conduction = silicon_results["conduction_band_minimum_ry"]
    assert gap == pytest.approx(conduction - valence)
    assert valence <= silicon_results["fermi_energy_ry"] <= conduction


def test_spin_polarized_results_validate_against_the_schema(iron_results):
    jsonschema.validate(instance=iron_results, schema=SCHEMA)


def test_spin_polarized_results_carry_both_magnetizations(iron_results):
    """A magnetic run reports both moments; the total alone cannot describe an AFM state."""
    assert iron_results["units"]["magnetization"] == "uB"
    total = iron_results["total_magnetization_bohr_magneton"]
    absolute = iron_results["absolute_magnetization_bohr_magneton"]
    assert absolute >= abs(total) - 1e-9


def test_the_schema_rejects_an_undeclared_key(silicon_results):
    """A permissive schema would pass every test above while checking nothing."""
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance={**silicon_results, "energy_in_hartrees": 1.0}, schema=SCHEMA)


def test_the_schema_rejects_a_missing_required_key(silicon_results):
    incomplete = {key: value for key, value in silicon_results.items() if key != "units"}
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=incomplete, schema=SCHEMA)
