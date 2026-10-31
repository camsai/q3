"""tests/regression/test_results_schema.py — results.json matches its published schema.

``schemas/q3/results.json`` is what a reader outside this repository -- a
results ledger, a platform importer, anyone consuming a finished run -- is
entitled to rely on. A schema nothing validates against is a document, not a
contract: it drifts from the writer the first time a key is added, and the
readers it was written for go back to guessing.

These tests close that loop from both ends:

* **From the writer**: every flavor fills ``generated::Results``, the C++ struct
  generated from this schema, and that struct writes exactly the keys the
  schema declares -- checked statically, without running anything.
* **From a real run**: the file a converged silicon SCF and a converged
  spin-polarized SCF actually leave on disk is validated against the schema,
  with its ``$ref``\\ s into ESSE resolved. Two cases rather than one because
  the optional blocks differ: magnetization and the DFT+U occupations exist
  only on the second.
* **On the schema itself**: that it rejects an unknown key and a missing
  required one, since a schema permissive enough to accept anything would pass
  both tests above while checking nothing. It also rejects a ``properties.records``
  entry ESSE does not define, which is what keeps new property records
  going to ESSE first (issue #314).
"""

import json
import re
import subprocess
from pathlib import Path

import jsonschema
import pytest
from q3.io.results.classes import Results

from helpers import (
    EXAMPLES_DIRECTORY,
    Q3_ROOT,
    apply_config_overlay,
    load_results_json,
    repository_schemas,
    results_schema,
)

RESULTS_WRITER_PATH = Q3_ROOT / "src" / "io" / "output" / "results.hpp"
GENERATED_RESULTS_PATH = Q3_ROOT / "src" / "generated" / "q3" / "results.hpp"

pytestmark = pytest.mark.skipif(
    repository_schemas() is None,
    reason=(
        "mat3ra-esse, which supplies the ESSE schemas the results schema "
        "references, is not installed"
    ),
)

# ── Schema loading ───────────────────────────────────────────────────

SCHEMA = results_schema()


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
    """Run q3 on an example plus an overlay; return results.json as written.

    The document itself, not the model: what is under test here is the file
    against the schema. ``load_results_json`` validates it on the way.
    """
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
    load_results_json(Path(run_directory) / "output", prefix)
    return json.loads((Path(run_directory) / "output" / f"{prefix}_results.json").read_text())


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
    """The writer fills the struct generated from the schema, which writes exactly its keys.

    The generated header is checked against the schema by the generator's own
    ``--check``; this ties the writer to it, so a key cannot reach a file by
    any other route.
    """
    assert "generated::Results results;" in RESULTS_WRITER_PATH.read_text(), (
        "the flavors no longer start from generated::Results; has new_results() moved?"
    )
    generated = GENERATED_RESULTS_PATH.read_text()
    serializer = generated[generated.index("const Results& data) {") :]
    serializer = serializer[: serializer.index("\n}\n")]
    emitted = set(re.findall(r'json_output\["([^"]+)"\]', serializer))
    assert emitted == set(SCHEMA["properties"])


def test_silicon_results_validate_against_the_schema(silicon_results):
    jsonschema.validate(instance=silicon_results, schema=SCHEMA)


def test_silicon_results_carry_the_derived_blocks(silicon_results):
    """The keys a reader of a semiconductor run needs, present and self-consistent."""
    results = Results.model_validate(silicon_results)
    assert results.runtime_log.converged is True
    assert results.properties.pressure is not None
    assert set(results.properties.force_contributions) >= {"local", "ewald", "nonlocal"}
    gaps = {gap.type.value: gap for gap in results.properties.band_gaps.values}
    valence = results.properties.band_edge_valence_max.value
    conduction = results.properties.band_edge_conduction_min.value
    assert gaps["indirect"].value == pytest.approx(conduction - valence)
    assert valence <= results.properties.fermi_energy.value <= conduction


def test_spin_polarized_results_validate_against_the_schema(iron_results):
    jsonschema.validate(instance=iron_results, schema=SCHEMA)


def test_spin_polarized_results_carry_both_magnetizations(iron_results):
    """A magnetic run reports both moments; the total alone cannot describe an AFM state."""
    properties = Results.model_validate(iron_results).properties
    total = properties.magnetization_total.value
    absolute = properties.magnetization_absolute.value
    assert absolute >= abs(total) - 1e-9


def test_the_schema_rejects_an_undeclared_key(silicon_results):
    """A permissive schema would pass every test above while checking nothing."""
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance={**silicon_results, "energy_in_hartrees": 1.0}, schema=SCHEMA)


def test_the_schema_rejects_a_missing_required_key(silicon_results):
    """A missing group, and a missing key inside one, are both refused."""
    without_compute = {key: value for key, value in silicon_results.items() if key != "runtime_log"}
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=without_compute, schema=SCHEMA)
    properties = {
        key: value
        for key, value in silicon_results["properties"].items()
        if key != "energy_contributions"
    }
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance={**silicon_results, "properties": properties}, schema=SCHEMA)


UNIT_SUFFIXES = ("_ry", "_ev", "_kbar", "_bohr", "_per_bohr", "_bohr_magneton", "_hartree")


def _keys(node):
    """Every object key in *node*, at any depth."""
    if isinstance(node, dict):
        for key, value in node.items():
            yield key
            yield from _keys(value)
    elif isinstance(node, list):
        for item in node:
            yield from _keys(item)


def test_silicon_results_are_esse_records_and_written_once(silicon_results):
    """Issue #314: what ESSE defines is an ESSE record, and is not also a key of its own."""
    records = silicon_results["properties"]["records"]
    properties = {record["name"]: record for record in records}
    assert len(properties) == len(records), "a property is listed twice"
    assert set(properties) == {
        "total_energy",
        "total_energy_contributions",
        "fermi_energy",
        "atomic_forces",
        "stress_tensor",
        "pressure",
        "band_gaps",
        "eigenvalues_at_kpoints",
        "band_edge_valence_max",
        "band_edge_conduction_min",
        "convergence_electronic",
    }
    assert not [key for key in _keys(silicon_results) if key.endswith(UNIT_SUFFIXES)], (
        "a key names its unit; state it in the quantity's units field instead"
    )
    forces = properties["atomic_forces"]["values"]
    assert [entry["id"] for entry in forces] == [
        coordinate["id"]
        for coordinate in silicon_results["structure"]["material"]["basis"]["coordinates"]
    ]
    gaps = {gap["type"]: gap for gap in properties["band_gaps"]["values"]}
    assert gaps["direct"]["value"] >= gaps["indirect"]["value"]


def test_the_schema_rejects_a_property_esse_does_not_define(silicon_results):
    """New property records go to ESSE first (AGENTS.md, "Output schemas")."""
    invented = {"name": "total_magnetization", "units": "uB", "value": 0.0}
    properties = silicon_results["properties"]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(
            instance={
                **silicon_results,
                "properties": {**properties, "records": [*properties["records"], invented]},
            },
            schema=SCHEMA,
        )
