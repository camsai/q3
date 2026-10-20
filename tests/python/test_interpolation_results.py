"""tests/python/test_interpolation_results.py — the Python model mirrors the results schema.

``schemas/q3/interpolation_results.json`` generates the engine's writer;
``q3.io.interpolation_results`` is written by hand. These tests hold the two to
each other block by block -- the same keys, no more and no fewer -- and pin the
reader's stance the schema states: every engine-written field is optional to
read, the workflow's ladder attempt is written whole, and a file the workflow
annotated round-trips unchanged, nulls and unknown keys included.
"""

import json
import typing
from pathlib import Path

import pytest
from q3.io import interpolation_results as model
from q3.workflows.band_distance import BandDistance

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent.parent
SCHEMA = json.loads((REPOSITORY_ROOT / "schemas" / "q3" / "interpolation_results.json").read_text())

#: Every titled object of the schema, and the model that stands for it.
BLOCKS = {
    "Interpolation results": model.InterpolationResults,
    "Interpolation results density of states": model.DensityOfStates,
    "Interpolation results interpolation": model.Interpolation,
    "Interpolation results spread": model.Spread,
    "Interpolation results disentanglement": model.Disentanglement,
    "Interpolation results excluded band": model.ExcludedBand,
    "Interpolation results quality": model.Quality,
    "Interpolation results quality spread": model.QualitySpread,
    "Interpolation results quality centers": model.QualityCenters,
    "Interpolation results hamiltonian decay": model.HamiltonianDecay,
    "Interpolation results ladder attempt": model.LadderAttempt,
    "Interpolation results band distance": model.BandDistanceReport,
}

#: The ladder attempt's keys that are null for an evaluation that raised.
NULLABLE_ATTEMPT_KEYS = {"eta_ev", "eta_maximum_ev", "localized", "localization_converged"}


def _titled_objects(node, found):
    """Every object in the schema that carries a title, with its node."""
    if isinstance(node, dict):
        if node.get("type") == "object" and "properties" in node and "title" in node:
            found[node["title"]] = node
        for value in node.values():
            _titled_objects(value, found)
    elif isinstance(node, list):
        for item in node:
            _titled_objects(item, found)
    return found


TITLED = _titled_objects(SCHEMA, {})


def test_every_described_block_has_a_model_and_every_model_a_block():
    assert set(TITLED) == set(BLOCKS)


@pytest.mark.parametrize("title", sorted(BLOCKS))
def test_the_model_carries_exactly_the_schemas_keys(title):
    assert set(TITLED[title]["properties"]) == set(BLOCKS[title].model_fields), title


def _classes_in(annotation) -> set:
    """Every class an annotation names, through ``X | None`` and ``list[X]``."""
    arguments = typing.get_args(annotation)
    if not arguments:
        return {annotation}
    return set().union(*(_classes_in(argument) for argument in arguments))


def _nested_block_title(specification: dict) -> str | None:
    """The titled block a property holds, directly or as the items of an array."""
    holder = (
        specification.get("items", {}) if specification.get("type") == "array" else specification
    )
    return holder.get("title") if holder.get("type") == "object" else None


@pytest.mark.parametrize("title", sorted(BLOCKS))
def test_a_block_nested_in_the_schema_is_typed_as_its_own_model(title):
    """The key sets could agree with every nested block left a plain dict; the nesting is held too."""
    fields = BLOCKS[title].model_fields
    for name, specification in TITLED[title]["properties"].items():
        nested = _nested_block_title(specification)
        if nested is not None:
            assert BLOCKS[nested] in _classes_in(fields[name].annotation), f"{title}.{name}"


def test_the_nesting_test_sees_the_blocks_it_is_for():
    nested = {
        _nested_block_title(specification)
        for block in TITLED.values()
        for specification in block["properties"].values()
    } - {None}
    assert nested == set(BLOCKS) - {"Interpolation results"}


@pytest.mark.parametrize("title", sorted(set(BLOCKS) - {"Interpolation results ladder attempt"}))
def test_every_engine_written_field_is_optional_to_read(title):
    required = [name for name, field in BLOCKS[title].model_fields.items() if field.is_required()]
    assert required == [], f"{title}: a reader must not require {required}"


def test_the_ladder_attempt_requires_what_describes_it_and_nulls_what_judges_it():
    attempt = TITLED["Interpolation results ladder attempt"]
    assert set(attempt["required"]) == set(attempt["properties"])
    nullable = {
        name for name, spec in attempt["properties"].items() if "null" in spec.get("type", [])
    }
    assert nullable == NULLABLE_ATTEMPT_KEYS
    required = {
        name for name, field in model.LadderAttempt.model_fields.items() if field.is_required()
    }
    assert required == set(attempt["properties"]) - NULLABLE_ATTEMPT_KEYS


def test_the_stubs_subset_reads():
    """The unit-test stub writes a handful of keys; the model reads them and nothing else."""
    results = model.InterpolationResults.model_validate(
        {
            "kpoints": [[0.0, 0.0, 0.0]],
            "eigenvalues_ry": [[[-0.5, 0.25]]],
            "fermi_energy_ry": -0.5,
            "interpolation": {
                "spread": {"localized": True, "localization_converged": True},
                "disentanglement": {"metallic": False, "bands_excluded_from_freeze": []},
                "quality": {"hamiltonian_decay": {"ratio": 0.01}},
            },
        }
    )
    assert results.calculation is None
    assert results.interpolation.spread.localized is True
    assert results.interpolation.spread.gauge_invariant_bohr_squared is None
    assert results.interpolation.quality.hamiltonian_decay.ratio == 0.01
    assert results.interpolation.quality.ladder is None


ANNOTATED = {
    "calculation": "interpolation",
    "fermi_energy_ry": 0.5,
    "provenance": {"generator": {"engine_name": "q3", "engine_commit": ""}},
    "interpolation": {
        "method": "wannier",
        "description": "wannier (projection gauge): 4 orbitals on 27 lattice vectors",
        "record": "./output/si.save/interpolation.q3ip",
        "number_of_bands": 4,
        "total_bloch_sum_weight": 27.0,
        "spread": {"gauge_invariant_bohr_squared": 0.0, "localized": False},
        "quality": {
            "hamiltonian_decay": {
                "outermost_shell_distance_bohr": 10.0,
                "outermost_shell_size": 12,
                "outermost_shell_maximum_ry": 0.001,
                "largest_hopping_ry": 0.1,
                "ratio": 0.01,
            },
            "status": "accept",
            "reason": "eta 0.9 meV below the gate",
            "ladder": [
                {
                    "rung": "default",
                    "grid": [4, 4, 4],
                    "settings": {"localization": "maximal"},
                    "status": "error:unconverged",
                    "reason": "Error: disentanglement did not converge in 4000 iterations.",
                    "eta_ev": None,
                    "eta_maximum_ev": None,
                    "localized": None,
                    "localization_converged": None,
                }
            ],
            "band_distance": {
                "eta_ev": 0.0009,
                "eta_maximum_ev": 0.004,
                "number_of_states": 200,
                "reference": "path",
                "number_of_kpoints": 50,
            },
        },
    },
}


def test_a_file_the_workflow_annotated_round_trips_with_its_nulls(tmp_path):
    path = tmp_path / "si_interpolation_results.json"
    path.write_text(json.dumps(ANNOTATED))
    results = model.InterpolationResults.read(path)
    assert results.interpolation.quality.status == "accept"
    assert results.interpolation.quality.ladder[0].eta_ev is None
    assert results.interpolation.quality.band_distance.number_of_kpoints == 50
    assert results.to_document() == ANNOTATED

    results.write(path)
    assert json.loads(path.read_text()) == ANNOTATED
    assert path.read_text().endswith("}\n")


def test_the_quality_block_is_made_where_a_file_carries_none_and_found_where_it_does():
    bare = model.InterpolationResults()
    block = bare.quality_block()
    block.status = "accept"
    assert bare.to_document() == {"interpolation": {"quality": {"status": "accept"}}}
    assert bare.quality_block() is block

    annotated = model.InterpolationResults.model_validate(ANNOTATED)
    assert annotated.quality_block() is annotated.interpolation.quality
    assert annotated.quality_block().reason == "eta 0.9 meV below the gate"


def test_a_newer_engines_key_survives_the_round_trip():
    document = json.loads(json.dumps(ANNOTATED))
    document["interpolation"]["spread"]["a_newer_key"] = [1, 2]
    document["a_newer_block"] = {"x": None}
    results = model.InterpolationResults.model_validate(document)
    assert results.to_document() == document


def test_band_distance_to_report_is_the_files_block():
    distance = BandDistance(eta=0.0009, eta_maximum=0.004, number_of_states=200)
    report = distance.to_report(reference="path", number_of_kpoints=50)
    assert isinstance(report, model.BandDistanceReport)
    assert report.model_dump() == ANNOTATED["interpolation"]["quality"]["band_distance"]
    assert distance.as_dict() == {
        "eta_ev": 0.0009,
        "eta_maximum_ev": 0.004,
        "number_of_states": 200,
    }
