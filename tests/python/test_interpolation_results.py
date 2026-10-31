"""tests/python/test_interpolation_results.py — reading the interpolation results file through its generated model.

``q3.io.interpolation.results`` is generated from
``schemas/q3/interpolation/results.json`` (``test_generated_models.py`` holds
the two together). What these tests pin is the reader's stance: the unit-test
stub's subset reads, a file the workflow annotated round-trips unchanged --
a raised ladder attempt's nulls and unknown keys included -- the quality
block is made where a file has none, and the density of states is ESSE's
record.
"""

import json
import typing

from mat3ra.esse.models.properties_directory.non_scalar.density_of_states import (
    DensityOfStatesPropertySchema,
)
from q3.io.interpolation.results import classes as model
from q3.workflows.band_distance import BandDistance


def test_the_stubs_subset_reads():
    """The unit-test stub writes a handful of keys; the model reads them and nothing else."""
    results = model.InterpolationResults.model_validate(
        {
            "properties": {
                "eigenvalues_at_kpoints": {
                    "name": "eigenvalues_at_kpoints",
                    "units": "Ry",
                    "values": [
                        {"kpoint": [0.0, 0.0, 0.0], "eigenvalues": [{"energies": [-0.5, 0.25]}]}
                    ],
                },
                "fermi_energy": {"name": "fermi_energy", "units": "Ry", "value": -0.5},
            },
            "construction": {
                "spread": {"localized": True, "localization_converged": True},
                "disentanglement": {"metallic": False, "bands_excluded_from_freeze": []},
            },
            "quality": {"hamiltonian_decay": {"ratio": 0.01}},
        }
    )
    assert results.calculation is None
    assert results.construction.spread.localized is True
    assert results.construction.spread.gauge_invariant_bohr_squared is None
    assert results.quality.hamiltonian_decay.ratio == 0.01
    assert results.quality.ladder is None


ANNOTATED = {
    "calculation": "interpolation",
    "provenance": {"generator": {"engine_name": "q3", "engine_commit": ""}},
    "properties": {"fermi_energy": {"name": "fermi_energy", "units": "Ry", "value": 0.5}},
    "construction": {
        "method": "wannier",
        "description": "wannier (projection gauge): 4 orbitals on 27 lattice vectors",
        "record": "./output/si.save/interpolation.q3ip",
        "number_of_bands": 4,
        "total_bloch_sum_weight": 27.0,
        "spread": {"gauge_invariant_bohr_squared": 0.0, "localized": False},
    },
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
}


def test_a_file_the_workflow_annotated_round_trips_with_its_nulls(tmp_path):
    path = tmp_path / "si_interpolation_results.json"
    path.write_text(json.dumps(ANNOTATED))
    results = model.InterpolationResults.read(path)
    assert results.quality.status == "accept"
    assert results.quality.ladder[0].eta_ev is None
    assert results.quality.band_distance.number_of_kpoints == 50
    assert results.to_document() == ANNOTATED

    results.write(path)
    assert json.loads(path.read_text()) == ANNOTATED
    assert path.read_text().endswith("}\n")


def test_the_quality_block_is_made_where_a_file_carries_none_and_found_where_it_does():
    bare = model.InterpolationResults()
    block = bare.quality_block()
    block.status = "accept"
    assert bare.to_document() == {"quality": {"status": "accept"}}
    assert bare.quality_block() is block

    annotated = model.InterpolationResults.model_validate(ANNOTATED)
    assert annotated.quality_block() is annotated.quality
    assert annotated.quality_block().reason == "eta 0.9 meV below the gate"


def test_a_newer_engines_key_survives_the_round_trip():
    document = json.loads(json.dumps(ANNOTATED))
    document["construction"]["spread"]["a_newer_key"] = [1, 2]
    document["a_newer_block"] = {"x": None}
    results = model.InterpolationResults.model_validate(document)
    assert results.to_document() == document


def test_band_distance_to_report_is_the_files_block():
    distance = BandDistance(eta=0.0009, eta_maximum=0.004, number_of_states=200)
    report = distance.to_report(reference="path", number_of_kpoints=50)
    assert isinstance(report, model.BandDistanceReport)
    assert report.model_dump() == ANNOTATED["quality"]["band_distance"]
    assert distance.as_dict() == {
        "eta_ev": 0.0009,
        "eta_maximum_ev": 0.004,
        "number_of_states": 200,
    }


def test_the_density_of_states_is_read_by_esses_model():
    """The block is ESSE's density_of_states record, read by ESSE's own model."""
    field = model.Properties.model_fields["density_of_states"]
    assert DensityOfStatesPropertySchema in typing.get_args(field.annotation)
    document = {
        "name": "density_of_states",
        "xAxis": {"label": "energy", "units": "Ry"},
        "yAxis": {"label": "density of states", "units": "states/unitcell"},
        "xDataArray": [-1.0, 0.0],
        "yDataSeries": [[0.5, 1.5]],
        "legend": [],
    }
    results = model.InterpolationResults.model_validate(
        {"properties": {"density_of_states": document}}
    )
    assert results.to_document()["properties"]["density_of_states"] == document
