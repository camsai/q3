"""tests/python/test_results_model.py — reading a run's results file through its generated model.

``q3.io.results`` is generated from ``schemas/q3/results.json``
(``test_generated_models.py`` holds the two together). What these tests pin is
what a reader relies on beyond the generated shape: each record read as ESSE's
own model by its name, Q3's own quantities as attributes, a record that
breaks its ESSE schema refused, the eigenvalues in the shape the band-structure
and DOS routines take, and a file round-tripping unchanged.
"""

import json
import typing

import pytest
from pydantic import ValidationError
from q3.io.results import classes as model
from q3.io.results.base import RecordsBlock


def test_every_record_the_schema_allows_reads_as_an_attribute_of_its_name():
    """The records are ESSE's models; each one the schema allows has its getter."""
    (records,) = [
        kind
        for kind in typing.get_args(model.Properties.model_fields["records"].annotation)
        if kind is not type(None)
    ]
    (union,) = typing.get_args(records)
    for record in typing.get_args(union):
        (name,) = [member.value for member in record.model_fields["name"].annotation]
        assert isinstance(getattr(RecordsBlock, name), property), name


DOCUMENT = {
    "structure": {"material": {"name": "Si", "basis": {"elements": [], "coordinates": []}}},
    "provenance": {"generator": {"engine_name": "q3", "engine_commit": ""}},
    "properties": {
        "records": [
            {"name": "total_energy", "units": "Ry", "value": -15.85},
            {
                "name": "total_energy_contributions",
                "units": "Ry",
                "hartree": {"name": "hartree", "value": 1.1},
                "ewald": {"name": "ewald", "value": -16.9},
                "temperatureEntropy": {"name": "temperature_entropy", "value": 0.0},
            },
            {
                "name": "atomic_forces",
                "units": "Ry/a.u.",
                "values": [
                    {"id": 2, "value": [-0.01, 0.0, 0.0]},
                    {"id": 1, "value": [0.01, 0.0, 0.0]},
                ],
            },
            {
                "name": "band_gaps",
                "values": [
                    {"type": "direct", "units": "Ry", "value": 1.0},
                    {"type": "indirect", "units": "Ry", "value": 0.9},
                ],
            },
        ],
        "energy_contributions": {
            "hubbard": {"name": "hubbard", "units": "Ry", "value": 0.25},
        },
        "isolated_system_correction": {
            "method": "makov-payne",
            "terms": {"first_order": {"name": "first_order", "units": "Ry", "value": -0.1}},
            "total": {"name": "total", "units": "Ry", "value": -0.1},
        },
    },
    "runtime_log": {
        "flavor": "scf",
        "converged": True,
        "incomplete": {"stress": "the Hubbard stress is not implemented for this projector"},
    },
}
# The records a spin-polarized run adds, listed with the others.
DOCUMENT["properties"]["records"] += [
    {"name": "magnetization_total", "units": "uB", "value": 2.2},
    {"name": "magnetization_absolute", "units": "uB", "value": 2.4},
    {"name": "band_edge_valence_max", "units": "Ry", "value": -0.2, "kpoint": [0.5, 0.0, 0.0]},
    {"name": "band_edge_conduction_min", "units": "Ry", "value": 0.1},
    {
        "name": "eigenvalues_at_kpoints",
        "units": "Ry",
        "values": [
            {
                "kpoint": [0.0, 0.0, 0.0],
                "weight": 1.5,
                "eigenvalues": [
                    {"spin": 0.5, "energies": [-0.5, 0.1], "occupations": [1.0, 0.0]},
                    {"spin": -0.5, "energies": [-0.4, 0.2], "occupations": [1.0, 0.0]},
                ],
            },
            {
                "kpoint": [0.5, 0.0, 0.0],
                "weight": 0.5,
                "eigenvalues": [
                    {"spin": 0.5, "energies": [-0.3, 0.3], "occupations": [1.0, 0.0]},
                    {"spin": -0.5, "energies": [-0.2, 0.4], "occupations": [0.5, 0.0]},
                ],
            },
        ],
    },
]


def test_each_record_is_read_as_esses_model_by_its_name():
    properties = model.Results.model_validate(DOCUMENT).properties
    assert properties.total_energy.value == -15.85
    assert properties.total_energy.units.value == "Ry"
    assert properties.total_energy_contributions.ewald.value == -16.9
    assert [entry.id for entry in properties.atomic_forces.values] == [2, 1]
    gaps = {gap.type.value: gap.value for gap in properties.band_gaps.values}
    assert gaps == {"direct": 1.0, "indirect": 0.9}
    assert properties.stress_tensor is None
    assert properties.magnetization_absolute.value == 2.4
    assert properties.magnetization_total.units.value == "uB"
    assert properties.band_edge_valence_max.kpoint == [0.5, 0.0, 0.0]
    assert properties.band_edge_conduction_min.value == 0.1


def test_q3s_own_quantities_read_as_attributes():
    results = model.Results.model_validate(DOCUMENT)
    properties = results.properties
    assert properties.energy_contributions.hubbard.value == 0.25
    assert properties.energy_contributions.dispersion is None
    assert properties.isolated_system_correction.terms["first_order"].value == -0.1
    assert results.runtime_log.incomplete.forces is None
    assert results.runtime_log.incomplete.stress.startswith("the Hubbard stress")


def test_a_record_that_breaks_its_esse_schema_is_refused():
    broken = {"properties": {"records": [{"name": "total_energy", "units": "Ry", "value": "low"}]}}
    with pytest.raises(ValidationError):
        model.Results.model_validate(broken)


def test_a_record_esse_has_no_model_for_here_is_refused():
    invented = {"properties": {"records": [{"name": "total_magnetization", "value": 0.0}]}}
    with pytest.raises(ValidationError):
        model.Results.model_validate(invented)


def test_eigenvalues_read_as_kpoint_spin_band_arrays():
    eigenvalues, occupations, weights = model.eigenvalue_arrays(
        model.Results.model_validate(DOCUMENT).properties.eigenvalues_at_kpoints
    )
    assert eigenvalues == [[[-0.5, 0.1], [-0.4, 0.2]], [[-0.3, 0.3], [-0.2, 0.4]]]
    assert occupations[1][1] == [0.5, 0.0]
    assert weights == [1.5, 0.5]


def test_eigenvalues_without_occupations_read_none_for_them():
    record = model.Results.model_validate(
        {
            "properties": {
                "records": [
                    {
                        "name": "eigenvalues_at_kpoints",
                        "units": "Ry",
                        "values": [{"eigenvalues": [{"energies": [0.1]}]}],
                    }
                ]
            }
        }
    ).properties.eigenvalues_at_kpoints
    assert model.eigenvalue_arrays(record) == ([[[0.1]]], None, [None])


def test_a_file_round_trips_unchanged(tmp_path):
    path = tmp_path / "si_results.json"
    path.write_text(json.dumps(DOCUMENT))
    assert model.Results.read(path).to_document() == DOCUMENT


def test_a_newer_engines_key_survives_the_round_trip():
    document = json.loads(json.dumps(DOCUMENT))
    document["runtime_log"]["a_newer_key"] = [1, 2]
    document["a_newer_block"] = {"x": 1}
    assert model.Results.model_validate(document).to_document() == document
