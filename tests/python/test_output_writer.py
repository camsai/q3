"""tests/python/test_output_writer.py — Unit tests for q3.io.output_writer.

Tests result extraction from both new flat schema and legacy nested schema
without invoking the q3 binary.
"""

import numpy as np
import pytest
from q3.errors import IncompleteDerivativeError, SCFNotConvergedError
from q3.io.output_writer import (
    extract_band_structure,
    extract_dos,
    extract_forces,
    extract_lowdin_charges,
    extract_pdos,
    extract_stress,
    extract_total_energy,
    raise_if_derivative_incomplete,
    raise_if_not_converged,
)
from q3.io.results.classes import Results

# ── Sample results dicts ──────────────────────────────────────────────────────


def _eigenvalues(energies, occupations=None, weights=None):
    """ESSE's ``eigenvalues_at_kpoints`` record for energies[k-point][spin][band] in Ry."""
    values = []
    for kpoint_index, channels in enumerate(energies):
        entry = {
            "eigenvalues": [
                {"energies": channel}
                | ({"occupations": occupations[kpoint_index][spin]} if occupations else {})
                for spin, channel in enumerate(channels)
            ]
        }
        if weights is not None:
            entry["weight"] = weights[kpoint_index]
        values.append(entry)
    return {"name": "eigenvalues_at_kpoints", "units": "Ry", "values": values}


_FLAT_RESULTS = {
    "runtime_log": {"flavor": "self_consistent_field", "converged": True},
    "properties": {
        "records": [
            {"name": "total_energy", "units": "Ry", "value": -15.8547},
            {"name": "fermi_energy", "units": "Ry", "value": 0.123},
            _eigenvalues(
                [
                    [[-0.5, -0.3, 0.1, 0.4]],  # kpoint 0, spin 0
                    [[-0.4, -0.2, 0.2, 0.5]],  # kpoint 1, spin 0
                ]
            ),
        ],
    },
}

# A minimal synthetic NSCF+projections result: one atom, 3 orbitals (a "1S"
# shell and a "2P" shell split across its two given m components), 2
# k-points, 2 bands.  Band 1's projections deliberately sum to 0.9 (not 1.0)
# to exercise a nonzero Löwdin spilling parameter.
_PDOS_RESULTS = {
    "runtime_log": {"flavor": "non_self_consistent_field", "converged": True},
    "properties": {
        "records": [
            {"name": "fermi_energy", "units": "Ry", "value": 0.05},
            _eigenvalues(
                [
                    [[-0.5, 0.5]],  # kpoint 0, spin 0
                    [[-0.4, 0.6]],  # kpoint 1, spin 0
                ],
                occupations=[
                    [[1.0, 0.5]],
                    [[1.0, 0.5]],
                ],
                weights=[1.0, 1.0],
            ),
        ],
        "orbital_projections": {
            "values": [
                [[1.0, 0.0, 0.0], [0.0, 0.4, 0.5]],  # kpoint 0: band 0, band 1
                [[1.0, 0.0, 0.0], [0.0, 0.4, 0.5]],  # kpoint 1: band 0, band 1
            ],
            "labels": [
                {
                    "atom_index": 0,
                    "species_type_index": 0,
                    "shell_label": "1S",
                    "angular_momentum": 0,
                    "magnetic_quantum_number": 0,
                },
                {
                    "atom_index": 0,
                    "species_type_index": 0,
                    "shell_label": "2P",
                    "angular_momentum": 1,
                    "magnetic_quantum_number": -1,
                },
                {
                    "atom_index": 0,
                    "species_type_index": 0,
                    "shell_label": "2P",
                    "angular_momentum": 1,
                    "magnetic_quantum_number": 0,
                },
            ],
        },
    },
}


# ── extract_total_energy ──────────────────────────────────────────────────────


def test_extract_total_energy_reads_the_esse_record():
    """extract_total_energy reads the ESSE 'total_energy' property."""
    energy = extract_total_energy(Results.model_validate(_FLAT_RESULTS))
    assert energy == pytest.approx(-15.8547)


def test_extract_total_energy_refuses_another_unit():
    """A total energy in eV is refused rather than read as Ry."""
    in_electron_volts = {
        "properties": {"records": [{"name": "total_energy", "units": "eV", "value": -215.7}]}
    }
    with pytest.raises(ValueError):
        extract_total_energy(Results.model_validate(in_electron_volts))


def test_extract_total_energy_missing_raises():
    """extract_total_energy raises KeyError when no total_energy is listed."""
    with pytest.raises(KeyError):
        extract_total_energy(Results.model_validate({"properties": {}}))


# ── extract_band_structure ────────────────────────────────────────────────────


def test_extract_band_structure_reads_the_eigenvalues_block():
    """extract_band_structure returns the eigenvalues block as [k-point][spin][band]."""
    kpoints = [(0.0, 0.0, 0.0), (0.5, 0.0, 0.0)]
    distances = [0.0, 0.5]
    result = extract_band_structure(
        Results.model_validate(_FLAT_RESULTS), kpoints=kpoints, kpath_distances=distances
    )
    assert result["fermi_energy_ry"] == pytest.approx(0.123)
    assert len(result["eigenvalues_ry"]) == 2
    assert len(result["eigenvalues_ry"][0][0]) == 4  # 4 bands, spin 0


def test_extract_band_structure_refuses_eigenvalues_in_another_unit():
    """Eigenvalues stated in eV are refused, not read as Ry."""
    records = _FLAT_RESULTS["properties"]["records"]
    in_electronvolts = dict(
        _FLAT_RESULTS,
        properties={"records": records[:-1] + [{**records[-1], "units": "eV"}]},
    )
    with pytest.raises(ValueError, match="'eV'"):
        extract_band_structure(
            Results.model_validate(in_electronvolts), kpoints=[], kpath_distances=[]
        )


# ── extract_dos ───────────────────────────────────────────────────────────────


def test_extract_dos_produces_correct_length():
    """extract_dos returns energies and DOS of the requested length."""
    dos = extract_dos(Results.model_validate(_FLAT_RESULTS), energy_steps=100)
    assert len(dos["energies_ry"]) == 100
    assert len(dos["dos"]) == 100


def test_extract_dos_is_normalized():
    """DOS integral ≈ total number of eigenvalues (rough check)."""
    dos_result = extract_dos(
        Results.model_validate(_FLAT_RESULTS), energy_steps=2000, broadening_ry=0.005
    )
    energies = np.array(dos_result["energies_ry"])
    dos_values = np.array(dos_result["dos"])
    de = energies[1] - energies[0]
    integral = float(dos_values.sum() * de)
    # 2 kpoints × 4 bands = 8 eigenvalues → integral ≈ 8
    assert integral == pytest.approx(8.0, rel=0.01)


def test_extract_dos_weighs_each_kpoint_by_its_weight():
    """The k-point weights the eigenvalues block states scale the DOS."""
    weighted = dict(
        _FLAT_RESULTS,
        properties={
            "records": _FLAT_RESULTS["properties"]["records"][:-1]
            + [
                _eigenvalues(
                    [[[-0.5, -0.3, 0.1, 0.4]], [[-0.4, -0.2, 0.2, 0.5]]], weights=[1.5, 0.5]
                )
            ],
        },
    )
    dos_result = extract_dos(
        Results.model_validate(weighted), energy_steps=2000, broadening_ry=0.005
    )
    energies = np.array(dos_result["energies_ry"])
    integral = float(np.array(dos_result["dos"]).sum() * (energies[1] - energies[0]))
    assert integral == pytest.approx(4 * 1.5 + 4 * 0.5, rel=0.01)


# ── extract_lowdin_charges ────────────────────────────────────────────────────


def test_extract_lowdin_charges_per_orbital_values():
    """Charges match the hand-computed Σ_kn wg(n,k)·proj_i(n,k) for _PDOS_RESULTS."""
    charges = extract_lowdin_charges(Results.model_validate(_PDOS_RESULTS))
    orbital_charges = [entry["charge"] for entry in charges["orbital"]]
    assert orbital_charges == pytest.approx([2.0, 0.4, 0.5])


def test_extract_lowdin_charges_totals_and_spilling():
    """total_charge, number_of_electrons, and spilling_parameter are consistent."""
    charges = extract_lowdin_charges(Results.model_validate(_PDOS_RESULTS))
    assert charges["number_of_electrons"] == pytest.approx(3.0)
    assert charges["total_charge"] == pytest.approx(2.9)
    assert charges["spilling_parameter"] == pytest.approx(1.0 - 2.9 / 3.0)


def test_extract_lowdin_charges_atom_shell_grouping():
    """Charges for the two '2P' m-components are summed into one shell entry."""
    charges = extract_lowdin_charges(Results.model_validate(_PDOS_RESULTS))
    by_shell = {entry["shell_label"]: entry["charge"] for entry in charges["atom_shell"]}
    assert set(by_shell) == {"1S", "2P"}
    assert by_shell["1S"] == pytest.approx(2.0)
    assert by_shell["2P"] == pytest.approx(0.9)  # 0.4 + 0.5, summed over m


def test_extract_lowdin_charges_missing_projections_raises():
    """extract_lowdin_charges raises KeyError without orbital_projections."""
    with pytest.raises(KeyError):
        extract_lowdin_charges(Results.model_validate(_FLAT_RESULTS))


# ── extract_pdos ───────────────────────────────────────────────────────────────


def test_extract_pdos_produces_correct_length():
    """extract_pdos returns energies and per-orbital DOS of the requested length."""
    pdos = extract_pdos(Results.model_validate(_PDOS_RESULTS), energy_steps=100)
    assert len(pdos["energies_ry"]) == 100
    for entry in pdos["orbital_pdos"]:
        assert len(entry["dos"]) == 100


def test_extract_pdos_orbital_and_shell_counts():
    """orbital_pdos has one entry per orbital; atom_shell_pdos is grouped by shell."""
    pdos = extract_pdos(Results.model_validate(_PDOS_RESULTS), energy_steps=50)
    assert len(pdos["orbital_pdos"]) == 3
    assert len(pdos["atom_shell_pdos"]) == 2


def test_extract_pdos_summed_pdos_equals_sum_of_orbitals():
    """summed_pdos is the elementwise sum of every orbital's PDOS curve."""
    pdos = extract_pdos(Results.model_validate(_PDOS_RESULTS), energy_steps=200)
    expected = np.sum([entry["dos"] for entry in pdos["orbital_pdos"]], axis=0)
    assert pdos["summed_pdos"] == pytest.approx(expected.tolist())


def test_extract_pdos_atom_shell_equals_grouped_sum():
    """The '2P' atom_shell curve equals the sum of its two m-component curves."""
    pdos = extract_pdos(Results.model_validate(_PDOS_RESULTS), energy_steps=200)
    p_orbitals = [entry["dos"] for entry in pdos["orbital_pdos"] if entry["shell_label"] == "2P"]
    expected = np.sum(p_orbitals, axis=0)
    (p_shell,) = [entry for entry in pdos["atom_shell_pdos"] if entry["shell_label"] == "2P"]
    assert p_shell["dos"] == pytest.approx(expected.tolist())


def test_extract_pdos_fermi_energy_defaults_from_results():
    """Without an override, fermi_energy_ry comes from the results' fermi_energy record."""
    pdos = extract_pdos(Results.model_validate(_PDOS_RESULTS), energy_steps=10)
    assert pdos["fermi_energy_ry"] == pytest.approx(0.05)


def test_extract_pdos_fermi_energy_override():
    """An explicit fermi_energy_ry overrides the value in results (e.g. SCF's)."""
    pdos = extract_pdos(
        Results.model_validate(_PDOS_RESULTS), energy_steps=10, fermi_energy_ry=0.99
    )
    assert pdos["fermi_energy_ry"] == pytest.approx(0.99)


def test_extract_pdos_missing_projections_raises():
    """extract_pdos raises KeyError without orbital_projections."""
    with pytest.raises(KeyError):
        extract_pdos(Results.model_validate(_FLAT_RESULTS))


# ── Derivatives ───────────────────────────────────────────────────────────────

_FORCE_RESULTS = {
    "runtime_log": {"flavor": "self_consistent_field", "converged": True},
    "properties": {
        "records": [
            {"name": "total_energy", "units": "Ry", "value": -15.8547},
            {
                "name": "atomic_forces",
                "units": "Ry/a.u.",
                "values": [
                    {"id": 2, "value": [-0.001, 0.002, -0.003]},
                    {"id": 1, "value": [0.001, -0.002, 0.003]},
                ],
            },
            {
                "name": "stress_tensor",
                "units": "kbar",
                "value": [[10.0, 0.5, 0.0], [0.5, 10.0, 0.0], [0.0, 0.0, 9.5]],
            },
        ]
    },
}


def test_extract_forces_returns_native_units_per_atom():
    """Forces come back as one triple per atom, in id order, in Q3's own Ry/Bohr."""
    forces = extract_forces(Results.model_validate(_FORCE_RESULTS))

    assert len(forces) == 2
    assert forces[0] == pytest.approx([0.001, -0.002, 0.003])
    assert forces[1] == pytest.approx([-0.001, 0.002, -0.003])


def test_extract_stress_returns_three_by_three_kbar():
    """Stress comes back as a 3x3 tensor in kbar, in Q3's own sign convention."""
    stress = extract_stress(Results.model_validate(_FORCE_RESULTS))

    assert len(stress) == 3
    assert all(len(row) == 3 for row in stress)
    assert stress[0][0] == pytest.approx(10.0)
    assert stress[2][2] == pytest.approx(9.5)


def test_extract_forces_and_stress_are_empty_when_not_computed():
    """A single-point energy run is a complete result, not an error."""
    assert extract_forces(Results.model_validate(_FLAT_RESULTS)) == []
    assert extract_stress(Results.model_validate(_FLAT_RESULTS)) == []


def _incomplete(**reasons):
    """``_FORCE_RESULTS`` with the derivatives named in *reasons* flagged incomplete."""
    runtime_log = {**_FORCE_RESULTS["runtime_log"], "incomplete": reasons}
    return {**_FORCE_RESULTS, "runtime_log": runtime_log}


def test_extract_forces_refuses_an_incomplete_derivative():
    """A force Q3 has disclaimed must not be handed back as if it were whole."""
    incomplete = _incomplete(forces="augmentation term omitted")

    with pytest.raises(IncompleteDerivativeError) as failure:
        extract_forces(Results.model_validate(incomplete))
    assert "augmentation term omitted" in str(failure.value)


def test_extract_stress_refuses_an_incomplete_derivative():
    """The same guard applies to the stress."""
    incomplete = _incomplete(stress="band groups span several ranks")

    with pytest.raises(IncompleteDerivativeError):
        extract_stress(Results.model_validate(incomplete))


def test_incomplete_forces_do_not_block_reading_the_stress():
    """Only the quantity actually asked for is checked."""
    incomplete = _incomplete(forces="augmentation term omitted")

    assert extract_stress(Results.model_validate(incomplete))[0][0] == pytest.approx(10.0)
    assert extract_total_energy(Results.model_validate(incomplete)) == pytest.approx(-15.8547)


def test_raise_if_derivative_incomplete_passes_a_clean_result():
    """Nothing is raised when Q3 has not flagged anything."""
    raise_if_derivative_incomplete(Results.model_validate(_FORCE_RESULTS), ("forces", "stress"))


# ── Convergence ───────────────────────────────────────────────────────────────

# What a run that ran out of iterations writes: the count it performed beside
# the limit its provenance recorded. The limit sits in the provenance because it
# is what the run was asked to do; the count is under runtime_log because it is
# what the run did.
_NOT_CONVERGED_RESULTS = {
    "runtime_log": {
        "flavor": "self_consistent_field",
        "converged": False,
        "self_consistent_field_iterations": 100,
    },
    "provenance": {"precision": {"maximum_self_consistent_field_iterations": 100}},
    "properties": {"records": [{"name": "total_energy", "units": "Ry", "value": -486.6}]},
}


def test_raise_if_not_converged_passes_a_converged_result():
    converged = {**_NOT_CONVERGED_RESULTS["runtime_log"], "converged": True}
    raise_if_not_converged(
        Results.model_validate(dict(_NOT_CONVERGED_RESULTS, runtime_log=converged))
    )


def test_raise_if_not_converged_accepts_a_result_that_does_not_say():
    """A result without the key was accepted before, and still is."""
    raise_if_not_converged(
        Results.model_validate(
            {"properties": {"records": [{"name": "total_energy", "units": "Ry", "value": -15.8}]}}
        )
    )


def test_raise_if_not_converged_reports_the_iterations_the_run_performed():
    with pytest.raises(SCFNotConvergedError) as failure:
        raise_if_not_converged(Results.model_validate(_NOT_CONVERGED_RESULTS), returncode=0)

    assert str(failure.value) == "SCF did not converge after 100 iterations; its limit was 100."
    assert failure.value.iterations == 100
    assert failure.value.maximum_iterations == 100
    assert failure.value.returncode == 0


def test_raise_if_not_converged_falls_back_to_the_limit_for_an_older_binary():
    """A binary that predates the count must not be reported as stopping at 0.

    Before the engine wrote ``self_consistent_field_iterations``, every failed
    run was reported "after 0 iterations" -- a copper atom that had run all 100.
    The limit is the one number such a result does carry, so the message states
    that and claims no count at all.
    """
    runtime_log = dict(_NOT_CONVERGED_RESULTS["runtime_log"])
    del runtime_log["self_consistent_field_iterations"]
    older_binary_results = dict(_NOT_CONVERGED_RESULTS, runtime_log=runtime_log)

    with pytest.raises(SCFNotConvergedError) as failure:
        raise_if_not_converged(Results.model_validate(older_binary_results))

    assert str(failure.value) == "SCF did not converge within its 100-iteration limit."
    assert failure.value.iterations is None
    assert failure.value.maximum_iterations == 100


def test_raise_if_not_converged_reports_a_count_without_provenance():
    without_provenance = dict(_NOT_CONVERGED_RESULTS)
    del without_provenance["provenance"]

    with pytest.raises(SCFNotConvergedError) as failure:
        raise_if_not_converged(Results.model_validate(without_provenance))

    assert str(failure.value) == "SCF did not converge after 100 iterations."
    assert failure.value.maximum_iterations is None


def test_raise_if_not_converged_invents_no_number_when_the_result_has_none():
    with pytest.raises(SCFNotConvergedError) as failure:
        raise_if_not_converged(Results.model_validate({"runtime_log": {"converged": False}}))

    assert str(failure.value) == "SCF did not converge."
    assert failure.value.iterations is None
    assert failure.value.maximum_iterations is None
