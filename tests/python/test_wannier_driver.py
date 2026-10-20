"""tests/python/test_wannier_driver.py — the protocol's result, and the chain's derivations.

The workflow tests run the protocol end to end through stub binaries; these
pin what the driver hands to others: the result's keys in the order every
caller reads them, and the mesh and k-path derivations both workflows now
take from ``EngineChain`` rather than each computing on its own.
"""

import pytest
import yaml
from q3.io.interpolation_results import InterpolationResults, LadderAttempt
from q3.workflows.band_distance import BandDistance
from q3.workflows.interpolation import EngineChain
from q3.workflows.wannier import WorkflowResult, off_mesh_reference_kpoints

#: The keys the workflow has always returned, in the order it documents them.
DOCUMENTED_KEYS = (
    "status",
    "reason",
    "grid",
    "number_of_wannier_functions",
    "trial_shells",
    "number_of_bands",
    "attempts",
    "band_distance",
    "quality",
    "interpolation_results",
    "reference_results",
    "reference_kpoints",
    "scf_results",
    "nscf_results",
    "fermi_energy_ry",
)

SILICON = {
    "name": "Si",
    "lattice": {
        "type": "FCC",
        "a": 3.866976,
        "b": 3.866975,
        "c": 3.866975,
        "alpha": 60,
        "beta": 60,
        "gamma": 60,
        "units": {"length": "angstrom", "angle": "degree"},
    },
    "basis": {
        "units": "crystal",
        "elements": [{"id": 0, "value": "Si"}, {"id": 1, "value": "Si"}],
        "coordinates": [
            {"id": 0, "value": [0.875, 0.875, 0.875]},
            {"id": 1, "value": [0.125, 0.125, 0.125]},
        ],
    },
}

#: The same cell as vectors alone: no type, so no high-symmetry table to name.
SILICON_AS_VECTORS = {
    **SILICON,
    "lattice": {
        "vectors": {
            "a": [3.866976, 0.0, 0.0],
            "b": [1.933488, 3.348897, 0.0],
            "c": [1.933488, 1.116299, 3.157321],
            "units": "angstrom",
        }
    },
}


def _chain(directory, material) -> EngineChain:
    directory.mkdir(parents=True, exist_ok=True)
    manifest = {
        "material": material,
        "model": {"type": "dft", "subtype": "gga", "functional": "pbe"},
        "method": {
            "type": "pseudopotential",
            "subtype": "norm-conserving",
            "cutoffs": {"wavefunction": 48.0, "density": 192.0},
            "pseudopotentials": {"Si": "Si.upf"},
            "scf": {"number_of_bands": 4},
        },
        "control": {"calculation": "scf", "outdir": "./output", "prefix": "si"},
    }
    path = directory / "nc_scf.yml"
    path.write_text(yaml.safe_dump(manifest, sort_keys=False))
    # The chain resolves its engine binary when it is built; nothing here runs it.
    binary = directory / "q3"
    binary.write_text("#!/bin/sh\nexit 0\n")
    binary.chmod(0o755)
    return EngineChain(path, q3_binary=binary, q3_utility_binary=binary, capture_output=True)


def test_the_result_keeps_the_documented_keys_in_their_order():
    attempt = LadderAttempt(
        rung="default",
        grid=[4, 4, 4],
        settings={"localization": "auto"},
        status="error:unknown",
        reason="Error: cannot open file",
    )
    result = WorkflowResult(
        status="reject",
        reason="Error: cannot open file",
        grid=(4, 4, 4),
        number_of_wannier_functions=8,
        trial_shells={"Si": [0, 1]},
        number_of_bands=16,
        attempts=(attempt,),
        band_distance=None,
        interpolation_results=None,
        reference_results=None,
        reference_kpoints=[(0.125, 0.125, 0.125)],
        scf_results=None,
        nscf_results=None,
        fermi_energy_ry=0.0,
    ).as_dict()
    assert tuple(result) == DOCUMENTED_KEYS
    assert result["grid"] == [4, 4, 4] and result["reference_kpoints"] == [[0.125, 0.125, 0.125]]
    # A raised attempt keeps the nulls of what it could not measure.
    assert result["attempts"][0]["eta_ev"] is None
    assert result["quality"] is None and result["band_distance"] is None


def test_the_results_quality_is_the_results_files_own_block():
    document = InterpolationResults.model_validate(
        {
            "calculation": "interpolation",
            "interpolation": {
                "quality": {"status": "accept", "hamiltonian_decay": {"ratio": 0.01}}
            },
        }
    )
    result = WorkflowResult(
        status="accept",
        reason="eta = 1.00 meV",
        grid=(4, 4, 4),
        number_of_wannier_functions=8,
        trial_shells={"Si": [0, 1]},
        number_of_bands=16,
        attempts=(),
        band_distance=BandDistance(eta=0.001, eta_maximum=0.004, number_of_states=512),
        interpolation_results=document,
        reference_results=None,
        reference_kpoints=[],
        scf_results=None,
        nscf_results=None,
        fermi_energy_ry=0.4,
    ).as_dict()
    assert result["quality"] is result["interpolation_results"]["interpolation"]["quality"]
    assert result["quality"] == {"status": "accept", "hamiltonian_decay": {"ratio": 0.01}}
    assert result["band_distance"] == {
        "eta_ev": 0.001,
        "eta_maximum_ev": 0.004,
        "number_of_states": 512,
    }


def test_the_chain_derives_the_mesh_from_the_material_or_validates_the_grid(tmp_path):
    with _chain(tmp_path, SILICON) as chain:
        assert chain.mesh((4, 4, 4), 0.2) == (4, 4, 4)
        rule = chain.mesh(None, 0.2)
        assert len(rule) == 3 and all(isinstance(n, int) and n >= 4 for n in rule)
        with pytest.raises(ValueError, match="three positive integers"):
            chain.mesh((4, 0, 4), 0.2)
    assert not chain.effective_scf_path.exists(), "leaving the chain removes its manifest"


def test_the_chain_reads_a_path_with_the_materials_lattice_type_or_refuses(tmp_path):
    chain = _chain(tmp_path / "typed", SILICON)
    try:
        assert chain.lattice_type() == "FCC"
        assert chain.lattice_type("BCC") == "BCC"
        kpoints, distances = chain.kpath(["G", "X"], None, 4)
        assert len(kpoints) == len(distances) > 2
        assert all(len(point) == 3 for point in kpoints)
    finally:
        chain.close()

    chain = _chain(tmp_path / "vectors", SILICON_AS_VECTORS)
    try:
        with pytest.raises(ValueError, match="lattice_type"):
            chain.kpath(["G", "X"], None, 4)
    finally:
        chain.close()


def test_off_mesh_reference_points_sit_half_a_step_off_the_mesh():
    points = off_mesh_reference_kpoints((4, 4, 4))
    assert len(points) == 64
    assert points[0] == (0.125, 0.125, 0.125)
