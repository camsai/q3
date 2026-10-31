"""
Tests for QE-format binary output files (wavefunction and charge density).

Reads charge-density.dat and wfcN.dat in Fortran unformatted format,
validates structure and cross-checks against JSON results.

Uses config overlays on the canonical examples to avoid duplicating
input files. All outputs go to temporary directories.
"""

import numpy as np
import pytest
from binary_readers import read_charge_density, read_wavefunction
from q3.io.results.classes import Results

from helpers import (
    EXAMPLES_DIRECTORY,
    apply_config_overlay,
    generate_qe_input,
    requires_qe,
    run_q3,
    run_qe_full,
)

# ── Configuration ────────────────────────────────────────────────────

BASE_EXAMPLE = EXAMPLES_DIRECTORY / "total_energy--semiconductor--silicon" / "nc.yml"

GAMMA_OVERLAY = {
    "model": {"subtype": "lda", "functional": "pz"},
    "method": {
        "cutoffs": {"wavefunction": 16.0},
        "pseudopotentials": {"Si": "../data/pseudopotentials/nc/Si.pz-vbc.UPF"},
        "scf": {
            "convergence_threshold": 1.0e-6,
            "mixing": {"beta": 0.7},
            "kgrid": {"dimensions": [1, 1, 1], "shifts": [0, 0, 0]},
        },
    },
    "control": {"prefix": "si_gamma"},
}

NON_GAMMA_OVERLAY = {
    "model": {"subtype": "lda", "functional": "pz"},
    "method": {
        "cutoffs": {"wavefunction": 16.0},
        "pseudopotentials": {"Si": "../data/pseudopotentials/nc/Si.pz-vbc.UPF"},
        "scf": {
            "convergence_threshold": 1.0e-6,
            "mixing": {"beta": 0.7},
            "kgrid": {"dimensions": [2, 2, 2], "shifts": [1, 1, 1]},
        },
    },
    "control": {"prefix": "si_shifted"},
}


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def gamma_binary_output(q3_command, tmp_path_factory):
    """Run gamma-only SCF and return paths to binary output files."""
    temporary_directory = tmp_path_factory.mktemp("binary_gamma")
    merged_yaml = apply_config_overlay(
        BASE_EXAMPLE,
        GAMMA_OVERLAY,
        temporary_directory,
    )
    result = run_q3(
        q3_command,
        merged_yaml,
        working_directory=temporary_directory,
    )
    assert result.returncode == 0, f"q3 failed:\n{result.stderr}\n{result.stdout}"

    output_directory = temporary_directory / "output"
    save_directory = output_directory / "si_gamma.save"
    results_json_path = output_directory / "si_gamma_results.json"

    return {
        "save_directory": save_directory,
        "results_json_path": results_json_path,
        "charge_density_path": save_directory / "charge-density.dat",
        "wavefunction_path": save_directory / "wfc1.dat",
    }


@pytest.fixture(scope="session")
def charge_density(gamma_binary_output):
    """Parse charge-density.dat once and share across tests."""
    return read_charge_density(gamma_binary_output["charge_density_path"])


@pytest.fixture(scope="session")
def wavefunction(gamma_binary_output):
    """Parse wfc1.dat once and share across tests."""
    return read_wavefunction(gamma_binary_output["wavefunction_path"])


@pytest.fixture(scope="session")
def results_json(gamma_binary_output):
    """Load JSON results once and share across tests."""
    return Results.read(gamma_binary_output["results_json_path"])


@pytest.fixture(scope="session")
def non_gamma_wavefunction(q3_command, tmp_path_factory):
    """Parse a shifted-k checkpoint whose nonzero header fields expose misalignment."""
    temporary_directory = tmp_path_factory.mktemp("binary_non_gamma")
    merged_yaml = apply_config_overlay(
        BASE_EXAMPLE,
        NON_GAMMA_OVERLAY,
        temporary_directory,
    )
    result = run_q3(q3_command, merged_yaml, working_directory=temporary_directory)
    assert result.returncode == 0, f"q3 failed:\n{result.stderr}\n{result.stdout}"
    return read_wavefunction(temporary_directory / "output" / "si_shifted.save" / "wfc1.dat")


# ── Charge density tests ────────────────────────────────────────────


def test_charge_density_readable(charge_density):
    """charge-density.dat has valid Fortran unformatted record structure."""
    assert charge_density["number_of_g_vectors"] > 0
    assert charge_density["number_of_spin_components"] == 1
    assert charge_density["density_g_space"].shape == (charge_density["number_of_g_vectors"],)


def test_charge_density_g_zero_first(charge_density):
    """G=0 should be the first G-vector (QE convention)."""
    assert np.all(charge_density["miller_indices"][0] == [0, 0, 0]), (
        f"First G-vector is {charge_density['miller_indices'][0]}, expected [0, 0, 0]"
    )


def test_charge_density_positive_at_g_zero(charge_density):
    """ρ(G=0) should be positive (integral of charge density)."""
    density_at_g_zero = charge_density["density_g_space"][0].real
    assert density_at_g_zero > 0, f"ρ(G=0) = {density_at_g_zero}, expected positive"


def test_charge_density_inversion_symmetry(charge_density):
    """Miller indices should use a consistent inversion symmetry convention.

    Full-sphere storage: both G and -G are present (pair_fraction > 0.9).
    Half-sphere (gamma-only): only G >= 0 stored (pair_fraction < 0.1).

    Note: the charge density writer currently hardcodes gamma_only=0 in the
    binary header regardless of actual calculation type, so we cannot rely on
    that flag. Instead, we verify that the storage is self-consistent — either
    nearly all G-vectors have inversion partners, or nearly none do.
    """
    miller_set = set(map(tuple, charge_density["miller_indices"]))
    paired_count = sum(
        1 for miller in charge_density["miller_indices"] if tuple(-miller) in miller_set
    )
    pair_fraction = paired_count / charge_density["number_of_g_vectors"]

    assert pair_fraction > 0.9 or pair_fraction < 0.1, (
        f"Inconsistent inversion symmetry: {pair_fraction:.0%} of G-vectors have "
        f"inversion partners (expected >90% for full-sphere or <10% for half-sphere)"
    )


# ── Wavefunction tests ──────────────────────────────────────────────


def test_wavefunction_readable(wavefunction):
    """wfc1.dat has valid Fortran unformatted record structure."""
    assert wavefunction["number_of_bands"] > 0
    assert wavefunction["number_of_plane_waves"] > 0
    assert wavefunction["coefficients"].shape == (
        wavefunction["number_of_bands"],
        wavefunction["max_plane_waves"],
    )


def test_wavefunction_kpoint_is_gamma(wavefunction):
    """k-point should be (0, 0, 0) for gamma-only calculation."""
    assert wavefunction["kpoint_index"] == 1
    assert np.allclose(
        wavefunction["kpoint_coordinates"],
        [0.0, 0.0, 0.0],
        atol=1e-10,
    )
    assert wavefunction["spin_index"] == 1
    assert wavefunction["gamma_only"] == 1
    assert wavefunction["scale_factor"] == pytest.approx(1.0)


def test_non_gamma_wavefunction_header_uses_packed_offsets(non_gamma_wavefunction):
    """A shifted k-point discriminates every field in QE's packed 44-byte header."""
    assert non_gamma_wavefunction["kpoint_index"] == 1
    assert np.linalg.norm(non_gamma_wavefunction["kpoint_coordinates"]) > 0.1
    assert non_gamma_wavefunction["spin_index"] == 1
    assert non_gamma_wavefunction["gamma_only"] == 0
    assert non_gamma_wavefunction["scale_factor"] == pytest.approx(1.0)


def test_wavefunction_band_count_matches_json(wavefunction, results_json):
    """Number of bands in binary should match JSON eigenvalue count."""
    number_of_eigenvalues = len(
        results_json.properties.eigenvalues_at_kpoints.values[0].eigenvalues[0].energies
    )
    assert wavefunction["number_of_bands"] == number_of_eigenvalues, (
        f"Binary has {wavefunction['number_of_bands']} bands, "
        f"JSON has {number_of_eigenvalues} eigenvalues"
    )


def test_wavefunction_normalization(wavefunction):
    """Each wavefunction should be approximately normalized.

    For gamma-only, only the G >= 0 half-sphere is stored, so the norm is:
        ||ψ||² = 2 Σ_{G>0} |c_G|² + |c_0|²
    which equals 2 * Σ_all |c_G|² - |c_0|² (since c_0 is counted once).
    For non-gamma, the norm is simply Σ|c_G|².
    """
    is_gamma_only = wavefunction["gamma_only"]
    for band_index in range(wavefunction["number_of_bands"]):
        coefficients = wavefunction["coefficients"][band_index]
        if is_gamma_only:
            norm_squared = 2.0 * np.sum(np.abs(coefficients) ** 2) - np.abs(coefficients[0]) ** 2
        else:
            norm_squared = np.sum(np.abs(coefficients) ** 2)
        assert 0.5 < norm_squared < 2.0, (
            f"Band {band_index} norm² = {norm_squared:.4f}, expected ~1.0"
        )


def test_wavefunction_orthogonality(wavefunction):
    """Wavefunctions should be approximately orthogonal: <ψ_i|ψ_j> ≈ δ_ij.

    For gamma-only, only the G >= 0 half-sphere is stored, so the inner product is:
        <ψ_i|ψ_j> = 2 Re(Σ_{G>0} c_i*(G) c_j(G)) + c_i*(0) c_j(0)
    which equals 2 Re(c_i† c_j) - Re(c_i*(0) c_j(0)).
    For non-gamma, the inner product is simply c_i† c_j.
    """
    is_gamma_only = wavefunction["gamma_only"]
    coefficients = wavefunction["coefficients"]

    if is_gamma_only:
        # <ψ_i|ψ_j> = 2 Re(Σ c_i* c_j) - Re(c_i*(0) c_j(0))
        naive_overlap = coefficients @ coefficients.conj().T
        g_zero_outer = np.outer(coefficients[:, 0].conj(), coefficients[:, 0])
        overlap_matrix = 2.0 * naive_overlap.real - g_zero_outer.real
    else:
        overlap_matrix = coefficients @ coefficients.conj().T

    for index_i in range(wavefunction["number_of_bands"]):
        for index_j in range(wavefunction["number_of_bands"]):
            expected = 1.0 if index_i == index_j else 0.0
            assert abs(overlap_matrix[index_i, index_j] - expected) < 0.01, (
                f"S[{index_i},{index_j}] = "
                f"{overlap_matrix[index_i, index_j]:.4f}, "
                f"expected {expected}"
            )


def test_wavefunction_reciprocal_vectors_nonzero(wavefunction):
    """Reciprocal lattice vectors should have non-zero norm."""
    for vector_index in range(3):
        norm = np.linalg.norm(wavefunction["reciprocal_vectors"][vector_index])
        assert norm > 0.1, f"Reciprocal vector b{vector_index + 1} has norm {norm:.4f}"


def test_non_gamma_kpoint_is_cartesian_inverse_bohr(non_gamma_wavefunction):
    """xk is Cartesian in 1/Bohr, as QE writes it: in the basis of the stored
    reciprocal vectors it must land on the 2x2x2 shifted grid (multiples of 1/4)."""
    reciprocal_vectors = non_gamma_wavefunction["reciprocal_vectors"]
    fractional = np.linalg.solve(
        reciprocal_vectors.T, np.asarray(non_gamma_wavefunction["kpoint_coordinates"])
    )
    assert np.allclose(fractional * 4.0, np.round(fractional * 4.0), atol=1e-8), fractional


def test_wavefunction_miller_indices_lie_inside_cutoff(non_gamma_wavefunction):
    """Read as QE's interleaved mill(3, igwx), every |k+G|^2 is inside ecutwfc.

    Read with the transposed (all-h, all-k, all-l) layout the triplets would be
    scrambled and most would fall far outside the sphere, so this pins the order.
    """
    wavefunction = non_gamma_wavefunction
    wavevectors = (
        wavefunction["miller_indices"] @ wavefunction["reciprocal_vectors"]
        + np.asarray(wavefunction["kpoint_coordinates"])[None, :]
    )
    kinetic_energy_ry = np.sum(wavevectors**2, axis=1)
    ecutwfc_ry = NON_GAMMA_OVERLAY["method"]["cutoffs"]["wavefunction"]
    assert kinetic_energy_ry.max() <= ecutwfc_ry * (1.0 + 1e-8)


# ── Interoperability with files Quantum ESPRESSO wrote ───────────────

QE_INTEROP_CUTOFF_RY = 16.0
QE_INTEROP_OVERLAY = {
    "method": {
        "cutoffs": {"wavefunction": QE_INTEROP_CUTOFF_RY, "density": 4 * QE_INTEROP_CUTOFF_RY},
        "scf": {"convergence_threshold": 1.0e-6},
    },
    "control": {"prefix": "si_interop"},
}


@pytest.fixture(scope="module")
def quantum_espresso_save(tmp_path_factory):
    """Run pw.x on the silicon example and return its .save directory."""
    temporary_directory = tmp_path_factory.mktemp("qe_interop")
    input_path = generate_qe_input(
        case_name="si_interop",
        material_name="silicon",
        pp_type="nc",
        pp_filename="Si.upf",
        ecutwfc=QE_INTEROP_CUTOFF_RY,
        ecutrho=4 * QE_INTEROP_CUTOFF_RY,
        kgrid=[2, 2, 2],
        kshift=[1, 1, 1],
        convergence_threshold=1.0e-6,
        output_directory=str(temporary_directory),
    )
    run_qe_full(input_path)
    save_directory = temporary_directory / "si_interop" / "si_interop.save"
    assert (save_directory / "charge-density.dat").is_file(), "pw.x wrote no charge-density.dat"
    return save_directory


@pytest.fixture(scope="module")
def q3_interop_save(q3_command, tmp_path_factory):
    """Run Q3 on the same silicon cell, cutoffs and k-grid as the pw.x fixture."""
    temporary_directory = tmp_path_factory.mktemp("q3_interop")
    merged_yaml = apply_config_overlay(BASE_EXAMPLE, QE_INTEROP_OVERLAY, temporary_directory)
    result = run_q3(q3_command, merged_yaml, working_directory=temporary_directory)
    assert result.returncode == 0, f"q3 failed:\n{result.stderr}\n{result.stdout}"
    return temporary_directory / "output" / "si_interop.save"


@requires_qe
def test_quantum_espresso_wavefunction_parses_with_shared_reader(quantum_espresso_save):
    """QE's own wfc1.dat parses with the reader Q3's files are checked against."""
    wavefunction = read_wavefunction(quantum_espresso_save / "wfc1.dat")
    wavevectors = (
        wavefunction["miller_indices"] @ wavefunction["reciprocal_vectors"]
        + np.asarray(wavefunction["kpoint_coordinates"])[None, :]
    )
    assert np.sum(wavevectors**2, axis=1).max() <= QE_INTEROP_CUTOFF_RY * (1.0 + 1e-8)
    assert wavefunction["kpoint_index"] == 1
    assert wavefunction["spin_index"] == 1
    assert wavefunction["scale_factor"] == pytest.approx(1.0)


@requires_qe
def test_charge_density_matches_quantum_espresso_layout(quantum_espresso_save, q3_interop_save):
    """Q3 and QE agree on reciprocal vectors, Miller convention and the G sphere."""
    reference = read_charge_density(quantum_espresso_save / "charge-density.dat")
    candidate = read_charge_density(q3_interop_save / "charge-density.dat")
    assert np.allclose(candidate["reciprocal_vectors"], reference["reciprocal_vectors"], atol=1e-8)
    assert set(map(tuple, candidate["miller_indices"])) == set(
        map(tuple, reference["miller_indices"])
    )
    assert candidate["density_g_space"][0].real == pytest.approx(
        reference["density_g_space"][0].real, rel=1e-6
    )
