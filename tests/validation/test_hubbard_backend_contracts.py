"""Executable refusal contracts for DFT+U backends that lack parity coverage."""

from helpers import EXAMPLES_DIRECTORY, apply_config_overlay, run_q3

SILICON_INPUT = EXAMPLES_DIRECTORY / "total_energy--semiconductor--silicon" / "nc.yml"
HUBBARD_OVERLAY = {
    "method": {
        "scf": {
            "diagonalization": "davidson",
            "hubbard": [{"species": "Si", "orbital_label": "p", "u_ev": 3.0}],
        }
    }
}
DIRECT_DIAGONALIZER_OVERLAY = {
    "method": {
        "scf": {
            "diagonalization": "direct",
            "hubbard": [{"species": "Si", "orbital_label": "p", "u_ev": 3.0}],
        }
    }
}


def test_gpu_refuses_hubbard_before_backend_initialization(q3_gpu_binary, tmp_path):
    """GPU DFT+U must fail before allocating or validating a device backend."""
    input_path = apply_config_overlay(SILICON_INPUT, HUBBARD_OVERLAY, tmp_path)
    completed = run_q3([q3_gpu_binary], input_path, working_directory=tmp_path)
    output = completed.stdout + completed.stderr

    assert completed.returncode != 0
    assert "DFT+U (hubbard) is not yet supported on GPU. Run on CPU instead." in output
    assert "Initializing GPU backends" not in output
    assert "Initializing Metal backends" not in output
    assert "Initializing CUDA backends" not in output
    assert "Initializing HIP backends" not in output
    assert "Initializing SYCL backends" not in output


def test_direct_diagonalizer_refuses_hubbard_before_scf(q3_command, tmp_path):
    """The direct diagonalizer must not silently omit the Hubbard Hamiltonian."""
    input_path = apply_config_overlay(SILICON_INPUT, DIRECT_DIAGONALIZER_OVERLAY, tmp_path)
    completed = run_q3(q3_command, input_path, working_directory=tmp_path)
    output = completed.stdout + completed.stderr

    assert completed.returncode != 0
    assert "DFT+U (hubbard) requires diagonalization: davidson" in output
    assert "SCF Iteration" not in output
