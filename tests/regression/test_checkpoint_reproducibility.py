"""Two runs of one binary on one input must leave byte-identical checkpoints.

A checkpoint is supposed to be a function of the calculation. When it is not,
two things break at once: a restart is not the run it claims to continue, and
no comparison of one run's ``.save/`` tree against another's can mean anything,
because the noise it reports has nothing to do with the change under test.

The defect this guards against was exactly that. ``write_kpoint_header`` filled
a local struct field by field and wrote it whole, and the struct was not whole:
the 8-byte-aligned double after its leading ``int32_t`` left four bytes of
padding that nothing assigned. Every ``wfc*.dat`` carried four bytes of
whatever the stack last held, so two runs of the same binary on the same input
differed in exactly those four bytes -- fourteen of nineteen cases in the
parity matrix, at file offset 8, every time.

This test asserts the property rather than the fix, so it also covers the next
record that grows a padding hole.
"""

import filecmp
import struct

from helpers import EXAMPLES_DIRECTORY, apply_config_overlay, run_q3

BASE_EXAMPLE = EXAMPLES_DIRECTORY / "total_energy--semiconductor--silicon" / "nc.yml"

# Small and fast, and deliberately *not* Gamma-only: the k-point header is the
# record that carried the uninitialized padding, and a run with several
# k-points writes several of them.
REPRODUCIBILITY_OVERLAY = {
    "model": {"subtype": "lda", "functional": "pz"},
    "method": {
        "cutoffs": {"wavefunction": 16.0},
        "pseudopotentials": {"Si": "../data/pseudopotentials/nc/Si.pz-vbc.UPF"},
        "scf": {
            "convergence_threshold": 1.0e-6,
            "kgrid": {"dimensions": [2, 2, 2], "shifts": [1, 1, 1]},
        },
    },
    # FFTW_MEASURE picks its plan by timing several, so the last bits of every
    # transform depend on what else the machine was doing. Deterministic
    # planning is what makes byte identity a statement about the code.
    "control": {"prefix": "si_repro", "use_deterministic_fft_planning": True},
}


def _run_once(q3_command, directory):
    """Run the shared input once into its own directory, returning the save tree."""
    input_path = apply_config_overlay(BASE_EXAMPLE, REPRODUCIBILITY_OVERLAY, directory)
    completed = run_q3(q3_command, input_path, working_directory=directory)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    save_directory = directory / "output" / "si_repro.save"
    assert save_directory.is_dir(), f"no save tree at {save_directory}"
    return save_directory


def test_two_runs_write_byte_identical_checkpoints(q3_command, tmp_path):
    first = _run_once(q3_command, tmp_path / "first")
    second = _run_once(q3_command, tmp_path / "second")

    first_names = sorted(path.name for path in first.iterdir() if path.is_file())
    second_names = sorted(path.name for path in second.iterdir() if path.is_file())
    assert first_names == second_names, "the two runs wrote different checkpoint files"
    assert first_names, "the run wrote no checkpoint files at all"

    # shallow=False forces a content comparison; the default compares stat()
    # signatures, and two files written seconds apart differ there anyway.
    matched, mismatched, errors = filecmp.cmpfiles(first, second, first_names, shallow=False)
    assert not errors, f"could not compare: {errors}"
    assert not mismatched, (
        f"{len(mismatched)} checkpoint file(s) differ between two runs of the same "
        f"binary on the same input: {mismatched}. A checkpoint must be a function "
        f"of the calculation."
    )
    assert sorted(matched) == first_names


def test_the_wavefunction_header_has_no_padding(q3_command, tmp_path):
    """The specific regression: the k-point header used to carry a padding word.

    It was a C struct dumped whole, so four bytes after the k-point index came
    from the stack. The header is now written field by field in Quantum
    ESPRESSO's packed 44-byte layout, so there is no padding left to leak, and
    checking the record length is what keeps it that way.
    """
    save_directory = _run_once(q3_command, tmp_path / "only")
    wavefunction_files = sorted(save_directory.glob("wfc*.dat"))
    assert wavefunction_files, "the run wrote no wavefunction checkpoints"

    for path in wavefunction_files:
        data = path.read_bytes()
        assert len(data) >= 52, f"{path.name} is shorter than its own header"
        (leading,) = struct.unpack("<i", data[0:4])
        (trailing,) = struct.unpack("<i", data[48:52])
        assert leading == 44 and trailing == 44, (
            f"{path.name} has a {leading}-byte k-point header; the packed layout "
            f"is 44 bytes, and anything longer carries padding"
        )
