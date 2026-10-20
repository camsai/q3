"""
MPI topology guard for the mean-field export
(`export_and_record_mean_field_under_topology`, io/mean_field_export/record_and_export.hpp).

Under `q3_mpi`, every multi-rank topology distributes the mean-field state:
k-point pools each hold only their own k-points, and band groups hold only a
G-vector slab of each band. The exporter has no gather step, so no rank can
write a correct file. Before the guard existed, both multi-rank shapes were
measured writing WRONG output while exiting 0:

  - 2 ranks, 1 pool (band groups): WFN with corrupted Fortran record framing
    -- both ranks interleaved slab-partial coefficient records (809 of 1546
    plane waves) onto the same file.
  - 2 ranks, 2 pools: a WELL-FORMED WFN containing 1 of 2 k-points with its
    weight renormalized to 1.0 -- undetectable by any BerkeleyGW consumer.

This file pins the guard from both sides: multi-rank runs must refuse loudly
and write nothing, and a single-rank `q3_mpi` run (1 pool x 1 rank -- the one
MPI shape where a rank does hold everything) must still export a complete,
valid file. The refusal must also not hang: the guard's decision is taken
identically on all ranks before any collective, and these tests run under a
subprocess timeout that would catch a reintroduced deadlock.
"""

import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

from helpers import EXAMPLES_DIRECTORY, apply_config_overlay
from helpers_berkeleygw import read_berkeleygw_wavefunction_file

pytestmark = pytest.mark.skipif(
    shutil.which("mpirun") is None,
    reason="mpirun not found — MPI tests require an MPI installation",
)


@pytest.fixture(scope="module")
def mpi_q3_bin(q3_bin):
    """The q3_mpi binary, or a skip when --q3-bin points at a serial build.

    The serial CI job runs this whole directory with `--q3-bin ./build/q3`.
    `mpirun -np 2 <serial q3>` launches two INDEPENDENT serial processes, each
    of which correctly performs the full export -- which is exactly the
    behavior the multi-rank refusal tests exist to distinguish from a real
    `q3_mpi` run. The energy-comparison MPI tests happen to pass in that mode
    (N identical serial runs agree with each other trivially); these tests are
    the first whose assertions can tell the difference, so they must not run
    against a serial binary at all.
    """
    if "mpi" not in Path(q3_bin).name and "build_mpi" not in str(q3_bin):
        pytest.skip(
            f"--q3-bin={q3_bin} is not an MPI build; the export topology guard "
            "only exists in q3_mpi (pass --q3-bin ./build_mpi/q3_mpi)"
        )
    return q3_bin


SILICON_EXAMPLE_DIRECTORY = EXAMPLES_DIRECTORY / "total_energy--semiconductor--silicon"
NC_INPUT = SILICON_EXAMPLE_DIRECTORY / "nc.yml"

# The silicon example's 2x2x2 shifted Monkhorst-Pack grid reduces to 2 IBZ
# k-points; a complete export must carry both.
EXPECTED_NUMBER_OF_KPOINTS = 2
EXPECTED_VALENCE_ELECTRONS = 8.0

EXPORT_ARTIFACTS = ("WFN", "RHO", "vxc.dat")
RECORD_FILE_NAME = "mean-field.q3mf"
REFUSAL_TEXT = "not supported under MPI"


def _run_mpi_export(q3_bin, run_directory, number_of_ranks, extra_arguments=()):
    """Run the silicon example with a BerkeleyGW export on N ranks; return
    (completed process, outdir)."""
    input_path = apply_config_overlay(
        NC_INPUT,
        {
            "control": {
                "mean_field_export": "berkeleygw",
                "use_deterministic_fft_planning": True,
            }
        },
        run_directory,
    )
    command = ["mpirun", "-np", str(number_of_ranks), q3_bin, str(input_path)]
    command += list(extra_arguments)
    result = subprocess.run(
        command,
        cwd=str(run_directory),
        capture_output=True,
        text=True,
        timeout=600,
    )
    with open(input_path) as file_handle:
        outdir = Path(yaml.safe_load(file_handle)["control"]["outdir"])
    return result, outdir


@pytest.mark.parametrize(
    (
        "number_of_ranks",
        "extra_arguments",
        "expected_number_of_pools",
        "topology_label",
        "writes_checkpoint",
    ),
    [
        # --pools 1 has to be said: with two irreducible k-points the default
        # gives each rank its own pool. One pool means 2 ranks form one band
        # group whose members each hold a G-slab of every band.
        (2, ("--pools", "1"), 1, "band groups (2 ranks, 1 pool)", False),
        # 2 pools of 1 rank each: each pool holds 1 of the 2 IBZ k-points.
        (2, ("--pools", "2"), 2, "k-point pools (2 ranks, 2 pools)", True),
    ],
)
def test_multi_rank_export_refuses_loudly_and_exports_nothing(
    mpi_q3_bin,
    tmp_path,
    number_of_ranks,
    extra_arguments,
    expected_number_of_pools,
    topology_label,
    writes_checkpoint,
):
    """Any data-distributing topology must refuse the export, not write a
    corrupt or silently-partial one -- and the SCF itself must still succeed."""
    result, outdir = _run_mpi_export(mpi_q3_bin, tmp_path, number_of_ranks, extra_arguments)
    assert result.returncode == 0, (
        f"{topology_label}: the SCF run itself should succeed even when the export "
        f"is refused (exit {result.returncode})\nstderr: {result.stderr[-2000:]}"
    )

    exported = [name for name in EXPORT_ARTIFACTS if (outdir / name).exists()]
    assert not exported, (
        f"{topology_label}: export files {exported} were written even though no rank "
        f"holds the complete mean field"
    )
    stray_records = list(outdir.rglob(RECORD_FILE_NAME))
    assert not stray_records, f"{topology_label}: mean-field record written: {stray_records}"

    combined_output = result.stdout + result.stderr
    assert REFUSAL_TEXT in combined_output, (
        f"{topology_label}: the refusal must be logged so the user learns why their "
        f"requested export is missing; got no such message"
    )

    # The checkpoint needs less than the export: a rank that is its own pool
    # holds each of its k-points whole, so it writes their files itself
    # (test_mpi_checkpoint.py checks what they hold). Ranks sharing a pool hold
    # slabs, and must refuse before creating a partial save directory.
    if writes_checkpoint:
        assert (outdir / "si_nc.save" / "effective-potential.q3cp").is_file(), (
            f"{topology_label}: single-rank pools wrote no checkpoint"
        )
    else:
        assert not (outdir / "si_nc.save").exists(), (
            f"{topology_label}: a partial MPI checkpoint directory was created"
        )
        assert "Checkpoint output is not supported when ranks share a k-point pool" in (
            combined_output
        )

    results = yaml.safe_load((outdir / "si_nc_results.json").read_text())
    execution = results["execution"]
    assert execution["number_of_processes"] == number_of_ranks
    assert execution["number_of_kpoint_pools"] == expected_number_of_pools
    assert execution["band_group_size"] == number_of_ranks // expected_number_of_pools


def test_single_rank_mpi_export_is_complete_and_valid(mpi_q3_bin, tmp_path):
    """1 pool x 1 rank is the one MPI topology where a rank holds everything;
    the guard must let it export, and the file must carry the full k-set."""
    result, outdir = _run_mpi_export(mpi_q3_bin, tmp_path, number_of_ranks=1)
    assert result.returncode == 0, f"single-rank run failed: {result.stderr[-2000:]}"

    for name in EXPORT_ARTIFACTS:
        assert (outdir / name).is_file(), f"single-rank q3_mpi export produced no {name}"
    assert (outdir / "si_nc.save" / RECORD_FILE_NAME).is_file(), (
        "single-rank q3_mpi export wrote no mean-field record"
    )

    # The independent reader validates all framing; then the completeness facts.
    wavefunction_header = read_berkeleygw_wavefunction_file(str(outdir / "WFN"))
    assert wavefunction_header["number_of_kpoints"] == EXPECTED_NUMBER_OF_KPOINTS
    assert abs(sum(wavefunction_header["kpoint_weights"]) - 1.0) < 1.0e-10
    # A slab-partial export (the pre-guard band-group failure) would show short
    # per-k-point plane-wave counts; the full spheres for this example are
    # 1546/1557 at the default cutoff, so >1000 is a safe completeness floor
    # that does not pin the exact basis size.
    assert all(
        int(count) > 1000 for count in wavefunction_header["number_of_gvectors_per_kpoint"]
    ), (
        f"suspiciously small plane-wave counts "
        f"{wavefunction_header['number_of_gvectors_per_kpoint']} -- G-slab rather than full sphere?"
    )
