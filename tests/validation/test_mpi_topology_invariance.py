"""
Topology invariance test for k-point pool parallelism.

Verifies that the Si FCC SCF total energy is identical (within 1e-8 Ry)
when run on 1, 2, and 4 MPI ranks with the same physical input.

Uses the canonical Si NC example from examples/total_energy--semiconductor--silicon/nc.yml.

Run:
    .venv/bin/pytest tests/validation/test_mpi_topology_invariance.py -v \\
        --q3-bin ./build_mpi/q3
"""

import json
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml
from q3.io.results.classes import Incomplete

from helpers import (
    EXAMPLES_DIRECTORY,
    apply_config_overlay,
    atomic_forces,
    energy_contribution,
    load_results_json,
    run_q3_and_load_results,
    stress_contributions,
    stress_tensor,
)

pytestmark = pytest.mark.skipif(
    shutil.which("mpirun") is None,
    reason="mpirun not found — MPI tests require an MPI installation",
)

# ── Paths ─────────────────────────────────────────────────────────────────────

SI_EXAMPLE_DIRECTORY = EXAMPLES_DIRECTORY / "total_energy--semiconductor--silicon"
SI_NC_INPUT = SI_EXAMPLE_DIRECTORY / "nc.yml"
SI_NC_PREFIX = "si_nc"

NI_LSDA_EXAMPLE_DIRECTORY = EXAMPLES_DIRECTORY / "total_energy--metal--magnetic--nickel"
NI_LSDA_INPUT = NI_LSDA_EXAMPLE_DIRECTORY / "nc.yml"

PAW_PSEUDOPOTENTIAL_DIRECTORY = EXAMPLES_DIRECTORY / "data" / "pseudopotentials" / "paw"
SILICON_CARBON_LATTICE_VECTORS_BOHR = (
    (-4.119180301160476, 0.0, 4.119180301160476),
    (0.0, 4.119180301160476, 4.119180301160476),
    (-4.119180301160476, 4.119180301160476, 0.0),
)

SILICON_CARBON_ATOMIC_POSITIONS_CRYSTAL = ((0.0, 0.0, 0.0), (0.27, 0.263, 0.257))

PAW_POOL_STRESS_TOLERANCE_KBAR = 1.0e-4
"""
How far the PAW case's stress may move between pool counts (in kbar).

Pool counts sum the same terms in a different order. That round-off can change where
the SCF stops, or a step of the eigensolver, which moves the result by part of one SCF
iteration, and at this case's 1e-12 Ry threshold one more iteration moves the stress
by 1.6e-5 kbar. On macOS the pool counts agree to 1e-10 kbar; on Linux CI they came
out 1.07e-5 kbar apart, the same in both ScaLAPACK jobs, while the energies and forces
stayed within their 1e-8 bounds. A missing or doubled pool reduction moves the stress
by kbar.
"""

PAW_POOL_STRESS_CONTRIBUTION_TOLERANCE_KBAR = 1.0e-3
"""
The same bound for each stress contribution (in kbar). The contributions partly cancel
in their sum and each moves more than the total: one more SCF iteration moves the
augmentation and local terms by up to 7e-5 kbar.
"""


# The q3_bin fixture is provided by tests/validation/conftest.py.

# ── Helpers ───────────────────────────────────────────────────────────────────


def run_si_scf_on_n_ranks(
    q3_bin: str,
    number_of_ranks: int,
    working_directory: Path,
    number_of_pools: int | None = None,
    distribute_bands: bool = False,
) -> float:
    """
    Run the Si NC SCF example on number_of_ranks MPI ranks in the given
    working directory (a per-test tmp_path — never the tracked examples/
    directory) and return the total energy in Ry.

    The Si NC example has 2 irreducible k-points, so any rank count above 2
    leaves ranks sharing a pool. Q3 refuses that layout unless it was asked for,
    which is what number_of_pools is for: passing it states that the shared-pool
    (band-group) path is the thing under test, rather than arriving there by
    default.

    distribute_bands switches what ranks sharing a pool divide: the bands rather
    than the reciprocal-space coefficients of every band.
    """
    working_directory.mkdir(parents=True, exist_ok=True)
    command = ["mpirun", "-np", str(number_of_ranks), q3_bin]
    if number_of_pools is not None:
        command += ["--pools", str(number_of_pools)]
    if distribute_bands:
        command += ["--distribute-bands"]
    results, _ = run_q3_and_load_results(
        command,
        SI_NC_INPUT,
        SI_NC_PREFIX,
        working_directory=str(working_directory),
        timeout=600,
    )
    return float(results.properties.total_energy.value)


def run_ni_lsda_scf_on_n_ranks(
    q3_bin: str,
    number_of_ranks: int,
    smearing_type: str,
    working_directory: Path,
) -> float:
    """
    Run the Ni FCC LSDA (collinear spin-polarized) SCF example, with the given
    smearing scheme, on number_of_ranks MPI ranks and return the total energy
    in Ry. The k-grid is overlaid down to 4x4x4+shift (from the committed
    example's 12x12x12+shift) for MPI-test speed -- but NOT smaller than that:
    at 2x2x2+shift this structure's cubic (Oh, 48-op) symmetry reduces to only
    2 irreducible k-points, which is fewer than the 4 ranks/pools this test
    requests. Q3 clamps the pool count down to the available k-point count in
    that case (a Q3 rule, not an upstream one), so a 2x2x2 grid would
    silently turn "4 pools" into "2 pools of 2 ranks" -- comparing that
    against np=1 exercises band parallelism unintentionally (and with a
    larger floating-point noise floor than pure k-point-pool parallelism),
    not the plain topology invariance this test name promises. 4x4x4+shift
    gives 10 irreducible k-points, comfortably above 4, so "4 pools" means
    what it says: band_size=1, pure k-point-pool parallelism, matching what
    test_topology_invariance_1_vs_4_ranks above already validates for the
    unpolarized Si case.

    Uses control.use_deterministic_fft_planning: true so that FFTW_MEASURE's
    per-process runtime-timed plan selection isn't itself a source of
    run-to-run/cross-topology variation on top of genuine MPI floating-point
    non-associativity -- see SMEARING_TOPOLOGY_TOLERANCE_RYDBERG's docstring
    for the measurements this enabled.
    """
    overlay = {
        "method": {
            "scf": {"kgrid": {"dimensions": [4, 4, 4]}, "smearing": {"type": smearing_type}}
        },
        "control": {
            "use_deterministic_fft_planning": True,
            "prefix": f"ni_lsda_{smearing_type.replace('-', '_')}",
        },
    }
    input_path = apply_config_overlay(NI_LSDA_INPUT, overlay, working_directory)
    command = ["mpirun", "-np", str(number_of_ranks), q3_bin]
    results, _ = run_q3_and_load_results(
        command,
        input_path,
        f"ni_lsda_{smearing_type.replace('-', '_')}",
        working_directory=str(working_directory),
        timeout=600,
    )
    return float(results.properties.total_energy.value)


def run_paw_properties_with_topology(
    q3_bin: str, number_of_ranks: int, number_of_pools: int, working_directory: Path
):
    """Run a non-symmetric PAW SiC case and return its complete results."""
    working_directory.mkdir(parents=True, exist_ok=True)
    material = {
        "lattice": {
            "vectors": dict(zip(("a", "b", "c"), SILICON_CARBON_LATTICE_VECTORS_BOHR)),
            "units": {"length": "bohr", "angle": "degree"},
        },
        "basis": {
            "units": "crystal",
            "elements": [{"id": 0, "value": "Si"}, {"id": 1, "value": "C"}],
            "coordinates": [
                {"id": index, "value": position}
                for index, position in enumerate(SILICON_CARBON_ATOMIC_POSITIONS_CRYSTAL)
            ],
        },
    }
    material_path = working_directory / "material.json"
    material_path.write_text(json.dumps(material, indent=4) + "\n")

    prefix = f"silicon_carbon_paw_r{number_of_ranks}_p{number_of_pools}"
    input_data = {
        "material": str(material_path),
        "model": {"type": "dft", "subtype": "gga", "functional": "pbe"},
        "method": {
            "type": "pseudopotential",
            "subtype": "paw",
            "cutoffs": {"wavefunction": 45.0, "density": 180.0},
            "pseudopotentials": {
                "Si": str(PAW_PSEUDOPOTENTIAL_DIRECTORY / "Si.upf"),
                "C": str(PAW_PSEUDOPOTENTIAL_DIRECTORY / "C.upf"),
            },
            "scf": {
                "convergence_threshold": 1.0e-12,
                "max_iterations": 300,
                "number_of_bands": 8,
                "mixing": {"mode": "broyden", "beta": 0.3},
                "kgrid": {"dimensions": [2, 2, 2], "shifts": [0, 0, 0]},
            },
        },
        "control": {
            "disable_symmetry": True,
            "number_of_kpoint_pools": number_of_pools,
            "calculation": "scf",
            "outdir": str(working_directory / "output"),
            "prefix": prefix,
            "use_deterministic_fft_planning": True,
        },
    }
    input_path = working_directory / "input.yml"
    input_path.write_text(yaml.safe_dump(input_data, sort_keys=False))
    command = [
        "mpirun",
        "-np",
        str(number_of_ranks),
        q3_bin,
        "--pools",
        str(number_of_pools),
    ]
    results, _ = run_q3_and_load_results(
        command,
        input_path,
        prefix,
        working_directory=str(working_directory),
        timeout=1200,
    )
    return results


# ── Tests ─────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("number_of_ranks", [1, 2, 4])
def test_si_scf_runs_on_n_ranks(q3_bin, number_of_ranks, tmp_path):
    """Si NC SCF converges without error on 1, 2, and 4 MPI ranks."""
    # Above 2 ranks the Si example must be told to share pools; see the helper.
    number_of_pools = 2 if number_of_ranks > 2 else None
    energy = run_si_scf_on_n_ranks(q3_bin, number_of_ranks, tmp_path, number_of_pools)
    assert energy < 0.0, f"total_energy should be negative on {number_of_ranks} ranks, got {energy}"


TOPOLOGY_INVARIANCE_TOLERANCE_RYDBERG = 1e-8
"""
Tolerance for topology invariance (in Ry).

MPI Allreduce operations sum floating-point numbers in a different order than
the serial path, causing bit-level differences of ~1e-10 Ry. This tolerance
accommodates that inherent floating-point non-associativity while still ensuring
physical equivalence (1e-8 Ry ≈ 1.4e-7 eV, well within DFT accuracy).
"""


def test_topology_invariance_1_vs_2_ranks(q3_bin, tmp_path):
    """Total energy agrees on 1 and 2 ranks within 1e-8 Ry (FP tolerance)."""
    energy_1 = run_si_scf_on_n_ranks(q3_bin, 1, tmp_path / "ranks1")
    energy_2 = run_si_scf_on_n_ranks(q3_bin, 2, tmp_path / "ranks2")
    difference = abs(energy_1 - energy_2)
    assert difference < TOPOLOGY_INVARIANCE_TOLERANCE_RYDBERG, (
        f"Topology invariance violated:\n"
        f"  E(1 rank) = {energy_1:.12f} Ry\n"
        f"  E(2 ranks) = {energy_2:.12f} Ry\n"
        f"  |diff|    = {difference:.2e} Ry  (threshold: {TOPOLOGY_INVARIANCE_TOLERANCE_RYDBERG:.0e} Ry)"
    )


def test_topology_invariance_1_vs_4_ranks(q3_bin, tmp_path):
    """Total energy agrees on 1 and 4 ranks within 1e-8 Ry (FP tolerance)."""
    energy_1 = run_si_scf_on_n_ranks(q3_bin, 1, tmp_path / "ranks1")
    energy_4 = run_si_scf_on_n_ranks(q3_bin, 4, tmp_path / "ranks4", number_of_pools=2)
    difference = abs(energy_1 - energy_4)
    assert difference < TOPOLOGY_INVARIANCE_TOLERANCE_RYDBERG, (
        f"Topology invariance violated:\n"
        f"  E(1 rank) = {energy_1:.12f} Ry\n"
        f"  E(4 ranks) = {energy_4:.12f} Ry\n"
        f"  |diff|    = {difference:.2e} Ry  (threshold: {TOPOLOGY_INVARIANCE_TOLERANCE_RYDBERG:.0e} Ry)"
    )


SHARED_POOL_FORCE_TOLERANCE_RY_PER_BOHR = 1.0e-8
"""
How far a force may move when two ranks share the pool (in Ry/bohr).

The same bound the pool counts use below. Measured on 2H-SiC: 1e-10 per
contribution and 5e-15 after symmetrization. The defect this guards against was
0.45 Ry/bohr there, and 0.2-0.6 Ry/bohr on the reported cells.
"""


def test_forces_are_the_same_when_ranks_share_a_kpoint_pool(q3_bin, tmp_path):
    """
    Two ranks in one pool split each k-point's plane waves, and the forces must not notice.

    The nonlocal force multiplies two overlaps, each a partial sum over plane waves when the
    ranks of a pool split them. They used to be multiplied before being summed across the
    pool, which drops every cross term between the two ranks' slabs. The energy needs no such
    product, so it stayed exact, while the forces came out up to 0.6 Ry/bohr wrong
    (camsai/q3#22). Every other case in this module gives each pool one rank, so none of them
    split the plane waves; this one does, and says so before comparing.
    """
    one_rank = run_paw_properties_with_topology(q3_bin, 1, 1, tmp_path / "ranks_1_pools_1")
    shared_pool = run_paw_properties_with_topology(q3_bin, 2, 1, tmp_path / "ranks_2_pools_1")

    assert shared_pool.runtime_log.execution.reciprocal_space_vectors_are_distributed, (
        "two ranks in one pool should split the plane waves, or this compares nothing"
    )
    assert (shared_pool.runtime_log.incomplete or Incomplete()).forces is None
    assert (
        abs(shared_pool.properties.total_energy.value - one_rank.properties.total_energy.value)
        < 1.0e-8
    )
    assert max(abs(component) for force in atomic_forces(one_rank) for component in force) > (
        1.0e-3
    )

    worst = max(
        abs(shared - single)
        for shared_force, single_force in zip(
            atomic_forces(shared_pool),
            atomic_forces(one_rank),
        )
        for shared, single in zip(shared_force, single_force)
    )
    assert worst < SHARED_POOL_FORCE_TOLERANCE_RY_PER_BOHR, (
        f"forces moved by {worst:.3e} Ry/bohr when two ranks shared the k-point pool\n"
        f"  one rank:    {atomic_forces(one_rank)}\n"
        f"  shared pool: {atomic_forces(shared_pool)}"
    )


def test_paw_energy_force_and_stress_are_kpoint_pool_invariant(q3_bin, tmp_path):
    """Committed PAW data give the same complete properties on 1, 2, and 4 pools."""
    results_by_number_of_pools = {
        number_of_pools: run_paw_properties_with_topology(
            q3_bin,
            number_of_pools,
            number_of_pools,
            tmp_path / f"pools_{number_of_pools}",
        )
        for number_of_pools in (1, 2, 4)
    }
    baseline = results_by_number_of_pools[1]
    assert not baseline.runtime_log.incomplete
    assert max(abs(component) for force in atomic_forces(baseline) for component in force) > (
        1.0e-3
    )
    assert max(
        abs(stress_tensor(baseline)[row][column]) for row in range(3) for column in range(3)
    ) > (1.0)

    for number_of_pools in (2, 4):
        candidate = results_by_number_of_pools[number_of_pools]
        assert (
            abs(candidate.properties.total_energy.value - baseline.properties.total_energy.value)
            < 1.0e-8
        )
        assert (
            abs(
                energy_contribution(candidate, "one_center")
                - energy_contribution(baseline, "one_center")
            )
            < 1.0e-8
        )
        assert (
            max(
                abs(
                    atomic_forces(candidate)[atom][direction]
                    - atomic_forces(baseline)[atom][direction]
                )
                for atom in range(len(atomic_forces(baseline)))
                for direction in range(3)
            )
            < 1.0e-8
        )
        assert (
            max(
                abs(stress_tensor(candidate)[row][column] - stress_tensor(baseline)[row][column])
                for row in range(3)
                for column in range(3)
            )
            < PAW_POOL_STRESS_TOLERANCE_KBAR
        )
        assert stress_contributions(candidate).keys() == stress_contributions(baseline).keys()
        assert (
            max(
                abs(
                    stress_contributions(candidate)[contribution][row][column]
                    - stress_contributions(baseline)[contribution][row][column]
                )
                for contribution in stress_contributions(baseline)
                for row in range(3)
                for column in range(3)
            )
            < PAW_POOL_STRESS_CONTRIBUTION_TOLERANCE_KBAR
        )


# ── Local Thomas-Fermi preconditioning under a shared pool ───────────────────

SI_REFERENCE_INPUT = SI_EXAMPLE_DIRECTORY / "reference" / "nc" / "scf_qe.in"


def run_si_with_local_thomas_fermi(
    q3_bin: str, number_of_ranks: int, working_directory: Path
) -> float:
    """
    The silicon reference input with mixing_mode = 'local-TF', which is how
    local Thomas-Fermi preconditioning is asked for, and with its pseudopotential
    and output directories made absolute so it runs from a scratch directory.
    Every rank shares the one pool. Returns the total energy in Ry.
    """
    working_directory.mkdir(parents=True, exist_ok=True)
    pseudopotential_directory = (EXAMPLES_DIRECTORY / "data" / "pseudopotentials" / "nc").resolve()
    output_directory = working_directory / "output"
    replacements = {
        "mixing_mode = 'plain'": "mixing_mode = 'local-TF'",
        "conv_thr = 1.0e-8": "conv_thr = 1.0e-12",
        "pseudo_dir = '../../../data/pseudopotentials/nc/'": (
            f"pseudo_dir = '{pseudopotential_directory}/'"
        ),
        "outdir = './tmp'": f"outdir = '{output_directory}'",
    }
    text = SI_REFERENCE_INPUT.read_text()
    for original, replacement in replacements.items():
        assert original in text, f"{SI_REFERENCE_INPUT.name} no longer contains {original!r}"
        text = text.replace(original, replacement)
    input_path = working_directory / "scf_local_tf.in"
    input_path.write_text(text)

    command = ["mpirun", "-np", str(number_of_ranks), q3_bin]
    if number_of_ranks > 1:
        command += ["--pools", "1"]
    completed = subprocess.run(
        command + [str(input_path)],
        capture_output=True,
        text=True,
        cwd=str(working_directory),
        timeout=300,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return float(load_results_json(output_directory, "qe_nc").properties.total_energy.value)


def test_local_thomas_fermi_preconditioning_under_a_shared_pool(q3_bin, tmp_path):
    """
    The preconditioner's average screening sets the diagonal of its
    conjugate-gradient solve, and was taken over each rank's own real-space
    planes. Ranks sharing a pool hold different planes, so the solves took
    different steps around the same collective transforms and the run hung on
    two ranks before its first SCF iteration finished. A hang fails this test
    through the subprocess timeout.
    """
    serial_energy = run_si_with_local_thomas_fermi(q3_bin, 1, tmp_path / "serial")
    shared_energy = run_si_with_local_thomas_fermi(q3_bin, 2, tmp_path / "shared")
    assert abs(shared_energy - serial_energy) < 1.0e-8, (
        f"serial {serial_energy:.12f} Ry, 2 ranks {shared_energy:.12f} Ry"
    )


# ── LSDA + smearing cases ────────────────────────────────────────────────────
#
# The tests above cover only the unpolarized, fixed-occupation Si NC example.
# Spin-polarized (LSDA) runs exercise a second Allreduce-summed quantity (the
# magnetization channel of the mixed density) and every metallic smearing
# scheme drives a separate Fermi-bisection Allreduce loop -- neither path was
# previously exercised on more than 1 MPI rank.


@pytest.mark.parametrize("number_of_ranks", [1, 2, 4])
def test_ni_lsda_scf_runs_on_n_ranks(q3_bin, number_of_ranks, tmp_path):
    """Ni FCC LSDA (gaussian-smeared) SCF converges without error on 1, 2, 4 ranks."""
    energy = run_ni_lsda_scf_on_n_ranks(q3_bin, number_of_ranks, "gaussian", tmp_path)
    assert energy < 0.0, f"total_energy should be negative on {number_of_ranks} ranks, got {energy}"


SMEARING_TOPOLOGY_TOLERANCE_RYDBERG = {
    "gaussian": 2.0e-7,
    "methfessel-paxton": 2.0e-7,
    "marzari-vanderbilt": 2.0e-7,
    "fermi-dirac": 1.0e-4,
}
"""
Per-scheme 1-vs-4-rank energy tolerances, calibrated from measured intrinsic
spreads on this exact system (Ni FCC LSDA, 4x4x4+shift, degauss 0.02 Ry).

Original calibration (3x np=4 + 2x np=1 repeated runs per scheme,
2026-07-27, post-desync-fix, WITHOUT use_deterministic_fft_planning):

    scheme              same-topology spread   cross-topology |dE|   moment
    gaussian            0 (bit-identical)      4.3e-08               0.6853
    methfessel-paxton   0 (bit-identical)      2.0e-09               0.6890
    marzari-vanderbilt  0 (bit-identical)      1.6e-08               0.7079
    fermi-dirac         up to ~1e-04 (np=4)    8.3e-05               0.304-0.306

The Si test's 1e-8 tolerance above is NOT achievable here even with fully
correct MPI: a metallic soft-moment system stops at slightly different points
of a nearly-degenerate moment valley depending on topology (different
Allreduce-vs-serial rounding and stopping iteration), giving genuine ~1e-8 to
~4e-8 converged-energy differences for the stiff schemes -- this is why
gaussian/methfessel-paxton/marzari-vanderbilt stay at 2e-7 (a ~5x margin over
the worst observed genuine delta) rather than tightening further below.

Fermi-Dirac at degauss 0.02 Ry was the outlier: the moment restoring force is
so soft that dr2 < 1e-8 is satisfied anywhere along a nearly flat valley, so
its OLD 5e-3 tolerance mostly covered run-to-run FFTW_MEASURE plan-timing
noise (energies spreading ~1e-4 between REPEATED IDENTICAL np=4 runs), not
genuine cross-topology physics. With use_deterministic_fft_planning: true
(2026-07-28, see TODOs.md's "Deterministic FFT planning knob" entry) that
noise source is eliminated: 3x np=1 + 3x np=4 repeated runs of this exact
Fermi-Dirac case gave energies bit-identical to 12 printed decimal digits
across ALL 6 runs (same-topology AND cross-topology spread both exactly
0.0 -- see this branch's PR description for the raw measurement). 1e-4 keeps
a large (>1000x) margin over that measurement rather than asserting the
literal zero observed on one small sample -- still a 50x tightening from the
original 5e-3, and matched to the ~1e-4 improvement suggested when the
deterministic-FFT knob was scoped.
"""


@pytest.mark.parametrize(
    "smearing_type", ["gaussian", "methfessel-paxton", "marzari-vanderbilt", "fermi-dirac"]
)
def test_ni_lsda_smearing_topology_invariance_1_vs_4_ranks(q3_bin, smearing_type, tmp_path):
    """LSDA total energy agrees on 1 vs 4 ranks per scheme-calibrated tolerance.

    Was xfail (nondeterministic tolerance breaches and MPI_ERR_TRUNCATE crashes)
    until the k-point-pool convergence-decision synchronization landed in
    compute_residual_between (see that function's docstring in
    src/layer_4_computation/iterator/kohn_sham/planewave/init.hpp for the
    root-cause writeup: per-rank FFTW_MEASURE plans made dr2 bit-divergent
    across pools, splitting the SCF-exit decision and desyncing the pools'
    collective call sequences). The crash mode is guarded by every scheme in
    this parametrization (a desync recurrence aborts the run); the energy
    assertion uses SMEARING_TOPOLOGY_TOLERANCE_RYDBERG — see its docstring for
    why 1e-8 is not physically achievable on this metallic soft-moment system.
    """
    tolerance = SMEARING_TOPOLOGY_TOLERANCE_RYDBERG[smearing_type]
    energy_1 = run_ni_lsda_scf_on_n_ranks(q3_bin, 1, smearing_type, tmp_path / "ranks1")
    energy_4 = run_ni_lsda_scf_on_n_ranks(q3_bin, 4, smearing_type, tmp_path / "ranks4")
    difference = abs(energy_1 - energy_4)
    assert difference < tolerance, (
        f"LSDA topology invariance violated ({smearing_type}):\n"
        f"  E(1 rank) = {energy_1:.12f} Ry\n"
        f"  E(4 ranks) = {energy_4:.12f} Ry\n"
        f"  |diff|    = {difference:.2e} Ry  (threshold: {tolerance:.0e} Ry)"
    )


# ── Deterministic FFT planning ───────────────────────────────────────────────


def run_ni_fermi_dirac_scf_deterministic(q3_bin: str, working_directory: Path) -> float:
    """
    Run the Ni FCC LSDA + Fermi-Dirac SCF example (the worst-case system for
    run-to-run reproducibility: soft, nearly-degenerate moment valley) on 4
    ranks / 4 pools with control.use_deterministic_fft_planning: true, and return
    the total energy in Ry.
    """
    overlay = {
        "method": {
            "scf": {"kgrid": {"dimensions": [4, 4, 4]}, "smearing": {"type": "fermi-dirac"}}
        },
        "control": {"use_deterministic_fft_planning": True, "prefix": "ni_fd_deterministic"},
    }
    input_path = apply_config_overlay(NI_LSDA_INPUT, overlay, working_directory)
    command = ["mpirun", "-np", "4", q3_bin]
    results, _ = run_q3_and_load_results(
        command,
        input_path,
        "ni_fd_deterministic",
        working_directory=str(working_directory),
        timeout=600,
    )
    return float(results.properties.total_energy.value)


# ── The starting guess under a shared pool ────────────────────────────────────


@pytest.mark.parametrize("example", ["us.yml", "paw.yml"], ids=["ultrasoft", "paw"])
def test_first_scf_iteration_is_the_same_on_a_shared_pool(q3_bin, example, tmp_path):
    """
    One SCF iteration measures the starting guess rather than the fixed point.
    The guess keeps the lowest Ritz vectors of H on the atomic orbitals, and
    that H read the density-grid potential on the wavefunction grid, unscaled.
    Wherever a dataset has more atomic orbitals than bands, as both of these
    do, the bands kept therefore depended on how the pool was split: the
    energy after one iteration on two ranks was 3.3e-4 Ry (ultrasoft) and
    1.2e-4 Ry (PAW) away from serial. The SCF still reached the same answer,
    by a different path and in a different number of iterations.
    """
    energies = []
    for number_of_ranks in (1, 2):
        working_directory = tmp_path / f"ranks_{number_of_ranks}"
        overlay = {
            "method": {"scf": {"max_iterations": 1}},
            "control": {"prefix": "first_iteration", "use_deterministic_fft_planning": True},
        }
        input_path = apply_config_overlay(
            SI_EXAMPLE_DIRECTORY / example, overlay, working_directory
        )
        command = ["mpirun", "-np", str(number_of_ranks), q3_bin]
        if number_of_ranks > 1:
            command += ["--pools", "1"]
        completed = subprocess.run(
            command + [str(input_path)],
            capture_output=True,
            text=True,
            cwd=str(working_directory),
            timeout=600,
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr
        results = load_results_json(working_directory / "output", "first_iteration")
        assert results.runtime_log.self_consistent_field_iterations == 1
        energies.append(float(results.properties.total_energy.value))
    assert abs(energies[1] - energies[0]) < 1.0e-9, (
        f"after one iteration: serial {energies[0]:.12f} Ry, 2 ranks {energies[1]:.12f} Ry"
    )


def test_deterministic_fft_planning_gives_bit_identical_repeated_runs(q3_bin, tmp_path):
    """
    control.use_deterministic_fft_planning: true makes repeated identical runs of
    the previously-nondeterministic Fermi-Dirac Ni case (see
    SMEARING_TOPOLOGY_TOLERANCE_RYDBERG's docstring above -- this is the same
    knob that let the fermi-dirac topology-invariance tolerance tighten from
    5e-3 to 1e-4) reproducible to floating-point-print precision -- confirming
    FFTW_ESTIMATE (fixed, no runtime timing) replaces FFTW_MEASURE
    (per-process-timed, the source of the run-to-run variation documented in
    KNOWN-ISSUES.md) end to end.
    """
    energies = [
        run_ni_fermi_dirac_scf_deterministic(q3_bin, tmp_path / f"run{i}") for i in range(3)
    ]
    spread = max(energies) - min(energies)
    assert spread < 1e-9, (
        f"use_deterministic_fft_planning did not give reproducible runs:\n"
        f"  energies = {[f'{e:.12f}' for e in energies]}\n"
        f"  spread   = {spread:.2e} Ry"
    )


# ── Default topology selection ────────────────────────────────────────────────


def test_default_topology_gives_each_rank_its_own_pool(q3_bin, tmp_path):
    """
    With no --pools, ranks are spread over k-points rather than piled into one
    pool. Nickel at 4x4x4+shift has 10 irreducible k-points, so 4 ranks become 4
    single-rank pools -- the layout with measured scaling.
    """
    working_directory = tmp_path / "auto"
    working_directory.mkdir(parents=True, exist_ok=True)
    overlay = {
        "method": {"scf": {"kgrid": {"dimensions": [4, 4, 4]}, "smearing": {"type": "gaussian"}}},
        "control": {"use_deterministic_fft_planning": True},
    }
    input_path = apply_config_overlay(NI_LSDA_INPUT, overlay, working_directory)
    completed = subprocess.run(
        ["mpirun", "-np", "4", q3_bin, str(input_path)],
        capture_output=True,
        text=True,
        cwd=str(working_directory),
        timeout=900,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "4 k-point pool(s) x 1 rank(s) per pool" in completed.stdout, completed.stdout


def silicon_configuration_with_absolute_paths():
    """
    The silicon example's material and pseudopotential paths are relative to
    the example directory. Staged anywhere else they have to be absolute, or a
    run fails on a missing file before reaching whatever is under test -- which
    would make the test pass for the wrong reason.
    """
    configuration = yaml.safe_load(SI_NC_INPUT.read_text())
    configuration["material"] = str((SI_EXAMPLE_DIRECTORY / configuration["material"]).resolve())
    for species, relative_path in configuration["method"]["pseudopotentials"].items():
        configuration["method"]["pseudopotentials"][species] = str(
            (SI_EXAMPLE_DIRECTORY / relative_path).resolve()
        )
    return configuration


def test_band_distribution_is_invariant_across_topologies(q3_bin, tmp_path):
    """
    Dividing bands gives the serial answer, alone and nested inside k-point pools.

    The two axes compose: pools partition the k-points, and within each pool the
    ranks partition that pool's bands. Nesting them is the arrangement upstream
    codes use, and it is worth pinning because the two reductions involved are
    easy to get subtly wrong -- the inter-pool communicator pairs ranks by their
    position within a pool, a pairing written when those positions held matching
    slabs of the density grid. Under band distribution they hold whole grids
    instead. That still reduces correctly, since every rank in a pool ends up
    with the same whole density once the bands are gathered back, but "still
    correct for a different reason" is exactly the kind of claim that should be
    a test rather than a comment.

    4 ranks as 2 pools of 2 exercises both splits at once; 4 ranks in 1 pool
    exercises the band split alone.
    """
    serial_energy = run_si_scf_on_n_ranks(q3_bin, 1, tmp_path / "serial")
    for label, number_of_ranks, number_of_pools in (
        ("bands_only", 4, 1),
        ("pools_and_bands", 4, 2),
    ):
        energy = run_si_scf_on_n_ranks(
            q3_bin,
            number_of_ranks,
            tmp_path / label,
            number_of_pools=number_of_pools,
            distribute_bands=True,
        )
        difference = abs(energy - serial_energy)
        assert difference < TOPOLOGY_INVARIANCE_TOLERANCE_RYDBERG, (
            f"band distribution changed the total energy ({label}):\n"
            f"  serial                 = {serial_energy:.9f} Ry\n"
            f"  {number_of_ranks} ranks / {number_of_pools} pool(s) = {energy:.9f} Ry\n"
            f"  |diff|                 = {difference:.2e} Ry  "
            f"(threshold: {TOPOLOGY_INVARIANCE_TOLERANCE_RYDBERG:.0e} Ry)"
        )


def test_sharing_a_pool_must_be_asked_for(q3_bin, tmp_path):
    """
    The silicon example has 2 irreducible k-points, so 4 ranks cannot each own
    one. Rather than silently distributing the FFT instead -- slower, and it
    disables gamma-only, stress, checkpointing and export -- the run stops and
    says what to do about it.
    """
    working_directory = tmp_path / "refused"
    working_directory.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        ["mpirun", "-np", "4", q3_bin, str(SI_NC_INPUT)],
        capture_output=True,
        text=True,
        cwd=str(working_directory),
        timeout=600,
    )
    assert completed.returncode != 0, "expected the implicit shared-pool layout to be refused"
    combined_output = completed.stdout + completed.stderr
    assert "irreducible k-point" in combined_output, combined_output
    assert "--pools" in combined_output, combined_output

    # The refusal is raised identically on every rank, and the first rank to
    # exit nonzero takes the whole job down with it -- including rank 0,
    # part-way through printing. So the explanation has to reach the user from
    # whichever rank got there first, not only from rank 0, whose console is
    # the one init_mpi() leaves unsilenced. Asserting on stderr specifically is
    # what pins that down: the unconditional path writes there, the rank-0-only
    # logger writes to stdout. Without it this test passed or failed on a race,
    # losing the message roughly a quarter of the time under load.
    assert "irreducible k-point" in completed.stderr, completed.stderr

    # MPI_Abort after MPI_Finalize is undefined behaviour -- and reachable here,
    # because unwinding out of create_scf_iterator_mpi destroys the execution
    # policy it took by value, whose destructor finalizes. OpenMPI answers it
    # with a wall of standard-violation warnings that buries the explanation
    # above, and exits 14 rather than the 1 the code asks for.
    assert "after MPI_FINALIZE" not in completed.stderr, completed.stderr


def test_a_failure_on_one_rank_alone_fails_the_job_rather_than_hanging(q3_bin, tmp_path):
    """
    A rank that fails on its own must take the job down and say why, not leave
    it wedged.

    Every step of the execution policy's teardown is collective -- MPI_Comm_free
    and MPI_Finalize no less than the barrier ahead of them -- and the policy is
    destroyed while unwinding, because create_scf_iterator_mpi takes it by value
    and the iterator owns it afterwards. So a rank unwinding alone would enter
    that teardown and block, waiting on peers that were still inside a physics
    collective and would never arrive. The job hung indefinitely rather than
    failing, and since the barrier sits ahead of the handler that reports the
    error, stderr stayed empty: no exit, no explanation. Measured on the code
    before the fix, this scenario ran past 75 seconds having printed nothing;
    after it, the job exits nonzero in about a second with the reason.

    The vehicle matters, and is easy to get wrong. The failing rank is given a
    pseudopotential path that does not exist -- not a missing *input* file --
    because pseudopotentials are loaded inside create_scf_iterator_mpi, past
    the point where the policy was moved into the callee. A missing input file
    throws in parse_input instead, while main still holds the policy, and never
    reaches the destructor while unwinding at all. That path did not hang, and
    a test built on it would pass whether or not this bug were present.
    """
    working_directory = tmp_path / "rank_local_failure"
    working_directory.mkdir(parents=True, exist_ok=True)

    sound_configuration = silicon_configuration_with_absolute_paths()
    sound_input = working_directory / "sound.yml"
    sound_input.write_text(yaml.safe_dump(sound_configuration))

    broken_configuration = silicon_configuration_with_absolute_paths()
    broken_configuration["method"]["pseudopotentials"] = {
        species: str(Path(path).with_name("absent_" + Path(path).name))
        for species, path in broken_configuration["method"]["pseudopotentials"].items()
    }
    broken_input = working_directory / "broken.yml"
    broken_input.write_text(yaml.safe_dump(broken_configuration))

    """
    MPMD: rank 0 gets the sound input and goes on to do physics, rank 1 gets
    the broken one and fails alone -- which is the case a collective refusal
    raised on every rank cannot cover. Ranks are numbered in app-context order,
    so the failing rank is rank 1. Two ranks over silicon's two irreducible
    k-points is one pool each; --pools states that rather than leaning on the
    default, so a change to the default cannot quietly change what is tested.
    """
    command = [
        "mpirun",
        "-np",
        "1",
        q3_bin,
        "--pools",
        "2",
        str(sound_input),
        ":",
        "-np",
        "1",
        q3_bin,
        "--pools",
        "2",
        str(broken_input),
    ]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            cwd=str(working_directory),
            timeout=300,
        )
    except subprocess.TimeoutExpired:
        pytest.fail(
            "the job hung: one rank failed and the run never exited. When this "
            "regresses, the cause is a collective call reached while unwinding "
            "-- MPIExecutionPolicy's teardown must run none while an exception "
            "is in flight, so that the handler in main_mpi.cpp can report the "
            "error and abort."
        )

    assert completed.returncode != 0, "a failure on one rank must fail the job"

    """
    On stderr specifically, and tagged with the rank that failed: the reporting
    path that survives a silenced non-root rank writes there, and rank 1's
    console logger is off.
    """
    assert "[rank 1] Fatal error" in completed.stderr, completed.stderr
    assert "absent_Si.upf" in completed.stderr, completed.stderr

    """
    Teardown must not have run: had it finalized MPI on the way out, the abort
    that follows would be an MPI_Abort after MPI_Finalize, which OpenMPI answers
    with standard-violation warnings that bury the message above.
    """
    assert "after MPI_FINALIZE" not in completed.stderr, completed.stderr


def test_sharing_a_pool_is_allowed_when_requested(q3_bin, tmp_path):
    """The same run succeeds once --pools states the intent."""
    energy = run_si_scf_on_n_ranks(q3_bin, 4, tmp_path / "requested", number_of_pools=2)
    assert energy < 0.0


# ── Gamma-only under a shared pool ────────────────────────────────────────────

GAMMA_ONLY_PREFIX = "si_gamma"
# Converged well past the 1e-8 Ry comparison: different topologies take
# different SCF paths to the same fixed point, and PAW at the examples' 1e-8
# threshold stops 4e-7 Ry short of it.
GAMMA_ONLY_OVERLAY = {
    "method": {
        "scf": {
            "convergence_threshold": 1.0e-12,
            "kgrid": {"dimensions": [1, 1, 1], "shifts": [0, 0, 0]},
        }
    },
    "control": {"prefix": GAMMA_ONLY_PREFIX, "use_deterministic_fft_planning": True},
}


def run_si_gamma_only_scf(
    q3_bin: str,
    example_input: Path,
    number_of_ranks: int,
    working_directory: Path,
    distribute_bands: bool = False,
) -> tuple[float, str]:
    """
    A Si example at Gamma alone -- a 1x1x1 unshifted grid, which is what
    switches the half-sphere storage on -- run with every rank in the one pool.
    Returns the total energy in Ry and the console output, which says whether
    the half sphere was kept.
    """
    working_directory.mkdir(parents=True, exist_ok=True)
    input_path = apply_config_overlay(example_input, GAMMA_ONLY_OVERLAY, working_directory)
    command = ["mpirun", "-np", str(number_of_ranks), q3_bin]
    if number_of_ranks > 1:
        command += ["--pools", "1"]
    if distribute_bands:
        command += ["--distribute-bands"]
    completed = subprocess.run(
        command + [str(input_path)],
        capture_output=True,
        text=True,
        cwd=str(working_directory),
        timeout=600,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    results = load_results_json(working_directory / "output", GAMMA_ONLY_PREFIX)
    return float(results.properties.total_energy.value), completed.stdout


@pytest.mark.parametrize(
    ("example_input", "number_of_ranks", "distribute_bands"),
    [
        (SI_NC_INPUT, 2, False),
        (SI_NC_INPUT, 4, False),
        (SI_NC_INPUT, 2, True),
        (SI_EXAMPLE_DIRECTORY / "us.yml", 2, False),
        (SI_EXAMPLE_DIRECTORY / "us.yml", 4, False),
        (SI_EXAMPLE_DIRECTORY / "paw.yml", 2, False),
    ],
    ids=[
        "nc_2_ranks_sticks",
        "nc_4_ranks_sticks",
        "nc_2_ranks_bands",
        "us_2_ranks_sticks",
        "us_4_ranks_sticks",
        "paw_2_ranks_sticks",
    ],
)
def test_gamma_only_survives_a_shared_pool(
    q3_bin, example_input, number_of_ranks, distribute_bands, tmp_path
):
    """
    Ranks sharing a pool used to fall back to the full sphere, because the
    slab FFT could put a G-vector and its negative on different ranks. The
    stick FFT keeps them together, and the gamma trick's origin correction is
    applied by the one rank whose plane-wave list begins with G = 0. The energy
    must match the serial half-sphere run, and the run must say it kept the
    optimisation rather than quietly giving it up. Band distribution keeps the
    whole half sphere on every rank and is the control.

    Ultrasoft and PAW make the same correction once more, in the projector
    overlaps that build the augmentation charge. That one asked the density
    list, which is complete on every rank, whether G = 0 came first, so every
    rank subtracted an origin term and the energy on two ranks was off by
    about 1e-3 Ry. Norm-conserving decks never reach that code.
    """
    serial_energy, _ = run_si_gamma_only_scf(q3_bin, example_input, 1, tmp_path / "serial")
    shared_energy, console = run_si_gamma_only_scf(
        q3_bin,
        example_input,
        number_of_ranks,
        tmp_path / f"shared_{number_of_ranks}",
        distribute_bands=distribute_bands,
    )
    assert "Gamma-only optimization enabled" in console
    assert "Gamma-only disabled" not in console
    assert abs(shared_energy - serial_energy) < 1.0e-8, (
        f"serial {serial_energy:.12f} Ry, {number_of_ranks} ranks {shared_energy:.12f} Ry"
    )


# ── Equal wavefunction and density grids under a shared pool ─────────────────

EQUAL_GRIDS_PREFIX = "si_equal_grids"


def run_si_with_equal_grids(
    q3_bin: str, number_of_ranks: int, kgrid: dict, working_directory: Path
) -> tuple[float, str]:
    """
    The Si NC example with its density cutoff lowered to twice the
    wavefunction cutoff, which makes the wavefunction grid -- sized for twice
    the wavefunction cutoff (four times on the exact grid), capped at the
    density grid -- the same size as the density grid. Every rank shares the
    one pool. Returns the total energy in Ry and the console output.
    """
    working_directory.mkdir(parents=True, exist_ok=True)
    with open(SI_NC_INPUT) as file_handle:
        wavefunction_cutoff = float(
            yaml.safe_load(file_handle)["method"]["cutoffs"]["wavefunction"]
        )
    overlay = {
        "method": {
            "cutoffs": {"density": 2.0 * wavefunction_cutoff},
            "scf": {"convergence_threshold": 1.0e-12, "kgrid": kgrid},
        },
        "control": {"prefix": EQUAL_GRIDS_PREFIX, "use_deterministic_fft_planning": True},
    }
    input_path = apply_config_overlay(SI_NC_INPUT, overlay, working_directory)
    command = ["mpirun", "-np", str(number_of_ranks), q3_bin]
    if number_of_ranks > 1:
        command += ["--pools", "1"]
    completed = subprocess.run(
        command + [str(input_path)],
        capture_output=True,
        text=True,
        cwd=str(working_directory),
        timeout=600,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    results = load_results_json(working_directory / "output", EQUAL_GRIDS_PREFIX)
    return float(results.properties.total_energy.value), completed.stdout


@pytest.mark.parametrize(
    "kgrid",
    [
        {"dimensions": [2, 2, 2], "shifts": [1, 1, 1]},
        {"dimensions": [1, 1, 1], "shifts": [0, 0, 0]},
    ],
    ids=["kpoints", "gamma"],
)
def test_equal_wave_and_density_grids_under_a_shared_pool(q3_bin, kgrid, tmp_path):
    """
    With the two grids the same size, the band density used to be accumulated
    straight onto the density grid. Ranks sharing a pool split that grid and
    the wavefunction grid by stick independently, so a plane wave whose stick
    the density grid gave to another rank was dropped: 0.79 Ry off at these
    k-points on two ranks, 1.5 Ry at Gamma. The examples keep the density
    cutoff at four times the wavefunction cutoff and never reached that path.
    """
    serial_energy, serial_console = run_si_with_equal_grids(q3_bin, 1, kgrid, tmp_path / "serial")
    shared_energy, _ = run_si_with_equal_grids(q3_bin, 2, kgrid, tmp_path / "shared")

    grid_lines = [
        line.split("=")[-1].strip()
        for line in serial_console.splitlines()
        if "FFT grid (rho)" in line or "FFT grid (wave)" in line
    ]
    assert len(grid_lines) == 2 and grid_lines[0] == grid_lines[1], (
        f"the density and wavefunction grids must be the same size here: {grid_lines}"
    )
    assert abs(shared_energy - serial_energy) < 1.0e-8, (
        f"serial {serial_energy:.12f} Ry, 2 ranks {shared_energy:.12f} Ry"
    )


def test_each_rank_writes_its_own_log_file(q3_bin, tmp_path):
    """
    --log-file names a file for rank 0; every other rank appends its rank index,
    because the console carries rank 0 alone and a rank that fails on its own
    would otherwise have nowhere to say so.

    init_mpi() has documented this since it was written, and it did not work:
    the file was registered as a logger of its own rather than as a sink on the
    default one, so nothing was ever written to it, and silence() then set the
    whole logger to off on non-root ranks -- which would have silenced the file
    too. Both are fixed; this pins the behavior the docstring promised.
    """
    working_directory = tmp_path / "per_rank_logs"
    working_directory.mkdir(parents=True, exist_ok=True)

    completed = subprocess.run(
        ["mpirun", "-np", "2", q3_bin, "--pools", "2", "--log-file", "run.log", str(SI_NC_INPUT)],
        capture_output=True,
        text=True,
        cwd=str(working_directory),
        timeout=900,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr

    root_log = working_directory / "run.log"
    rank_one_log = working_directory / "run.log.1"
    assert root_log.is_file(), "rank 0 wrote no log file"
    assert rank_one_log.is_file(), "rank 1 wrote no log file of its own"
    assert root_log.stat().st_size > 0, "rank 0's log file is empty"
    assert rank_one_log.stat().st_size > 0, (
        "rank 1's log file is empty: silencing its console must not silence its file"
    )


# ── Isolated-system corrections under a shared pool ──────────────────────────

SI_ATOM_INPUT = EXAMPLES_DIRECTORY / "total_energy--atom--silicon" / "nc_charged.yml"
SI_ATOM_MATERIAL = EXAMPLES_DIRECTORY / "data" / "materials" / "silicon_atom.json"
ISOLATED_PREFIX = "silicon_atom_isolated"


def run_off_centre_silicon_atom(
    q3_bin: str,
    correction: str,
    lattice_overrides: dict,
    number_of_ranks: int,
    working_directory: Path,
) -> dict:
    """
    The charged silicon atom example with the atom moved off the centre of its
    box, so that its density changes when the axes are permuted, run with every
    rank in the one pool. Returns the results JSON.
    """
    working_directory.mkdir(parents=True, exist_ok=True)
    material = json.loads(SI_ATOM_MATERIAL.read_text())
    material["basis"]["coordinates"][0]["value"] = [0.30, 0.45, 0.60]
    material["lattice"].update(lattice_overrides)
    material_path = working_directory / "material.json"
    material_path.write_text(json.dumps(material, indent=4) + "\n")
    overlay = {
        "material": str(material_path),
        "control": {
            "prefix": ISOLATED_PREFIX,
            "isolated_system_correction": correction,
            "use_deterministic_fft_planning": True,
        },
    }
    input_path = apply_config_overlay(SI_ATOM_INPUT, overlay, working_directory)
    command = ["mpirun", "-np", str(number_of_ranks), q3_bin]
    if number_of_ranks > 1:
        command += ["--pools", "1"]
    completed = subprocess.run(
        command + [str(input_path)],
        capture_output=True,
        text=True,
        cwd=str(working_directory),
        timeout=600,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return load_results_json(working_directory / "output", ISOLATED_PREFIX)


@pytest.mark.parametrize(
    ("correction", "lattice_overrides"),
    [
        ("makov-payne", {}),
        ("martyna-tuckerman", {"type": "ORC", "b": 9.0, "c": 10.0}),
    ],
    ids=["makov_payne_cubic", "martyna_tuckerman_orthorhombic"],
)
def test_isolated_system_corrections_under_a_shared_pool(
    q3_bin, correction, lattice_overrides, tmp_path
):
    """
    Both corrections need the position of every grid point, and took it from
    the local index plus local_offset() planes, which is the layout of a slab
    along the first dimension. Ranks sharing a pool hold planes along the third
    dimension, so each point was placed at a permutation of its coordinates.
    For this atom on two ranks in one pool the Makov-Payne second-order term
    came out 0.163 Ry against 0.014 Ry serial, and the Martyna-Tuckerman
    corrected energy -18.857 Ry against -3.834 Ry. Makov-Payne accepts cubic
    cells only, where the permutation leaves a centred atom unchanged, hence
    the off-centre atom; the box with unequal sides does the same for the
    Martyna-Tuckerman kernel.
    """
    serial = run_off_centre_silicon_atom(
        q3_bin, correction, lattice_overrides, 1, tmp_path / "serial"
    )
    shared = run_off_centre_silicon_atom(
        q3_bin, correction, lattice_overrides, 2, tmp_path / "shared"
    )
    serial_correction = serial.properties.isolated_system_correction
    shared_correction = shared.properties.isolated_system_correction
    assert serial_correction.method == correction
    assert abs(shared.properties.total_energy.value - serial.properties.total_energy.value) < 1.0e-8
    serial_total = serial_correction.total.value
    shared_total = shared_correction.total.value
    assert abs(shared_total - serial_total) < 1.0e-8, (
        f"{correction}: serial {serial_total:.10f} Ry, 2 ranks {shared_total:.10f} Ry"
    )
