"""
CPU ↔ GPU energy-parity test (release blocker 15a).

Verifies that the GPU backend reproduces the serial CPU total energy on the
same physical input:

  * FP64:            |ΔE| ≤ 1e-10 Ry (double precision end to end)
  * mixed precision: |ΔE| ≤ 1e-5  Ry (FP32 subspace / projector work with
                                       FP64 accumulation)

Covers three code paths that have historically diverged independently:
  * NC, shifted 2×2×2 k-grid   — the general multi-k complex path
  * NC, Γ-only                 — half-sphere G-vectors + real DGEMM path
                                 (the path of the 2026-07 Γ-only GPU bugs)
  * US, shifted 2×2×2 k-grid   — ultrasoft augmentation + nonlocal projectors
  * PAW, shifted 2×2×2 k-grid  — augmentation + one-center corrections

Plus spin (LSDA, nspin=2: Fe/Ni/Cr) and non-Gaussian smearing schemes
(Methfessel-Paxton, Fermi-Dirac, Marzari-Vanderbilt: Cu) — these route through
the same generic execution-policy abstractions as the cases above (no
scheme- or spin-specific GPU code exists), but had no GPU coverage before.

Every run plans its FFTs deterministically, so the comparison sees the two
backends' arithmetic and not FFTW's per-process plan choice; see
DETERMINISTIC_FFT_PLANNING_OVERLAY.

The GPU binary is resolved by the session-scoped ``q3_gpu_binary`` fixture in
tests/conftest.py (``--q3-gpu-bin``, or auto-detected: Metal ``build/q3_gpu``
on macOS, ``build_cuda/q3_gpu_cuda`` on CUDA runners, ...).

Run:
    .venv/bin/pytest tests/validation/test_gpu_energy_parity.py -v \\
        --q3-bin ./build/q3 --q3-gpu-bin ./build/q3_gpu
"""

import pytest

from helpers import EXAMPLES_DIRECTORY, deep_merge, run_overlay_and_read_energy

# ── Tolerances ────────────────────────────────────────────────────────────────

FP64_PARITY_TOLERANCE_RYDBERG = 1e-10
"""
FP64 CPU↔GPU parity bound (roadmap release blocker 15a).

GPU BLAS/FFT reductions sum in a different order than the serial CPU path, so
bit-identity is not guaranteed. Double precision throughout keeps the
difference far below this bound: at the 1e-14 Ry level on the original silicon
cases (measured on Metal and CUDA), and at most 2.3e-13 Ry over all ten cases
on Metal once both sides plan their FFTs the same way (see
DETERMINISTIC_FFT_PLANNING_OVERLAY). Left to FFTW_MEASURE, plan choice alone
moved chromium's antiferromagnet by more than this bound.
"""

MIXED_PRECISION_TOLERANCE_RYDBERG = 1e-5
"""
Mixed-precision parity bound: FP32/TF32 subspace GEMMs and projector work
introduce ~1e-9–1e-7 Ry error on these systems; 1e-5 Ry is the acceptance
bound from the GPU acceleration plan (§ mixed precision).
"""

# ── Test cases ────────────────────────────────────────────────────────────────

SI_EXAMPLE_DIRECTORY = EXAMPLES_DIRECTORY / "total_energy--semiconductor--silicon"
CU_EXAMPLE_DIRECTORY = EXAMPLES_DIRECTORY / "total_energy--metal--copper"
FE_EXAMPLE_DIRECTORY = EXAMPLES_DIRECTORY / "total_energy--metal--magnetic--iron"
NI_EXAMPLE_DIRECTORY = EXAMPLES_DIRECTORY / "total_energy--metal--magnetic--nickel"
CR_EXAMPLE_DIRECTORY = EXAMPLES_DIRECTORY / "total_energy--metal--magnetic--chromium"

GAMMA_ONLY_OVERLAY = {"method": {"scf": {"kgrid": {"dimensions": [1, 1, 1], "shifts": [0, 0, 0]}}}}

# (case_id, base input, overlay, results prefix)
PARITY_CASES = [
    ("si_nc_2x2x2", SI_EXAMPLE_DIRECTORY / "nc.yml", None, "si_nc"),
    ("si_nc_gamma_only", SI_EXAMPLE_DIRECTORY / "nc.yml", GAMMA_ONLY_OVERLAY, "si_nc"),
    ("si_us_2x2x2", SI_EXAMPLE_DIRECTORY / "us.yml", None, "si_us"),
    ("si_paw_2x2x2", SI_EXAMPLE_DIRECTORY / "paw.yml", None, "si_paw"),
]

# Smearing schemes and LSDA (nspin=2) route through the same generic
# execution-policy abstractions as the cases above (no scheme- or
# spin-specific GPU/MPI code exists), but had never actually been run on
# the GPU backend before. Separate list so a regression here is easy to
# tell apart from the original three code-path cases.
SPIN_AND_SMEARING_PARITY_CASES = [
    ("cu_nc_methfessel_paxton", CU_EXAMPLE_DIRECTORY / "nc_mp.yml", None, "cu_nc_mp"),
    ("cu_nc_fermi_dirac", CU_EXAMPLE_DIRECTORY / "nc_fd.yml", None, "cu_nc_fd"),
    ("cu_nc_marzari_vanderbilt", CU_EXAMPLE_DIRECTORY / "nc_mv.yml", None, "cu_nc_mv"),
    ("fe_bcc_fm_lsda", FE_EXAMPLE_DIRECTORY / "nc.yml", None, "fe_nc"),
    ("ni_fcc_fm_lsda", NI_EXAMPLE_DIRECTORY / "nc.yml", None, "ni_nc"),
    ("cr_bcc_afm_lsda", CR_EXAMPLE_DIRECTORY / "nc.yml", None, "cr_afm_nc"),
]

DETERMINISTIC_FFT_PLANNING_OVERLAY = {"control": {"use_deterministic_fft_planning": True}}
"""
Every run here, serial and GPU, plans its FFTs with FFTW_ESTIMATE.

FFTW_MEASURE, the default, times candidate algorithms in each process. Two runs
of one binary then take bit-different SCF trajectories, and a soft-moment system
stops at a different point of its moment valley each time (KNOWN-ISSUES.md,
"Run-to-run energy variation for soft-moment systems under MPI"). Measured on
the Metal build, 2026-10-09, three runs each of the serial and GPU binaries:

  * chromium's antiferromagnet spread over 5.0e-10 Ry serial and 8.4e-10 Ry GPU;
  * its CPU↔GPU difference reached 7.8e-10 Ry, and 7 of the 9 pairings exceeded
    the FP64 bound, so the case failed more often than it passed;
  * the other metals moved by up to 3e-11 Ry.

With fixed plans every run repeated bit for bit. Seven of the ten cases agreed
exactly between CPU and GPU, and the largest difference was 2.3e-13 Ry (PAW
silicon). Metal has no GPU FFT, so both sides plan through FFTW. On CUDA the
GPU Davidson transforms through cuFFT, and the flag fixes the FFTW plans the
serial binary and the rest of the GPU binary still use.
"""

ALL_PARITY_CASES = [
    (case_id, base_yaml, deep_merge(DETERMINISTIC_FFT_PLANNING_OVERLAY, overlay or {}), prefix)
    for case_id, base_yaml, overlay, prefix in PARITY_CASES + SPIN_AND_SMEARING_PARITY_CASES
]
CASE_IDS = [case[0] for case in ALL_PARITY_CASES]


# ── Helpers ───────────────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def serial_energy_for_case(q3_command, tmp_path_factory):
    """Return a lazily computed, process-cached serial CPU reference energy."""
    energies = {}

    def get_energy(case_id, base_yaml, overlay, prefix):
        if case_id in energies:
            return energies[case_id]
        run_directory = tmp_path_factory.mktemp(f"serial_{case_id}")
        energies[case_id] = run_overlay_and_read_energy(
            q3_command, base_yaml, overlay, prefix, run_directory
        )
        return energies[case_id]

    return get_energy


# ── Tests ─────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("case_id,base_yaml,overlay,prefix", ALL_PARITY_CASES, ids=CASE_IDS)
def test_gpu_fp64_energy_parity(
    q3_gpu_binary, serial_energy_for_case, tmp_path, case_id, base_yaml, overlay, prefix
):
    """GPU FP64 total energy matches serial CPU within 1e-10 Ry."""
    gpu_energy = run_overlay_and_read_energy([q3_gpu_binary], base_yaml, overlay, prefix, tmp_path)
    serial_energy = serial_energy_for_case(case_id, base_yaml, overlay, prefix)
    difference = abs(gpu_energy - serial_energy)
    assert difference < FP64_PARITY_TOLERANCE_RYDBERG, (
        f"FP64 CPU↔GPU parity violated for {case_id}:\n"
        f"  E(serial CPU) = {serial_energy:.14f} Ry\n"
        f"  E(GPU FP64)   = {gpu_energy:.14f} Ry\n"
        f"  |diff|        = {difference:.2e} Ry"
        f"  (threshold: {FP64_PARITY_TOLERANCE_RYDBERG:.0e} Ry)"
    )


@pytest.mark.parametrize("case_id,base_yaml,overlay,prefix", ALL_PARITY_CASES, ids=CASE_IDS)
def test_gpu_mixed_precision_energy_parity(
    q3_gpu_binary, serial_energy_for_case, tmp_path, case_id, base_yaml, overlay, prefix
):
    """GPU mixed-precision total energy matches serial CPU within 1e-5 Ry."""
    gpu_energy = run_overlay_and_read_energy(
        [q3_gpu_binary],
        base_yaml,
        overlay,
        prefix,
        tmp_path,
        extra_args=("--precision", "mixed"),
    )
    serial_energy = serial_energy_for_case(case_id, base_yaml, overlay, prefix)
    difference = abs(gpu_energy - serial_energy)
    assert difference < MIXED_PRECISION_TOLERANCE_RYDBERG, (
        f"Mixed-precision CPU↔GPU parity violated for {case_id}:\n"
        f"  E(serial CPU) = {serial_energy:.14f} Ry\n"
        f"  E(GPU mixed)  = {gpu_energy:.14f} Ry\n"
        f"  |diff|        = {difference:.2e} Ry"
        f"  (threshold: {MIXED_PRECISION_TOLERANCE_RYDBERG:.0e} Ry)"
    )
