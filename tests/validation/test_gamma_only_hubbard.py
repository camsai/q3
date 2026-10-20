"""
DFT+U at a single Gamma point, where Q3 stores only half of G-space.

DFT+U was refused outright for Gamma-only runs until 2026-08-09, which is
precisely the mode a user reaches for on a large cell -- a defect or an
adsorbate with a Hubbard correction on it. The refusal was honest: at an
unshifted Gamma point Q3 keeps one G of each +-G pair, and its half sphere
*includes* G = 0, so every overlap taken against that list is half of what it
stands for. The completed value is ``2*Re[half]`` with the G = 0 term -- counted
twice by the doubling -- subtracted once, except where the summand carries a
factor of G_alpha and therefore vanishes at G = 0. Getting that pair of rules
wrong is what made four force terms wrong simultaneously in this same code on
2026-08-08 (KNOWN-ISSUES.md).

Four places in the DFT+U path take such an overlap: the occupation matrix
``<phi_m|S|psi>``, the ``<beta_j|phi_m>`` that builds S|phi> for ultrasoft and
PAW, the Hamiltonian's ``|phi> V <phi|`` term, and both branches of the force.

**Two independent gates**, in the order they must be trusted:

1. *The energy, against Q3's own full sphere.* Both runs sample exactly one
   k-point at Gamma; the only difference is the basis, because ``kgrid:
   [1,1,1]`` triggers the half-sphere storage while an explicit
   ``kpoints_list`` leaves the grid dimensions unset so the detection does not
   fire. Any half-sphere bookkeeping error moves the total energy, and nothing
   else does.
2. *The force, against a central difference of Q3's own Gamma-only energy.*
   This is downstream of the first: a force checked against an energy that is
   itself wrong proves nothing, which is why the energy gate comes first.

Measured, with the correction in place and with each half of it removed:

===========================  ==========  ==========================
quantity                     correct     with the correction removed
===========================  ==========  ==========================
E_total, gamma vs full       8e-9 Ry     8.3e-2 Ry (no doubling)
E_total, gamma vs full       8e-9 Ry     1.2e-6 Ry (no G=0 term)
F/(-dE/dR), ultrasoft        0.999971    1.0934
F/(-dE/dR), norm-conserving  0.999979    1.0334
===========================  ==========  ==========================

The G = 0 term is the small one here and that is expected, not luck: a Hubbard
manifold has l >= 1, and an orbital with l >= 1 vanishes at G = 0. It survives
at 1e-6 Ry through the augmented projector S|phi>, and it is the term whose
absence skewed the KB nonlocal force unevenly rather than scaling it, so it is
gated rather than dismissed.

Run:
    .venv/bin/pytest tests/validation/test_gamma_only_hubbard.py -v --q3-bin ./build/q3
"""

import json
import math
import re
import subprocess
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).resolve().parents[2]
SILICON_MATERIAL = REPOSITORY / "examples" / "data" / "materials" / "silicon.json"
PSEUDOPOTENTIALS = REPOSITORY / "examples" / "data" / "pseudopotentials"

DATASETS = {
    "ultrasoft": ("ultrasoft", PSEUDOPOTENTIALS / "us" / "Si_PBE_USPP.UPF"),
    "norm_conserving": ("norm_conserving", PSEUDOPOTENTIALS / "nc" / "Si.upf"),
}
"""
The two Hubbard projector conventions, which are two different code paths.

Ultrasoft (and PAW, which shares its branch) projects with S|phi>, so its
occupation matrix and its force both run through ``<beta_j|phi_m>`` as well;
norm-conserving projects with |phi> and never touches that overlap. Both are
exercised, because a Gamma correction applied to one and not the other would
pass a single-dataset test.
"""

BOHR_TO_ANGSTROM = 0.529177249
SILICON_PRIMITIVE_A_ANGSTROM = 3.866976

FIRST_CELL_VECTOR_BOHR = (
    -0.5 * SILICON_PRIMITIVE_A_ANGSTROM * math.sqrt(2.0) / BOHR_TO_ANGSTROM,
    0.0,
    +0.5 * SILICON_PRIMITIVE_A_ANGSTROM * math.sqrt(2.0) / BOHR_TO_ANGSTROM,
)
"""
a1 of ``silicon.json``, in bohr, read from the same two rules Q3 applies to it.

The material states an FCC lattice with the ESSE primitive vector length `a`;
the YAML reader multiplies that by sqrt(2) to get the conventional alat, and
``lattice.hpp``'s Bravais index 2 lays out a1 = alat/2 (-1, 0, 1). Both steps
are spelled out because projecting the force onto the wrong axis makes a
correct force look badly wrong -- 1.9x, on the ultrasoft gate this one mirrors
-- and the failure gives no hint that the axis is what is at fault.
"""

BASE_DISPLACEMENT = 0.020
"""
Atom 2's offset from its lattice site along a1, in *every* run of the triple.

The diamond site is symmetric, so a difference centred on the ideal position
vanishes identically and the test would measure nothing.
"""

FINITE_DIFFERENCE_STEP = 0.002
"""The fractional step taken around that base displacement."""

CONVERGENCE_THRESHOLD = "1.0e-12"
"""
``convergence_threshold`` bounds the *density* residual, not the energy.

At 1e-8 the converged energies still scatter by ~5e-6 Ry between neighbouring
geometries, which is larger than the energy difference across a small
displacement, and the finite difference then wanders with step size instead of
converging. The same threshold is used for the energy gate, where the quantity
being compared is 1e-8 Ry.
"""

ENERGY_TOLERANCE_RYDBERG = 1.0e-7
"""
|E(gamma-only) - E(full sphere)|, for the same single Gamma point.

Measured 8e-9 Ry, which is the printed resolution. Removing the G = 0 term
gives 1.2e-6 and removing the doubling gives 8.3e-2, so this sits ~15x above
the noise and ~15x below the smaller of the two defects it exists to catch.
"""

FORCE_TOLERANCE = 1.0e-3
"""
|F(analytic) / F(finite-difference) - 1|.

Measured 2.9e-5 (ultrasoft) and 2.1e-5 (norm-conserving), both dominated by the
O(h^2) truncation of the central difference. The defects it catches are 9.3%
and 3.3%.
"""

INPUT_TEMPLATE = """material: {material}
model: {{type: dft, subtype: gga, functional: pbe}}
method:
  type: pseudopotential
  subtype: {subtype}
  cutoffs: {{wavefunction: 30.0, density: 200.0}}
  pseudopotentials: {{Si: {pseudopotential}}}
  scf:
    convergence_threshold: {convergence_threshold}
    max_iterations: 300
    mixing: {{mode: broyden, beta: 0.3}}
    smearing: {{type: gaussian, degauss: 0.005}}
{kpoints}    number_of_bands: 8
{hubbard}control: {{calculation: scf, outdir: {outdir}, prefix: si{control_extra}}}
"""

GAMMA_ONLY_KPOINTS = "    kgrid: {dimensions: [1, 1, 1], shifts: [0, 0, 0]}\n"

FULL_SPHERE_KPOINTS_LIST = ", kpoints_list: [[0.0, 0.0, 0.0, 2.0]]"
"""
The same single k-point at Gamma, stated explicitly instead of as a grid.

``build_global_kpoint_grid_from_input`` takes the list verbatim and never calls
``KPointsGrid::generate``, so the grid dimensions stay at their {0,0,0} default
and the ``1x1x1 and unshifted`` test that switches on the half sphere does not
fire. Weight 2.0 is what a generated 1x1x1 grid carries: the spin degeneracy is
embedded in the k-point weights.
"""

HUBBARD_BLOCK = "    hubbard:\n      - {species: Si, orbital_label: p, u_ev: 3.0}\n"
"""
U = 3 eV on silicon's 3p shell -- the same manifold and value as the committed
QE deck in ``examples/.../reference/us_hubbard/scf_qe.in``, so this starts from
a configuration the repository already converges.
"""


def write_input(directory, dataset, gamma_only, displacement, hubbard=True):
    subtype, pseudopotential = DATASETS[dataset]
    material = json.loads(SILICON_MATERIAL.read_text())
    material["basis"]["coordinates"][1]["value"] = [0.125 + displacement, 0.125, 0.125]
    material_path = directory / "silicon.json"
    material_path.write_text(json.dumps(material))

    input_path = directory / "si.yml"
    input_path.write_text(
        INPUT_TEMPLATE.format(
            material=material_path,
            subtype=subtype,
            pseudopotential=pseudopotential,
            convergence_threshold=CONVERGENCE_THRESHOLD,
            kpoints=GAMMA_ONLY_KPOINTS if gamma_only else "",
            hubbard=HUBBARD_BLOCK if hubbard else "",
            outdir=directory / "out",
            control_extra="" if gamma_only else FULL_SPHERE_KPOINTS_LIST,
        )
    )
    return input_path


def run_q3(q3_bin, directory, dataset, gamma_only, displacement, hubbard=True):
    """Run q3 on a generated deck; return (stdout, results.json dict).

    Both are returned because the two gates below need different things: the
    energy and force are in results.json, but the per-atom Tr[n] occupation
    trace and the per-atom-per-contribution force breakdown have no JSON
    field (see hubbard_manifold_traces() and
    test_the_gamma_only_hubbard_force_contribution_is_not_zero()) and the
    half-sphere/full-sphere basis check right below is inherently a "did
    this code path run" smoke check, not a physics result -- both stay
    stdout-based on purpose, not as a migration gap.
    """
    input_path = write_input(directory, dataset, gamma_only, displacement, hubbard)
    completed = subprocess.run(
        [str(q3_bin), input_path.name],
        cwd=str(directory),
        capture_output=True,
        text=True,
        timeout=3600,
    )
    assert completed.returncode == 0, completed.stdout[-3000:]
    results = json.loads((directory / "out" / "si_results.json").read_text())
    assert results["converged"], (
        "the run did not converge, and an unconverged run reports no forces at all"
    )
    assert ("half-sphere" in completed.stdout) == gamma_only, (
        "this deck did not take the basis it was written for:\n" + completed.stdout[:2000]
    )
    return completed.stdout, results


def hubbard_manifold_traces(output):
    """Per-atom Hubbard occupation-matrix trace Tr[n]; genuinely stdout-only.

    FlavorResult has no field for this today (it's a per-atom SCF diagnostic
    printed by the reporter, not part of results.json) -- a real, documented
    gap, not an oversight.
    """
    matches = re.findall(r"DFT\+U: atom (\d+) Tr\[n\] = ([\d.]+)", output)
    assert matches, "no Tr[n] diagnostic in Q3 output"
    return [float(trace) for _, trace in matches[-2:]]


# ── Gate 1: the energy, against Q3's own full sphere ─────────────────────────


@pytest.fixture(scope="module", params=sorted(DATASETS))
def energy_pair(request, q3_bin, tmp_path_factory):
    """The same single Gamma point run on the half sphere and on the full one.

    Displaced off-symmetry, which matters here for a reason the force gate does
    not share: the occupation matrix is symmetrized over the point group in the
    full-sphere run and deliberately not in the Gamma-only one (the half sphere
    breaks the group, the same reason the density symmetrizer stands down). On
    the ideal diamond site that difference is real and worth 1.6e-7 Ry, which
    would swamp what this gate is measuring. Off-symmetry the group is trivial,
    the group average is the identity in both runs, and the comparison isolates
    the half-sphere bookkeeping.
    """
    dataset = request.param
    outputs = {}
    for gamma_only in (True, False):
        directory = tmp_path_factory.mktemp(f"{dataset}_{'gamma' if gamma_only else 'full'}")
        outputs[gamma_only] = run_q3(q3_bin, directory, dataset, gamma_only, BASE_DISPLACEMENT)
    return dataset, outputs[True], outputs[False]


def test_gamma_only_hubbard_energy_matches_the_full_sphere(energy_pair):
    dataset, (_, gamma_results), (_, full_sphere_results) = energy_pair
    gamma = gamma_results["total_energy_ry"]
    full_sphere = full_sphere_results["total_energy_ry"]
    print(f"\n  {dataset}: E(gamma) = {gamma:.9f}  E(full sphere) = {full_sphere:.9f} Ry")
    assert abs(gamma - full_sphere) < ENERGY_TOLERANCE_RYDBERG, (
        f"the Gamma-only DFT+U total energy differs from the same single k-point over the "
        f"full sphere by {abs(gamma - full_sphere):.3e} Ry, which is the half-sphere "
        "bookkeeping being wrong somewhere between the occupation matrix, S|phi> and the "
        "Hamiltonian term"
    )


def test_gamma_only_hubbard_energy_and_occupations_match_the_full_sphere(energy_pair):
    """
    E_U and Tr[n] on their own, because the total can hide them.

    E_U is ~0.31 Ry against a total of ~-14.2, so a proportional error in the
    occupation matrix that would be obvious here is only a fifth as visible in
    the total. Tr[n] is the quantity that says whether the projection found the
    shell at all -- it is what caught the missing overlap operator for PAW.

    E_U comes from results.json (hubbard_energy_ry); Tr[n] has no JSON field
    and stays stdout-based -- see hubbard_manifold_traces()'s docstring.
    """
    dataset, (gamma_stdout, gamma_results), (full_sphere_stdout, full_sphere_results) = energy_pair
    gamma = gamma_results["hubbard_energy_ry"]
    full_sphere = full_sphere_results["hubbard_energy_ry"]
    gamma_traces = hubbard_manifold_traces(gamma_stdout)
    full_sphere_traces = hubbard_manifold_traces(full_sphere_stdout)
    print(
        f"\n  {dataset}: E_U {gamma:.9f} vs {full_sphere:.9f} Ry   "
        f"Tr[n] {gamma_traces} vs {full_sphere_traces}"
    )
    assert abs(gamma - full_sphere) < ENERGY_TOLERANCE_RYDBERG, (
        f"E_U differs between the two bases by {abs(gamma - full_sphere):.3e} Ry"
    )
    for gamma_trace, full_sphere_trace in zip(gamma_traces, full_sphere_traces):
        assert abs(gamma_trace - full_sphere_trace) < 1.0e-3, (
            f"Tr[n] differs between the two bases: {gamma_traces} vs {full_sphere_traces}. "
            "Halving the projection scales Tr[n] by about a quarter, so this is what a "
            "missing 2*Re[...] looks like before it reaches the energy"
        )


def test_the_gamma_only_run_carries_a_hubbard_energy_at_all(energy_pair):
    """
    The premise of the gates above.

    Two runs that both computed no Hubbard term would agree perfectly and prove
    nothing. This is also the direct guard on the refusal that used to stand at
    SCF setup: DFT+U with a Gamma-only basis raised rather than ran.
    """
    dataset, (_, gamma_results), _ = energy_pair
    energy = gamma_results["hubbard_energy_ry"]
    print(f"\n  {dataset}: E_U = {energy:.9f} Ry")
    assert energy > 1.0e-3, f"the Gamma-only DFT+U energy is {energy:.3e} Ry, effectively zero"


# ── Gate 2: the force, against Q3's own Gamma-only energy ────────────────────


@pytest.fixture(scope="module", params=sorted(DATASETS))
def force_triple(request, q3_bin, tmp_path_factory):
    dataset = request.param
    outputs = {}
    for label, offset in (
        ("baseline", 0.0),
        ("plus", FINITE_DIFFERENCE_STEP),
        ("minus", -FINITE_DIFFERENCE_STEP),
    ):
        directory = tmp_path_factory.mktemp(f"{dataset}_force_{label}")
        outputs[label] = run_q3(q3_bin, directory, dataset, True, BASE_DISPLACEMENT + offset)
    return dataset, outputs


def test_gamma_only_hubbard_force_is_the_gradient_of_q3s_own_energy(force_triple):
    dataset, outputs = force_triple
    length = math.sqrt(sum(component**2 for component in FIRST_CELL_VECTOR_BOHR))
    unit_axis = [component / length for component in FIRST_CELL_VECTOR_BOHR]
    step_bohr = FINITE_DIFFERENCE_STEP * length

    _, baseline_results = outputs["baseline"]
    _, plus_results = outputs["plus"]
    _, minus_results = outputs["minus"]
    force = baseline_results["forces_ry_per_bohr"][1]
    analytic = sum(force[index] * unit_axis[index] for index in range(3))
    numerical = -(plus_results["total_energy_ry"] - minus_results["total_energy_ry"]) / (
        2.0 * step_bohr
    )

    assert abs(numerical) > 1e-3, "the displacement is not producing a resolvable force"
    ratio = analytic / numerical
    print(
        f"\n  {dataset}: F(analytic) = {analytic:+.8f}  -dE/dx = {numerical:+.8f} Ry/bohr  "
        f"ratio = {ratio:.6f}"
    )
    assert abs(ratio - 1.0) < FORCE_TOLERANCE, (
        f"the Gamma-only DFT+U force is not the gradient of Q3's own energy: analytic "
        f"{analytic:.8f} vs finite difference {numerical:.8f} Ry/bohr (ratio {ratio:.6f})"
    )


def test_the_gamma_only_hubbard_force_contribution_is_not_zero(force_triple):
    """
    The premise of the gate above, and a direct guard on the derivative overlaps.

    The Hubbard term is ~11% of the total force on this atom for the ultrasoft
    dataset. Dropping the Gamma treatment in the force path alone leaves the
    total within 9% of its own gradient while the Hubbard contribution itself
    falls by a factor of five -- a much sharper signal than the ratio, and the
    reason both are asserted.
    """
    dataset, outputs = force_triple
    baseline_stdout, _ = outputs["baseline"]
    matches = re.findall(
        r"atom\s+2 hubbard:\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)",
        baseline_stdout,
    )
    assert matches, "no Hubbard force contribution reported"
    magnitude = math.sqrt(sum(float(value) ** 2 for value in matches[-1]))
    print(f"\n  {dataset}: |F_hubbard| on atom 2 = {magnitude:.3e} Ry/bohr")
    assert magnitude > 3.0e-3, (
        f"the Gamma-only Hubbard force is {magnitude:.2e} Ry/bohr; with the half-sphere "
        "treatment removed from the force path it drops to ~1.7e-3"
    )
