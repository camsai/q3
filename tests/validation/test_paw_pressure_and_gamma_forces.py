"""
PAW stress and Gamma-only forces against Q3's own energy, on a committed dataset.

Three identities that a defect in a PAW derivative breaks while the total
energy stays right -- which is how such a defect survives every comparison of
energies, and why the reference here is Q3's own energy rather than another
code's:

1. **The pressure is -dE/dV**, taken on a fixed plane-wave basis.
2. **The stress does not depend on the choice of origin.** Translating every
   atom by the same vector is not a physical change.
3. **At a single unshifted Gamma point the force is -dE/dx.** That is where Q3
   stores only half of G-space, so every force sum has to reconstruct the
   other half -- the one sampling no other PAW finite-difference gate takes,
   and exactly the one a molecule in a box runs on.

Silicon on ``examples/data/pseudopotentials/paw/Si.upf`` throughout: a PAW
dataset with a partial core charge, which the origin gate needs, because the
defect it exists for lived in the core-correction stress. Nothing here needs a
reference code or a dataset the repository does not ship, so all three run
wherever Q3 is built.

Every number below was measured on these inputs. A number "with the defect
restored" comes from rebuilding Q3 with that one change put back; the energy
is the same either way, so only the analytic side of a gate moves.

PAW forces away from Gamma are gated by the SiC DFT+U central difference in
``test_paw_forces_and_stress.py``; ultrasoft and norm-conserving Gamma-only
forces by ``test_gamma_only_hubbard.py``.

Run:
    .venv/bin/pytest tests/validation/test_paw_pressure_and_gamma_forces.py -v --q3-bin ./build/q3
"""

import json
import math
from pathlib import Path

import pytest

from helpers import BOHR_RADIUS_ANGSTROM, load_results_json, run_q3

REPOSITORY = Path(__file__).resolve().parents[2]
SILICON_MATERIAL = REPOSITORY / "examples" / "data" / "materials" / "silicon.json"
PAW_DATASET = REPOSITORY / "examples" / "data" / "pseudopotentials" / "paw" / "Si.upf"

RYDBERG_PER_BOHR_CUBED_TO_KILOBAR = 147105.078

COMMITTED_ORIGIN = ((0.875, 0.875, 0.875), (0.125, 0.125, 0.125))
"""The basis as ``silicon.json`` states it, with the inversion centre at the origin."""

SHIFTED_ORIGIN = ((0.0, 0.0, 0.0), (0.25, 0.25, 0.25))
"""The same crystal translated by an eighth of the cell along each axis."""

WAVEFUNCTION_CUTOFF_RYDBERG = 30.0
DENSITY_CUTOFF_RYDBERG = 240.0
"""
Far from converged, and none of the three identities needs it to be.

The density cutoff is chosen for the origin gate: it gives a 40^3 density grid,
on which an eighth of the cell is exactly five grid steps, so both origins
sample the same discretized crystal.
"""

STRESS_KPOINT_GRID = 4
GAMMA_ONLY_KPOINT_GRID = 1
"""
Unshifted in both cases. A 1x1x1 grid with no shift is what switches Q3 to the
half-sphere storage, and the Gamma gate asserts that it did.
"""

CONVERGENCE_THRESHOLD = "1.0e-12"
"""
The analytic force and stress are evaluated once, from the final state, so
their error is first order in the SCF residual, where the energy's is second
order. At 1e-10 every gate below reads the same to within its noise (pressure
gap -0.0059 against -0.0053 kbar, origin difference 7e-4 against 2e-4 kbar,
Gamma ratio unchanged); 1e-12 keeps that floor far below each bound.
"""

VOLUME_STEP = 0.0025
"""
The fractional volume change on each side of the pressure's central difference.

Its own truncation error, (h^2/6) V^2 d^2P/dV^2, measured 0.081, 0.020, 0.0053
and 0.0010 kbar at steps of 1%, 0.5%, 0.25% and 0.1%: it scales as h^2 and
extrapolates to 2e-4 kbar at zero step, which is the analytic pressure's own SCF
noise (it moves by 4e-4 kbar between mixing settings). A quarter of a percent
keeps the truncation an order of magnitude under the bound, while the energy
difference, 5e-5 Ry, stays far above the SCF noise.
"""

PRESSURE_TOLERANCE_KILOBAR = 0.05
"""
|P(analytic) - (-dE/dV)|.

Measured 0.0052 to 0.0054 kbar over five runs, nearly all of it VOLUME_STEP's
truncation. Absolute rather than relative, because this pressure is a 5.4 kbar
residual between terms of up to 3000 kbar. The smallest of those terms is the
augmentation stress, at -137 kbar; with its off-diagonal projector pairs counted
once instead of twice -- a slip in the triangle storage it is built on -- the
gap opens to 4.36 kbar, 87 times this bound.
"""

ORIGIN_TOLERANCE_KILOBAR = 0.1
"""
|P(committed origin) - P(shifted origin)|.

Measured 1.7e-4 to 4.3e-4 kbar over five runs. What is left is SCF convergence,
not the grid: the two origins differ by 3e-5 to 5e-4 kbar across cutoffs and
mixing settings, also when both are given identical grids. A density grid the
shift does not land on (36^3, at 200 Ry) adds the egg-box effect, 0.025 kbar.
The bound is four times that, so it does not lean on the grid staying
commensurate, and 8600 times below the defect it exists for (see the test).
"""

GAMMA_BASE_DISPLACEMENT = 0.020
"""
The second atom's offset from its site along a1, in crystal units, in every run.

A diamond site is symmetric, so a central difference about the ideal position
vanishes identically and would measure nothing.
"""

GAMMA_FINITE_DIFFERENCE_STEP = 0.002
"""The crystal-coordinate step taken either side of that offset, 0.0146 bohr."""

FIRST_CELL_VECTOR_DIRECTION = (-1.0 / math.sqrt(2.0), 0.0, 1.0 / math.sqrt(2.0))
"""
The unit vector along a1 = alat/2 (-1, 0, 1), the layout of Bravais index 2.

Spelled out because projecting the force onto the wrong axis makes a correct
force look badly wrong, and nothing in that failure points at the axis.
"""

GAMMA_FORCE_TOLERANCE = 1.0e-3
"""
|F(analytic) / F(finite difference) - 1|.

Measured 9.4e-5. It is not the step's truncation -- the ratio reads 0.999912,
0.999906 and 0.999879 at steps of 0.001, 0.002 and 0.004, converging to a
difference of 7e-6 Ry/bohr rather than to zero -- and not the half sphere,
whose force matches an explicit full-sphere Gamma run to 7e-9 Ry/bohr, so the
full sphere carries the same residual. The defect it exists for is in the test.
"""

INPUT_TEMPLATE = """material: {material}
model: {{type: dft, subtype: gga, functional: pbe}}
method:
  type: pseudopotential
  subtype: paw
  cutoffs: {{wavefunction: {wavefunction_cutoff:.12f}, density: {density_cutoff:.12f}}}
  pseudopotentials: {{Si: {pseudopotential}}}
  scf:
    convergence_threshold: {convergence_threshold}
    max_iterations: 300
    mixing: {{mode: broyden, beta: 0.7}}
    kgrid: {{dimensions: [{grid}, {grid}, {grid}], shifts: [0, 0, 0]}}
control: {{calculation: scf, outdir: {outdir}, prefix: si}}
"""


def write_input(directory, positions, kpoint_grid, volume_fraction=0.0):
    """
    Silicon at `positions`, its cell scaled to (1 + volume_fraction) V on an unchanged basis.

    Both cutoffs scale as V^(-2/3), exactly as every |G|^2 does, so each
    G-vector inside a cutoff stays inside it: the plane-wave and density bases
    and both FFT grids are the same at every volume.
    """
    material = json.loads(SILICON_MATERIAL.read_text())
    length_scale = (1.0 + volume_fraction) ** (1.0 / 3.0)
    for parameter in ("a", "b", "c"):
        material["lattice"][parameter] *= length_scale
    for coordinate, position in zip(material["basis"]["coordinates"], positions):
        coordinate["value"] = list(position)
    material_path = directory / "silicon.json"
    material_path.write_text(json.dumps(material))

    basis_scale = (1.0 + volume_fraction) ** (-2.0 / 3.0)
    input_path = directory / "si.yml"
    input_path.write_text(
        INPUT_TEMPLATE.format(
            material=material_path,
            wavefunction_cutoff=WAVEFUNCTION_CUTOFF_RYDBERG * basis_scale,
            density_cutoff=DENSITY_CUTOFF_RYDBERG * basis_scale,
            pseudopotential=PAW_DATASET,
            convergence_threshold=CONVERGENCE_THRESHOLD,
            grid=kpoint_grid,
            outdir=directory / "out",
        )
    )
    return input_path


def run_silicon(q3_command, directory, positions, kpoint_grid, volume_fraction=0.0):
    """Run one SCF in `directory` and return (console output, results.json)."""
    directory.mkdir(parents=True, exist_ok=True)
    input_path = write_input(directory, positions, kpoint_grid, volume_fraction)
    completed = run_q3(q3_command, input_path, working_directory=directory, timeout=1800)
    assert completed.returncode == 0, (
        f"q3 exited with {completed.returncode}\nstdout:\n{completed.stdout[-3000:]}\n"
        f"stderr:\n{completed.stderr[-3000:]}"
    )
    results = load_results_json(directory / "out", "si")
    assert results["converged"], "the SCF did not converge, so its derivatives mean nothing"
    return completed.stdout, results


def primitive_vector_length_bohr():
    """|a1| = |a2| = |a3| of ``silicon.json``: the material states it directly, in angstrom."""
    material = json.loads(SILICON_MATERIAL.read_text())
    return material["lattice"]["a"] / BOHR_RADIUS_ANGSTROM


def cell_volume_bohr_cubed():
    """
    The volume Q3 builds from ``silicon.json``, in bohr^3.

    An FCC primitive vector of length `a` belongs to a conventional cube of edge
    a*sqrt(2), which holds four primitive cells.
    """
    conventional_edge = primitive_vector_length_bohr() * math.sqrt(2.0)
    return conventional_edge**3 / 4.0


@pytest.fixture(scope="module")
def stress_baseline(q3_command, tmp_path_factory):
    """Equilibrium silicon at the committed origin; both stress gates start here."""
    directory = tmp_path_factory.mktemp("paw_stress") / "baseline"
    _, results = run_silicon(q3_command, directory, COMMITTED_ORIGIN, STRESS_KPOINT_GRID)
    return results


def test_paw_pressure_is_the_volume_derivative_of_q3s_own_energy(
    q3_command, stress_baseline, tmp_path
):
    """
    P = -dE/dV, by a central difference of Q3's own energy on a fixed basis.

    A plane-wave stress is the strain derivative at a fixed set of G-vectors.
    E(V) at a fixed *cutoff* regenerates the basis at each volume, so it also
    carries the Pulay term the analytic stress leaves out: at this cutoff a
    fixed-cutoff difference sits 0.20 kbar from the analytic pressure, a basis
    error that would read as a stress error. Scaling the cutoffs with the cell
    (`write_input`) removes the term at any cutoff, which is what lets this run
    so far below convergence.
    """
    analytic = stress_baseline["pressure_kbar"]
    energies = {}
    for sign in (+1, -1):
        _, results = run_silicon(
            q3_command,
            tmp_path / f"volume{sign:+d}",
            COMMITTED_ORIGIN,
            STRESS_KPOINT_GRID,
            volume_fraction=sign * VOLUME_STEP,
        )
        energies[sign] = results["total_energy_ry"]

    numerical = (
        -(energies[+1] - energies[-1])
        / (2.0 * VOLUME_STEP * cell_volume_bohr_cubed())
        * RYDBERG_PER_BOHR_CUBED_TO_KILOBAR
    )
    print(
        f"\n  P(analytic) = {analytic:+.5f}  -dE/dV = {numerical:+.5f} kbar  "
        f"difference {analytic - numerical:+.2e}"
    )
    assert abs(analytic - numerical) < PRESSURE_TOLERANCE_KILOBAR, (
        f"the PAW pressure is not the volume derivative of Q3's own energy: analytic "
        f"{analytic:.5f} vs finite difference {numerical:.5f} kbar"
    )


def test_stress_does_not_depend_on_the_choice_of_origin(q3_command, stress_baseline, tmp_path):
    """
    Translating the basis is not a physical change, so the stress must not move.

    This is the invariant that caught the core-correction phase error. That term
    pairs V*_xc(G) against the core density, which is built with S(G) = sum
    e^{-iG.tau}; the stress used the conjugate sign, and under tau -> tau + t
    the two same-signed factors leave a residual e^{+2iG.t}. With the conjugate
    put back, the shifted origin's pressure is 865.32 kbar against the committed
    origin's 5.45, while the energies still agree.

    The committed origin alone could never show it. With the inversion centre
    at the origin S(G) is real, so S and its conjugate are the same number and
    the defect leaves the committed pressure exactly where it was -- which is
    also why the volume-derivative gate above passes with it in place.
    """
    reference = stress_baseline["pressure_kbar"]
    _, shifted = run_silicon(
        q3_command, tmp_path / "shifted_origin", SHIFTED_ORIGIN, STRESS_KPOINT_GRID
    )
    shifted_pressure = shifted["pressure_kbar"]
    print(
        f"\n  P at (7/8, 1/8) = {reference:+.5f}  P at (0, 1/4) = {shifted_pressure:+.5f} kbar  "
        f"difference {shifted_pressure - reference:+.2e}"
    )
    assert abs(shifted_pressure - reference) < ORIGIN_TOLERANCE_KILOBAR, (
        f"the stress depends on the choice of origin: {reference:.5f} vs {shifted_pressure:.5f} "
        f"kbar for the same crystal, whose energies are {stress_baseline['total_energy_ry']:.10f} "
        f"and {shifted['total_energy_ry']:.10f} Ry"
    )


def test_gamma_only_forces_are_the_gradient_of_q3s_own_energy(q3_command, tmp_path):
    """
    The Gamma-only half sphere, where every PAW force term once had a defect.

    At a single unshifted Gamma point Q3 keeps one G of each +-G pair, and a
    G-sum over that list is half the full-sphere sum whenever the summand is
    even under G -> -G. The local, core-correction, Ewald and augmentation
    forces all are, and none of them carried the factor of two; the nonlocal
    term instead double counted G = 0, which Q3's half sphere includes, in
    `2*Re[<beta|psi>_half]`. With all five restored the ratio below reads
    0.716; the nonlocal double count alone gives 0.817 and the augmentation
    factor alone 1.062. The smallest single piece, the core-correction factor,
    is worth 2.3% of this force by its share of it (ratio 1.023), still 23 times
    the bound.

    The energies were right throughout -- only the derivatives were wrong -- so
    a relaxation would have converged to the wrong geometry silently.
    """
    step = GAMMA_FINITE_DIFFERENCE_STEP
    outputs = {}
    for label, offset in (("baseline", 0.0), ("plus", +step), ("minus", -step)):
        displaced = (0.125 + GAMMA_BASE_DISPLACEMENT + offset, 0.125, 0.125)
        outputs[label] = run_silicon(
            q3_command, tmp_path / label, (COMMITTED_ORIGIN[0], displaced), GAMMA_ONLY_KPOINT_GRID
        )

    baseline_output, baseline = outputs["baseline"]
    # The one fact only the console states: which basis the run took.
    assert "half-sphere" in baseline_output, (
        "this input was meant to exercise the Gamma-only half sphere:\n" + baseline_output[:2000]
    )

    force = baseline["forces_ry_per_bohr"][1]
    analytic = sum(force[index] * FIRST_CELL_VECTOR_DIRECTION[index] for index in range(3))
    step_bohr = step * primitive_vector_length_bohr()
    numerical = -(
        outputs["plus"][1]["total_energy_ry"] - outputs["minus"][1]["total_energy_ry"]
    ) / (2.0 * step_bohr)

    assert abs(numerical) > 1.0e-3, "the displacement is not producing a resolvable force"
    ratio = analytic / numerical
    print(
        f"\n  Gamma-only: F(analytic) = {analytic:+.8f}  -dE/dx = {numerical:+.8f} Ry/bohr  "
        f"ratio = {ratio:.6f}"
    )
    assert abs(ratio - 1.0) < GAMMA_FORCE_TOLERANCE, (
        f"the Gamma-only PAW force is not the gradient of Q3's own energy: analytic "
        f"{analytic:.8f} vs finite difference {numerical:.8f} Ry/bohr (ratio {ratio:.6f})"
    )
