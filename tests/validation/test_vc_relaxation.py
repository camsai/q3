"""
Validation tests for the Variable-Cell Relaxation workflow in Q3.

Exercises the ASE FrechetCellFilter/ExpCellFilter + BFGS workflow
(q3.workflows.vc_relax.run_vc_relax_workflow), which drives simultaneous
ionic + cell relaxation through Q3ASECalculator.

  1. Stress sign: a compressed cell must show positive physical pressure
     (pushing it to expand) and a stretched cell negative pressure -- the
     sign ASE's cell filters rely on to move the cell in the right
     direction at all.
  2. Convergence + QE cross-check: Si strained +2% isotropically
     recovers the unstrained equilibrium lattice constant and matches an
     independent QE vc-relax run from the identical starting geometry.
  3. Dispersion-aware relaxation: DFT-D3(BJ) forces/stress flow through
     the same cell-filter code path used by run_vc_relax_workflow.
  4. Target-pressure relaxation: the cell Q3 settles on under a 10 GPa
     external pressure, cross-checked against the volume an independent
     Birch-Murnaghan E(V) fit puts at that pressure. This is the
     capability QE states as `&CELL press`, which the QE parser refuses
     while naming this workflow instead.

Historical note: prior to this test file, run_vc_relax_workflow had three
compounding bugs that made it non-functional for any real calculation --
see the docstrings on the individual fixes in ase_calculator.py and
relax.py for details. Test 2 below is the end-to-end proof that the fixed
pipeline actually reproduces an independent QE vc-relax result.
"""

import math

import numpy as np
import pytest
import yaml
from ase.filters import FrechetCellFilter
from q3.calculators.ase_calculator import Q3ASECalculator
from q3.structure.ase_atoms import material_json_to_ase_atoms
from q3.workflows.equation_of_state import run_equation_of_state_workflow
from q3.workflows.vc_relax import run_vc_relax_workflow
from scipy.optimize import brentq

from helpers import (
    EXAMPLES_DIRECTORY,
    QE_AVAILABLE,
    QE_PW,
    birch_murnaghan_pressure_gpa,
    generate_qe_input,
    run_qe_vc_relax,
)

RY_TO_EV = 13.6058
BOHR_TO_ANGSTROM = 0.529177210903

SI_TOTAL_ENERGY_DIR = EXAMPLES_DIRECTORY / "total_energy--semiconductor--silicon"
VC_RELAX_DIR = EXAMPLES_DIRECTORY / "relaxation--semiconductor--silicon"

# The unstrained Si primitive lattice vector length (examples/data/materials/silicon.json),
# a MaterialsProject PBE-relaxed value -- the target vc-relax should recover from a
# +2% strained starting point.
SI_UNSTRAINED_A_PRIMITIVE_ANGSTROM = 3.866976


# ── 1. Stress sign convention (fast regression) ───────────────────────────────


def test_compressed_and_stretched_si_give_opposite_sign_physical_pressure(q3_command, tmp_path):
    """A compressed cell must show positive physical pressure (wants to expand);
    a stretched cell must show negative physical pressure (wants to contract).

    This is the sign ExpCellFilter/FrechetCellFilter's generalized cell force
    is built from (see stress_kbar_to_ase_voigt in ase_calculator.py); get it
    backwards and variable-cell relaxation moves the cell away from
    equilibrium instead of toward it.
    """

    material_path = EXAMPLES_DIRECTORY / "data" / "materials" / "silicon.json"

    def physical_pressure_ev_ang3(scale_factor, label):
        atoms = material_json_to_ase_atoms(material_path)
        atoms.set_cell(atoms.get_cell() * scale_factor, scale_atoms=True)
        atoms.calc = Q3ASECalculator(
            SI_TOTAL_ENERGY_DIR / "nc.yml",
            # mode='subprocess': this test deliberately gives Q3 a DIFFERENT
            # cell than the template YAML's own material file, which
            # mode='bindings' cannot do at all (see Q3ASECalculator's
            # bindings-mode cell guard).
            mode="subprocess",
            q3_binary=q3_command[0],
            workdir=tmp_path / f"stress_sign_{label}",
        )
        stress_ev_ang3 = atoms.get_stress()  # ASE Voigt order, tensile-positive convention
        return -(stress_ev_ang3[0] + stress_ev_ang3[1] + stress_ev_ang3[2]) / 3.0

    compressed_pressure = physical_pressure_ev_ang3(0.97, "compressed")
    stretched_pressure = physical_pressure_ev_ang3(1.03, "stretched")

    assert compressed_pressure > 0, (
        f"Compressed Si cell should have positive physical pressure (pushing it to "
        f"expand back out), got {compressed_pressure:.6f} eV/Å³"
    )
    assert stretched_pressure < 0, (
        f"Stretched Si cell should have negative physical pressure (pulling it back "
        f"in), got {stretched_pressure:.6f} eV/Å³"
    )


# ── 2. Convergence from strained cell + QE cross-check ────────────────────────


def test_vc_relax_recovers_equilibrium_from_2pct_strained_cell(q3_command, tmp_path):
    """Si strained +2% isotropically relaxes back to (near) the unstrained
    equilibrium lattice constant, and agrees with an independent QE vc-relax
    run started from the identical strained geometry.
    """

    result = run_vc_relax_workflow(
        VC_RELAX_DIR / "vc_relax_nc.yml",
        fmax=5.0e-3 * RY_TO_EV / BOHR_TO_ANGSTROM,
        steps=30,
        q3_binary=q3_command[0],
        workdir=tmp_path / "vc_relax_workdir",
        trajectory_path=tmp_path / "vc_relax.traj",
    )
    assert result["converged"], f"vc-relax did not converge within 30 steps: {result}"

    final_cell_angstrom = np.array(result["final_cell_angstrom"])
    # V = a_conventional^3 / 4 for the FCC primitive (rhombohedral) cell
    # (the same identity compute_celldm1 uses in reverse); a_conventional
    # itself is sqrt(2) times the ESSE/Q3 "primitive" a used throughout this
    # file (see q3.structure.ase_atoms.material_json_to_ase_atoms for that convention).
    # Isotropic recovery is expected here (no shear-driving anisotropy from a
    # purely isotropic starting strain on a cubic crystal).
    final_volume_angstrom3 = abs(np.linalg.det(final_cell_angstrom))
    final_a_conventional_angstrom = (final_volume_angstrom3 * 4.0) ** (1.0 / 3.0)
    final_a_primitive_angstrom = final_a_conventional_angstrom / math.sqrt(2.0)

    relative_deviation = (
        abs(final_a_primitive_angstrom - SI_UNSTRAINED_A_PRIMITIVE_ANGSTROM)
        / SI_UNSTRAINED_A_PRIMITIVE_ANGSTROM
    )
    assert relative_deviation < 0.005, (
        f"vc-relax from +2% strain recovered a={final_a_primitive_angstrom:.6f} Å, "
        f"expected within 0.5% of the unstrained a={SI_UNSTRAINED_A_PRIMITIVE_ANGSTROM:.6f} Å "
        f"(deviation: {relative_deviation * 100:.4f}%)"
    )

    if not QE_AVAILABLE:
        pytest.skip(f"QE LibXC build not available at {QE_PW} for cross-check")

    qe_input = generate_qe_input(
        case_name="si_nc_vc_relax",
        material_name="silicon_strained_2pct",
        pp_type="nc",
        pp_filename="Si.upf",
        ecutwfc=48.0,
        ecutrho=192.0,
        kgrid=[2, 2, 2],
        kshift=[1, 1, 1],
        number_of_bands=8,
        calculation="vc-relax",
        output_directory=str(tmp_path),
    )
    qe_final_volume_bohr3 = run_qe_vc_relax(qe_input)
    qe_final_volume_angstrom3 = qe_final_volume_bohr3 * BOHR_TO_ANGSTROM**3
    qe_final_a_conventional_angstrom = (qe_final_volume_angstrom3 * 4.0) ** (1.0 / 3.0)
    qe_final_a_primitive_angstrom = qe_final_a_conventional_angstrom / math.sqrt(2.0)

    relative_diff_vs_qe = abs(final_a_primitive_angstrom - qe_final_a_primitive_angstrom) / (
        qe_final_a_primitive_angstrom
    )
    assert relative_diff_vs_qe < 0.005, (
        f"Q3 vc-relax final a={final_a_primitive_angstrom:.6f} Å differs from QE's "
        f"{qe_final_a_primitive_angstrom:.6f} Å by {relative_diff_vs_qe * 100:.4f}% "
        f"(tolerance: 0.5%)"
    )


# ── 3. Dispersion-aware relaxation (wiring smoke test) ────────────────────────


def test_vc_relax_dispersion_forces_and_stress_flow_through_cell_filter(q3_command, tmp_path):
    """DFT-D3(BJ) forces/stress reach ASE's cell filter through the same code
    path run_vc_relax_workflow uses, and change the generalized cell force
    relative to a non-dispersion run.

    This is a wiring smoke test (cheap cutoff/k-grid), not a quantitative
    validation -- see test_qe_graphite_d3_dispersion.py for the
    production-accuracy D3 forces/stress validation against QE.
    """
    graphite_material_path = EXAMPLES_DIRECTORY / "data" / "materials" / "graphite.json"
    pseudopotential_path = (
        EXAMPLES_DIRECTORY / "data" / "pseudopotentials" / "nc" / "C.upf"
    ).resolve()

    def build_config(include_dispersion):
        config = {
            "material": str(graphite_material_path),
            "model": {"type": "dft", "subtype": "gga", "functional": "pbe"},
            "method": {
                "type": "pseudopotential",
                "subtype": "norm-conserving",
                "cutoffs": {"wavefunction": 30.0, "density": 120.0},
                "pseudopotentials": {"C": str(pseudopotential_path)},
                "scf": {
                    "convergence_threshold": 1.0e-6,
                    "smearing": {"type": "gaussian", "degauss": 0.02},
                    "kgrid": {"dimensions": [4, 4, 2], "shifts": [0, 0, 0]},
                },
            },
            "control": {"calculation": "scf", "outdir": "./output", "prefix": "graphite_smoke"},
        }
        if include_dispersion:
            config["model"]["dispersion"] = "d3bj"
        return config

    def cell_filter_forces(include_dispersion, label):
        config = build_config(include_dispersion)
        yaml_path = tmp_path / f"{label}.yml"
        with open(yaml_path, "w") as file_handle:
            yaml.safe_dump(config, file_handle)

        atoms = material_json_to_ase_atoms(graphite_material_path)
        atoms.calc = Q3ASECalculator(
            yaml_path,
            # Match run_vc_relax_workflow's mode exactly, since this test's
            # purpose is to validate that same cell-filter code path.
            mode="subprocess",
            q3_binary=q3_command[0],
            workdir=tmp_path / f"workdir_{label}",
        )
        cell_filter = FrechetCellFilter(atoms)
        return cell_filter.get_forces()

    forces_without_dispersion = cell_filter_forces(False, "no_d3")
    forces_with_dispersion = cell_filter_forces(True, "with_d3")

    assert np.all(np.isfinite(forces_without_dispersion))
    assert np.all(np.isfinite(forces_with_dispersion))
    assert forces_without_dispersion.shape == forces_with_dispersion.shape

    # The D3 correction must change the generalized cell-filter force
    # (positions 4:6, the cell degrees of freedom for this 4-atom cell) --
    # otherwise the dispersion stress never reached the filter at all.
    cell_force_difference = np.abs(forces_with_dispersion[4:] - forces_without_dispersion[4:]).max()
    assert cell_force_difference > 1e-6, (
        "DFT-D3(BJ) dispersion did not change the cell-filter generalized force; "
        "dispersion stress may not be flowing through Q3ASECalculator -> "
        "FrechetCellFilter."
    )


# ── 4. Target-pressure relaxation ─────────────────────────────────────────────

# 10 GPa: deep enough that the answer is nowhere near the zero-pressure cell
# (silicon contracts ~9 % in volume), shallow enough to stay well inside the
# diamond phase and inside the range a third-order Birch-Murnaghan fit at a
# fixed 48 Ry cutoff can be trusted over.
TARGET_PRESSURE_GIGAPASCAL = 10.0


def test_vc_relax_under_target_pressure_matches_the_equation_of_state_volume(q3_command, tmp_path):
    """Relaxing Si at 10 GPa must land on the volume the E(V) fit puts at 10 GPa.

    Two independent routes to the same physical quantity, sharing only the
    engine:

    * ``run_vc_relax_workflow(target_pressure_gpa=10)`` minimises the enthalpy
      E + P V variationally, driven by Q3's *analytic stress tensor*, and is
      free to change the cell's shape as well as its size;
    * ``run_equation_of_state_workflow`` fits Birch-Murnaghan to *total
      energies* over a volume scan, and P(V) = 10 GPa is solved on that fit
      here, never looking at a stress.

    A sign error in the pV term, a GPa-to-eV/A^3 slip, or an inconsistency
    between Q3's stress and its own energy derivative would separate them
    immediately. The zero-pressure leg is the control: it must reproduce the
    equation of state's equilibrium volume.

    The example's own 2x2x2 k-grid is kept deliberately -- both routes use it,
    so the comparison is between two methods rather than between two samplings,
    and the whole test stays around twenty seconds.
    """

    scf_input = SI_TOTAL_ENERGY_DIR / "nc.yml"
    # q3_command may be a full command (e.g. "mpirun -np 2 ./build/q3"); the
    # binary is its last token.
    q3_binary = q3_command[-1]

    equation_of_state = run_equation_of_state_workflow(
        scf_input,
        linear_scale_factors=(0.96, 0.97, 0.98, 0.99, 1.00, 1.01, 1.02),
        workdir=tmp_path / "equation_of_state",
        q3_binary=q3_binary,
    )
    equilibrium_volume = equation_of_state["volume"]["value"]
    target_volume = brentq(
        lambda volume: (
            birch_murnaghan_pressure_gpa(volume, equation_of_state) - TARGET_PRESSURE_GIGAPASCAL
        ),
        0.7 * equilibrium_volume,
        equilibrium_volume,
    )

    for target_pressure, predicted_volume in (
        (0.0, equilibrium_volume),
        (TARGET_PRESSURE_GIGAPASCAL, target_volume),
    ):
        result = run_vc_relax_workflow(
            scf_input,
            fmax=0.01,
            steps=30,
            target_pressure_gpa=target_pressure,
            q3_binary=q3_binary,
            workdir=tmp_path / f"vc_relax_{target_pressure:g}",
            trajectory_path=tmp_path / f"vc_relax_{target_pressure:g}.traj",
        )
        relaxed_volume = result["final_volume_angstrom3"]
        print(
            f"\n  P={target_pressure:4.1f} GPa: vc-relax V={relaxed_volume:.4f} A^3 "
            f"in {result['steps']} steps, equation of state V={predicted_volume:.4f} A^3, "
            f"difference {(relaxed_volume - predicted_volume) / predicted_volume * 100:+.3f} %"
        )
        assert result["converged"], f"vc-relax at {target_pressure} GPa did not converge: {result}"
        assert relaxed_volume == pytest.approx(predicted_volume, rel=0.005)

    # The point of a target pressure is that it actually compresses.
    assert target_volume < equilibrium_volume
