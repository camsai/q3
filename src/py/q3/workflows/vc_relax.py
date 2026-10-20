"""q3.workflows.vc_relax — Variable-cell relaxation workflow using ASE.

Runs simultaneous ionic + cell relaxation using ASE's ``FrechetCellFilter``
and the ``BFGS`` optimizer.

Relaxation to a **target pressure** is supported through *target_pressure_gpa*:
this is Q3's answer to Quantum ESPRESSO's ``&CELL press``, which the QE input
parser refuses precisely because it is this workflow's parameter and not an
engine setting (``src/io/input/parser/qe.hpp``).

Example::

    from q3.workflows import run_vc_relax_workflow

    result = run_vc_relax_workflow(
        scf_input_path="examples/total_energy--semiconductor--silicon/nc.yml",
        fmax=0.01,
        q3_binary="./build/q3",
    )
    print("Relaxed lattice constant:", result["final_cell_angstrom"])
"""

from __future__ import annotations

from pathlib import Path

from ase.filters import FrechetCellFilter
from ase.optimize import BFGS
from mat3ra.code.constants import Coefficients as _Coefficients

from q3.calculators.ase_calculator import Q3ASECalculator
from q3.properties.stress import pressure_gpa_to_ev_per_angstrom3
from q3.workflows.relax import read_relaxation_input


def run_vc_relax_workflow(
    scf_input_path: str | Path,
    *,
    fmax: float = 0.02,
    steps: int = 100,
    use_symmetry: bool = True,
    target_pressure_gpa: float = 0.0,
    q3_binary: str | Path | None = None,
    launcher: str | list[str] = (),
    trajectory_path: str | Path | None = None,
    workdir: str | Path | None = None,
    capture_output: bool = True,
) -> dict:
    """Relax ionic positions and cell shape simultaneously.

    Uses ASE's ``FrechetCellFilter`` to relax both the
    atomic positions and the cell vectors so that the full stress tensor
    (not just its trace) vanishes.  The cell volume is NOT held fixed: the
    filter drives the system to the zero-external-pressure equilibrium
    volume, so a strained starting cell will expand or contract as needed.

    Parameters
    ----------
    scf_input_path:
        Path to a Q3 YAML input file, or a pw.x vc-relax deck (any ``ibrav``;
        pass its ``press``, divided by 10, as *target_pressure_gpa*).
    fmax:
        Convergence criterion: maximum force/stress component in eV/Å.
    steps:
        Maximum number of BFGS steps.
    use_symmetry:
        If True (default), apply the FixSymmetry constraint to the atoms
        before wrapping them in the cell filter -- constrains both ionic
        displacements and cell changes to preserve the starting space group,
        speeding up convergence for symmetric starting cells the same way
        ``run_relax_workflow``'s identical parameter already does for
        ionic-only relaxation.
    target_pressure_gpa:
        External hydrostatic pressure to relax against, in GPa, positive for
        compression (QE states the same quantity as ``&CELL press``, in kbar:
        divide by 10).  The default of 0.0 relaxes to vanishing stress, as
        before.  A non-zero value hands the cell filter the enthalpy
        H = E + P V to minimize instead of E, so the converged cell is the one
        whose own stress balances *P* rather than the one whose stress
        vanishes.

        Two consequences worth knowing before reading a number out of this.
        The pV term breaks ASE's energy/force consistency (the filter's reported
        energy is the enthalpy while ``final_energy_ry`` below stays the plain
        DFT total energy), which is expected and does not affect the BFGS
        gradient it actually descends.  And the stress Q3 computes at fixed
        plane-wave cutoff carries the usual Pulay error, so at high target
        pressures the balance point is only as trustworthy as the cutoff --
        cross-check against the pressure of
        ``q3.workflows.equation_of_state``'s E(V) fit, which reaches the same
        answer through energies alone and is biased differently.
    q3_binary:
        Path to the binary each step runs: ``q3``, ``q3_mpi`` or a GPU binary.
    launcher:
        Command each step runs the binary under, e.g. ``"srun -n 64"``.
    trajectory_path:
        ASE trajectory output path.
    workdir:
        Directory for Q3 per-step files.
    capture_output:
        Whether to capture binary stdout/stderr.

    Returns
    -------
    dict
        Contains ``"converged"``, ``"steps"``, ``"final_energy_ry"``,
        ``"final_cell_angstrom"`` (3×3 cell matrix in Angström),
        ``"final_volume_angstrom3"`` and ``"target_pressure_gpa"``.
    """
    scf_input_path = Path(scf_input_path).expanduser().resolve()
    trajectory_path = Path(trajectory_path or "vc_relax.traj").expanduser().resolve()

    calculator = Q3ASECalculator(
        scf_input_path,
        # Force subprocess mode: bindings mode's SCFSolver has no API to
        # update the cell mid-run (see Q3ASECalculator.calculate()'s
        # bindings branch), so 'auto' would silently pick bindings whenever
        # q3_core is importable and re-solve the ORIGINAL cell on every
        # step regardless of what the cell filter does.
        mode="subprocess",
        q3_binary=q3_binary,
        launcher=launcher,
        workdir=workdir,
        capture_output=capture_output,
    )
    atoms, _ = read_relaxation_input(scf_input_path, use_symmetry, calculator)

    # ASE states the external pressure in its own eV/A^3 unit system and adds
    # +P*V to the energy, so a positive value compresses -- the same sign
    # convention as QE's press and as Q3's own reported pressure.
    cell_filter = FrechetCellFilter(
        atoms, scalar_pressure=pressure_gpa_to_ev_per_angstrom3(target_pressure_gpa)
    )
    optimizer = BFGS(cell_filter, trajectory=str(trajectory_path))
    converged = optimizer.run(fmax=fmax, steps=steps)

    final_energy_ev = atoms.get_potential_energy()

    return {
        "converged": converged,
        "steps": optimizer.nsteps,
        "final_energy_ry": final_energy_ev / _Coefficients.RYDBERG,
        "final_cell_angstrom": atoms.get_cell().tolist(),
        "final_volume_angstrom3": float(atoms.get_volume()),
        "target_pressure_gpa": float(target_pressure_gpa),
    }
