"""q3.workflows.relax — Geometry relaxation workflow using ASE BFGS.

Runs SCF via Q3 and uses ASE's BFGS optimizer to relax the geometry.

Example::

    from q3.workflows import run_relax_workflow

    result = run_relax_workflow(
        scf_input_path="examples/total_energy--semiconductor--silicon/nc.yml",
        fmax=0.01,
        steps=100,
        q3_binary="./build/q3",
    )
    print(result["final_energy_ry"], "Ry")
"""

from __future__ import annotations

import logging
from pathlib import Path

import yaml
from ase import Atoms
from ase.constraints import FixSymmetry
from ase.optimize import BFGS
from mat3ra.code.constants import Coefficients as _Coefficients

from q3.binary import ExecutionPolicy
from q3.calculators.ase_calculator import Q3ASECalculator
from q3.structure.ase_atoms import material_json_to_ase_atoms


def read_relaxation_input(
    scf_input_path: Path, use_symmetry: bool, calculator: Q3ASECalculator
) -> tuple[Atoms, dict]:
    """The starting atoms, carrying *calculator*, and the relaxation controls of an input.

    Shared by ``run_relax_workflow`` and ``run_vc_relax_workflow``.  A Q3 YAML
    input names its ESSE material and states its controls.  Any other input --
    a pw.x deck, with any ``ibrav`` -- is read by the engine alone: it runs once
    at its own structure, and the run's results give the material (``if_pos``
    as ``basis.constraints``) and, under ``provenance.relaxation``, the deck's
    ``forc_conv_thr`` and ``nstep``; that run is the relaxation's first step.
    Either way the atoms come from ``material_json_to_ase_atoms``.  With
    *use_symmetry*, they carry ``FixSymmetry`` unless a constraint already holds
    a coordinate fixed, which can break the starting symmetry.
    """
    if scf_input_path.suffix.lower() in (".yml", ".yaml"):
        with open(scf_input_path) as file_handle:
            config = yaml.safe_load(file_handle)
        atoms = material_json_to_ase_atoms(
            (scf_input_path.parent / config.get("material", "")).resolve()
        )
        atoms.calc = calculator
        control = config.get("control", {})
    else:
        atoms, results = calculator.run_input_as_written()
        relaxation = (results.provenance or {}).get("relaxation", {})
        control = {
            "relax_force_threshold": relaxation.get("force_threshold_ry_per_bohr"),
            "relax_max_steps": relaxation.get("maximum_steps"),
        }
        control = {key: value for key, value in control.items() if value is not None}

    if use_symmetry and not atoms.constraints:
        try:
            atoms.set_constraint(FixSymmetry(atoms))
        except ImportError:
            logging.warning(
                "FixSymmetry constraint could not be loaded; proceeding without symmetry constraints."
            )
    return atoms, control


def run_relax_workflow(
    scf_input_path: str | Path,
    *,
    fmax: float | None = None,
    steps: int | None = None,
    maxstep_angstrom: float = 0.04,
    bfgs_alpha: float = 30.0,
    use_symmetry: bool = True,
    mode: str = "auto",
    q3_binary: str | Path | None = None,
    execution_policy: ExecutionPolicy | None = None,
    trajectory_path: str | Path | None = None,
    workdir: str | Path | None = None,
    capture_output: bool = True,
) -> dict:
    """Relax ionic positions using ASE BFGS + Q3 forces.

    Parameters
    ----------
    scf_input_path:
        Path to a Q3 YAML input file for the SCF calculation, or a pw.x relax
        deck (any ``ibrav``), whose ``forc_conv_thr``, ``nstep`` and ``if_pos``
        flags are read the same way.

        This same file may also carry ``control.relax_max_steps`` and
        ``control.relax_force_threshold`` (the ``calculation: relaxation``
        YAML convention every ``examples/relaxation--*/`` file uses) --
        when *fmax* or *steps* is not passed explicitly, they default to
        whatever this file states. The C++ engine itself never reads
        ``calculation: relaxation`` directly (see ``create_flavor()`` in
        ``src/layer_4_computation/flavors/factory.hpp`` -- multi-step
        workflows are Python-orchestrated by design); this is what
        actually consumes those fields. ``control.relax_energy_threshold``
        is schema-accepted and set by every example file, but this
        workflow has no energy-based convergence criterion (ASE ``BFGS``
        is run with only *fmax*/*steps*) -- a warning is logged, not
        silently ignored, when a file sets it.
    fmax:
        Maximum force on any atom for convergence criterion in eV/Å.
        Defaults to *scf_input_path*'s own ``control.relax_force_threshold``
        (Ry/bohr, converted to eV/Å) if present, else 0.02.
    steps:
        Maximum number of optimization steps. Defaults to *scf_input_path*'s
        own ``control.relax_max_steps`` if present, else 100.
    maxstep_angstrom:
        Maximum displacement per BFGS step in Å.  Choose this to be smaller
        than the expected atom displacement to equilibrium.  For stiff
        covalent systems (Si 3×3×3 near equilibrium, ~0.013 Å away), use
        0.005–0.01 Å.  For soft metals (Cu 3×3×3) 0.1–0.2 Å is appropriate.
    bfgs_alpha:
        Initial guess for the Hessian diagonal (curvature of energy surface).
        Lower values produce more aggressive initial steps.  The default of
        30.0 was tuned on Si NC 3×3×3 and converges in 3 BFGS steps (vs 4
        with the ASE default of 70.0).  Values below ~20 can cause
        oscillations; values above ~50 are overly conservative.
    use_symmetry:
        If True (default), apply the FixSymmetry constraint to the atoms.
    mode:
        Execution mode for the calculator ('auto', 'subprocess', or 'bindings').
    q3_binary:
        Path to the binary each step runs: ``q3``, ``q3_mpi`` or a GPU binary.
    execution_policy:
        The execution policy every step starts the binary with: ranks,
        k-point pools, the launcher (see :class:`q3.binary.ExecutionPolicy`).
    trajectory_path:
        Where to write the ASE ``.traj`` trajectory file.  Defaults to
        ``relax.traj`` in the current directory.
    workdir:
        Directory for Q3 step inputs/outputs.
    capture_output:
        Whether to capture binary stdout/stderr.

    Returns
    -------
    dict
        Contains:
        - ``"converged"`` — ``True`` if ``fmax`` criterion was met.
        - ``"steps"`` — number of optimization steps taken.
        - ``"final_energy_ry"`` — final total energy in Rydberg.
        - ``"final_forces_ry_per_bohr"`` — forces at final geometry.
    """
    scf_input_path = Path(scf_input_path).expanduser().resolve()

    if trajectory_path is None:
        if workdir is not None:
            trajectory_path = Path(workdir) / "relax.traj"
        else:
            trajectory_path = Path("relax.traj")
    else:
        trajectory_path = Path(trajectory_path)
    trajectory_path = trajectory_path.expanduser().resolve()
    trajectory_path.parent.mkdir(parents=True, exist_ok=True)

    calculator = Q3ASECalculator(
        scf_input_path,
        mode=mode,
        q3_binary=q3_binary,
        execution_policy=execution_policy,
        workdir=workdir,
        capture_output=capture_output,
    )
    atoms, input_control = read_relaxation_input(scf_input_path, use_symmetry, calculator)

    # ── Defaults from scf_input_path itself ────────────────────────────────────
    # The calculation: relaxation YAML convention (every
    # examples/relaxation--*/ file) states relax_max_steps and
    # relax_force_threshold directly -- read them here so those files are
    # actually runnable through this workflow with no other arguments,
    # rather than carrying fields nothing consumes. Explicit keyword
    # arguments always win over the file. Matches the treatment applied to
    # run_band_structure_workflow's band_path/band_path_points_per_segment.
    if fmax is None:
        relax_force_threshold_ry_per_bohr = input_control.get("relax_force_threshold")
        fmax = (
            0.02
            if relax_force_threshold_ry_per_bohr is None
            else relax_force_threshold_ry_per_bohr * _Coefficients.ry_bohr_to_eV_A
        )
    if steps is None:
        steps = input_control.get("relax_max_steps", 100)
    if input_control.get("relax_energy_threshold") is not None:
        logging.warning(
            "%s sets control.relax_energy_threshold, but run_relax_workflow() has "
            "no energy-based convergence criterion (ASE BFGS is run with only "
            "fmax/steps) -- this value has no effect.",
            scf_input_path,
        )

    # Full BFGS builds the complete inverse Hessian from force/displacement
    # history, matching QE's trust-radius BFGS in step efficiency.  The alpha
    # parameter controls the initial Hessian diagonal — lower values produce
    # more aggressive first steps.  alpha=30 was empirically tuned: it
    # converges Si NC 3×3×3 in 3 steps (vs 4 with the ASE default of 70),
    # matching QE's 2-step BFGS in total force evaluations (3 vs 3).
    optimizer = BFGS(
        atoms,
        trajectory=str(trajectory_path),
        maxstep=maxstep_angstrom,
        alpha=bfgs_alpha,
    )
    converged = optimizer.run(fmax=fmax, steps=steps)

    final_energy_ev = atoms.get_potential_energy()
    final_forces_ev_per_angstrom = atoms.get_forces()

    return {
        "converged": converged,
        "steps": optimizer.nsteps,
        "final_energy_ry": final_energy_ev / _Coefficients.RYDBERG,
        "final_forces_ry_per_bohr": (
            final_forces_ev_per_angstrom / _Coefficients.ry_bohr_to_eV_A
        ).tolist(),
    }
