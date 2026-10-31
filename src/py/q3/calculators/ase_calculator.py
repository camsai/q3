"""q3.calculators.ase_calculator — ASE Calculator wrapping Q3.

``Q3ASECalculator`` implements the ASE Calculator interface so that Q3 can
be used as the DFT engine inside ASE workflows (geometry optimization, MD,
NEB, etc.).

It operates in Mode A (subprocess): each call to ``get_forces()`` / ``get_stress()``
spawns a fresh ``q3`` process with the current atomic positions written to a
temporary YAML input file.

Example::

    from ase.io import read
    from ase.optimize import BFGS
    from q3.calculators import Q3ASECalculator

    atoms = read("si.cif")
    atoms.calc = Q3ASECalculator(
        scf_input_path="examples/total_energy--semiconductor--silicon/nc.yml",
        q3_binary="./build/q3",
    )
    opt = BFGS(atoms, trajectory="si_relax.traj")
    opt.run(fmax=0.01)
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from ase.calculators.calculator import Calculator, all_changes
from ase.io import espresso
from mat3ra.code.constants import Coefficients as _Coefficients

from q3.binary import ExecutionPolicy, execution_policy_or_serial, find_binary
from q3.errors import Q3RuntimeError
from q3.io.output_writer import (
    extract_total_energy,
    raise_if_derivative_incomplete,
    raise_if_not_converged,
)
from q3.io.results.classes import Results
from q3.structure.ase_atoms import ase_atoms_to_material_json, material_json_to_ase_atoms

try:
    import q3_core
except ImportError:  # the compiled extension is not built: "auto" runs the binary
    q3_core = None

# Re-exported for the consumers that imported it from here before it moved to
# q3.io.output_writer, where every reader of a Q3 result can reach it.
__all__ = [
    "Q3ASECalculator",
    "raise_if_derivative_incomplete",
    "stress_kbar_to_ase_voigt",
]

# ASE uses eV and Angstrom; Q3 uses Ry and Bohr.
# All conversion factors come from the canonical mat3ra.code.constants source.
_RY_TO_EV = _Coefficients.RYDBERG
_RY_PER_BOHR_TO_EV_PER_ANG = _Coefficients.ry_bohr_to_eV_A
# 1 kbar = 0.1 GPa.
_KBAR_TO_EV_PER_ANG3 = 0.1 / _Coefficients.EV_PER_ANGSTROM3_TO_GPA


def stress_kbar_to_ase_voigt(stress_kbar: list) -> Any:
    """Convert Q3's 3x3 stress tensor (kbar) to ASE's Voigt-order stress (eV/Å³).

    Q3 reports the stress tensor in the Nielsen-Martin convention,
    sigma = -(1/Omega) dE/dstrain (see stress_tensor/planewave/init.hpp),
    under which a compressed cell has POSITIVE diagonal stress (it matches
    the physical pressure, like Quantum ESPRESSO's convention).  ASE instead
    uses sigma = +(1/Omega) dE/dstrain (see ase/stress.py's elasticity-tensor
    docstring), under which a compressed cell has NEGATIVE diagonal stress
    ("tensile positive").  The two conventions are exact negatives of each
    other, so the sign must be flipped here -- without it, ASE's cell filters
    (ExpCellFilter/FrechetCellFilter) would compute the generalized cell
    force with the wrong sign and drive variable-cell relaxation to expand
    under compression and contract under tension, i.e. away from equilibrium
    instead of toward it.
    """
    stress_matrix = np.array(stress_kbar)
    voigt_kbar = np.array(
        [
            stress_matrix[0][0],
            stress_matrix[1][1],
            stress_matrix[2][2],
            stress_matrix[1][2],
            stress_matrix[0][2],
            stress_matrix[0][1],
        ]
    )
    return -voigt_kbar * _KBAR_TO_EV_PER_ANG3


class Q3ASECalculator(Calculator):
    """ASE Calculator that runs Q3 via subprocess or the nanobind ``q3_core`` bindings.

    Parameters
    ----------
    scf_input_path:
        Path to a Q3 YAML input file or a pw.x deck (any ``ibrav``).  Every
        single-point calculation runs this file unchanged, in its own
        directory, with ``--material`` naming the current structure: the
        engine keeps the file's settings and species and takes only the cell
        and positions.  Steps share the file's ``outdir`` and ``prefix``, and
        each starts from the density the previous one saved there.
    mode:
        ``"auto"`` (default) picks ``"bindings"`` when ``q3_core`` is
        importable, else ``"subprocess"``. Bindings mode only supports a
        FIXED cell (see ``calculate()``'s bindings branch) -- pass
        ``mode="subprocess"`` explicitly for variable-cell relaxation.
    q3_binary:
        Path to the binary each calculation runs: ``q3``, ``q3_mpi`` or a GPU
        binary.  Auto-detected when ``None``.
    execution_policy:
        The execution policy each calculation starts the binary with: ranks,
        k-point pools, the launcher (see :class:`q3.binary.ExecutionPolicy`).
        Anything but the serial policy needs ``"subprocess"`` mode, which
        ``"auto"`` then picks: the in-process bindings run on one process.
    workdir:
        Directory where each step's structure (``step_NNNN/material.json``) is written.  Defaults
        to a persistent ``q3_ase_workdir/`` in the current directory.
    capture_output:
        When ``True`` (default), binary stdout/stderr are captured.  Set to
        ``False`` for debugging.
    """

    implemented_properties = ["energy", "forces", "stress"]

    def __init__(
        self,
        scf_input_path: str | Path,
        *,
        mode: str = "auto",
        q3_binary: str | Path | None = None,
        execution_policy: ExecutionPolicy | None = None,
        workdir: str | Path | None = None,
        capture_output: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self._scf_input_path = Path(scf_input_path).expanduser().resolve()
        self._workdir = (
            Path(workdir).expanduser().resolve() if workdir else (Path.cwd() / "q3_ase_workdir")
        )
        self._capture_output = capture_output
        self._step = 0
        self._mode = mode

        self._execution_policy = execution_policy_or_serial(execution_policy)
        if self._scf_input_path.suffix.lower() in (".yml", ".yaml"):
            with open(self._scf_input_path) as file_handle:
                self._template_config: dict = yaml.safe_load(file_handle)
            control = self._template_config.get("control", {})
        else:
            # A pw.x deck runs through the binary: the bindings read only Q3's YAML.
            self._mode = "subprocess"
            with open(self._scf_input_path) as file_handle:
                control = espresso.read_fortran_namelist(file_handle)[0].get("control", {})
        # Where the engine, run in the input's directory, writes each step's results.
        self._results_path = (
            self._scf_input_path.parent
            / control.get("outdir", "./")
            / f"{control.get('prefix', 'q3')}_results.json"
        )

        # Resolve mode
        if self._mode == "auto":
            in_process = q3_core is not None and self._execution_policy.is_default
            self._mode = "bindings" if in_process else "subprocess"
        if self._mode == "bindings" and not self._execution_policy.is_default:
            raise ValueError(
                "Q3ASECalculator: mode='bindings' runs in this process and cannot honour "
                f"the execution policy {self._execution_policy!r}. Use mode='subprocess' (or 'auto')."
            )

        self._bindings_reference_cell = None
        if self._mode == "bindings":
            self._solver = q3_core.SCFSolver(str(self._scf_input_path))
            # SCFSolver parses its cell ONCE, from this YAML's material file,
            # and has no API to update it -- solve() only ever takes
            # fractional coordinates. Cache the cell that file actually
            # describes so calculate() can catch (rather than silently
            # ignore) a caller passing it Atoms with a different cell,
            # whether that mismatch exists from the very first call or
            # appears mid-run (e.g. a variable-cell relaxation).
            material_relative_path = self._template_config.get("material", "")
            material_path = (self._scf_input_path.parent / material_relative_path).resolve()
            if material_path.exists():
                self._bindings_reference_cell = np.array(
                    material_json_to_ase_atoms(material_path).get_cell()
                )
        elif self._mode == "subprocess":
            self._binary = find_binary(q3_binary)
        else:
            raise ValueError(
                f"Invalid mode: {self._mode}. Must be one of 'auto', 'subprocess', or 'bindings'."
            )

    # ---------------------------------------------------------------------- #
    # ASE interface                                                            #
    # ---------------------------------------------------------------------- #

    def calculate(
        self,
        atoms=None,
        properties=("energy",),
        system_changes=all_changes,
    ) -> None:
        """Run a single-point Q3 calculation for *atoms*."""
        super().calculate(atoms, properties, system_changes)

        if self._mode == "bindings":
            # SCFSolver.solve() (src/bindings/q3_core_api.hpp) takes ONLY
            # fractional coordinates: it always solves the cell parsed from
            # this YAML's OWN material file, on every call, including the
            # first -- there is no API to pass it a different cell at all.
            # Silently ignoring a mismatch here would mean either (a) a
            # caller who builds `atoms` with an already-different cell gets
            # results for a cell it never asked about, or (b) a caller doing
            # variable-cell relaxation sees the optimizer "converge" against
            # a frozen potential-energy surface. Both fail silently without
            # this check.
            if self._bindings_reference_cell is not None and not np.allclose(
                np.array(atoms.get_cell()), self._bindings_reference_cell, rtol=1e-8, atol=1e-8
            ):
                raise ValueError(
                    "Q3ASECalculator: mode='bindings' always solves the cell from this "
                    "YAML's own material file and cannot accept a different one (the C++ "
                    "SCFSolver bindings have no API to update it). Construct with "
                    "mode='subprocess' to use a different or changing cell."
                )
            coordinates = atoms.get_scaled_positions().tolist()
            result_str = self._solver.solve(coordinates, write_output_files=True)
            q3_results = Results.model_validate_json(result_str)
        else:
            self._step += 1
            step_directory = self._workdir / f"step_{self._step:04d}"
            step_directory.mkdir(parents=True, exist_ok=True)
            material_path = step_directory / "material.json"
            material_path.write_text(json.dumps(ase_atoms_to_material_json(atoms)))
            q3_results = self._run_binary("--material", str(material_path))

        self._store_results(q3_results, properties, len(atoms))

    def run_input_as_written(self):
        """Run the input once, at its own structure, and return that structure as ASE atoms.

        The structure is the one the engine built (the results' ``material``), so any
        input it reads -- a pw.x deck with any ``ibrav`` -- can be relaxed without a
        reader of its own, and forces line up with positions by construction.  The
        atoms carry this calculator with the run's energy, forces and stress already
        stored, so the first step of a relaxation is this run, not a repeat of it.

        Returns the atoms and the run's results.
        """
        self._step += 1
        q3_results = self._run_binary()
        atoms = material_json_to_ase_atoms(q3_results.structure.material)
        atoms.calc = self
        self.atoms = atoms.copy()
        self._store_results(q3_results, ("energy", "forces", "stress"), len(atoms))
        return atoms, q3_results

    def _run_binary(self, *arguments: str) -> Results:
        """Run the binary on the input, in its directory, and read the results it wrote."""
        self._results_path.unlink(missing_ok=True)
        proc = subprocess.run(
            self._execution_policy.command(self._binary, str(self._scf_input_path), *arguments),
            cwd=self._scf_input_path.parent,
            capture_output=self._capture_output,
            text=True,
        )
        if proc.returncode != 0:
            stderr = proc.stderr if self._capture_output else ""
            raise Q3RuntimeError(
                f"Q3ASECalculator: q3 exited with code {proc.returncode} at step {self._step}.",
                returncode=proc.returncode,
                stderr=stderr,
            )
        if not self._results_path.exists():
            raise FileNotFoundError(f"{self._results_path} not found after step {self._step}.")
        results = Results.read(self._results_path)
        self._execution_policy.check_results(results)
        return results

    def _store_results(self, q3_results: Results, properties, number_of_atoms: int) -> None:
        """Convert a Q3 result to ASE's units and conventions, into ``self.results``."""
        raise_if_not_converged(q3_results)

        # Convert energy from Ry to eV.
        energy_ry = extract_total_energy(q3_results)
        self.results["energy"] = energy_ry * _RY_TO_EV

        # Refuse to hand back a derivative Q3 has told us is incomplete.
        raise_if_derivative_incomplete(q3_results, properties)

        # Convert forces from Ry/Bohr to eV/Å.
        forces = q3_results.properties.atomic_forces
        forces_ry_bohr = (
            [atom.value for atom in sorted(forces.values, key=lambda atom: atom.id)]
            if forces
            else []
        )
        if forces_ry_bohr:
            self.results["forces"] = np.array(forces_ry_bohr) * _RY_PER_BOHR_TO_EV_PER_ANG
        elif "forces" in properties:
            self.results["forces"] = np.zeros((number_of_atoms, 3))

        # Convert stress from Q3's kbar (Nielsen-Martin convention) to ASE's
        # Voigt-order eV/Å³ (tensile-positive convention) -- see
        # stress_kbar_to_ase_voigt's docstring for why the sign flips.
        stress = q3_results.properties.stress_tensor
        stress_kbar = [row.root for row in stress.value] if stress else None
        if stress_kbar and "stress" in properties:
            self.results["stress"] = stress_kbar_to_ase_voigt(stress_kbar)
