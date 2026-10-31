"""q3.workflows.equation_of_state — Birch-Murnaghan equation of state from a volume scan.

Scales the cell uniformly, runs SCF at each point through ``Q3ASECalculator``,
and fits a third-order Birch-Murnaghan equation of state to the E(V) curve:
the equilibrium volume V0, the bulk modulus B0 and its pressure derivative
B0'. Requires ASE.

Only the orchestration lives here.  The strain is ``q3.structure.strain``, the
fit and the stress-to-pressure conversion are ``q3.properties`` -- both marked
as the pieces to move to mat3ra-made.

Example::

    from q3.workflows import run_equation_of_state_workflow

    result = run_equation_of_state_workflow(
        "examples/total_energy--semiconductor--silicon/nc.yml",
        workdir="eos_scan",
        q3_binary="./build/q3",
    )
    print(result["volume"]["value"], "A^3")
    print(result["bulk_modulus_gpa"], "GPa")

Accuracy notes for anyone reading a number out of this
------------------------------------------------------
* **The cutoff is fixed in Rydberg, the basis is not.**  Compressing a cell at
  fixed cutoff changes how many plane waves the basis holds, which biases E(V)
  and therefore B0 (the Pulay/basis-set error of any volume scan).  Use a cutoff
  well above the one that converges the total energy, and widen the scan no
  further than it has to be.
* **Fractional coordinates are held fixed**, which is exact for a structure
  whose sites are fixed by symmetry (silicon's diamond basis, any single-site
  metal) and an approximation for one with a free internal coordinate, where
  each point would want its ionic positions relaxed first.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml

from q3.binary import ExecutionPolicy
from q3.calculators.ase_calculator import Q3ASECalculator
from q3.properties.equation_of_state import DEFAULT_LINEAR_SCALE_FACTORS, fit_birch_murnaghan
from q3.properties.stress import hydrostatic_pressure_gpa
from q3.structure.ase_atoms import material_json_to_ase_atoms
from q3.structure.strain import isotropic_strain_matrix, strain_atoms


def run_equation_of_state_workflow(
    scf_input_path: str | Path,
    *,
    linear_scale_factors=DEFAULT_LINEAR_SCALE_FACTORS,
    q3_binary: str | Path | None = None,
    execution_policy: ExecutionPolicy | None = None,
    workdir: str | Path | None = None,
    capture_output: bool = True,
) -> dict[str, Any]:
    """Scan cell volumes, run SCF at each, and fit a Birch-Murnaghan equation of state.

    Parameters
    ----------
    scf_input_path:
        A Q3 YAML input whose ``material`` names a material JSON file.
    linear_scale_factors:
        Multipliers applied to the **lattice vectors**, so volume scales as the
        cube.  The default
        (``q3.properties.equation_of_state.DEFAULT_LINEAR_SCALE_FACTORS``) spans
        +-4 % in length (+-12 % in volume).
    q3_binary, execution_policy, workdir, capture_output:
        Forwarded to ``Q3ASECalculator``; *workdir* collects one
        ``step_NNNN/`` directory per point with the exact input that produced it.

    Returns
    -------
    dict
        ``"volume"`` (V0) and ``"total_energy"`` (E0) as ESSE properties,
        ``"bulk_modulus_gpa"`` and ``"bulk_modulus_pressure_derivative"`` (ESSE
        has no schema for either), and the scan itself: ``"volumes_angstrom3"``,
        ``"energies_ev"`` and ``"pressures_gpa"`` -- the pressure at each point
        from Q3's own stress tensor, compression positive, which the fit never
        sees and which therefore checks it.
    """
    scf_input_path = Path(scf_input_path).expanduser().resolve()
    with open(scf_input_path) as file_handle:
        config = yaml.safe_load(file_handle)

    atoms = material_json_to_ase_atoms((scf_input_path.parent / config["material"]).resolve())
    # Subprocess mode: every point is a different cell, which the bindings
    # path cannot solve (see Q3ASECalculator.calculate()).
    atoms.calc = Q3ASECalculator(
        scf_input_path,
        mode="subprocess",
        q3_binary=q3_binary,
        execution_policy=execution_policy,
        workdir=workdir,
        capture_output=capture_output,
    )

    reference_cell = np.array(atoms.get_cell())
    volumes: list[float] = []
    energies: list[float] = []
    pressures: list[float] = []
    for scale in linear_scale_factors:
        strain_atoms(atoms, reference_cell, isotropic_strain_matrix(scale))
        # Stress first: the same SCF also stores the energy, which is then read
        # from the calculator's cache instead of costing a second run.
        pressures.append(hydrostatic_pressure_gpa(atoms.get_stress()))
        energies.append(float(atoms.get_potential_energy()))
        volumes.append(float(atoms.get_volume()))

    return {
        **fit_birch_murnaghan(volumes, energies),
        "volumes_angstrom3": volumes,
        "energies_ev": energies,
        "pressures_gpa": pressures,
    }
