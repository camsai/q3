"""q3.workflows.elastic — Cubic elastic constants from a strain-stress scan.

Strains the cell, runs SCF at each strain through ``Q3ASECalculator``, and takes
C11, C12 and C44 as the slopes of Q3's stress against the strain.  Requires ASE.

Only the orchestration lives here.  The strain set and the fit are
``q3.properties.elastic_constants``, whose docstring says why every strain is
applied in both signs; the strain itself is ``q3.structure.strain`` -- both
marked as the pieces to move to mat3ra-made.

**Cubic symmetry only.**  One uniaxial and one shear pattern determine the three
independent constants of a cubic crystal and nothing else; on a lower-symmetry
cell they would return weighted averages of several real constants.  The
strains are applied in the frame ``q3.structure.ase_atoms`` builds, which for
Q3's FCC, BCC and simple cubic lattices has the cubic axes along x, y and z.

**C44 is the clamped-ion value.**  Fractional coordinates are held fixed, so the
sublattice relaxation a shear induces in a crystal with a free internal
coordinate is not included: for silicon that is the whole difference between
~100 GPa and the ~75 GPa that compares with the measured 79.6 GPa.  C11 and C12
have no such contribution -- the uniaxial strains preserve each site's symmetry.

Example::

    from q3.workflows import run_cubic_elastic_constants_workflow

    result = run_cubic_elastic_constants_workflow(
        "examples/total_energy--semiconductor--silicon/nc.yml",
        workdir="elastic_scan",
        q3_binary="./build/q3",
    )
    print(result["c11_gpa"], result["c12_gpa"], result["c44_gpa"])
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml

from q3.calculators.ase_calculator import Q3ASECalculator
from q3.properties.elastic_constants import (
    DEFAULT_STRAIN_MAGNITUDES,
    cubic_voigt_strain_set,
    fit_cubic_elastic_constants,
)
from q3.properties.stress import voigt_stress_gpa
from q3.structure.ase_atoms import material_json_to_ase_atoms
from q3.structure.strain import strain_atoms, strain_matrix_from_voigt


def run_cubic_elastic_constants_workflow(
    scf_input_path: str | Path,
    *,
    strain_magnitudes=DEFAULT_STRAIN_MAGNITUDES,
    q3_binary: str | Path | None = None,
    workdir: str | Path | None = None,
    capture_output: bool = True,
) -> dict[str, Any]:
    """Strain the cell, collect Q3's stress at each strain, and fit C11, C12 and C44.

    Runs ``4 * len(strain_magnitudes) + 1`` SCF calculations: both signs of every
    magnitude for a uniaxial-x and a yz-shear strain, plus the unstrained cell.

    Parameters
    ----------
    scf_input_path:
        A Q3 YAML input whose ``material`` names a material JSON file.
    strain_magnitudes:
        Positive magnitudes, each applied in both signs.  Shear strains are
        engineering strains (eps_4 = 2 eps_yz).
    q3_binary, workdir, capture_output:
        Forwarded to ``Q3ASECalculator``; *workdir* collects one
        ``step_NNNN/`` directory per point with the exact input that produced it.

    Returns
    -------
    dict
        ``"c11_gpa"``, ``"c12_gpa"`` and ``"c44_gpa"`` (ESSE has no schema for
        elastic constants), and ``"points"``: per run, its ``"pattern"``
        (``"reference"`` first, then ``"uniaxial_x"`` / ``"shear_yz"``), its
        Voigt ``"voigt_strain"`` and Q3's Voigt ``"voigt_stress_gpa"``, tensile
        positive.  The reference point's stress says how far the input cell is
        from equilibrium -- the constants are those of that pre-stressed cell.
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
        workdir=workdir,
        capture_output=capture_output,
    )

    reference_cell = np.array(atoms.get_cell())
    points: list[dict[str, Any]] = []
    for pattern, voigt_strain in cubic_voigt_strain_set(strain_magnitudes):
        strain_atoms(atoms, reference_cell, strain_matrix_from_voigt(voigt_strain))
        points.append(
            {
                "pattern": pattern,
                "voigt_strain": voigt_strain.tolist(),
                "voigt_stress_gpa": voigt_stress_gpa(atoms.get_stress()).tolist(),
            }
        )

    return {**fit_cubic_elastic_constants(points), "points": points}
