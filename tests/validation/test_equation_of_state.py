"""Validation of the Birch-Murnaghan equation-of-state workflow on silicon.

Runs ``q3.workflows.run_equation_of_state_workflow`` end to end against the real
binary and checks the three numbers it exists to produce -- the equilibrium
volume V0, the bulk modulus B0 and its pressure derivative B0' -- against
published silicon values and against Q3's own stress tensor.

The last of those is the strongest check here and needs no external reference at
all: the fit sees only E(V), while the pressure at each point is computed
independently by the stress calculator.  If the fit, the eV/angstrom^3-to-GPa
conversion or the cell-volume bookkeeping were wrong, the two would part company
immediately.

Published comparisons used below (PBE unless stated):

* B0: **~88-89 GPa** for PBE silicon; the experimental value is 97.9 GPa, which
  PBE is well known to underestimate by roughly a tenth.  A test that demanded
  the experimental number would be testing the functional, not the workflow.
* B0': ~4.2, experiment and PBE alike.
* V0: the ``examples/data/materials/silicon.json`` cell is itself a PBE-relaxed
  MaterialsProject structure, so a correct scan must come back to it.

Run::

    python -m pytest tests/validation/test_equation_of_state.py -v --q3-bin ./build/q3
"""

import numpy as np
import pytest
from q3.workflows import run_equation_of_state_workflow

from helpers import EXAMPLES_DIRECTORY, apply_config_overlay, birch_murnaghan_pressure_gpa

SILICON_SCF_INPUT = EXAMPLES_DIRECTORY / "total_energy--semiconductor--silicon" / "nc.yml"

# The primitive FCC cell of examples/data/materials/silicon.json (a = 3.866976 A
# as the primitive vector length), in angstrom^3.
SILICON_REFERENCE_VOLUME_ANGSTROM3 = 40.8883

# The example ships a 2x2x2 k-grid, which is enough for the total energy it is
# an example of but leaves ~0.15 % in V0. 4x4x4 is converged for this quantity
# (6x6x6 moves B0 by under 0.1 %) and keeps the nine-point scan near half a
# minute.
CONVERGED_KGRID_OVERLAY = {
    "method": {"scf": {"kgrid": {"dimensions": [4, 4, 4], "shifts": [1, 1, 1]}}}
}


@pytest.fixture(scope="module")
def silicon_equation_of_state(q3_command, tmp_path_factory):
    """One nine-point volume scan, shared by every assertion in this module."""
    directory = tmp_path_factory.mktemp("equation_of_state")
    scf_input = apply_config_overlay(
        SILICON_SCF_INPUT, CONVERGED_KGRID_OVERLAY, directory / "input"
    )
    return run_equation_of_state_workflow(
        scf_input,
        workdir=directory / "scan",
        # q3_command may be a full command (e.g. "mpirun -np 2 ./build/q3"); the
        # binary is its last token, the same rule tests/validation/conftest.py
        # uses to find q3_utility beside it.
        q3_binary=q3_command[-1],
    )


def test_results_are_esse_properties(silicon_equation_of_state):
    """V0 and E0 come back as the ESSE ``volume`` and ``total_energy`` properties."""
    assert silicon_equation_of_state["volume"]["name"] == "volume"
    assert silicon_equation_of_state["volume"]["units"] == "angstrom^3"
    assert silicon_equation_of_state["total_energy"]["name"] == "total_energy"
    assert silicon_equation_of_state["total_energy"]["units"] == "eV"


def test_equilibrium_volume_returns_to_the_relaxed_input_cell(silicon_equation_of_state):
    """The scan must land back on the PBE-relaxed cell it was centred on.

    Not a tautology: the scan measures nine cells and the fit is free to put its
    minimum anywhere among them, so recovering the starting volume is evidence
    that Q3's E(V) really does minimise there -- and it is the single number
    that would move if the FCC conventional-lattice-constant conversion in
    ``q3.structure`` were dropped.
    """
    fitted = silicon_equation_of_state["volume"]["value"]
    assert fitted == pytest.approx(SILICON_REFERENCE_VOLUME_ANGSTROM3, rel=0.01)


def test_bulk_modulus_matches_the_published_pbe_silicon_value(silicon_equation_of_state):
    bulk_modulus = silicon_equation_of_state["bulk_modulus_gpa"]
    # Wide band first: a sign error, a Ry/eV slip or a bohr/angstrom slip all
    # land orders of magnitude outside it rather than a few percent.
    assert 50.0 < bulk_modulus < 150.0
    # Then the physics: PBE silicon is 88-89 GPa.
    assert bulk_modulus == pytest.approx(88.5, rel=0.05)


def test_pressure_derivative_is_near_four(silicon_equation_of_state):
    """B0' ~ 4.2 for silicon; 4 is also where the third-order term vanishes.

    A fit that returned exactly 4.0 would be suspicious rather than reassuring
    -- it is what a second-order Birch-Murnaghan degenerates to -- so the window
    is set around the published value, not around the degenerate one.
    """
    assert silicon_equation_of_state["bulk_modulus_pressure_derivative"] == pytest.approx(
        4.2, abs=0.8
    )


def test_fitted_pressures_agree_with_the_computed_stress_tensor(silicon_equation_of_state):
    """P(V) from the E(V) fit against P(V) from Q3's stress -- two independent routes.

    The fit is built from energies alone and never sees a stress; the stress is
    an analytic derivative computed inside the engine.  Agreement over the whole
    scan is a statement about both, and it is what would break first if the
    volume bookkeeping or the unit conversion were wrong.
    """
    volumes = np.asarray(silicon_equation_of_state["volumes_angstrom3"])
    fitted = birch_murnaghan_pressure_gpa(volumes, silicon_equation_of_state)
    measured = np.asarray(silicon_equation_of_state["pressures_gpa"])
    for volume, one_fitted, one_measured in zip(volumes, fitted, measured):
        print(
            f"\n  V={volume:8.4f} A^3: stress {one_measured:+8.4f} GPa, "
            f"fit {one_fitted:+8.4f} GPa, diff {one_measured - one_fitted:+7.4f}"
        )
    assert np.max(np.abs(measured - fitted)) < 0.15


def test_pressure_rises_monotonically_under_compression(silicon_equation_of_state):
    """Q3's own measured pressures must be monotonic in volume over the scan.

    A non-monotonic P(V) would mean the scan crossed a mechanical instability or
    that a point failed to converge to the same solution as its neighbours;
    either invalidates the fit that follows.
    """
    pressures = np.asarray(silicon_equation_of_state["pressures_gpa"])
    assert np.all(np.diff(pressures) < 0.0)
