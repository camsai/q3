"""
Supercell folding consistency (machine-validated, no reference code needed).

The same crystal computed in a supercell must give the same energy per formula
unit as in its primitive cell whenever the two k-point meshes sample the same
points. A 2x2x2 supercell of FCC silicon on a shifted 2x2x2 Monkhorst-Pack mesh
unfolds exactly onto the primitive cell's shifted 4x4x4 mesh: every supercell
point sits at 1/8 or 3/8 of the primitive reciprocal vectors, and its eight
images under the supercell reciprocal lattice fill the 4x4x4 set
{1/8, 3/8, 5/8, 7/8} on each axis. The plane-wave spheres and the FFT grids
fold the same way, so

    E(16-atom cell, 2x2x2 shifted)  ==  8 x E(2-atom cell, 4x4x4 shifted)

holds to the SCF convergence level, term by term (Ewald, Hartree,
exchange-correlation, band energy). Any treatment that differs between the two
cells -- symmetry above all -- breaks the relation, which no comparison against
an external code can catch when the primitive cell agrees with that code.

Until 2026-09-10 the density symmetrizer discarded every operation carrying a
fractional translation once it saw a supercell, keeping 12 of the 384
space-group operations. Those 12 happen to map the shifted mesh onto itself, so
the supercell converged to the unsymmetrized-mesh density and sat +1.44e-4 Ry
above eight primitive cells (camsai/q3#6).

The same report found that ``control.disable_symmetry`` never moved the energy:
it only skipped the mesh reduction while the density was symmetrized regardless.
It now uses no symmetry at all, so the raw shifted mesh gives a different (and
less converged) energy than the symmetrized one -- by the gap QE shows -- while
the folding relation holds either way, because the raw supercell mesh unfolds
onto the raw primitive mesh just as the symmetrized ones do.

Run:
    .venv/bin/pytest tests/validation/test_supercell_folding_consistency.py -v --q3-bin ./build/q3
    (add --tier2 for the no-symmetry supercell leg)
"""

import pytest

from helpers import EXAMPLES_DIRECTORY, run_overlay_and_read_energy

SILICON_DIRECTORY = EXAMPLES_DIRECTORY / "total_energy--semiconductor--silicon"

# 2-atom primitive cell; the committed deck samples a shifted 2x2x2 mesh.
PRIMITIVE_INPUT = SILICON_DIRECTORY / "nc.yml"
PRIMITIVE_PREFIX = "si_nc"
# 16-atom 2x2x2 supercell of the same crystal, shifted 2x2x2 mesh as committed.
SUPERCELL_INPUT = SILICON_DIRECTORY / "nc_2x2x2.yml"
SUPERCELL_PREFIX = "si_nc_2x2x2"
FORMULA_UNITS_PER_SUPERCELL = 8

# The primitive mesh that the supercell's 2x2x2 mesh unfolds onto.
PRIMITIVE_4X4X4_OVERLAY = {"method": {"scf": {"kgrid": {"dimensions": [4, 4, 4]}}}}
NO_SYMMETRY_OVERLAY = {"control": {"disable_symmetry": True}}
PRIMITIVE_4X4X4_NO_SYMMETRY_OVERLAY = {**PRIMITIVE_4X4X4_OVERLAY, **NO_SYMMETRY_OVERLAY}

FOLDING_TOLERANCE_RYDBERG = 1e-6
"""
Bound on |E(supercell) - 8 E(primitive)|, for the whole 16-atom cell.

The defect this test guards was +1.44e-4 Ry. With both cells symmetrized over
their full space groups the total folds to +2.8e-7 Ry, and that much is the
Ewald term alone (+2.9e-7 Ry, from the committed supercell coordinates; QE 7.5
shows the same +2.4e-7 Ry). The bound sits 3.6x above what is left and 140x
below the defect.
"""


@pytest.fixture(scope="module")
def primitive_energy(q3_command, tmp_path_factory):
    """E(2-atom cell, 4x4x4 shifted), Ry."""
    run_directory = tmp_path_factory.mktemp("folding_primitive_4x4x4")
    return run_overlay_and_read_energy(
        q3_command, PRIMITIVE_INPUT, PRIMITIVE_4X4X4_OVERLAY, PRIMITIVE_PREFIX, run_directory
    )


@pytest.fixture(scope="module")
def supercell_energy(q3_command, tmp_path_factory):
    """E(16-atom cell, 2x2x2 shifted), Ry, deck as committed."""
    run_directory = tmp_path_factory.mktemp("folding_supercell_2x2x2")
    return run_overlay_and_read_energy(
        q3_command, SUPERCELL_INPUT, None, SUPERCELL_PREFIX, run_directory
    )


def test_supercell_energy_folds_onto_primitive_cell(primitive_energy, supercell_energy):
    """Eight primitive cells on the unfolded mesh are one supercell on the folded one."""
    folded_energy = FORMULA_UNITS_PER_SUPERCELL * primitive_energy
    difference = supercell_energy - folded_energy
    assert abs(difference) < FOLDING_TOLERANCE_RYDBERG, (
        f"Supercell folding consistency violated:\n"
        f"  E(2-atom, 4x4x4 shifted)   = {primitive_energy:.12f} Ry\n"
        f"  8 x that                   = {folded_energy:.12f} Ry\n"
        f"  E(16-atom, 2x2x2 shifted)  = {supercell_energy:.12f} Ry\n"
        f"  E(supercell) - 8 E(prim)   = {difference:+.3e} Ry"
        f" ({difference / (2 * FORMULA_UNITS_PER_SUPERCELL):+.2e} Ry/atom)"
    )


SYMMETRIZATION_GAP_BOUNDS_RYDBERG = (1.5e-5, 2.1e-5)
"""
Bounds on E(no symmetry) - E(symmetric) for the 2-atom cell on the shifted
4x4x4 mesh.

A shifted Monkhorst-Pack mesh is not invariant under the FCC point group, so
the symmetrized density is that of a larger k-set (the mesh plus its images)
and its energy is lower than the raw mesh's. QE 7.5 puts the gap at +1.796e-5
Ry on these decks and Q3 measures +1.797e-5 Ry; before the flag also
disabled density symmetrization the gap was exactly zero.
"""

UNSYMMETRIZED_SUPERCELL_ENERGY_BEFORE_FIX_RYDBERG = -135.400345677
"""
What nc_2x2x2.yml produced before 2026-09-10, with or without the flag: the
raw-mesh energy, which the primitive cell must now reproduce (divided by eight)
when it too uses no symmetry.
"""


@pytest.fixture(scope="module")
def primitive_energy_without_symmetry(q3_command, tmp_path_factory):
    """E(2-atom cell, 4x4x4 shifted, no symmetry), Ry."""
    run_directory = tmp_path_factory.mktemp("folding_primitive_4x4x4_no_symmetry")
    return run_overlay_and_read_energy(
        q3_command,
        PRIMITIVE_INPUT,
        PRIMITIVE_4X4X4_NO_SYMMETRY_OVERLAY,
        PRIMITIVE_PREFIX,
        run_directory,
    )


def test_disable_symmetry_gives_the_unsymmetrized_mesh(
    primitive_energy, primitive_energy_without_symmetry
):
    """Without symmetry the shifted mesh is sampled as is, and the energy shows it."""
    gap = primitive_energy_without_symmetry - primitive_energy
    lower_bound, upper_bound = SYMMETRIZATION_GAP_BOUNDS_RYDBERG
    assert lower_bound < gap < upper_bound, (
        f"E(no symmetry) - E(symmetric) = {gap:+.3e} Ry, expected within "
        f"({lower_bound:.1e}, {upper_bound:.1e}) Ry:\n"
        f"  E(symmetric)   = {primitive_energy:.12f} Ry\n"
        f"  E(no symmetry) = {primitive_energy_without_symmetry:.12f} Ry"
    )
    folded_before_fix = (
        UNSYMMETRIZED_SUPERCELL_ENERGY_BEFORE_FIX_RYDBERG / FORMULA_UNITS_PER_SUPERCELL
    )
    difference = primitive_energy_without_symmetry - folded_before_fix
    assert abs(difference) < FOLDING_TOLERANCE_RYDBERG, (
        f"The raw-mesh primitive energy should be the old supercell energy over eight:\n"
        f"  E(no symmetry)                 = {primitive_energy_without_symmetry:.12f} Ry\n"
        f"  E(supercell before fix) / 8    = {folded_before_fix:.12f} Ry\n"
        f"  |diff|                         = {abs(difference):.2e} Ry"
    )


@pytest.mark.slow
def test_supercell_energy_folds_without_symmetry(
    q3_command, primitive_energy_without_symmetry, tmp_path
):
    """The raw supercell mesh unfolds onto the raw primitive mesh, so folding holds here too.

    Tier 2: the unreduced 16-atom run is the most expensive leg of the module (about 35 s
    here), and the relation it checks holds by the same mechanism as the symmetric one.
    Measured +2.9e-7 Ry.
    """
    supercell_energy_without_symmetry = run_overlay_and_read_energy(
        q3_command, SUPERCELL_INPUT, NO_SYMMETRY_OVERLAY, SUPERCELL_PREFIX, tmp_path
    )
    folded_energy = FORMULA_UNITS_PER_SUPERCELL * primitive_energy_without_symmetry
    difference = supercell_energy_without_symmetry - folded_energy
    assert abs(difference) < FOLDING_TOLERANCE_RYDBERG, (
        f"Supercell folding consistency violated without symmetry:\n"
        f"  E(2-atom, 4x4x4 shifted)   = {primitive_energy_without_symmetry:.12f} Ry\n"
        f"  8 x that                   = {folded_energy:.12f} Ry\n"
        f"  E(16-atom, 2x2x2 shifted)  = {supercell_energy_without_symmetry:.12f} Ry\n"
        f"  E(supercell) - 8 E(prim)   = {difference:+.3e} Ry"
    )
