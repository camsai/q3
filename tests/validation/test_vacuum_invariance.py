"""
An isolated atom's energy must not depend on how much vacuum surrounds it.

Until 2026-10-07 the local pseudopotential V_loc(q) was interpolated from a uniform
q-table with its 1/q² divergence included, and a 4-point fit across that divergence
is wrong near the origin. Only a cell with a long direction has G-vectors that close
to it, so bulk cells never showed the error; a slab or a molecule picked up an energy
that grew roughly as c⁵ with the cell length c -- -59 mRy on a Ni/CO slab in a 20 Å
cell, where Quantum ESPRESSO's energy stays flat. The unit tests in
test_local_potential_table.cpp pin the table against the exact transform; this pins
the consequence, without needing another code: one neutral silicon atom in a
6 x 6 x c Å cell gives the same energy at c = 20 Å and c = 30 Å.

The cells start at 20 Å because the atoms, 6 Å apart in the plane, form layers that
still feel each other across 10 Å of vacuum: ESPRESSO 7.6 puts 0.25 mRy between 10 Å
and 20 Å for this deck, and Q3 agrees. From 20 Å on it puts 0.02 μRy between them.

Run:
    .venv/bin/pytest tests/validation/test_vacuum_invariance.py -v --q3-bin ./build/q3
"""

import pytest

from helpers import EXAMPLES_DIRECTORY, run_q3

ANGSTROM_IN_BOHR = 1.8897261245650618

CELL_LENGTHS_ANGSTROM = (20.0, 30.0)

TOLERANCE_MILLI_RYDBERG = 0.01
"""
How far apart the two cells may be.

The interpolated tail put 0.51 mRy between them. What is left without it is the two
cells' own difference, mostly their different FFT grids: 0.44 μRy measured, and
0.02 μRy with the wavefunction grid sized as ESPRESSO sizes it.
"""


def _deck(cell_length_angstrom):
    """A neutral Si atom in a 6 x 6 x c Å cell, at the centre of the long axis."""
    pseudopotential_directory = EXAMPLES_DIRECTORY / "data" / "pseudopotentials" / "nc"
    return f"""&CONTROL
  calculation = 'scf'
  pseudo_dir = '{pseudopotential_directory}/'
  outdir = './output'
  prefix = 'silicon_atom'
/
&SYSTEM
  ibrav = 0
  celldm(1) = {ANGSTROM_IN_BOHR}
  nat = 1
  ntyp = 1
  ecutwfc = 25.0
  ecutrho = 100.0
  nbnd = 8
  occupations = 'smearing'
  smearing = 'gaussian'
  degauss = 0.01
/
&ELECTRONS
  conv_thr = 1.0d-10
  mixing_beta = 0.3
  electron_maxstep = 200
/
ATOMIC_SPECIES
  Si 28.0855 Si.upf
CELL_PARAMETERS {{alat}}
  6.0 0.0 0.0
  0.0 6.0 0.0
  0.0 0.0 {cell_length_angstrom}
ATOMIC_POSITIONS {{crystal}}
  Si 0.0 0.0 0.5
K_POINTS automatic
  1 1 1 0 0 0
"""


def _total_energy(q3_command, directory, cell_length_angstrom):
    run_directory = directory / f"c{cell_length_angstrom:g}"
    run_directory.mkdir()
    deck = run_directory / "scf.in"
    deck.write_text(_deck(cell_length_angstrom))
    result = run_q3(q3_command, deck, working_directory=run_directory, timeout=900)
    assert result.returncode == 0, f"q3 failed:\n{result.stdout[-3000:]}\n{result.stderr[-2000:]}"
    for line in result.stdout.splitlines():
        if "total energy" in line and "=" in line and "Ry" in line:
            try:
                return float(line.split("=")[1].split("Ry")[0])
            except ValueError:
                continue
    pytest.fail(f"q3 printed no total energy:\n{result.stdout[-3000:]}")


def test_an_isolated_atom_does_not_feel_the_vacuum_around_it(q3_command, tmp_path):
    shorter, longer = CELL_LENGTHS_ANGSTROM
    near = _total_energy(q3_command, tmp_path, shorter)
    far = _total_energy(q3_command, tmp_path, longer)
    difference_milli_rydberg = (far - near) * 1000.0
    print(
        f"\n  E(c = {shorter:g} Å) = {near:.9f} Ry, E(c = {longer:g} Å) = {far:.9f} Ry, "
        f"difference {difference_milli_rydberg:+.5f} mRy (tolerance {TOLERANCE_MILLI_RYDBERG} mRy)"
    )
    assert abs(difference_milli_rydberg) < TOLERANCE_MILLI_RYDBERG, (
        f"one Si atom's energy moved by {difference_milli_rydberg:+.5f} mRy when its cell "
        f"grew from {shorter:g} Å to {longer:g} Å; an isolated atom should not feel the vacuum"
    )
