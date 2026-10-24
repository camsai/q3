"""A QE deck's plain mixing converges a metallic slab as pw.x's does (self-validated).

pw.x's default mixing_mode = 'plain' is Broyden with no preconditioning. Q3's QE reader
mapped it to Broyden but left the preconditioner at Q3's own default, which switches
Kerker on whenever smearing is on. Kerker damps the long-wavelength charge, and in a slab
that is the charge sloshing across the vacuum: the slowest mode to converge. A 64-atom
Ni(111) + CO slab took Q3 219 SCF iterations against pw.x's 103.

The deck below is a four-layer Cu(111) slab with 15 angstrom of vacuum, the committed
ultrasoft Cu dataset and the default plain mixing. Measured on it: pw.x 7.6, 22 iterations;
Q3 with the old reader, 46; Q3 now, 27. The bound sits between the two Q3 counts.

Run:
    .venv/bin/pytest tests/validation/test_qe_plain_mixing_slab.py -v --q3-bin ./build/q3
"""

import math
import shutil

from helpers import EXAMPLES_DIRECTORY, load_results_json, run_q3

CU_ULTRASOFT = EXAMPLES_DIRECTORY / "data" / "pseudopotentials" / "us" / "cu_pbe_v1.2.uspp.F.UPF"
ITERATION_BOUND = 36
"""Q3 converges this deck in 27 iterations; Kerker-preconditioned mixing takes 46."""


def cu_slab_deck(layers=4, vacuum_angstrom=15.0):
    lattice_constant = 3.61
    in_plane = lattice_constant / math.sqrt(2)
    spacing = lattice_constant / math.sqrt(3)
    height = (layers - 1) * spacing + vacuum_angstrom
    stacking = [(0.0, 0.0), (1 / 3, 2 / 3), (2 / 3, 1 / 3)]
    first = 0.5 - (layers - 1) * spacing / 2 / height
    cell = [(in_plane, 0, 0), (-in_plane / 2, in_plane * math.sqrt(3) / 2, 0), (0, 0, height)]
    rows = "\n".join(f"  {x:.10f} {y:.10f} {z:.10f}" for x, y, z in cell)
    atoms = "\n".join(
        f"  Cu {stacking[i % 3][0]:.10f} {stacking[i % 3][1]:.10f} {first + i * spacing / height:.10f}"
        for i in range(layers)
    )
    return f"""&CONTROL
  calculation = 'scf'
  prefix = 'cu_slab'
  pseudo_dir = './'
  outdir = './output'
/
&SYSTEM
  ibrav = 0
  nat = {layers}
  ntyp = 1
  ecutwfc = 30.0
  ecutrho = 240.0
  occupations = 'smearing'
  smearing = 'gaussian'
  degauss = 0.01
/
&ELECTRONS
  conv_thr = 1.0d-8
  mixing_beta = 0.1
/
ATOMIC_SPECIES
  Cu 63.546 {CU_ULTRASOFT.name}
CELL_PARAMETERS angstrom
{rows}
ATOMIC_POSITIONS crystal
{atoms}
K_POINTS automatic
  6 6 1 0 0 0
"""


def test_plain_mixing_converges_a_metallic_slab_like_pwx(q3_command, tmp_path):
    shutil.copy(CU_ULTRASOFT, tmp_path / CU_ULTRASOFT.name)
    deck = tmp_path / "cu_slab.in"
    deck.write_text(cu_slab_deck())
    completed = run_q3(list(q3_command) + ["--format", "qe"], deck, working_directory=tmp_path)
    assert completed.returncode == 0, (
        f"q3 exited with {completed.returncode}\n"
        f"stdout:\n{completed.stdout[-4000:]}\nstderr:\n{completed.stderr}"
    )
    results = load_results_json(tmp_path / "output", "cu_slab")
    assert results.get("converged"), completed.stdout[-3000:]
    iterations = int(results["self_consistent_field_iterations"])
    assert iterations <= ITERATION_BOUND, (
        f"the Cu(111) slab took {iterations} SCF iterations (bound {ITERATION_BOUND}); "
        "Kerker-preconditioned mixing takes 46 on it, pw.x's plain mixing 22"
    )
