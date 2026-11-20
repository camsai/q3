"""
A Quantum ESPRESSO deck computes the forces and the stress it asks for, as in pw.x.

pw.x computes forces only under ``tprnfor`` (on by itself for ``relax`` and
``vc-relax``) and the stress only under ``tstress`` (on by itself for
``vc-relax``). Q3 used to compute both for every deck, which on a large cell is
a sizeable share of a run the deck never asked for. These tests pin the three
ways a run can now come to hold them: the deck's own keys, the ``--forces`` and
``--stress`` flags a driver passes, and a Q3 YAML input, which always does.
"""

import pytest

from helpers import EXAMPLES_DIRECTORY, run_q3_and_load_results

PREFIX = "silicon"

DECK = """&CONTROL
  calculation = 'scf'
  prefix = '{prefix}'
  pseudo_dir = '{pseudo_dir}'
  outdir = './output'
{control_keys}/
&SYSTEM
  ibrav = 2
  celldm(1) = 10.2
  nat = 2
  ntyp = 1
  ecutwfc = 12.0
/
&ELECTRONS
  conv_thr = 1.0e-8
/
ATOMIC_SPECIES
  Si 28.086 Si.pz-vbc.UPF
ATOMIC_POSITIONS alat
  Si 0.00 0.00 0.00
  Si 0.27 0.25 0.25
K_POINTS automatic
  2 2 2 0 0 0
"""


def run_deck(q3_command, directory, control_keys="", flags=()):
    """Run the displaced-silicon deck with extra &CONTROL keys and command-line flags."""
    directory.mkdir(parents=True, exist_ok=True)
    deck_path = directory / "scf.in"
    deck_path.write_text(
        DECK.format(
            prefix=PREFIX,
            pseudo_dir=EXAMPLES_DIRECTORY / "data" / "pseudopotentials" / "nc",
            control_keys=control_keys,
        )
    )
    results, _ = run_q3_and_load_results(list(q3_command) + list(flags), deck_path, PREFIX)
    return results


def test_a_plain_scf_deck_computes_neither(q3_command, tmp_path):
    results = run_deck(q3_command, tmp_path / "plain")
    assert results.properties.total_energy is not None
    assert results.properties.atomic_forces is None
    assert results.properties.stress_tensor is None
    assert results.properties.pressure is None


@pytest.mark.parametrize(
    "control_keys, flags",
    [
        ("  tprnfor = .true.\n  tstress = .true.\n", ()),
        ("", ("--forces", "--stress")),
    ],
    ids=["deck-keys", "command-line-flags"],
)
def test_asked_for_both_are_computed(q3_command, tmp_path, control_keys, flags):
    results = run_deck(q3_command, tmp_path / "both", control_keys, flags)
    forces = results.properties.atomic_forces
    assert forces is not None and len(forces.values) == 2
    # The second atom is displaced along x, so the force on it is not zero.
    assert max(abs(component) for atom in forces.values for component in atom.value) > 1e-3
    assert results.properties.stress_tensor is not None
    assert results.properties.pressure is not None


def test_each_is_switched_separately(q3_command, tmp_path):
    forces_only = run_deck(q3_command, tmp_path / "forces", "  tprnfor = .true.\n")
    assert forces_only.properties.atomic_forces is not None
    assert forces_only.properties.stress_tensor is None

    stress_only = run_deck(q3_command, tmp_path / "stress", flags=("--stress",))
    assert stress_only.properties.atomic_forces is None
    assert stress_only.properties.stress_tensor is not None
