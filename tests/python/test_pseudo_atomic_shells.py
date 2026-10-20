"""tests/python/test_pseudo_atomic_shells.py — the shells a dataset lists, as ``q3.io`` reads them.

What the reader makes of the repository's own datasets, and what the trial-orbital
rule then does with it, is held in ``test_wannier_workflow.py``. This file holds
the reader's own parts: the threshold it shares with the engine, the core a
shell's principal quantum number is counted up from, and the label it keeps.
"""

import re
from pathlib import Path

import pytest
from q3.io.pseudo_atomic_shells import ZERO_NORM_THRESHOLD, PseudoAtomicShell, pseudo_atomic_shells
from q3.io.pseudo_atomic_shells.principal_quantum_numbers import (
    core_subshells,
    principal_quantum_numbers_from_the_core,
)

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent.parent
ENGINE_READER = REPOSITORY_ROOT / "src/io/pseudopotential/upf_reader/utilities.hpp"


def test_the_zero_norm_threshold_is_the_engines():
    declared = re.search(r"zero_norm_threshold\s*=\s*([0-9.eE+-]+)\s*;", ENGINE_READER.read_text())
    assert declared is not None, "the engine's zero-norm threshold moved; point this test at it"
    assert float(declared.group(1)) == ZERO_NORM_THRESHOLD


@pytest.mark.parametrize(
    ("core_electrons", "outermost"),
    [
        (2, (1, 0)),  # [He]
        (10, (2, 1)),  # [Ne]
        (18, (3, 1)),  # [Ar]
        (28, (3, 2)),  # [Ar]3d, which only filling by n and then l reaches
        (54, (5, 1)),  # [Xe], which only filling by n + l reaches
    ],
)
def test_a_core_is_closed_subshells_in_one_of_the_two_filling_orders(core_electrons, outermost):
    core = core_subshells(core_electrons)
    assert core is not None
    assert sum(2 * (2 * angular + 1) for _, angular in core) == core_electrons
    assert core[-1] == outermost


def test_a_core_no_filling_order_closes_is_refused():
    assert core_subshells(0) == []
    assert core_subshells(3) is None
    assert core_subshells(11) is None


def test_principal_quantum_numbers_continue_above_the_core_per_angular_momentum():
    # Copper with 19 valence electrons: a [Ne] core, so 3s 4s, 3p 4p, 3d 4d.
    assert principal_quantum_numbers_from_the_core("Cu", 19.0, [0, 0, 1, 1, 2, 2]) == [
        3,
        4,
        3,
        4,
        3,
        4,
    ]
    # Copper with 11: [Ar], so 4s, 4p, and the 3d no core subshell precedes.
    assert principal_quantum_numbers_from_the_core("Cu", 11.0, [0, 1, 2]) == [4, 4, 3]
    # No core at all: each l starts at its own lowest n.
    assert principal_quantum_numbers_from_the_core("H", 1.0, [0, 1, 2]) == [1, 2, 3]


def test_what_does_not_fix_a_core_gives_no_numbers():
    assert principal_quantum_numbers_from_the_core("Xx", 19.0, [0]) is None
    assert principal_quantum_numbers_from_the_core("Cu", 18.5, [0]) is None
    assert principal_quantum_numbers_from_the_core("Cu", 26.0, [0]) is None


def test_a_shell_keeps_the_label_its_file_gives_it_in_either_version(tmp_path):
    version_two = tmp_path / "v2.upf"
    version_two.write_text(
        '<UPF version="2.0.1">\n<PP_PSWFC>\n'
        '<PP_CHI.1 type="real" size="2" l="0" occupation="2.0" label="3S">0.1 0.1</PP_CHI.1>\n'
        '<PP_CHI.2 type="real" size="2" l="1" occupation="2.0">0.1 0.1</PP_CHI.2>\n'
        "</PP_PSWFC>\n</UPF>\n"
    )
    shells = pseudo_atomic_shells(version_two)
    assert [shell.label for shell in shells] == ["3S", None]
    assert [shell.index for shell in shells] == [0, 1]
    # One shell without a label, and no header to count from: no numbers at all.
    assert [shell.principal_quantum_number for shell in shells] == [None, None]

    version_one = tmp_path / "v1.UPF"
    version_one.write_text(
        "<PP_PSWFC>\n4S    0  1.00          Wavefunction\n 0.0 0.0\n"
        "3D    2 10.00          Wavefunction\n 0.0 0.0\n</PP_PSWFC>\n"
    )
    shells = pseudo_atomic_shells(version_one)
    assert [(shell.label, shell.principal_quantum_number) for shell in shells] == [
        ("4S", 4),
        ("3D", 3),
    ]
    assert [shell.occupation for shell in shells] == [1.0, 10.0]


def test_a_shell_without_an_angular_momentum_is_refused(tmp_path):
    dataset = tmp_path / "no_l.upf"
    dataset.write_text(
        '<PP_PSWFC>\n<PP_CHI.1 type="real" size="2" occupation="2.0">0.1 0.1</PP_CHI.1>\n'
        "</PP_PSWFC>\n"
    )
    with pytest.raises(ValueError, match="no angular momentum"):
        pseudo_atomic_shells(dataset)


def test_what_a_shell_says_of_itself():
    occupied = PseudoAtomicShell(index=0, angular_momentum=2, occupation=10.0)
    assert (occupied.orbitals, occupied.occupied, occupied.usable) == (5, True, True)
    empty = PseudoAtomicShell(index=1, angular_momentum=1, occupation=0.0)
    assert (empty.orbitals, empty.occupied, empty.usable) == (3, False, True)
    unused = PseudoAtomicShell(index=2, angular_momentum=0, occupation=-1.0)
    assert not unused.usable
    without_a_radial_function = PseudoAtomicShell(
        index=3, angular_momentum=0, occupation=2.0, zero_norm=True
    )
    assert not without_a_radial_function.usable
