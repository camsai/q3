"""q3.workflows.wannier.trial_shells — which pseudo-atomic shells become trial orbitals.

The rule, over the shells :mod:`q3.io.pseudo_atomic_shells` reads from a
dataset. :class:`~q3.workflows.wannier.trial_orbitals.TrialOrbitals` applies it
to every species of a material, and the protocol writes the answer into the
capture manifest as ``control.interpolation.trial_shells``.
"""

from __future__ import annotations

from pathlib import Path

from q3.io.pseudo_atomic_shells import PseudoAtomicShell, pseudo_atomic_shells


def _semicore(shells: list[PseudoAtomicShell]) -> set[int]:
    """The occupied shells below the atom's valence, by index.

    With principal quantum numbers: an occupied shell whose n is under the
    outermost occupied shell's, less one for d and two for f -- copper's 3s and
    3p are semicore beside its 4s, its 3d is not. Without them, file order is
    all there is: of the occupied shells sharing an l, all but the last.
    """
    occupied = [shell for shell in shells if shell.usable and shell.occupied]
    if not occupied:
        return set()
    if all(shell.principal_quantum_number is not None for shell in occupied):
        outermost = max(shell.principal_quantum_number for shell in occupied)
        return {
            shell.index
            for shell in occupied
            if shell.principal_quantum_number < outermost - max(0, shell.angular_momentum - 1)
        }
    semicore: set[int] = set()
    for angular_momentum in {shell.angular_momentum for shell in occupied}:
        same = [shell.index for shell in occupied if shell.angular_momentum == angular_momentum]
        semicore.update(same[:-1])
    return semicore


def trial_shells(shells: list[PseudoAtomicShell]) -> list[int]:
    """The shells the capture builds trial orbitals from: every occupied one, and the empty
    ones that are not a second channel of an l the atom already has in valence.

    An augmented dataset lists partial waves beside its atom's valence: a PAW
    file's unoccupied second s and p, an ultrasoft file's empty 4s and 4p. Where
    an occupied valence shell of the same l exists, such a shell describes
    nothing that shell does not, and as a trial orbital it only doubles the
    functions asked for and the band window sized from them. Where the only
    occupied shell of that l is semicore, the empty one is the valence channel
    and stays: copper's 4p, beside its 3p. A shell the engine will not use (see
    :attr:`PseudoAtomicShell.usable`) is never kept and is nobody's valence
    partner. Semicore shells themselves stay: leaving them out is a separate
    choice, with the band window below the valence manifold it would need.
    """
    semicore = _semicore(shells)
    valence_channels = {
        shell.angular_momentum
        for shell in shells
        if shell.usable and shell.occupied and shell.index not in semicore
    }
    return [
        shell.index
        for shell in shells
        if shell.usable and (shell.occupied or shell.angular_momentum not in valence_channels)
    ]


def number_of_trial_orbitals(shells: list[PseudoAtomicShell]) -> int:
    """How many trial orbitals one atom with these shells contributes: 2l + 1 per trial shell."""
    kept = set(trial_shells(shells))
    return sum(shell.orbitals for shell in shells if shell.index in kept)


def pseudo_atomic_orbitals_in(upf_path: str | Path) -> int:
    """:func:`number_of_trial_orbitals` of the shells one dataset lists."""
    return number_of_trial_orbitals(pseudo_atomic_shells(upf_path))
