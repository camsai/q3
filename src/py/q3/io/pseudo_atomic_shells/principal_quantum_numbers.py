"""q3.io.pseudo_atomic_shells.principal_quantum_numbers — a shell's n, counted up from the core.

A dataset whose shells carry no labels -- a PAW dataset written by ATOMPAW --
still fixes each shell's principal quantum number: its header names the element
and the valence charge, and the electrons the pseudopotential removed fill
closed subshells.
"""

from __future__ import annotations

from mat3ra.periodic_table import PERIODIC_TABLE

#: How far a header's valence charge may sit from a whole number of electrons.
INTEGER_CHARGE_TOLERANCE = 1.0e-6

#: Closed subshells (n, l), in the two orders a pseudopotential's core follows:
#: by n and then l (gallium's 28 core electrons end on 3d, not 4s), and by n + l
#: (xenon's 54 end on 5p, below the 4f the first order would fill before it).
_SUBSHELLS = [
    (principal, angular) for principal in range(1, 8) for angular in range(min(principal, 4))
]
_CORE_FILLING_ORDERS = (
    sorted(_SUBSHELLS),
    sorted(_SUBSHELLS, key=lambda subshell: (subshell[0] + subshell[1], subshell[0])),
)


def core_subshells(core_electrons: int) -> list[tuple[int, int]] | None:
    """The closed subshells (n, l) holding exactly *core_electrons*, or None if no filling order does."""
    for order in _CORE_FILLING_ORDERS:
        filled: list[tuple[int, int]] = []
        electrons = 0
        for principal, angular in order:
            if electrons >= core_electrons:
                break
            filled.append((principal, angular))
            electrons += 2 * (2 * angular + 1)
        if electrons == core_electrons:
            return filled
    return None


def principal_quantum_numbers_from_the_core(
    symbol: str, valence_charge: float, angular_momenta: list[int]
) -> list[int] | None:
    """Each shell's n, counted up from the element's core; None when the two do not fix it.

    Z - z_valence electrons fill closed subshells, and the shells of each l
    continue above that l's last core subshell in the order the file lists
    them: copper with 19 valence electrons has a [Ne] core, so its two s shells
    are 3s and 4s and its one occupied p shell is 3p. Z comes from
    ``mat3ra-periodic-table``, which is the atomic number this needs and the
    only thing a dataset's header leaves out.
    """
    if (
        symbol not in PERIODIC_TABLE
        or abs(valence_charge - round(valence_charge)) > INTEGER_CHARGE_TOLERANCE
    ):
        return None
    core = core_subshells(PERIODIC_TABLE[symbol]["atomic_number"] - round(valence_charge))
    if core is None:
        return None
    following = {
        angular: 1
        + max(
            (principal for principal, core_angular in core if core_angular == angular),
            default=angular,
        )
        for angular in set(angular_momenta)
    }
    numbers = []
    for angular in angular_momenta:
        numbers.append(following[angular])
        following[angular] += 1
    return numbers
