"""
Reading forces out of a finished q3 run, shared by the force validation tests.

Every number a force assertion reads comes from the run's results.json; the
console output is kept beside it for what is only ever said there.
"""

from typing import NamedTuple

BOHR_TO_ANGSTROM = 0.529177249


class Q3Run(NamedTuple):
    """One finished q3 run: what it wrote, and what a person saw.

    Every number an assertion in this file reads comes from `results`. The
    console output is kept beside it for the two things that are only ever
    said there -- which code path a run took, and whether the human-facing
    report still prints at all.
    """

    results: dict
    standard_output: str


def force_on_atom(run, atom_index):
    """The total force on one atom, in Ry/bohr."""
    return run.results["forces_ry_per_bohr"][atom_index]


def force_contribution(run, name):
    """One named term of that force, per atom -- 'hubbard', 'nonlocal', ..."""
    contributions = run.results["force_contributions_ry_per_bohr"]
    assert name in contributions, (
        f"no {name} force contribution in the results: {list(contributions)}"
    )
    return contributions[name]


def force_contribution_on_atom(run, name, atom_index):
    """That term's contribution to the force on one atom."""
    return force_contribution(run, name)[atom_index]
