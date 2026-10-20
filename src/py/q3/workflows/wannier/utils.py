"""q3.workflows.wannier.utils — helpers of the protocol that hold no state.

Pure functions of their arguments: how many bands a capture keeps, where the
reference k-points go, what the next mesh is, which threshold pairs are left to
try, whether a results file reports states its dataset cannot represent, and
the tolerance a disentanglement that ran out of iterations would have converged
under.
"""

from __future__ import annotations

import math
import re

from q3.io.interpolation_results import InterpolationResults
from q3.workflows.interpolation import Grid, KPoint, mesh_for_spacing, validated_grid
from q3.workflows.wannier.settings import (
    BAND_WINDOW_EXTRA,
    BAND_WINDOW_FACTOR,
    DEFAULT_DISENTANGLEMENT_TOLERANCE,
    LOOSENED_TOLERANCE_MARGIN,
    MAXIMUM_DISENTANGLEMENT_TOLERANCE,
    MAXIMUM_REFERENCE_KPOINTS,
    PRINTED_SPREAD_RESOLUTION_BOHR_SQUARED,
    PROJECTABILITY_MAXIMUM_LADDER,
    PROJECTABILITY_MINIMUM_LADDER,
    SMALLEST_SPREAD_BOHR_SQUARED,
)

_SPREADS_LINE = re.compile(r"(?m)^.*?\bdiagnostic_spreads_bohr_squared:\s*([^A-Za-z\n]*)$")
_NUMBER = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def band_window(wannier_functions: int) -> int:
    """ceil(1.5 N) + 4: room above the manifold for the disentangler to choose from."""
    return int(math.ceil(BAND_WINDOW_FACTOR * wannier_functions)) + BAND_WINDOW_EXTRA


def off_mesh_reference_kpoints(
    grid: Grid, maximum_points: int = MAXIMUM_REFERENCE_KPOINTS
) -> list[KPoint]:
    """Points half a mesh step off every mesh point, on a sub-grid capped in size.

    At the mesh points the model is exact by construction, so the honest
    reference is between them: (j s + 1/2) / n along each axis, with the
    stride s the smallest that keeps the count under *maximum_points*. Every
    point is exactly half a step from a mesh point on every axis.
    """
    dimensions = validated_grid(grid)
    stride = 1
    while math.prod(math.ceil(points / stride) for points in dimensions) > maximum_points:
        stride += 1
    first, second, third = (
        [(index * stride + 0.5) / points for index in range(math.ceil(points / stride))]
        for points in dimensions
    )
    return [(x, y, z) for x in first for y in second for z in third]


def finer_mesh(mesh: Grid, grid: Grid | None, lattice: dict, spacing: float) -> Grid | None:
    """The mesh one halving of the spacing gives, or None when it would not change.

    A grid the rule chose is re-derived at half the spacing; an explicit grid is
    doubled on every axis, which is what halving the spacing does to it. A
    spacing already fine enough that halving lands on the same integers means
    the mesh rung has nothing to offer, and the caller rejects instead.
    """
    finer = (
        mesh_for_spacing(lattice, spacing / 2.0)
        if grid is None
        else (2 * mesh[0], 2 * mesh[1], 2 * mesh[2])
    )
    return None if tuple(finer) == tuple(mesh) else finer


def threshold_walk(tried: frozenset[tuple[float, float]]) -> list[tuple[float, float]]:
    """The published grid, minimum outer and maximum inner, minus what was already tried."""
    walk = []
    for minimum in PROJECTABILITY_MINIMUM_LADDER:
        for maximum in PROJECTABILITY_MAXIMUM_LADDER:
            if (minimum, maximum) not in tried and minimum < maximum:
                walk.append((minimum, maximum))
    return walk


def window_holds_unrepresentable_states(results: InterpolationResults) -> bool:
    """Whether the classifier reported bands it kept out of an automatic metal window."""
    verdict = results.interpolation.disentanglement if results.interpolation else None
    return verdict is not None and bool(
        verdict.frozen_window_automatic and verdict.metallic and verdict.bands_excluded_from_freeze
    )


def one_figure_up(value: float) -> float:
    """*value* rounded up to one significant figure, so a tolerance reads as 7e-06."""
    exponent = math.floor(math.log10(value))
    mantissa = math.ceil(value / 10.0**exponent - 1.0e-9)
    return float(f"{mantissa}e{exponent}")


def last_disentanglement_spreads(failure_text: str) -> list[float]:
    """The gauge-invariant spreads a disentanglement that did not converge reached, oldest first.

    Read off the ``diagnostic_spreads_bohr_squared:`` line the utility prints
    after the code, where they are numbers and not part of a sentence.
    """
    match = _SPREADS_LINE.search(failure_text)
    return [float(value) for value in _NUMBER.findall(match.group(1))] if match else []


def loosened_disentanglement_tolerance(
    failure_text: str, tolerance: float = DEFAULT_DISENTANGLEMENT_TOLERANCE
) -> float | None:
    """The tolerance a disentanglement that ran out of iterations would have converged under.

    The failure prints the spreads of its last five iterations. A tolerance a
    margin above the relative change among them is one the same iteration had
    already met by the time it gave up, and the iteration is deterministic, so
    a retry under it stops no later than the failure did: it costs no more, and
    what it returns is judged by the gate like any other model. A larger budget
    is the other answer, and on the augmented campaign rows 4000 iterations
    already took one and a half to two hours.

    None when there are no spreads to read, when the change is too large to
    call the iteration settled (``MAXIMUM_DISENTANGLEMENT_TOLERANCE``), or when
    the tolerance would be no looser than *tolerance*, the one that failed.
    """
    spreads = last_disentanglement_spreads(failure_text)
    if len(spreads) < 2:
        return None
    change = max(max(spreads) - min(spreads), PRINTED_SPREAD_RESOLUTION_BOHR_SQUARED)
    loosened = one_figure_up(
        LOOSENED_TOLERANCE_MARGIN * change / max(abs(spreads[-1]), SMALLEST_SPREAD_BOHR_SQUARED)
    )
    if loosened > MAXIMUM_DISENTANGLEMENT_TOLERANCE or loosened <= tolerance:
        return None
    return loosened
