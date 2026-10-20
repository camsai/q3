"""q3.workflows.wannier.settings — every fixed choice of the automated protocol, in one place.

The gate a model is held to, the published threshold grid and the frozen
windows the retry ladder walks, the budgets it doubles, the tolerances it
loosens a disentanglement by, the band window's rule, and the cap on the
reference set. A number that decides what the protocol does is here and
nowhere else, each with the reason it has the value it has; the rungs, the
gate and the driver import them.
"""

from __future__ import annotations

import numpy as np

# ── the accept gate ──────────────────────────────────────────────────────────
#: The accept gate: converged gauge, eta below this and eta_max below the next.
ACCEPT_ETA_EV = 0.010
ACCEPT_ETA_MAXIMUM_EV = 0.100
#: The gate's verdicts are read by people, who read band distances in millielectronvolts.
MILLIELECTRONVOLT_PER_ELECTRONVOLT = 1000.0

# ── the reference the model is judged against ────────────────────────────────
#: At most this many off-mesh reference k-points when no path is asked for.
MAXIMUM_REFERENCE_KPOINTS = 125
#: How the results file names the reference a band distance was measured on.
REFERENCE_ON_A_PATH = "path"
REFERENCE_OFF_THE_MESH = "half-shifted mesh"
#: How many times the mesh rung may halve the spacing and recapture.
DEFAULT_MAXIMUM_MESH_REFINEMENTS = 1

# ── the band window ──────────────────────────────────────────────────────────
#: The band window: this many times the Wannier count, plus this many.
BAND_WINDOW_FACTOR = 1.5
BAND_WINDOW_EXTRA = 4

# ── the ladder: thresholds, budgets, windows ─────────────────────────────────
#: The published optimisation grid, maximum varying fastest.
PROJECTABILITY_MINIMUM_LADDER = (0.01, 0.02)
PROJECTABILITY_MAXIMUM_LADDER = tuple(
    round(float(value), 4) for value in np.linspace(0.99, 0.85, 15)
)
DEFAULT_PROJECTABILITY_MINIMUM = 0.01
DEFAULT_PROJECTABILITY_MAXIMUM = 0.95
#: The localiser's default budget, and the one the stalled rung demands: twice that.
DEFAULT_LOCALIZATION_ITERATIONS = 3000
STALLED_LOCALIZATION_ITERATIONS = 2 * DEFAULT_LOCALIZATION_ITERATIONS
#: The published frozen window, E_F + 2 eV, which `auto` applies; the window
#: rung lowers it in these steps when it freezes too much, and widens it by
#: these steps when a converged model still misses the gate.
DEFAULT_FROZEN_WINDOW_EV = 2.0
LOWERED_WINDOWS_EV = (1.0, 0.0)
RAISED_WINDOW_STEPS_EV = (1.0, 2.0)
#: The window that freezes the occupied manifold and nothing above the Fermi level.
OCCUPIED_MANIFOLD_WINDOW_EV = 0.0

# ── a disentanglement that ran out of iterations ─────────────────────────────
#: The disentangler's own tolerance: the relative change in the gauge-invariant
#: spread, over five iterations, below which its subspace counts as converged.
DEFAULT_DISENTANGLEMENT_TOLERANCE = 1.0e-10
#: The engine prints the spreads it gave up on with six decimals, so a change
#: smaller than this cannot be read off the failure.
PRINTED_SPREAD_RESOLUTION_BOHR_SQUARED = 1.0e-6
#: How far above the printed drift a loosened tolerance sits: the engine's window
#: reaches one iteration further back than the five spreads it prints, and a
#: drift is rarely exactly linear.
LOOSENED_TOLERANCE_MARGIN = 3.0
#: A drift that would need a tolerance looser than this is an iteration that is
#: not settling at all, and it is not retried.
MAXIMUM_DISENTANGLEMENT_TOLERANCE = 1.0e-4
#: Floor on the spread a relative drift is measured against, so a spread of exactly zero has one.
SMALLEST_SPREAD_BOHR_SQUARED = 1.0e-30
