"""q3.workflows.wannier.rungs — the rungs of the retry ladder, one class per module, in order.

A rung is a decision over the ladder's state (:mod:`q3.workflows.wannier.ladder_state`):
whether it applies to the attempt the last rung left, and what to try -- which
evaluates the saved record again under different settings and promotes them
when they help. By contract a rung re-evaluates the SAVED record, with no
DFT: that invariant is what makes the ladder cheap, and it is why the mesh,
the one step that recaptures, is the protocol's outer loop and not a rung here.

:data:`LADDER` is the order. Each rung judges what the last one left, so the
order is the ladder; the protocol walks it once per mesh and stops early only
when a rung ends the whole ladder (``stop_reason``).
"""

from __future__ import annotations

from q3.workflows.wannier.rungs.rung import Rung
from q3.workflows.wannier.rungs.stalled_rung import StalledRung
from q3.workflows.wannier.rungs.thresholds_rung import ThresholdsRung
from q3.workflows.wannier.rungs.unrepresentable_window_rung import UnrepresentableWindowRung
from q3.workflows.wannier.rungs.window_rung import WindowRung

#: The rungs in the order the driver walks them, once per mesh.
LADDER: tuple[type[Rung], ...] = (
    StalledRung,
    UnrepresentableWindowRung,
    WindowRung,
    ThresholdsRung,
)

__all__ = [
    "LADDER",
    "Rung",
    "StalledRung",
    "ThresholdsRung",
    "UnrepresentableWindowRung",
    "WindowRung",
]
