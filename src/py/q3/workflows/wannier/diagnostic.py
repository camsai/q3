"""q3.workflows.wannier.diagnostic — :class:`Diagnostic`, the engine's failure codes.

Spelled exactly as the engine's enum spells them
(``src/layer_4_computation/interpolator/method/wannier/diagnostic.hpp``); a test
reads that header and holds the two equal.
"""

from __future__ import annotations

from enum import StrEnum

from q3.errors import Q3RuntimeError, diagnostic_code
from q3.workflows.wannier.route import Route


class Diagnostic(StrEnum):
    """The engine's codes, spelled as its enum spells them."""

    DISENTANGLEMENT_UNCONVERGED = "disentanglement_unconverged"
    DISENTANGLEMENT_UNSUPPORTED = "disentanglement_unsupported"
    FROZEN_WINDOW_OVERFULL = "frozen_window_overfull"
    KEPT_BANDS_UNDERFULL = "kept_bands_underfull"
    GAUGE_SINGULAR = "gauge_singular"
    GAUGE_COLLAPSED = "gauge_collapsed"
    OVERLAPS_NOT_ORTHONORMAL = "overlaps_not_orthonormal"
    LOCALIZATION_UNSUPPORTED = "localization_unsupported"
    TWO_POINT_AXIS = "two_point_axis"

    @property
    def route(self) -> Route:
        """Which rung a failure with this code belongs to."""
        return _ROUTES[self]

    @classmethod
    def from_code(cls, code: str | None) -> Diagnostic | None:
        """The code as this module knows it; a newer engine's, or none, reads as None."""
        if code is None:
            return None
        try:
            return cls(code)
        except ValueError:
            return None

    @classmethod
    def from_message(cls, message: str) -> Diagnostic | None:
        """The code on the ``diagnostic:`` line of a failure's text, or None without one.

        The line may carry the logger's prefix; a code this module does not
        know (a newer engine's) reads as None, so the prose decides as it
        would for an older one.
        """
        return cls.from_code(diagnostic_code(message))

    @classmethod
    def of(cls, error: Q3RuntimeError) -> Diagnostic | None:
        """The code of a failed run: what the runner read off its output, or what the text says."""
        if error.diagnostic is not None:
            return cls.from_code(error.diagnostic)
        return cls.from_message(error.text)


_ROUTES = {
    Diagnostic.DISENTANGLEMENT_UNCONVERGED: Route.UNCONVERGED,
    Diagnostic.DISENTANGLEMENT_UNSUPPORTED: Route.LOCALIZATION,
    Diagnostic.FROZEN_WINDOW_OVERFULL: Route.INFEASIBLE,
    Diagnostic.KEPT_BANDS_UNDERFULL: Route.INFEASIBLE,
    Diagnostic.GAUGE_SINGULAR: Route.INFEASIBLE,
    Diagnostic.GAUGE_COLLAPSED: Route.INFEASIBLE,
    Diagnostic.OVERLAPS_NOT_ORTHONORMAL: Route.UNKNOWN,
    Diagnostic.LOCALIZATION_UNSUPPORTED: Route.LOCALIZATION,
    Diagnostic.TWO_POINT_AXIS: Route.LOCALIZATION,
}
