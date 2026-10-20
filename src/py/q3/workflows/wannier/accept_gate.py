"""q3.workflows.wannier.accept_gate — :class:`AcceptGate`, the judge of the retry ladder.

The thresholds a model is held to, fixed for the whole run, and the verdict on
one evaluation against them.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, ConfigDict

from q3.io.interpolation_results import InterpolationResults
from q3.workflows.band_distance import BandDistance, band_distance, eigenvalues_electronvolt
from q3.workflows.wannier.attempt_status import AttemptStatus
from q3.workflows.wannier.settings import MILLIELECTRONVOLT_PER_ELECTRONVOLT
from q3.workflows.wannier.verdict import Verdict


class AcceptGate(BaseModel):
    """The thresholds a model is judged against, fixed for the whole run."""

    model_config = ConfigDict(frozen=True)

    eta_ev: float
    eta_maximum_ev: float
    energy_offset_ev: float
    smearing_ev: float

    def judge(
        self, results: InterpolationResults, reference_ev: np.ndarray, fermi_energy_ev: float
    ) -> Verdict:
        """eta against the reference, and the verdict on this evaluation."""
        distance = band_distance(
            reference_ev,
            eigenvalues_electronvolt(results),
            fermi_energy_ev,
            energy_offset=self.energy_offset_ev,
            smearing=self.smearing_ev,
        )
        spread = results.interpolation.spread if results.interpolation is not None else None
        localized = spread.localized if spread is not None else None
        converged = spread.localization_converged if spread is not None else None
        status, reason = self._status_and_reason(distance, localized, converged)
        return Verdict(
            distance=distance,
            status=status,
            reason=reason,
            # As the file gives them for a gauge that was not localised -- absent stays
            # absent -- and as plain booleans once it was.
            localized=localized if not localized else True,
            localization_converged=converged if not localized else bool(converged),
        )

    def _status_and_reason(
        self, distance: BandDistance, localized: bool | None, converged: bool | None
    ) -> tuple[AttemptStatus, str]:
        """The first thing this model falls short on, or that it falls short on nothing."""
        if not localized:
            return AttemptStatus.RETRY, "the gauge was not localised"
        if not converged:
            return AttemptStatus.RETRY, "the localisation stalled short of a stationary gauge"
        eta_mev = distance.eta * MILLIELECTRONVOLT_PER_ELECTRONVOLT
        eta_maximum_mev = distance.eta_maximum * MILLIELECTRONVOLT_PER_ELECTRONVOLT
        if distance.eta > self.eta_ev:
            gate_mev = self.eta_ev * MILLIELECTRONVOLT_PER_ELECTRONVOLT
            return (
                AttemptStatus.RETRY,
                f"eta = {eta_mev:.1f} meV is above the {gate_mev:.0f} meV gate",
            )
        if distance.eta_maximum > self.eta_maximum_ev:
            gate_mev = self.eta_maximum_ev * MILLIELECTRONVOLT_PER_ELECTRONVOLT
            return (
                AttemptStatus.RETRY,
                f"eta_max = {eta_maximum_mev:.1f} meV is above the {gate_mev:.0f} meV gate",
            )
        return (
            AttemptStatus.ACCEPT,
            f"eta = {eta_mev:.2f} meV, eta_max = {eta_maximum_mev:.1f} meV, gauge stationary",
        )
