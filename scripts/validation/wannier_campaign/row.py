"""One row of the campaign's table."""

from __future__ import annotations

from pydantic import BaseModel
from q3.io.interpolation.results.classes import BandDistanceReport, Quality

from .case import CampaignCase
from .utils import millielectronvolts


class CampaignRow(BaseModel):
    """One row per case; an error is a row carrying its reason, not a gap in the table.

    A row that ran carries every column, a null where the run produced nothing
    for it; a row that raised carries only its reason, which is why the dict a
    row becomes leaves unset fields out rather than writing nulls for them.
    """

    name: str
    family: str
    formula: str
    variant: str
    atoms: int
    seconds: float
    note: str
    status: str
    reason: str
    wannier_functions: int | None = None
    bands: int | None = None
    grid: str | None = None
    rung: str
    attempts: int
    eta_first_mev: float | None = None
    eta_best_mev: float | None = None
    eta_mev: float | None = None
    eta_max_mev: float | None = None
    spread_variance_bohr_squared: float | None = None
    decay_ratio: float | None = None
    ladder: list[dict] | None = None

    @classmethod
    def from_error(cls, case: CampaignCase, seconds: float, error: str | None) -> CampaignRow:
        """The row of a case whose run raised."""
        return cls(
            **cls._identity(case, seconds),
            status="error",
            reason=error or "unknown",
            rung="",
            attempts=0,
        )

    @classmethod
    def from_result(cls, case: CampaignCase, result: dict, seconds: float) -> CampaignRow:
        """The row of a case the workflow judged, from what it returned."""
        attempts = result["attempts"]
        accepted = [attempt for attempt in attempts if attempt["status"] == "accept"]
        etas = [attempt["eta_ev"] for attempt in attempts if attempt.get("eta_ev") is not None]
        distance = BandDistanceReport.model_validate(result.get("band_distance") or {})
        quality = Quality.model_validate(result.get("quality") or {})
        # eta on the first rung is what the protocol's defaults deliver; the best says
        # whether a retry helped; eta itself is the accepted attempt's, or the last one's.
        return cls(
            **cls._identity(case, seconds),
            status=result["status"],
            reason=result["reason"],
            wannier_functions=result["number_of_wannier_functions"],
            bands=result["number_of_bands"],
            grid="x".join(str(value) for value in result["grid"]),
            rung=(accepted[0] if accepted else attempts[-1] if attempts else {}).get("rung", ""),
            attempts=len(attempts),
            eta_first_mev=millielectronvolts(attempts[0].get("eta_ev") if attempts else None, 2),
            eta_best_mev=millielectronvolts(min(etas, default=None), 2),
            eta_mev=millielectronvolts(distance.eta_ev, 2),
            eta_max_mev=millielectronvolts(distance.eta_maximum_ev, 1),
            spread_variance_bohr_squared=(
                quality.spread.variance_bohr_squared if quality.spread is not None else None
            ),
            decay_ratio=(
                quality.hamiltonian_decay.ratio if quality.hamiltonian_decay is not None else None
            ),
            ladder=attempts,
        )

    def as_dict(self) -> dict:
        """The row as it is written to campaign.json: only the columns this row has."""
        return self.model_dump(exclude_unset=True)

    @staticmethod
    def _identity(case: CampaignCase, seconds: float) -> dict:
        return {
            "name": case.name,
            "family": case.family,
            "formula": case.formula,
            "variant": case.variant,
            "atoms": case.number_of_atoms,
            "seconds": round(seconds, 1),
            "note": case.note,
        }
