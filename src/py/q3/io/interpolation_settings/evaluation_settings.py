"""q3.io.interpolation_settings.evaluation_settings — :class:`EvaluationSettings`, one model of the ``control.interpolation`` block."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field

from q3.io.interpolation_settings.interpolation_settings import InterpolationSettings
from q3.io.interpolation_settings.mesh import Mesh

IterationBudget = Annotated[int, Field(ge=1)]
Projectability = Annotated[float, Field(ge=0.0, le=1.0)]


class EvaluationSettings(InterpolationSettings):
    """The evaluation request ``q3_utility interpolate`` reads: the subspace, the gauge, the DOS.

    Disentanglement and its thresholds belong here rather than on the capture:
    the record stores the raw window, so the same capture serves a run with
    disentanglement and a run without.
    """

    #: The dense mesh for the interpolated density of states, [n1, n2, n3].
    dos_mesh: Mesh | None = None
    #: How the N states are chosen out of the window: as they stand, or by classification.
    disentanglement: Literal["none", "projectability"] | None = None
    disentanglement_maximum_iterations: IterationBudget | None = None
    disentanglement_convergence_tolerance: Annotated[float, Field(gt=0.0)] | None = None
    #: How the gauge inside the subspace is fixed: "auto" localises where the record allows.
    localization: Literal["none", "auto", "maximal"] | None = None
    localization_maximum_iterations: IterationBudget | None = None
    projectability_minimum: Projectability | None = None
    projectability_maximum: Projectability | None = None
    #: In electronvolts above the Fermi level, or "auto" for the classifier's own rule.
    frozen_window_above_fermi_ev: float | Literal["auto"] | None = None
