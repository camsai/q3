"""q3.io.interpolation_settings — the ``control.interpolation`` block, typed, one model per step.

One block serves two steps of the same feature, and which step is meant is
decided by ``control.calculation``, not by the block: on an NSCF run it is
the build request the engine reads (:class:`CaptureSettings`), on
``calculation: interpolation`` the evaluation request ``q3_utility interpolate``
reads (:class:`EvaluationSettings`). ``schemas/q3/input.json`` describes the
keys and their defaults; ``tests/python/test_interpolation_settings.py`` holds
these models to it, and the schema's defaults to the engine's own.

Each writes itself with :meth:`InterpolationSettings.to_block`: the keys that
were set and nothing else, so a manifest carries only the decisions that were
made and the engine's defaults apply to the rest. Unknown keys are refused, so
a rung of the retry ladder that misspells a threshold fails at once rather
than evaluating the model it thought it had changed; and each field holds the
schema's bounds, so a block a model wrote is one the parser accepts. ``mesh``
is the rule the two meshes of the block share: three positive integers.
"""

from q3.io.interpolation_settings.capture_settings import CaptureSettings
from q3.io.interpolation_settings.evaluation_settings import EvaluationSettings
from q3.io.interpolation_settings.interpolation_settings import InterpolationSettings
from q3.io.interpolation_settings.mesh import Grid, Mesh, validated_grid

__all__ = [
    "CaptureSettings",
    "EvaluationSettings",
    "Grid",
    "InterpolationSettings",
    "Mesh",
    "validated_grid",
]
