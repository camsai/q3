"""q3.io.interpolation.results.classes — the results file of ``q3_utility interpolate``, typed.

The models are generated from ``schemas/q3/interpolation/results.json`` into
``q3.generated.interpolation.results`` (one module per schema file), the same schema the engine's writer is
generated from, and are named here as the workflow reads them. Every field is
optional and unknown keys are kept, so the unit-test stub's subset and a newer
engine's file both read. A key the file carries is written back as it was,
nulls included: a raised ladder attempt has no band distance, and the file
says so with a null a reader can tell from "not recorded".
"""

from q3.generated.interpolation.results import (
    InterpolationResults,
)
from q3.generated.interpolation.results.construction import (
    InterpolationResultsConstruction as Construction,
)
from q3.generated.interpolation.results.construction.disentanglement import (
    InterpolationResultsDisentanglement as Disentanglement,
)
from q3.generated.interpolation.results.construction.disentanglement.bands_excluded_from_freeze import (
    InterpolationResultsExcludedBand as ExcludedBand,
)
from q3.generated.interpolation.results.construction.spread import (
    InterpolationResultsSpread as Spread,
)
from q3.generated.interpolation.results.properties import (
    InterpolationResultsProperties as Properties,
)
from q3.generated.interpolation.results.quality import (
    InterpolationResultsQuality as Quality,
)
from q3.generated.interpolation.results.quality.band_distance import (
    InterpolationResultsBandDistance as BandDistanceReport,
)
from q3.generated.interpolation.results.quality.centers import (
    InterpolationResultsQualityCenters as QualityCenters,
)
from q3.generated.interpolation.results.quality.hamiltonian_decay import (
    InterpolationResultsHamiltonianDecay as HamiltonianDecay,
)
from q3.generated.interpolation.results.quality.ladder import (
    InterpolationResultsLadderAttempt as LadderAttempt,
)
from q3.generated.interpolation.results.quality.spread import (
    InterpolationResultsQualitySpread as QualitySpread,
)

__all__ = [
    "BandDistanceReport",
    "Construction",
    "Disentanglement",
    "ExcludedBand",
    "HamiltonianDecay",
    "InterpolationResults",
    "LadderAttempt",
    "Properties",
    "Quality",
    "QualityCenters",
    "QualitySpread",
    "Spread",
]
