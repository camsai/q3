"""q3.io.interpolation_results — the results file of ``q3_utility interpolate``, typed.

One contract, on the Python side. The models here mirror
``schemas/q3/interpolation_results.json`` block for block and key for key --
the same schema the engine's writer is generated from -- and
``tests/python/test_interpolation_results.py`` holds them to it, so a key
renamed in the schema without a rename here fails at unit-test speed rather
than in a driver at campaign speed.

Every field the engine writes defaults to ``None``. The schema's ``required``
is the producer's guarantee, present in every file the utility writes; it is
not a reader's precondition, because the unit-test stub writes a subset, an
older file lacks the blocks added since, and the workflow reads all of them
through this one model. Unknown keys are kept (``extra="allow"``) for the same
reason in the other direction: a newer engine's key round-trips instead of
being dropped.

The one block whose ``None`` values are data is the ladder: a raised attempt
has no band distance, and the file says so with a null the reader can tell
from "not recorded". :meth:`InterpolationResults.to_document` keeps those
nulls and drops every other absent field, which is exactly the file the
engine wrote plus what the workflow added.
"""

from q3.io.interpolation_results.band_distance_report import BandDistanceReport
from q3.io.interpolation_results.density_of_states import DensityOfStates
from q3.io.interpolation_results.disentanglement import Disentanglement
from q3.io.interpolation_results.excluded_band import ExcludedBand
from q3.io.interpolation_results.hamiltonian_decay import HamiltonianDecay
from q3.io.interpolation_results.interpolation import Interpolation
from q3.io.interpolation_results.interpolation_results import InterpolationResults
from q3.io.interpolation_results.ladder_attempt import LadderAttempt
from q3.io.interpolation_results.quality import Quality
from q3.io.interpolation_results.quality_centers import QualityCenters
from q3.io.interpolation_results.quality_spread import QualitySpread
from q3.io.interpolation_results.results_block import ResultsBlock
from q3.io.interpolation_results.spread import Spread

__all__ = [
    "BandDistanceReport",
    "DensityOfStates",
    "Disentanglement",
    "ExcludedBand",
    "HamiltonianDecay",
    "Interpolation",
    "InterpolationResults",
    "LadderAttempt",
    "Quality",
    "QualityCenters",
    "QualitySpread",
    "ResultsBlock",
    "Spread",
]
