"""q3.workflows.wannier — the automated Wannier protocol: capture, judge, retry.

``run_interpolation_workflow`` runs the mechanics with whatever it is told.
This module is the protocol around them, the part that decides and checks
without a person in the loop (Vitale et al., npj Comput. Mater. 6, 66 (2020);
Qiao, Pizzi, Marzari, npj Comput. Mater. 9, 208 (2023)):

1. the k-mesh from a reciprocal-space spacing (``mesh_for_spacing``, 0.2 per
   angstrom), never written by hand;
2. the trial orbitals and their count from the pseudopotentials' own atomic
   shells, less the empty channels an augmented dataset repeats for an l its
   atom already has in valence (``trial_shells``), so the band window is
   ``ceil(1.5 N) + 4`` and the subspace is chosen by projectability
   disentanglement with the published frozen window, E_F + 2 eV for metal and
   insulator alike -- the same window the band distance is measured on;
3. a reference calculation the model is judged against -- a path, if one is
   asked for, otherwise a set of k-points half a mesh step off every mesh
   point, where the error is largest -- and the band distance eta between the
   two;
4. accept, or climb the retry ladder: a disentanglement that runs out of
   iterations is retried under the tolerance its own last spreads call for,
   which then holds for the rest of the ladder; a stalled localisation is
   retried as a demand with twice the budget; a metal whose window holds states the
   dataset cannot represent (the classifier reports them) and still stalls is
   retried with the occupied manifold alone frozen; a window that freezes
   more states than there are Wannier functions is lowered, and a converged
   model that still misses the gate gets a wider one; what is left walks the
   published grid of projectability thresholds; a model still too far off has
   its mesh spacing halved and is recaptured; and what survives none of that
   is rejected with the whole history, never shipped silently.

The singular-gauge rule lives in the classifier
itself (``BandClassifier``): a state the trial orbitals cannot reach is never
frozen, and the exclusion is reported, so that rung needs no retry here.

Every rung but the last re-runs ``q3_utility interpolate`` against the saved
record, which takes seconds and no DFT; only the mesh rung recaptures.

This module is the facade, and the package is one class per module.
``driver`` is the entry point; ``protocol`` runs the mesh loop under a
caller's ``protocol_options`` and returns a ``workflow_result``; ``settings``
holds every fixed choice and ``utils`` the helpers with no state;
``ladder_state``, ``accept_gate`` and ``verdict`` are what the rungs share,
``evaluation`` is the one way a saved record is evaluated, and ``rungs`` is the
ladder itself, a module per rung; ``diagnostic`` spells the engine's failure
codes, ``route`` what the ladder does about each, and ``attempt_status`` and
``rung_name`` the words the history is written in; ``trial_shells`` is the rule
that chooses trial orbitals among a dataset's shells, which
``q3.io.pseudo_atomic_shells`` reads, and ``trial_orbitals`` is that rule over
every species of a material.

Example::

    from q3.workflows.wannier import run_automatic_wannier_workflow

    result = run_automatic_wannier_workflow(
        "examples/band_structure--metal--copper--wannier/nc_scf.yml"
    )
    result["status"]  # "accept" | "reject"
    result["band_distance"]  # {"eta_ev": ..., "eta_maximum_ev": ..., ...}
    result["attempts"]  # one entry per rung fired, with its settings and outcome
    result["trial_shells"]  # {"Cu": [0, 1, 2, 3, 4]}: the PP_CHI shells the capture used
"""

from __future__ import annotations

from q3.io.interpolation.results.classes import LadderAttempt
from q3.io.pseudo_atomic_shells import PseudoAtomicShell, pseudo_atomic_shells
from q3.workflows.wannier.accept_gate import AcceptGate
from q3.workflows.wannier.attempt_status import AttemptStatus
from q3.workflows.wannier.diagnostic import Diagnostic
from q3.workflows.wannier.driver import run_automatic_wannier_workflow
from q3.workflows.wannier.evaluation import Evaluation
from q3.workflows.wannier.ladder_state import LadderState
from q3.workflows.wannier.protocol import WannierProtocol
from q3.workflows.wannier.protocol_options import ProtocolOptions
from q3.workflows.wannier.route import Route
from q3.workflows.wannier.rung_name import RungName
from q3.workflows.wannier.rungs import LADDER
from q3.workflows.wannier.settings import (
    ACCEPT_ETA_EV,
    ACCEPT_ETA_MAXIMUM_EV,
    BAND_WINDOW_EXTRA,
    BAND_WINDOW_FACTOR,
    DEFAULT_DISENTANGLEMENT_TOLERANCE,
    DEFAULT_FROZEN_WINDOW_EV,
    DEFAULT_LOCALIZATION_ITERATIONS,
    DEFAULT_PROJECTABILITY_MAXIMUM,
    DEFAULT_PROJECTABILITY_MINIMUM,
    LOOSENED_TOLERANCE_MARGIN,
    LOWERED_WINDOWS_EV,
    MAXIMUM_DISENTANGLEMENT_TOLERANCE,
    MAXIMUM_REFERENCE_KPOINTS,
    PRINTED_SPREAD_RESOLUTION_BOHR_SQUARED,
    PROJECTABILITY_MAXIMUM_LADDER,
    PROJECTABILITY_MINIMUM_LADDER,
    RAISED_WINDOW_STEPS_EV,
)
from q3.workflows.wannier.trial_orbitals import TrialOrbitals
from q3.workflows.wannier.trial_shells import (
    number_of_trial_orbitals,
    pseudo_atomic_orbitals_in,
    trial_shells,
)
from q3.workflows.wannier.utils import (
    band_window,
    last_disentanglement_spreads,
    loosened_disentanglement_tolerance,
    off_mesh_reference_kpoints,
)
from q3.workflows.wannier.verdict import Verdict
from q3.workflows.wannier.workflow_result import WorkflowResult

__all__ = [
    "ACCEPT_ETA_EV",
    "ACCEPT_ETA_MAXIMUM_EV",
    "AcceptGate",
    "AttemptStatus",
    "BAND_WINDOW_EXTRA",
    "BAND_WINDOW_FACTOR",
    "DEFAULT_DISENTANGLEMENT_TOLERANCE",
    "DEFAULT_FROZEN_WINDOW_EV",
    "DEFAULT_LOCALIZATION_ITERATIONS",
    "DEFAULT_PROJECTABILITY_MAXIMUM",
    "DEFAULT_PROJECTABILITY_MINIMUM",
    "Diagnostic",
    "Evaluation",
    "LADDER",
    "LOOSENED_TOLERANCE_MARGIN",
    "LOWERED_WINDOWS_EV",
    "LadderAttempt",
    "LadderState",
    "MAXIMUM_DISENTANGLEMENT_TOLERANCE",
    "MAXIMUM_REFERENCE_KPOINTS",
    "PRINTED_SPREAD_RESOLUTION_BOHR_SQUARED",
    "PROJECTABILITY_MAXIMUM_LADDER",
    "PROJECTABILITY_MINIMUM_LADDER",
    "ProtocolOptions",
    "PseudoAtomicShell",
    "RAISED_WINDOW_STEPS_EV",
    "Route",
    "RungName",
    "TrialOrbitals",
    "Verdict",
    "WannierProtocol",
    "WorkflowResult",
    "band_window",
    "last_disentanglement_spreads",
    "loosened_disentanglement_tolerance",
    "number_of_trial_orbitals",
    "off_mesh_reference_kpoints",
    "pseudo_atomic_orbitals_in",
    "pseudo_atomic_shells",
    "run_automatic_wannier_workflow",
    "trial_shells",
]
