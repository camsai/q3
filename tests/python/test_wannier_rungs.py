"""tests/python/test_wannier_rungs.py — each rung's condition and what it adopts, against a scripted evaluation.

The workflow tests drive the whole ladder through stub binaries; these hold
the rungs one at a time to their contracts: when each applies, what it
promotes, what it records without adopting, and that the state a rung hands
on is a new value rather than the old one changed.
"""

import pytest
from q3.io.interpolation.results.classes import InterpolationResults, LadderAttempt
from q3.io.interpolation_settings import EvaluationSettings
from q3.workflows.band_distance import eigenvalues_ev
from q3.workflows.wannier.accept_gate import AcceptGate
from q3.workflows.wannier.attempt_status import AttemptStatus
from q3.workflows.wannier.diagnostic import Diagnostic
from q3.workflows.wannier.ladder_state import LadderState
from q3.workflows.wannier.rung_name import RungName
from q3.workflows.wannier.rungs import (
    LADDER,
    StalledRung,
    ThresholdsRung,
    UnrepresentableWindowRung,
    WindowRung,
)
from q3.workflows.wannier.settings import (
    DEFAULT_LOCALIZATION_ITERATIONS,
    PROJECTABILITY_MAXIMUM_LADDER,
)
from q3.workflows.wannier.utils import finer_mesh, threshold_walk

MESH = (4, 4, 4)
DEFAULT_SETTINGS = EvaluationSettings(
    localization="auto", projectability_minimum=0.01, projectability_maximum=0.95
)


def _model(results: dict | None) -> InterpolationResults | None:
    return InterpolationResults.model_validate(results) if results is not None else None


class ScriptedEvaluation:
    """An Evaluation whose attempts follow a script: each entry says what the next attempt becomes."""

    def __init__(self, script, maximum_threshold_attempts=None):
        self.script = list(script)
        self.calls = []
        self.mesh = MESH
        self.maximum_threshold_attempts = maximum_threshold_attempts

    def attempt(self, state, rung, settings):
        entry = self.script.pop(0)
        self.calls.append((rung, settings.to_block()))
        status = AttemptStatus(entry["status"])
        record = LadderAttempt(
            rung=rung,
            grid=list(MESH),
            settings=settings.to_block(),
            status=status,
            reason=entry.get("reason", ""),
            eta_ev=entry.get("eta_ev"),
            eta_maximum_ev=entry.get("eta_maximum_ev"),
            localized=entry.get("localized"),
            localization_converged=entry.get("converged"),
        )
        results = None if status.value.startswith("error:") else _model(entry.get("results", {}))
        return state.with_attempt(record, results, entry.get("diagnostic"))


def _state(status, *, converged=None, results=None, diagnostic=None, reason="", eta_ev=None):
    record = LadderAttempt(
        rung=RungName.DEFAULT,
        grid=list(MESH),
        settings=DEFAULT_SETTINGS.to_block(),
        status=AttemptStatus(status),
        reason=reason,
        eta_ev=eta_ev,
        localized=True if converged is not None else None,
        localization_converged=converged,
    )
    return LadderState(
        record=record,
        results=_model(results),
        diagnostic=diagnostic,
        settings=DEFAULT_SETTINGS,
        tried=frozenset({(0.01, 0.95)}),
        attempts=(record,),
    )


def test_a_state_before_its_first_attempt_has_no_record_and_answers_no_to_everything():
    state = LadderState(settings=DEFAULT_SETTINGS, tried=frozenset())
    assert state.record is None and state.status is None
    assert not (state.accepted or state.evaluated or state.stalled or state.short_of_the_gate)


def test_with_attempt_extends_the_history_without_touching_the_state_it_came_from():
    first = _state("retry", converged=False, results={})
    second = first.with_attempt(first.record, _model({}), None)
    third = second.with_attempt(second.record, None, Diagnostic.GAUGE_SINGULAR)
    assert len(first.attempts) == 1
    assert len(second.attempts) == 2
    assert len(third.attempts) == 3
    assert third.results is None and third.diagnostic is Diagnostic.GAUGE_SINGULAR
    assert first.adopting(DEFAULT_SETTINGS.updated(localization="maximal")).settings != (
        first.settings
    )
    assert first.stopping("no").stop_reason == "no" and first.stop_reason is None


def test_settings_are_updated_by_validation_so_a_misspelt_threshold_fails_at_once():
    assert DEFAULT_SETTINGS.updated(projectability_maximum=0.9).projectability_maximum == 0.9
    assert DEFAULT_SETTINGS.updated(localization=None).localization is None
    with pytest.raises(ValueError):
        DEFAULT_SETTINGS.updated(projectability_maximun=0.9)
    with pytest.raises(ValueError):
        DEFAULT_SETTINGS.updated(localization="always")


def test_the_ladder_is_walked_in_this_order():
    assert LADDER == (StalledRung, UnrepresentableWindowRung, WindowRung, ThresholdsRung)
    assert [rung.name for rung in LADDER] == ["stalled", "window", "window", "thresholds"]


def test_the_stalled_rung_demands_localisation_with_twice_the_budget():
    evaluation = ScriptedEvaluation([{"status": "accept", "converged": True}])
    rung = StalledRung(evaluation)
    assert not rung.applies(_state("retry", converged=True, results={}))
    assert not rung.applies(_state("error:infeasible"))
    state = _state("retry", converged=False, results={})
    assert rung.applies(state)

    after = rung.attempt(state)
    assert evaluation.calls[0][0] == "stalled"
    assert evaluation.calls[0][1]["localization"] == "maximal"
    assert evaluation.calls[0][1]["localization_maximum_iterations"] == (
        2 * DEFAULT_LOCALIZATION_ITERATIONS
    )
    assert after.accepted and after.settings.localization == "maximal"
    assert after.stop_reason is None


def test_a_demand_the_localiser_refuses_ends_the_ladder():
    evaluation = ScriptedEvaluation(
        [{"status": "error:localization", "reason": "Error: no overlaps"}]
    )
    after = StalledRung(evaluation).attempt(_state("retry", converged=False, results={}))
    assert after.stop_reason == "Error: no overlaps"
    assert after.settings == DEFAULT_SETTINGS, "a refused demand promotes nothing"


def test_the_unrepresentable_window_rung_freezes_the_occupied_manifold_when_it_helps():
    unrepresentable = {
        "construction": {
            "disentanglement": {
                "frozen_window_automatic": True,
                "metallic": True,
                "bands_excluded_from_freeze": [{"band": 4, "kpoints": 3}],
            }
        }
    }
    rung = UnrepresentableWindowRung(ScriptedEvaluation([{"status": "retry", "converged": True}]))
    assert not rung.applies(_state("retry", converged=False, results={}))
    state = _state("retry", converged=False, results=unrepresentable)
    assert rung.applies(state)
    after = rung.attempt(state)
    assert after.settings.frozen_window_above_fermi_ev == 0.0

    stalled_again = UnrepresentableWindowRung(
        ScriptedEvaluation([{"status": "retry", "converged": False}])
    ).attempt(state)
    assert stalled_again.settings.frozen_window_above_fermi_ev is None


def test_the_window_rung_lowers_on_the_classifiers_code_and_on_nothing_else():
    by_code = _state("error:infeasible", diagnostic=Diagnostic.FROZEN_WINDOW_OVERFULL)
    by_words_alone = _state("error:infeasible", reason="k-point 3 freezes 11 bands")
    coded_otherwise = _state(
        "error:infeasible", diagnostic=Diagnostic.GAUGE_SINGULAR, reason="freezes, it says"
    )
    evaluation = ScriptedEvaluation(
        [{"status": "error:infeasible"}, {"status": "retry", "converged": True, "eta_ev": 0.02}]
    )
    rung = WindowRung(evaluation)
    assert rung.applies(by_code)
    assert not rung.applies(by_words_alone), "a sentence routes nothing"
    assert not rung.applies(coded_otherwise), "a code that is not the window's is believed"

    after = rung.attempt(by_code)
    assert [call[1]["frozen_window_above_fermi_ev"] for call in evaluation.calls] == [1.0, 0.0]
    assert after.settings.frozen_window_above_fermi_ev == 0.0
    assert len(after.attempts) == 3


def test_the_window_rung_widens_a_converged_model_and_keeps_only_what_improves():
    judged = {"construction": {"disentanglement": {"frozen_window_above_fermi_ev": 2.0}}}
    state = _state("retry", converged=True, results=judged, eta_ev=0.020)
    evaluation = ScriptedEvaluation(
        [
            {"status": "retry", "converged": True, "eta_ev": 0.015, "results": judged},
            {"status": "retry", "converged": True, "eta_ev": 0.030, "results": judged},
        ]
    )
    rung = WindowRung(evaluation)
    assert rung.applies(state)

    after = rung.attempt(state)
    assert [call[1]["frozen_window_above_fermi_ev"] for call in evaluation.calls] == [3.0, 4.0]
    assert after.record.eta_ev == 0.015, "the wider window that improved is the model judged"
    assert after.settings.frozen_window_above_fermi_ev == 3.0
    assert len(after.attempts) == 3, "the attempt that did not improve is still history"


def test_the_thresholds_rung_walks_the_published_grid_skipping_what_was_tried():
    walk = threshold_walk(frozenset({(0.01, 0.95)}))
    assert walk[0] == (0.01, 0.99)
    assert (0.01, 0.95) not in walk
    assert len(walk) == 2 * len(PROJECTABILITY_MAXIMUM_LADDER) - 1

    evaluation = ScriptedEvaluation(
        [{"status": "retry", "converged": True}, {"status": "accept", "converged": True}],
        maximum_threshold_attempts=5,
    )
    rung = ThresholdsRung(evaluation)
    state = _state("retry", converged=True, results={})
    assert rung.applies(state)
    assert not rung.applies(_state("accept", converged=True, results={}))
    assert not rung.applies(_state("error:localization"))

    after = rung.attempt(state)
    assert [call[1]["projectability_maximum"] for call in evaluation.calls] == [0.99, 0.98]
    assert after.accepted
    assert (0.01, 0.98) in after.tried
    assert after.settings == DEFAULT_SETTINGS, "a pair is tried, not promoted"


def test_the_gate_judges_the_gauge_before_the_distance():
    gate = AcceptGate(eta_ev=0.010, eta_maximum_ev=0.100, energy_offset_ev=2.0, smearing_ev=0.1)
    document = {
        "properties": {
            "eigenvalues_at_kpoints": {
                "name": "eigenvalues_at_kpoints",
                "units": "Ry",
                "values": [{"eigenvalues": [{"energies": [-0.5, 0.25]}]}],
            }
        },
        "construction": {"spread": {"localized": True, "localization_converged": True}},
    }
    reference = eigenvalues_ev(_model(document))
    verdict = gate.judge(_model(document), reference, 0.0)
    assert verdict.status is AttemptStatus.ACCEPT and "gauge stationary" in verdict.reason

    document["construction"]["spread"]["localization_converged"] = False
    verdict = gate.judge(_model(document), reference, 0.0)
    assert verdict.status is AttemptStatus.RETRY and "stalled" in verdict.reason
    assert (verdict.localized, verdict.localization_converged) == (True, False)

    document["construction"]["spread"]["localized"] = False
    verdict = gate.judge(_model(document), reference, 0.0)
    assert verdict.status is AttemptStatus.RETRY and "not localised" in verdict.reason


def test_finer_mesh_doubles_an_explicit_grid():
    assert finer_mesh((4, 4, 4), (4, 4, 4), {}, 0.2) == (8, 8, 8)
