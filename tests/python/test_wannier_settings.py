"""tests/python/test_wannier_settings.py — the protocol's copies of the engine's defaults are the engine's.

``q3.workflows.wannier.settings`` states four numbers the manifest's schema
also states, because the ladder starts from the engine's defaults and doubles,
loosens or walks away from them: the two projectability thresholds, the
localiser's budget and the disentangler's tolerance. A default that moved on
one side and not the other would make the first rung evaluate something other
than what a manifest that says nothing evaluates. The published frozen window
is held to the classifier's own constant the same way.
"""

import json
import re
from pathlib import Path

from q3.workflows.wannier import settings
from q3.workflows.wannier.attempt_status import AttemptStatus
from q3.workflows.wannier.route import Route
from q3.workflows.wannier.rung_name import RungName

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent.parent
SCHEMA_BLOCK = json.loads((REPOSITORY_ROOT / "schemas" / "q3" / "input.json").read_text())[
    "properties"
]["control"]["properties"]["interpolation"]["properties"]
LADDER_ATTEMPT = json.loads(
    (REPOSITORY_ROOT / "schemas" / "q3" / "interpolation_results.json").read_text()
)["properties"]["interpolation"]["properties"]["quality"]["properties"]["ladder"]["items"][
    "properties"
]
THRESHOLDS_HEADER = (
    REPOSITORY_ROOT
    / "src/layer_4_computation/interpolator/method/wannier/settings/band_classification_thresholds.hpp"
)


def test_the_defaults_the_ladder_starts_from_are_the_manifests():
    assert (
        settings.DEFAULT_PROJECTABILITY_MINIMUM == SCHEMA_BLOCK["projectability_minimum"]["default"]
    )
    assert (
        settings.DEFAULT_PROJECTABILITY_MAXIMUM == SCHEMA_BLOCK["projectability_maximum"]["default"]
    )
    assert (
        settings.DEFAULT_LOCALIZATION_ITERATIONS
        == SCHEMA_BLOCK["localization_maximum_iterations"]["default"]
    )
    assert (
        settings.DEFAULT_DISENTANGLEMENT_TOLERANCE
        == SCHEMA_BLOCK["disentanglement_convergence_tolerance"]["default"]
    )
    assert settings.STALLED_LOCALIZATION_ITERATIONS == 2 * settings.DEFAULT_LOCALIZATION_ITERATIONS


def test_the_published_window_is_the_classifiers_own():
    declared = re.search(
        r"AUTOMATIC_WINDOW_RYDBERG\s*=\s*([0-9.]+)\s*/\s*constants::RY_TO_EV",
        THRESHOLDS_HEADER.read_text(),
    )
    assert declared is not None, "the classifier's automatic window moved; point this test at it"
    assert float(declared.group(1)) == settings.DEFAULT_FROZEN_WINDOW_EV
    assert SCHEMA_BLOCK["frozen_window_above_fermi_ev"]["default"] == "auto"


def test_the_windows_the_ladder_walks_bracket_the_published_one():
    assert all(window < settings.DEFAULT_FROZEN_WINDOW_EV for window in settings.LOWERED_WINDOWS_EV)
    assert settings.LOWERED_WINDOWS_EV[-1] == settings.OCCUPIED_MANIFOLD_WINDOW_EV
    assert all(step > 0.0 for step in settings.RAISED_WINDOW_STEPS_EV)


def test_the_words_of_the_history_are_the_ones_the_results_file_documents():
    assert [name.value for name in RungName] == [
        "default",
        "unconverged",
        "stalled",
        "window",
        "thresholds",
        "mesh",
    ]
    assert {status.value for status in AttemptStatus} == {"accept", "retry", "recapture"} | {
        "error:" + route.value for route in Route
    }


def test_the_results_schema_names_every_word_the_history_is_written_in():
    rungs = LADDER_ATTEMPT["rung"]["description"]
    assert all(f'"{name.value}"' in rungs for name in RungName), rungs
    statuses = LADDER_ATTEMPT["status"]["description"]
    assert all(f'"{word}"' in statuses for word in ("accept", "retry", "recapture")), statuses
    assert '"error:<route>"' in statuses
    assert all(f'"{route.value}"' in statuses for route in Route), statuses
