"""tests/python/test_interpolation_settings.py — the block models mirror the schema, and its defaults the engine's.

``control.interpolation`` is declared in ``schemas/q3/input.json`` with the
defaults the engine applies; ``q3.io.interpolation_settings`` writes the block
and ``src/layer_4_computation/interpolator/input_data/specification.hpp``
reads it, taking each knob's default from the settings struct that owns it
(``method/wannier/settings/``). The two models must cover the block's keys exactly, write only what
was set, refuse what the engine would not read, and the schema's defaults must
be the engine's -- a knob whose default moved on one side and not the other
would apply silently to every manifest that leaves it out.
"""

import json
import re
from pathlib import Path

import jsonschema
import pytest
from q3.io.interpolation_settings import CaptureSettings, EvaluationSettings

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent.parent
SCHEMA_BLOCK = json.loads((REPOSITORY_ROOT / "schemas" / "q3" / "input.json").read_text())[
    "properties"
]["control"]["properties"]["interpolation"]
SPECIFICATION = (
    REPOSITORY_ROOT / "src/layer_4_computation/interpolator/input_data/specification.hpp"
)
SETTINGS_DIRECTORY = (
    REPOSITORY_ROOT / "src/layer_4_computation/interpolator/method/wannier/settings"
)


def test_the_two_models_together_cover_the_block_and_nothing_else():
    covered = set(CaptureSettings.model_fields) | set(EvaluationSettings.model_fields)
    assert covered == set(SCHEMA_BLOCK["properties"])
    assert set(CaptureSettings.model_fields) & set(EvaluationSettings.model_fields) == {"method"}


def test_only_what_was_set_is_written():
    assert EvaluationSettings().to_block() == {"method": "wannier"}
    assert EvaluationSettings(dos_mesh=(4, 4, 4), projectability_maximum=0.9).to_block() == {
        "method": "wannier",
        "dos_mesh": [4, 4, 4],
        "projectability_maximum": 0.9,
    }
    assert CaptureSettings(grid=(2, 2, 2), export_formats=[]).to_block() == {
        "method": "wannier",
        "grid": [2, 2, 2],
    }
    assert CaptureSettings(grid=[4, 4, 4], trial_shells={"Si": [0, 2]}).to_block()[
        "trial_shells"
    ] == {"Si": [0, 2]}


def test_a_key_the_engine_does_not_read_is_refused():
    with pytest.raises(ValueError, match="projectability_maximmum"):
        EvaluationSettings(projectability_maximmum=0.9)
    with pytest.raises(ValueError, match="localization"):
        EvaluationSettings(localization="most")
    with pytest.raises(ValueError, match="grid"):
        CaptureSettings()


@pytest.mark.parametrize(
    "settings",
    [
        CaptureSettings(grid=(4, 4, 4)),
        CaptureSettings(
            grid=[6, 6, 4],
            bands=[5, 12],
            projection_centers=[(0, 0, 0), (0.25, 0.25, 0.25)],
            projection_decay_bohr=1.5,
            trial_shells={"Cu": [0, 1, 2, 3, 4], "Si": [0, 2]},
            export_formats=["wannier90"],
        ),
        EvaluationSettings(),
        EvaluationSettings(
            dos_mesh=(24, 24, 24),
            disentanglement="projectability",
            disentanglement_maximum_iterations=8000,
            disentanglement_convergence_tolerance=7.0e-6,
            localization="maximal",
            localization_maximum_iterations=6000,
            projectability_minimum=0.02,
            projectability_maximum=0.85,
            frozen_window_above_fermi_ev=0.0,
        ),
        EvaluationSettings(frozen_window_above_fermi_ev="auto", localization="none"),
    ],
)
def test_a_block_a_model_writes_is_one_the_schema_accepts(settings):
    jsonschema.Draft7Validator(SCHEMA_BLOCK).validate(settings.to_block())


@pytest.mark.parametrize(
    ("model", "outside_the_schema"),
    [
        (CaptureSettings, {"grid": [4, 4]}),
        (CaptureSettings, {"grid": [4, 0, 4]}),
        (CaptureSettings, {"grid": [4, 4, 4.5]}),
        (CaptureSettings, {"grid": [4, 4, 4], "method": "wanier"}),
        (CaptureSettings, {"grid": [4, 4, 4], "bands": [0, 8]}),
        (CaptureSettings, {"grid": [4, 4, 4], "bands": [1, 8, 9]}),
        (CaptureSettings, {"grid": [4, 4, 4], "projection_centers": [[0.0, 0.0]]}),
        (CaptureSettings, {"grid": [4, 4, 4], "projection_decay_bohr": -1.0}),
        (CaptureSettings, {"grid": [4, 4, 4], "trial_shells": {"Si": []}}),
        (CaptureSettings, {"grid": [4, 4, 4], "trial_shells": {"Si": [-1]}}),
        (CaptureSettings, {"grid": [4, 4, 4], "trial_shells": {"Si": [0, 0]}}),
        (CaptureSettings, {"grid": [4, 4, 4], "export_formats": ["wannier91"]}),
        (EvaluationSettings, {"dos_mesh": [24, 24]}),
        (EvaluationSettings, {"dos_mesh": [24, -24, 24]}),
        (EvaluationSettings, {"disentanglement": "windows"}),
        (EvaluationSettings, {"disentanglement_maximum_iterations": 0}),
        (EvaluationSettings, {"disentanglement_convergence_tolerance": 0.0}),
        (EvaluationSettings, {"localization_maximum_iterations": 0}),
        (EvaluationSettings, {"projectability_minimum": -0.01}),
        (EvaluationSettings, {"projectability_maximum": 1.5}),
    ],
)
def test_what_the_schema_refuses_the_model_refuses(model, outside_the_schema):
    block = {"method": "wannier", **outside_the_schema}
    assert list(jsonschema.Draft7Validator(SCHEMA_BLOCK).iter_errors(block)), (
        "the schema accepts this block, so the case proves nothing"
    )
    with pytest.raises(ValueError):
        model(**outside_the_schema)


def test_a_mesh_is_refused_in_the_words_a_caller_can_act_on():
    with pytest.raises(ValueError, match="grid must be three positive integers"):
        CaptureSettings(grid=[4, 0, 4])
    with pytest.raises(ValueError, match="dos_mesh must be three positive integers"):
        EvaluationSettings(dos_mesh=[4, 4])
    with pytest.raises(ValueError, match="dos_mesh must be three positive integers"):
        EvaluationSettings().updated(dos_mesh=[0, 4, 4])


def test_the_window_is_a_number_or_auto():
    assert (
        EvaluationSettings(frozen_window_above_fermi_ev="auto").to_block()[
            "frozen_window_above_fermi_ev"
        ]
        == "auto"
    )
    assert (
        EvaluationSettings(frozen_window_above_fermi_ev=1).to_block()[
            "frozen_window_above_fermi_ev"
        ]
        == 1.0
    )
    with pytest.raises(ValueError):
        EvaluationSettings(frozen_window_above_fermi_ev="wide")


_DEFAULTED_MEMBER = re.compile(
    r"^\s+(?:Real|int|bool|std::string)\s+(\w+)\s*=\s*([^;]+);", re.MULTILINE
)
_SETTINGS_REFERENCE = re.compile(r"^(\w+)\{\}\.(\w+)$")


def _defaulted_members(header: Path) -> dict[str, str]:
    """Every defaulted member a header declares, as its initialiser text."""
    return {name: value.strip() for name, value in _DEFAULTED_MEMBER.findall(header.read_text())}


def _engine_defaults() -> dict[str, str]:
    """Every defaulted member of InterpolationSpecification, as the value the engine applies.

    The specification states a knob's default once, by naming the settings
    struct that owns it (``DisentanglementSettings{}.maximum_iterations``);
    such a reference is followed into that struct's header, so what is
    compared with the schema is the number and not the spelling of where it
    lives.
    """
    defaults = _defaulted_members(SPECIFICATION)
    for name, value in defaults.items():
        reference = _SETTINGS_REFERENCE.match(value)
        if reference is None:
            continue
        struct, member = reference.groups()
        header = SETTINGS_DIRECTORY / (re.sub(r"(?<!^)(?=[A-Z])", "_", struct).lower() + ".hpp")
        defaults[name] = _defaulted_members(header)[member]
    return defaults


def test_the_schemas_defaults_are_the_engines():
    engine = _engine_defaults()
    numeric = {
        "projectability_minimum": 0.01,
        "projectability_maximum": 0.95,
        "localization_maximum_iterations": 3000,
        "disentanglement_maximum_iterations": 4000,
        "disentanglement_convergence_tolerance": 1.0e-10,
        "projection_decay_bohr": 1.0,
    }
    for key, value in numeric.items():
        assert SCHEMA_BLOCK["properties"][key]["default"] == value, key
        assert float(engine[key]) == value, f"{key}: specification.hpp says {engine[key]}"
    assert SCHEMA_BLOCK["properties"]["localization"]["default"] == "auto"
    assert engine["localization"] == '"auto"'
    # An empty `disentanglement` string in the engine is the schema's "none".
    assert SCHEMA_BLOCK["properties"]["disentanglement"]["default"] == "none"
    assert "disentanglement" not in engine
    # "auto" in the manifest is the engine's automatic flag, not a number.
    assert SCHEMA_BLOCK["properties"]["frozen_window_above_fermi_ev"]["default"] == "auto"
    assert engine["frozen_window_is_automatic"] == "true"
