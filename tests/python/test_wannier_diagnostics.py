"""tests/python/test_wannier_diagnostics.py — the codes are the engine's, and they route first.

The engine's ``WannierDiagnostic`` enumerators and this side's ``Diagnostic``
are two spellings of one list. Reading the enumerators out of the one header
that declares them keeps the two equal at unit-test speed; the rest pins that
the code is the only thing that routes -- a failure without one is unknown to
the ladder whatever its sentence says -- and that the numbers a failure
carries are read off their own line.
"""

import re
from pathlib import Path

import pytest
from q3.errors import Q3RuntimeError
from q3.workflows.wannier.attempt_status import AttemptStatus
from q3.workflows.wannier.diagnostic import Diagnostic
from q3.workflows.wannier.route import Route
from q3.workflows.wannier.utils import last_disentanglement_spreads

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent.parent
DIAGNOSTIC_HEADER = (
    REPOSITORY_ROOT / "src/layer_4_computation/interpolator/method/wannier/diagnostic.hpp"
)

PREFIX = "[2026-09-17 19:14:14.460] [error] "
UNCONVERGED = (
    "disentanglement did not converge in 4000 iterations. Last gauge-invariant spreads "
    "(bohr^2): 32.869717, 32.869701, 32.869684, 32.869668, 32.869651. A subspace that has "
    "not converged is not returned as if it had."
)


def _engine_enumerators() -> set[str]:
    text = DIAGNOSTIC_HEADER.read_text()
    block = re.search(r"enum class WannierDiagnostic \{(.*?)\};", text, re.DOTALL)
    assert block is not None, "the engine's enum moved; point this test at it"
    return set(re.findall(r"^\s+([a-z_]+),\s*$", block.group(1), re.MULTILINE))


def test_the_codes_are_the_engines_enumerators_spelled_the_same():
    assert {diagnostic.value for diagnostic in Diagnostic} == _engine_enumerators()


def test_every_code_has_a_route():
    assert {diagnostic.route for diagnostic in Diagnostic} <= set(Route)
    assert Diagnostic.DISENTANGLEMENT_UNCONVERGED.route is Route.UNCONVERGED
    assert Diagnostic.FROZEN_WINDOW_OVERFULL.route is Route.INFEASIBLE
    assert Diagnostic.LOCALIZATION_UNSUPPORTED.route is Route.LOCALIZATION


def test_the_code_line_is_read_through_the_loggers_prefix():
    message = f"{PREFIX}Error: {UNCONVERGED}\n{PREFIX}diagnostic: disentanglement_unconverged\n"
    assert Diagnostic.from_message(message) is Diagnostic.DISENTANGLEMENT_UNCONVERGED
    assert Diagnostic.from_message("Error: something else\n") is None
    assert Diagnostic.from_message(f"{PREFIX}diagnostic: a_newer_code\n") is None


def test_the_code_alone_decides_the_status_whatever_the_prose_says():
    # The prose says "freezes" and the code says the record cannot be localised: the code routes.
    message = (
        "Error: this k-point freezes too much, or so a reading of the words would say\n"
        "diagnostic: two_point_axis\n"
    )
    assert (
        AttemptStatus.of_failure(Diagnostic.from_message(message))
        is AttemptStatus.ERROR_LOCALIZATION
    )
    assert (
        AttemptStatus.of_failure(Diagnostic.KEPT_BANDS_UNDERFULL) is AttemptStatus.ERROR_INFEASIBLE
    )
    assert (
        AttemptStatus.of_failure(Diagnostic.OVERLAPS_NOT_ORTHONORMAL) is AttemptStatus.ERROR_UNKNOWN
    )


@pytest.mark.parametrize(
    "message",
    [
        UNCONVERGED,
        "disentanglement: k-point 3 freezes 11 bands but only 10 Wannier functions were asked for",
        "localization needs the neighbour overlaps M_mn(k,b), and this record carries none",
        "cannot open file",
    ],
)
def test_a_failure_without_a_code_is_one_the_ladder_does_not_know(message):
    """Words route nothing, however familiar: a failure the engine gave no code is unknown."""
    assert Diagnostic.from_message(message) is None
    assert AttemptStatus.of_failure(Diagnostic.from_message(message)) is AttemptStatus.ERROR_UNKNOWN


def test_every_status_a_failure_can_have_is_spelled_error_then_its_route():
    assert {status.value for status in AttemptStatus if status.value.startswith("error:")} == {
        "error:" + route.value for route in Route
    }


def test_a_failed_run_says_its_own_text_and_its_error_line():
    failure = Q3RuntimeError(
        "q3_utility exited with code 1.",
        returncode=1,
        stderr=f"{PREFIX}Error: {UNCONVERGED}\n{PREFIX}diagnostic: disentanglement_unconverged\n",
        diagnostic="disentanglement_unconverged",
    )
    assert failure.error_line == f"Error: {UNCONVERGED}"
    assert "q3_utility exited with code 1." in failure.text
    assert Diagnostic.of(failure) is Diagnostic.DISENTANGLEMENT_UNCONVERGED
    assert (
        Q3RuntimeError("the binary said nothing useful").error_line
        == "the binary said nothing useful"
    )


def test_the_spreads_are_read_off_their_own_line_and_never_out_of_the_sentence():
    with_line = f"Error: {UNCONVERGED}\ndiagnostic_spreads_bohr_squared: 1.000000, 2.000000\n"
    assert last_disentanglement_spreads(with_line) == [1.0, 2.0]
    assert last_disentanglement_spreads(f"{PREFIX}diagnostic_spreads_bohr_squared: 3.5\n") == [3.5]
    assert last_disentanglement_spreads(UNCONVERGED) == [], "the sentence is for a reader"
    assert last_disentanglement_spreads("Error: something else") == []
