"""
Validation test: a run reports how many SCF iterations it performed.

`converged` says that a run failed, not how far it got. The engine writes the
count as `self_consistent_field_iterations`, beside the limit its provenance
records as `maximum_self_consistent_field_iterations`, and the SCF loop's two
exits must both get it right: converging, and running out of budget. The second
is the exit every failed run takes, and the one where a loop counter that
stepped past its budget would report an iteration that never ran.

Runs the silicon norm-conserving example, which converges in seconds. FFT
planning is made deterministic, so runs given different budgets follow one
trajectory for as long as both last -- the budget bounds the loop and nothing
else, which makes a run cut short a prefix of a longer one.
"""

import pytest
from q3.binary import run_q3
from q3.errors import SCFNotConvergedError

from helpers import EXAMPLES_DIRECTORY, apply_config_overlay, run_q3_and_load_results

SILICON_INPUT = EXAMPLES_DIRECTORY / "total_energy--semiconductor--silicon" / "nc.yml"
PREFIX = "si_nc"


def _overlay(maximum_iterations=None):
    overlay = {"control": {"use_deterministic_fft_planning": True}}
    if maximum_iterations is not None:
        overlay["method"] = {"scf": {"max_iterations": maximum_iterations}}
    return overlay


def _run(q3_command, run_directory, maximum_iterations=None):
    input_path = apply_config_overlay(SILICON_INPUT, _overlay(maximum_iterations), run_directory)
    results, _ = run_q3_and_load_results(
        q3_command, input_path, PREFIX, working_directory=str(run_directory)
    )
    return results


def test_a_run_that_exhausts_its_budget_reports_that_budget(q3_command, tmp_path):
    results = _run(q3_command, tmp_path, maximum_iterations=3)

    assert results["converged"] is False
    assert results["self_consistent_field_iterations"] == 3
    assert results["provenance"]["precision"]["maximum_self_consistent_field_iterations"] == 3


def test_a_converged_run_reports_the_smallest_budget_that_converges(q3_command, tmp_path):
    """Pin the count exactly, without reading it back out of the log.

    If a run converges on iteration k, the same run limited to k iterations
    must still converge, and limited to k - 1 must not -- so a count off by one
    in either direction fails one of the two.
    """
    unlimited = _run(q3_command, tmp_path / "unlimited")
    converged_on = unlimited["self_consistent_field_iterations"]
    limit = unlimited["provenance"]["precision"]["maximum_self_consistent_field_iterations"]
    assert unlimited["converged"] is True
    assert 2 <= converged_on < limit

    exactly_enough = _run(q3_command, tmp_path / "exactly_enough", maximum_iterations=converged_on)
    assert exactly_enough["converged"] is True
    assert exactly_enough["self_consistent_field_iterations"] == converged_on

    one_short = _run(q3_command, tmp_path / "one_short", maximum_iterations=converged_on - 1)
    assert one_short["converged"] is False
    assert one_short["self_consistent_field_iterations"] == converged_on - 1


def test_run_q3_reports_the_count_the_engine_wrote(q3_command, tmp_path):
    """The Python layer and the engine, together: the error names the real count."""
    if len(q3_command) != 1:
        pytest.skip("run_q3 takes the path of a binary, not a launcher command")
    input_path = apply_config_overlay(SILICON_INPUT, _overlay(maximum_iterations=3), tmp_path)

    with pytest.raises(SCFNotConvergedError) as failure:
        run_q3(input_path, q3_binary=q3_command[0])

    assert str(failure.value) == "SCF did not converge after 3 iterations; its limit was 3."
    assert failure.value.iterations == 3
    assert failure.value.maximum_iterations == 3
