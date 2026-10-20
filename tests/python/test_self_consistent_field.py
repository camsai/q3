"""tests/python/test_self_consistent_field.py — Unit tests for q3.flavors.self_consistent_field.

run_scf runs in one of two modes, chosen only by whether the compiled q3_core
extension imports: in-process through it (Mode B), or by launching the q3
binary (Mode A). Which one a caller gets is an accident of the build, so a
result has to be treated the same either way -- and a result that did not
converge was not: Mode A raised, while Mode B returned it as though it were an
answer.

Neither mode needs the engine here. Mode B runs against a stand-in q3_core
patched over the module's own; Mode A against a stand-in binary that exits 0, as the
engine does when it does not converge, beside the results file a run leaves.
"""

import json
import types
from pathlib import Path

import pytest
from q3.errors import SCFNotConvergedError
from q3.flavors import run_scf

# What a run that used its whole iteration budget reports: the count it
# performed, and the limit its provenance recorded.
_NOT_CONVERGED_RESULTS = {
    "flavor": "self_consistent_field",
    "converged": False,
    "self_consistent_field_iterations": 100,
    "provenance": {"precision": {"maximum_self_consistent_field_iterations": 100}},
    "total_energy_ry": -486.6177018,
}


def _write_input(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    input_path = directory / "input.yml"
    input_path.write_text("control:\n  calculation: scf\n  outdir: output\n  prefix: copper\n")
    return input_path


def _install_in_process_engine(monkeypatch, results: dict) -> list[str]:
    """Make run_scf's ``q3_core`` a stand-in whose run_scf returns *results*.

    Returns the list of input paths it is called with, so a test can tell Mode
    B actually ran rather than falling through to Mode A.
    """
    calls: list[str] = []

    def run_scf_in_process(input_path: str, write_output_files: bool = True) -> str:
        calls.append(input_path)
        return json.dumps(results)

    stand_in = types.ModuleType("q3_core")
    stand_in.run_scf = run_scf_in_process
    monkeypatch.setattr("q3.flavors.self_consistent_field.q3_core", stand_in)
    return calls


def test_in_process_run_refuses_a_result_that_did_not_converge(tmp_path, monkeypatch):
    calls = _install_in_process_engine(monkeypatch, _NOT_CONVERGED_RESULTS)
    input_path = _write_input(tmp_path)

    with pytest.raises(SCFNotConvergedError) as failure:
        run_scf(input_path)

    assert calls == [str(input_path.resolve())]
    assert str(failure.value) == "SCF did not converge after 100 iterations; its limit was 100."
    assert failure.value.iterations == 100
    assert failure.value.maximum_iterations == 100
    assert failure.value.returncode == -1  # no process ran, so there is no exit status


def test_in_process_run_returns_a_converged_result(tmp_path, monkeypatch):
    converged_results = dict(_NOT_CONVERGED_RESULTS, converged=True)
    calls = _install_in_process_engine(monkeypatch, converged_results)

    assert run_scf(_write_input(tmp_path)) == converged_results
    assert len(calls) == 1


def test_both_modes_refuse_the_same_result_in_the_same_words(tmp_path, monkeypatch):
    """Whether a failed run is reported must not depend on how Q3 was built."""
    _install_in_process_engine(monkeypatch, _NOT_CONVERGED_RESULTS)
    with pytest.raises(SCFNotConvergedError) as in_process_failure:
        run_scf(_write_input(tmp_path / "in_process"))

    subprocess_input = _write_input(tmp_path / "subprocess")
    output_directory = subprocess_input.parent / "output"
    output_directory.mkdir()
    (output_directory / "copper_results.json").write_text(json.dumps(_NOT_CONVERGED_RESULTS))
    binary = subprocess_input.parent / "q3"
    binary.write_text("#!/bin/sh\nexit 0\n")
    binary.chmod(0o755)
    # No extension is what selects Mode A -- even on a machine where it is built.
    monkeypatch.setattr("q3.flavors.self_consistent_field.q3_core", None)
    with pytest.raises(SCFNotConvergedError) as subprocess_failure:
        run_scf(subprocess_input, q3_binary=binary)

    assert str(in_process_failure.value) == str(subprocess_failure.value)
    assert in_process_failure.value.iterations == subprocess_failure.value.iterations
    assert (
        in_process_failure.value.maximum_iterations == subprocess_failure.value.maximum_iterations
    )
