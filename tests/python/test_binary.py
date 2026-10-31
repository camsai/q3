import json
from pathlib import Path

import pytest
from q3.binary import _find_results_json, run_q3, run_q3_utility
from q3.errors import Q3RuntimeError, SCFNotConvergedError, diagnostic_code, first_error_line


def _write_input(path: Path) -> None:
    path.write_text("control:\n  outdir: output\n  prefix: silicon\n")


def test_find_results_json_prefers_canonical_prefixed_name(tmp_path):
    input_path = tmp_path / "input.yml"
    output_directory = tmp_path / "output"
    output_directory.mkdir()
    _write_input(input_path)

    canonical_path = output_directory / "silicon_results.json"
    compatibility_path = output_directory / "results.json"
    canonical_path.write_text("{}\n")
    compatibility_path.write_text("{}\n")

    assert _find_results_json(input_path) == canonical_path


def test_find_results_json_accepts_legacy_unprefixed_name(tmp_path):
    input_path = tmp_path / "input.yml"
    output_directory = tmp_path / "output"
    output_directory.mkdir()
    _write_input(input_path)

    compatibility_path = output_directory / "results.json"
    compatibility_path.write_text("{}\n")

    assert _find_results_json(input_path) == compatibility_path


# ── Non-convergence ──────────────────────────────────────────────────────────

"""
run_q3 is where a failed SCF became "did not converge after 0 iterations": it
read an `iterations` key the engine has never written. These run it for real,
subprocess and all, against a stand-in executable that exits 0 the way the
engine does on non-convergence, beside the results file the engine would have
left.
"""


def _stage_finished_run(tmp_path: Path, results: dict) -> tuple[Path, Path]:
    """Write an input, the results a run of it left, and a binary that exits 0."""
    input_path = tmp_path / "input.yml"
    _write_input(input_path)
    output_directory = tmp_path / "output"
    output_directory.mkdir()
    (output_directory / "silicon_results.json").write_text(json.dumps(results))

    binary = tmp_path / "q3"
    binary.write_text("#!/bin/sh\nexit 0\n")
    binary.chmod(0o755)
    return input_path, binary


def test_run_q3_reports_how_many_iterations_a_failed_run_performed(tmp_path):
    input_path, binary = _stage_finished_run(
        tmp_path,
        {
            "runtime_log": {
                "flavor": "self_consistent_field",
                "converged": False,
                "self_consistent_field_iterations": 100,
            },
            "provenance": {"precision": {"maximum_self_consistent_field_iterations": 100}},
        },
    )

    with pytest.raises(SCFNotConvergedError) as failure:
        run_q3(input_path, q3_binary=binary)

    assert "after 100 iterations" in str(failure.value)
    assert failure.value.iterations == 100
    assert failure.value.returncode == 0


def test_run_q3_states_the_limit_when_an_older_binary_reports_no_count(tmp_path):
    input_path, binary = _stage_finished_run(
        tmp_path,
        {
            "runtime_log": {"flavor": "self_consistent_field", "converged": False},
            "provenance": {"precision": {"maximum_self_consistent_field_iterations": 100}},
        },
    )

    with pytest.raises(SCFNotConvergedError) as failure:
        run_q3(input_path, q3_binary=binary)

    assert "within its 100-iteration limit" in str(failure.value)
    assert "0 iterations" not in str(failure.value)
    assert failure.value.iterations is None


def test_run_q3_returns_a_converged_result(tmp_path):
    converged_results = {"runtime_log": {"flavor": "self_consistent_field", "converged": True}}
    input_path, binary = _stage_finished_run(tmp_path, converged_results)

    assert run_q3(input_path, q3_binary=binary).to_document() == converged_results


# ── q3_utility failures when the output is shown ────────────────────────────────


def test_run_q3_utility_keeps_the_failure_when_it_shows_the_output(tmp_path, capsys):
    """
    Showing the output used to send it only to the terminal, so a failure came
    back as "exited with code 2" and nothing else -- and the Wannier ladder,
    which routes on the diagnostic, rejected every such failure as unknown.
    """
    utility = tmp_path / "q3_utility"
    utility.write_text(
        "#!/bin/sh\necho 'disentanglement: iterating'\n"
        "echo 'Error: disentanglement did not converge in 4000 iterations.' >&2\nexit 2\n"
    )
    utility.chmod(0o755)

    with pytest.raises(Q3RuntimeError) as failure:
        run_q3_utility(["interpolate", "run.yml"], q3_utility_binary=utility, capture_output=False)

    assert "did not converge" in failure.value.stderr
    assert failure.value.returncode == 2
    shown = capsys.readouterr().out
    assert "disentanglement: iterating" in shown
    assert "did not converge" in shown


def test_run_q3_keeps_the_failure_when_it_shows_the_output(tmp_path, capsys):
    """The engine runner, like the utility's: shown output is still the error's to carry."""
    binary = tmp_path / "q3"
    binary.write_text(
        "#!/bin/sh\necho '[info] scf: iterating'\n"
        "echo '[error] Error: control.interpolation: the pseudopotentials carry augmentation "
        "charges'\nexit 1\n"
    )
    binary.chmod(0o755)
    _write_input(tmp_path / "run.yml")

    with pytest.raises(Q3RuntimeError) as failure:
        run_q3(tmp_path / "run.yml", q3_binary=binary, capture_output=False)

    assert "carry augmentation charges" in failure.value.stderr
    assert failure.value.command[0] == str(binary)
    assert failure.value.diagnostic is None
    assert "scf: iterating" in capsys.readouterr().out


def test_a_failed_run_carries_the_code_the_binary_printed(tmp_path):
    utility = tmp_path / "q3_utility"
    utility.write_text(
        "#!/bin/sh\necho 'Error: the k-mesh is 2x2x2, and the neighbour direction (1, 0, 0) "
        "reaches the same k-point as its opposite.' >&2\n"
        "echo 'diagnostic: two_point_axis' >&2\nexit 1\n"
    )
    utility.chmod(0o755)

    with pytest.raises(Q3RuntimeError) as failure:
        run_q3_utility(["interpolate", "run.yml"], q3_utility_binary=utility)

    assert failure.value.diagnostic == "two_point_axis"
    assert failure.value.command == [str(utility), "interpolate", "run.yml"]
    assert first_error_line(failure.value.stderr).startswith("Error: the k-mesh is 2x2x2")


def test_the_error_line_is_read_from_after_the_loggers_prefix():
    text = (
        "[2026-09-17 19:14:14.460] [info] disentanglement: iterating\n"
        "[2026-09-17 19:14:14.461] [error] Error: disentanglement did not converge\n"
        "[2026-09-17 19:14:14.461] [error] diagnostic: disentanglement_unconverged\n"
        "    stage_0: 11.3 ms\n"
    )
    assert first_error_line(text) == "Error: disentanglement did not converge"
    assert diagnostic_code(text) == "disentanglement_unconverged"
    assert first_error_line("nothing to see") is None
    assert diagnostic_code("diagnostic_spreads_bohr_squared: 1.0, 2.0") is None
