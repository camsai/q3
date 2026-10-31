"""tests/python/test_compare_results_json.py — Unit tests for the parity comparator.

Covers ``scripts/validation/compare_results_json.py``: exact-by-default comparison, the
volatile-leaf exemptions, structural differences, directory mode including the
byte comparison that puts ``.save/`` under the gate, and the
``numerics-change`` escape: its tolerance with a mandatory reason, and the matrix
rows a pull request declares it moves.

The two cases the improvement program names as the gate's own self-test are
here by name: a no-op change must pass, and a 1e-14 perturbation must fail.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT_PATH = REPOSITORY_ROOT / "scripts" / "validation" / "compare_results_json.py"


def _load_comparator():
    """Import the comparator script as a module (it is a script, not a package)."""
    spec = importlib.util.spec_from_file_location("compare_results_json", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


comparator = _load_comparator()

BASELINE_RESULTS = {
    "total_energy_ry": -15.793461234567,
    "fermi_energy_ry": 0.4312,
    "forces_ry_per_bohr": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    "stress_kbar": [[1.5, 0.0, 0.0], [0.0, 1.5, 0.0], [0.0, 0.0, 1.5]],
    "eigenvalues_ry": [[-0.2, 0.1, 0.3, 0.3]],
    "runtime_log": {
        "flavor": "scf",
        "converged": True,
        "timings_ms": {"total": 1234.5, "davidson": 900.1},
        "execution": {"number_of_processes": 1, "peak_resident_memory_bytes": 104857600},
    },
    "provenance": {"generator": {"engine_name": "q3", "engine_commit": "abc123"}},
}


def _write(path, document):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


# ── Exact comparison ──────────────────────────────────────────────────────────


def test_identical_documents_agree():
    assert comparator.compare_documents(BASELINE_RESULTS, BASELINE_RESULTS, 0.0) == []


def test_a_no_op_change_passes(tmp_path):
    """The gate's first self-test: a pull request that moves no numbers is green."""
    base = _write(tmp_path / "base" / "si_results.json", BASELINE_RESULTS)
    head = _write(tmp_path / "head" / "si_results.json", BASELINE_RESULTS)
    assert comparator.main([str(base), str(head)]) == 0


def test_a_one_part_in_1e14_perturbation_fails(tmp_path):
    """The gate's second self-test: the smallest drift a double can carry is caught."""
    perturbed = dict(BASELINE_RESULTS)
    perturbed["total_energy_ry"] = BASELINE_RESULTS["total_energy_ry"] * (1.0 + 1e-14)
    assert perturbed["total_energy_ry"] != BASELINE_RESULTS["total_energy_ry"]

    base = _write(tmp_path / "base" / "si_results.json", BASELINE_RESULTS)
    head = _write(tmp_path / "head" / "si_results.json", perturbed)
    assert comparator.main([str(base), str(head)]) == 1

    differences = comparator.compare_documents(BASELINE_RESULTS, perturbed, 0.0)
    assert len(differences) == 1
    assert differences[0].startswith("/total_energy_ry:")


def test_a_difference_deep_in_an_array_is_found_by_pointer():
    moved = json.loads(json.dumps(BASELINE_RESULTS))
    moved["eigenvalues_ry"][0][2] = 0.30000000000001
    differences = comparator.compare_documents(BASELINE_RESULTS, moved, 0.0)
    assert differences == [f"/eigenvalues_ry/0/2: base {0.3!r} vs head {0.30000000000001!r}"]


# ── Volatile leaves ───────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("pointer_path", "replacement"),
    [
        (("runtime_log", "timings_ms", "total"), 9999.9),
        (("runtime_log", "execution", "peak_resident_memory_bytes"), 209715200),
        (("provenance", "generator", "engine_commit"), "def456"),
    ],
)
def test_volatile_leaves_do_not_break_parity(pointer_path, replacement):
    """Timings, peak memory and the engine's own commit differ run to run by design."""
    moved = json.loads(json.dumps(BASELINE_RESULTS))
    target = moved
    for key in pointer_path[:-1]:
        target = target[key]
    target[pointer_path[-1]] = replacement
    assert comparator.compare_documents(BASELINE_RESULTS, moved, 0.0) == []


def test_a_non_volatile_execution_leaf_still_breaks_parity():
    """The volatile list is per-leaf, not per-subtree: topology still has to match."""
    moved = json.loads(json.dumps(BASELINE_RESULTS))
    moved["runtime_log"]["execution"]["number_of_processes"] = 4
    assert comparator.compare_documents(BASELINE_RESULTS, moved, 0.0) != []


# ── Structure ─────────────────────────────────────────────────────────────────


def test_a_removed_key_is_a_difference():
    moved = json.loads(json.dumps(BASELINE_RESULTS))
    del moved["fermi_energy_ry"]
    assert comparator.compare_documents(BASELINE_RESULTS, moved, 0.0) == [
        "/fermi_energy_ry: present in base, missing from head"
    ]


def test_an_added_key_is_a_difference():
    moved = json.loads(json.dumps(BASELINE_RESULTS))
    moved["hubbard_energy_ry"] = 0.25
    assert comparator.compare_documents(BASELINE_RESULTS, moved, 0.0) == [
        "/hubbard_energy_ry: missing from base, present in head"
    ]


def test_a_shortened_array_is_a_difference():
    moved = json.loads(json.dumps(BASELINE_RESULTS))
    moved["forces_ry_per_bohr"] = moved["forces_ry_per_bohr"][:1]
    differences = comparator.compare_documents(BASELINE_RESULTS, moved, 0.0)
    assert len(differences) == 3
    assert all("present in base, missing from head" in message for message in differences)


def test_an_emptied_array_is_a_difference():
    """A container leaf exists so that emptying one cannot read as no difference."""
    moved = json.loads(json.dumps(BASELINE_RESULTS))
    moved["forces_ry_per_bohr"] = []
    assert comparator.compare_documents(BASELINE_RESULTS, moved, 0.0) != []


def test_a_boolean_is_not_a_number():
    """``True == 1.0`` in Python; a flag turning into a count is still a difference."""
    moved = json.loads(json.dumps(BASELINE_RESULTS))
    moved["runtime_log"]["converged"] = 1
    assert comparator.compare_documents(BASELINE_RESULTS, moved, 0.0) != []


def test_matching_not_a_numbers_agree():
    assert comparator.values_agree(float("nan"), float("nan"), 0.0)
    assert not comparator.values_agree(float("nan"), 0.0, 0.0)


# ── The numerics-change escape ────────────────────────────────────────────────


def test_a_relative_tolerance_admits_a_small_move():
    moved = json.loads(json.dumps(BASELINE_RESULTS))
    moved["total_energy_ry"] *= 1.0 + 1e-14
    assert comparator.compare_documents(BASELINE_RESULTS, moved, 1e-12) == []
    assert comparator.compare_documents(BASELINE_RESULTS, moved, 0.0) != []


def test_a_relative_tolerance_still_catches_a_real_move():
    moved = json.loads(json.dumps(BASELINE_RESULTS))
    moved["total_energy_ry"] *= 1.0 + 1e-6
    assert comparator.compare_documents(BASELINE_RESULTS, moved, 1e-12) != []


def test_a_tolerance_without_a_reason_is_refused(tmp_path, capsys):
    base = _write(tmp_path / "base" / "si_results.json", BASELINE_RESULTS)
    head = _write(tmp_path / "head" / "si_results.json", BASELINE_RESULTS)
    assert comparator.main([str(base), str(head), "--tolerance", "1e-12"]) == 2
    assert "needs --reason" in capsys.readouterr().err


def test_a_tolerance_with_a_reason_is_accepted(tmp_path):
    base = _write(tmp_path / "base" / "si_results.json", BASELINE_RESULTS)
    head = _write(tmp_path / "head" / "si_results.json", BASELINE_RESULTS)
    assert (
        comparator.main([str(base), str(head), "--tolerance", "1e-12", "--reason", "PDOS S metric"])
        == 0
    )


# ── Rows declared to move ─────────────────────────────────────────────────────


def _two_row_runs(tmp_path, moved_row):
    """A base and a head run of two rows, with one row's energy moved in the head."""
    moved = json.loads(json.dumps(BASELINE_RESULTS))
    moved["total_energy_ry"] += 1e-6
    for row in ("copper-smearing", "silicon-insulator"):
        _write(tmp_path / "base" / row / "output" / "x_results.json", BASELINE_RESULTS)
        head_document = moved if row == moved_row else BASELINE_RESULTS
        _write(tmp_path / "head" / row / "output" / "x_results.json", head_document)
    return [str(tmp_path / "base"), str(tmp_path / "head")]


def test_differences_confined_to_declared_rows_pass(tmp_path, capsys):
    escape = ["--tolerance", "1e-12", "--reason", "a smeared metal stays neutral"]
    arguments = _two_row_runs(tmp_path, "copper-smearing") + escape + ["--moving-rows", "copper-*"]
    assert comparator.main(arguments) == 0
    output = capsys.readouterr().out
    assert "copper-smearing: 1 difference(s), declared to move" in output
    assert "outside the 1 row(s) declared to move" in output


def test_a_row_not_declared_still_fails(tmp_path, capsys):
    escape = ["--tolerance", "1e-12", "--reason", "metals only"]
    arguments = (
        _two_row_runs(tmp_path, "silicon-insulator") + escape + ["--moving-rows", "copper-*"]
    )
    assert comparator.main(arguments) == 1
    captured = capsys.readouterr()
    assert "silicon-insulator/output/x_results.json/total_energy_ry" in captured.err
    assert "copper-smearing: declared to move, but identical" in captured.out


def test_a_row_pattern_that_matches_nothing_is_refused(tmp_path, capsys):
    escape = ["--tolerance", "1e-12", "--reason", "a typo in the row name"]
    arguments = _two_row_runs(tmp_path, "copper-smearing") + escape + ["--moving-rows", "coper-*"]
    assert comparator.main(arguments) == 2
    assert "match no row" in capsys.readouterr().err


def test_declared_rows_need_a_reason(tmp_path, capsys):
    arguments = _two_row_runs(tmp_path, "copper-smearing") + ["--moving-rows", "copper-*"]
    assert comparator.main(arguments) == 2
    assert "needs --reason" in capsys.readouterr().err


def test_declared_rows_need_two_run_directories(tmp_path, capsys):
    base = _write(tmp_path / "base" / "si_results.json", BASELINE_RESULTS)
    head = _write(tmp_path / "head" / "si_results.json", BASELINE_RESULTS)
    arguments = [str(base), str(head), "--reason", "rows", "--moving-rows", "si*"]
    assert comparator.main(arguments) == 2
    assert "two run directories" in capsys.readouterr().err


def test_checkpoints_of_a_declared_row_do_not_fail(tmp_path):
    """The ``.save/`` bytes are what no tolerance can pass, so the declaration covers them."""
    for side, payload in (("base", b"\x01\x02"), ("head", b"\x01\x03")):
        checkpoint = tmp_path / side / "copper-smearing" / "output" / "cu.save" / "wfc1.dat"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(payload)
    arguments = [str(tmp_path / "base"), str(tmp_path / "head"), "--tolerance", "1e-12"]
    arguments += ["--reason", "metals move", "--moving-rows", "copper-smearing"]
    assert comparator.main(arguments) == 0


# ── Directory mode ────────────────────────────────────────────────────────────


def test_directory_mode_compares_every_results_file(tmp_path):
    _write(tmp_path / "base" / "case-a" / "output" / "a_results.json", BASELINE_RESULTS)
    _write(tmp_path / "head" / "case-a" / "output" / "a_results.json", BASELINE_RESULTS)
    assert comparator.main([str(tmp_path / "base"), str(tmp_path / "head")]) == 0


def test_directory_mode_names_the_file_a_difference_is_in(tmp_path):
    moved = json.loads(json.dumps(BASELINE_RESULTS))
    moved["total_energy_ry"] += 1e-9
    _write(tmp_path / "base" / "case-a" / "output" / "a_results.json", BASELINE_RESULTS)
    _write(tmp_path / "head" / "case-a" / "output" / "a_results.json", moved)
    differences = comparator.compare(tmp_path / "base", tmp_path / "head", 0.0)
    assert len(differences) == 1
    assert differences[0].startswith("case-a/output/a_results.json/total_energy_ry:")


def test_directory_mode_compares_checkpoints_byte_for_byte(tmp_path):
    """The ``.save/`` tree is under the gate too, and it is not JSON."""
    for side, payload in (("base", b"\x01\x02\x03"), ("head", b"\x01\x02\x04")):
        checkpoint = tmp_path / side / "case-a" / "output" / "si.save" / "wfc1.dat"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(payload)
    differences = comparator.compare(tmp_path / "base", tmp_path / "head", 0.0)
    assert differences == ["case-a/output/si.save/wfc1.dat: bytes differ"]


def test_directory_mode_reports_a_file_only_one_side_produced(tmp_path):
    _write(tmp_path / "base" / "case-a" / "output" / "a_results.json", BASELINE_RESULTS)
    _write(tmp_path / "head" / "case-a" / "output" / "a_results.json", BASELINE_RESULTS)
    _write(tmp_path / "head" / "case-a" / "output" / "extra.json", {"x": 1})
    differences = comparator.compare(tmp_path / "base", tmp_path / "head", 0.0)
    assert differences == ["case-a/output/extra.json: produced by head, not by base"]


def test_comparing_a_file_against_a_directory_is_an_error(tmp_path, capsys):
    base = _write(tmp_path / "base" / "si_results.json", BASELINE_RESULTS)
    (tmp_path / "head").mkdir(parents=True, exist_ok=True)
    assert comparator.main([str(base), str(tmp_path / "head")]) == 2
    assert "not the same kind of path" in capsys.readouterr().err


def test_a_missing_input_is_an_error_not_a_pass(tmp_path, capsys):
    base = _write(tmp_path / "base" / "si_results.json", BASELINE_RESULTS)
    assert comparator.main([str(base), str(tmp_path / "head" / "si_results.json")]) == 2
    assert "could not run" in capsys.readouterr().err
