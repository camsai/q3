"""tests/python/test_run_benchmarks.py — Unit tests for the benchmark harness.

Covers ``scripts/benchmark/run_benchmarks.py``: the deck manifest (paths resolved
relative to the manifest, as AGENTS.md § Testing requires), deck selection by
identifier and by tag, the input rewriting that lets a deck run from a
throwaway directory, the split between stage times and work counts, and the
three ways a run can fail to produce a usable record.

The engine is replaced by a stub binary here. What is under test is the
harness's bookkeeping, and a real SCF run would make these tests minutes long
and dependent on a build.
"""

import importlib.util
import json
import stat
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT_PATH = REPOSITORY_ROOT / "scripts" / "benchmark" / "run_benchmarks.py"
MATRIX_PATH = REPOSITORY_ROOT / "tests" / "regression" / "benchmark_matrix.yml"


def _load_harness():
    """Import the harness script as a module (it is a script, not a package)."""
    spec = importlib.util.spec_from_file_location("run_benchmarks", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


harness = _load_harness()

# A stand-in for q3: reads the prepared input the way the real binary does,
# and writes a results file with the same timings_ms shape. {behaviour} is
# substituted per test.
STUB_BINARY = """#!/usr/bin/env python3
import json
import sys
from pathlib import Path

import yaml

manifest = yaml.safe_load(Path(sys.argv[1]).read_text())
control = manifest["control"]
output_directory = Path(control["outdir"])
output_directory.mkdir(parents=True, exist_ok=True)
{behaviour}
"""

WRITE_RESULTS = """
results = {"properties": {"records": [
    {"name": "total_energy", "units": "Ry", "value": -16.913417},
]}, "runtime_log": {
    "converged": True,
    "self_consistent_field_iterations": 7,
    "execution": {"peak_resident_memory_bytes": 39624704},
    "timings_ms": {
        "seed": 86.7,
        "orthogonalize": 28.6,
        "hamiltonian_application": 125.6,
        "subspace_update": 18.4,
        "diagonalize": 2.1,
        "collapse": 27.5,
        "total": 290.0,
        "fft_calls": 929.0,
        "davidson_iterations": 53.0,
        "hamiltonian_applications": 320.0,
    },
}}
(output_directory / f"{control['prefix']}_results.json").write_text(json.dumps(results))
"""

WRITE_RESULTS_WITHOUT_TIMINGS = """
results = {
    "runtime_log": {"converged": False},
    "properties": {"records": [{"name": "total_energy", "units": "Ry", "value": -16.9}]},
}
(output_directory / f"{control['prefix']}_results.json").write_text(json.dumps(results))
"""

FAIL_LOUDLY = """
print("engine exploded")
sys.exit(3)
"""


@pytest.fixture
def stub_binary(tmp_path):
    def build(behaviour=WRITE_RESULTS):
        path = tmp_path / "q3_stub.py"
        path.write_text(STUB_BINARY.format(behaviour=behaviour))
        path.chmod(path.stat().st_mode | stat.S_IEXEC)
        return path

    return build


@pytest.fixture
def deck(tmp_path):
    """A deck whose material and pseudopotential are named relative to itself."""
    deck_directory = tmp_path / "examples" / "case"
    deck_directory.mkdir(parents=True)
    (deck_directory.parent / "material.json").write_text("{}")
    (deck_directory / "Si.upf").write_text("pseudopotential")
    input_path = deck_directory / "nc.yml"
    input_path.write_text(
        yaml.safe_dump(
            {
                "material": "../material.json",
                "method": {
                    "pseudopotentials": {"Si": "Si.upf"},
                    "scf": {"convergence_threshold": 1.0e-8, "max_iterations": 40},
                },
                "control": {"outdir": "./output", "prefix": "si_nc"},
            }
        )
    )
    return {"identifier": "silicon", "input_path": input_path}


# ── The deck manifest ─────────────────────────────────────────────────────────


def test_committed_matrix_resolves_every_deck():
    """The real matrix points at files that exist, from wherever it is read."""
    matrix, decks = harness.load_matrix(MATRIX_PATH)
    assert matrix["repetitions"] >= 1
    assert decks
    for entry in decks:
        assert entry["input_path"].is_file()


def test_committed_matrix_holds_decks_cheap_enough_for_continuous_integration():
    """At least one deck is tagged for CI, or the benchmark job has nothing to run."""
    _, decks = harness.load_matrix(MATRIX_PATH)
    assert [entry for entry in decks if "ci" in (entry.get("tags") or [])]


def test_matrix_paths_are_relative_to_the_manifest():
    """AGENTS.md § Testing: no machine-specific absolute path in a committed manifest."""
    matrix = yaml.safe_load(MATRIX_PATH.read_text())
    for entry in matrix["decks"]:
        assert not Path(entry["input"]).is_absolute()


def test_matrix_missing_a_required_key_is_rejected(tmp_path):
    path = tmp_path / "matrix.yml"
    path.write_text(yaml.safe_dump({"decks": [{"identifier": "nameless"}]}))
    with pytest.raises(ValueError, match="input"):
        harness.load_matrix(path)


def test_matrix_pointing_at_a_missing_deck_is_rejected(tmp_path):
    path = tmp_path / "matrix.yml"
    path.write_text(yaml.safe_dump({"decks": [{"identifier": "gone", "input": "nowhere.yml"}]}))
    with pytest.raises(ValueError, match="missing input file"):
        harness.load_matrix(path)


def test_empty_matrix_is_rejected(tmp_path):
    path = tmp_path / "matrix.yml"
    path.write_text(yaml.safe_dump({"decks": []}))
    with pytest.raises(ValueError, match="No decks"):
        harness.load_matrix(path)


# ── Deck selection ────────────────────────────────────────────────────────────


DECKS = [
    {"identifier": "cheap", "tags": ["ci"]},
    {"identifier": "expensive", "tags": ["full"]},
    {"identifier": "untagged"},
]


def test_selection_by_identifier():
    assert [entry["identifier"] for entry in harness.select_decks(DECKS, ["cheap"], None)] == [
        "cheap"
    ]


def test_selection_by_tag_excludes_untagged_decks():
    selected = harness.select_decks(DECKS, None, ["ci"])
    assert [entry["identifier"] for entry in selected] == ["cheap"]


def test_selection_defaults_to_every_deck():
    assert len(harness.select_decks(DECKS, None, None)) == 3


def test_unknown_deck_is_rejected_rather_than_silently_running_nothing():
    with pytest.raises(ValueError, match="Unknown deck"):
        harness.select_decks(DECKS, ["typo"], None)


def test_selection_matching_nothing_is_an_error():
    with pytest.raises(ValueError, match="matched nothing"):
        harness.select_decks(DECKS, None, ["gpu"])


# ── Preparing a deck to run ───────────────────────────────────────────────────


def test_prepared_input_resolves_paths_and_redirects_output(deck, tmp_path):
    run_directory = tmp_path / "run"
    run_directory.mkdir()
    prepared = yaml.safe_load(harness.prepare_input(deck, run_directory).read_text())

    assert Path(prepared["material"]).is_absolute()
    assert Path(prepared["material"]).is_file()
    assert Path(prepared["method"]["pseudopotentials"]["Si"]).is_file()
    assert prepared["control"]["outdir"] == str(run_directory / "output")
    assert prepared["control"]["prefix"] == "silicon"
    assert prepared["control"]["use_deterministic_fft_planning"] is True


def test_prepared_input_applies_the_deck_overlay(deck, tmp_path):
    """An overlay merges into the deck rather than replacing whole branches."""
    run_directory = tmp_path / "run"
    run_directory.mkdir()
    deck["overlay"] = {"method": {"scf": {"max_iterations": 3}}}
    prepared = yaml.safe_load(harness.prepare_input(deck, run_directory).read_text())

    assert prepared["method"]["scf"]["max_iterations"] == 3
    assert prepared["method"]["scf"]["convergence_threshold"] == 1.0e-8
    assert prepared["method"]["pseudopotentials"]["Si"]


def test_absolute_paths_in_a_deck_are_left_alone(deck, tmp_path):
    manifest = yaml.safe_load(deck["input_path"].read_text())
    manifest["material"] = "/absolute/material.json"
    deck["input_path"].write_text(yaml.safe_dump(manifest))
    run_directory = tmp_path / "run"
    run_directory.mkdir()

    prepared = yaml.safe_load(harness.prepare_input(deck, run_directory).read_text())
    assert prepared["material"] == "/absolute/material.json"


# ── Reading the engine's timings ──────────────────────────────────────────────


def test_counts_are_separated_from_stage_times():
    """The engine emits both in one block; a count is not a millisecond."""
    stage_times, counts = harness.split_timings(
        {"total": 290.0, "seed": 86.7, "fft_calls": 929.0, "davidson_iterations": 53.0}
    )
    assert stage_times == {"total": 290.0, "seed": 86.7}
    assert counts == {"fft_calls": 929, "davidson_iterations": 53}


def test_a_run_records_timings_counts_and_wall_time(deck, stub_binary, tmp_path):
    record = harness.run_once(
        stub_binary(), deck, 1, tmp_path / "run", 60, {"PATH": "/usr/bin:/bin"}
    )

    assert record["deck"] == "silicon"
    assert record["repetition"] == 1
    assert record["timings_ms"]["total"] == 290.0
    assert record["counts"] == {
        "fft_calls": 929,
        "davidson_iterations": 53,
        "hamiltonian_applications": 320,
    }
    assert record["wall_seconds"] > 0.0
    assert record["converged"] is True
    assert record["self_consistent_field_iterations"] == 7
    assert record["peak_resident_memory_bytes"] == 39624704


def test_a_failed_run_reports_the_console_tail(deck, stub_binary, tmp_path):
    """A benchmark that silently records zero for a crashed run is worse than none."""
    with pytest.raises(RuntimeError, match="engine exploded"):
        harness.run_once(
            stub_binary(FAIL_LOUDLY), deck, 1, tmp_path / "run", 60, {"PATH": "/usr/bin:/bin"}
        )


def test_a_run_without_timings_is_an_error_not_a_zero(deck, stub_binary, tmp_path):
    with pytest.raises(RuntimeError, match="no timings_ms"):
        harness.run_once(
            stub_binary(WRITE_RESULTS_WITHOUT_TIMINGS),
            deck,
            1,
            tmp_path / "run",
            60,
            {"PATH": "/usr/bin:/bin"},
        )


def test_a_binary_that_is_not_executable_is_rejected(tmp_path):
    path = tmp_path / "not_a_binary"
    path.write_text("")
    with pytest.raises(ValueError, match="not executable"):
        harness.resolve_executable(path)


# ── Provenance ────────────────────────────────────────────────────────────────


def test_machine_label_names_a_class_of_machine_not_an_instance():
    """Two runs on the same box must agree, or no baseline is ever comparable."""
    assert harness.derive_machine_label() == harness.derive_machine_label()
    assert " " not in harness.derive_machine_label()


def test_every_record_carries_its_own_machine_and_commit(deck, stub_binary, tmp_path, monkeypatch):
    """End to end: records from two machines can be concatenated without confusion."""
    matrix_path = tmp_path / "matrix.yml"
    matrix_path.write_text(
        yaml.safe_dump(
            {
                "repetitions": 1,
                "decks": [{"identifier": "silicon", "input": str(deck["input_path"])}],
            }
        )
    )
    output_directory = tmp_path / "out"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_benchmarks.py",
            "--q3-bin",
            str(stub_binary()),
            "--matrix",
            str(matrix_path),
            "--output-directory",
            str(output_directory),
            "--machine-label",
            "test-runner",
        ],
    )
    assert harness.main() == 0

    payload = json.loads((output_directory / "records.json").read_text())
    assert payload["schema"] == "q3-benchmark-records-1"
    assert payload["metadata"]["machine"]["label"] == "test-runner"
    assert payload["metadata"]["binary"]["sha256"]
    assert payload["records"][0]["machine"] == "test-runner"
    assert payload["records"][0]["commit"] == payload["metadata"]["git_commit"]


def test_recorded_metadata_holds_no_machine_specific_absolute_path(tmp_path):
    """AGENTS.md § Testing, and the export gate: no home directory in a committed record."""
    inside = harness.PROJECT_ROOT / "scripts" / "benchmark" / "run_benchmarks.py"
    assert harness.describe_repository_path(inside) == "scripts/benchmark/run_benchmarks.py"

    outside = tmp_path / "somewhere" / "q3"
    assert harness.describe_repository_path(outside) == "q3"


def test_the_harness_runs_as_a_script(stub_binary, deck, tmp_path):
    """The command line the CI job actually invokes."""
    matrix_path = tmp_path / "matrix.yml"
    matrix_path.write_text(
        yaml.safe_dump(
            {
                "repetitions": 1,
                "decks": [
                    {"identifier": "silicon", "input": str(deck["input_path"]), "tags": ["ci"]}
                ],
            }
        )
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--q3-bin",
            str(stub_binary()),
            "--matrix",
            str(matrix_path),
            "--tag",
            "ci",
            "--output-directory",
            str(tmp_path / "out"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert (tmp_path / "out" / "records.json").is_file()
