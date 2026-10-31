"""tests/python/test_execution_policy.py — one execution policy, every engine run of a workflow.

A workflow is several engine runs: a band structure is an SCF and then an NSCF
along the path, a relaxation one SCF per geometry. The launcher and the
binary's parallel options used to reach the relaxation's steps alone, and only
by its own keyword, so the NSCF of every band structure ran on one process
whatever the SCF was given. These run the workflows against a stand-in
launcher and binary that record how they were called, and check that every
run was launched the same way.
"""

import json
import sys
import types
from pathlib import Path

import jsonschema
import pytest
import yaml
from pydantic import ValidationError
from q3.binary import ExecutionPolicy, run_q3
from q3.calculators.ase_calculator import Q3ASECalculator
from q3.errors import Q3RuntimeError
from q3.flavors import run_scf
from q3.io.results.classes import Results
from q3.workflows import run_band_structure_workflow

SILICON_MATERIAL = (
    Path(__file__).resolve().parents[2] / "examples" / "data" / "materials" / "silicon.json"
)

SCHEMA = json.loads(
    (Path(__file__).resolve().parents[2] / "schemas" / "q3" / "execution_policy.json").read_text()
)

NUMBER_OF_PROCESSES = 4
BINARY_OPTIONS = ("--pools", "4")


# ── The value itself ─────────────────────────────────────────────────────────


def test_one_process_on_the_defaults_is_the_binary_alone():
    execution_policy = ExecutionPolicy()
    assert execution_policy.is_default
    assert execution_policy.command("q3", "scf.yml") == ["q3", "scf.yml"]


def test_more_processes_go_through_the_launcher_and_options_follow_the_input():
    execution_policy = ExecutionPolicy(
        number_of_processes=2, number_of_kpoint_pools=2, distribute_bands=True
    )
    command = execution_policy.command(Path("/opt/q3_mpi"), "scf.yml", "--material", "step.json")
    assert command == [
        "mpirun",
        "-np",
        "2",
        "/opt/q3_mpi",
        "scf.yml",
        "--material",
        "step.json",
        "--pools",
        "2",
        "--distribute-bands",
    ]
    assert not execution_policy.is_default


def test_a_launcher_string_is_split_as_a_shell_would_and_takes_the_rank_count():
    execution_policy = ExecutionPolicy(
        number_of_processes=64, launcher="srun -n {number_of_processes} --gpus-per-task 1"
    )
    assert execution_policy.launcher == (
        "srun",
        "-n",
        "{number_of_processes}",
        "--gpus-per-task",
        "1",
    )
    assert execution_policy.launch_words == ["srun", "-n", "64", "--gpus-per-task", "1"]


def test_a_binary_option_on_one_process_is_not_serial():
    execution_policy = ExecutionPolicy(precision="mixed")
    assert execution_policy.launch_words == []
    assert execution_policy.command("q3_gpu", "scf.yml") == [
        "q3_gpu",
        "scf.yml",
        "--precision",
        "mixed",
    ]
    assert not execution_policy.is_default


def test_results_from_another_process_count_are_refused():
    """A serial binary under mpirun runs one copy per process, each reporting one."""
    execution_policy = ExecutionPolicy(number_of_processes=4)
    execution_policy.check_results(
        Results.model_validate({"runtime_log": {"execution": {"number_of_processes": 4}}})
    )
    execution_policy.check_results(Results.model_validate({}))
    with pytest.raises(Q3RuntimeError, match="MPI build"):
        execution_policy.check_results(
            Results.model_validate({"runtime_log": {"execution": {"number_of_processes": 1}}})
        )


def test_the_model_has_the_fields_the_schema_describes():
    assert set(ExecutionPolicy.model_fields) == set(SCHEMA["properties"])
    defaults = ExecutionPolicy().model_dump(mode="json")
    for name, description in SCHEMA["properties"].items():
        assert defaults[name] == description["default"], name


def test_a_policy_dumps_to_what_the_schema_accepts_and_loads_back():
    execution_policy = ExecutionPolicy(
        number_of_processes=8,
        number_of_kpoint_pools=8,
        precision="double",
        launcher="srun -n {number_of_processes}",
    )
    dumped = execution_policy.model_dump(mode="json")
    jsonschema.validate(dumped, SCHEMA)
    assert ExecutionPolicy.model_validate(dumped) == execution_policy


@pytest.mark.parametrize(
    "fields",
    [
        {"ranks": 4},
        {"number_of_processes": 0},
        {"number_of_kpoint_pools": 0},
        {"precision": "single"},
    ],
)
def test_the_model_and_the_schema_refuse_the_same_values(fields):
    with pytest.raises(ValidationError):
        ExecutionPolicy(**fields)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(fields, SCHEMA)


def test_a_policy_is_frozen():
    with pytest.raises(ValidationError):
        ExecutionPolicy().number_of_processes = 2


# ── Stand-ins that record how they were launched ─────────────────────────────


def _stand_ins(tmp_path: Path) -> tuple[Path, Path, Path]:
    """A launcher and a binary that log their arguments; the binary leaves a result.

    The binary reads the input as the engine would -- outdir and prefix, and the
    k-point list of an NSCF -- and writes the results file run_q3 reads back.
    """
    log_path = tmp_path / "launches.jsonl"
    launcher = tmp_path / "launch"
    launcher.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        f"with open({str(log_path)!r}, 'a') as log:\n"
        "    log.write(json.dumps({'launcher': sys.argv[1:]}) + '\\n')\n"
        "skip = 3\n"
        "os.execv(sys.argv[skip], sys.argv[skip:])\n"
    )
    binary = tmp_path / "q3_stand_in"
    binary.write_text(
        f"#!{sys.executable}\n"
        "import json, sys, pathlib, yaml\n"
        "input_path = pathlib.Path(next(a for a in sys.argv[1:] if a.endswith('.yml')))\n"
        "config = yaml.safe_load(input_path.read_text())\n"
        "control = config.get('control', {})\n"
        "outdir = input_path.parent / control.get('outdir', '.')\n"
        "outdir.mkdir(parents=True, exist_ok=True)\n"
        "kpoints = control.get('kpoints_list') or [None]\n"
        "results = {'runtime_log': {'flavor': control['calculation'], 'converged': True},\n"
        "           'properties': {\n"
        "               'records': [{'name': 'fermi_energy', 'units': 'Ry', 'value': 0.0},\n"
        "                           {'name': 'total_energy', 'units': 'Ry', 'value': -15.8},\n"
        "                           {'name': 'eigenvalues_at_kpoints', 'units': 'Ry', 'values': [\n"
        "                   {'eigenvalues': [{'energies': [0.1, 0.2, 0.3, 0.4]}]} for _ in kpoints]}]}}\n"
        "(outdir / (control.get('prefix', 'q3') + '_results.json')).write_text(json.dumps(results))\n"
        f"with open({str(log_path)!r}, 'a') as log:\n"
        "    log.write(json.dumps({'binary': sys.argv[1:],\n"
        "                          'calculation': control['calculation']}) + '\\n')\n"
    )
    for script in (launcher, binary):
        script.chmod(0o755)
    return launcher, binary, log_path


def _launches(log_path: Path) -> list[dict]:
    return [json.loads(line) for line in log_path.read_text().splitlines()]


def _execution_policy(launcher: Path) -> ExecutionPolicy:
    """Four ranks on four pools, started through the logging stand-in launcher."""
    return ExecutionPolicy(
        number_of_processes=NUMBER_OF_PROCESSES,
        number_of_kpoint_pools=NUMBER_OF_PROCESSES,
        launcher=[str(launcher), "-n", "{number_of_processes}"],
    )


# ── Every unit of a workflow ─────────────────────────────────────────────────


def test_the_band_structure_runs_its_scf_and_its_nscf_on_the_same_setup(tmp_path):
    launcher, binary, log_path = _stand_ins(tmp_path)
    scf_input = tmp_path / "scf.yml"
    scf_input.write_text(
        yaml.safe_dump(
            {
                "material": str(SILICON_MATERIAL),
                "control": {
                    "calculation": "band_structure",
                    "outdir": "output",
                    "prefix": "si",
                    "band_path": ["G", "X"],
                    "band_path_points_per_segment": 3,
                },
            }
        )
    )

    result = run_band_structure_workflow(
        scf_input, q3_binary=binary, execution_policy=_execution_policy(launcher)
    )

    launches = _launches(log_path)
    runs = [entry for entry in launches if "binary" in entry]
    assert [run["calculation"] for run in runs] == [
        "self_consistent_field",
        "nscf",
    ]
    assert [entry["launcher"][:2] for entry in launches[::2]] == [
        ["-n", str(NUMBER_OF_PROCESSES)]
    ] * 2
    for run in runs:
        assert run["binary"][-len(BINARY_OPTIONS) :] == list(BINARY_OPTIONS)
    assert len(result["band_structure"]["eigenvalues_ry"]) == len(
        result["band_structure"]["kpoints"]
    )


def test_run_q3_without_a_setup_runs_the_binary_alone(tmp_path):
    _, binary, log_path = _stand_ins(tmp_path)
    scf_input = tmp_path / "scf.yml"
    scf_input.write_text(
        yaml.safe_dump({"control": {"calculation": "self_consistent_field", "outdir": "output"}})
    )

    run_q3(scf_input, q3_binary=binary)

    (launch,) = _launches(log_path)
    assert launch["binary"] == [str(scf_input.resolve())]


# ── The in-process path cannot honour a setup ────────────────────────────────


def _in_process_engine(monkeypatch, module: str) -> list[str]:
    calls: list[str] = []
    stand_in = types.ModuleType("q3_core")
    stand_in.run_scf = lambda path, write_output_files=True: calls.append(path) or "{}"
    stand_in.SCFSolver = lambda path: calls.append(path)
    monkeypatch.setattr(f"{module}.q3_core", stand_in)
    return calls


def test_run_scf_launches_the_binary_rather_than_drop_the_setup(tmp_path, monkeypatch):
    in_process_calls = _in_process_engine(monkeypatch, "q3.flavors.self_consistent_field")
    launcher, binary, log_path = _stand_ins(tmp_path)
    scf_input = tmp_path / "scf.yml"
    scf_input.write_text(
        yaml.safe_dump({"control": {"calculation": "self_consistent_field", "outdir": "output"}})
    )

    run_scf(scf_input, q3_binary=binary, execution_policy=_execution_policy(launcher))

    assert in_process_calls == []
    assert any("launcher" in entry for entry in _launches(log_path))


def test_the_calculator_picks_the_binary_for_a_setup_and_refuses_bindings(tmp_path, monkeypatch):
    in_process_calls = _in_process_engine(monkeypatch, "q3.calculators.ase_calculator")
    launcher, binary, _ = _stand_ins(tmp_path)
    scf_input = tmp_path / "scf.yml"
    scf_input.write_text(yaml.safe_dump({"control": {"calculation": "self_consistent_field"}}))

    calculator = Q3ASECalculator(
        scf_input, q3_binary=binary, execution_policy=_execution_policy(launcher)
    )
    assert calculator._mode == "subprocess"
    assert in_process_calls == []

    with pytest.raises(ValueError, match="execution policy"):
        Q3ASECalculator(
            scf_input,
            mode="bindings",
            q3_binary=binary,
            execution_policy=_execution_policy(launcher),
        )
