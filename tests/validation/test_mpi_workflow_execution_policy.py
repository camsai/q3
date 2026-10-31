"""
Every engine run of a Python workflow, started on `q3_mpi` with one execution policy.

A workflow is several engine runs: a band structure is an SCF and then an NSCF
along the path, a relaxation one SCF per geometry. The execution policy handed
to the workflow has to reach each of them -- before it did, only the
relaxation's steps took a launcher, and `q3_mpi` refused the NSCF outright.

The pin: run the real workflows through a launcher that records, after each
run, the `execution` block the engine wrote into that run's results, and check
that every run -- not just the first -- reports the ranks and pools asked for.
tests/python/test_execution_policy.py pins the commands against a stand-in
binary; this pins what the engine itself says it ran on.
"""

import json
import shutil
import sys
from pathlib import Path

import pytest
from q3.binary import ExecutionPolicy
from q3.workflows import run_band_structure_workflow, run_relax_workflow

from helpers import EXAMPLES_DIRECTORY

pytestmark = pytest.mark.skipif(
    shutil.which("mpirun") is None,
    reason="mpirun not found — MPI tests require an MPI installation",
)

NUMBER_OF_PROCESSES = 2

RECORDING_LAUNCHER = """
import json, pathlib, subprocess, sys, yaml
log_path, command = sys.argv[1], sys.argv[2:]
exit_code = subprocess.call(command)
input_path = pathlib.Path(next(word for word in command if word.endswith(".yml")))
control = yaml.safe_load(input_path.read_text()).get("control", {})
results_path = (
    input_path.parent / control.get("outdir", ".") / (control.get("prefix", "q3") + "_results.json")
)
runtime_log = json.loads(results_path.read_text())["runtime_log"] if results_path.exists() else {}
with open(log_path, "a") as log:
    log.write(json.dumps({"exit_code": exit_code, "flavor": runtime_log.get("flavor"),
                          "execution": runtime_log.get("execution")}) + "\\n")
sys.exit(exit_code)
"""


@pytest.fixture
def recorded_execution_policy(tmp_path):
    """Two ranks on two pools, through mpirun wrapped in a launcher that records each run."""
    launcher_path = tmp_path / "recording_launcher.py"
    launcher_path.write_text(RECORDING_LAUNCHER)
    log_path = tmp_path / "runs.jsonl"
    execution_policy = ExecutionPolicy(
        number_of_processes=NUMBER_OF_PROCESSES,
        number_of_kpoint_pools=NUMBER_OF_PROCESSES,
        launcher=[
            sys.executable,
            str(launcher_path),
            str(log_path),
            "mpirun",
            "-np",
            "{number_of_processes}",
            "--oversubscribe",
        ],
    )
    return execution_policy, log_path


def _recorded_runs(log_path: Path) -> list[dict]:
    return [json.loads(line) for line in log_path.read_text().splitlines()]


def _assert_every_run_used_the_policy(runs):
    for run in runs:
        assert run["exit_code"] == 0, run
        assert run["execution"]["number_of_processes"] == NUMBER_OF_PROCESSES, run
        assert run["execution"]["number_of_kpoint_pools"] == NUMBER_OF_PROCESSES, run
        assert run["execution"]["peak_resident_memory_bytes"] > 0, run


def test_band_structure_runs_its_scf_and_its_nscf_on_the_policy(
    q3_bin, recorded_execution_policy, tmp_path
):
    execution_policy, log_path = recorded_execution_policy
    example = shutil.copytree(
        EXAMPLES_DIRECTORY / "band_structure--semiconductor--silicon", tmp_path / "bands"
    )
    (tmp_path / "data").symlink_to(EXAMPLES_DIRECTORY / "data")

    run_band_structure_workflow(
        example / "nc.yml", q3_binary=q3_bin, execution_policy=execution_policy
    )

    runs = _recorded_runs(log_path)
    assert [run["flavor"] for run in runs] == [
        "self_consistent_field",
        "non_self_consistent_field",
    ]
    _assert_every_run_used_the_policy(runs)


def test_relaxation_runs_every_step_on_the_policy(q3_bin, recorded_execution_policy, tmp_path):
    execution_policy, log_path = recorded_execution_policy
    example = shutil.copytree(
        EXAMPLES_DIRECTORY / "relaxation--semiconductor--silicon", tmp_path / "relax"
    )
    (tmp_path / "data").symlink_to(EXAMPLES_DIRECTORY / "data")

    run_relax_workflow(
        example / "nc.yml",
        q3_binary=q3_bin,
        execution_policy=execution_policy,
        mode="subprocess",
        steps=3,
        workdir=tmp_path / "relax_work",
    )

    runs = _recorded_runs(log_path)
    assert len(runs) >= 2, "the displaced geometry needs at least one step after the first"
    _assert_every_run_used_the_policy(runs)
