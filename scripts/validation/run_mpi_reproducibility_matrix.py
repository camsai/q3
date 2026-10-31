#!/usr/bin/env python3
"""Run the Package 2 serial, MPI, and optional GPU reproducibility matrix.

The matrix uses fresh run directories and compares every reported energy
component plus stable result metadata. Each topology is repeated three times
by default so a deterministic same-topology spread is distinguished from a
cross-topology physics difference.

It doubles as the benchmarking harness, which is why it records timing and
placement alongside the energies:

  * ``timings_ms`` from results.json -- the Davidson stage breakdown and call
    counts, read from the file rather than scraped from console text. A run
    that aborts has no timings block, which shows up as a missing key rather
    than as a plausible-looking blank.
  * The node the work actually ran on, and the intra-pool split it used.
  * Whether the replicated-G fallback fired. A run that fell back is measuring
    different work, not the same work more slowly, so a timing comparison that
    ignores it is comparing two different calculations.

Launchers: ``--launcher mpirun`` (default) for a workstation, ``--launcher
srun`` inside a Slurm allocation. srun is what makes multi-node topologies and
``--nodes`` meaningful; under ``salloc <command>`` the script body itself runs
on the login node, so anything that must run on the allocation has to go
through srun rather than being invoked directly.
"""

import argparse
import copy
import hashlib
import json
import logging
import math
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "py"))

from q3.io.results.classes import (  # noqa: E402 -- needs the sys.path entry above
    Results,
    eigenvalue_arrays,
)

# The units the tolerances are stated in: the engine writes every energy in Ry
# and the forces in Ry/a.u., and each quantity's own ``units`` says so.
ENERGY_UNITS = "Ry"
FORCE_UNITS = "Ry/a.u."
# Compared exactly: runtime_log's flavor and converged, and why the forces are
# incomplete (None when they are not). The stress is left out: whether it is
# complete depends on the topology, and validate_stress_refusal checks it.
EXACT_METADATA_FIELDS = (
    "flavor",
    "converged",
    "forces_incomplete",
)
# The ESSE property records compared, by name.
ESSE_ENERGIES = ("total_energy", "fermi_energy")
# The ESSE total_energy_contributions terms compared, by ESSE's own keys.
ESSE_CONTRIBUTIONS = (
    "eigenvalues",
    "hartree",
    "exchange_correlation",
    "ewald",
    "temperatureEntropy",
)
# Q3's energy_contributions ESSE has no key for, by Q3's keys.
Q3_CONTRIBUTIONS = ("double_counting", "one_center", "hubbard")
THREAD_ENVIRONMENT = {
    "BLIS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "OMP_DYNAMIC": "FALSE",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}
DEFAULT_TOPOLOGY_IDENTIFIERS = (
    "serial",
    "mpi_1_rank",
    "mpi_2_ranks",
    "mpi_4_ranks",
)


@dataclass(frozen=True)
class ReproducibilityCase:
    identifier: str
    manifest: str
    prefix: str
    overlay: dict
    total_energy_tolerance_rydberg: float = 1.0e-8
    component_energy_tolerance_rydberg: float = 1.0e-8
    # Benchmark decks cap max_iterations to time a fixed amount of work, so
    # they stop before convergence on purpose. Their energies are still
    # compared across topologies -- the same truncated calculation has to give
    # the same truncated answer -- but "did not converge" is the intended
    # outcome rather than a failure.
    requires_convergence: bool = True
    # Excluded from the default selection: minutes to hours each, and the
    # largest needs about 45 GB per rank with the basis replicated.
    is_benchmark_only: bool = False


@dataclass(frozen=True)
class ExecutionTopology:
    identifier: str
    command: tuple[str, ...]
    binary_path: str
    number_of_processes: int
    number_of_kpoint_pools: int | None
    number_of_nodes: int = 1
    # Which split is used inside a pool. Q3 has two and they are not
    # interchangeable: the default distributes reciprocal space (G-vector
    # slabs), --distribute-bands distributes bands and sets group_size() to 1.
    # Recorded per topology because a timing is not interpretable without it.
    distributes_bands: bool = False
    threads_per_process: int = 1


@dataclass
class RunRecord:
    case: str
    topology: str
    repetition: int
    wall_seconds: float
    result_path: str
    results: dict
    # Benchmarking context. wall_seconds alone is not usable on a cluster:
    # srun startup measured 2-9 s here, which swamps any run shorter than a
    # minute, so the solver's own total is the figure to compare and the wall
    # is kept only to make that gap visible.
    timings_ms: dict | None = None
    # The allocation's nodes, not the step's -- see read_allocation_node_names.
    allocation_node_names: str = ""
    number_of_nodes: int = 1
    distributes_bands: bool = False
    threads_per_process: int = 1
    replicated_fallback_active: bool = False


HUBBARD_OVERLAY = {
    "method": {
        "cutoffs": {"wavefunction": 30.0, "density": 120.0},
        "scf": {
            "convergence_threshold": 1.0e-16,
            "kgrid": {"dimensions": [4, 4, 4]},
            "hubbard": [{"species": "Ni", "orbital_label": "3D", "u_ev": 3.0}],
        },
    }
}

REPRODUCIBILITY_CASES = (
    ReproducibilityCase(
        identifier="silicon_norm_conserving",
        manifest="examples/total_energy--semiconductor--silicon/nc.yml",
        prefix="si_nc",
        overlay={},
    ),
    ReproducibilityCase(
        identifier="silicon_makov_payne",
        manifest="examples/total_energy--atom--silicon/nc_charged.yml",
        prefix="silicon_atom",
        overlay={"control": {"isolated_system_correction": "makov-payne"}},
    ),
    ReproducibilityCase(
        identifier="silicon_martyna_tuckerman",
        manifest="examples/total_energy--atom--silicon/nc_charged.yml",
        prefix="silicon_atom",
        overlay={"control": {"isolated_system_correction": "martyna-tuckerman"}},
    ),
    ReproducibilityCase(
        identifier="nickel_lsda",
        manifest="examples/total_energy--metal--magnetic--nickel/nc.yml",
        prefix="ni_nc",
        overlay={
            "method": {
                "cutoffs": {"wavefunction": 30.0, "density": 120.0},
                "scf": {
                    "convergence_threshold": 1.0e-16,
                    "kgrid": {"dimensions": [4, 4, 4]},
                },
            }
        },
        component_energy_tolerance_rydberg=1.0e-6,
    ),
    ReproducibilityCase(
        identifier="nickel_lsda_hubbard_atomic",
        manifest="examples/total_energy--metal--magnetic--nickel/nc.yml",
        prefix="ni_nc",
        overlay={
            **HUBBARD_OVERLAY,
            "method": {
                **HUBBARD_OVERLAY["method"],
                "scf": {
                    **HUBBARD_OVERLAY["method"]["scf"],
                    "hubbard_projectors": "atomic",
                },
            },
        },
        component_energy_tolerance_rydberg=1.0e-6,
    ),
    ReproducibilityCase(
        identifier="nickel_lsda_hubbard_ortho_atomic",
        manifest="examples/total_energy--metal--magnetic--nickel/nc.yml",
        prefix="ni_nc",
        overlay={
            **HUBBARD_OVERLAY,
            "method": {
                **HUBBARD_OVERLAY["method"],
                "scf": {
                    **HUBBARD_OVERLAY["method"]["scf"],
                    "hubbard_projectors": "ortho-atomic",
                },
            },
        },
        component_energy_tolerance_rydberg=1.0e-6,
    ),
    ReproducibilityCase(
        identifier="silicon_ultrasoft",
        manifest="examples/total_energy--semiconductor--silicon/us.yml",
        prefix="si_us",
        overlay={"method": {"scf": {"convergence_threshold": 1.0e-16}}},
        component_energy_tolerance_rydberg=1.0e-6,
    ),
    ReproducibilityCase(
        identifier="silicon_paw",
        manifest="examples/total_energy--semiconductor--silicon/paw.yml",
        prefix="si_paw",
        overlay={"method": {"scf": {"convergence_threshold": 1.0e-16}}},
        component_energy_tolerance_rydberg=1.0e-6,
    ),
    # ─── Benchmark ladder ────────────────────────────────────────────────────
    # Four decks chosen because they exercise different parallelism, not
    # because they are four sizes of the same thing. Opt in with --case; none
    # runs by default. See docs/benchmarks/ for the protocol.
    #
    # Energies are still compared across topologies: a benchmark deck that
    # gives a different answer on four ranks than on one is reporting a bug,
    # whatever its timings say.
    ReproducibilityCase(
        identifier="small_silicon_kpoints",
        manifest="examples/total_energy--semiconductor--silicon/nc_kpt_8x8x8.yml",
        prefix="si_nc_kpt_8x8x8",
        overlay={},
        is_benchmark_only=True,
    ),
    # 27 atoms, 280 bands, one k-point: pools are unusable, so this is the
    # deck for threading, the intra-pool splits and the GPU. max_iterations: 3
    # in the deck itself, hence requires_convergence=False.
    ReproducibilityCase(
        identifier="medium_copper_gamma",
        manifest="examples/total_energy--metal--copper/nc_3x3x3_bench.yml",
        prefix="cu_nc_3x3x3_bench",
        overlay={},
        # A capped deck is compared mid-convergence, where the topologies have
        # not yet agreed: distributing G-vectors or bands reorders the sums, so
        # the iterate they stop on differs slightly.
        #
        # The total and its components need different tolerances, and the gap
        # between them is not slack -- it is what the total energy being
        # variational means. Near the solution the total is stationary with
        # respect to density error and so is second-order in it, while each
        # component is first-order and they move in compensating directions.
        # Measured on the copper deck: total energies agree to better than
        # 1e-2 Ry across every topology, while double_counting_correction and
        # exchange_correlation differ by 1.6e-1 and 4.6e-2 Ry.
        #
        # So the total stays the meaningful cross-topology invariant, tight
        # enough that a lost k-point or a broken reduction still fails it, and
        # the components are loosened to where a truncated run can legitimately
        # land. Converged decks keep the 1e-8 default for both.
        total_energy_tolerance_rydberg=1.0e-2,
        component_energy_tolerance_rydberg=1.0,
        requires_convergence=False,
        is_benchmark_only=True,
    ),
    # 432 atoms, 1040 bands, ~167k plane waves. Its own comment says it needs
    # about 45 GB per rank with the basis replicated, so it does not fit a
    # 40 GB A100 that way and a 512 GB node holds roughly eleven such ranks.
    # That is the point of it: it forces the distribution axes to be used.
    # Capped here rather than in the deck, which has no max_iterations.
    ReproducibilityCase(
        identifier="large_silicon_gamma",
        manifest="examples/total_energy--semiconductor--silicon/nc_6x6x6.yml",
        prefix="si_nc_6x6x6",
        overlay={"method": {"scf": {"max_iterations": 3}}},
        # A capped deck is compared mid-convergence, where the topologies have
        # not yet agreed: distributing G-vectors or bands reorders the sums, so
        # the iterate they stop on differs slightly.
        #
        # The total and its components need different tolerances, and the gap
        # between them is not slack -- it is what the total energy being
        # variational means. Near the solution the total is stationary with
        # respect to density error and so is second-order in it, while each
        # component is first-order and they move in compensating directions.
        # Measured on the copper deck: total energies agree to better than
        # 1e-2 Ry across every topology, while double_counting_correction and
        # exchange_correlation differ by 1.6e-1 and 4.6e-2 Ry.
        #
        # So the total stays the meaningful cross-topology invariant, tight
        # enough that a lost k-point or a broken reduction still fails it, and
        # the components are loosened to where a truncated run can legitimately
        # land. Converged decks keep the 1e-8 default for both.
        total_energy_tolerance_rydberg=1.0e-2,
        component_energy_tolerance_rydberg=1.0,
        requires_convergence=False,
        is_benchmark_only=True,
    ),
    # 12x12x6 k-grid: the only ladder entry with enough irreducible k-points
    # for pools alone to cross a node boundary.
    ReproducibilityCase(
        identifier="wide_k_graphite",
        manifest="examples/total_energy--semimetal--graphite/nc_d3.yml",
        prefix="graphite_nc_d3",
        overlay={"method": {"scf": {"max_iterations": 3}}},
        # A capped deck is compared mid-convergence, where the topologies have
        # not yet agreed: distributing G-vectors or bands reorders the sums, so
        # the iterate they stop on differs slightly.
        #
        # The total and its components need different tolerances, and the gap
        # between them is not slack -- it is what the total energy being
        # variational means. Near the solution the total is stationary with
        # respect to density error and so is second-order in it, while each
        # component is first-order and they move in compensating directions.
        # Measured on the copper deck: total energies agree to better than
        # 1e-2 Ry across every topology, while double_counting_correction and
        # exchange_correlation differ by 1.6e-1 and 4.6e-2 Ry.
        #
        # So the total stays the meaningful cross-topology invariant, tight
        # enough that a lost k-point or a broken reduction still fails it, and
        # the components are loosened to where a truncated run can legitimately
        # land. Converged decks keep the 1e-8 default for both.
        total_energy_tolerance_rydberg=1.0e-2,
        component_energy_tolerance_rydberg=1.0,
        requires_convergence=False,
        is_benchmark_only=True,
    ),
)


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial-binary", required=True, type=Path)
    parser.add_argument("--mpi-binary", required=True, type=Path)
    parser.add_argument("--gpu-binary", type=Path)
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=int, default=600)
    parser.add_argument(
        "--case",
        dest="selected_case_identifiers",
        action="append",
        choices=[case.identifier for case in REPRODUCIBILITY_CASES],
        help="Run only this case; repeat the option to select more than one.",
    )
    parser.add_argument(
        "--topology",
        dest="selected_topology_identifiers",
        action="append",
        choices=[
            *DEFAULT_TOPOLOGY_IDENTIFIERS,
            "mpi_2_ranks_2_pools",
            "mpi_4_ranks_2_pools",
            "mpi_4_ranks_4_pools",
            "mpi_2_ranks_bands",
            "mpi_4_ranks_bands",
            "mpi_4_ranks_2_pools_bands",
            "mpi_8_ranks_8_pools",
            "mpi_16_ranks_16_pools",
            "mpi_2_nodes_8_ranks",
            "mpi_2_nodes_16_ranks",
            "gpu",
        ],
        help="Run only this topology; repeat the option to select more than one.",
    )
    parser.add_argument(
        "--launcher",
        choices=("mpirun", "srun"),
        default="mpirun",
        help=(
            "How to start MPI ranks. Use srun inside a Slurm allocation: it is "
            "required for the multi-node topologies, and under `salloc <command>` "
            "it is the only way the work reaches the allocated nodes at all."
        ),
    )
    parser.add_argument(
        "--convergence-threshold",
        type=float,
        help="Override method.scf.convergence_threshold for a recorded diagnostic run.",
    )
    return parser.parse_args()


def configure_logging(output_directory):
    logger = logging.getLogger("mpi_reproducibility_matrix")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    for handler in (
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(output_directory / "matrix.log", encoding="utf-8"),
    ):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def create_output_directory(requested_directory):
    if requested_directory is None:
        timestamp = datetime.now().strftime("%Y-%m-%dT%H%M%S")
        requested_directory = (
            PROJECT_ROOT
            / "agents"
            / "workdir"
            / "tmp"
            / "reproducibility"
            / f"mpi_matrix_{timestamp}"
        )
    output_directory = requested_directory.expanduser().resolve()
    if output_directory.exists() and any(output_directory.iterdir()):
        raise ValueError(f"Output directory is not empty: {output_directory}")
    output_directory.mkdir(parents=True, exist_ok=True)
    return output_directory


def resolve_executable(path, label):
    resolved = path.expanduser().resolve()
    if not resolved.is_file() or not os.access(resolved, os.X_OK):
        raise ValueError(f"{label} is not executable: {resolved}")
    return resolved


def compute_binary_digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for block in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def deep_merge(base, overlay):
    merged = copy.deepcopy(base)
    for key, value in overlay.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def prepare_manifest(case, topology, run_directory, convergence_threshold):
    source_path = PROJECT_ROOT / case.manifest
    with source_path.open(encoding="utf-8") as file_handle:
        manifest = deep_merge(yaml.safe_load(file_handle), case.overlay)

    material_path = Path(manifest["material"])
    if not material_path.is_absolute():
        manifest["material"] = str((source_path.parent / material_path).resolve())
    for element, pseudopotential in manifest["method"]["pseudopotentials"].items():
        pseudopotential_path = Path(pseudopotential)
        if not pseudopotential_path.is_absolute():
            manifest["method"]["pseudopotentials"][element] = str(
                (source_path.parent / pseudopotential_path).resolve()
            )

    control = manifest.setdefault("control", {})
    control["outdir"] = str(run_directory / "output")
    control["prefix"] = case.prefix
    control["use_deterministic_fft_planning"] = True
    if convergence_threshold is not None:
        manifest["method"]["scf"]["convergence_threshold"] = convergence_threshold
    if topology.number_of_kpoint_pools is None:
        control.pop("number_of_kpoint_pools", None)
    else:
        control["number_of_kpoint_pools"] = topology.number_of_kpoint_pools

    input_path = run_directory / "input.yml"
    with input_path.open("w", encoding="utf-8") as file_handle:
        yaml.safe_dump(manifest, file_handle, sort_keys=False)
    return input_path


def run_one(
    case,
    topology,
    repetition,
    output_directory,
    timeout_seconds,
    environment,
    convergence_threshold,
):
    run_directory = output_directory / case.identifier / topology.identifier / f"run_{repetition}"
    run_directory.mkdir(parents=True)
    input_path = prepare_manifest(case, topology, run_directory, convergence_threshold)
    standard_output_path = run_directory / "standard_output.log"
    command = [*topology.command, str(input_path)]
    started = time.perf_counter()
    with standard_output_path.open("w", encoding="utf-8") as standard_output:
        completed = subprocess.run(
            command,
            cwd=run_directory,
            env=environment,
            stdout=standard_output,
            stderr=subprocess.STDOUT,
            timeout=timeout_seconds,
            check=False,
        )
    wall_seconds = time.perf_counter() - started
    if completed.returncode != 0:
        output_tail = standard_output_path.read_text(errors="replace")[-20000:]
        raise RuntimeError(
            f"{' '.join(command)} exited with code {completed.returncode}\n{output_tail}"
        )
    result_path = run_directory / "output" / f"{case.prefix}_results.json"
    with result_path.open(encoding="utf-8") as file_handle:
        results = json.load(file_handle)
    if case.requires_convergence and not results["runtime_log"].get("converged"):
        raise RuntimeError(f"Calculation did not converge: {run_directory}")
    console_text = standard_output_path.read_text(errors="replace")
    return RunRecord(
        case=case.identifier,
        topology=topology.identifier,
        repetition=repetition,
        wall_seconds=wall_seconds,
        result_path=str(result_path),
        results=results,
        timings_ms=results["runtime_log"].get("timings_ms"),
        allocation_node_names=read_allocation_node_names(console_text),
        number_of_nodes=topology.number_of_nodes,
        distributes_bands=topology.distributes_bands,
        threads_per_process=topology.threads_per_process,
        replicated_fallback_active=("Falling back to replicated" in console_text),
    )


def read_allocation_node_names(console_text):
    """Nodes in the ALLOCATION, which is not the same as the nodes a step used.

    Recorded because runs in this session landed on six different nodes and
    nothing establishes they are interchangeable: a timing compared across them
    without knowing that measures the machine, not the code.

    The limitation is deliberate and worth stating rather than papering over.
    This reads the allocation's node list from the launching environment, so
    every topology in one allocation reports the same value -- a single-node
    topology inside a two-node allocation lists both. It is provenance for the
    run, not proof that a multi-node topology actually spanned nodes. Q3's log
    reports rank and pool counts but no hostnames, so confirming that needs
    either a separate `srun hostname` or a per-rank hostname in the log.

    Empty outside Slurm, where there is one machine and the question does not
    arise.
    """
    for variable in ("SLURM_JOB_NODELIST", "SLURM_NODELIST"):
        value = os.environ.get(variable)
        if value:
            return value
    return ""


def nested_shape(value):
    if not isinstance(value, list):
        return ()
    return (len(value), *(nested_shape(value[0]) if value else ()))


def flatten_numeric_values(value):
    if isinstance(value, list):
        return [item for nested in value for item in flatten_numeric_values(nested)]
    return [float(value)]


def energies(properties):
    """The energies *properties* carries, by their ESSE (or Q3) name, in ENERGY_UNITS."""
    values = {}
    for name in ESSE_ENERGIES:
        record = properties.record(name)
        if record is not None:
            values[name] = record.value
    contributions = properties.total_energy_contributions
    if contributions is not None:
        terms = contributions.model_dump(by_alias=True, exclude_none=True)
        values.update({key: terms[key]["value"] for key in ESSE_CONTRIBUTIONS if key in terms})
    for key in Q3_CONTRIBUTIONS:
        quantity = getattr(properties.energy_contributions, key, None)
        if quantity is not None:
            values[key] = quantity.value
    return values


def comparable_fields(results):
    """*results* with the quantities it carries read out as plain numbers.

    The comparison below works field by field, so runtime_log's metadata, the ESSE
    records results.json carries under "properties" and Q3's own unit-carrying
    quantities are read,
    each in the unit the tolerances are stated in (ENERGY_UNITS, FORCE_UNITS):
    the energies by name, atomic_forces in atom order and the eigenvalues
    arrays indexed [k-point][spin][band]. The file itself is not changed.
    """
    model = Results.model_validate(results)
    runtime_log = model.runtime_log
    properties = model.properties
    fields = {
        "flavor": runtime_log.flavor,
        "converged": runtime_log.converged,
        "forces_incomplete": runtime_log.incomplete.forces if runtime_log.incomplete else None,
        "energies": energies(properties),
    }
    forces = properties.atomic_forces
    if forces is not None:
        fields["forces"] = [atom.value for atom in forces.values]
    eigenvalue_record = properties.eigenvalues_at_kpoints
    if eigenvalue_record is not None:
        eigenvalues, occupations, weights = eigenvalue_arrays(eigenvalue_record)
        fields["eigenvalues"] = eigenvalues
        fields["kpoint_weights"] = weights
        fields["kpoint_coordinates"] = [
            kpoint.kpoint or [] for kpoint in eigenvalue_record.values or []
        ]
        if occupations is not None:
            fields["occupations"] = occupations
    correction = properties.isolated_system_correction
    if correction is not None:
        fields["isolated_system_correction"] = {
            "method": correction.method,
            "energies": {
                "total": correction.total.value,
                "corrected_total_energy": correction.corrected_total_energy.value,
                **{name: term.value for name, term in correction.terms.items()},
            },
        }
    return fields


def compare_results(
    reference,
    candidate,
    total_energy_tolerance,
    component_energy_tolerance,
    context,
):
    reference = comparable_fields(reference)
    candidate = comparable_fields(candidate)
    failures = []
    reference_energies = reference["energies"]
    candidate_energies = candidate["energies"]
    for name in sorted(reference_energies.keys() | candidate_energies.keys()):
        tolerance = total_energy_tolerance if name == "total_energy" else component_energy_tolerance
        if (name in reference_energies) != (name in candidate_energies):
            failures.append(f"{context}: presence of energy {name} differs")
        elif not math.isclose(
            float(reference_energies[name]),
            float(candidate_energies[name]),
            abs_tol=tolerance,
            rel_tol=0.0,
        ):
            failures.append(
                f"{context}: energy {name} differs by "
                f"{abs(float(reference_energies[name]) - float(candidate_energies[name])):.3e}"
                f" {ENERGY_UNITS}"
            )

    for field in EXACT_METADATA_FIELDS:
        if reference.get(field) != candidate.get(field):
            failures.append(f"{context}: metadata {field} differs")

    for field in (
        "kpoint_weights",
        "kpoint_coordinates",
        "eigenvalues",
        "occupations",
        "forces",
    ):
        if nested_shape(reference.get(field, [])) != nested_shape(candidate.get(field, [])):
            failures.append(f"{context}: shape of {field} differs")

    reference_weights = reference.get("kpoint_weights", [])
    candidate_weights = candidate.get("kpoint_weights", [])
    if len(reference_weights) == len(candidate_weights):
        for index, (reference_weight, candidate_weight) in enumerate(
            zip(reference_weights, candidate_weights, strict=True)
        ):
            if not math.isclose(
                float(reference_weight), float(candidate_weight), abs_tol=1.0e-12, rel_tol=0.0
            ):
                failures.append(f"{context}: kpoint_weights[{index}] differs")

    for field, tolerance in (
        ("kpoint_coordinates", 1.0e-12),
        ("eigenvalues", component_energy_tolerance),
        ("occupations", component_energy_tolerance),
    ):
        reference_values = reference.get(field, [])
        candidate_values = candidate.get(field, [])
        if nested_shape(reference_values) != nested_shape(candidate_values):
            continue
        for index, (reference_value, candidate_value) in enumerate(
            zip(
                flatten_numeric_values(reference_values),
                flatten_numeric_values(candidate_values),
                strict=True,
            )
        ):
            if not math.isclose(reference_value, candidate_value, abs_tol=tolerance, rel_tol=0.0):
                failures.append(
                    f"{context}: {field} value {index} differs by "
                    f"{abs(reference_value - candidate_value):.3e}"
                )

    reference_correction = reference.get("isolated_system_correction")
    candidate_correction = candidate.get("isolated_system_correction")
    if (reference_correction is None) != (candidate_correction is None):
        failures.append(f"{context}: isolated-system correction presence differs")
    elif reference_correction is not None:
        if reference_correction.get("method") != candidate_correction.get("method"):
            failures.append(f"{context}: isolated-system correction method differs")
        for name, reference_value in reference_correction["energies"].items():
            if not math.isclose(
                float(reference_value),
                float(candidate_correction["energies"][name]),
                abs_tol=total_energy_tolerance,
                rel_tol=0.0,
            ):
                failures.append(f"{context}: correction {name} differs")
    return failures


def validate_stress_refusal(case, topology, results):
    """Check that an unsupported stress path says so, rather than going quiet.

    Skipped for the capped benchmark decks. They stop before convergence on
    purpose, and the SCF flavor then skips forces and stress entirely --
    "skipping forces and stress - SCF did not converge" -- so there is no
    stress result to have been marked incomplete. Asserting the refusal
    machinery here would be asserting it on a code path that never ran.
    """
    if not case.requires_convergence:
        return []

    expected_reasons = []
    if case.identifier == "nickel_lsda_hubbard_ortho_atomic":
        expected_reasons.append("ortho-atomic projectors")
    execution_metadata = results["runtime_log"].get("execution", {})
    reciprocal_space_vectors_are_distributed = execution_metadata.get(
        "reciprocal_space_vectors_are_distributed"
    )
    if not isinstance(reciprocal_space_vectors_are_distributed, bool):
        return [f"{case.identifier}/{topology.identifier}: execution metadata is absent"]
    if reciprocal_space_vectors_are_distributed:
        expected_reasons.append("MPI band group")

    context = f"{case.identifier}/{topology.identifier}"
    incomplete = results["runtime_log"].get("incomplete", {})
    if not expected_reasons:
        if "stress" in incomplete:
            return [f"{context}: stress is unexpectedly marked incomplete"]
        return []

    failures = []
    if "stress" not in incomplete:
        failures.append(f"{context}: unsupported stress path is not marked incomplete")
    reason = incomplete.get("stress", "")
    for expected_reason in expected_reasons:
        if expected_reason not in reason:
            failures.append(f"{context}: stress refusal omits {expected_reason!r}")
    return failures


def validate_execution_topology(topology, results, context):
    execution_metadata = results["runtime_log"].get("execution", {})
    failures = []
    if execution_metadata.get("number_of_processes") != topology.number_of_processes:
        failures.append(f"{context}: actual process count differs from the launch topology")

    number_of_kpoint_pools = execution_metadata.get("number_of_kpoint_pools")
    band_group_size = execution_metadata.get("band_group_size")
    if not isinstance(number_of_kpoint_pools, int) or number_of_kpoint_pools < 1:
        failures.append(f"{context}: actual k-point pool count is invalid")
    if not isinstance(band_group_size, int) or band_group_size < 1:
        failures.append(f"{context}: actual band-group size is invalid")
    if (
        isinstance(number_of_kpoint_pools, int)
        and isinstance(band_group_size, int)
        and number_of_kpoint_pools * band_group_size != topology.number_of_processes
    ):
        failures.append(f"{context}: actual pool and band-group sizes do not cover all processes")
    if (
        topology.number_of_kpoint_pools is not None
        and isinstance(number_of_kpoint_pools, int)
        and number_of_kpoint_pools > topology.number_of_kpoint_pools
    ):
        failures.append(f"{context}: actual pool count exceeds the requested count")
    return failures


def write_summary(output_directory, metadata, records, failures):
    serialized_records = []
    for record in records:
        serialized = asdict(record)
        serialized_records.append(serialized)
    summary = {"metadata": metadata, "records": serialized_records, "failures": failures}
    with (output_directory / "summary.json").open("w", encoding="utf-8") as file_handle:
        json.dump(summary, file_handle, indent=2)
        file_handle.write("\n")


def main():
    arguments = parse_arguments()
    if arguments.repetitions < 1:
        raise ValueError("--repetitions must be positive")
    output_directory = create_output_directory(arguments.output_directory)
    logger = configure_logging(output_directory)
    serial_binary = resolve_executable(arguments.serial_binary, "serial binary")
    mpi_binary = resolve_executable(arguments.mpi_binary, "MPI binary")
    gpu_binary = (
        resolve_executable(arguments.gpu_binary, "GPU binary")
        if arguments.gpu_binary is not None
        else None
    )

    def launch(ranks, nodes=1):
        """Prefix that puts `ranks` processes on `nodes` nodes.

        mpirun is right on a workstation. Inside a Slurm allocation it is srun:
        it is the only launcher that reaches the allocated nodes at all (under
        `salloc <command>` the script body runs on the login node), and -N is
        what makes a multi-node topology mean anything rather than silently
        packing every rank onto one node.
        """
        if arguments.launcher == "srun":
            return ("srun", "-N", str(nodes), "-n", str(ranks))
        if nodes != 1:
            raise ValueError("multi-node topologies require --launcher srun")
        return ("mpirun", "-np", str(ranks))

    def mpi_topology(identifier, ranks, pools, nodes=1, distribute_bands=False):
        command = (
            *launch(ranks, nodes),
            str(mpi_binary),
            "--pools",
            str(pools),
            *(("--distribute-bands",) if distribute_bands else ()),
        )
        return ExecutionTopology(
            identifier,
            command,
            str(mpi_binary),
            ranks,
            pools,
            number_of_nodes=nodes,
            distributes_bands=distribute_bands,
        )

    available_topologies = [
        ExecutionTopology("serial", (str(serial_binary),), str(serial_binary), 1, None),
        # Every MPI topology goes through mpi_topology() so the launcher is a
        # single decision rather than a literal repeated in each entry. These
        # were hard-coded to mpirun, which meant --launcher srun silently
        # applied to the new topologies only and the old ones still tried to
        # exec mpirun -- absent on a Cray machine.
        mpi_topology("mpi_1_rank", 1, 1),
        mpi_topology("mpi_2_ranks", 2, 1),
        mpi_topology("mpi_2_ranks_2_pools", 2, 2),
        mpi_topology("mpi_4_ranks", 4, 1),
        mpi_topology("mpi_4_ranks_2_pools", 4, 2),
        mpi_topology("mpi_4_ranks_4_pools", 4, 4),
        # --distribute-bands is a different split from the default, not a
        # variation of it, so it gets its own topologies rather than a flag on
        # the existing ones. It is also the less-travelled path: the default
        # reciprocal-space split is what every measurement so far exercised.
        mpi_topology("mpi_2_ranks_bands", 2, 1, distribute_bands=True),
        mpi_topology("mpi_4_ranks_bands", 4, 1, distribute_bands=True),
        mpi_topology("mpi_4_ranks_2_pools_bands", 4, 2, distribute_bands=True),
        # Single-node counterparts to the multi-node topologies below. The
        # question a two-node run answers is what crossing the boundary costs,
        # and that is only answerable against the same rank count on one node.
        mpi_topology("mpi_8_ranks_8_pools", 8, 8),
        mpi_topology("mpi_16_ranks_16_pools", 16, 16),
        # Multi-node. srun only; mpirun raises rather than quietly packing
        # every rank onto one node and reporting it as a two-node result.
        mpi_topology("mpi_2_nodes_8_ranks", 8, 8, nodes=2),
        mpi_topology("mpi_2_nodes_16_ranks", 16, 16, nodes=2),
    ]
    if gpu_binary is not None:
        available_topologies.append(
            ExecutionTopology("gpu", (str(gpu_binary),), str(gpu_binary), 1, None)
        )

    # Benchmark decks run only when asked for by name. They take minutes to
    # hours, and the largest needs roughly 45 GB per rank, so sweeping them in
    # by default would turn a reproducibility check into an overnight job.
    selected_case_identifiers = set(
        arguments.selected_case_identifiers
        or [case.identifier for case in REPRODUCIBILITY_CASES if not case.is_benchmark_only]
    )
    selected_topology_identifiers = set(
        arguments.selected_topology_identifiers or DEFAULT_TOPOLOGY_IDENTIFIERS
    )
    if "gpu" in selected_topology_identifiers and gpu_binary is None:
        raise ValueError("--topology gpu requires --gpu-binary")
    cases = [case for case in REPRODUCIBILITY_CASES if case.identifier in selected_case_identifiers]
    topologies = [
        topology
        for topology in available_topologies
        if topology.identifier in selected_topology_identifiers
    ]

    metadata = {
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip(),
        "git_worktree_dirty": bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        ),
        "repetitions": arguments.repetitions,
        "convergence_threshold_override": arguments.convergence_threshold,
        "cases": [
            {
                "identifier": case.identifier,
                "manifest": case.manifest,
                "total_energy_tolerance_rydberg": case.total_energy_tolerance_rydberg,
                "component_energy_tolerance_rydberg": (case.component_energy_tolerance_rydberg),
            }
            for case in cases
        ],
        "topologies": [topology.identifier for topology in topologies],
        "binaries": {
            topology.identifier: {
                "path": topology.binary_path,
                "sha256": compute_binary_digest(Path(topology.binary_path)),
            }
            for topology in topologies
        },
    }
    logger.info("Commit: %s", metadata["git_commit"])
    logger.info("Worktree dirty: %s", metadata["git_worktree_dirty"])
    for identifier, binary_metadata in metadata["binaries"].items():
        logger.info("Binary %s: %s", identifier, binary_metadata["sha256"])

    environment = os.environ.copy()
    environment.update(THREAD_ENVIRONMENT)
    records = []
    failures = []
    total_runs = sum(
        arguments.repetitions
        * sum(
            topology.identifier != "gpu" or case.identifier == "silicon_norm_conserving"
            for topology in topologies
        )
        for case in cases
    )
    completed_runs = 0
    for case in cases:
        case_topologies = [
            topology
            for topology in topologies
            if topology.identifier != "gpu" or case.identifier == "silicon_norm_conserving"
        ]
        for topology in case_topologies:
            for repetition in range(1, arguments.repetitions + 1):
                record = run_one(
                    case,
                    topology,
                    repetition,
                    output_directory,
                    arguments.timeout_seconds,
                    environment,
                    arguments.convergence_threshold,
                )
                records.append(record)
                failures.extend(validate_stress_refusal(case, topology, record.results))
                failures.extend(
                    validate_execution_topology(
                        topology,
                        record.results,
                        f"{case.identifier}/{topology.identifier}",
                    )
                )
                completed_runs += 1
                logger.info(
                    "Run %d/%d: %s / %s / repetition %d, %.3f s, E=%.12f Ry",
                    completed_runs,
                    total_runs,
                    case.identifier,
                    topology.identifier,
                    repetition,
                    record.wall_seconds,
                    energies(record.results)["total_energy"],
                )

    for case in cases:
        case_records = [record for record in records if record.case == case.identifier]
        reference = next(
            (
                record
                for record in case_records
                if record.topology == "serial" and record.repetition == 1
            ),
            case_records[0],
        )
        for record in case_records:
            failures.extend(
                compare_results(
                    reference.results,
                    record.results,
                    case.total_energy_tolerance_rydberg,
                    case.component_energy_tolerance_rydberg,
                    f"{case.identifier}/{record.topology}/run_{record.repetition}",
                )
            )

    write_summary(output_directory, metadata, records, failures)
    if failures:
        for failure in failures:
            logger.error("FAIL: %s", failure)
        logger.error("Matrix failed with %d comparison errors", len(failures))
        return 1
    logger.info("Matrix passed: %d runs, all components and metadata invariant", len(records))
    logger.info("Summary: %s", output_directory / "summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
