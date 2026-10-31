#!/usr/bin/env python3
"""Run the committed benchmark decks and write their timings out as JSON.

    scripts/benchmark/run_benchmarks.py --q3-bin build/q3
    scripts/benchmark/run_benchmarks.py --q3-bin build/q3 --tag ci --repetitions 2
    scripts/benchmark/run_benchmarks.py --q3-bin build/q3 --deck silicon_norm_conserving

The decks live in ``tests/regression/benchmark_matrix.yml``; this script runs
each one the requested number of times and writes one JSON record per run.

Where the numbers come from
---------------------------

Nothing here measures anything the engine does not already measure. Every
per-phase figure is read from the ``timings_ms`` block that a run writes into
``<prefix>_results.json`` (assembled in
``src/layer_4_computation/flavors/self_consistent_field.hpp`` from the
profiling counters in ``src/layer_0_core/profiling/counters.hpp``). Reading the
file rather than scraping the console means a run that aborted before writing
one shows up as a missing block, not as a plausible-looking zero.

That block mixes two kinds of number, and this script separates them:
``fft_calls``, ``davidson_iterations`` and ``hamiltonian_applications`` are
counts of work done, everything else is milliseconds. The split matters when
comparing two runs -- a stage that got slower because it ran more often is a
different finding from one that got slower per call, and the times alone cannot
tell them apart.

Wall time is recorded too, as the outer bound: it includes process startup,
input parsing, pseudopotential reading, the property run and writing the
results file, none of which the solver's own total covers.

What is held fixed
------------------

Both knobs below are set by this script rather than left to the environment,
because a benchmark that measures a different configuration on each machine is
not a benchmark:

* One thread per process (``OMP_NUM_THREADS`` and the vendor equivalents),
  so a runner with more cores does not silently measure a different code path.
* ``control.use_deterministic_fft_planning: true``, which selects FFTW's
  fixed-heuristic planner over ``FFTW_MEASURE``. ``FFTW_MEASURE`` times
  candidate plans at run time, so it both costs a variable amount up front and
  can pick a different plan on a loaded machine. The deterministic planner
  gives up some FFT throughput to make repeated runs comparable.

Compare two of these files with ``scripts/benchmark/compare_benchmarks.py``.
"""

import argparse
import copy
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MATRIX_PATH = PROJECT_ROOT / "tests" / "regression" / "benchmark_matrix.yml"

# Keys inside the engine's timings_ms block that are counts of work, not
# milliseconds. Everything else in that block is a stage time.
COUNT_KEYS = ("fft_calls", "davidson_iterations", "hamiltonian_applications")

# One thread per process, matching
# scripts/validation/run_mpi_reproducibility_matrix.py.
# A benchmark that picks up however many threads the machine happens to allow
# measures the machine, not the change under review.
SINGLE_THREAD_ENVIRONMENT = {
    "BLIS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "OMP_DYNAMIC": "FALSE",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--q3-bin", required=True, type=Path, help="q3 binary to benchmark")
    parser.add_argument(
        "--matrix",
        type=Path,
        default=DEFAULT_MATRIX_PATH,
        help="benchmark deck manifest (default: tests/regression/benchmark_matrix.yml)",
    )
    parser.add_argument(
        "--deck",
        dest="selected_deck_identifiers",
        action="append",
        help="run only this deck; repeat the option to select more than one",
    )
    parser.add_argument(
        "--tag",
        dest="selected_tags",
        action="append",
        help="run only decks carrying this tag (e.g. 'ci'); repeatable",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        help="runs per deck (default: the manifest's own repetitions value)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="JSON file to write (default: <output-directory>/records.json)",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        help="working directory for the runs (default: agents/workdir/tmp/benchmarks/<stamp>)",
    )
    parser.add_argument(
        "--machine-label",
        help=(
            "name for the machine these numbers describe. Defaults to a label "
            "derived from the platform. Timings are only comparable within one "
            "label, so a CI runner should pass its runner class here."
        ),
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=1800,
        help="per-run timeout (default: 1800)",
    )
    return parser.parse_args()


def derive_machine_label():
    """A label for the machine, stable across runs on the same one.

    Deliberately coarse: it names a class of machine rather than an instance,
    because the question a baseline answers is "is this the same kind of box",
    and a hostname would make every developer's baseline unique and useless.
    """
    processor = platform.machine() or "unknown"
    return f"{platform.system().lower()}-{processor}-{os.cpu_count() or 0}-core"


def read_processor_model():
    """The CPU model string, where the platform offers one.

    Provenance only. Two runners of the same class can still hold different
    silicon generations, and that difference is worth seeing in the record
    rather than discovering as an unexplained 30% shift.
    """
    cpuinfo_path = Path("/proc/cpuinfo")
    if cpuinfo_path.is_file():
        for line in cpuinfo_path.read_text(errors="replace").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def describe_machine(label):
    return {
        "label": label,
        "system": platform.system(),
        "release": platform.release(),
        "architecture": platform.machine(),
        "processor": read_processor_model(),
        "logical_cores": os.cpu_count() or 0,
    }


def run_git_command(*arguments):
    completed = subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), *arguments],
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else ""


def read_git_commit():
    """The checkout's commit, and whether it has uncommitted changes.

    A timing that cannot say what produced it is not evidence, so both halves
    are recorded: "slower than some earlier build" is not a finding.
    """
    return {
        "git_commit": run_git_command("rev-parse", "HEAD"),
        "git_worktree_dirty": bool(run_git_command("status", "--porcelain")),
    }


def compute_file_digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for block in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def describe_repository_path(path):
    """A path as the repository sees it, never as this machine does.

    Committed records must not carry a machine-specific absolute path (AGENTS.md
    § Testing), and a home directory in an exported file fails the public-export
    gate. Anything outside the checkout is reduced to its file name.
    """
    try:
        return str(Path(path).resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return Path(path).name


def read_binary_version(binary_path):
    """``q3 --version``: what the binary says it is.

    Q3 bakes the commit in at build time and reports it as unknown for a build
    from a modified tree, so this answers a question the checkout's own commit
    does not: whether the binary being timed corresponds to a commit at all.
    Recorded next to the binary's SHA-256 because a timing whose binary cannot
    be identified is a number, not evidence. A binary too old to know the flag
    reports an empty string rather than failing the run.
    """
    completed = subprocess.run(
        [str(binary_path), "--version"], capture_output=True, text=True, check=False
    )
    return completed.stdout.strip() if completed.returncode == 0 else ""


def resolve_executable(path):
    resolved = path.expanduser().resolve()
    if not resolved.is_file() or not os.access(resolved, os.X_OK):
        raise ValueError(f"q3 binary is not executable: {resolved}")
    return resolved


def load_matrix(matrix_path):
    """Read the deck manifest, resolving each deck's path relative to it.

    Deck paths are stored relative to the manifest, per AGENTS.md § Testing, so
    the matrix works from a git worktree and carries no machine-specific path.
    """
    with matrix_path.open(encoding="utf-8") as file_handle:
        matrix = yaml.safe_load(file_handle)
    decks = matrix.get("decks") or []
    if not decks:
        raise ValueError(f"No decks listed in {matrix_path}")
    for deck in decks:
        for required_key in ("identifier", "input"):
            if required_key not in deck:
                raise ValueError(f"Deck in {matrix_path} is missing '{required_key}': {deck}")
        deck["input_path"] = (matrix_path.parent / deck["input"]).resolve()
        if not deck["input_path"].is_file():
            raise ValueError(f"Deck '{deck['identifier']}' points at a missing input file")
    return matrix, decks


def select_decks(decks, selected_deck_identifiers, selected_tags):
    selected = decks
    if selected_deck_identifiers:
        known = {deck["identifier"] for deck in decks}
        unknown = sorted(set(selected_deck_identifiers) - known)
        if unknown:
            raise ValueError(f"Unknown deck(s): {', '.join(unknown)}")
        selected = [deck for deck in selected if deck["identifier"] in selected_deck_identifiers]
    if selected_tags:
        wanted = set(selected_tags)
        selected = [deck for deck in selected if wanted & set(deck.get("tags") or [])]
    if not selected:
        raise ValueError("Deck selection matched nothing")
    return selected


def deep_merge(base, overlay):
    merged = copy.deepcopy(base)
    for key, value in overlay.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def prepare_input(deck, run_directory):
    """Write the deck into the run directory with its paths and outputs fixed.

    The example decks name their material and pseudopotentials relative to
    themselves, so a copy placed anywhere else has to resolve them first.
    """
    source_path = deck["input_path"]
    with source_path.open(encoding="utf-8") as file_handle:
        manifest = deep_merge(yaml.safe_load(file_handle), deck.get("overlay") or {})

    material_path = Path(manifest["material"])
    if not material_path.is_absolute():
        manifest["material"] = str((source_path.parent / material_path).resolve())
    pseudopotentials = manifest["method"]["pseudopotentials"]
    for element, pseudopotential in pseudopotentials.items():
        pseudopotential_path = Path(pseudopotential)
        if not pseudopotential_path.is_absolute():
            pseudopotentials[element] = str((source_path.parent / pseudopotential_path).resolve())

    control = manifest.setdefault("control", {})
    control["outdir"] = str(run_directory / "output")
    control["prefix"] = deck["identifier"]
    control["use_deterministic_fft_planning"] = True

    prepared_path = run_directory / "input.yml"
    with prepared_path.open("w", encoding="utf-8") as file_handle:
        yaml.safe_dump(manifest, file_handle, sort_keys=False)
    return prepared_path


def total_energy_ry(results: dict) -> float | None:
    """The value of the results' ESSE ``total_energy`` record, if it is in Ry.

    Read from the JSON rather than through ``q3.io.results``: this script runs
    without the package installed.
    """
    for record in results.get("properties", {}).get("records", []):
        if record.get("name") == "total_energy" and record.get("units") == "Ry":
            return record["value"]
    return None


def split_timings(timings):
    """Separate the stage milliseconds from the work counts."""
    stage_times = {key: float(value) for key, value in timings.items() if key not in COUNT_KEYS}
    counts = {key: int(timings[key]) for key in COUNT_KEYS if key in timings}
    return stage_times, counts


def run_once(binary_path, deck, repetition_index, run_directory, timeout_seconds, environment):
    run_directory.mkdir(parents=True)
    prepared_path = prepare_input(deck, run_directory)
    console_path = run_directory / "console.log"

    started = time.perf_counter()
    with console_path.open("w", encoding="utf-8") as console:
        completed = subprocess.run(
            [str(binary_path), str(prepared_path)],
            cwd=run_directory,
            env=environment,
            stdout=console,
            stderr=subprocess.STDOUT,
            timeout=timeout_seconds,
            check=False,
        )
    wall_seconds = time.perf_counter() - started

    if completed.returncode != 0:
        console_tail = console_path.read_text(errors="replace")[-8000:]
        raise RuntimeError(
            f"{binary_path} exited with code {completed.returncode} on deck "
            f"'{deck['identifier']}'\n{console_tail}"
        )

    results_path = run_directory / "output" / f"{deck['identifier']}_results.json"
    with results_path.open(encoding="utf-8") as file_handle:
        results = json.load(file_handle)

    runtime_log = results.get("runtime_log", {})
    timings = runtime_log.get("timings_ms")
    if not timings:
        raise RuntimeError(
            f"Deck '{deck['identifier']}' wrote no timings_ms block. The binary predates the "
            "in-results timing counters, or the run stopped before the flavor finished."
        )
    stage_times, counts = split_timings(timings)

    return {
        "deck": deck["identifier"],
        "repetition": repetition_index,
        "wall_seconds": wall_seconds,
        "timings_ms": stage_times,
        "counts": counts,
        "converged": bool(runtime_log.get("converged")),
        "self_consistent_field_iterations": runtime_log.get("self_consistent_field_iterations"),
        "total_energy_ry": total_energy_ry(results),
        "peak_resident_memory_bytes": (runtime_log.get("execution") or {}).get(
            "peak_resident_memory_bytes"
        ),
    }


def create_output_directory(requested_directory):
    if requested_directory is None:
        stamp = datetime.now().strftime("%Y-%m-%dT%H%M%S")
        requested_directory = PROJECT_ROOT / "agents" / "workdir" / "tmp" / "benchmarks" / stamp
    output_directory = requested_directory.expanduser().resolve()
    if output_directory.exists() and any(output_directory.iterdir()):
        raise ValueError(f"Output directory is not empty: {output_directory}")
    output_directory.mkdir(parents=True, exist_ok=True)
    return output_directory


def report(records):
    print(f"\n{'deck':<34} {'rep':>3} {'solver ms':>11} {'wall s':>8} {'converged':>10}")
    for record in records:
        print(
            f"{record['deck']:<34} {record['repetition']:>3} "
            f"{record['timings_ms'].get('total', 0.0):>11.1f} "
            f"{record['wall_seconds']:>8.2f} {str(record['converged']):>10}"
        )


def main():
    arguments = parse_arguments()
    binary_path = resolve_executable(arguments.q3_bin)
    matrix, decks = load_matrix(arguments.matrix.resolve())
    selected_decks = select_decks(
        decks, arguments.selected_deck_identifiers, arguments.selected_tags
    )
    repetitions = arguments.repetitions or int(matrix.get("repetitions", 3))
    if repetitions < 1:
        raise ValueError("--repetitions must be at least 1")

    output_directory = create_output_directory(arguments.output_directory)
    output_path = arguments.output or (output_directory / "records.json")
    machine_label = arguments.machine_label or derive_machine_label()
    commit = read_git_commit()

    environment = {**os.environ, **SINGLE_THREAD_ENVIRONMENT}

    records = []
    for deck in selected_decks:
        for repetition_index in range(1, repetitions + 1):
            run_directory = output_directory / deck["identifier"] / f"repetition_{repetition_index}"
            print(f"  {deck['identifier']} repetition {repetition_index}/{repetitions} ...")
            record = run_once(
                binary_path,
                deck,
                repetition_index,
                run_directory,
                arguments.timeout_seconds,
                environment,
            )
            # Each record carries its own machine and commit so that records
            # from several machines can be concatenated without losing which
            # box produced which number.
            record["machine"] = machine_label
            record["commit"] = commit["git_commit"]
            records.append(record)

    payload = {
        "schema": "q3-benchmark-records-1",
        "metadata": {
            "recorded_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            **commit,
            "machine": describe_machine(machine_label),
            "binary": {
                "name": describe_repository_path(binary_path),
                "sha256": compute_file_digest(binary_path),
                "version": read_binary_version(binary_path),
            },
            "matrix": describe_repository_path(arguments.matrix),
            "repetitions": repetitions,
            "single_threaded": True,
            "deterministic_fft_planning": True,
        },
        "records": records,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file_handle:
        json.dump(payload, file_handle, indent=4, sort_keys=True)
        file_handle.write("\n")

    report(records)
    print(f"\nWrote {output_path}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
        print(str(error), file=sys.stderr)
        sys.exit(1)
