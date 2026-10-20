#!/usr/bin/env python3
"""Run every case in the A/B parity matrix with one binary, into one tree.

This is the execution half of the parity gate; ``compare_results_json.py`` is
the comparison half. The gate runs this script twice in the same job -- once
with the binary built from the pull request's merge base, once with the binary
built from its head -- and then compares the two trees.

Both runs read the **head** checkout's inputs. The matrix lives there, so it is
the only checkout guaranteed to contain every case it names, and using one
input tree for both sides means the only difference between the runs is the
binary itself. That is what the comparison is trying to isolate.

Layout
------

``--out DIR`` receives one subdirectory per case, holding nothing but what the
engine wrote::

    DIR / silicon - norm - conserving / output / si_nc_results.json
    DIR / silicon - norm - conserving / output / si_nc.save / ...

The merged input and the captured stdout go to ``--log-dir`` instead, which
defaults to a sibling of ``--out``. They are deliberately kept out of the
compared tree: the merged input names absolute paths into the checkout and the
log carries wall-clock timings, so neither can be compared byte for byte, and a
comparator that had to know which files to skip would be a comparator with an
allowlist.

Each case runs with its own directory as the working directory and an
``outdir`` of ``./output``, so any path the engine records in a checkpoint is
relative and identical on both sides.

Usage::

    python scripts/validation/run_parity_matrix.py --q3-bin ./build/q3 --out base_run
    python scripts/validation/run_parity_matrix.py --q3-bin ./build/q3 --out head_run --case silicon-ultrasoft
"""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
import time
from pathlib import Path

import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MATRIX_PATH = REPOSITORY_ROOT / "tests" / "regression" / "parity_matrix.yml"
DEFAULT_TIMEOUT_SECONDS = 1800


class MatrixError(RuntimeError):
    """The matrix could not be read, or a case in it is malformed."""


def deep_merge(base: dict, overlay: dict) -> dict:
    """Deep-merge ``overlay`` into ``base``, returning a new dict."""
    merged = copy.deepcopy(base)
    for key, value in overlay.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def resolve_relative_paths(config: dict, base_directory: Path) -> dict:
    """Rewrite an input's material and pseudopotential paths as absolute ones.

    The merged input is written somewhere other than the example directory it
    came from, so paths relative to that directory would no longer resolve.
    Same rule as ``tests/helpers.py``.
    """
    if isinstance(config.get("material"), str):
        material_path = Path(config["material"])
        if not material_path.is_absolute():
            config["material"] = str((base_directory / material_path).resolve())

    pseudopotentials = config.get("method", {}).get("pseudopotentials", {})
    if isinstance(pseudopotentials, dict):
        for element, pseudopotential_path in pseudopotentials.items():
            path = Path(pseudopotential_path)
            if not path.is_absolute():
                pseudopotentials[element] = str((base_directory / path).resolve())
    return config


def load_matrix(matrix_path: Path) -> tuple[dict, list[dict]]:
    """Read the matrix file, returning its defaults and its case list."""
    try:
        document = yaml.safe_load(matrix_path.read_text(encoding="utf-8"))
    except OSError as error:
        raise MatrixError(f"cannot read {matrix_path}: {error}") from error
    if not isinstance(document, dict) or "cases" not in document:
        raise MatrixError(f"{matrix_path} has no 'cases' list")

    defaults = document.get("defaults") or {}
    cases = document["cases"]
    identifiers = [case.get("id") for case in cases]
    if None in identifiers:
        raise MatrixError(f"{matrix_path} has a case without an 'id'")
    duplicates = {name for name in identifiers if identifiers.count(name) > 1}
    if duplicates:
        raise MatrixError(f"{matrix_path} repeats case id(s): {', '.join(sorted(duplicates))}")
    return defaults, cases


def write_merged_input(
    source_path: Path, overrides: dict, destination: Path, output_directory_name: str
) -> Path:
    """Write one case's input: the example, plus the case's and the matrix's overrides."""
    config = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    if overrides:
        config = deep_merge(config, overrides)
    resolve_relative_paths(config, source_path.parent)
    config.setdefault("control", {})["outdir"] = f"./{output_directory_name}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(yaml.dump(config, default_flow_style=False), encoding="utf-8")
    return destination


def run_one_input(
    q3_binary: Path, input_path: Path, working_directory: Path, timeout_seconds: int
) -> subprocess.CompletedProcess:
    """Run the engine once, from ``working_directory``."""
    return subprocess.run(
        [str(q3_binary), str(input_path)],
        cwd=str(working_directory),
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        check=False,
    )


def run_case(case: dict, defaults: dict, q3_binary: Path, out_root: Path, log_root: Path) -> dict:
    """Run one matrix case (its prerequisite first, when it has one)."""
    identifier = case["id"]
    overrides = deep_merge(defaults.get("overrides") or {}, case.get("overrides") or {})
    timeout_seconds = int(
        case.get("timeout_seconds", defaults.get("timeout_seconds", DEFAULT_TIMEOUT_SECONDS))
    )

    case_directory = out_root / identifier
    case_directory.mkdir(parents=True, exist_ok=True)
    log_directory = log_root / identifier
    log_directory.mkdir(parents=True, exist_ok=True)

    inputs = []
    if case.get("prerequisite"):
        # A prerequisite gets the case's overrides plus its own. The band-path
        # examples need the second kind: their SCF file asks for
        # `calculation: band_structure`, which is the Python layer's
        # multi-step workflow and which the engine refuses outright, so the
        # prerequisite has to be redirected to a plain single-point SCF while
        # the NSCF file it feeds stays exactly as it ships.
        prerequisite_overrides = deep_merge(overrides, case.get("prerequisite_overrides") or {})
        inputs.append(
            ("prerequisite", REPOSITORY_ROOT / case["prerequisite"], prerequisite_overrides)
        )
    inputs.append(("input", REPOSITORY_ROOT / case["input"], overrides))

    started_at = time.monotonic()
    for role, source_path, role_overrides in inputs:
        if not source_path.exists():
            raise MatrixError(f"case '{identifier}': {role} {source_path} does not exist")
        merged_path = write_merged_input(
            source_path, role_overrides, log_directory / f"{role}-{source_path.name}", "output"
        )
        completed = run_one_input(q3_binary, merged_path, case_directory, timeout_seconds)
        (log_directory / f"{role}-{source_path.stem}.log").write_text(
            completed.stdout + completed.stderr, encoding="utf-8"
        )
        if completed.returncode != 0:
            return {
                "id": identifier,
                "status": "failed",
                "role": role,
                "returncode": completed.returncode,
                "seconds": round(time.monotonic() - started_at, 1),
            }

    return {
        "id": identifier,
        "status": "ran",
        "seconds": round(time.monotonic() - started_at, 1),
    }


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--q3-bin", type=Path, required=True, help="the q3 binary to run")
    parser.add_argument("--out", type=Path, required=True, help="where the run artifacts go")
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="where merged inputs and stdout go (default: a '-logs' sibling of --out)",
    )
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX_PATH)
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        dest="cases",
        help="run only this case id; repeatable",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_argument_parser().parse_args(argv)

    q3_binary = arguments.q3_bin.resolve()
    if not q3_binary.exists():
        print(f"No such binary: {q3_binary}", file=sys.stderr)
        return 2

    out_root = arguments.out.resolve()
    log_root = (
        arguments.log_dir.resolve()
        if arguments.log_dir is not None
        else out_root.parent / f"{out_root.name}-logs"
    )

    try:
        defaults, cases = load_matrix(arguments.matrix)
    except MatrixError as error:
        print(f"Parity matrix: {error}", file=sys.stderr)
        return 2

    if arguments.cases:
        wanted = set(arguments.cases)
        unknown = wanted - {case["id"] for case in cases}
        if unknown:
            print(f"No such case id(s): {', '.join(sorted(unknown))}", file=sys.stderr)
            return 2
        cases = [case for case in cases if case["id"] in wanted]

    results = []
    for case in cases:
        print(f"[parity] {case['id']} ...", flush=True)
        try:
            outcome = run_case(case, defaults, q3_binary, out_root, log_root)
        except MatrixError as error:
            print(f"Parity matrix: {error}", file=sys.stderr)
            return 2
        except subprocess.TimeoutExpired:
            outcome = {"id": case["id"], "status": "timed-out"}
        results.append(outcome)
        print(f"[parity] {case['id']}: {outcome['status']}", flush=True)

    out_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)
    (log_root / "manifest.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    failures = [outcome for outcome in results if outcome["status"] != "ran"]
    if failures:
        print(
            f"{len(failures)} of {len(results)} parity cases did not run: "
            f"{', '.join(outcome['id'] for outcome in failures)}",
            file=sys.stderr,
        )
        return 1
    print(f"All {len(results)} parity cases ran.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
