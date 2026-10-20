#!/usr/bin/env python3
"""
Regenerate the committed `reference.json` for the Wannier interface fixtures.

Runs the reference chain -- `pw.x` scf, `pw.x` nscf on the full unreduced
grid, `wannier90.x -pp`, `pw2wannier90.x` -- in a scratch directory, then
reduces its output to the handful of numbers the parity gates actually
compare and writes them beside the inputs.

Why a JSON summary rather than the files themselves: the chain's own outputs
run to hundreds of kilobytes per fixture and are regenerable, so committing
them would bloat the repository for no gain -- the same reasoning that makes
`*.out` gitignored and `reference.json` committed for the plain QE cases (see
refresh_qe_references.py).

**Only gauge-invariant quantities are recorded.** Raw M and A elements depend
on each code's arbitrary per-k phase choice, so two correct codes disagree on
them elementwise; recording them would invite a comparison that fails for no
physical reason. What is stored instead survives any per-k unitary: the
overlap-squared sums, Omega_I, the b-vector shell and its weights, and the
eigenvalues.

    .venv/bin/python tests/validation/refresh_wannier_references.py

Per AGENTS.md the reference binaries are the LibXC build under
`reference/dft/q-e/build_libxc/bin/`; this script never builds anything.
Each fixture runs in its own scratch directory because `pw.x` copies stdin to
a fixed name in its working directory -- two runs sharing a cwd silently swap
inputs.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "tests"))

from helpers_wannier import (  # noqa: E402
    compute_b_vectors,
    gauge_invariant_spread,
    read_eigenvalue_file,
    read_neighbor_list_file,
    read_overlap_file,
    read_projection_file,
    solve_finite_difference_weights,
)

from helpers import QE_BIN_DIR  # noqa: E402

# The shared QE reference build's `bin/`, by tests/helpers.py's one rule:
# $Q3_QE_BIN_DIR if set, else reference/dft/q-e/build_libxc/bin under any
# ancestor of the repository (which is what makes it work from a git worktree);
# absent, the conventional location, so a "binary not found" message names a
# plausible path.
REFERENCE_BINARY_DIRECTORY = QE_BIN_DIR

FIXTURES = {
    "band_structure--semiconductor--silicon--wannier": {
        "seedname": "si",
        "pseudopotentials": ["nc/Si.upf"],
        "description": (
            "Si diamond, norm-conserving, 48/192 Ry, 4x4x4 Gamma-centred development "
            "grid. The 4 occupied bands form an isolated manifold, so Omega_I from raw "
            "M is a clean cross-code comparison with no disentanglement involved."
        ),
    }
}

NOTE = (
    "Gauge-invariant summary of the QE + Wannier90 reference chain for the inputs "
    "beside this file. The chain's own outputs (.mmn/.amn/.eig/.nnkp/.out) are not "
    "committed; regenerate with tests/validation/refresh_wannier_references.py when "
    "an input or the reference build changes. Only quantities invariant under a "
    "per-k unitary are recorded -- raw M and A elements are gauge-dependent and must "
    "never be compared across codes."
)


def _run(binary: str, arguments: list[str], working_directory: Path, log_name: str) -> None:
    executable = REFERENCE_BINARY_DIRECTORY / binary
    if not executable.is_file():
        raise SystemExit(
            f"reference binary not found: {executable}\n"
            f"Build it as an incremental target in the existing QE build tree; "
            f"never rebuild QE from scratch (AGENTS.md)."
        )
    log_path = working_directory / log_name
    with log_path.open("w") as log:
        result = subprocess.run(
            [str(executable), *arguments],
            cwd=working_directory,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode != 0:
        tail = "\n".join(log_path.read_text().splitlines()[-20:])
        raise SystemExit(f"{binary} failed (exit {result.returncode}); last lines:\n{tail}")


def run_chain(reference_directory: Path, seedname: str, pseudopotentials: list[str]) -> Path:
    """Run the four-step chain in a fresh scratch directory; return that directory."""
    scratch = Path(tempfile.mkdtemp(prefix=f"wannier_reference_{seedname}_"))
    for name in ("scf.in", "nscf.in", f"{seedname}.win", f"{seedname}.pw2wan"):
        shutil.copy(reference_directory / name, scratch / name)
    for relative in pseudopotentials:
        source = REPOSITORY_ROOT / "examples" / "data" / "pseudopotentials" / relative
        shutil.copy(source, scratch / Path(relative).name)

    print(f"  scf            ({scratch})")
    _run("pw.x", ["-i", "scf.in"], scratch, "scf.out")
    print("  nscf (full grid, nosym)")
    _run("pw.x", ["-i", "nscf.in"], scratch, "nscf.out")
    print("  wannier90 -pp")
    _run("wannier90.x", ["-pp", seedname], scratch, "w90pp.out")
    print("  pw2wannier90")
    _run("pw2wannier90.x", ["-i", f"{seedname}.pw2wan"], scratch, "pw2wan.out")
    return scratch


def summarize(scratch: Path, seedname: str, description: str) -> dict:
    """Reduce the chain's output to the gauge-invariant numbers the gates use."""
    overlaps = read_overlap_file(scratch / f"{seedname}.mmn")
    projections = read_projection_file(scratch / f"{seedname}.amn")
    eigenvalues = read_eigenvalue_file(scratch / f"{seedname}.eig")
    neighbors = read_neighbor_list_file(scratch / f"{seedname}.nnkp")

    if not np.array_equal(overlaps.neighbor_kpoint, neighbors.neighbor_kpoint):
        raise SystemExit(
            "the .mmn and .nnkp disagree on the neighbour table; the chain is "
            "internally inconsistent and its output must not be archived"
        )
    if not np.array_equal(overlaps.periodic_image_shift, neighbors.periodic_image_shift):
        raise SystemExit("the .mmn and .nnkp disagree on the periodic image shifts")

    b_vectors = compute_b_vectors(neighbors)
    weights = solve_finite_difference_weights(b_vectors)
    omega_invariant = gauge_invariant_spread(
        overlaps.overlaps, weights, projections.number_of_wannier_functions
    )
    overlap_squared_sum = (np.abs(overlaps.overlaps) ** 2).sum(axis=(2, 3))
    singular_values = np.linalg.svd(
        overlaps.overlaps.reshape(-1, overlaps.number_of_bands, overlaps.number_of_bands),
        compute_uv=False,
    )

    return {
        "note": NOTE,
        "description": description,
        "chain": [
            "pw.x -i scf.in",
            "pw.x -i nscf.in",
            f"wannier90.x -pp {seedname}",
            f"pw2wannier90.x -i {seedname}.pw2wan",
        ],
        "dimensions": {
            "number_of_bands": overlaps.number_of_bands,
            "number_of_kpoints": overlaps.number_of_kpoints,
            "number_of_neighbors": overlaps.number_of_neighbors,
            "number_of_wannier_functions": projections.number_of_wannier_functions,
        },
        "b_vector_shell": {
            "magnitude_inverse_angstrom": float(np.linalg.norm(b_vectors[0], axis=1)[0]),
            "weights_angstrom_squared": [float(value) for value in weights],
            "completeness_residual": float(
                np.abs(
                    np.einsum("b,bi,bj->ij", weights, b_vectors[0], b_vectors[0]) - np.eye(3)
                ).max()
            ),
        },
        "gauge_invariant_spread_angstrom_squared": float(omega_invariant),
        "maximum_singular_value_of_overlap": float(singular_values.max()),
        "overlap_squared_sum_by_kpoint_and_neighbor": [
            [float(value) for value in row] for row in overlap_squared_sum
        ],
        "eigenvalues_electron_volt": [
            [float(value) for value in row] for row in eigenvalues.eigenvalues
        ],
    }


def main() -> int:
    for fixture_name, specification in FIXTURES.items():
        reference_directory = REPOSITORY_ROOT / "examples" / fixture_name / "reference"
        if not reference_directory.is_dir():
            print(f"skipping {fixture_name}: no reference directory", file=sys.stderr)
            continue
        print(f"{fixture_name}:")
        scratch = run_chain(
            reference_directory, specification["seedname"], specification["pseudopotentials"]
        )
        summary = summarize(scratch, specification["seedname"], specification["description"])
        output_path = reference_directory / "reference.json"
        output_path.write_text(json.dumps(summary, indent=2) + "\n")
        print(
            f"  wrote {output_path.relative_to(REPOSITORY_ROOT)} "
            f"(Omega_I = {summary['gauge_invariant_spread_angstrom_squared']:.6f} Ang^2, "
            f"max singular value = {summary['maximum_singular_value_of_overlap']:.6f})"
        )
        print(f"  scratch kept for inspection: {scratch}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
