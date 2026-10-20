#!/usr/bin/env python3
"""
Regenerate the committed `reference.json` beside every QE input in `examples/`.

`*.out` is gitignored, so QE's own output files exist only on the machine that
ran them -- which means a test reading them directly passes locally and fails on
a fresh clone with a missing-file error. These JSON files are the committed
record of what QE printed; the `.out` stays as provenance for whoever has it.

Run this after changing a QE input, adding a case, or rebuilding the reference
binary. It reads the `.out` files that are present and leaves the rest alone, so
running it with a partial set will not erase entries -- but it will only refresh
what it can see, so regenerate on a machine that has run the full set.

    .venv/bin/python tests/validation/refresh_qe_references.py

Per AGENTS.md, QE reference runs must come from the LibXC build at
`reference/dft/q-e/build_libxc/bin/pw.x` so the comparison is apples to apples.
"""

import json
import re
import sys
from pathlib import Path

EXAMPLES_DIRECTORY = Path(__file__).resolve().parents[2] / "examples"

QUANTITIES = (
    ("total_energy_rydberg", r"^!\s+total energy\s+=\s+(-?\d+\.\d+)"),
    ("total_magnetization_bohr_magneton", r"^\s+total magnetization\s+=\s+(-?\d+\.\d+)"),
    ("absolute_magnetization_bohr_magneton", r"^\s+absolute magnetization\s+=\s+(-?\d+\.\d+)"),
    ("hubbard_energy_rydberg", r"^\s+Hubbard energy\s+=\s+(-?\d+\.\d+)"),
    (
        "total_pressure_kbar",
        r"^\s+total\s+stress\s+\(Ry/bohr\*\*3\)\s+\(kbar\)\s+P=\s*(-?\d+\.\d+)",
    ),
    ("hubbard_stress_kbar", r"^\s+hubbard stress \(kbar\)\s+(-?\d+\.\d+)"),
)
"""
What to lift out of a QE run.

The last two need `tstress = .true.`; `hubbard stress` additionally needs
`verbosity = 'high'`, because QE prints the per-term stress decomposition only
when `iverbosity > 0` -- without it there is a total and nothing else. Both are
recorded as the xx component alone: every deck carrying them is cubic, so the
tensor is isotropic and the off-diagonals are zero to the two decimals QE prints.
"""

NOTE = (
    "Numbers Quantum ESPRESSO printed for the input files beside this one. "
    "The .out files themselves are gitignored, so these are what the test suite "
    "compares against; regenerate with tests/validation/refresh_qe_references.py "
    "when an input or the reference build changes."
)


def parse_output(output_path):
    """The converged quantities QE printed, or None if it did not converge."""
    text = output_path.read_text()
    record = {}
    for name, pattern in QUANTITIES:
        matches = re.findall(pattern, text, re.M)
        if matches:
            record[name] = float(matches[-1])
    if "total_energy_rydberg" not in record:
        return None
    version = re.search(r"Program PWSCF v\.([\d.]+)", text)
    if version:
        record["quantum_espresso_version"] = version.group(1)
    return record


def main():
    by_directory = {}
    for output_path in sorted(EXAMPLES_DIRECTORY.glob("*/reference/*/*.out")):
        input_path = output_path.with_suffix(".in")
        if not input_path.exists():
            continue
        record = parse_output(output_path)
        if record is None:
            print(f"skipping {output_path}: no converged total energy", file=sys.stderr)
            continue
        by_directory.setdefault(output_path.parent, {})[input_path.name] = record

    if not by_directory:
        print("no QE .out files found -- nothing to refresh", file=sys.stderr)
        return 1

    for directory, records in sorted(by_directory.items()):
        existing = {}
        reference_path = directory / "reference.json"
        if reference_path.exists():
            existing = json.loads(reference_path.read_text()).get("inputs", {})
        merged = {**existing, **records}
        payload = {
            "note": NOTE,
            "binary": "reference/dft/q-e/build_libxc/bin/pw.x",
            "libxc_version": "7.0.0",
            "inputs": merged,
        }
        reference_path.write_text(json.dumps(payload, indent=4, sort_keys=True) + "\n")
        print(f"{reference_path.relative_to(EXAMPLES_DIRECTORY.parent)}: {len(merged)} input(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
