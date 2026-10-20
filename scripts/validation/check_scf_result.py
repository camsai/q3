"""Assert that a container SCF converged *and* landed on the right number.

The container smoke tests used to grep for ``"converged": true`` and stop
there. Convergence is the weaker half of the claim: a container built against
the wrong LibXC, a different FFTW precision, or a mis-set cutoff converges
happily onto a different ground state, and the grep passes. The energy is what
says the image computes what the native build computes.

Usage::

    python3 scripts/validation/check_scf_result.py <results.json> [--expected RY] [--tolerance RY]

The default expectation is the silicon norm-conserving example, whose value is
also pinned in tests/validation/test_examples.py and recorded in that example's
reference/ directory.
"""

import argparse
import json
import sys
from pathlib import Path

# tests/validation/test_examples.py EXPECTED_DATA["nc"]["energy"]. Kept as a
# literal rather than imported: this runs on a bare runner with the repository
# checked out but nothing installed, and reaching into the test package would
# drag pytest and the package's dependencies in behind it.
DEFAULT_EXPECTED_ENERGY_RY = -16.91341791

# Wide enough that a rebuild of the same code on a different architecture
# passes -- amd64 and arm64 agree to about 2e-8 Ry -- and far tighter than any
# real defect. Swapping the exchange-correlation functional or the cutoff moves
# this number by milli-Rydberg or more.
DEFAULT_TOLERANCE_RY = 1e-6


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, help="a run's results JSON file")
    parser.add_argument("--expected", type=float, default=DEFAULT_EXPECTED_ENERGY_RY)
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE_RY)
    arguments = parser.parse_args()

    if not arguments.results.is_file():
        print(f"FAIL {arguments.results}: no results file was written")
        return 1

    data = json.loads(arguments.results.read_text(encoding="utf-8"))

    if data.get("converged") is not True:
        print(f"FAIL {arguments.results}: SCF did not converge")
        return 1

    energy = data.get("total_energy_ry")
    if energy is None:
        print(f"FAIL {arguments.results}: no total_energy_ry in the results")
        return 1

    difference = abs(energy - arguments.expected)
    if difference > arguments.tolerance:
        print(
            f"FAIL {arguments.results}: total energy {energy!r} Ry differs from "
            f"{arguments.expected!r} by {difference:.3e} Ry, over the "
            f"{arguments.tolerance:.0e} tolerance"
        )
        return 1

    print(
        f"OK {arguments.results}: converged, total energy {energy} Ry "
        f"(within {difference:.1e} of the reference)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
