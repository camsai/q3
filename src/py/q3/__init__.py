"""Q3 — Python interface to the Q3 plane-wave DFT code.

Top-level package for the Python layer.  Provides:

    * Flavor wrappers (SCF, NSCF) that invoke the ``q3`` binary via subprocess.
    * Workflow helpers (band structure, DOS, relax, vc-relax).
    * An ASE Calculator (``Q3ASECalculator``).
    * I/O utilities for reading/writing Q3 YAML inputs and ESSE-format results.

Example::

    from q3.flavors import run_scf
    from q3.io.input_writer import write_input

    result = run_scf("path/to/input.yml", q3_binary="./build/q3")
    print(result["total_energy_ry"])
"""

from q3.errors import (
    Q3Error,
    Q3InputError,
    Q3RuntimeError,
    Q3VersionMismatchError,
    SCFNotConvergedError,
)

__all__ = [
    "Q3Error",
    "Q3InputError",
    "Q3RuntimeError",
    "Q3VersionMismatchError",
    "SCFNotConvergedError",
]
