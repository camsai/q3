"""q3.flavors.non_self_consistent_field — NSCF flavor wrapper.

Calls the ``q3`` engine in ``non_self_consistent_field`` mode.  Requires a
prior SCF run to have written an ``effective-potential.q3cp`` checkpoint.

Mode B (preferred): If the ``q3_core`` nanobind extension module is importable,
the calculation runs in-process (no subprocess overhead).

Mode A (fallback): If ``q3_core`` is not available, the ``q3`` binary is
invoked via subprocess and ``results.json`` is read back.

Typical usage::

    from q3.flavors import run_nscf
    from q3.io.results.classes import eigenvalue_arrays

    results = run_nscf("path/to/nscf_input.yml", q3_binary="./build/q3")
    eigenvalues, occupations, weights = eigenvalue_arrays(results.properties.eigenvalues_at_kpoints)
    # eigenvalues is a list[list[list[float]]] in Ry
    # indexed as [kpoint_index][spin_index][band_index]
"""

from __future__ import annotations

from pathlib import Path

from q3.binary import ExecutionPolicy, execution_policy_or_serial, run_q3
from q3.io.input_writer import ensure_calculation_type
from q3.io.results.classes import Results

try:
    import q3_core
except ImportError:  # the compiled extension is not built: Mode A runs the binary
    q3_core = None


def run_nscf(
    input_path: str | Path,
    *,
    q3_binary: str | Path | None = None,
    execution_policy: ExecutionPolicy | None = None,
    capture_output: bool = True,
) -> Results:
    """Run a non-self-consistent field (single-pass diagonalization) calculation.

    Parameters
    ----------
    input_path:
        Path to a Q3 YAML input file with ``calculation: nscf`` (or
        ``non_self_consistent_field``) and a ``kpoints_list`` section.
    q3_binary:
        Explicit path to the ``q3`` binary.  Used only in Mode A (subprocess).
        Auto-detected when ``None``.
    execution_policy:
        The execution policy the binary is started with (ranks, pools, the
        launcher).  Anything but the serial policy runs Mode A, since the
        in-process bindings can honour neither a launcher nor a binary option.
    capture_output:
        Pass ``False`` to stream binary stdout/stderr to the terminal.
        Used only in Mode A (subprocess).

    Returns
    -------
    Results
        The run's ``results.json``.  ``properties.eigenvalues_at_kpoints`` lists
        the eigenvalues at each requested k-point (``eigenvalue_arrays`` reshapes
        them).

    Raises
    ------
    Q3InputError
        If the input file specifies a non-NSCF calculation type or is missing
        ``kpoints_list``.
    Q3RuntimeError
        If the engine exits with a non-zero status (e.g. checkpoint missing).
    """
    input_path = ensure_calculation_type(
        input_path,
        expected="non_self_consistent_field",
        aliases=("nscf", "bands"),
    )

    execution_policy = execution_policy_or_serial(execution_policy)

    # ── Mode B: in-process via nanobind extension ─────────────────────────────
    if q3_core is not None and execution_policy.is_default:
        return Results.model_validate_json(q3_core.run_nscf(str(input_path)))

    # ── Mode A: subprocess ────────────────────────────────────────────────────
    return run_q3(
        input_path,
        q3_binary=q3_binary,
        execution_policy=execution_policy,
        capture_output=capture_output,
    )
