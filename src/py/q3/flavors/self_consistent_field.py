"""q3.flavors.self_consistent_field — SCF flavor wrapper.

Calls the ``q3`` engine to run a self-consistent field calculation.

Mode B (preferred): If the ``q3_core`` nanobind extension module is importable,
the calculation runs in-process (no subprocess overhead).

Mode A (fallback): If ``q3_core`` is not available, the ``q3`` binary is
invoked via subprocess and ``results.json`` is read back.

Callers do not need to know which mode is active — the API is identical.

Example::

    from q3.flavors import run_scf

    results = run_scf("path/to/input.yml", q3_binary="./build/q3")
    print(f"Total energy: {results.properties.total_energy.value} Ry")
"""

from __future__ import annotations

from pathlib import Path

from q3.binary import ExecutionPolicy, execution_policy_or_serial, run_q3
from q3.io.input_writer import ensure_calculation_type
from q3.io.output_writer import raise_if_not_converged
from q3.io.results.classes import Results

try:
    import q3_core
except ImportError:  # the compiled extension is not built: Mode A runs the binary
    q3_core = None


def run_scf(
    input_path: str | Path,
    *,
    q3_binary: str | Path | None = None,
    execution_policy: ExecutionPolicy | None = None,
    capture_output: bool = True,
) -> Results:
    """Run a self-consistent field calculation.

    Parameters
    ----------
    input_path:
        Path to a Q3 YAML input file.  The ``calculation`` field may be
        ``scf`` or ``self_consistent_field``; both are accepted.
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
        The run's ``results.json``.

    Raises
    ------
    Q3InputError
        If the input file specifies a non-SCF calculation type.
    Q3RuntimeError
        If the engine exits with a non-zero status.
    SCFNotConvergedError
        If the engine runs successfully but SCF does not converge, in either
        mode.
    """
    input_path = ensure_calculation_type(
        input_path, expected="self_consistent_field", aliases=("scf",)
    )

    execution_policy = execution_policy_or_serial(execution_policy)

    # ── Mode B: in-process via nanobind extension ─────────────────────────────
    if q3_core is not None and execution_policy.is_default:
        results = Results.model_validate_json(q3_core.run_scf(str(input_path)))
        # The engine returns normally from a run that did not converge, and
        # run_q3 refuses such a result in Mode A. Which mode runs depends only
        # on whether the extension was built, so the refusal cannot depend on
        # it either: without this, the same calculation raised in one
        # environment and returned a plausible-looking energy in the other.
        raise_if_not_converged(results)
        return results

    # ── Mode A: subprocess ────────────────────────────────────────────────────
    return run_q3(
        input_path,
        q3_binary=q3_binary,
        execution_policy=execution_policy,
        capture_output=capture_output,
    )
