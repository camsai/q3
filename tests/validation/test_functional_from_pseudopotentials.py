"""
An input that names no exchange-correlation functional runs the one its
pseudopotentials were generated with (machine-validated, no reference code needed).

That is pw.x's rule when ``input_dft`` is absent. Q3 used to fall back on
Perdew-Zunger LDA instead, so a deck with PBE pseudopotentials and no
functional ran as LDA, converged, and reported success. On the silicon example
below, with ``model.functional`` removed, it gave -16.894765600 Ry against
-16.913417930 Ry for the same deck with ``functional: pbe``: 18.65 mRy for the
two-atom cell, 127 meV/atom.

Removing the key from the committed PBE example must reproduce the PBE run, and
the provenance record in the results file must name the functional that ran.

Run:
    .venv/bin/pytest tests/validation/test_functional_from_pseudopotentials.py -v --q3-bin ./build/q3
"""

import subprocess

import pytest

from helpers import EXAMPLES_DIRECTORY, apply_config_overlay, load_results_json

SILICON_INPUT = EXAMPLES_DIRECTORY / "total_energy--semiconductor--silicon" / "nc.yml"
SILICON_PREFIX = "si_nc"

SAME_CALCULATION_TOLERANCE_RYDBERG = 1e-9
"""
Both runs are the same calculation once the functional is resolved, so they
agree to rounding (measured: at most 7e-15 Ry apart over four repeats). The
defect this guards against is 1.87e-2 Ry.
"""


def run_and_load_results(q3_command, overlay, run_directory):
    """Run the silicon example with an overlay in its own directory; return results.json."""
    merged_input = apply_config_overlay(SILICON_INPUT, overlay, run_directory)
    completed = subprocess.run(
        list(q3_command) + [str(merged_input)],
        cwd=str(run_directory),
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert completed.returncode == 0, (
        f"q3 exited with code {completed.returncode}\n"
        f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )
    return load_results_json(run_directory / "output", SILICON_PREFIX)


@pytest.fixture(scope="module")
def named_pbe_results(q3_command, tmp_path_factory):
    """The committed example, which names ``functional: pbe``."""
    return run_and_load_results(q3_command, {}, tmp_path_factory.mktemp("functional_named"))


@pytest.fixture(scope="module")
def unnamed_results(q3_command, tmp_path_factory):
    """The same example with ``model.functional`` removed; its dataset states PBE."""
    return run_and_load_results(
        q3_command, {"model": {"functional": None}}, tmp_path_factory.mktemp("functional_unnamed")
    )


def test_unnamed_functional_reproduces_the_pseudopotentials_functional(
    named_pbe_results, unnamed_results
):
    """No functional in the input gives the PBE energy, not the LDA one."""
    named = float(named_pbe_results["total_energy_ry"])
    unnamed = float(unnamed_results["total_energy_ry"])
    difference = abs(unnamed - named)
    assert difference < SAME_CALCULATION_TOLERANCE_RYDBERG, (
        f"An input naming no functional did not run its pseudopotentials' functional:\n"
        f"  E(functional: pbe)    = {named:.12f} Ry\n"
        f"  E(no functional)      = {unnamed:.12f} Ry\n"
        f"  |diff|                = {difference:.2e} Ry"
    )


def test_provenance_names_the_functional_that_ran(unnamed_results, named_pbe_results):
    """The results file records PBE, so the resolved functional is visible after the run.

    It records it as ``pbe``, the spelling every reader's name is lower-cased to,
    and the same whether the deck named it or the dataset supplied it.
    """
    recorded = unnamed_results["provenance"]["model"]["exchange_correlation_functional"]
    named = named_pbe_results["provenance"]["model"]["exchange_correlation_functional"]
    assert recorded == "pbe", f"provenance records {recorded!r}, expected 'pbe'"
    assert named == recorded, f"a named functional is recorded as {named!r}, not {recorded!r}"
