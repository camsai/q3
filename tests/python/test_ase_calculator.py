"""tests/python/test_ase_calculator.py — Unit tests for q3.calculators.ase_calculator.

Tests the Q3-to-ASE stress sign/unit conversion in isolation, without
invoking the q3 binary or requiring ASE to be installed (the helper under
test only depends on numpy). The tests at the end are the exception: they
drive the calculator itself, against a stand-in for the binary rather than the
engine.
"""

import json

import numpy as np
import pytest
from mat3ra.code.constants import Coefficients
from q3.calculators.ase_calculator import (
    Q3ASECalculator,
    raise_if_derivative_incomplete,
    stress_kbar_to_ase_voigt,
)
from q3.errors import IncompleteDerivativeError, SCFNotConvergedError

# ── Fixtures ─────────────────────────────────────────────────────────────────

# A diagonal-only, compressive (Q3/Nielsen-Martin convention: positive
# diagonal = positive physical pressure = compressed cell) stress tensor,
# with distinct off-diagonal shear components to catch a row/column swap.
_Q3_STRESS_KBAR = [
    [100.0, 5.0, 6.0],
    [5.0, 200.0, 7.0],
    [6.0, 7.0, 300.0],
]

_KBAR_TO_EV_ANG3 = 1.0 / 1602.176634


def test_diagonal_components_are_negated_and_unit_converted():
    """Q3's compressive-positive kbar maps to ASE's tensile-positive eV/Å³."""
    voigt = stress_kbar_to_ase_voigt(_Q3_STRESS_KBAR)
    assert voigt[0] == pytest.approx(-100.0 * _KBAR_TO_EV_ANG3)
    assert voigt[1] == pytest.approx(-200.0 * _KBAR_TO_EV_ANG3)
    assert voigt[2] == pytest.approx(-300.0 * _KBAR_TO_EV_ANG3)


def test_off_diagonal_components_use_correct_voigt_order_and_sign():
    """Voigt order is (xx, yy, zz, yz, xz, xy); shear terms are negated too."""
    voigt = stress_kbar_to_ase_voigt(_Q3_STRESS_KBAR)
    assert voigt[3] == pytest.approx(-7.0 * _KBAR_TO_EV_ANG3)  # yz
    assert voigt[4] == pytest.approx(-6.0 * _KBAR_TO_EV_ANG3)  # xz
    assert voigt[5] == pytest.approx(-5.0 * _KBAR_TO_EV_ANG3)  # xy


def test_compressive_pressure_becomes_negative_ase_stress():
    """A purely hydrostatic, compressive (positive-pressure) Q3 stress must
    come out NEGATIVE in ASE's convention -- this is the sign that, if
    flipped, would make ExpCellFilter/FrechetCellFilter push a compressed
    cell to compress further instead of expand back toward equilibrium.
    """
    hydrostatic_compressive_kbar = [[50.0, 0.0, 0.0], [0.0, 50.0, 0.0], [0.0, 0.0, 50.0]]
    voigt = stress_kbar_to_ase_voigt(hydrostatic_compressive_kbar)
    assert all(component < 0.0 for component in voigt[:3])


def test_tensile_q3_stress_becomes_positive_ase_stress():
    """The mirror case: a negative-pressure (tensile/stretched) Q3 stress
    must come out POSITIVE in ASE's convention.
    """
    hydrostatic_tensile_kbar = [[-50.0, 0.0, 0.0], [0.0, -50.0, 0.0], [0.0, 0.0, -50.0]]
    voigt = stress_kbar_to_ase_voigt(hydrostatic_tensile_kbar)
    assert all(component > 0.0 for component in voigt[:3])


def test_zero_stress_round_trips_to_zero():
    voigt = stress_kbar_to_ase_voigt([[0.0] * 3 for _ in range(3)])
    assert all(component == pytest.approx(0.0) for component in voigt)


# ── Refusing a derivative Q3 says is incomplete ──────────────────────────────

"""
Q3 reports derivative terms it cannot compute. Atomic and ortho-atomic force
and stress are complete for every supported dataset family. The fixtures below
preserve the refusal contract for any future projector or backend whose
derivatives are not complete.

The fixtures below are synthetic results JSON, not runs: what is under test is
the refusal mechanism, which is independent of which term happens to be missing
this month, so they are left describing a generic incomplete pair rather than
being retired every time one of them is filled in.

The failure this guards against is not a crash. A geometry optimization driven
by such a force converges, to the wrong geometry, with nothing in its output
saying why; before these flags existed the only trace was a warning line in a
log nothing reads programmatically.
"""

INCOMPLETE_DERIVATIVE_RESULTS = {
    "converged": True,
    "total_energy_ry": -15.517022664,
    "forces_ry_per_bohr": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    "forces_are_incomplete": True,
    "forces_incomplete_reason": "the selected projector type has no complete force derivative",
    "stress_is_incomplete": True,
    "stress_incomplete_reason": ("the selected projector type has no complete stress derivative"),
}

COMPLETE_RESULTS = {
    "converged": True,
    "total_energy_ry": -15.7,
    "forces_ry_per_bohr": [[0.01, 0.0, 0.0]],
}


def test_energy_only_is_allowed_when_the_derivatives_are_incomplete():
    """The energy is correct on exactly the systems whose forces are not."""
    raise_if_derivative_incomplete(INCOMPLETE_DERIVATIVE_RESULTS, ["energy"])


def test_incomplete_forces_are_refused_when_asked_for():
    with pytest.raises(IncompleteDerivativeError) as caught:
        raise_if_derivative_incomplete(INCOMPLETE_DERIVATIVE_RESULTS, ["energy", "forces"])
    assert caught.value.quantity == "forces"
    assert "force derivative" in caught.value.reason


def test_incomplete_stress_is_refused_when_asked_for():
    with pytest.raises(IncompleteDerivativeError) as caught:
        raise_if_derivative_incomplete(INCOMPLETE_DERIVATIVE_RESULTS, ["stress"])
    assert caught.value.quantity == "stress"


def test_a_complete_result_passes_every_property():
    """The guard must be invisible to every system that does not trip it."""
    raise_if_derivative_incomplete(COMPLETE_RESULTS, ["energy", "forces", "stress"])


# ── Non-convergence ──────────────────────────────────────────────────────────


def test_a_failed_scf_says_how_many_iterations_it_performed(tmp_path):
    """The calculator used to report every failed SCF as 0 iterations.

    It read an ``iterations`` key the engine has never written. Run through
    the subprocess path against a stand-in binary that leaves the results a
    run out of iterations would, and exits 0 as the engine does.
    """
    ase = pytest.importorskip("ase")

    template_path = tmp_path / "template.yml"
    template_path.write_text("material: absent.json\ncontrol:\n  calculation: scf\n")
    binary = tmp_path / "q3"
    binary.write_text(
        "#!/bin/sh\n"
        "cat > q3_results.json <<'RESULTS'\n"
        '{"converged": false, "self_consistent_field_iterations": 100,'
        ' "provenance": {"precision": {"maximum_self_consistent_field_iterations": 100}}}\n'
        "RESULTS\n"
    )
    binary.chmod(0o755)

    atoms = ase.Atoms("Cu", positions=[(0.0, 0.0, 0.0)], cell=(10.0, 10.0, 10.0), pbc=True)
    atoms.calc = Q3ASECalculator(
        template_path, mode="subprocess", q3_binary=binary, workdir=tmp_path / "work"
    )

    with pytest.raises(SCFNotConvergedError) as failure:
        atoms.get_potential_energy()

    assert "after 100 iterations" in str(failure.value)
    assert failure.value.iterations == 100


# ── A non-YAML input's starting structure ──────────────────────────────────


def test_a_deck_run_as_written_gives_the_engines_structure_and_is_the_first_step(tmp_path):
    """A pw.x deck is read by the engine alone: its results' material is the structure.

    The stand-in binary counts its calls, so the test also shows the run is
    reused as the relaxation's first evaluation rather than repeated.
    """
    deck_path = tmp_path / "relax.in"
    deck_path.write_text("&control\n  calculation = 'relax'\n  prefix = 'si'\n/\n")
    results = {
        "converged": True,
        "total_energy_ry": -16.0,
        "forces_ry_per_bohr": [[0.0, 0.0, 0.0], [0.01, 0.0, 0.0]],
        "stress_kbar": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        "material": {
            "lattice": {
                "vectors": {
                    "a": [-2.7, 0.0, 2.7],
                    "b": [0.0, 2.7, 2.7],
                    "c": [-2.7, 2.7, 0.0],
                    "units": "angstrom",
                }
            },
            "basis": {
                "units": "crystal",
                "elements": [{"id": 1, "value": "Si"}, {"id": 2, "value": "Si"}],
                "coordinates": [
                    {"id": 1, "value": [0.0, 0.0, 0.0]},
                    {"id": 2, "value": [0.26, 0.25, 0.25]},
                ],
                "constraints": [
                    {"id": 1, "value": [False, False, False]},
                    {"id": 2, "value": [True, True, True]},
                ],
            },
        },
    }
    binary = tmp_path / "q3"
    binary.write_text(
        "#!/bin/sh\n"
        "echo run >> calls.log\n"
        f"cat > si_results.json <<'RESULTS'\n{json.dumps(results)}\nRESULTS\n"
    )
    binary.chmod(0o755)
    calculator = Q3ASECalculator(deck_path, q3_binary=binary, workdir=tmp_path / "work")

    atoms, run_results = calculator.run_input_as_written()

    assert run_results["total_energy_ry"] == -16.0
    assert atoms.get_chemical_symbols() == ["Si", "Si"]
    assert np.allclose(atoms.get_scaled_positions()[1], [0.26, 0.25, 0.25])
    assert len(atoms.constraints) == 1  # the first atom, held by if_pos 0 0 0
    assert np.isclose(atoms.get_potential_energy(), -16.0 * Coefficients.RYDBERG)
    assert np.allclose(
        atoms.get_forces(apply_constraint=False)[1, 0], 0.01 * Coefficients.ry_bohr_to_eV_A
    )
    assert atoms.get_stress().shape == (6,)
    assert (tmp_path / "calls.log").read_text().count("run") == 1
