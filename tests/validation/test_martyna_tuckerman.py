"""
Validation test for the Martyna-Tuckerman isolated-system correction.

Overlays control.isolated_system_correction: martyna-tuckerman onto the same
charged-cell example test_makov_payne.py uses
(examples/total_energy--semiconductor--silicon/nc_charged.yml: FCC Si,
celldm(1)=10.3344 bohr, tot_charge=+1.0, PBE, NC pseudopotential) and checks
that the nested results.json object is populated and internally consistent.

SCOPE NOTE (see MartynaTuckermanCorrectionCalculator's class docstring for the
full statement): Q3 implements the POST-SCF (non-self-consistent) evaluation
of Martyna-Tuckerman -- evaluated once on the density already converged under
the ordinary periodic Coulomb kernel. This differs architecturally from QE's
`assume_isolated='m-t'`, which is self-consistent (the corrected potential
feeds back into the SCF loop, so QE's density itself relaxes differently). A
literal Q3-vs-QE total-energy comparison would therefore conflate that scope
difference with any actual formula difference, so -- unlike
test_makov_payne.py, which has a directly comparable QE reference for its
(also post-SCF) formula -- this test's numeric anchors below are MEASURED
directly from this exact Q3 configuration, not cross-validated against QE,
mirroring the precedent set by the DFT+U ortho-atomic regression anchors in
tests/integration/test_scf_integration.cpp
(NiFCC_HubbardOrthoAtomicConvergesToDifferentEnergyThanAtomic) for a value
this project could not (at the time) cross-check against an established
code either. Self-consistent Martyna-Tuckerman, which WOULD be directly
comparable to QE's own assume_isolated='m-t' total energy, is tracked as
future work in TODOs.md.

Also note: FCC Si's valence density is a delocalized covalent-bonding
density, not a compact/molecule-like one -- exactly the kind of system that
violates MartynaTuckermanCorrectionCalculator's own documented "density
localized within half the cell" validity assumption (its cheap locality
check fires a warning for this system: only ~54% of |charge density| lies
within the heuristic locality radius). This test exists to pin the
correction's WIRING (factory, YAML parsing, results.json output) with a
regression anchor, exactly like test_makov_payne.py's role for its own
scheme -- it is not a claim that this specific number is the physically
"correct" isolated-Si+ energy. The C++ unit tests
(tests/unit/layer_5_properties/isolated_system_correction/planewave/test_martyna_tuckerman.cpp)
are where the formula's own physical correctness is actually established, on
synthetic systems built to satisfy (or deliberately probe) that validity
assumption.
"""

from helpers import EXAMPLES_DIRECTORY, apply_config_overlay, run_q3_and_load_results

SILICON_EXAMPLE_DIRECTORY = EXAMPLES_DIRECTORY / "total_energy--semiconductor--silicon"
NC_CHARGED_INPUT = SILICON_EXAMPLE_DIRECTORY / "nc_charged.yml"

# MEASURED self-anchors for this exact Q3 configuration (see module docstring for why
# these are self-anchors, not a QE cross-check). Regenerate by rerunning this exact
# overlay against ./build/q3 if the underlying SCF/formula ever legitimately changes;
# do not hand-edit to make a failing test pass.
MEASURED_ENERGY_CORRECTION_RYDBERG = 0.022991603075479674
MEASURED_TOTAL_ENERGY_RYDBERG = -17.299615094052516
CORRECTION_TOLERANCE_RYDBERG = 1.0e-4
TOTAL_ENERGY_TOLERANCE_RYDBERG = 1.0e-4


def test_martyna_tuckerman_correction_self_consistent_anchor(q3_command, tmp_path):
    overlay = {"control": {"isolated_system_correction": "martyna-tuckerman"}}
    input_path = apply_config_overlay(NC_CHARGED_INPUT, overlay, tmp_path)
    results, _ = run_q3_and_load_results(
        q3_command, input_path, "si_nc_charged", working_directory=str(tmp_path)
    )

    correction = results.get("isolated_system_correction")
    assert correction is not None, (
        "results.json is missing the isolated_system_correction object -- "
        "control.isolated_system_correction: martyna-tuckerman should have populated it"
    )
    assert correction["method"] == "martyna-tuckerman"

    # Martyna-Tuckerman reports a single "energy" term (see
    # IsolatedSystemCorrectionResult's scheme-agnostic terms_in_ry design) -- unlike
    # Makov-Payne's first_order_ry/second_order_ry pair.
    assert "energy_ry" in correction
    assert "first_order_ry" not in correction
    assert "second_order_ry" not in correction

    energy_correction = float(correction["energy_ry"])
    corrected_total = float(correction["corrected_total_energy_ry"])
    total_energy = float(results["total_energy_ry"])

    assert abs(total_energy - MEASURED_TOTAL_ENERGY_RYDBERG) < TOTAL_ENERGY_TOLERANCE_RYDBERG, (
        f"uncorrected total_energy_ry = {total_energy:.8f} Ry drifted from the measured "
        f"anchor {MEASURED_TOTAL_ENERGY_RYDBERG:.8f} Ry -- the SCF itself changed, which "
        f"would also invalidate the correction anchor below"
    )
    assert (
        abs(energy_correction - MEASURED_ENERGY_CORRECTION_RYDBERG) < CORRECTION_TOLERANCE_RYDBERG
    ), (
        f"Martyna-Tuckerman energy correction = {energy_correction:.8f} Ry drifted from "
        f"its measured regression anchor {MEASURED_ENERGY_CORRECTION_RYDBERG:.8f} Ry "
        f"(self-anchored, not QE-cross-validated -- see module docstring)"
    )

    assert abs(float(correction["total_ry"]) - energy_correction) < 1.0e-12, (
        "total_ry must equal the single 'energy' term for Martyna-Tuckerman"
    )
    assert abs(corrected_total - (total_energy + energy_correction)) < 1.0e-9, (
        "isolated_system_correction.corrected_total_energy_ry must equal "
        "total_energy_ry + energy_ry"
    )


def test_martyna_tuckerman_not_applied_without_opt_in(q3_command, tmp_path):
    """Without control.isolated_system_correction, no correction object is written."""
    input_path = apply_config_overlay(NC_CHARGED_INPUT, {}, tmp_path)
    results, _ = run_q3_and_load_results(
        q3_command, input_path, "si_nc_charged", working_directory=str(tmp_path)
    )
    assert "isolated_system_correction" not in results
