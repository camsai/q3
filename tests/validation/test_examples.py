"""
Validation tests for q3: ensure the canonical example inputs run,
converge, and produce physically accurate results.

Unlike regression tests (which use overlays to create custom configs),
these tests run the production examples as-is — validating that the
documentation examples work correctly.
"""

from q3.io.results.classes import Results

from helpers import (
    EXAMPLES_DIRECTORY,
    TOLERANCE_EXAMPLES_RYDBERG,
    TOLERANCE_STRESS_REGRESSION_KBAR,
    atomic_forces,
    energy_contribution,
    pressure_from_tensor,
    run_q3,
    stress_contributions,
    stress_tensor,
)

# stdout stress-line label -> results.json stress_contributions key.
# self_consistent_field.hpp's add_stress_contribution() calls are the JSON
# keys; the stdout labels are stress_tensor/planewave/init.hpp's
# print_contribution() calls.
STRESS_LABEL_TO_JSON_KEY = {
    "core corr": "core_correction",
    "Ewald": "ewald",
    "XC": "exchange_correlation",
    "Hartree": "hartree",
    "kinetic": "kinetic",
    "local PP": "local_pseudopotential",
    "nonlocal": "nonlocal_pseudopotential",
    "augment": "augmentation",
}

EXPECTED_DATA = {
    "nc": {
        "energy": -16.91341791,
        "energy_contributions": {
            "ewald": -16.67997255,
            "exchange_correlation": -6.16953852,
            "hartree": 1.13363851,
            "one_electron": 4.80245465,
        },
        "forces": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        "stress": {
            "core corr": -770.32,
            "Ewald": -2964.19,
            "XC": -164.14,
            "Hartree": 201.46,
            "kinetic": 2176.30,
            "local PP": -152.11,
            "nonlocal": 1676.68,
            "total": 3.69,
        },
    },
    "us": {
        "energy": -15.71731181,
        "energy_contributions": {
            "ewald": -16.67997254,
            "exchange_correlation": -4.82245256,
            "hartree": 1.13646631,
            "one_electron": 4.64864698,
        },
        "forces": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        "stress": {
            "augment": -1774.01,
            "Ewald": -2964.19,
            "XC": -788.72,
            "Hartree": 201.96,
            "kinetic": 2621.77,
            "local PP": -1231.03,
            "nonlocal": 2164.52,
            "total": 4.32,
        },
    },
}

EXPECTED_DATA_COPPER = {
    "nc": {
        "energy": -386.58337932,
        "energy_contributions": {
            "ewald": -241.86590336,
            "exchange_correlation": -48.25774811,
            "hartree": 119.25780124,
            "one_electron": -215.71491204,
        },
        "forces": [[0.0, 0.0, 0.0]],
        "stress": {
            "core corr": -23051.97,
            "Ewald": -148035.05,
            "XC": -2384.78,
            "Hartree": 72992.71,
            "kinetic": 225951.26,
            "local PP": -21898.83,
            "nonlocal": -103517.93,
            "total": 55.41,
        },
    },
    "us": {
        "energy": -403.77922845,
        "energy_contributions": {
            "ewald": -241.86590342,
            "exchange_correlation": -38.81411584,
            "hartree": 145.95388090,
            "one_electron": -269.05049097,
        },
        "forces": [[0.0, 0.0, 0.0]],
        "stress": {
            "augment": 47234.26,
            "core corr": -662.95,
            "Ewald": -148035.04,
            "XC": -22057.19,
            "Hartree": 89331.69,
            "kinetic": 50485.44,
            "local PP": -7890.64,
            "nonlocal": 38854.14,
            "total": 25.45,
        },
    },
    "nc_888": {
        # 8x8x8 Monkhorst-Pack grid (512 → 60 IBZ k-points after symmetry reduction).
        # Energy reference from converged Q3 runs (np=1/2/4 agree to < 1e-9 Ry).
        "energy": -386.58688924,
        "energy_contributions": {
            "ewald": -241.86590336,
            "exchange_correlation": -48.25599289,
            "hartree": 119.24160087,
            "one_electron": -215.70580302,
        },
        "forces": [[0.0, 0.0, 0.0]],
        "stress": {
            "core corr": -23051.85,
            "Ewald": -148035.05,
            "XC": -2383.78,
            "Hartree": 72982.33,
            "kinetic": 225932.19,
            "local PP": -21892.07,
            "nonlocal": -103512.98,
            "total": 38.78,
        },
    },
}


def run_example_and_validate(
    q3_command,
    example_directory,
    yml_filename,
    prefix,
    expected,
    energy_tolerance=None,
    stress_tolerance=None,
):
    """Run a q3 example and validate total energy, contributions, forces, and stresses."""
    yml_path = example_directory / yml_filename

    # Clean up old results to ensure we verify the new run.
    json_path = example_directory / "output" / f"{prefix}_results.json"
    if json_path.exists():
        try:
            json_path.unlink()
        except OSError:
            pass

    result = run_q3(q3_command, yml_path, working_directory=example_directory)
    assert result.returncode == 0, (
        f"q3 example run failed for {yml_filename}:\n{result.stderr}\n{result.stdout}"
    )

    assert json_path.exists(), f"Output results JSON not found: {json_path}"
    data = Results.read(json_path)

    assert data.runtime_log.converged is True, f"SCF did not converge for {yml_filename}!"

    if energy_tolerance is None:
        energy_tolerance = TOLERANCE_EXAMPLES_RYDBERG
    if stress_tolerance is None:
        stress_tolerance = TOLERANCE_STRESS_REGRESSION_KBAR

    # 1. Compare total energy.
    computed_energy = data.properties.total_energy.value
    difference = abs(computed_energy - expected["energy"])
    assert difference < energy_tolerance, (
        f"Total energy {computed_energy:.8f} Ry differs from expected "
        f"{expected['energy']:.8f} Ry by {difference:.6f} Ry "
        f"(tolerance: {energy_tolerance} Ry)"
    )

    # 2. Compare energy contributions.
    # In Q3's results, "one_electron" = band_energy + double_counting_correction,
    # matching QE's definition: E_one_electron = Σ(ε_n) + double_counting_correction.
    band_energy = data.properties.total_energy_contributions.eigenvalues.value
    double_counting_correction = energy_contribution(data, "double_counting")
    contributions_from_data = {
        "ewald": data.properties.total_energy_contributions.ewald.value,
        "exchange_correlation": data.properties.total_energy_contributions.exchange_correlation.value,
        "hartree": data.properties.total_energy_contributions.hartree.value,
        "one_electron": band_energy + double_counting_correction,
    }
    for term, expected_value in expected["energy_contributions"].items():
        assert term in contributions_from_data, (
            f"Missing energy contribution key in results.json: {term}"
        )
        computed_value = contributions_from_data[term]
        difference_term = abs(computed_value - expected_value)
        assert difference_term < energy_tolerance, (
            f"Energy term '{term}' value {computed_value:.8f} Ry differs from "
            f"expected {expected_value:.8f} Ry by {difference_term:.6f} Ry "
            f"(tolerance: {energy_tolerance} Ry)"
        )

    # 3. Compare forces.
    computed_forces = atomic_forces(data)
    assert len(computed_forces) == len(expected["forces"]), "Force vector length mismatch"
    for atom_index, (computed_force, expected_force) in enumerate(
        zip(computed_forces, expected["forces"])
    ):
        for dimension in range(3):
            difference_force = abs(computed_force[dimension] - expected_force[dimension])
            assert difference_force < 1e-5, (
                f"Atom {atom_index + 1} force component {dimension} value "
                f"{computed_force[dimension]:.8f} differs from expected "
                f"{expected_force[dimension]:.8f} (tolerance: 1e-5)"
            )

    # 4. Compare stress contributions. expected["stress"] uses the old stdout
    # labels (STRESS_LABEL_TO_JSON_KEY maps them to results.json's keys);
    # "total" has no per-contribution JSON key, so it's the trace/3 of the
    # top-level stress_kbar tensor instead, the same quantity the old
    # "Pressure = ..." stdout line reported.
    stress_contributions_kbar = stress_contributions(data)
    for term, expected_value in expected["stress"].items():
        if term == "total":
            computed_value = pressure_from_tensor(stress_tensor(data))
        else:
            json_key = STRESS_LABEL_TO_JSON_KEY[term]
            assert json_key in stress_contributions_kbar, (
                f"Missing stress contribution key in results.json: {json_key} (for '{term}')"
            )
            computed_value = pressure_from_tensor(stress_contributions_kbar[json_key])
        difference_stress = abs(computed_value - expected_value)
        assert difference_stress < stress_tolerance, (
            f"Stress contribution '{term}' value {computed_value:.2f} kbar differs from "
            f"expected {expected_value:.2f} kbar by {difference_stress:.2f} kbar "
            f"(tolerance: {stress_tolerance} kbar)"
        )


# ── Tests ────────────────────────────────────────────────────────────

SILICON_EXAMPLE_DIRECTORY = EXAMPLES_DIRECTORY / "total_energy--semiconductor--silicon"


def test_silicon_norm_conserving(q3_command):
    """Validate Silicon Norm-Conserving PBE example."""
    run_example_and_validate(
        q3_command,
        SILICON_EXAMPLE_DIRECTORY,
        "nc.yml",
        "si_nc",
        EXPECTED_DATA["nc"],
    )


def test_silicon_ultrasoft(q3_command):
    """Validate Silicon Ultrasoft PZ example."""
    run_example_and_validate(
        q3_command,
        SILICON_EXAMPLE_DIRECTORY,
        "us.yml",
        "si_us",
        EXPECTED_DATA["us"],
    )


COPPER_EXAMPLE_DIRECTORY = EXAMPLES_DIRECTORY / "total_energy--metal--copper"


def test_copper_norm_conserving(q3_command):
    """Validate Copper Norm-Conserving PBE example (metal with smearing)."""
    run_example_and_validate(
        q3_command,
        COPPER_EXAMPLE_DIRECTORY,
        "nc.yml",
        "cu_nc",
        EXPECTED_DATA_COPPER["nc"],
        energy_tolerance=5e-3,
    )


def test_copper_ultrasoft(q3_command):
    """Validate Copper Ultrasoft PBE example (metal with smearing)."""
    run_example_and_validate(
        q3_command,
        COPPER_EXAMPLE_DIRECTORY,
        "us.yml",
        "cu_us",
        EXPECTED_DATA_COPPER["us"],
        energy_tolerance=5e-3,
    )


def test_copper_norm_conserving_888(q3_command):
    """Validate Copper NC 8x8x8 k-grid example (denser mesh, MPI parallelism benchmark)."""
    run_example_and_validate(
        q3_command,
        COPPER_EXAMPLE_DIRECTORY,
        "nc_888.yml",
        "cu_nc_888",
        EXPECTED_DATA_COPPER["nc_888"],
        # Wider tolerances to accommodate cross-platform differences:
        # individual energy/stress components shift when the SCF converges
        # to a slightly different charge density with different LibXC
        # versions or compiler optimizations.
        energy_tolerance=5e-3,
        stress_tolerance=5.0,
    )
