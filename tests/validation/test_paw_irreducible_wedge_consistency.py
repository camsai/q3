"""The irreducible wedge has to give the full zone's answer (self-validated).

A symmetry-reduced mesh samples one k-point per star. Everything accumulated
over it stands for the whole zone only once it has been symmetrized over the
crystal's operations, and PAW has two such quantities, not one: the charge
density on the grid, and the on-site occupations rho_ij that the one-centre
Hartree and exchange-correlation terms are built from. The grid density has
always been symmetrized. rho_ij was not, and because it reaches the total
energy without passing through a grid, nothing downstream repaired it --
levels that symmetry makes degenerate came out split, and the total energy
sat above the full mesh's.

Measured before the fix, at the settings below: diamond PAW put its Gamma
valence maximum at 13.7274 / 13.7427 / 13.7465 eV against the full mesh's
13.7393 three times, and 0.19 mRy above it in total energy; silicon PAW 0.25
mRy. The ultrasoft and norm-conserving controls were clean then and are clean
now -- their only consumer of rho_ij is the augmentation charge, which the
density symmetrizer repairs on the grid.

Both halves of each pair are Q3 runs, so this needs no reference code: the
full mesh is its own oracle, and `control.disable_symmetry` is how to ask for
it.

Run:
    .venv/bin/pytest tests/validation/test_paw_irreducible_wedge_consistency.py -v --q3-bin ./build/q3
"""

import pytest

from helpers import (
    EXAMPLES_DIRECTORY,
    apply_config_overlay,
    eigenvalues_ry,
    kpoint_weights,
    load_results_json,
    run_q3,
)

RYDBERG_IN_ELECTRONVOLTS = 13.605693122994

DIAMOND_PAW = EXAMPLES_DIRECTORY / "total_energy--insulator--diamond" / "paw.yml"
SILICON_DIRECTORY = EXAMPLES_DIRECTORY / "total_energy--semiconductor--silicon"

CASES = {
    # name: (input, prefix, ecutwfc, ecutrho)
    "diamond-paw": (DIAMOND_PAW, "c_paw", 30.0, 240.0),
    "silicon-paw": (SILICON_DIRECTORY / "paw.yml", "si_paw", 30.0, 240.0),
    "silicon-ultrasoft": (SILICON_DIRECTORY / "us.yml", "si_us", 30.0, 240.0),
    "silicon-norm-conserving": (SILICON_DIRECTORY / "nc.yml", "si_nc", 30.0, 120.0),
}

WEDGE_ENERGY_TOLERANCE_RYDBERG = 5.0e-6
"""
Both sums are over the same zone, so what is left is discretization noise in
quantities the two runs compute differently -- the k-point list, and with it
the FFT and mixing histories. Measured after the fix: diamond PAW 7e-7 Ry,
silicon PAW below 1e-10, and the two silicon controls 9e-8 and 3e-9. The bound
sits an order of magnitude above the worst of those and 40x below the 1.9e-4 Ry
the defect produced.
"""

DEGENERACY_TOLERANCE_ELECTRONVOLTS = 1.0e-3
"""
Diamond's Gamma valence maximum is a symmetry-forced triplet. It came out
19 meV wide and is now 0.1 meV wide; 1 meV is loose enough for the residual
and 19x below the defect.
"""


def run_pair(q3_command, tmp_path_factory, case):
    """One reduced-wedge run and one full-mesh run on the same 2x2x2 mesh."""
    base, prefix, wavefunction_cutoff, density_cutoff = CASES[case]
    results = {}
    for label, reduced in (("wedge", True), ("full", False)):
        directory = tmp_path_factory.mktemp(f"{case}-{label}")
        overlay = {
            "method": {
                "cutoffs": {
                    "wavefunction": wavefunction_cutoff,
                    "density": density_cutoff,
                },
                "scf": {
                    # PAW's rho_ij channel converges behind the density; see
                    # KNOWN-ISSUES.md on the PAW energy scatter at 1e-8.
                    "convergence_threshold": 1.0e-12,
                    "kgrid": {"dimensions": [2, 2, 2], "shifts": [0, 0, 0]},
                },
            },
            "control": {"disable_symmetry": not reduced},
        }
        merged = apply_config_overlay(base, overlay, directory)
        completed = run_q3(q3_command, merged, working_directory=directory, timeout=900)
        assert completed.returncode == 0, (
            f"{case} ({label}) exited with {completed.returncode}\n"
            f"stdout:\n{completed.stdout[-4000:]}\nstderr:\n{completed.stderr}"
        )
        results[label] = load_results_json(directory / "output", prefix)
    return results


def gamma_bands_in_electronvolts(results):
    """Bands at Gamma, which a Gamma-centred mesh generates and keeps first.

    Asserted rather than assumed: Gamma is a star of one, so its weight is the
    unreduced 2/8 whether or not the mesh was reduced.
    """
    weights = kpoint_weights(results)
    assert weights[0] == pytest.approx(0.25), (
        f"first k-point has weight {weights[0]}, so it is not the unreduced Gamma"
    )
    bands = eigenvalues_ry(results)[0]
    if bands and isinstance(bands[0], list):
        bands = bands[0]
    return [value * RYDBERG_IN_ELECTRONVOLTS for value in bands]


@pytest.mark.parametrize("case", list(CASES))
def test_the_wedge_reproduces_the_full_meshs_total_energy(q3_command, tmp_path_factory, case):
    results = run_pair(q3_command, tmp_path_factory, case)
    wedge_energy = float(results["wedge"].properties.total_energy.value)
    full_energy = float(results["full"].properties.total_energy.value)
    assert wedge_energy == pytest.approx(full_energy, abs=WEDGE_ENERGY_TOLERANCE_RYDBERG), (
        f"{case}: wedge {wedge_energy:.10f} Ry vs full mesh {full_energy:.10f} Ry "
        f"({(wedge_energy - full_energy) * 1e3:+.4f} mRy)"
    )


def test_the_wedge_keeps_diamonds_valence_maximum_degenerate(q3_command, tmp_path_factory):
    """The three-fold Gamma valence maximum, which the unsymmetrized rho_ij split."""
    results = run_pair(q3_command, tmp_path_factory, "diamond-paw")
    wedge_bands = gamma_bands_in_electronvolts(results["wedge"])
    full_bands = gamma_bands_in_electronvolts(results["full"])

    valence_maximum = wedge_bands[1:4]
    spread = max(valence_maximum) - min(valence_maximum)
    assert spread < DEGENERACY_TOLERANCE_ELECTRONVOLTS, (
        f"Gamma valence maximum split by {spread * 1e3:.1f} meV: "
        f"{[f'{value:.5f}' for value in valence_maximum]}"
    )
    for band_index in range(1, 4):
        assert wedge_bands[band_index] == pytest.approx(
            full_bands[band_index], abs=DEGENERACY_TOLERANCE_ELECTRONVOLTS
        ), (
            f"band {band_index}: wedge {wedge_bands[band_index]:.5f} eV vs full {full_bands[band_index]:.5f} eV"
        )
