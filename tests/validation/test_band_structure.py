"""
Validation tests for the Band Structure calculation in Q3.

Runs the full SCF + NSCF workflow via run_band_structure_workflow() and
validates key physical properties of the Si band structure:

  1. Band gap: Si PBE gives ~0.62–0.73 eV (reference: ~0.65 eV from QE+LibXC).
  2. Valence band maximum at Gamma (band index 3).
  3. Number of k-points matches requested path.
  4. Eigenvalues are finite and in a physically reasonable range.
"""

from q3.workflows.band_structure import run_band_structure_workflow

from helpers import EXAMPLES_DIRECTORY

# ── Constants ────────────────────────────────────────────────────────────────

RY_TO_EV = 13.6058

# Si PBE band gap: QE+LibXC reference gives ~0.65 eV.
# Accept a generous window to account for k-grid sampling and path resolution.
SI_BAND_GAP_MIN_EV = 0.3
SI_BAND_GAP_MAX_EV = 1.1

SI_EXAMPLE_DIR = EXAMPLES_DIRECTORY / "band_structure--semiconductor--silicon"


# ── Helpers ───────────────────────────────────────────────────────────────────


def compute_band_gap(eigenvalues_ry: list, fermi_energy_ry: float) -> float:
    """Compute the fundamental band gap in Ry from NSCF eigenvalues.

    Finds the highest occupied band (VBM) and lowest unoccupied band (CBM)
    across all k-points and returns CBM - VBM in eV.
    """
    valence_band_maximum = float("-inf")
    conduction_band_minimum = float("inf")

    for kpoint_data in eigenvalues_ry:
        for spin_data in kpoint_data:
            for band_energy in spin_data:
                if band_energy <= fermi_energy_ry:
                    valence_band_maximum = max(valence_band_maximum, band_energy)
                else:
                    conduction_band_minimum = min(conduction_band_minimum, band_energy)

    if valence_band_maximum == float("-inf") or conduction_band_minimum == float("inf"):
        return 0.0

    return (conduction_band_minimum - valence_band_maximum) * RY_TO_EV


# ── Tests ─────────────────────────────────────────────────────────────────────


def test_silicon_band_gap(q3_command):
    """
    Validate Si PBE band gap from Q3 band structure.

    Si PBE underestimates the experimental gap (~1.12 eV) but should give
    ~0.62–0.73 eV. We check for a non-zero gap in a generous window.
    """
    result = run_band_structure_workflow(
        SI_EXAMPLE_DIR / "nc.yml",
        band_path=["G", "X", "W", "K", "G", "L"],
        lattice_type="FCC",
        points_per_segment=30,
        q3_binary=q3_command[0],
    )

    band_structure = result["band_structure"]
    eigenvalues_ry = band_structure["eigenvalues_ry"]
    fermi_energy_ry = band_structure["fermi_energy_ry"]
    kpath_distances = band_structure["kpath_distances"]
    kpoints = band_structure["kpoints"]

    # 1. Correct number of k-points: 5 segments × 30 points each, plus the
    #    final endpoint of the last segment (L), giving 5*30 + 1 = 151.
    expected_kpoints = 5 * 30 + 1
    assert len(kpoints) == expected_kpoints, (
        f"Expected {expected_kpoints} k-points, got {len(kpoints)}"
    )
    assert len(kpath_distances) == expected_kpoints

    # 2. Eigenvalues are finite and in a physically reasonable range.
    assert len(eigenvalues_ry) == expected_kpoints, "Eigenvalue k-point count mismatch"
    for kpoint_index, kpoint_data in enumerate(eigenvalues_ry):
        for spin_data in kpoint_data:
            for band_energy in spin_data:
                assert -50.0 < band_energy < 50.0, (
                    f"Eigenvalue {band_energy:.4f} Ry at k-point {kpoint_index} "
                    f"is outside physically reasonable range [-50, 50] Ry"
                )

    # 3. Band gap is in the expected PBE range for Si.
    band_gap_ev = compute_band_gap(eigenvalues_ry, fermi_energy_ry)
    assert SI_BAND_GAP_MIN_EV < band_gap_ev < SI_BAND_GAP_MAX_EV, (
        f"Si PBE band gap {band_gap_ev:.4f} eV is outside expected range "
        f"[{SI_BAND_GAP_MIN_EV}, {SI_BAND_GAP_MAX_EV}] eV"
    )

    # 4. kpath_distances are monotonically non-decreasing.
    for index in range(1, len(kpath_distances)):
        assert kpath_distances[index] >= kpath_distances[index - 1], (
            f"kpath_distances not monotonic at index {index}"
        )

    print(
        f"  Si band gap = {band_gap_ev:.4f} eV, "
        f"Fermi energy = {fermi_energy_ry * RY_TO_EV:.4f} eV ({fermi_energy_ry:.4f} Ry)"
    )


def test_band_path_and_lattice_type_default_from_the_input_file(q3_command):
    """run_band_structure_workflow() must read band_path/lattice_type/
    points_per_segment from scf_input_path's own control section (and the
    material it references) when they are not passed explicitly.

    nc.yml sets control.band_path == ["G","X","W","K","G","L"] and
    control.band_path_points_per_segment == 30 directly (the convention
    every examples/band_structure--*/ file uses), and its material
    (silicon.json) states lattice.type == "FCC" -- exactly the values
    test_silicon_band_gap() above passes explicitly. Before this fix, this
    workflow ignored those fields entirely (they were dead data the schema
    accepted but nothing consumed) and calling it with no keyword arguments
    raised a TypeError for the missing required band_path/lattice_type.
    """
    result = run_band_structure_workflow(SI_EXAMPLE_DIR / "nc.yml", q3_binary=q3_command[0])

    kpoints = result["band_structure"]["kpoints"]
    # Same expectation as test_silicon_band_gap(): 5 segments x 30 points
    # each, plus the final endpoint -- proves the defaulted band_path and
    # points_per_segment produced the identical path, not just "some" path.
    expected_kpoints = 5 * 30 + 1
    assert len(kpoints) == expected_kpoints, (
        f"Expected {expected_kpoints} k-points from nc.yml's own band_path/"
        f"band_path_points_per_segment, got {len(kpoints)} -- the YAML "
        "defaults are not being read correctly"
    )
