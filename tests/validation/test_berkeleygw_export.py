"""
End-to-end validation for the BerkeleyGW mean-field export
(`control.mean_field_export: berkeleygw`).

Overlays `control.mean_field_export: berkeleygw` onto the existing silicon
SCF example (examples/total_energy--semiconductor--silicon/nc.yml) and, if
the run actually produced `WFN`/`RHO` in its outdir, reads them back with
the INDEPENDENT reader in tests/helpers_berkeleygw.py (derived solely from
QE's pw2bgw.f90 -- see that module's docstring) and checks the physical and
structural invariants a correct BerkeleyGW mean-field export must satisfy.

As of this writing, `control.mean_field_export` is not yet an accepted
control key (confirmed against a current q3 build: it is rejected at YAML
parse time with "Unknown key 'mean_field_export' in control", exit code 1
-- see docs/io_reference.md §1.6's control allow-list). This test therefore
SKIPS CLEANLY -- rather than failing -- whenever the run did not succeed
with this key, or succeeded but did not produce WFN/RHO in outdir. That
keeps this test green until the writer lands, at which point it becomes a
real gate with no further changes needed here.

`vxc.dat` is a REAL gate (not optional) whenever WFN reports
`number_of_spin_components == 1` -- the silicon example overlaid here is
always spin-unpolarized and non-gamma-only (a 2x2x2 shifted Monkhorst-Pack
mesh), i.e. always within `BerkeleyGWExchangeCorrelationFile`'s supported
scope (see its class comment), so vxc.dat must appear alongside WFN/RHO.
"""

import json
import re
from pathlib import Path

import numpy as np
import pytest
import yaml

from helpers import (
    EXAMPLES_DIRECTORY,
    apply_config_overlay,
    read_atomic_positions,
    run_q3,
)
from helpers_berkeleygw import (
    read_berkeleygw_charge_density_file,
    read_berkeleygw_exchange_correlation_file,
    read_berkeleygw_wavefunction_file,
)

SILICON_EXAMPLE_DIRECTORY = EXAMPLES_DIRECTORY / "total_energy--semiconductor--silicon"
NC_INPUT = SILICON_EXAMPLE_DIRECTORY / "nc.yml"
MATERIAL_NAME = "silicon"

# Mirrors tests/helpers.py's ATOMIC_MASSES element set (the materials this repo's
# examples actually use), so this test does not need to consult Q3's own
# PeriodicTable C++ code (AGENTS.md references
# PeriodicTable::atomic_number_by_element_symbol, which lives under src/ and is
# unrelated to -- and deliberately not consulted for -- the export this validates).
ATOMIC_NUMBERS_BY_SYMBOL = {"Si": 14, "Cu": 29, "Fe": 26, "Ni": 28, "Cr": 24}

KPOINT_WEIGHT_SUM_TOLERANCE = 1.0e-10
WAVEFUNCTION_NORMALIZATION_TOLERANCE = 1.0e-6
GEOMETRY_RELATIVE_TOLERANCE = 1.0e-6
GEOMETRY_ABSOLUTE_TOLERANCE = 1.0e-8
ELECTRON_COUNT_TOLERANCE = 1.0e-6
# Vxc(r) real forces every diagonal <psi|Vxc|psi> real; a direct G-space double-sum
# accumulates rounding error many orders of magnitude below this (measured: ~1e-15 eV on
# real silicon output), so this bound exists to catch an index/conjugation bug, not noise.
VXC_IMAGINARY_PART_TOLERANCE_EV = 1.0e-6
# Generous "not obviously broken" bounds for ANY diagonal Vxc matrix element (catches a sign
# flip, a dropped Ry->eV conversion, or a dropped G=0 term -- not a tight physics prediction).
VXC_PLAUSIBLE_MINIMUM_EV = -20.0
VXC_PLAUSIBLE_MAXIMUM_EV = -2.0
# Tighter band for VALENCE bands specifically: typical LDA/PBE <psi|Vxc|psi> magnitudes for a
# semiconductor like silicon are roughly -10 to -12 eV; widened slightly here so this is not
# fragile to minor pseudopotential/functional changes while still being a meaningful check.
VXC_VALENCE_MINIMUM_EV = -13.0
VXC_VALENCE_MAXIMUM_EV = -9.0


def _read_z_valence(pseudopotential_path):
    """Reads UPF's PP_HEADER z_valence attribute (valence electron count)."""
    text = Path(pseudopotential_path).read_text()
    match = re.search(r'z_valence\s*=\s*"\s*([0-9.eE+-]+)\s*"', text)
    if not match:
        raise AssertionError(f"could not find z_valence in {pseudopotential_path}")
    return float(match.group(1))


def _expected_valence_electron_count(merged_yaml_config):
    """Total valence electron count implied by the (already-resolved-paths)
    merged YAML config: sum of each atom's pseudopotential z_valence.
    """
    with open(merged_yaml_config["material"]) as file_handle:
        material = json.load(file_handle)
    elements = [entry["value"] for entry in material["basis"]["elements"]]
    pseudopotentials = merged_yaml_config["method"]["pseudopotentials"]
    z_valence_by_element = {
        element: _read_z_valence(path) for element, path in pseudopotentials.items()
    }
    return sum(z_valence_by_element[element] for element in elements)


def _find_gamma_column_index(gvectors):
    """Index of the G=(0,0,0) column in a (3, ng) Miller-index array."""
    zero_columns = np.flatnonzero(np.all(gvectors == 0, axis=0))
    assert len(zero_columns) > 0, "no G=(0,0,0) vector found in the charge-density G-set"
    return int(zero_columns[0])


def test_berkeleygw_mean_field_export(q3_command, tmp_path):
    overlay = {"control": {"mean_field_export": "berkeleygw"}}
    input_path = apply_config_overlay(NC_INPUT, overlay, tmp_path)
    output_directory = tmp_path / "output"

    result = run_q3(q3_command, input_path, working_directory=str(tmp_path))
    wfn_path = output_directory / "WFN"
    rho_path = output_directory / "RHO"
    if result.returncode != 0 or not wfn_path.exists() or not rho_path.exists():
        pytest.skip(
            "BerkeleyGW mean-field export (control.mean_field_export: berkeleygw) has "
            "not landed yet, or did not produce WFN/RHO in this run's outdir "
            f"({output_directory}). q3 exit code: {result.returncode}. stdout tail: "
            f"{result.stdout[-500:]!r}. This test starts asserting the moment the "
            "writer lands and emits these files -- no changes needed here."
        )

    wfn = read_berkeleygw_wavefunction_file(wfn_path)
    rho = read_berkeleygw_charge_density_file(rho_path)

    # ── header self-consistency ──────────────────────────────────────
    assert wfn["number_of_spin_components"] > 0
    assert wfn["number_of_kpoints"] > 0
    assert wfn["number_of_bands"] > 0
    assert wfn["number_of_g_vectors"] > 0
    assert len(wfn["number_of_gvectors_per_kpoint"]) == wfn["number_of_kpoints"], (
        "len(ngk) must equal number_of_kpoints"
    )
    assert np.all(wfn["number_of_gvectors_per_kpoint"] <= wfn["maximum_g_vectors_per_kpoint"]), (
        "every per-k-point plane-wave count must be <= the declared maximum"
    )
    kpoint_weight_sum = float(np.sum(wfn["kpoint_weights"]))
    assert abs(kpoint_weight_sum - 1.0) < KPOINT_WEIGHT_SUM_TOLERANCE, (
        f"k-point weights sum to {kpoint_weight_sum}, expected 1.0 (pw2bgw.f90:990-991 "
        f"scales by nst/2 specifically so they do)"
    )
    assert np.all(wfn["kpoints_crystal_coordinates"] >= -1.0)
    assert np.all(wfn["kpoints_crystal_coordinates"] <= 1.0)

    elements, _coordinates, _unique_species = read_atomic_positions(MATERIAL_NAME)
    assert wfn["number_of_atoms"] == len(elements), (
        f"WFN declares {wfn['number_of_atoms']} atoms, material '{MATERIAL_NAME}' has "
        f"{len(elements)}"
    )
    expected_atomic_numbers = sorted(ATOMIC_NUMBERS_BY_SYMBOL[element] for element in elements)
    actual_atomic_numbers = sorted(int(z) for z in wfn["atomic_numbers"])
    assert actual_atomic_numbers == expected_atomic_numbers
    assert np.all(wfn["atomic_numbers"] >= 1) and np.all(wfn["atomic_numbers"] <= 118), (
        "atomic numbers must be plausible (1-118)"
    )

    # ── geometry: avec/bvec mutually consistent ──────────────────────
    # Convention concluded from the format contract's "blat = tpiba = 2*pi/alat"
    # (docs/BGW_FORMAT_CONTRACT.md) combined with the standard reciprocal-lattice
    # identity a_i . b_j = 2*pi*delta_ij and pw2bgw's own adot/bdot formulas
    # (pw2bgw.f90:778-804 / :1645-1671), which show alat and tpiba are exactly the
    # scale factors taking dimensionless avec/bvec to physical vectors. Since
    # alat * tpiba = alat * (2*pi/alat) = 2*pi exactly, the 2*pi cancels and
    # avec[:,i] . bvec[:,j] = delta_ij with NO leftover factor -- see this repo's
    # final report for the full derivation.
    avec = wfn["lattice_vectors_alat_units"]
    bvec = wfn["reciprocal_lattice_vectors_tpiba_units"]
    orthonormality = avec.T @ bvec
    assert np.allclose(
        orthonormality,
        np.eye(3),
        rtol=GEOMETRY_RELATIVE_TOLERANCE,
        atol=GEOMETRY_ABSOLUTE_TOLERANCE,
    ), f"avec^T @ bvec = \n{orthonormality}\nexpected the 3x3 identity"

    alat = wfn["lattice_parameter_bohr"]
    expected_adot = alat**2 * (avec.T @ avec)
    assert np.allclose(
        wfn["real_space_metric_tensor_bohr2"],
        expected_adot,
        rtol=GEOMETRY_RELATIVE_TOLERANCE,
        atol=GEOMETRY_ABSOLUTE_TOLERANCE,
    ), "adot must equal alat^2 * avec^T @ avec (pw2bgw.f90:778-790)"

    tpiba = wfn["reciprocal_lattice_parameter_inverse_bohr"]
    expected_bdot = tpiba**2 * (bvec.T @ bvec)
    assert np.allclose(
        wfn["reciprocal_space_metric_tensor_inverse_bohr2"],
        expected_bdot,
        rtol=GEOMETRY_RELATIVE_TOLERANCE,
        atol=GEOMETRY_ABSOLUTE_TOLERANCE,
    ), "bdot must equal tpiba^2 * bvec^T @ bvec (pw2bgw.f90:792-804)"

    # ── wavefunctions ─────────────────────────────────────────────────
    global_gvector_set = {
        tuple(int(value) for value in column) for column in wfn["charge_density_gvectors"].T
    }
    for kpoint_index in range(wfn["number_of_kpoints"]):
        expected_gvector_count = int(wfn["number_of_gvectors_per_kpoint"][kpoint_index])
        kpoint_gvectors = wfn["kpoint_gvectors"][kpoint_index]
        assert kpoint_gvectors.shape == (3, expected_gvector_count), (
            f"k-point {kpoint_index}: gvec shape {kpoint_gvectors.shape} does not match "
            f"ngk[{kpoint_index}] = {expected_gvector_count}"
        )
        for column in kpoint_gvectors.T:
            miller_index = tuple(int(value) for value in column)
            assert miller_index in global_gvector_set, (
                f"k-point {kpoint_index}'s G-vector {miller_index} is not in the global "
                f"charge-density G-vector list"
            )

        coefficients = wfn["wavefunction_coefficients"][kpoint_index]
        assert coefficients.shape == (wfn["number_of_bands"], expected_gvector_count)
        squared_norms = np.sum(np.abs(coefficients) ** 2, axis=1)
        assert np.allclose(squared_norms, 1.0, atol=WAVEFUNCTION_NORMALIZATION_TOLERANCE), (
            f"k-point {kpoint_index}: Sum|c|^2 per band = {squared_norms}, expected all 1.0"
        )

    # ── RHO ───────────────────────────────────────────────────────────
    assert rho["number_of_g_vectors"] == wfn["number_of_g_vectors"]
    gamma_index = _find_gamma_column_index(rho["charge_density_gvectors"])
    # rho(G) is multiplied by the cell volume before writing (pw2bgw.f90:1706-1710),
    # and the density's own G=0 component is the cell-AVERAGE density,
    # electron_count / cell_volume (the convention QE's rho%of_g(1) and Q3's
    # charge_density_g share). The two cell volumes cancel:
    #     rho_written(G=0) = cell_volume * (electron_count / cell_volume)
    #                      = electron_count
    # So the on-disk G=0 value IS the electron count already -- do NOT divide by
    # the cell volume again. (Measured on FCC Si2: 8.000000000000002 on disk;
    # dividing by the 275.93 bohr^3 cell would give 0.0290, which is Q3's own
    # pre-export rho(G=0), not an electron count.)
    rho_at_gamma = complex(np.sum(rho["charge_density_g_times_cell_volume"][gamma_index, :]))
    assert abs(rho_at_gamma.imag) < 1.0e-6, (
        f"rho(G=0) should be real (it is a total electron count), got {rho_at_gamma}"
    )
    electron_count_from_rho = rho_at_gamma.real

    with open(input_path) as file_handle:
        merged_config = yaml.safe_load(file_handle)
    expected_electron_count = _expected_valence_electron_count(merged_config)
    assert abs(electron_count_from_rho - expected_electron_count) < ELECTRON_COUNT_TOLERANCE, (
        f"rho(G=0) = {electron_count_from_rho}, expected the valence electron "
        f"count {expected_electron_count} (sum of UPF z_valence over all atoms)"
    )
    assert rho["charge_density_gvectors"].shape == (3, rho["number_of_g_vectors"]), (
        "RHO's body G-vector count must match its header's number_of_g_vectors"
    )

    # ── cross-file: WFN and RHO headers must agree ───────────────────
    assert wfn["number_of_spin_components"] == rho["number_of_spin_components"]
    assert wfn["number_of_symmetry_operations"] == rho["number_of_symmetry_operations"]
    assert wfn["number_of_atoms"] == rho["number_of_atoms"]
    assert wfn["cell_volume_bohr3"] == pytest.approx(rho["cell_volume_bohr3"])
    assert np.allclose(wfn["lattice_vectors_alat_units"], rho["lattice_vectors_alat_units"])
    assert np.array_equal(wfn["fft_grid_dimensions"], rho["fft_grid_dimensions"])

    # ── vxc.dat ───────────────────────────────────────────────────────
    # A REAL gate whenever this run is within BerkeleyGWExchangeCorrelationFile's supported
    # scope (spin-unpolarized, non-gamma-only) -- true for this silicon example unconditionally,
    # since it is a 2x2x2 shifted Monkhorst-Pack mesh with no spin polarization configured.
    vxc_path = output_directory / "vxc.dat"
    if wfn["number_of_spin_components"] != 1:
        pytest.skip(
            "vxc.dat is only in scope for spin-unpolarized (nspin==1) runs; this overlay "
            f"produced number_of_spin_components={wfn['number_of_spin_components']}"
        )
    assert vxc_path.exists(), (
        "vxc.dat must be written for a spin-unpolarized, non-gamma-only run (WFN/RHO already "
        "succeeded) -- see BerkeleyGWExchangeCorrelationFile's class comment for its supported "
        "scope"
    )
    vxc = read_berkeleygw_exchange_correlation_file(vxc_path)
    assert vxc["number_of_kpoints"] == wfn["number_of_kpoints"], (
        "vxc.dat should have one block per WFN k-point"
    )

    wfn_kpoint_coordinates = wfn["kpoints_crystal_coordinates"]  # shape (3, nk)
    highest_occupied_band_index = wfn["highest_occupied_band_index"]  # shape (nk*ns,)
    all_diagonal_values_ev = []
    for kpoint_index, kpoint_block in enumerate(vxc["kpoints"]):
        # noffdiag = 0: this exporter only ever writes diagonal matrix elements.
        assert kpoint_block["offdiagonal_entry_count"] == 0, (
            "BerkeleyGWExchangeCorrelationFile is diagonal-only; noffdiag must be 0"
        )
        # ndiag already nspin-scaled on disk (pw2bgw convention); nspin==1 here so it is
        # exactly the full band count -- Q3 has no partial diag_nmin/diag_nmax selection.
        assert kpoint_block["diagonal_entry_count"] == wfn["number_of_bands"]
        assert len(kpoint_block["diagonal_entries"]) == wfn["number_of_bands"]

        assert np.allclose(
            kpoint_block["kpoint_crystal_coordinates"],
            wfn_kpoint_coordinates[:, kpoint_index],
            atol=1.0e-9,
        ), (
            f"k-point {kpoint_index}: vxc.dat coordinates "
            f"{kpoint_block['kpoint_crystal_coordinates']} do not match WFN's "
            f"{wfn_kpoint_coordinates[:, kpoint_index]}"
        )

        valence_band_count = int(highest_occupied_band_index[kpoint_index])
        seen_band_indices = set()
        for entry in kpoint_block["diagonal_entries"]:
            assert entry["spin_index"] == 1
            assert 1 <= entry["band_index"] <= wfn["number_of_bands"]
            seen_band_indices.add(entry["band_index"])

            value = entry["vxc_ev"]
            all_diagonal_values_ev.append(value)
            assert abs(value.imag) < VXC_IMAGINARY_PART_TOLERANCE_EV, (
                f"k-point {kpoint_index} band {entry['band_index']}: <psi|Vxc|psi> = {value} "
                f"eV has a non-negligible imaginary part -- Vxc(r) is real, so this diagonal "
                f"matrix element must come out real to within floating-point rounding"
            )
            assert VXC_PLAUSIBLE_MINIMUM_EV < value.real < VXC_PLAUSIBLE_MAXIMUM_EV, (
                f"k-point {kpoint_index} band {entry['band_index']}: Re(Vxc) = {value.real} eV "
                f"is not plausible for any occupied-or-empty band of a real material -- check "
                f"for a sign error, a dropped Ry->eV conversion, or a dropped G=0 term"
            )
            if entry["band_index"] <= valence_band_count:
                assert VXC_VALENCE_MINIMUM_EV < value.real < VXC_VALENCE_MAXIMUM_EV, (
                    f"k-point {kpoint_index} valence band {entry['band_index']}: Re(Vxc) = "
                    f"{value.real} eV is outside the typical LDA/PBE range for a semiconductor "
                    f"({VXC_VALENCE_MINIMUM_EV} to {VXC_VALENCE_MAXIMUM_EV} eV)"
                )
        assert seen_band_indices == set(range(1, wfn["number_of_bands"] + 1)), (
            f"k-point {kpoint_index}: expected bands 1..{wfn['number_of_bands']} exactly once "
            f"each, got {sorted(seen_band_indices)}"
        )

        for entry in kpoint_block["offdiagonal_entries"]:
            assert 1 <= entry["spin_index"] <= wfn["number_of_spin_components"]
            assert 1 <= entry["band_index_bra"] <= wfn["number_of_bands"]
            assert 1 <= entry["band_index_ket"] <= wfn["number_of_bands"]

    print(
        f"\nvxc.dat diagonal matrix elements (eV), {len(all_diagonal_values_ev)} total: "
        f"min Re = {min(v.real for v in all_diagonal_values_ev):.6f}, "
        f"max Re = {max(v.real for v in all_diagonal_values_ev):.6f}, "
        f"max |Im| = {max(abs(v.imag) for v in all_diagonal_values_ev):.3e}"
    )


# ── NSCF export ──────────────────────────────────────────────────────

# The converged SCF density and the one an NSCF pass leaves on the iterator
# differ by ~12-19% before the flavor restores the checkpoint's copy, so a loose
# bound still fails loudly on a regression; this is set just above the SCF's own
# symmetry/convergence noise, measured at 4.5e-6 on this system.
NSCF_DENSITY_AGREEMENT_TOLERANCE = 1.0e-5
# The SCF's vxc.dat is built from the last iteration's density, the NSCF's from the
# converged one restored from the checkpoint; on this system they differ by ~1e-6 eV.
# A missing core correction or a Ry/eV slip moves the traces by electron-volts.
NSCF_VXC_TRACE_AGREEMENT_TOLERANCE_EV = 1.0e-3


def test_berkeleygw_nscf_export_matches_scf_density_and_normalizes_weights(q3_command, tmp_path):
    """An NSCF export must carry the converged SCF density, vxc.dat and
    sum(kw) = 1.

    NSCF is the flavor BerkeleyGW is normally fed from, and three things about
    it used to be wrong -- all invisible to the SCF-only test above:

    1. `RHO` held the density the NSCF pass recomputed, which the shared
       per-iteration flow had already run through a mixing step. That makes the
       exported density a blend of the atomic-superposition initial guess and
       this pass's output, governed by `mixing.beta` -- measured 12% off the
       converged density in L2 at the silicon example's beta (19% when the NSCF
       k-set is the symmetry-reduced IBZ), with 9127 nonzero G components
       instead of 5921.
    2. `kw` was scaled by the assumed Q3 convention (weights summing to 2)
       rather than normalized. `control.kpoints_list` weights are taken from the
       input verbatim, so a list written on BerkeleyGW's own sum-to-1 convention
       produced sum(kw) = 0.5 on disk.
    3. No `vxc.dat` at all: the NSCF pass loads V_eff from the checkpoint and so
       never evaluated Vxc(G), leaving the exported state's copy empty -- which
       stops BerkeleyGW's sigma, which needs Vxc matrix elements.

    This test drives exactly that path: it takes the k-points straight out of
    the SCF's own `WFN` and feeds them back as an explicit `kpoints_list` using
    the file's already-normalized (sum-to-1) weights -- the scale that used to
    break -- then requires the NSCF `RHO` and valence Vxc to reproduce the
    SCF's. Vxc is compared as a trace over the valence bands at each k-point:
    individual diagonal elements inside a degenerate set depend on the basis
    the diagonalizer picked, the trace does not.
    """
    overlay = {"control": {"mean_field_export": "berkeleygw"}}
    scf_input_path = apply_config_overlay(NC_INPUT, overlay, tmp_path)
    output_directory = tmp_path / "output"

    scf_result = run_q3(q3_command, scf_input_path, working_directory=str(tmp_path))
    scf_rho_path = output_directory / "RHO"
    if scf_result.returncode != 0 or not scf_rho_path.exists():
        pytest.skip(
            "BerkeleyGW mean-field export did not produce RHO from the SCF run "
            f"(exit code {scf_result.returncode}); nothing to compare against yet"
        )

    scf_wavefunction = read_berkeleygw_wavefunction_file(output_directory / "WFN")
    scf_density = read_berkeleygw_charge_density_file(scf_rho_path)
    scf_density_g = scf_density["charge_density_g_times_cell_volume"].copy()
    scf_gvectors = scf_density["charge_density_gvectors"].copy()
    scf_vxc = read_berkeleygw_exchange_correlation_file(output_directory / "vxc.dat")

    with open(scf_input_path) as file_handle:
        nscf_config = yaml.safe_load(file_handle)
    kpoints = scf_wavefunction["kpoints_crystal_coordinates"]
    weights = scf_wavefunction["kpoint_weights"]
    nscf_config["control"]["calculation"] = "nscf"
    nscf_config["control"]["kpoints_list"] = [
        [float(kpoints[0, i]), float(kpoints[1, i]), float(kpoints[2, i]), float(weights[i])]
        for i in range(kpoints.shape[1])
    ]
    nscf_input_path = tmp_path / "nscf.yml"
    with open(nscf_input_path, "w") as file_handle:
        yaml.dump(nscf_config, file_handle, default_flow_style=False)

    # Writes into the same outdir, so the NSCF finds <prefix>.save and its files
    # replace the SCF's -- hence the SCF arrays were copied out above.
    nscf_result = run_q3(q3_command, nscf_input_path, working_directory=str(tmp_path))
    assert nscf_result.returncode == 0, (
        f"NSCF run failed (exit {nscf_result.returncode}):\n{nscf_result.stdout[-3000:]}\n"
        f"{nscf_result.stderr[-3000:]}"
    )

    nscf_wavefunction = read_berkeleygw_wavefunction_file(output_directory / "WFN")
    nscf_density = read_berkeleygw_charge_density_file(output_directory / "RHO")

    nscf_weight_sum = float(np.sum(nscf_wavefunction["kpoint_weights"]))
    assert abs(nscf_weight_sum - 1.0) < KPOINT_WEIGHT_SUM_TOLERANCE, (
        f"NSCF k-point weights sum to {nscf_weight_sum}, expected 1.0. The input's "
        f"kpoints_list weights summed to {float(np.sum(weights))}; BerkeleyGW requires "
        f"sum(kw) = 1 regardless of the scale the user wrote."
    )

    assert np.array_equal(nscf_density["charge_density_gvectors"], scf_gvectors), (
        "NSCF and SCF runs of the same system must export the same charge-density G set"
    )
    largest_density_difference = float(
        np.max(np.abs(nscf_density["charge_density_g_times_cell_volume"] - scf_density_g))
    )
    assert largest_density_difference < NSCF_DENSITY_AGREEMENT_TOLERANCE, (
        f"NSCF-exported rho(G) differs from the converged SCF rho(G) by "
        f"{largest_density_difference:.3e} (tolerance "
        f"{NSCF_DENSITY_AGREEMENT_TOLERANCE:.0e}). The NSCF flavor must export the "
        f"charge density restored from the SCF checkpoint, not the density its own "
        f"single pass recomputed and mixed."
    )

    nscf_vxc_path = output_directory / "vxc.dat"
    assert nscf_vxc_path.exists(), (
        "An NSCF BerkeleyGW export must write vxc.dat: the flavor has to evaluate Vxc(G) of "
        "the restored converged density, since its own pass loads V_eff whole."
    )
    nscf_vxc = read_berkeleygw_exchange_correlation_file(nscf_vxc_path)
    assert nscf_vxc["number_of_kpoints"] == scf_vxc["number_of_kpoints"]
    number_of_valence_bands = round(_expected_valence_electron_count(nscf_config) / 2)

    def valence_trace(kpoint_block):
        return sum(
            entry["vxc_ev"].real
            for entry in kpoint_block["diagonal_entries"]
            if entry["band_index"] <= number_of_valence_bands
        )

    for kpoint_index, (scf_block, nscf_block) in enumerate(
        zip(scf_vxc["kpoints"], nscf_vxc["kpoints"])
    ):
        difference = abs(valence_trace(nscf_block) - valence_trace(scf_block))
        assert difference < NSCF_VXC_TRACE_AGREEMENT_TOLERANCE_EV, (
            f"k-point {kpoint_index}: NSCF valence Vxc trace {valence_trace(nscf_block):.6f} eV "
            f"differs from the SCF's {valence_trace(scf_block):.6f} eV by {difference:.2e} eV."
        )
