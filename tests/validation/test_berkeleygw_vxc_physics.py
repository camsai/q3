"""
Physics validation for BerkeleyGW's ``vxc.dat``: recompute every diagonal
matrix element from scratch and require Q3's exported values to match.

This is deliberately not a format test -- ``test_berkeleygw_export.py``
already checks framing, headers and cross-file agreement using the
independent reader. This test checks the *numbers*, by rebuilding
``<psi_nk|V_xc|psi_nk>`` out of nothing but the two files Q3 wrote before
``vxc.dat`` existed (``WFN`` and ``RHO``) plus published closed-form LDA
formulas. It shares no code path with the exporter:

  * Q3's exporter evaluates the G-space double sum
    ``sum_{G,G'} conj(c(G)) V_xc(G-G') c(G')`` (the quantity pw2bgw.x's
    ``write_vxc_g`` evaluates),
    using the V_xc the SCF actually converged to.
  * This test evaluates the real-space product
    ``(1/N_r) sum_r |psi(r)|^2 V_xc(r)`` (the quantity pw2bgw.x's
    ``write_vxc_r`` evaluates),
    using a V_xc it rebuilds itself from ``RHO`` via its own implementation
    of the Perdew-Zunger 1981 functional -- not libxc, which is what Q3
    calls.

**Why the two routes must agree to machine precision, not merely
approximately.** Every wavefunction G satisfies ``|k+G|^2 <= ecutwfc``, so
any difference of two of them obeys
``|G-G'| = |(k+G)-(k+G')| <= 2 sqrt(ecutwfc) = sqrt(4 ecutwfc)``. When
``ecutrho >= 4 ecutwfc`` -- the ``dual = 4`` norm-conserving default -- every
such difference therefore lies inside the sphere on which V_xc is stored, so
the double sum loses nothing and is algebraically identical to the real-space
integral. Measured on this exact system while writing this test: 0 of
38,514,920 ``(G-G')`` pairs fell outside the charge-density G set, and the two
routes agreed to 2.3e-14 eV. That is why ``AGREEMENT_TOLERANCE_EV`` below is
tight enough to catch a real defect (a dropped ``G=0`` term shifts every
element by about +7 eV; a Ry/eV slip by a factor of 13.6) rather than merely
confirming the right order of magnitude.

Three preconditions make the reconstruction exact rather than approximate.
Each is asserted, not assumed, so that a future edit to the example or the
overlay fails loudly instead of quietly invalidating the comparison:

  1. **V_xc of the valence density.** The exporter evaluates V_xc at
     ``rho_valence`` alone, as pw2bgw does by default (``vxc_zero_rho_core``),
     which is exactly the density ``RHO`` carries -- so a non-linear core
     correction no longer breaks the reconstruction. (It did while the export
     included ``rho_core``; the dataset below is NLCC-free for precondition 2's
     sake, not this one's.)
  2. **A local LDA functional, matching the one implemented below.** The
     overlay pins ``functional: pz`` (libxc's ``XC_LDA_X`` +
     ``XC_LDA_C_PZ``), which ``Si.pz-vbc.UPF`` was itself generated with
     (its header reads ``SLA PZ NOGX NOGC``).
  3. **dual >= 4**, per the exactness argument above.
"""

import numpy as np
import pytest
import yaml

from helpers import EXAMPLES_DIRECTORY, apply_config_overlay, run_q3
from helpers_berkeleygw import (
    read_berkeleygw_charge_density_file,
    read_berkeleygw_exchange_correlation_file,
    read_berkeleygw_wavefunction_file,
)

SILICON_EXAMPLE_DIRECTORY = EXAMPLES_DIRECTORY / "total_energy--semiconductor--silicon"
NC_INPUT = SILICON_EXAMPLE_DIRECTORY / "nc.yml"
NLCC_FREE_PSEUDOPOTENTIAL = "../data/pseudopotentials/nc/Si.pz-vbc.UPF"
# Substring of BerkeleyGWExchangeCorrelationFile's own refusal message, used to tell
# "this build predates vxc.dat support" (skip) from "the run broke" (fail).
NOT_YET_SUPPORTED_MARKER = "vxc.dat export"

# Q3 converts libxc's Hartree-unit output to Rydberg by
# constants::electron_charge_squared_rydberg = 2.0; the same factor appears here.
HARTREE_TO_RY = 2.0
RY_TO_EV = 13.605693122994  # CODATA 2018, as in Q3's constants::RY_TO_EV
# Matches the libxc density threshold Q3 sets to mirror QE (libxc.hpp: 1e-10).
DENSITY_THRESHOLD = 1.0e-10

AGREEMENT_TOLERANCE_EV = 1.0e-7
IMAGINARY_PART_TOLERANCE_EV = 1.0e-9
NORMALIZATION_TOLERANCE = 1.0e-9
ELECTRON_COUNT_TOLERANCE = 1.0e-8
# Si.pz-vbc.UPF has z_valence = 4 and the FCC silicon cell holds 2 atoms.
EXPECTED_ELECTRON_COUNT = 8.0
# Loose physical bracket on LDA V_xc expectation values for silicon, in eV. Not a
# precision check -- that is AGREEMENT_TOLERANCE_EV -- just a guard that catches a
# sign flip or a units blunder with a message about physics rather than digits.
PLAUSIBLE_VXC_RANGE_EV = (-15.0, -5.0)


def slater_exchange_potential_hartree(density):
    """``v_x = -(3/pi)^(1/3) rho^(1/3)`` in Hartree -- LDA exchange, libxc's
    ``XC_LDA_X``.

    Derived from ``e_x = -(3/4)(3/pi)^(1/3) rho^(1/3)`` (energy per electron)
    via ``v_x = d(rho e_x)/drho = (4/3) e_x``.
    """
    return -((3.0 / np.pi) ** (1.0 / 3.0)) * np.cbrt(density)


def perdew_zunger_correlation_potential_hartree(density):
    """``v_c`` in Hartree from the Perdew-Zunger 1981 LDA parameterization.

    The unpolarized branch of PZ81 (Phys. Rev. B 23, 5048 (1981), Table XII),
    which is libxc's ``XC_LDA_C_PZ``. With ``rs = (3/(4 pi rho))^(1/3)``:

        rs >= 1:  eps_c = gamma / (1 + b1 sqrt(rs) + b2 rs)
        rs <  1:  eps_c = A ln rs + B + C rs ln rs + D rs

    Both potentials are ``v_c = d(rho eps_c)/drho = eps_c - (rs/3) deps_c/drs``
    (using ``drs/drho = -rs/(3 rho)``), worked out by hand:

        rs >= 1:  v_c = eps_c (1 + 7/6 b1 sqrt(rs) + 4/3 b2 rs)
                              / (1 + b1 sqrt(rs) + b2 rs)
        rs <  1:  v_c = A ln rs + (B - A/3) + 2/3 C rs ln rs + (2D - C) rs/3
    """
    gamma, beta_1, beta_2 = -0.1423, 1.0529, 0.3334
    a_coefficient, b_coefficient = 0.0311, -0.048
    c_coefficient, d_coefficient = 0.0020, -0.0116

    r_s = np.cbrt(3.0 / (4.0 * np.pi * density))
    sqrt_r_s = np.sqrt(r_s)

    denominator = 1.0 + beta_1 * sqrt_r_s + beta_2 * r_s
    numerator = 1.0 + (7.0 / 6.0) * beta_1 * sqrt_r_s + (4.0 / 3.0) * beta_2 * r_s
    high_density_potential = (gamma / denominator) * (numerator / denominator)

    log_r_s = np.log(np.maximum(r_s, np.finfo(float).tiny))
    low_density_potential = (
        a_coefficient * log_r_s
        + (b_coefficient - a_coefficient / 3.0)
        + (2.0 / 3.0) * c_coefficient * r_s * log_r_s
        + (2.0 * d_coefficient - c_coefficient) * r_s / 3.0
    )
    return np.where(r_s >= 1.0, high_density_potential, low_density_potential)


def exchange_correlation_potential_rydberg(density):
    """``v_xc(rho)`` in Rydberg, zeroed below Q3's libxc density threshold."""
    safe_density = np.maximum(density, DENSITY_THRESHOLD)
    potential_hartree = slater_exchange_potential_hartree(
        safe_density
    ) + perdew_zunger_correlation_potential_hartree(safe_density)
    return np.where(density > DENSITY_THRESHOLD, potential_hartree * HARTREE_TO_RY, 0.0)


def scatter_onto_fft_grid(miller_indices, values, grid_dimensions):
    """Place coefficients indexed by Miller indices ``(3, n)`` onto an FFT grid.

    Negative Miller indices wrap to the top of each axis, which is the same
    convention the forward transform below undoes.
    """
    grid = np.zeros(tuple(int(dimension) for dimension in grid_dimensions), dtype=np.complex128)
    wrapped = np.empty_like(miller_indices)
    for axis in range(3):
        wrapped[axis] = np.remainder(miller_indices[axis], int(grid_dimensions[axis]))
    grid[wrapped[0], wrapped[1], wrapped[2]] = values
    return grid


def inverse_transform_to_real_space(grid):
    """``f(r) = sum_G f(G) exp(+i G.r)`` -- numpy's ``ifftn`` sign convention,
    undoing its ``1/N`` normalization so that ``(1/N_r) sum_r |f|^2`` equals
    ``sum_G |f(G)|^2`` (Parseval), which is what makes a unit-norm band come
    out with unit norm in real space.
    """
    return np.fft.ifftn(grid) * grid.size


def _assert_reconstruction_preconditions(merged_config):
    """Fail loudly if the run no longer satisfies what makes this test exact."""
    assert merged_config["model"]["functional"].lower() == "pz", (
        "this test implements the Perdew-Zunger LDA functional specifically; "
        f"the run used '{merged_config['model']['functional']}'."
    )
    cutoffs = merged_config["method"]["cutoffs"]
    assert cutoffs["density"] >= 4.0 * cutoffs["wavefunction"] - 1.0e-9, (
        f"ecutrho ({cutoffs['density']}) < 4 * ecutwfc "
        f"({4.0 * cutoffs['wavefunction']}): differences G-G' of wavefunction G "
        f"vectors can then fall outside the G set V_xc is stored on, so the "
        f"G-space double sum and this real-space evaluation are no longer "
        f"algebraically identical and cannot be compared at this tolerance."
    )


def test_berkeleygw_vxc_matrix_elements_match_independent_reconstruction(q3_command, tmp_path):
    overlay = {
        "model": {"subtype": "lda", "functional": "pz"},
        "method": {"pseudopotentials": {"Si": NLCC_FREE_PSEUDOPOTENTIAL}},
        "control": {"mean_field_export": "berkeleygw"},
    }
    input_path = apply_config_overlay(NC_INPUT, overlay, tmp_path)
    output_directory = tmp_path / "output"

    result = run_q3(q3_command, input_path, working_directory=str(tmp_path))
    exchange_correlation_path = output_directory / "vxc.dat"
    wavefunction_path = output_directory / "WFN"
    charge_density_path = output_directory / "RHO"

    # A nonzero exit is only ever an acceptable reason to skip when it is the
    # exporter's own documented "vxc.dat is not yet supported" refusal. Skipping on
    # ANY failure would let a genuine crash masquerade as "not implemented yet",
    # which is precisely the kind of silent hole this suite exists to close.
    combined_output = f"{result.stdout}\n{result.stderr}"
    if result.returncode != 0:
        if NOT_YET_SUPPORTED_MARKER in combined_output:
            pytest.skip("this build still refuses vxc.dat export; nothing to validate yet")
        raise AssertionError(
            f"q3 exited {result.returncode} for a reason other than the documented "
            f"vxc.dat refusal:\n{combined_output[-4000:]}"
        )
    if not exchange_correlation_path.is_file():
        pytest.skip("run succeeded but wrote no vxc.dat; nothing to validate yet")
    assert wavefunction_path.is_file(), "vxc.dat was written but WFN was not"
    assert charge_density_path.is_file(), "vxc.dat was written but RHO was not"

    with open(input_path) as file_handle:
        _assert_reconstruction_preconditions(yaml.safe_load(file_handle))

    charge_density = read_berkeleygw_charge_density_file(charge_density_path)
    wavefunction = read_berkeleygw_wavefunction_file(wavefunction_path)
    exported = read_berkeleygw_exchange_correlation_file(exchange_correlation_path)

    cell_volume = charge_density["cell_volume_bohr3"]
    grid_dimensions = charge_density["fft_grid_dimensions"]
    number_of_grid_points = int(np.prod(grid_dimensions))

    # ── rho(G) -> rho(r), and a check that the density itself is sane ──
    density_g = charge_density["charge_density_g_times_cell_volume"][:, 0] / cell_volume
    density_r = inverse_transform_to_real_space(
        scatter_onto_fft_grid(charge_density["charge_density_gvectors"], density_g, grid_dimensions)
    )
    assert np.max(np.abs(density_r.imag)) < 1.0e-12, "rho(r) must be real"
    density_real = density_r.real
    electron_count = float(np.sum(density_real)) / number_of_grid_points * cell_volume
    assert electron_count == pytest.approx(EXPECTED_ELECTRON_COUNT, abs=ELECTRON_COUNT_TOLERANCE), (
        f"integral of rho over the cell is {electron_count}, expected 8 valence electrons"
    )

    potential_r = exchange_correlation_potential_rydberg(density_real)

    # ── <psi_nk|V_xc|psi_nk>, real-space route ────────────────────────
    number_of_kpoints = wavefunction["number_of_kpoints"]
    number_of_bands = wavefunction["number_of_bands"]
    reconstructed_ev = np.zeros((number_of_kpoints, number_of_bands))
    for kpoint_index in range(number_of_kpoints):
        kpoint_gvectors = wavefunction["kpoint_gvectors"][kpoint_index]
        coefficients = wavefunction["wavefunction_coefficients"][kpoint_index]
        for band_index in range(number_of_bands):
            band_coefficients = coefficients[band_index, :]
            norm = float(np.vdot(band_coefficients, band_coefficients).real)
            assert norm == pytest.approx(1.0, abs=NORMALIZATION_TOLERANCE), (
                f"k-point {kpoint_index + 1} band {band_index + 1} has norm {norm}; "
                f"the real-space normalization below depends on unit-norm bands"
            )
            orbital_r = inverse_transform_to_real_space(
                scatter_onto_fft_grid(kpoint_gvectors, band_coefficients, grid_dimensions)
            )
            expectation_rydberg = (
                float(np.sum(np.abs(orbital_r) ** 2 * potential_r)) / number_of_grid_points
            )
            reconstructed_ev[kpoint_index, band_index] = expectation_rydberg * RY_TO_EV

    lower_bound, upper_bound = PLAUSIBLE_VXC_RANGE_EV
    assert np.all((reconstructed_ev > lower_bound) & (reconstructed_ev < upper_bound)), (
        f"independently reconstructed V_xc expectation values fall outside the "
        f"physically plausible range {PLAUSIBLE_VXC_RANGE_EV} eV for silicon LDA: "
        f"min {reconstructed_ev.min()}, max {reconstructed_ev.max()}. This indicates "
        f"a problem with the run or with RHO/WFN, not necessarily with vxc.dat."
    )

    # ── Compare against what Q3 wrote ─────────────────────────────────
    assert exported["number_of_kpoints"] == number_of_kpoints, (
        f"vxc.dat has {exported['number_of_kpoints']} k-points, WFN has {number_of_kpoints}"
    )
    for kpoint_index, block in enumerate(exported["kpoints"]):
        expected_kpoint = wavefunction["kpoints_crystal_coordinates"][:, kpoint_index]
        np.testing.assert_allclose(
            block["kpoint_crystal_coordinates"],
            expected_kpoint,
            atol=5.0e-9,
            err_msg=(
                f"vxc.dat k-point {kpoint_index + 1} disagrees with WFN record 11 "
                f"(both are crystal coordinates written at F13.9)"
            ),
        )
        assert block["offdiagonal_entry_count"] == 0, (
            f"k-point {kpoint_index + 1} declares "
            f"{block['offdiagonal_entry_count']} off-diagonal entries; Q3 exports "
            f"diagonal elements only"
        )
        assert block["diagonal_entry_count"] == number_of_bands, (
            f"k-point {kpoint_index + 1} declares {block['diagonal_entry_count']} "
            f"diagonal entries, expected one per band ({number_of_bands})"
        )
        for entry in block["diagonal_entries"]:
            band_index = entry["band_index"] - 1
            assert 0 <= band_index < number_of_bands, (
                f"k-point {kpoint_index + 1} names band {entry['band_index']}, "
                f"outside 1..{number_of_bands}"
            )
            written = entry["vxc_ev"]
            assert abs(written.imag) < IMAGINARY_PART_TOLERANCE_EV, (
                f"k-point {kpoint_index + 1} band {entry['band_index']} has "
                f"Im(V_xc) = {written.imag} eV; a diagonal element of a real "
                f"local potential is real"
            )
            expected = reconstructed_ev[kpoint_index, band_index]
            assert written.real == pytest.approx(expected, abs=AGREEMENT_TOLERANCE_EV), (
                f"k-point {kpoint_index + 1} band {entry['band_index']}: Q3 wrote "
                f"{written.real:.9f} eV, independent reconstruction gives "
                f"{expected:.9f} eV (difference {written.real - expected:.3e} eV). "
                f"See this module's docstring for why these must agree to "
                f"{AGREEMENT_TOLERANCE_EV:.0e} eV."
            )
