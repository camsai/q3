#pragma once

/**
 * MakovPayneCorrectionCalculator — isolated (charged) system electrostatic correction.
 *
 * Corrects the spurious electrostatic self-interaction of a net-charged periodic
 * cell with its own periodic images (Makov, G. and Payne, M.C., "Periodic boundary
 * conditions in ab initio calculations," Phys. Rev. B 51, 4014 (1995)). The closed-form
 * monopole and dipole+quadrupole expressions of that paper are implemented here.
 *
 * First order (monopole/Madelung) term:
 *   E_1st = alpha_M(ibrav) * q^2 / L
 * Second order (dipole+quadrupole) term:
 *   E_2nd = -(2*pi/3) * (q*Tr[Q] - |p|^2) / L^3 * e^2
 * where:
 *   q      = Z_tot - N_electrons        (net cell charge; +1 for one electron removed)
 *   Tr[Q]  = Tr[Q_ion] - Tr[Q_el]       (trace of the second-moment tensor about x0,
 *                                        e*bohr^2)
 *   p      = p_ion - p_el               (total dipole about x0; |p|^2 in e^2*bohr^2)
 *   x0     = sum_i(Z_i * tau_i) / sum_i(Z_i)   (valence-charge-weighted ionic center)
 *   L      = alat (see the Madelung-constant note below)
 *   e^2    = 2 in Rydberg units, so the e^2/2 factor of E_1st is folded away and both
 *            terms are returned in Ry.
 * Both terms already carry the sign such that
 *   corrected_total_energy = total_energy + E_1st + E_2nd,
 * and they are reported as the "first_order" and "second_order" terms of
 * IsolatedSystemCorrectionResult.
 *
 * Quadrupole sign. The second-order term above carries the quadrupole contribution
 * with the sign opposite to the one printed in the 1995 paper's Eq. 15. That
 * discrepancy is not Q3's own reading of the paper: it is an observation also recorded
 * upstream in Quantum ESPRESSO. The sign actually in force
 * here is the one pinned by the cross-code reference values in
 * tests/validation/test_makov_payne.py; re-deriving the Eq. 15 sign from the paper is
 * an open item.
 *
 * Madelung constants. 2.8373 (SC), 2.8883 (FCC), 2.8885 (BCC) are the Madelung
 * constants of a point charge in a compensating uniform background, normalised to
 * L = Omega^(1/3) (Omega = cell volume): they are (4*pi/3)^(1/3) = 1.61199 times the
 * classic per-particle values 1.76012 / 1.79175 / 1.79186 Ry*bohr. They are applied
 * here with L = alat, which equals Omega^(1/3) only for SC -- for FCC and BCC, alat
 * exceeds Omega^(1/3) by 4^(1/3) and 2^(1/3) respectively. L = alat is the convention
 * that reproduces the reference numbers pinned in tests/validation/test_makov_payne.py
 * (FCC Si: 2.8883 / 10.3344 bohr = 0.2794840 Ry at q = +1); whether the intended
 * reading is that interoperability convention or a different definition of L is an
 * open item.
 *
 * Cubic lattices only (ibrav 1/2/3 = SC/FCC/BCC). ibrav=-3 (BCC's alternate
 * primitive-vector axes) and ibrav=0 (explicit/"free" lattice vectors) are both
 * rejected, because the Madelung-constant lookup is keyed by the ibrav index itself
 * and neither value lies in 1..3. This is stricter than Quantum ESPRESSO, which first
 * resolves ibrav=0 to a Bravais index from the lattice vectors and only then applies
 * the same 1..3 bound.
 *
 * Electronic dipole/quadrupole moments are accumulated over the dense (charge
 * density) real-space grid with minimum-image wrapping of (r - x0) into the cell
 * before use -- unwrapped ionic positions are NOT wrapped, matching Quantum ESPRESSO's
 * convention (reproducing QE's number requires reproducing whatever atomic-position
 * gauge the reference calculation used). NLCC core charge is excluded, since
 * charge_density_real is the pure valence density (core charge is a separate array
 * added only where XC needs it) -- the same choice QE makes, moments taken over the
 * valence density alone and not the core charge.
 *
 * Under band parallelism, charge_density_real holds this rank's part of the grid.
 * Grid coordinates come from FFTGrid::global_real_space_index(), and the seven electronic
 * moments are reduced within the k-point pool through the injected execution
 * policy. K-point-pool mixing has already summed the converged density across
 * pools, so no across-pool reduction belongs here.
 */

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "layer_0_core/types/constants.hpp"
#include "layer_0_core/types/init.hpp"
#include "layer_1_structure/crystal/crystal.hpp"
#include "layer_3_representation/basis/planewave/init.hpp"
#include "layer_5_properties/isolated_system_correction/planewave/init.hpp"

namespace q3 {

class MakovPayneCorrectionCalculator : public IsolatedSystemCorrectionCalculator {
  public:
    [[nodiscard]] IsolatedSystemCorrectionResult
    compute(const NormConservingPlanewaveKohnShamSCFState& scf_state) const override {
        const Crystal& crystal_structure = *scf_state.crystal_structure;
        int bravais_lattice_index = crystal_structure.bravais_lattice_index();

        Real madelung_constant = 0.0;
        if (bravais_lattice_index == 1) {
            madelung_constant = 2.8373; // simple cubic
        } else if (bravais_lattice_index == 2) {
            madelung_constant = 2.8883; // FCC
        } else if (bravais_lattice_index == 3) {
            madelung_constant = 2.8885; // BCC
        } else {
            throw std::runtime_error(
                "Makov-Payne correction is only defined for cubic lattices (QE ibrav "
                "1/2/3 = SC/FCC/BCC); got bravais_lattice_index=" +
                std::to_string(bravais_lattice_index) +
                " (0 = explicit/'free' lattice vectors, -3 = BCC's alternate-axes "
                "primitive-vector convention -- QE's own bound check rejects -3 too, "
                "not just non-cubic lattices)");
        }

        Real alat = crystal_structure.lattice_parameter();
        int number_of_atoms = crystal_structure.number_of_atoms();

        // x0 = valence-charge-weighted ionic center, in alat units (Cartesian).
        Real total_valence_charge = 0.0;
        Vector3D ionic_center_alat{0.0, 0.0, 0.0};
        for (int atom_index = 0; atom_index < number_of_atoms; atom_index++) {
            Real valence_charge =
                crystal_structure.valence_charge(crystal_structure.species_type_index(atom_index));
            total_valence_charge += valence_charge;
            ionic_center_alat +=
                valence_charge * crystal_structure.convert_from_crystal_to_cartesian(atom_index);
        }
        ionic_center_alat = ionic_center_alat / total_valence_charge;

        /**
         * Ionic dipole/second moments about x0 -- NOT minimum-image wrapped, while the
         * electronic grid sum below is. This unwrapped-ion gauge is part of what the
         * cross-code reference values in tests/validation/test_makov_payne.py pin:
         * reproducing them requires the same gauge for the atomic positions.
         */
        Vector3D ionic_dipole_alat{0.0, 0.0, 0.0};
        Vector3D ionic_quadrupole_alat{0.0, 0.0, 0.0};
        for (int atom_index = 0; atom_index < number_of_atoms; atom_index++) {
            Real valence_charge =
                crystal_structure.valence_charge(crystal_structure.species_type_index(atom_index));
            Vector3D delta =
                crystal_structure.convert_from_crystal_to_cartesian(atom_index) - ionic_center_alat;
            ionic_dipole_alat += valence_charge * delta;
            ionic_quadrupole_alat.x += valence_charge * delta.x * delta.x;
            ionic_quadrupole_alat.y += valence_charge * delta.y * delta.y;
            ionic_quadrupole_alat.z += valence_charge * delta.z * delta.z;
        }
        Vector3D ionic_dipole = ionic_dipole_alat * alat;
        Vector3D ionic_quadrupole = ionic_quadrupole_alat * (alat * alat);

        /**
         * Electronic dipole/quadrupole: dense real-space grid sum, minimum-image
         * wrapping (r - x0) into the cell via fractional coordinates before use.
         */
        const FFTGrid& density_grid = scf_state.planewave_basis->fft_rho();
        int grid_dimension_1 = density_grid.grid_dimension_1();
        int grid_dimension_2 = density_grid.grid_dimension_2();
        int grid_dimension_3 = density_grid.grid_dimension_3();
        int total_grid_points = density_grid.total_grid_points();
        int local_grid_points = density_grid.local_grid_points();

        Matrix3x3 lattice_vectors = crystal_structure.lattice_vectors(); // alat units
        Matrix3x3 lattice_vectors_transpose = lattice_vectors.transpose();
        Matrix3x3 lattice_inverse_transpose = lattice_vectors.inverse().transpose();

        int number_of_spin_components = static_cast<int>(scf_state.charge_density_real.size());

        Real electron_count_raw = 0.0;
        Vector3D electronic_dipole_raw{0.0, 0.0, 0.0};
        Vector3D electronic_quadrupole_raw{0.0, 0.0, 0.0};

        /**
         * One sweep over this rank's part of the dense grid, run once post-SCF.
         * The grid translates each local storage index to the global flat index
         * before the fractional coordinate is derived. Serial execution visits
         * the complete grid with the two indices equal, preserving its original
         * arithmetic.
         */
        for (int local_flat_index = 0; local_flat_index < local_grid_points; local_flat_index++) {
            int global_flat_index = density_grid.global_real_space_index(local_flat_index);
            int index_3 = global_flat_index % grid_dimension_3;
            int index_2 = (global_flat_index / grid_dimension_3) % grid_dimension_2;
            int index_1 = global_flat_index / (grid_dimension_3 * grid_dimension_2);
            Vector3D fractional_grid_point{static_cast<Real>(index_1) / grid_dimension_1,
                                           static_cast<Real>(index_2) / grid_dimension_2,
                                           static_cast<Real>(index_3) / grid_dimension_3};
            Vector3D grid_point_alat = lattice_vectors_transpose * fractional_grid_point;
            Vector3D displacement = grid_point_alat - ionic_center_alat;

            Vector3D fractional_displacement = lattice_inverse_transpose * displacement;
            fractional_displacement.x -= std::round(fractional_displacement.x);
            fractional_displacement.y -= std::round(fractional_displacement.y);
            fractional_displacement.z -= std::round(fractional_displacement.z);
            Vector3D wrapped_displacement = lattice_vectors_transpose * fractional_displacement;

            Real total_density_at_point = 0.0;
            for (int spin_index = 0; spin_index < number_of_spin_components; spin_index++) {
                total_density_at_point +=
                    scf_state.charge_density_real[spin_index][local_flat_index];
            }

            electron_count_raw += total_density_at_point;
            electronic_dipole_raw += total_density_at_point * wrapped_displacement;
            electronic_quadrupole_raw.x +=
                total_density_at_point * wrapped_displacement.x * wrapped_displacement.x;
            electronic_quadrupole_raw.y +=
                total_density_at_point * wrapped_displacement.y * wrapped_displacement.y;
            electronic_quadrupole_raw.z +=
                total_density_at_point * wrapped_displacement.z * wrapped_displacement.z;
        }

        std::vector<Real> electronic_moments = {
            electron_count_raw,
            electronic_dipole_raw.x,
            electronic_dipole_raw.y,
            electronic_dipole_raw.z,
            electronic_quadrupole_raw.x,
            electronic_quadrupole_raw.y,
            electronic_quadrupole_raw.z,
        };
        if (scf_state.execution_policy != nullptr) {
            scf_state.execution_policy->reduce_within_pool_vector(electronic_moments);
        }
        electron_count_raw = electronic_moments[0];
        electronic_dipole_raw =
            Vector3D{electronic_moments[1], electronic_moments[2], electronic_moments[3]};
        electronic_quadrupole_raw =
            Vector3D{electronic_moments[4], electronic_moments[5], electronic_moments[6]};

        Real grid_sum_to_integral_scale =
            crystal_structure.cell_volume() / static_cast<Real>(total_grid_points);
        Real electron_count = electron_count_raw * grid_sum_to_integral_scale;
        Vector3D electronic_dipole = electronic_dipole_raw * grid_sum_to_integral_scale * alat;
        Vector3D electronic_quadrupole =
            electronic_quadrupole_raw * grid_sum_to_integral_scale * (alat * alat);

        Real net_charge = total_valence_charge - electron_count;
        Vector3D dipole = ionic_dipole - electronic_dipole;
        Real quadrupole_trace = (ionic_quadrupole.x - electronic_quadrupole.x) +
                                (ionic_quadrupole.y - electronic_quadrupole.y) +
                                (ionic_quadrupole.z - electronic_quadrupole.z);
        Real dipole_norm_squared = dipole.x * dipole.x + dipole.y * dipole.y + dipole.z * dipole.z;

        Real e2 = constants::electron_charge_squared_rydberg;

        Real first_order_in_ry = madelung_constant * net_charge * net_charge / alat;
        Real second_order_in_ry = -(2.0 / 3.0 * constants::pi) *
                                  (net_charge * quadrupole_trace - dipole_norm_squared) /
                                  (alat * alat * alat) * e2;

        IsolatedSystemCorrectionResult result;
        result.applied = true;
        result.method = name();
        result.terms_in_ry = {{"first_order", first_order_in_ry},
                              {"second_order", second_order_in_ry}};
        return result;
    }

    [[nodiscard]] std::string name() const override {
        return "makov-payne";
    }
};

} // namespace q3
