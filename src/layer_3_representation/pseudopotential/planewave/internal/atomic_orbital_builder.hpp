#pragma once

/**
 * AtomicOrbitalGSpaceBuilder — shared construction of atomic pseudo-orbitals in G-space.
 *
 * Builds |φ_Ilm(k+G)⟩ = i^l · S_I(k+G) · Y_lm(k+G) · χ_nl(|k+G|) for every
 * occupied atomic wavefunction (n,l) on every atom I and every m in the shell.
 * Every factor is one factor of the Rayleigh expansion
 *   e^{iq·r} = 4π Σ_lm i^l j_l(qr) Y*_lm(q̂) Y_lm(r̂)
 * applied to a spherical function centred at τ_I: i^l and Y_lm from the
 * expansion, S_I = e^{-i(k+G)·τ_I} from the displacement, χ_nl from the radial
 * transform tabulated in wavefunction_atomic.hpp. This is the construction
 * shared by two otherwise-unrelated consumers:
 *   - The SCF starting-guess (NormConservingPlanewaveWavefunctionInitializer),
 *     which pads the orbital set with random vectors up to the band count.
 *   - PDOS / Löwdin projection (layer_5_properties/projections), which needs
 *     the orbitals themselves (no padding) plus stable per-orbital labels.
 * Extracted here so both consumers call one proven implementation instead of
 * two independently-maintained copies of the same physics.
 */

#include <cmath>
#include <complex>
#include <string>
#include <vector>

#include "layer_0_core/math/interpolation/lagrange.hpp"
#include "layer_0_core/math/phase_factor.hpp"
#include "layer_0_core/math/spherical/harmonics.hpp"
#include "layer_0_core/types/init.hpp"
#include "layer_1_structure/crystal/crystal.hpp"
#include "layer_1_structure/reciprocal_space/kpoints_grid.hpp"
#include "layer_3_representation/basis/planewave/init.hpp"
#include "layer_3_representation/pseudopotential/set.hpp"

namespace q3 {

/// Identifies one atomic orbital in the ordering produced by build_atomic_orbitals_g_space().
struct AtomicOrbitalLabel {
    int atom_index;
    int species_type_index;
    std::string shell_label;     // e.g. "3D", the UPF PP_CHI label verbatim
    int angular_momentum;        // l
    int magnetic_quantum_number; // m, 0..2l in the same order as Y_lm storage (l² + m)
    /**
     * The dataset's own occupancy for this shell.
     *
     * Carried so a shell can be selected without a label: UPF makes PP_CHI's
     * label optional and PseudoDojo's PAW files omit it, leaving `l` and this
     * as the only distinguishing information. A Hubbard U applies to the
     * partially-filled valence shell, which is the one with the occupancy.
     */
    Real shell_occupation{0.0};
    /**
     * The shell's 0-based position among its species' atomic wavefunctions, as the file lists
     * them.
     */
    int shell_index{0};
};

/**
 * Total number of atomic orbitals: sum of (2l+1) over every occupied UPF PP_CHI
 * shell, over every atom.
 *
 * "Occupied" is the UPF file-format convention, not a physical statement: a
 * negative PP_CHI occupancy marks a shell that is not to be used as a starting
 * orbital, and the UPF reader writes that marker itself for any shell whose
 * radial function has zero norm. Any code reading the same file gets the same
 * count, which is why this agrees with what other plane-wave codes report.
 */
[[nodiscard]] inline int count_atomic_orbitals(const Crystal& crystal_structure,
                                               const PseudopotentialSet& pseudopotential_set) {
    int number_of_atomic_orbitals = 0;
    for (int atom_index = 0; atom_index < crystal_structure.number_of_atoms(); atom_index++) {
        int type_index = crystal_structure.species_type_index(atom_index);
        const auto& pseudopotential = pseudopotential_set[type_index];
        for (const auto& atomic_wavefunction : pseudopotential.wavefunctions_atomic) {
            if (atomic_wavefunction.occupations >= 0.0) {
                number_of_atomic_orbitals += 2 * atomic_wavefunction.angular_momentum + 1;
            }
        }
    }
    return number_of_atomic_orbitals;
}

/**
 * Per-orbital labels, in the same atom-major / shell-major / m-minor order as
 * build_atomic_orbitals_g_space() fills its output.
 */
[[nodiscard]] inline std::vector<AtomicOrbitalLabel>
enumerate_atomic_orbitals(const Crystal& crystal_structure,
                          const PseudopotentialSet& pseudopotential_set) {
    std::vector<AtomicOrbitalLabel> labels;
    labels.reserve(count_atomic_orbitals(crystal_structure, pseudopotential_set));
    for (int atom_index = 0; atom_index < crystal_structure.number_of_atoms(); atom_index++) {
        int type_index = crystal_structure.species_type_index(atom_index);
        const auto& pseudopotential = pseudopotential_set[type_index];
        const auto& shells = pseudopotential.wavefunctions_atomic;
        for (std::size_t shell_index = 0; shell_index < shells.size(); shell_index++) {
            const auto& atomic_wavefunction = shells[shell_index];
            if (atomic_wavefunction.occupations < 0.0)
                continue;
            int angular_momentum = atomic_wavefunction.angular_momentum;
            for (int magnetic_index = 0; magnetic_index < 2 * angular_momentum + 1;
                 magnetic_index++) {
                labels.push_back({atom_index,
                                  type_index,
                                  atomic_wavefunction.label,
                                  angular_momentum,
                                  magnetic_index,
                                  atomic_wavefunction.occupations,
                                  static_cast<int>(shell_index)});
            }
        }
    }
    return labels;
}

/**
 * Build G-space atomic orbitals for one k-point: orbitals[orbital_index][g_index],
 * exactly count_atomic_orbitals(...) rows (no padding), ordered identically to
 * enumerate_atomic_orbitals(). `atomic_wavefunction_table` is the q-space radial
 * table from tabulate_atomic_wavefunctions_in_q_space().
 */
inline void build_atomic_orbitals_g_space(
    int kpoint_index,
    int number_of_plane_waves,
    const std::vector<int>& g_vector_indices,
    const Crystal& crystal_structure,
    const KPointsGrid& kpoint_grid,
    const PlanewaveBasis& planewave_basis,
    const PseudopotentialSet& pseudopotential_set,
    const std::vector<std::vector<std::vector<Real>>>& atomic_wavefunction_table,
    Real q_spacing,
    int number_of_q_points,
    std::vector<std::vector<Complex>>& orbitals) {
    Real two_pi_over_lattice_parameter = crystal_structure.two_pi_over_lattice_parameter();
    Vector3D kpoint_crystal = kpoint_grid[kpoint_index].coordinates;

    const Matrix3x3& reciprocal_lattice = crystal_structure.reciprocal_lattice_vectors();
    Vector3D kpoint_cartesian;
    for (int dimension = 0; dimension < 3; dimension++) {
        kpoint_cartesian[dimension] = kpoint_crystal.x * reciprocal_lattice[0][dimension] +
                                      kpoint_crystal.y * reciprocal_lattice[1][dimension] +
                                      kpoint_crystal.z * reciprocal_lattice[2][dimension];
    }

    int maximum_angular_momentum = 0;
    for (int species_index = 0; species_index < pseudopotential_set.size(); species_index++) {
        for (const auto& atomic_wavefunction :
             pseudopotential_set[species_index].wavefunctions_atomic) {
            if (atomic_wavefunction.occupations >= 0.0) {
                maximum_angular_momentum =
                    std::max(maximum_angular_momentum, atomic_wavefunction.angular_momentum);
            }
        }
    }
    int number_of_lm_components = (maximum_angular_momentum + 1) * (maximum_angular_momentum + 1);

    std::vector<std::array<Real, 3>> gk_vectors(number_of_plane_waves);
    std::vector<Real> q_plus_g_magnitude_squared(number_of_plane_waves);
    std::vector<Real> q_plus_g_magnitude(number_of_plane_waves);

    for (int g_index = 0; g_index < number_of_plane_waves; g_index++) {
        const auto& g_vector = planewave_basis.g_vectors()[g_vector_indices[g_index]];
        gk_vectors[g_index][0] = kpoint_cartesian.x + g_vector.g_cart.x;
        gk_vectors[g_index][1] = kpoint_cartesian.y + g_vector.g_cart.y;
        gk_vectors[g_index][2] = kpoint_cartesian.z + g_vector.g_cart.z;
        q_plus_g_magnitude_squared[g_index] = gk_vectors[g_index][0] * gk_vectors[g_index][0] +
                                              gk_vectors[g_index][1] * gk_vectors[g_index][1] +
                                              gk_vectors[g_index][2] * gk_vectors[g_index][2];
        q_plus_g_magnitude[g_index] =
            std::sqrt(q_plus_g_magnitude_squared[g_index]) * two_pi_over_lattice_parameter;
    }

    std::vector<std::vector<Real>> spherical_harmonics_values;
    compute_real_spherical_harmonics_batch(number_of_lm_components,
                                           number_of_plane_waves,
                                           gk_vectors,
                                           q_plus_g_magnitude_squared,
                                           spherical_harmonics_values);

    int number_of_species = static_cast<int>(pseudopotential_set.size());
    // atomic_wavefunction_values[g_index][type_index][wavefunction_index]
    std::vector<std::vector<std::vector<Real>>> atomic_wavefunction_values(number_of_plane_waves);
    for (int g_index = 0; g_index < number_of_plane_waves; g_index++) {
        atomic_wavefunction_values[g_index].resize(0);
        for (int type_index = 0; type_index < number_of_species; type_index++) {
            int number_of_wavefunctions_for_type =
                static_cast<int>(pseudopotential_set[type_index].wavefunctions_atomic.size());
            std::vector<Real> atomic_wavefunction_values_for_type(number_of_wavefunctions_for_type,
                                                                  0.0);
            for (int wavefunction_index = 0; wavefunction_index < number_of_wavefunctions_for_type;
                 wavefunction_index++) {
                if (pseudopotential_set[type_index]
                        .wavefunctions_atomic[wavefunction_index]
                        .occupations < 0.0)
                    continue;
                atomic_wavefunction_values_for_type[wavefunction_index] =
                    lagrange_interpolation_4point(
                        atomic_wavefunction_table[type_index][wavefunction_index],
                        q_plus_g_magnitude[g_index],
                        q_spacing,
                        number_of_q_points);
            }
            atomic_wavefunction_values[g_index].push_back(
                std::move(atomic_wavefunction_values_for_type));
        }
    }

    int number_of_atomic_orbitals = count_atomic_orbitals(crystal_structure, pseudopotential_set);
    orbitals.assign(number_of_atomic_orbitals,
                    std::vector<Complex>(number_of_plane_waves, Complex(0.0, 0.0)));

    int orbital_index = 0;
    for (int atom_index = 0; atom_index < crystal_structure.number_of_atoms(); atom_index++) {
        Vector3D atomic_position_cartesian =
            crystal_structure.convert_from_crystal_to_cartesian(atom_index); // units of alat
        int type_index = crystal_structure.species_type_index(atom_index);
        const auto& pseudopotential = pseudopotential_set[type_index];

        Real phase_argument = (kpoint_cartesian.x * atomic_position_cartesian.x +
                               kpoint_cartesian.y * atomic_position_cartesian.y +
                               kpoint_cartesian.z * atomic_position_cartesian.z) *
                              constants::two_pi;
        Complex kpoint_phase = compute_phase_factor(phase_argument);

        std::vector<Complex> structure_factor(number_of_plane_waves);
        for (int g_index = 0; g_index < number_of_plane_waves; g_index++) {
            const auto& g_vector = planewave_basis.g_vectors()[g_vector_indices[g_index]];
            Real phase_g = constants::two_pi * (g_vector.g_cart.x * atomic_position_cartesian.x +
                                                g_vector.g_cart.y * atomic_position_cartesian.y +
                                                g_vector.g_cart.z * atomic_position_cartesian.z);
            structure_factor[g_index] = kpoint_phase * compute_phase_factor(phase_g);
        }

        for (int wavefunction_index = 0;
             wavefunction_index < static_cast<int>(pseudopotential.wavefunctions_atomic.size());
             wavefunction_index++) {
            if (pseudopotential.wavefunctions_atomic[wavefunction_index].occupations < 0.0)
                continue;

            int angular_momentum =
                pseudopotential.wavefunctions_atomic[wavefunction_index].angular_momentum;
            // i^l phase factor (Condon-Shortley)
            Complex angular_momentum_phase;
            switch (angular_momentum % 4) {
                case 0:
                    angular_momentum_phase = Complex(1.0, 0.0);
                    break;
                case 1:
                    angular_momentum_phase = Complex(0.0, 1.0);
                    break;
                case 2:
                    angular_momentum_phase = Complex(-1.0, 0.0);
                    break;
                default:
                    angular_momentum_phase = Complex(0.0, -1.0);
                    break;
            }

            for (int magnetic_index = 0; magnetic_index < 2 * angular_momentum + 1;
                 magnetic_index++) {
                int combined_lm_index = angular_momentum * angular_momentum + magnetic_index;
                for (int g_index = 0; g_index < number_of_plane_waves; g_index++) {
                    orbitals[orbital_index][g_index] =
                        angular_momentum_phase * structure_factor[g_index] *
                        Complex(
                            spherical_harmonics_values[g_index][combined_lm_index] *
                                atomic_wavefunction_values[g_index][type_index][wavefunction_index],
                            0.0);
                }
                orbital_index++;
            }
        }
    }
}

} // namespace q3
