#pragma once

/**
 * Nonlocal (Kleinman-Bylander) pseudopotential stress tensor contribution.
 *
 * Computes the nonlocal pseudopotential stress using KB projectors and their
 * derivatives with respect to strain (Bessel and spherical harmonic components).
 *
 * For norm-conserving pseudopotential_set, the bare D matrix (Dion) is used.
 * For ultrasoft pseudopotential_set, the effective D matrix (Deeq) is used,
 * and an eigenvalue × q_ij correction is subtracted.
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <string>
#include <vector>

#include "layer_0_core/math/linear_algebra/blas.hpp"
#include "layer_0_core/math/phase_factor.hpp"
#include "layer_0_core/math/spherical/harmonics.hpp"
#include "layer_0_core/math/tolerances.hpp"
#include "layer_0_core/types/init.hpp"
#include "layer_0_core/types/matrix.hpp"
#include "layer_1_structure/crystal/crystal.hpp"
#include "layer_1_structure/reciprocal_space/kpoints_grid.hpp"
#include "layer_3_representation/basis/planewave/init.hpp"
#include "layer_3_representation/pseudopotential/planewave/internal/projector.hpp"
#include "layer_3_representation/quantum_system/kohn_sham/planewave/norm_conserving/init.hpp"
#include "layer_4_computation/state/kohn_sham/planewave/norm_conserving/init.hpp"
#include "layer_5_properties/stress_tensor/planewave/contributions.hpp"

namespace q3 {
using namespace q3::blas;

class NonlocalPseudopotentialStressTensorCalculator {
  public:
    NonlocalPseudopotentialStressTensorCalculator(
        const Crystal& crystal_structure,
        const PlanewaveBasis& planewave_basis,
        const KPointsGrid& kpoint_grid,
        const NormConservingPlanewaveKohnShamSCFState& scf_state,
        StressTensorContributions& result,
        const std::string& derivative_method = "analytical")
        : crystal_structure_(crystal_structure)
        , planewave_basis_(planewave_basis)
        , kpoint_grid_(kpoint_grid)
        , scf_state_(scf_state)
        , result_(result)
        , derivative_method_(derivative_method) {}

    void compute() {
        bool use_effective_d_matrix = scf_state_.quantum_system().has_ultrasoft();
        const Matrix3x3& reciprocal_lattice_vectors =
            crystal_structure_.reciprocal_lattice_vectors();
        const auto& pseudopotential_basis = scf_state_.quantum_system().pseudopotential_basis();

        int maximum_angular_momentum = 0;
        int largest_atom_projector_count = 0;
        for (int atom_index = 0; atom_index < crystal_structure_.number_of_atoms(); atom_index++) {
            int species_type_index = crystal_structure_.species_type_index(atom_index);
            for (int beta_index = 0;
                 beta_index < pseudopotential_basis.number_of_projectors(species_type_index);
                 beta_index++) {
                maximum_angular_momentum = std::max(
                    maximum_angular_momentum,
                    pseudopotential_basis.beta_angular_momentum(species_type_index, beta_index));
            }
            largest_atom_projector_count =
                std::max(largest_atom_projector_count,
                         pseudopotential_basis.number_of_projector_channels(species_type_index));
        }
        int number_of_lm_components =
            (maximum_angular_momentum + 1) * (maximum_angular_momentum + 1);

        int number_of_kpoints = kpoint_grid_.number_of_kpoints();
        /**
         * kpoint_state_index runs over number_of_kpoints * number_of_spin_components "quantum
         * states" (spin-major: spin_index * number_of_kpoints + kpoint_index, matching
         * PlanewaveKohnShamQuantumSystem's storage convention) — looping kpoint_index alone
         * silently drops the second spin channel for LSDA (nspin=2). kpoint_grid_[kpoint_index]
         * itself is NOT spin-doubled: same physical k-point and already spin-halved weight for
         * both channels.
         */
        for (int spin_index = 0; spin_index < scf_state_.number_of_spin_components();
             spin_index++) {
            for (int kpoint_index = 0; kpoint_index < number_of_kpoints; kpoint_index++) {
                int kpoint_state_index = spin_index * number_of_kpoints + kpoint_index;
                /**
                 * number_of_plane_waves/g_vector_indices_at_kpoint are PlanewaveBasis-derived
                 * (the G-vector set for a k-point does not depend on spin) — physical
                 * kpoint_index. occupation/wavefunction_coefficient/eigenvalue are
                 * QuantumSystem-derived (per spin channel) — spin-doubled kpoint_state_index.
                 */
                int number_of_plane_waves_kpoint = scf_state_.number_of_plane_waves(kpoint_index);
                const auto& plane_wave_indices_kpoint =
                    scf_state_.g_vector_indices_at_kpoint(kpoint_index);
                Vector3D kpoint_coordinates_crystal = kpoint_grid_[kpoint_index].coordinates;
                Real kpoint_weight = kpoint_grid_[kpoint_index].weight;

                Vector3D kpoint_coordinates_cartesian = crystal_to_cartesian_kpoint(
                    kpoint_coordinates_crystal, reciprocal_lattice_vectors);

                // Collect occupied band indices and weights
                std::vector<int> occupied_band_indices;
                std::vector<Real> occupied_band_weights;
                for (int band_index = 0; band_index < scf_state_.number_of_bands(); band_index++) {
                    Real occupation = scf_state_.occupation(kpoint_state_index, band_index);
                    if (occupation < math::tolerances::occupation_threshold)
                        continue;
                    occupied_band_indices.push_back(band_index);
                    occupied_band_weights.push_back(kpoint_weight * occupation);
                }
                int number_of_occupied_bands = static_cast<int>(occupied_band_indices.size());
                if (number_of_occupied_bands == 0)
                    continue;

                // Compute k+G vectors in Cartesian coordinates
                std::vector<Real> kpoint_plus_g_x, kpoint_plus_g_y, kpoint_plus_g_z;
                std::vector<Real> kpoint_plus_g_norm_inverse;
                compute_kpoint_plus_g_vectors(kpoint_coordinates_cartesian,
                                              plane_wave_indices_kpoint,
                                              number_of_plane_waves_kpoint,
                                              kpoint_plus_g_x,
                                              kpoint_plus_g_y,
                                              kpoint_plus_g_z,
                                              kpoint_plus_g_norm_inverse);

                /**
                 * The k+G geometry and its spherical harmonics do not depend on the
                 * atom, so they are computed once per k-point. Only the structure
                 * factor does, and it is formed per atom as that atom is reached.
                 * This used to be a 56-byte record per projector per plane wave --
                 * on the 54-atom silicon cell over a gigabyte of the same numbers.
                 */
                KpointPlusGGeometry geometry;
                compute_kpoint_plus_g_geometry(kpoint_coordinates_cartesian,
                                               plane_wave_indices_kpoint,
                                               number_of_plane_waves_kpoint,
                                               number_of_lm_components,
                                               geometry);

                // Collect occupied wavefunctions into contiguous matrix (NPW × Nocc)
                std::vector<Complex> wavefunction_matrix(
                    static_cast<std::size_t>(number_of_plane_waves_kpoint) *
                    number_of_occupied_bands);
                for (int occupied_index = 0; occupied_index < number_of_occupied_bands;
                     occupied_index++) {
                    int band_index = occupied_band_indices[occupied_index];
                    for (int g_index = 0; g_index < number_of_plane_waves_kpoint; g_index++) {
                        wavefunction_matrix[static_cast<std::size_t>(occupied_index) *
                                                number_of_plane_waves_kpoint +
                                            g_index] =
                            scf_state_.wavefunction_coefficient(
                                kpoint_state_index, band_index, g_index);
                    }
                }

                // Precompute eigenvalues for occupied bands (needed for US)
                std::vector<Real> occupied_eigenvalues(number_of_occupied_bands);
                if (use_effective_d_matrix) {
                    for (int band_n = 0; band_n < number_of_occupied_bands; band_n++) {
                        occupied_eigenvalues[band_n] = scf_state_.eigenvalue(
                            kpoint_state_index, occupied_band_indices[band_n]);
                    }
                }

                int g_zero_plane_wave_index =
                    planewave_basis_.is_gamma_only()
                        ? find_g_zero_plane_wave_index(plane_wave_indices_kpoint,
                                                       number_of_plane_waves_kpoint)
                        : -1;

                /**
                 * The D matrix couples projectors of the same atom only, so every
                 * term below is a sum over atoms of an atom's own projectors. The
                 * cell is therefore worked through in blocks of consecutive atoms,
                 * each holding its projectors and their six strain derivatives for
                 * as long as it takes to contract them with the occupied bands. The
                 * whole cell at once is what took the 54-atom silicon cell from
                 * 2.3 GiB through the SCF to over 4 GiB here.
                 *
                 * A block carries an eighth as many projectors as there are occupied
                 * bands, so its seven arrays stay below the occupied wavefunctions,
                 * but never fewer than 64: every contraction copies the wavefunctions
                 * into the BLAS's own buffers, and with a handful of projectors per
                 * call that copy, not the product, is the cost. Measured with
                 * Accelerate on this cell, 18 rows per call took five times as long
                 * per projector as the whole cell in one call, and 378 rows were
                 * within 10% of it.
                 */
                constexpr int minimum_projectors_per_block = 64;
                int projectors_per_block =
                    std::max(largest_atom_projector_count,
                             std::max(number_of_occupied_bands / 8, minimum_projectors_per_block));
                BlockWorkspace workspace;
                workspace.allocate(projectors_per_block, number_of_plane_waves_kpoint);

                std::array<std::vector<Real>, 6> radial_strain_weights;
                const std::vector<Real>* kpoint_plus_g_components[3] = {
                    &kpoint_plus_g_x, &kpoint_plus_g_y, &kpoint_plus_g_z};
                for (int strain_pair = 0; strain_pair < 6; strain_pair++) {
                    const auto& component_a =
                        *kpoint_plus_g_components[bessel_direction_a[strain_pair]];
                    const auto& component_b =
                        *kpoint_plus_g_components[bessel_direction_b[strain_pair]];
                    radial_strain_weights[strain_pair].resize(number_of_plane_waves_kpoint);
                    for (int g_index = 0; g_index < number_of_plane_waves_kpoint; g_index++) {
                        radial_strain_weights[strain_pair][g_index] =
                            component_a[g_index] * component_b[g_index] *
                            kpoint_plus_g_norm_inverse[g_index];
                    }
                }

                int block_first_atom = 0;
                while (block_first_atom < crystal_structure_.number_of_atoms()) {
                    int block_end_atom = block_first_atom;
                    int block_projectors = 0;
                    while (block_end_atom < crystal_structure_.number_of_atoms()) {
                        int atom_projectors = pseudopotential_basis.number_of_projector_channels(
                            crystal_structure_.species_type_index(block_end_atom));
                        if (block_projectors > 0 &&
                            block_projectors + atom_projectors > projectors_per_block) {
                            break;
                        }
                        block_projectors += atom_projectors;
                        block_end_atom++;
                    }
                    if (block_projectors > 0) {
                        accumulate_block(spin_index,
                                         kpoint_coordinates_cartesian,
                                         plane_wave_indices_kpoint,
                                         number_of_plane_waves_kpoint,
                                         geometry,
                                         kpoint_plus_g_components,
                                         radial_strain_weights,
                                         wavefunction_matrix,
                                         number_of_occupied_bands,
                                         occupied_band_weights,
                                         occupied_eigenvalues,
                                         g_zero_plane_wave_index,
                                         use_effective_d_matrix,
                                         block_first_atom,
                                         block_end_atom,
                                         block_projectors,
                                         workspace);
                    }
                    block_first_atom = block_end_atom;
                }
            }
        }

        symmetrize_stress_tensor(result_.nonlocal);
    }

  private:
    [[nodiscard]] int find_g_zero_plane_wave_index(const std::vector<int>& plane_wave_indices,
                                                   int number_of_plane_waves) const {
        for (int plane_wave_index = 0; plane_wave_index < number_of_plane_waves;
             plane_wave_index++) {
            if (planewave_basis_.g_vectors()[plane_wave_indices[plane_wave_index]]
                    .magnitude_squared < math::tolerances::numerical_zero_threshold) {
                return plane_wave_index;
            }
        }
        return -1;
    }

    /**
     * Completes a Gamma half-sphere contraction and leaves G = 0 single-counted.
     */
    static void complete_gamma_only_overlap_matrix(int number_of_plane_waves,
                                                   int number_of_left_functions,
                                                   int number_of_right_functions,
                                                   const Complex* left_functions,
                                                   const Complex* right_functions,
                                                   int g_zero_plane_wave_index,
                                                   std::vector<Complex>& overlaps) {
        for (int left_function_index = 0; left_function_index < number_of_left_functions;
             left_function_index++) {
            for (int right_function_index = 0; right_function_index < number_of_right_functions;
                 right_function_index++) {
                Real g_zero_product =
                    g_zero_plane_wave_index >= 0
                        ? (std::conj(left_functions[static_cast<size_t>(left_function_index) *
                                                        number_of_plane_waves +
                                                    g_zero_plane_wave_index]) *
                           right_functions[static_cast<size_t>(right_function_index) *
                                               number_of_plane_waves +
                                           g_zero_plane_wave_index])
                              .real()
                        : 0.0;
                size_t overlap_index =
                    static_cast<size_t>(right_function_index) * number_of_left_functions +
                    left_function_index;
                overlaps[overlap_index] =
                    Complex(2.0 * overlaps[overlap_index].real() - g_zero_product, 0.0);
            }
        }
    }

    /**
     * Convert k-point from crystal to Cartesian coordinates.
     */
    static Vector3D crystal_to_cartesian_kpoint(const Vector3D& kpoint_crystal,
                                                const Matrix3x3& reciprocal_lattice_vectors) {
        Vector3D result;
        for (int direction = 0; direction < 3; direction++) {
            result[direction] = kpoint_crystal.x * reciprocal_lattice_vectors[0][direction] +
                                kpoint_crystal.y * reciprocal_lattice_vectors[1][direction] +
                                kpoint_crystal.z * reciprocal_lattice_vectors[2][direction];
        }
        return result;
    }

    /**
     * Compute k+G vectors and their inverse norms for all plane waves at a k-point.
     */
    void compute_kpoint_plus_g_vectors(const Vector3D& kpoint_coordinates_cartesian,
                                       const std::vector<int>& plane_wave_indices_kpoint,
                                       int number_of_plane_waves_kpoint,
                                       std::vector<Real>& kpoint_plus_g_x,
                                       std::vector<Real>& kpoint_plus_g_y,
                                       std::vector<Real>& kpoint_plus_g_z,
                                       std::vector<Real>& kpoint_plus_g_norm_inverse) const {
        kpoint_plus_g_x.resize(number_of_plane_waves_kpoint);
        kpoint_plus_g_y.resize(number_of_plane_waves_kpoint);
        kpoint_plus_g_z.resize(number_of_plane_waves_kpoint);
        kpoint_plus_g_norm_inverse.resize(number_of_plane_waves_kpoint);

        for (int g_index = 0; g_index < number_of_plane_waves_kpoint; g_index++) {
            const auto& g_vector = planewave_basis_.g_vectors()[plane_wave_indices_kpoint[g_index]];
            kpoint_plus_g_x[g_index] = (kpoint_coordinates_cartesian.x + g_vector.g_cart.x) *
                                       crystal_structure_.two_pi_over_lattice_parameter();
            kpoint_plus_g_y[g_index] = (kpoint_coordinates_cartesian.y + g_vector.g_cart.y) *
                                       crystal_structure_.two_pi_over_lattice_parameter();
            kpoint_plus_g_z[g_index] = (kpoint_coordinates_cartesian.z + g_vector.g_cart.z) *
                                       crystal_structure_.two_pi_over_lattice_parameter();
            Real kpoint_plus_g_absolute =
                std::sqrt(kpoint_plus_g_x[g_index] * kpoint_plus_g_x[g_index] +
                          kpoint_plus_g_y[g_index] * kpoint_plus_g_y[g_index] +
                          kpoint_plus_g_z[g_index] * kpoint_plus_g_z[g_index]);
            kpoint_plus_g_norm_inverse[g_index] =
                (kpoint_plus_g_absolute > math::tolerances::bessel_argument_zero_threshold)
                    ? 1.0 / kpoint_plus_g_absolute
                    : 0.0;
        }
    }

    static constexpr int bessel_direction_a[6] = {0, 0, 0, 1, 1, 2};
    static constexpr int bessel_direction_b[6] = {0, 1, 2, 1, 2, 2};

    /**
     * The atom-independent part of the projector geometry at one k-point: k+G in
     * units of 2π/a and in atomic units, and the real spherical harmonics and
     * their three Cartesian derivatives at every k+G.
     */
    struct KpointPlusGGeometry {
        std::vector<std::array<Real, 3>> direction_reciprocal;
        std::vector<Real> magnitude_squared_reciprocal;
        std::vector<Real> magnitude_bohr;
        std::vector<std::vector<Real>> spherical_harmonics;
        std::vector<std::vector<std::vector<Real>>> spherical_harmonic_derivatives;
    };

    void compute_kpoint_plus_g_geometry(const Vector3D& kpoint_coordinates_cartesian,
                                        const std::vector<int>& plane_wave_indices_kpoint,
                                        int number_of_plane_waves_kpoint,
                                        int number_of_lm_components,
                                        KpointPlusGGeometry& geometry) const {
        geometry.direction_reciprocal.resize(number_of_plane_waves_kpoint);
        geometry.magnitude_squared_reciprocal.resize(number_of_plane_waves_kpoint);
        geometry.magnitude_bohr.resize(number_of_plane_waves_kpoint);
        for (int g_index = 0; g_index < number_of_plane_waves_kpoint; g_index++) {
            const auto& g_vector = planewave_basis_.g_vectors()[plane_wave_indices_kpoint[g_index]];
            Vector3D kpg;
            kpg.x = kpoint_coordinates_cartesian.x + g_vector.g_cart.x;
            kpg.y = kpoint_coordinates_cartesian.y + g_vector.g_cart.y;
            kpg.z = kpoint_coordinates_cartesian.z + g_vector.g_cart.z;
            Real kpg_norm_recip = kpg.norm();
            geometry.direction_reciprocal[g_index] = {kpg.x, kpg.y, kpg.z};
            geometry.magnitude_squared_reciprocal[g_index] = kpg_norm_recip * kpg_norm_recip;
            geometry.magnitude_bohr[g_index] =
                kpg_norm_recip * crystal_structure_.two_pi_over_lattice_parameter();
        }
        compute_real_spherical_harmonics_batch(number_of_lm_components,
                                               number_of_plane_waves_kpoint,
                                               geometry.direction_reciprocal,
                                               geometry.magnitude_squared_reciprocal,
                                               geometry.spherical_harmonics);
        geometry.spherical_harmonic_derivatives.assign(3, {});
        for (int polarization_index = 0; polarization_index < 3; polarization_index++) {
            compute_spherical_harmonic_derivatives_batch(
                number_of_lm_components,
                number_of_plane_waves_kpoint,
                geometry.direction_reciprocal,
                geometry.magnitude_squared_reciprocal,
                polarization_index,
                geometry.spherical_harmonic_derivatives[polarization_index]);
        }
    }

    /// The projectors and the six strain derivatives of each: seven kinds of column.
    static constexpr int number_of_projector_kinds = 7;

    /**
     * One block's projectors and strain derivatives in a single column-major
     * array, allocated once per k-point for the largest block and reused: a
     * column of plane waves per projector, the block's projectors first and then
     * each strain component's derivatives in the same order. Holding the seven
     * kinds side by side lets one matrix product contract all of them with the
     * bands.
     */
    struct BlockWorkspace {
        std::vector<Complex> columns;

        void allocate(int maximum_projectors, int number_of_plane_waves) {
            columns.resize(static_cast<std::size_t>(number_of_projector_kinds) *
                           static_cast<std::size_t>(maximum_projectors) * number_of_plane_waves);
        }
    };

    /**
     * Build the projectors of the atoms [first_atom, end_atom) and their strain
     * derivatives, contract them with the occupied bands, and add the block's
     * contribution to the nonlocal stress.
     *
     * The strain derivative of <β|ψ> along (a,b) has a radial part,
     * dβ/dq (k+G)_a (k+G)_b / |k+G| Y_lm, and an angular part,
     * β ∂Y_lm/∂(k+G)_a (k+G)_b. Both enter the (a,b) component with the same
     * D-matrix element and band weights, so they are summed into one projector
     * per component before contracting: six products with the bands rather
     * than twelve, and no band-sized array to reweight. Rebuilding those
     * reweighted bands for every block is what made the first blocked version
     * half again as slow as the whole-cell one.
     */
    void accumulate_block(int spin_index,
                          const Vector3D& kpoint_coordinates_cartesian,
                          const std::vector<int>& plane_wave_indices_kpoint,
                          int number_of_plane_waves_kpoint,
                          const KpointPlusGGeometry& geometry,
                          const std::vector<Real>* const kpoint_plus_g_components[3],
                          const std::array<std::vector<Real>, 6>& radial_strain_weights,
                          const std::vector<Complex>& wavefunction_matrix,
                          int number_of_occupied_bands,
                          const std::vector<Real>& occupied_band_weights,
                          const std::vector<Real>& occupied_eigenvalues,
                          int g_zero_plane_wave_index,
                          bool use_effective_d_matrix,
                          int first_atom,
                          int end_atom,
                          int number_of_projectors,
                          BlockWorkspace& workspace) {
        const auto& pseudopotential_basis = scf_state_.quantum_system().pseudopotential_basis();
        Real inverse_sqrt_cell_volume = 1.0 / std::sqrt(crystal_structure_.cell_volume());
        Real inverse_cell_volume = 1.0 / crystal_structure_.cell_volume();
        Real inverse_two_pi_over_lattice_parameter =
            1.0 / crystal_structure_.two_pi_over_lattice_parameter();
        std::size_t plane_waves = static_cast<std::size_t>(number_of_plane_waves_kpoint);

        {
            std::vector<Complex> structure_factor(number_of_plane_waves_kpoint);
            std::vector<Real> beta_derivative_at_g(number_of_plane_waves_kpoint);
            std::vector<Real> beta_at_g(number_of_plane_waves_kpoint);
            int projector_index = 0;
            for (int atom_index = first_atom; atom_index < end_atom; atom_index++) {
                int index_type = crystal_structure_.species_type_index(atom_index);
                Vector3D atomic_position =
                    crystal_structure_.convert_from_crystal_to_cartesian(atom_index);
                for (int g_index = 0; g_index < number_of_plane_waves_kpoint; g_index++) {
                    const auto& g_vector =
                        planewave_basis_.g_vectors()[plane_wave_indices_kpoint[g_index]];
                    Real phase =
                        constants::two_pi * (g_vector.g_cart.dot(atomic_position) +
                                             kpoint_coordinates_cartesian.dot(atomic_position));
                    structure_factor[g_index] = compute_phase_factor(phase);
                }
                for (int index_beta = 0;
                     index_beta < pseudopotential_basis.number_of_projectors(index_type);
                     index_beta++) {
                    int angular_momentum =
                        pseudopotential_basis.beta_angular_momentum(index_type, index_beta);
                    Complex prefactor =
                        inverse_sqrt_cell_volume * compute_angular_phase_factor(angular_momentum);
                    for (int g_index = 0; g_index < number_of_plane_waves_kpoint; g_index++) {
                        beta_derivative_at_g[g_index] = pseudopotential_basis.interpolate_dbeta(
                            index_type, index_beta, geometry.magnitude_bohr[g_index]);
                        beta_at_g[g_index] = pseudopotential_basis.interpolate_beta(
                            index_type, index_beta, geometry.magnitude_bohr[g_index]);
                    }
                    for (int magnetic_projection : harmonic_order_sequence(angular_momentum)) {
                        int lm_index = harmonic_combined_degree_order_index(angular_momentum,
                                                                            magnetic_projection);
                        Complex* projector =
                            workspace.columns.data() +
                            static_cast<std::size_t>(projector_index) * plane_waves;
                        std::array<Complex*, 6> strain_projector;
                        for (int strain_pair = 0; strain_pair < 6; strain_pair++) {
                            strain_projector[strain_pair] =
                                workspace.columns.data() +
                                static_cast<std::size_t>((strain_pair + 1) * number_of_projectors +
                                                         projector_index) *
                                    plane_waves;
                        }
                        for (int g_index = 0; g_index < number_of_plane_waves_kpoint; g_index++) {
                            Complex phased = structure_factor[g_index] * prefactor;
                            Real spherical_harmonic =
                                geometry.spherical_harmonics[g_index][lm_index];
                            projector[g_index] = beta_at_g[g_index] * spherical_harmonic * phased;
                            Complex radial =
                                beta_derivative_at_g[g_index] * spherical_harmonic * phased;
                            /**
                             * β divided by 2π/a to match the unit convention of the
                             * spherical-harmonic derivatives, which are in (2π/a)⁻¹.
                             */
                            Complex angular_scale =
                                beta_at_g[g_index] * inverse_two_pi_over_lattice_parameter * phased;
                            for (int strain_pair = 0; strain_pair < 6; strain_pair++) {
                                int direction_a = bessel_direction_a[strain_pair];
                                int direction_b = bessel_direction_b[strain_pair];
                                strain_projector[strain_pair][g_index] =
                                    radial * radial_strain_weights[strain_pair][g_index] +
                                    angular_scale *
                                        geometry.spherical_harmonic_derivatives[direction_a]
                                                                               [g_index][lm_index] *
                                        (*kpoint_plus_g_components[direction_b])[g_index];
                            }
                        }
                        projector_index++;
                    }
                }
            }
        }

        /**
         * All seven kinds against the occupied bands in one product:
         * overlaps(7·Nproj, Nocc) = columns†(7·Nproj, NPW) Ψ(NPW, Nocc). Rows
         * [0, Nproj) are ⟨β|ψ⟩; rows [(s+1)·Nproj, (s+2)·Nproj) are the
         * derivatives along strain component s.
         */
        Complex alpha_blas(1.0, 0.0);
        Complex beta_blas(0.0, 0.0);
        int blas_rows = number_of_projector_kinds * number_of_projectors;
        int blas_k_pw = number_of_plane_waves_kpoint;
        int blas_n_bands = number_of_occupied_bands;
        std::vector<Complex> overlaps(static_cast<std::size_t>(blas_rows) *
                                          static_cast<std::size_t>(number_of_occupied_bands),
                                      Complex(0.0, 0.0));
        zgemm_("C",
               "N",
               &blas_rows,
               &blas_n_bands,
               &blas_k_pw,
               reinterpret_cast<const std::complex<double>*>(&alpha_blas),
               reinterpret_cast<const std::complex<double>*>(workspace.columns.data()),
               &blas_k_pw,
               reinterpret_cast<const std::complex<double>*>(wavefunction_matrix.data()),
               &blas_k_pw,
               reinterpret_cast<const std::complex<double>*>(&beta_blas),
               reinterpret_cast<std::complex<double>*>(overlaps.data()),
               &blas_rows);
        if (planewave_basis_.is_gamma_only()) {
            complete_gamma_only_overlap_matrix(number_of_plane_waves_kpoint,
                                               blas_rows,
                                               number_of_occupied_bands,
                                               workspace.columns.data(),
                                               wavefunction_matrix.data(),
                                               g_zero_plane_wave_index,
                                               overlaps);
        }

        // Accumulate this block's atoms: projector pairs within each atom, bands inline.
        int projector_offset = 0;
        for (int atom_index = first_atom; atom_index < end_atom; atom_index++) {
            int species_type_index = crystal_structure_.species_type_index(atom_index);
            int number_of_betas = pseudopotential_basis.number_of_projectors(species_type_index);

            std::vector<int> projector_beta_indices, projector_magnetic_indices;
            int projector_start_index = projector_offset;
            for (int beta_index = 0; beta_index < number_of_betas; beta_index++) {
                int angular_momentum =
                    pseudopotential_basis.beta_angular_momentum(species_type_index, beta_index);
                for (int magnetic_projection : harmonic_order_sequence(angular_momentum)) {
                    projector_beta_indices.push_back(beta_index);
                    projector_magnetic_indices.push_back(magnetic_projection);
                    projector_offset++;
                }
            }
            int number_of_projectors_atom = static_cast<int>(projector_beta_indices.size());

            for (int projector_a_local = 0; projector_a_local < number_of_projectors_atom;
                 projector_a_local++) {
                for (int projector_b_local = 0; projector_b_local < number_of_projectors_atom;
                     projector_b_local++) {
                    int angular_momentum_a = pseudopotential_basis.beta_angular_momentum(
                        species_type_index, projector_beta_indices[projector_a_local]);
                    int angular_momentum_b = pseudopotential_basis.beta_angular_momentum(
                        species_type_index, projector_beta_indices[projector_b_local]);

                    // Band-independent D matrix element
                    Real d_element =
                        scf_state_.quantum_system().d_matrix_element(atom_index,
                                                                     species_type_index,
                                                                     projector_a_local,
                                                                     projector_b_local,
                                                                     spin_index);

                    /**
                     * Augmentation overlap for US, gated per species as well as on
                     * `use_effective_d_matrix`: that flag is `has_ultrasoft()`, a
                     * property of the whole system, true in a mixed cell while
                     * individual species are still norm-conserving. The forces twin in
                     * `forces/.../nonlocal.hpp` carries this same per-species gate.
                     */
                    Real overlap_integral = 0.0;
                    if (use_effective_d_matrix &&
                        pseudopotential_basis.pseudopotential_set()[species_type_index]
                            .has_augmentation_charges() &&
                        angular_momentum_a == angular_momentum_b &&
                        projector_magnetic_indices[projector_a_local] ==
                            projector_magnetic_indices[projector_b_local]) {
                        overlap_integral =
                            static_cast<const UltrasoftPlanewaveKohnShamQuantumSystem&>(
                                scf_state_.quantum_system())
                                .ultrasoft_basis()
                                .augmentation_overlap_integral(
                                    species_type_index,
                                    projector_beta_indices[projector_a_local],
                                    projector_beta_indices[projector_b_local]);
                    }

                    if (std::abs(d_element) < math::tolerances::d_matrix_element_threshold &&
                        std::abs(overlap_integral) < math::tolerances::d_matrix_element_threshold)
                        continue;

                    int global_a = projector_start_index + projector_a_local;
                    int global_b = projector_start_index + projector_b_local;

                    Real diagonal_sum = 0.0;
                    Real eigenvalue_diagonal_sum = 0.0;
                    Real strain_sums[6] = {};
                    Real eigenvalue_strain_sums[6] = {};
                    bool has_augmentation_term =
                        use_effective_d_matrix &&
                        std::abs(overlap_integral) > math::tolerances::d_matrix_element_threshold;

                    for (int band_n = 0; band_n < number_of_occupied_bands; band_n++) {
                        Real weight = occupied_band_weights[band_n];
                        std::size_t band_offset = static_cast<std::size_t>(band_n) * blas_rows;
                        Complex becp_a_conjugate = std::conj(overlaps[band_offset + global_a]);

                        Complex becp_b_value = overlaps[band_offset + global_b];
                        Real diagonal_term = weight * (becp_a_conjugate * becp_b_value).real();
                        diagonal_sum += diagonal_term;

                        Real eigenvalue =
                            has_augmentation_term ? occupied_eigenvalues[band_n] : 0.0;
                        eigenvalue_diagonal_sum += diagonal_term * eigenvalue;
                        for (int strain_pair = 0; strain_pair < 6; strain_pair++) {
                            Real strain_term =
                                weight * 2.0 *
                                (becp_a_conjugate *
                                 overlaps[band_offset +
                                          static_cast<std::size_t>((strain_pair + 1) *
                                                                   number_of_projectors) +
                                          global_b])
                                    .real();
                            strain_sums[strain_pair] += strain_term;
                            eigenvalue_strain_sums[strain_pair] += strain_term * eigenvalue;
                        }
                    }

                    for (int direction = 0; direction < 3; direction++) {
                        result_.nonlocal[direction][direction] +=
                            d_element * diagonal_sum * inverse_cell_volume;
                    }
                    for (int strain_pair = 0; strain_pair < 6; strain_pair++) {
                        int direction_a = bessel_direction_a[strain_pair];
                        int direction_b = bessel_direction_b[strain_pair];
                        result_.nonlocal[direction_a][direction_b] +=
                            d_element * strain_sums[strain_pair] * inverse_cell_volume;
                    }

                    if (has_augmentation_term) {
                        for (int direction = 0; direction < 3; direction++) {
                            result_.nonlocal[direction][direction] -=
                                overlap_integral * eigenvalue_diagonal_sum * inverse_cell_volume;
                        }
                        for (int strain_pair = 0; strain_pair < 6; strain_pair++) {
                            int direction_a = bessel_direction_a[strain_pair];
                            int direction_b = bessel_direction_b[strain_pair];
                            result_.nonlocal[direction_a][direction_b] -=
                                overlap_integral * eigenvalue_strain_sums[strain_pair] *
                                inverse_cell_volume;
                        }
                    }
                }
            }
        }
    }

    /// Symmetrize the upper triangle of a 3x3 stress tensor.
    static void symmetrize_stress_tensor(Real stress[3][3]) {
        for (int direction_a = 0; direction_a < 3; direction_a++) {
            for (int direction_b = direction_a + 1; direction_b < 3; direction_b++) {
                stress[direction_b][direction_a] = stress[direction_a][direction_b];
            }
        }
    }

    const Crystal& crystal_structure_;
    const PlanewaveBasis& planewave_basis_;
    const KPointsGrid& kpoint_grid_;
    const NormConservingPlanewaveKohnShamSCFState& scf_state_;
    StressTensorContributions& result_;
    const std::string derivative_method_;
};

} // namespace q3
