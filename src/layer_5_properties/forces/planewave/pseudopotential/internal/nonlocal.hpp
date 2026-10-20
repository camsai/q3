#pragma once

/**
 * Nonlocal (Kleinman-Bylander) pseudopotential force contribution.
 *
 * Computes the Hellmann-Feynman force from the nonlocal pseudopotential
 * using KB projectors and their derivatives with respect to atomic positions.
 *
 * For norm-conserving PPs the bare ionic D matrix is used -- `d_ion_matrix`,
 * read from the UPF `PP_DIJ` record. For ultrasoft and PAW it is the screened
 * D matrix, D_ion plus the integral of v_eff against Q_ij, which is per spin
 * channel and is supplied by the nonlocal effective potential.
 *
 * Gamma-point correction
 * ----------------------
 * When the gamma-only optimisation is active (1×1×1 k-grid, no shift), Q3
 * stores only the positive half-sphere G-vectors. The half-sphere overlaps
 * A_half = <β|(-iG)|ψ>_half and B_half = <β|ψ>_half relate to their
 * full-sphere counterparts by:
 *
 *   A_full = 2 · Re[A_half],   B_full = 2 · Re[B_half]
 *
 * so the correct Hellmann-Feynman force is:
 *
 *   F = 2D · Re[A_full · conj(B_full)] = 2D · 4 · Re[A_half] · Re[B_half]
 *
 * The naive formula Re[A_half · conj(B_half)] is wrong because the Im·Im
 * cross term is non-zero for the complex half-sphere coefficients produced
 * by the Davidson solver, causing incorrect forces at the gamma point.
 */

#include <cmath>
#include <complex>
#include <vector>

#include "layer_0_core/math/linear_algebra/blas.hpp"
#include "layer_0_core/math/tolerances.hpp"
#include "layer_0_core/types/init.hpp"
#include "layer_0_core/types/matrix.hpp"
#include "layer_1_structure/crystal/crystal.hpp"
#include "layer_1_structure/reciprocal_space/kpoints_grid.hpp"
#include "layer_3_representation/basis/planewave/init.hpp"
#include "layer_3_representation/pseudopotential/planewave/internal/projector.hpp"
#include "layer_3_representation/quantum_system/kohn_sham/planewave/norm_conserving/init.hpp"
#include "layer_3_representation/quantum_system/kohn_sham/planewave/ultrasoft/init.hpp"
#include "layer_4_computation/state/kohn_sham/planewave/norm_conserving/init.hpp"

namespace q3 {
using namespace q3::blas;

class NonlocalPseudopotentialForceCalculator {
  public:
    NonlocalPseudopotentialForceCalculator(const Crystal& crystal_structure,
                                           const PlanewaveBasis& planewave_basis,
                                           const KPointsGrid& kpoint_grid,
                                           const NormConservingPlanewaveKohnShamSCFState& scf_state)
        : crystal_structure_(crystal_structure)
        , planewave_basis_(planewave_basis)
        , kpoint_grid_(kpoint_grid)
        , scf_state_(scf_state) {}

    [[nodiscard]] std::vector<Vector3D> compute() const {
        bool is_gamma_only = planewave_basis_.is_gamma_only();
        int number_of_kpoints = kpoint_grid_.number_of_kpoints();
        std::vector<Vector3D> nonlocal_force(crystal_structure_.number_of_atoms(), {0.0, 0.0, 0.0});
        int total_number_of_kpoint_sets =
            number_of_kpoints * (scf_state_.number_of_spin_components() == 2 ? 2 : 1);
        const bool has_ultrasoft = scf_state_.quantum_system().has_ultrasoft();

        for (int kpoint_spin_index = 0; kpoint_spin_index < total_number_of_kpoint_sets;
             kpoint_spin_index++) {
            int kpoint_index = (kpoint_spin_index % number_of_kpoints);
            int number_of_plane_waves = scf_state_.number_of_plane_waves(kpoint_index);
            const auto& plane_wave_indices = scf_state_.g_vector_indices_at_kpoint(kpoint_index);
            Vector3D kpoint_vector = kpoint_grid_[kpoint_index].coordinates;
            Real kpoint_weight = kpoint_grid_[kpoint_index].weight;

            const Matrix3x3& reciprocal_lattice_vectors =
                crystal_structure_.reciprocal_lattice_vectors();
            Vector3D kpoint_cartesian;
            for (int direction = 0; direction < 3; direction++) {
                kpoint_cartesian[direction] =
                    kpoint_vector.x * reciprocal_lattice_vectors[0][direction] +
                    kpoint_vector.y * reciprocal_lattice_vectors[1][direction] +
                    kpoint_vector.z * reciprocal_lattice_vectors[2][direction];
            }

            Matrix<Complex> projector_functions;
            compute_kleinman_bylander_projectors(
                kpoint_vector,
                crystal_structure_,
                planewave_basis_,
                scf_state_.quantum_system().pseudopotential_basis().pseudopotential_set(),
                scf_state_.quantum_system().pseudopotential_basis().beta_function_table(),
                plane_wave_indices,
                number_of_plane_waves,
                scf_state_.quantum_system().pseudopotential_basis().total_number_of_projectors(),
                projector_functions);

            // Collect occupied band indices and weights
            std::vector<int> occupied_band_indices;
            std::vector<Real> occupied_band_weights;
            std::vector<Real> occupied_band_occupations;
            /**
             * Needed for the ultrasoft/PAW overlap term below, which is
             * band-dependent through the eigenvalue where the D term is not.
             */
            std::vector<Real> occupied_band_eigenvalues;
            for (int band_index = 0; band_index < scf_state_.number_of_bands(); band_index++) {
                Real occupation = scf_state_.occupation(kpoint_spin_index, band_index);
                if (occupation < math::tolerances::occupation_threshold)
                    continue;
                occupied_band_indices.push_back(band_index);
                occupied_band_weights.push_back(kpoint_weight * occupation);
                occupied_band_occupations.push_back(occupation);
                occupied_band_eigenvalues.push_back(
                    scf_state_.eigenvalue(kpoint_spin_index, band_index));
            }
            int number_of_occupied_bands = static_cast<int>(occupied_band_indices.size());
            if (number_of_occupied_bands == 0)
                continue;

            int number_of_projectors =
                scf_state_.quantum_system().pseudopotential_basis().total_number_of_projectors();

            // Precompute -i(k+G) components
            std::vector<std::vector<Complex>> minus_i_k_plus_g(
                3, std::vector<Complex>(number_of_plane_waves));
            for (int plane_wave_index = 0; plane_wave_index < number_of_plane_waves;
                 plane_wave_index++) {
                const auto& g_vector =
                    planewave_basis_.g_vectors()[plane_wave_indices[plane_wave_index]];
                for (int direction = 0; direction < 3; direction++) {
                    Real k_plus_g_direction =
                        (g_vector.g_cart[direction] + kpoint_cartesian[direction]) *
                        crystal_structure_.two_pi_over_lattice_parameter();
                    minus_i_k_plus_g[direction][plane_wave_index] =
                        Complex(0.0, -k_plus_g_direction);
                }
            }

            /**
             * We need 4 overlaps per band:
             * 0: <beta | psi>
             * 1,2,3: <beta | -i(k+G) psi> for x,y,z
             */
            std::vector<std::vector<Complex>> batched_overlaps(
                4, std::vector<Complex>(number_of_projectors * number_of_occupied_bands));

            std::vector<Complex> modified_wavefunction(number_of_plane_waves *
                                                       number_of_occupied_bands);

            Complex alpha_blas(1.0, 0.0);
            Complex beta_blas(0.0, 0.0);
            int blas_rows = number_of_projectors;
            int blas_columns = number_of_occupied_bands;
            int blas_inner_dimension = number_of_plane_waves;

            for (int component = 0; component < 4; component++) {
                for (int occupied_band_index = 0; occupied_band_index < number_of_occupied_bands;
                     occupied_band_index++) {
                    int band_index = occupied_band_indices[occupied_band_index];
                    int offset = occupied_band_index * number_of_plane_waves;
                    for (int g_index = 0; g_index < number_of_plane_waves; g_index++) {
                        Complex psi = scf_state_.wavefunction_coefficient(
                            kpoint_spin_index, band_index, g_index);
                        if (component == 0) {
                            modified_wavefunction[offset + g_index] = psi;
                        } else {
                            modified_wavefunction[offset + g_index] =
                                psi * minus_i_k_plus_g[component - 1][g_index];
                        }
                    }
                }

                zgemm_("C",
                       "N",
                       &blas_rows,
                       &blas_columns,
                       &blas_inner_dimension,
                       reinterpret_cast<const std::complex<double>*>(&alpha_blas),
                       reinterpret_cast<const std::complex<double>*>(projector_functions.data()),
                       &blas_inner_dimension,
                       reinterpret_cast<const std::complex<double>*>(modified_wavefunction.data()),
                       &blas_inner_dimension,
                       reinterpret_cast<const std::complex<double>*>(&beta_blas),
                       reinterpret_cast<std::complex<double>*>(batched_overlaps[component].data()),
                       &blas_rows);
            }

            /**
             * When ranks share a k-point pool, each holds only its slab of this
             * k-point's plane waves, so every overlap above is a partial sum
             * over G. The force multiplies two of them, so they are summed
             * across the pool here, before any product. Summing the per-rank
             * forces instead drops every cross term between two ranks' slabs:
             * the energy, linear in the projections the SCF already reduced,
             * stayed exact while the forces came out up to 0.6 Ry/bohr wrong
             * (camsai/q3#22). The DFT+U force reduces its overlaps the same
             * way. A no-op when a pool has one rank.
             */
            if (scf_state_.execution_policy != nullptr) {
                for (auto& component_overlaps : batched_overlaps)
                    scf_state_.execution_policy->reduce_within_group(
                        component_overlaps.data(), static_cast<int>(component_overlaps.size()));
            }

            /**
             * The G = 0 term that `2·Re[B_half]` double counts.
             *
             * Q3's gamma-only half sphere *includes* G = 0 (the selection rule
             * is `index_1 >= 0` when the other two vanish), so
             * `2·Re[<beta|psi>_half] = <beta|psi>_full + beta(0)psi(0)`. The
             * derivative overlap needs no such correction — its summand carries
             * a factor G_alpha, which is zero there — which is why leaving it
             * out skewed the components unevenly rather than scaling them: it
             * only touches projectors with beta(0) != 0, i.e. l = 0. Silicon's
             * Gamma force came out right in x and 1.36x too large in y.
             *
             * Quantum ESPRESSO corrects the same double count, in a different
             * place: it subtracts inside the overlap routine itself, so both
             * overlaps are corrected there and the correction on the derivative
             * one is a no-op for the same reason it is skipped here. The
             * derivation above rests on this file's own selection rule; the
             * upstream agreement is corroboration, not its basis.
             */
            std::vector<Real> gamma_g0_overlap;
            if (is_gamma_only) {
                gamma_g0_overlap.assign(
                    static_cast<size_t>(number_of_projectors) * number_of_occupied_bands, 0.0);
                int g0_index = -1;
                for (int plane_wave_index = 0; plane_wave_index < number_of_plane_waves;
                     plane_wave_index++) {
                    if (planewave_basis_.g_vectors()[plane_wave_indices[plane_wave_index]]
                            .magnitude_squared < math::tolerances::numerical_zero_threshold) {
                        g0_index = plane_wave_index;
                        break;
                    }
                }
                if (g0_index >= 0) {
                    for (int projector = 0; projector < number_of_projectors; projector++) {
                        for (int occupied_band_index = 0;
                             occupied_band_index < number_of_occupied_bands;
                             occupied_band_index++) {
                            Complex psi = scf_state_.wavefunction_coefficient(
                                kpoint_spin_index,
                                occupied_band_indices[occupied_band_index],
                                g0_index);
                            gamma_g0_overlap[static_cast<size_t>(projector) *
                                                 number_of_occupied_bands +
                                             occupied_band_index] =
                                (std::conj(projector_functions[projector][g0_index]) * psi).real();
                        }
                    }
                }
            }

            // Accumulate forces
            int current_projector_index = 0;
            for (int atom_index = 0; atom_index < crystal_structure_.number_of_atoms();
                 atom_index++) {
                int species_index = crystal_structure_.species_type_index(atom_index);
                int number_of_beta_functions =
                    scf_state_.quantum_system().pseudopotential_basis().number_of_projectors(
                        species_index);

                std::vector<int> beta_indices, magnetic_quantum_numbers, angular_momenta;
                int starting_projector_index = current_projector_index;
                for (int beta_function_index = 0; beta_function_index < number_of_beta_functions;
                     beta_function_index++) {
                    int angular_momentum =
                        scf_state_.quantum_system().pseudopotential_basis().beta_angular_momentum(
                            species_index, beta_function_index);
                    for (int magnetic_projection : harmonic_order_sequence(angular_momentum)) {
                        beta_indices.push_back(beta_function_index);
                        magnetic_quantum_numbers.push_back(magnetic_projection);
                        angular_momenta.push_back(angular_momentum);
                        current_projector_index++;
                    }
                }
                int number_of_projectors_on_atom = static_cast<int>(beta_indices.size());

                for (int projector_a_index = 0; projector_a_index < number_of_projectors_on_atom;
                     projector_a_index++) {
                    for (int projector_b_index = 0;
                         projector_b_index < number_of_projectors_on_atom;
                         projector_b_index++) {
                        /**
                         * D is per spin channel for ultrasoft and PAW, and the
                         * loop above is spin-major over `kpoint_spin_index`, so
                         * the channel is the quotient by the k-point count.
                         */
                        Real hamiltonian_d_coefficient =
                            scf_state_.quantum_system().d_matrix_element(atom_index,
                                                                         species_index,
                                                                         projector_a_index,
                                                                         projector_b_index,
                                                                         kpoint_spin_index /
                                                                             number_of_kpoints);

                        /**
                         * The generalized-eigenvalue term, for ultrasoft and PAW.
                         *
                         * These solve H|psi> = eps S|psi> with S = 1 + sum q_ij
                         * |p_i><p_j|, so S moves with the atom too and the force
                         * contracts (D_ij - eps_nk q_ij), not D_ij alone. The
                         * stress has always carried this correction; the force
                         * did not, which left it ~6% high on silicon even after
                         * the augmentation term was repaired -- and unlike a
                         * missing augmentation term, this one is band-dependent,
                         * so it cannot be absorbed into any scale factor.
                         *
                         * q_ij is diagonal in (l, m): the augmentation overlap
                         * integral is a radial quantity and the angular parts
                         * are orthonormal.
                         */
                        /**
                         * `has_ultrasoft` is a whole-system flag: a cell with
                         * one PAW species and one norm-conserving species takes
                         * the ultrasoft iterator. Asking the norm-conserving
                         * species for an augmentation overlap dereferences a
                         * null `as_ultrasoft`, and the assert that would catch
                         * it is compiled out in Release. Gate per species.
                         */
                        Real augmentation_overlap = 0.0;
                        if (has_ultrasoft &&
                            scf_state_.quantum_system()
                                .pseudopotential_basis()
                                .pseudopotential_set()[species_index]
                                .has_augmentation_charges() &&
                            angular_momenta[projector_a_index] ==
                                angular_momenta[projector_b_index] &&
                            magnetic_quantum_numbers[projector_a_index] ==
                                magnetic_quantum_numbers[projector_b_index]) {
                            augmentation_overlap =
                                static_cast<const UltrasoftPlanewaveKohnShamQuantumSystem&>(
                                    scf_state_.quantum_system())
                                    .ultrasoft_basis()
                                    .augmentation_overlap_integral(species_index,
                                                                   beta_indices[projector_a_index],
                                                                   beta_indices[projector_b_index]);
                        }

                        if (std::abs(hamiltonian_d_coefficient) <
                                math::tolerances::d_matrix_element_threshold &&
                            std::abs(augmentation_overlap) <
                                math::tolerances::d_matrix_element_threshold) {
                            continue;
                        }

                        int global_projector_a_index = starting_projector_index + projector_a_index;
                        int global_projector_b_index = starting_projector_index + projector_b_index;

                        Vector3D force_sum = {0.0, 0.0, 0.0};
                        Vector3D eigenvalue_weighted_sum = {0.0, 0.0, 0.0};

                        for (int occupied_band_index = 0;
                             occupied_band_index < number_of_occupied_bands;
                             occupied_band_index++) {
                            Real weight = occupied_band_weights[occupied_band_index];
                            Real eigenvalue = occupied_band_eigenvalues[occupied_band_index];

                            Complex overlap_b_0 =
                                batched_overlaps[0][occupied_band_index * number_of_projectors +
                                                    global_projector_b_index];
                            Complex overlap_b_0_conj = std::conj(overlap_b_0);

                            for (int direction = 0; direction < 3; direction++) {
                                Complex overlap_a_dir =
                                    batched_overlaps[direction + 1]
                                                    [occupied_band_index * number_of_projectors +
                                                     global_projector_a_index];

                                Complex spin_summed_force_contribution =
                                    overlap_a_dir * overlap_b_0_conj;
                                // k-point weight already includes the spin degeneracy factor of 2.
                                Real term =
                                    is_gamma_only
                                        // A_full · B_full; see the G=0 note above.
                                        ? (2.0 * overlap_a_dir.real()) *
                                              (2.0 * overlap_b_0.real() -
                                               gamma_g0_overlap[static_cast<size_t>(
                                                                    global_projector_b_index) *
                                                                    number_of_occupied_bands +
                                                                occupied_band_index])
                                        : spin_summed_force_contribution.real();
                                force_sum[direction] += weight * term;
                                eigenvalue_weighted_sum[direction] += weight * eigenvalue * term;
                            }
                        }

                        for (int direction = 0; direction < 3; direction++) {
                            nonlocal_force[atom_index][direction] +=
                                2.0 * (hamiltonian_d_coefficient * force_sum[direction] -
                                       augmentation_overlap * eigenvalue_weighted_sum[direction]);
                        }
                    }
                }
            }
        }

        return nonlocal_force;
    }

  private:
    const Crystal& crystal_structure_;
    const PlanewaveBasis& planewave_basis_;
    const KPointsGrid& kpoint_grid_;
    const NormConservingPlanewaveKohnShamSCFState& scf_state_;
};

} // namespace q3
