#pragma once

/**
 * UltrasoftPseudopotentialOverlapOperator — S = I + Σ q_ij |β_i⟩⟨β_j|.
 *
 * Ultrasoft pseudopotential_set relax the norm-conservation constraint, adding the
 * augmentation overlap built from the Kleinman-Bylander projectors and the q_ij
 * augmentation charges. This makes the eigenproblem generalized (zhegv).
 */

#include <algorithm>
#include <complex>
#include <cstring>
#include <vector>

#include "layer_0_core/math/linear_algebra/blas.hpp"
#include "layer_0_core/math/tolerances.hpp"
#include "layer_0_core/types/init.hpp"
#include "layer_0_core/types/matrix.hpp"
#include "layer_3_representation/pseudopotential/planewave/internal/projector.hpp"
#include "layer_3_representation/quantum_system/kohn_sham/planewave/ultrasoft/init.hpp"
#include "layer_4_computation/diagonalizer/planewave/internal/overlap/pseudopotential/init.hpp"

namespace q3 {

class UltrasoftPseudopotentialOverlapOperator : public PseudopotentialOverlapOperator {
  public:
    using PseudopotentialOverlapOperator::PseudopotentialOverlapOperator;

    [[nodiscard]] bool is_identity() const override {
        return false;
    }

    void apply(int number_of_plane_waves,
               const Complex* wavefunction,
               Complex* overlap_wavefunction) const override {
        // Copy input to output: S = I + augmentation
        std::memcpy(overlap_wavefunction, wavefunction, number_of_plane_waves * sizeof(Complex));

        int total_projectors = static_cast<int>(kleinman_bylander_potential_.rows());
        if (total_projectors == 0) {
            return;
        }

        // Ensure q-matrix is cached
        if (cached_q_matrix_.empty()) {
            build_q_matrix();
        }

        // 1. Compute <β|ψ> for all projectors
        std::vector<Complex> projector_overlaps(total_projectors, Complex(0.0, 0.0));
        bool is_gamma_only = quantum_system_.basis().is_gamma_only();

        if (is_gamma_only) {
            int double_number_of_plane_waves = 2 * number_of_plane_waves;
            double alpha_two = 2.0;
            double double_beta_zero = 0.0;
            int one_increment = 1;

            std::vector<double> real_projector_overlaps(total_projectors, 0.0);
            const double* real_kleinman_bylander_potential =
                reinterpret_cast<const double*>(kleinman_bylander_potential_.data());
            const double* real_wavefunction = reinterpret_cast<const double*>(wavefunction);

            q3::blas::dgemv_("T",
                             &double_number_of_plane_waves,
                             &total_projectors,
                             &alpha_two,
                             real_kleinman_bylander_potential,
                             &double_number_of_plane_waves,
                             real_wavefunction,
                             &one_increment,
                             &double_beta_zero,
                             real_projector_overlaps.data(),
                             &one_increment);

            /**
             * G = 0 correction: the origin is removed from the double count on
             * the rank whose plane-wave list begins with it, and nowhere else.
             */
            bool has_zero_g_vector = quantum_system_.basis().gamma_origin_is_local();
            if (has_zero_g_vector) {
                for (int projector_index = 0; projector_index < total_projectors;
                     ++projector_index) {
                    Complex kleinman_bylander_g0 = kleinman_bylander_potential_[projector_index][0];
                    Complex wavefunction_g0 = wavefunction[0];
                    real_projector_overlaps[projector_index] -=
                        (std::conj(kleinman_bylander_g0) * wavefunction_g0).real();
                }
            }

            for (int projector_index = 0; projector_index < total_projectors; ++projector_index) {
                projector_overlaps[projector_index] =
                    Complex(real_projector_overlaps[projector_index], 0.0);
            }
        } else {
            Complex alpha_one(1.0, 0.0);
            Complex beta_zero_val(0.0, 0.0);
            int one_increment = 1;
            auto* cast_alpha = reinterpret_cast<const std::complex<double>*>(&alpha_one);
            auto* cast_beta_zero = reinterpret_cast<const std::complex<double>*>(&beta_zero_val);
            auto* cast_kleinman_bylander =
                reinterpret_cast<const std::complex<double>*>(kleinman_bylander_potential_.data());
            auto* cast_wavefunction = reinterpret_cast<const std::complex<double>*>(wavefunction);
            auto* cast_projector_overlaps =
                reinterpret_cast<std::complex<double>*>(projector_overlaps.data());

            q3::blas::zgemv_("C",
                             &number_of_plane_waves,
                             &total_projectors,
                             cast_alpha,
                             cast_kleinman_bylander,
                             &number_of_plane_waves,
                             cast_wavefunction,
                             &one_increment,
                             cast_beta_zero,
                             cast_projector_overlaps,
                             &one_increment);
        }

        execution_policy_.reduce_within_group(projector_overlaps.data(), total_projectors);

        // 2. Apply q-matrix: projector_overlaps_with_q_matrix = Q × projector_overlaps
        std::vector<Complex> projector_overlaps_with_q_matrix(total_projectors, Complex(0.0, 0.0));
        Complex alpha_one_val(1.0, 0.0);
        Complex beta_zero_val(0.0, 0.0);
        int one_increment = 1;
        auto* cast_alpha = reinterpret_cast<const std::complex<double>*>(&alpha_one_val);
        auto* cast_beta_zero = reinterpret_cast<const std::complex<double>*>(&beta_zero_val);
        auto* cast_q_matrix =
            reinterpret_cast<const std::complex<double>*>(cached_q_matrix_.data());
        auto* cast_projector_overlaps =
            reinterpret_cast<std::complex<double>*>(projector_overlaps.data());
        auto* cast_projector_overlaps_with_q_matrix =
            reinterpret_cast<std::complex<double>*>(projector_overlaps_with_q_matrix.data());

        q3::blas::zgemv_("N",
                         &total_projectors,
                         &total_projectors,
                         cast_alpha,
                         cast_q_matrix,
                         &total_projectors,
                         cast_projector_overlaps,
                         &one_increment,
                         cast_beta_zero,
                         cast_projector_overlaps_with_q_matrix,
                         &one_increment);

        // 3. S|ψ⟩ += β × projector_overlaps_with_q_matrix
        Complex beta_one(1.0, 0.0);
        auto* cast_one = reinterpret_cast<const std::complex<double>*>(&beta_one);
        auto* cast_kleinman_bylander =
            reinterpret_cast<const std::complex<double>*>(kleinman_bylander_potential_.data());
        auto* cast_overlap_wavefunction =
            reinterpret_cast<std::complex<double>*>(overlap_wavefunction);

        q3::blas::zgemv_("N",
                         &number_of_plane_waves,
                         &total_projectors,
                         cast_alpha,
                         cast_kleinman_bylander,
                         &number_of_plane_waves,
                         cast_projector_overlaps_with_q_matrix,
                         &one_increment,
                         cast_one,
                         cast_overlap_wavefunction,
                         &one_increment);
    }

    /**
     * Efficient batched S|ψ⟩ for ultrasoft pseudopotentials.
     *
     * Uses zgemm for computing all <β|ψ> projections across all vectors at once,
     * then applies the q-matrix and adds back via zgemm.
     */
    void apply_batch(int number_of_plane_waves,
                     int number_of_vectors,
                     const Complex* wavefunctions,
                     Complex* overlap_wavefunctions) const override {
        // Start with S = I: copy all vectors
        std::memcpy(overlap_wavefunctions,
                    wavefunctions,
                    static_cast<size_t>(number_of_plane_waves) * number_of_vectors *
                        sizeof(Complex));

        int total_projectors = static_cast<int>(kleinman_bylander_potential_.rows());
        if (total_projectors == 0) {
            return;
        }

        /**
         * Step 1: Compute all projections projector_overlaps(total_projectors, number_of_vectors) =
         * β^H × ψ
         */
        std::vector<Complex> projector_overlaps(
            static_cast<size_t>(total_projectors) * number_of_vectors, Complex(0.0, 0.0));
        bool is_gamma_only = quantum_system_.basis().is_gamma_only();

        if (is_gamma_only) {
            int double_number_of_plane_waves = 2 * number_of_plane_waves;
            double alpha_two = 2.0;
            double double_beta_zero = 0.0;

            std::vector<double> real_projector_overlaps(
                static_cast<size_t>(total_projectors) * number_of_vectors, 0.0);
            const double* real_kleinman_bylander_potential =
                reinterpret_cast<const double*>(kleinman_bylander_potential_.data());
            const double* real_wavefunctions = reinterpret_cast<const double*>(wavefunctions);

            // Compute 2 * Re(KB^H · wavefunctions) via DGEMM
            q3::blas::dgemm_("T",
                             "N",
                             &total_projectors,
                             &number_of_vectors,
                             &double_number_of_plane_waves,
                             &alpha_two,
                             real_kleinman_bylander_potential,
                             &double_number_of_plane_waves,
                             real_wavefunctions,
                             &double_number_of_plane_waves,
                             &double_beta_zero,
                             real_projector_overlaps.data(),
                             &total_projectors);

            /**
             * G = 0 correction: the origin is removed from the double count on
             * the rank whose plane-wave list begins with it, and nowhere else.
             */
            bool has_zero_g_vector = quantum_system_.basis().gamma_origin_is_local();
            if (has_zero_g_vector) {
                for (int vector_index = 0; vector_index < number_of_vectors; ++vector_index) {
                    Complex wavefunction_g0 =
                        wavefunctions[static_cast<size_t>(vector_index) * number_of_plane_waves];
                    for (int projector_index = 0; projector_index < total_projectors;
                         ++projector_index) {
                        Complex kleinman_bylander_g0 =
                            kleinman_bylander_potential_[projector_index][0];
                        size_t index = static_cast<size_t>(projector_index) +
                                       static_cast<size_t>(vector_index) * total_projectors;
                        real_projector_overlaps[index] -=
                            (std::conj(kleinman_bylander_g0) * wavefunction_g0).real();
                    }
                }
            }

            for (size_t index = 0; index < real_projector_overlaps.size(); ++index) {
                projector_overlaps[index] = Complex(real_projector_overlaps[index], 0.0);
            }
        } else {
            Complex alpha_blas(1.0, 0.0);
            Complex beta_blas_zero(0.0, 0.0);
            auto* cast_alpha = reinterpret_cast<const std::complex<double>*>(&alpha_blas);
            auto* cast_zero = reinterpret_cast<const std::complex<double>*>(&beta_blas_zero);
            auto* cast_kleinman_bylander =
                reinterpret_cast<const std::complex<double>*>(kleinman_bylander_potential_.data());
            auto* cast_wavefunctions = reinterpret_cast<const std::complex<double>*>(wavefunctions);
            auto* cast_projector_overlaps =
                reinterpret_cast<std::complex<double>*>(projector_overlaps.data());

            q3::blas::zgemm_("C",
                             "N",
                             &total_projectors,
                             &number_of_vectors,
                             &number_of_plane_waves,
                             cast_alpha,
                             cast_kleinman_bylander,
                             &number_of_plane_waves,
                             cast_wavefunctions,
                             &number_of_plane_waves,
                             cast_zero,
                             cast_projector_overlaps,
                             &total_projectors);
        }

        execution_policy_.reduce_within_group(projector_overlaps.data(),
                                              total_projectors * number_of_vectors);

        /**
         * Step 2: Apply q-matrix: projector_overlaps_with_q_matrix(i, vec) = Σ_j q_ij *
         * projector_overlaps(j, vec)
         */
        if (cached_q_matrix_.empty()) {
            build_q_matrix();
        }
        std::vector<Complex> projector_overlaps_with_q_matrix(
            static_cast<size_t>(total_projectors) * number_of_vectors, Complex(0.0, 0.0));
        Complex alpha_blas(1.0, 0.0);
        Complex beta_blas_zero(0.0, 0.0);
        auto* cast_alpha = reinterpret_cast<const std::complex<double>*>(&alpha_blas);
        auto* cast_zero = reinterpret_cast<const std::complex<double>*>(&beta_blas_zero);
        auto* cast_q_matrix =
            reinterpret_cast<const std::complex<double>*>(cached_q_matrix_.data());
        auto* cast_projector_overlaps =
            reinterpret_cast<std::complex<double>*>(projector_overlaps.data());
        auto* cast_projector_overlaps_with_q_matrix =
            reinterpret_cast<std::complex<double>*>(projector_overlaps_with_q_matrix.data());

        q3::blas::zgemm_("N",
                         "N",
                         &total_projectors,
                         &number_of_vectors,
                         &total_projectors,
                         cast_alpha,
                         cast_q_matrix,
                         &total_projectors,
                         cast_projector_overlaps,
                         &total_projectors,
                         cast_zero,
                         cast_projector_overlaps_with_q_matrix,
                         &total_projectors);

        // Step 3: S|ψ⟩ += β × projector_overlaps_with_q_matrix
        Complex beta_blas_one(1.0, 0.0);
        auto* cast_one = reinterpret_cast<const std::complex<double>*>(&beta_blas_one);
        auto* cast_kleinman_bylander =
            reinterpret_cast<const std::complex<double>*>(kleinman_bylander_potential_.data());
        auto* cast_overlap_wavefunctions =
            reinterpret_cast<std::complex<double>*>(overlap_wavefunctions);

        q3::blas::zgemm_("N",
                         "N",
                         &number_of_plane_waves,
                         &number_of_vectors,
                         &total_projectors,
                         cast_alpha,
                         cast_kleinman_bylander,
                         &number_of_plane_waves,
                         cast_projector_overlaps_with_q_matrix,
                         &total_projectors,
                         cast_one,
                         cast_overlap_wavefunctions,
                         &number_of_plane_waves);
    }

    void build_dense_matrix(int number_of_plane_waves,
                            std::vector<Complex>& overlap_matrix) const override {
        overlap_matrix.assign(number_of_plane_waves * number_of_plane_waves, Complex(0.0, 0.0));
        for (int g_index = 0; g_index < number_of_plane_waves; g_index++) {
            overlap_matrix[g_index + g_index * number_of_plane_waves] = Complex(1.0, 0.0);
        }

        int projector_index_offset = 0;
        for (int atom_index = 0; atom_index < crystal_structure_.number_of_atoms(); atom_index++) {
            int species_type_index = crystal_structure_.species_type_index(atom_index);
            int number_of_projectors =
                quantum_system_.pseudopotential_basis().number_of_projectors(species_type_index);
            std::vector<int> projector_angular_momenta, projector_magnetic_projections,
                projector_beta_indices;
            for (int beta_index = 0; beta_index < number_of_projectors; beta_index++) {
                int angular_momentum =
                    quantum_system_.pseudopotential_basis().beta_angular_momentum(
                        species_type_index, beta_index);
                for (int magnetic_projection : harmonic_order_sequence(angular_momentum)) {
                    projector_angular_momenta.push_back(angular_momentum);
                    projector_magnetic_projections.push_back(magnetic_projection);
                    projector_beta_indices.push_back(beta_index);
                }
            }
            int number_of_projectors_for_atom = static_cast<int>(projector_angular_momenta.size());
            for (int projector_index_a = 0; projector_index_a < number_of_projectors_for_atom;
                 projector_index_a++) {
                for (int projector_index_b = 0; projector_index_b < number_of_projectors_for_atom;
                     projector_index_b++) {
                    Real overlap_coefficient =
                        static_cast<const UltrasoftPlanewaveKohnShamQuantumSystem&>(quantum_system_)
                            .ultrasoft_basis()
                            .augmentation_overlap_integral(
                                species_type_index,
                                projector_beta_indices[projector_index_a],
                                projector_beta_indices[projector_index_b]);
                    if (projector_angular_momenta[projector_index_a] !=
                            projector_angular_momenta[projector_index_b] ||
                        projector_magnetic_projections[projector_index_a] !=
                            projector_magnetic_projections[projector_index_b])
                        overlap_coefficient = 0.0;
                    if (std::abs(overlap_coefficient) <
                        math::tolerances::d_matrix_element_threshold)
                        continue;
                    for (int g_index = 0; g_index < number_of_plane_waves; g_index++) {
                        for (int plane_wave_index_jg = 0;
                             plane_wave_index_jg < number_of_plane_waves;
                             plane_wave_index_jg++) {
                            overlap_matrix[g_index + plane_wave_index_jg * number_of_plane_waves] +=
                                Complex(overlap_coefficient, 0.0) *
                                kleinman_bylander_potential_[projector_index_offset +
                                                             projector_index_a][g_index] *
                                std::conj(kleinman_bylander_potential_[projector_index_offset +
                                                                       projector_index_b]
                                                                      [plane_wave_index_jg]);
                        }
                    }
                }
            }
            projector_index_offset += number_of_projectors_for_atom;
        }
    }

  private:
    /**
     * Build the dense q-matrix Q(total_projectors × total_projectors) in column-major layout.
     * Q(i,j) = q_ab for the corresponding beta functions if l,m quantum numbers match.
     */
    void build_q_matrix() const {
        int total_projectors = static_cast<int>(kleinman_bylander_potential_.rows());
        cached_q_matrix_.assign(static_cast<size_t>(total_projectors) * total_projectors,
                                Complex(0.0, 0.0));

        int starting_projector_index = 0;
        for (int atom_index = 0; atom_index < crystal_structure_.number_of_atoms(); atom_index++) {
            int species_index = crystal_structure_.species_type_index(atom_index);
            int number_of_projectors =
                quantum_system_.pseudopotential_basis().number_of_projectors(species_index);

            std::vector<int> angular_momentum_l, magnetic_projection_m, beta_projector_index;
            for (int index_projector = 0; index_projector < number_of_projectors;
                 index_projector++) {
                int angular_momentum =
                    quantum_system_.pseudopotential_basis().beta_angular_momentum(species_index,
                                                                                  index_projector);
                for (int magnetic_projection : harmonic_order_sequence(angular_momentum)) {
                    angular_momentum_l.push_back(angular_momentum);
                    magnetic_projection_m.push_back(magnetic_projection);
                    beta_projector_index.push_back(index_projector);
                }
            }

            int number_of_projectors_atom = static_cast<int>(angular_momentum_l.size());
            for (int index_a = 0; index_a < number_of_projectors_atom; index_a++) {
                for (int index_b = 0; index_b < number_of_projectors_atom; index_b++) {
                    if (angular_momentum_l[index_a] != angular_momentum_l[index_b] ||
                        magnetic_projection_m[index_a] != magnetic_projection_m[index_b]) {
                        continue;
                    }
                    int beta_a = beta_projector_index[index_a];
                    int beta_b = beta_projector_index[index_b];
                    Real q_matrix_element =
                        static_cast<const UltrasoftPlanewaveKohnShamQuantumSystem&>(quantum_system_)
                            .ultrasoft_basis()
                            .augmentation_overlap_integral(species_index, beta_a, beta_b);
                    if (std::abs(q_matrix_element) > math::tolerances::d_matrix_element_threshold) {
                        int global_a = starting_projector_index + index_a;
                        int global_b = starting_projector_index + index_b;
                        // Column-major: Q(global_a, global_b)
                        cached_q_matrix_[static_cast<size_t>(global_a) +
                                         static_cast<size_t>(global_b) * total_projectors] =
                            Complex(q_matrix_element, 0.0);
                    }
                }
            }
            starting_projector_index += number_of_projectors_atom;
        }
    }

    mutable std::vector<Complex> cached_q_matrix_;
};

} // namespace q3
