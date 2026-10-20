#pragma once

/**
 * SerialPlanewaveSubspaceManager — CPU serial implementation of subspace operations.
 *
 * Extracted from PlanewaveDavidsonDiagonalizer. Contains all subspace linear algebra:
 * orthogonalization, subspace matrix construction, diagonalization, Ritz vector
 * extraction, and subspace collapse.
 *
 * Constructor takes the execution policy for band-group collective operations
 * and creates a SubspaceDiagonalizer for the eigenvalue problem.
 */

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstring>
#include <memory>
#include <vector>

#include "layer_0_core/execution/init.hpp"
#include "layer_0_core/math/linear_algebra/blas.hpp"
#include "layer_0_core/math/tolerances.hpp"
#include "layer_0_core/types/matrix.hpp"
#include "layer_4_computation/diagonalizer/planewave/internal/overlap/init.hpp"
#include "layer_4_computation/diagonalizer/planewave/internal/subspace_manager/init.hpp"
#include "layer_4_computation/diagonalizer/planewave/subspace/factory.hpp"

namespace q3 {

class SerialPlanewaveSubspaceManager : public PlanewaveSubspaceManager {
  public:
    explicit SerialPlanewaveSubspaceManager(const execution::ExecutionPolicy& execution_policy)
        : execution_policy_{execution_policy}
        , subspace_diagonalizer_{SubspaceDiagonalizerFactory::create(execution_policy)} {}

    void set_gamma_origin_is_local(bool origin_is_local) override {
        gamma_origin_is_local_ = origin_is_local;
    }

    int orthogonalize_batch(int starting_vector_index,
                            int number_of_candidates,
                            int number_of_plane_waves,
                            bool is_gamma_only,
                            const OverlapOperator* overlap_operator,
                            Complex* basis_vectors,
                            Complex* overlap_basis_vectors) override {
        if (number_of_candidates == 0) {
            return 0;
        }
        int leading_dimension = std::max(number_of_plane_waves, 1);
        int double_leading_dimension = 2 * leading_dimension;

        /**
         * When the overlap operator is the identity -- every norm-conserving
         * run -- S|v> is |v>, and the caller hands us the same buffer twice
         * rather than storing a second copy of the basis. Reads then resolve
         * correctly on their own; what must not happen is applying an update
         * to the same memory twice, so every write below is guarded.
         */
        const bool overlap_is_aliased = (overlap_basis_vectors == basis_vectors);

        if (!overlap_is_aliased) {
            overlap_operator->apply_batch(
                number_of_plane_waves,
                number_of_candidates,
                &basis_vectors[starting_vector_index * number_of_plane_waves],
                &overlap_basis_vectors[starting_vector_index * number_of_plane_waves]);
        }

        int number_of_existing = starting_vector_index;
        if (number_of_existing > 0) {
            if (is_gamma_only) {
                int double_number_of_plane_waves = 2 * number_of_plane_waves;
                double alpha_two = 2.0;
                double d_beta_zero = 0.0;

                auto* real_basis = reinterpret_cast<double*>(const_cast<Complex*>(basis_vectors));
                auto* real_candidates_sv = reinterpret_cast<double*>(
                    &overlap_basis_vectors[starting_vector_index * number_of_plane_waves]);

                std::vector<double> overlap_coefficients(number_of_existing * number_of_candidates,
                                                         0.0);
                q3::blas::dgemm_("T",
                                 "N",
                                 &number_of_existing,
                                 &number_of_candidates,
                                 &double_number_of_plane_waves,
                                 &alpha_two,
                                 real_basis,
                                 &double_leading_dimension,
                                 real_candidates_sv,
                                 &double_leading_dimension,
                                 &d_beta_zero,
                                 overlap_coefficients.data(),
                                 &number_of_existing);

                if (gamma_origin_is_local_) {
                    for (int candidate_index = 0; candidate_index < number_of_candidates;
                         ++candidate_index) {
                        for (int basis_index = 0; basis_index < number_of_existing; ++basis_index) {
                            Complex basis_g0 = basis_vectors[basis_index * number_of_plane_waves];
                            Complex sv_g0 =
                                overlap_basis_vectors[(starting_vector_index + candidate_index) *
                                                      number_of_plane_waves];
                            overlap_coefficients[candidate_index * number_of_existing +
                                                 basis_index] -=
                                (std::conj(basis_g0) * sv_g0).real();
                        }
                    }
                }

                execution_policy_.reduce_within_group(overlap_coefficients.data(),
                                                      number_of_existing * number_of_candidates);

                auto* real_candidates = reinterpret_cast<double*>(
                    &basis_vectors[starting_vector_index * number_of_plane_waves]);
                auto* real_overlap_basis = reinterpret_cast<double*>(overlap_basis_vectors);

                double alpha_neg_one = -1.0;
                double beta_one = 1.0;
                q3::blas::dgemm_("N",
                                 "N",
                                 &double_number_of_plane_waves,
                                 &number_of_candidates,
                                 &number_of_existing,
                                 &alpha_neg_one,
                                 real_basis,
                                 &double_leading_dimension,
                                 overlap_coefficients.data(),
                                 &number_of_existing,
                                 &beta_one,
                                 real_candidates,
                                 &double_leading_dimension);

                if (!overlap_is_aliased) {
                    /**
                     * Already subtracted from the basis; repeating it here would double it.
                     */
                    q3::blas::dgemm_("N",
                                     "N",
                                     &double_number_of_plane_waves,
                                     &number_of_candidates,
                                     &number_of_existing,
                                     &alpha_neg_one,
                                     real_overlap_basis,
                                     &double_leading_dimension,
                                     overlap_coefficients.data(),
                                     &number_of_existing,
                                     &beta_one,
                                     real_candidates_sv,
                                     &double_leading_dimension);
                }
            } else {
                Complex alpha_one(1.0, 0.0), beta_zero(0.0, 0.0);
                auto* cast_basis =
                    reinterpret_cast<std::complex<double>*>(const_cast<Complex*>(basis_vectors));
                auto* cast_candidates_sv = reinterpret_cast<std::complex<double>*>(
                    &overlap_basis_vectors[starting_vector_index * number_of_plane_waves]);
                auto* cast_alpha = reinterpret_cast<std::complex<double>*>(&alpha_one);
                auto* cast_beta = reinterpret_cast<std::complex<double>*>(&beta_zero);

                std::vector<Complex> overlap_coefficients(number_of_existing * number_of_candidates,
                                                          Complex(0.0, 0.0));
                auto* cast_coeffs =
                    reinterpret_cast<std::complex<double>*>(overlap_coefficients.data());
                q3::blas::zgemm_("C",
                                 "N",
                                 &number_of_existing,
                                 &number_of_candidates,
                                 &number_of_plane_waves,
                                 cast_alpha,
                                 cast_basis,
                                 &leading_dimension,
                                 cast_candidates_sv,
                                 &leading_dimension,
                                 cast_beta,
                                 cast_coeffs,
                                 &number_of_existing);

                execution_policy_.reduce_within_group(overlap_coefficients.data(),
                                                      number_of_existing * number_of_candidates);

                auto* cast_candidates = reinterpret_cast<std::complex<double>*>(
                    &basis_vectors[starting_vector_index * number_of_plane_waves]);
                auto* cast_overlap_basis =
                    reinterpret_cast<std::complex<double>*>(overlap_basis_vectors);

                Complex alpha_neg_one(-1.0, 0.0), beta_one(1.0, 0.0);
                auto* cast_alpha_neg = reinterpret_cast<std::complex<double>*>(&alpha_neg_one);
                auto* cast_beta_one = reinterpret_cast<std::complex<double>*>(&beta_one);
                q3::blas::zgemm_("N",
                                 "N",
                                 &number_of_plane_waves,
                                 &number_of_candidates,
                                 &number_of_existing,
                                 cast_alpha_neg,
                                 cast_basis,
                                 &leading_dimension,
                                 cast_coeffs,
                                 &number_of_existing,
                                 cast_beta_one,
                                 cast_candidates,
                                 &leading_dimension);

                if (!overlap_is_aliased) {
                    /**
                     * Already subtracted from the basis; repeating it here would double it.
                     */
                    q3::blas::zgemm_("N",
                                     "N",
                                     &number_of_plane_waves,
                                     &number_of_candidates,
                                     &number_of_existing,
                                     cast_alpha_neg,
                                     cast_overlap_basis,
                                     &leading_dimension,
                                     cast_coeffs,
                                     &number_of_existing,
                                     cast_beta_one,
                                     cast_candidates_sv,
                                     &leading_dimension);
                }
            }
        }

        // Attempt Cholesky QR to orthogonalize candidates against each other in batch
        if (is_gamma_only) {
            int double_number_of_plane_waves = 2 * number_of_plane_waves;
            auto* real_candidates = reinterpret_cast<double*>(
                &basis_vectors[starting_vector_index * number_of_plane_waves]);
            auto* real_candidates_sv = reinterpret_cast<double*>(
                &overlap_basis_vectors[starting_vector_index * number_of_plane_waves]);

            std::vector<double> cholesky_factor(number_of_candidates * number_of_candidates, 0.0);
            double alpha_two = 2.0;
            double beta_zero = 0.0;
            q3::blas::dgemm_("T",
                             "N",
                             &number_of_candidates,
                             &number_of_candidates,
                             &double_number_of_plane_waves,
                             &alpha_two,
                             real_candidates,
                             &double_leading_dimension,
                             real_candidates_sv,
                             &double_leading_dimension,
                             &beta_zero,
                             cholesky_factor.data(),
                             &number_of_candidates);

            if (gamma_origin_is_local_) {
                for (int column_index = 0; column_index < number_of_candidates; ++column_index) {
                    for (int row_index = 0; row_index < number_of_candidates; ++row_index) {
                        Complex vi_g0 = basis_vectors[(starting_vector_index + row_index) *
                                                      number_of_plane_waves];
                        Complex svj_g0 =
                            overlap_basis_vectors[(starting_vector_index + column_index) *
                                                  number_of_plane_waves];
                        cholesky_factor[column_index * number_of_candidates + row_index] -=
                            (std::conj(vi_g0) * svj_g0).real();
                    }
                }
            }

            execution_policy_.reduce_within_group(cholesky_factor.data(),
                                                  number_of_candidates * number_of_candidates);

            int info = 0;
            q3::blas::dpotrf_(
                "U", &number_of_candidates, cholesky_factor.data(), &number_of_candidates, &info);
            if (info == 0) {
                double d_alpha_one = 1.0;
                q3::blas::dtrsm_("R",
                                 "U",
                                 "N",
                                 "N",
                                 &double_number_of_plane_waves,
                                 &number_of_candidates,
                                 &d_alpha_one,
                                 cholesky_factor.data(),
                                 &number_of_candidates,
                                 real_candidates,
                                 &double_leading_dimension);
                if (!overlap_is_aliased) {
                    /**
                     * The basis was already solved against this factor.
                     */
                    q3::blas::dtrsm_("R",
                                     "U",
                                     "N",
                                     "N",
                                     &double_number_of_plane_waves,
                                     &number_of_candidates,
                                     &d_alpha_one,
                                     cholesky_factor.data(),
                                     &number_of_candidates,
                                     real_candidates_sv,
                                     &double_leading_dimension);
                }
                return number_of_candidates;
            }
        } else {
            auto* cast_candidates = reinterpret_cast<std::complex<double>*>(
                &basis_vectors[starting_vector_index * number_of_plane_waves]);
            auto* cast_candidates_sv = reinterpret_cast<std::complex<double>*>(
                &overlap_basis_vectors[starting_vector_index * number_of_plane_waves]);

            std::vector<Complex> cholesky_factor(number_of_candidates * number_of_candidates,
                                                 Complex(0.0, 0.0));
            Complex alpha_one(1.0, 0.0), beta_zero(0.0, 0.0);
            auto* cast_alpha = reinterpret_cast<std::complex<double>*>(&alpha_one);
            auto* cast_beta = reinterpret_cast<std::complex<double>*>(&beta_zero);
            auto* cast_cholesky_factor =
                reinterpret_cast<std::complex<double>*>(cholesky_factor.data());
            q3::blas::zgemm_("C",
                             "N",
                             &number_of_candidates,
                             &number_of_candidates,
                             &number_of_plane_waves,
                             cast_alpha,
                             cast_candidates,
                             &leading_dimension,
                             cast_candidates_sv,
                             &leading_dimension,
                             cast_beta,
                             cast_cholesky_factor,
                             &number_of_candidates);

            execution_policy_.reduce_within_group(cholesky_factor.data(),
                                                  number_of_candidates * number_of_candidates);

            int info = 0;
            q3::blas::zpotrf_(
                "U", &number_of_candidates, cast_cholesky_factor, &number_of_candidates, &info);
            if (info == 0) {
                Complex z_alpha_one(1.0, 0.0);
                auto* cast_alpha_one = reinterpret_cast<std::complex<double>*>(&z_alpha_one);
                q3::blas::ztrsm_("R",
                                 "U",
                                 "N",
                                 "N",
                                 &number_of_plane_waves,
                                 &number_of_candidates,
                                 cast_alpha_one,
                                 cast_cholesky_factor,
                                 &number_of_candidates,
                                 cast_candidates,
                                 &leading_dimension);
                if (!overlap_is_aliased) {
                    /**
                     * The basis was already solved against this factor.
                     */
                    q3::blas::ztrsm_("R",
                                     "U",
                                     "N",
                                     "N",
                                     &number_of_plane_waves,
                                     &number_of_candidates,
                                     cast_alpha_one,
                                     cast_cholesky_factor,
                                     &number_of_candidates,
                                     cast_candidates_sv,
                                     &leading_dimension);
                }
                return number_of_candidates;
            }
        }

        // Fallback: Modified Gram-Schmidt loop
        int accepted_count = 0;
        int one_increment = 1;
        std::vector<double> batch_coefficients_buffer(number_of_candidates, 0.0);
        std::vector<Complex> z_batch_coefficients_buffer(number_of_candidates, Complex(0.0, 0.0));

        for (int candidate_index = 0; candidate_index < number_of_candidates; ++candidate_index) {
            int source_index = starting_vector_index + candidate_index;
            int destination_index = starting_vector_index + accepted_count;

            if (destination_index != source_index) {
                std::memcpy(&basis_vectors[destination_index * number_of_plane_waves],
                            &basis_vectors[source_index * number_of_plane_waves],
                            number_of_plane_waves * sizeof(Complex));
                if (!overlap_is_aliased) {
                    /**
                     * The basis copy above already moved these coefficients.
                     */
                    std::memcpy(&overlap_basis_vectors[destination_index * number_of_plane_waves],
                                &overlap_basis_vectors[source_index * number_of_plane_waves],
                                number_of_plane_waves * sizeof(Complex));
                }
            }

            auto* current_vector = reinterpret_cast<std::complex<double>*>(
                &basis_vectors[destination_index * number_of_plane_waves]);
            auto* current_overlap = reinterpret_cast<std::complex<double>*>(
                &overlap_basis_vectors[destination_index * number_of_plane_waves]);

            int number_of_batch_existing = destination_index - starting_vector_index;
            if (number_of_batch_existing > 0) {
                if (is_gamma_only) {
                    int double_number_of_plane_waves = 2 * number_of_plane_waves;
                    double alpha_two = 2.0;
                    double d_beta_zero = 0.0;
                    int one_col = 1;

                    auto* real_batch_basis = reinterpret_cast<double*>(
                        &basis_vectors[starting_vector_index * number_of_plane_waves]);
                    auto* real_sv = reinterpret_cast<double*>(current_overlap);

                    q3::blas::dgemm_("T",
                                     "N",
                                     &number_of_batch_existing,
                                     &one_col,
                                     &double_number_of_plane_waves,
                                     &alpha_two,
                                     real_batch_basis,
                                     &double_leading_dimension,
                                     real_sv,
                                     &double_leading_dimension,
                                     &d_beta_zero,
                                     batch_coefficients_buffer.data(),
                                     &number_of_batch_existing);

                    if (gamma_origin_is_local_) {
                        for (int basis_index = 0; basis_index < number_of_batch_existing;
                             ++basis_index) {
                            Complex basis_g0 = basis_vectors[(starting_vector_index + basis_index) *
                                                             number_of_plane_waves];
                            Complex sv_g0 =
                                overlap_basis_vectors[destination_index * number_of_plane_waves];
                            batch_coefficients_buffer[basis_index] -=
                                (std::conj(basis_g0) * sv_g0).real();
                        }
                    }

                    execution_policy_.reduce_within_group(batch_coefficients_buffer.data(),
                                                          number_of_batch_existing);

                    double alpha_neg_one = -1.0;
                    double beta_one = 1.0;
                    q3::blas::dgemv_("N",
                                     &double_number_of_plane_waves,
                                     &number_of_batch_existing,
                                     &alpha_neg_one,
                                     real_batch_basis,
                                     &double_leading_dimension,
                                     batch_coefficients_buffer.data(),
                                     &one_increment,
                                     &beta_one,
                                     reinterpret_cast<double*>(current_vector),
                                     &one_increment);

                    auto* real_batch_overlap = reinterpret_cast<double*>(
                        &overlap_basis_vectors[starting_vector_index * number_of_plane_waves]);
                    if (!overlap_is_aliased) {
                        /**
                         * Already subtracted from the basis vector.
                         */
                        q3::blas::dgemv_("N",
                                         &double_number_of_plane_waves,
                                         &number_of_batch_existing,
                                         &alpha_neg_one,
                                         real_batch_overlap,
                                         &double_leading_dimension,
                                         batch_coefficients_buffer.data(),
                                         &one_increment,
                                         &beta_one,
                                         reinterpret_cast<double*>(current_overlap),
                                         &one_increment);
                    }
                } else {
                    int one_col = 1;
                    Complex alpha_one(1.0, 0.0), beta_zero(0.0, 0.0);
                    auto* cast_batch_basis = reinterpret_cast<std::complex<double>*>(
                        &basis_vectors[starting_vector_index * number_of_plane_waves]);
                    auto* cast_alpha = reinterpret_cast<std::complex<double>*>(&alpha_one);
                    auto* cast_beta = reinterpret_cast<std::complex<double>*>(&beta_zero);
                    auto* cast_coeffs =
                        reinterpret_cast<std::complex<double>*>(z_batch_coefficients_buffer.data());

                    q3::blas::zgemm_("C",
                                     "N",
                                     &number_of_batch_existing,
                                     &one_col,
                                     &number_of_plane_waves,
                                     cast_alpha,
                                     cast_batch_basis,
                                     &leading_dimension,
                                     current_overlap,
                                     &leading_dimension,
                                     cast_beta,
                                     cast_coeffs,
                                     &number_of_batch_existing);

                    execution_policy_.reduce_within_group(z_batch_coefficients_buffer.data(),
                                                          number_of_batch_existing);

                    Complex alpha_neg_one(-1.0, 0.0), beta_one(1.0, 0.0);
                    auto* cast_alpha_neg = reinterpret_cast<std::complex<double>*>(&alpha_neg_one);
                    auto* cast_beta_one = reinterpret_cast<std::complex<double>*>(&beta_one);
                    q3::blas::zgemv_("N",
                                     &number_of_plane_waves,
                                     &number_of_batch_existing,
                                     cast_alpha_neg,
                                     cast_batch_basis,
                                     &leading_dimension,
                                     cast_coeffs,
                                     &one_increment,
                                     cast_beta_one,
                                     current_vector,
                                     &one_increment);

                    auto* cast_batch_overlap = reinterpret_cast<std::complex<double>*>(
                        &overlap_basis_vectors[starting_vector_index * number_of_plane_waves]);
                    if (!overlap_is_aliased) {
                        /**
                         * Already subtracted from the basis vector.
                         */
                        q3::blas::zgemv_("N",
                                         &number_of_plane_waves,
                                         &number_of_batch_existing,
                                         cast_alpha_neg,
                                         cast_batch_overlap,
                                         &leading_dimension,
                                         cast_coeffs,
                                         &one_increment,
                                         cast_beta_one,
                                         current_overlap,
                                         &one_increment);
                    }
                }
            }

            Real norm_squared = gamma_aware_real_dot(
                is_gamma_only, number_of_plane_waves, current_vector, current_overlap);

            if (norm_squared > math::tolerances::iterative_solver_norm_threshold) {
                double inverse_norm = 1.0 / std::sqrt(norm_squared);
                q3::blas::zdscal_(
                    &number_of_plane_waves, &inverse_norm, current_vector, &one_increment);
                if (!overlap_is_aliased) {
                    /**
                     * The basis vector carries the scaling already.
                     */
                    q3::blas::zdscal_(
                        &number_of_plane_waves, &inverse_norm, current_overlap, &one_increment);
                }
                accepted_count++;
            }
        }
        return accepted_count;
    }

    void update_subspace_matrices(int start_column,
                                  int new_columns,
                                  int total_basis_size,
                                  int maximum_basis_size,
                                  int number_of_plane_waves,
                                  bool is_gamma_only,
                                  const Complex* basis_vectors,
                                  const Complex* hamiltonian_basis_vectors,
                                  const Complex* overlap_basis_vectors,
                                  Complex* subspace_hamiltonian,
                                  Complex* subspace_overlap) override {
        int leading_dimension = std::max(number_of_plane_waves, 1);
        Complex alpha_one(1.0, 0.0), beta_zero(0.0, 0.0);
        auto* cast_basis =
            reinterpret_cast<std::complex<double>*>(const_cast<Complex*>(basis_vectors));
        auto* cast_h_new = reinterpret_cast<std::complex<double>*>(
            const_cast<Complex*>(&hamiltonian_basis_vectors[start_column * number_of_plane_waves]));
        auto* cast_s_new = reinterpret_cast<std::complex<double>*>(
            const_cast<Complex*>(&overlap_basis_vectors[start_column * number_of_plane_waves]));

        if (is_gamma_only) {
            int double_number_of_plane_waves = 2 * number_of_plane_waves;
            int double_leading_dimension = 2 * leading_dimension;
            double alpha_two = 2.0;
            double d_beta_zero = 0.0;

            auto* real_basis = reinterpret_cast<double*>(const_cast<Complex*>(basis_vectors));
            auto* real_h_new = reinterpret_cast<double*>(const_cast<Complex*>(
                &hamiltonian_basis_vectors[start_column * number_of_plane_waves]));
            auto* real_s_new = reinterpret_cast<double*>(
                const_cast<Complex*>(&overlap_basis_vectors[start_column * number_of_plane_waves]));

            std::vector<double> h_block_real(static_cast<size_t>(total_basis_size) * new_columns,
                                             0.0);
            q3::blas::dgemm_("T",
                             "N",
                             &total_basis_size,
                             &new_columns,
                             &double_number_of_plane_waves,
                             &alpha_two,
                             real_basis,
                             &double_leading_dimension,
                             real_h_new,
                             &double_leading_dimension,
                             &d_beta_zero,
                             h_block_real.data(),
                             &total_basis_size);

            std::vector<double> s_block_real(static_cast<size_t>(total_basis_size) * new_columns,
                                             0.0);
            q3::blas::dgemm_("T",
                             "N",
                             &total_basis_size,
                             &new_columns,
                             &double_number_of_plane_waves,
                             &alpha_two,
                             real_basis,
                             &double_leading_dimension,
                             real_s_new,
                             &double_leading_dimension,
                             &d_beta_zero,
                             s_block_real.data(),
                             &total_basis_size);

            for (int col = 0; col < new_columns; ++col) {
                int source_col = start_column + col;
                Complex h_right_g0 = hamiltonian_basis_vectors[source_col * number_of_plane_waves];
                Complex s_right_g0 = overlap_basis_vectors[source_col * number_of_plane_waves];
                for (int row = 0; row < total_basis_size; ++row) {
                    Complex left_g0 = basis_vectors[row * number_of_plane_waves];
                    int real_index = col * total_basis_size + row;
                    if (gamma_origin_is_local_) {
                        h_block_real[real_index] -= (std::conj(left_g0) * h_right_g0).real();
                        s_block_real[real_index] -= (std::conj(left_g0) * s_right_g0).real();
                    }

                    int sub_index = (start_column + col) * maximum_basis_size + row;
                    subspace_hamiltonian[sub_index] = Complex(h_block_real[real_index], 0.0);
                    subspace_overlap[sub_index] = Complex(s_block_real[real_index], 0.0);
                }
            }
        } else {
            auto* cast_alpha = reinterpret_cast<std::complex<double>*>(&alpha_one);
            auto* cast_beta = reinterpret_cast<std::complex<double>*>(&beta_zero);

            auto* cast_h_block = reinterpret_cast<std::complex<double>*>(
                &subspace_hamiltonian[start_column * maximum_basis_size]);
            q3::blas::zgemm_("C",
                             "N",
                             &total_basis_size,
                             &new_columns,
                             &number_of_plane_waves,
                             cast_alpha,
                             cast_basis,
                             &leading_dimension,
                             cast_h_new,
                             &leading_dimension,
                             cast_beta,
                             cast_h_block,
                             &maximum_basis_size);

            auto* cast_s_block = reinterpret_cast<std::complex<double>*>(
                &subspace_overlap[start_column * maximum_basis_size]);
            q3::blas::zgemm_("C",
                             "N",
                             &total_basis_size,
                             &new_columns,
                             &number_of_plane_waves,
                             cast_alpha,
                             cast_basis,
                             &leading_dimension,
                             cast_s_new,
                             &leading_dimension,
                             cast_beta,
                             cast_s_block,
                             &maximum_basis_size);
        }

        int new_block_size = maximum_basis_size * new_columns;
        int new_block_offset = start_column * maximum_basis_size;
        execution_policy_.reduce_within_group(&subspace_hamiltonian[new_block_offset],
                                              new_block_size);
        execution_policy_.reduce_within_group(&subspace_overlap[new_block_offset], new_block_size);
    }

    int diagonalize_subspace(int basis_size,
                             int maximum_basis_size,
                             bool is_gamma_only,
                             const Complex* subspace_hamiltonian,
                             const Complex* subspace_overlap,
                             Real* eigenvalues,
                             Complex* eigenvector_matrix) override {
        /**
         * The solver overwrites both inputs, and the caller's matrices carry
         * the next iteration's incremental update, so it works on copies. The
         * Hamiltonian's copy is the eigenvector buffer itself -- the solver
         * returns the eigenvectors where the matrix was -- and only the overlap
         * needs a buffer of its own. Both are compacted from the caller's
         * leading dimension to basis_size. That is two matrices of basis_size
         * squared here rather than three, on every rank.
         */
        diagonalize_overlap_copy_workspace_.resize(static_cast<std::size_t>(basis_size) *
                                                   basis_size);
        for (int column = 0; column < basis_size; ++column) {
            for (int row = 0; row < basis_size; ++row) {
                eigenvector_matrix[column * basis_size + row] =
                    subspace_hamiltonian[column * maximum_basis_size + row];
                diagonalize_overlap_copy_workspace_[column * basis_size + row] =
                    subspace_overlap[column * maximum_basis_size + row];
            }
        }

        /**
         * Which ranks run the eigensolver, in three cases:
         *
         *   - a collective solver (ScaLAPACK) must be entered by EVERY rank of
         *     the group — gating on the root would deadlock — and returns
         *     redundant results, so no broadcast follows;
         *   - replicated coefficient vectors (group_size() <= 1) mean every rank
         *     already holds the same matrices, so each solves its own copy and
         *     there is no group to broadcast over. This includes the fallback
         *     that disables band distribution AFTER this manager was built, where
         *     group_rank_index() keeps its pool-local value and is not
         *     necessarily 0;
         *   - distributed vectors with a non-collective solver: the group root
         *     solves and broadcasts.
         *
         * The two conditions must agree. Testing group_rank_index() == 0 to
         * solve but group_size() > 1 to broadcast leaves the non-root ranks of a
         * replicated group with neither a result nor a broadcast, and they then
         * diverge from the root at the next collective.
         */
        const bool all_ranks_solve = subspace_diagonalizer_->is_collective();
        const bool vectors_are_replicated = execution_policy_.group_size() <= 1;
        const bool this_rank_solves =
            all_ranks_solve || vectors_are_replicated || execution_policy_.group_rank_index() == 0;
        int info = 0;
        if (this_rank_solves) {
            info = subspace_diagonalizer_->solve(basis_size,
                                                 eigenvector_matrix,
                                                 diagonalize_overlap_copy_workspace_.data(),
                                                 eigenvalues,
                                                 eigenvector_matrix,
                                                 is_gamma_only);
        }

        if (!all_ranks_solve && !vectors_are_replicated) {
            execution_policy_.broadcast_from_group_root(eigenvector_matrix,
                                                        basis_size * basis_size);
            execution_policy_.broadcast_from_group_root(eigenvalues, basis_size);
            Real info_as_real = static_cast<Real>(info);
            execution_policy_.broadcast_from_group_root(&info_as_real, 1);
            info = static_cast<int>(info_as_real);
        }

        if (info != 0) {
            log::warn("LAPACK eigensolver returned info={} in Davidson subspace (basis_size={})",
                      info,
                      basis_size);
        }
        return info;
    }

    /**
     * output = basis * coefficients, divided by band when bands are distributed.
     *
     * The product is (plane_waves x basis_size) times (basis_size x bands), so
     * its columns are bands and each rank can form its own contiguous range
     * against the matching column slice of the coefficient matrix. One
     * allgather then completes the array on every rank. Because every block has
     * exactly one source, the assembled result is identical on all ranks -- the
     * property the replicated basis depends on and redundant ZGEMMs cannot give.
     * The inputs must already agree across ranks, which the Davidson loop
     * guarantees by broadcasting the basis and the eigenvector matrix.
     */
    void multiply_basis_by_coefficients_across_bands(int number_of_plane_waves,
                                                     int number_of_bands,
                                                     int basis_size,
                                                     const std::complex<double>* basis,
                                                     const std::complex<double>* coefficients,
                                                     std::complex<double>* output) const {
        int leading_dimension = std::max(number_of_plane_waves, 1);
        Complex alpha_one(1.0, 0.0), beta_zero(0.0, 0.0);
        auto* cast_alpha = reinterpret_cast<std::complex<double>*>(&alpha_one);
        auto* cast_beta = reinterpret_cast<std::complex<double>*>(&beta_zero);
        int band_group_size = execution_policy_.band_distribution_size();
        if (band_group_size <= 1 || number_of_bands < band_group_size) {
            q3::blas::zgemm_("N",
                             "N",
                             &number_of_plane_waves,
                             &number_of_bands,
                             &basis_size,
                             cast_alpha,
                             const_cast<std::complex<double>*>(basis),
                             &leading_dimension,
                             const_cast<std::complex<double>*>(coefficients),
                             &basis_size,
                             cast_beta,
                             output,
                             &leading_dimension);
            return;
        }
        auto partition =
            execution::partition_bands(number_of_bands,
                                       band_group_size,
                                       execution_policy_.band_distribution_rank_index(),
                                       number_of_plane_waves);
        int local_bands = partition.local_number_of_bands;
        if (local_bands > 0) {
            std::size_t column_offset = static_cast<std::size_t>(partition.local_first_band);
            q3::blas::zgemm_(
                "N",
                "N",
                &number_of_plane_waves,
                &local_bands,
                &basis_size,
                cast_alpha,
                const_cast<std::complex<double>*>(basis),
                &leading_dimension,
                const_cast<std::complex<double>*>(coefficients + column_offset * basis_size),
                &basis_size,
                cast_beta,
                output + column_offset * number_of_plane_waves,
                &leading_dimension);
        }
        execution_policy_.allgather_across_bands(reinterpret_cast<Complex*>(output),
                                                 partition.counts.data(),
                                                 partition.displacements.data());
    }

    int collapse_subspace(int basis_size,
                          int number_of_plane_waves,
                          int number_of_bands,
                          int maximum_basis_size,
                          bool is_gamma_only,
                          bool is_identity_overlap,
                          const OverlapOperator* overlap_operator,
                          const Complex* subspace_hamiltonian,
                          const Complex* subspace_overlap,
                          Complex* basis_vectors,
                          Complex* hamiltonian_basis_vectors,
                          Complex* overlap_basis_vectors,
                          Complex* rotation_scratch) override {
        std::vector<Real> eigenvalues_expanded(maximum_basis_size, 0.0);
        std::vector<Complex> eigenvector_matrix(static_cast<std::size_t>(basis_size) * basis_size);
        /**
         * The subspace the caller holds is complete through basis_size, so its
         * eigenvectors are the rotation. Rebuilding it from the vectors is kept
         * only as the fallback, for the recovery collapse that follows a failed
         * diagonalisation of these very matrices.
         */
        int info = -1;
        if (subspace_hamiltonian != nullptr && subspace_overlap != nullptr) {
            info = diagonalize_subspace(basis_size,
                                        maximum_basis_size,
                                        is_gamma_only,
                                        subspace_hamiltonian,
                                        subspace_overlap,
                                        eigenvalues_expanded.data(),
                                        eigenvector_matrix.data());
        }
        if (info != 0) {
            build_subspace_matrices_and_diagonalize(basis_size,
                                                    number_of_plane_waves,
                                                    is_gamma_only,
                                                    overlap_operator,
                                                    basis_vectors,
                                                    hamiltonian_basis_vectors,
                                                    overlap_basis_vectors,
                                                    eigenvalues_expanded,
                                                    eigenvector_matrix);
        }
        /**
         * This diagonalisation is local to the collapse, so it is not covered
         * by the Davidson loop's eigenvector broadcast. Without republishing it,
         * each rank would rotate its band range by a slightly different matrix
         * and the allgathered basis would be a stitch of inconsistent rotations.
         */
        execution_policy_.broadcast_across_bands(eigenvector_matrix.data(),
                                                 static_cast<int>(eigenvector_matrix.size()));

        /**
         * Only the first number_of_bands columns survive a collapse, so only
         * that much is rotated and only that much is copied back. Columns past
         * it are left as they were rather than zeroed: every reader downstream
         * is bounded by basis_size, which the caller resets to number_of_bands,
         * and the expansion writes each new column before anything reads it.
         */
        std::size_t collapsed_size = static_cast<std::size_t>(number_of_bands) *
                                     static_cast<std::size_t>(number_of_plane_waves);

        auto* cast_basis = reinterpret_cast<std::complex<double>*>(basis_vectors);
        auto* cast_h_basis = reinterpret_cast<std::complex<double>*>(hamiltonian_basis_vectors);
        auto* cast_s_basis = reinterpret_cast<std::complex<double>*>(overlap_basis_vectors);
        auto* cast_eigvec = reinterpret_cast<std::complex<double>*>(eigenvector_matrix.data());
        auto* cast_scratch = reinterpret_cast<std::complex<double>*>(rotation_scratch);

        /**
         * One scratch buffer does for all three rotations, because each one
         * reads a different array than it writes: rotate into the scratch,
         * copy back, move on. The overlap array is the exception and needs no
         * rotation at all when it aliases the basis -- it already holds the
         * rotated vectors by the time the first copy-back lands.
         */
        multiply_basis_by_coefficients_across_bands(number_of_plane_waves,
                                                    number_of_bands,
                                                    basis_size,
                                                    cast_basis,
                                                    cast_eigvec,
                                                    cast_scratch);
        std::memcpy(basis_vectors, rotation_scratch, collapsed_size * sizeof(Complex));

        multiply_basis_by_coefficients_across_bands(number_of_plane_waves,
                                                    number_of_bands,
                                                    basis_size,
                                                    cast_h_basis,
                                                    cast_eigvec,
                                                    cast_scratch);
        std::memcpy(hamiltonian_basis_vectors, rotation_scratch, collapsed_size * sizeof(Complex));

        /**
         * Skip the overlap rotation only when the overlap array is the basis
         * itself. An identity overlap is not enough: the CPU workspace aliases
         * S|V> to |V> for it, so rotating the basis has already rotated S|V>,
         * but the device workspace keeps S|V> in a buffer of its own. Keying
         * the skip on the flag left that buffer unrotated after every collapse,
         * so the next orthogonalisation and subspace update read stale S|V>,
         * and a run that collapsed often converged to the wrong answer -- 1.77
         * Ry low on the two-atom silicon deck at expansion factor 2, which
         * collapses every iteration. Factor 4 hid it: small decks converge
         * before the basis fills.
         */
        const bool overlap_needs_rotation =
            !is_identity_overlap || overlap_basis_vectors != basis_vectors;
        if (overlap_needs_rotation) {
            multiply_basis_by_coefficients_across_bands(number_of_plane_waves,
                                                        number_of_bands,
                                                        basis_size,
                                                        cast_s_basis,
                                                        cast_eigvec,
                                                        cast_scratch);
            std::memcpy(overlap_basis_vectors, rotation_scratch, collapsed_size * sizeof(Complex));
        }

        enforce_gamma_g0_real(is_gamma_only, number_of_plane_waves, number_of_bands, basis_vectors);
        enforce_gamma_g0_real(
            is_gamma_only, number_of_plane_waves, number_of_bands, hamiltonian_basis_vectors);
        enforce_gamma_g0_real(
            is_gamma_only, number_of_plane_waves, number_of_bands, overlap_basis_vectors);
        return number_of_bands;
    }

    void extract_eigenvectors(int basis_size,
                              int number_of_plane_waves,
                              int number_of_bands,
                              int kpoint_index_state,
                              bool is_gamma_only,
                              const Complex* eigenvector_matrix,
                              const Complex* basis_vectors,
                              const Real* eigenvalues,
                              Complex* rotated_scratch,
                              PlanewaveKohnShamQuantumSystem& quantum_system) override {
        int leading_dimension = std::max(number_of_plane_waves, 1);
        int leading_basis_dimension = std::max(basis_size, 1);
        Complex alpha_one(1.0, 0.0), beta_zero(0.0, 0.0);
        auto* cast_basis =
            reinterpret_cast<std::complex<double>*>(const_cast<Complex*>(basis_vectors));
        auto* cast_eigvec =
            reinterpret_cast<std::complex<double>*>(const_cast<Complex*>(eigenvector_matrix));
        auto* cast_result = reinterpret_cast<std::complex<double>*>(rotated_scratch);
        auto* cast_alpha = reinterpret_cast<std::complex<double>*>(&alpha_one);
        auto* cast_beta = reinterpret_cast<std::complex<double>*>(&beta_zero);

        q3::blas::zgemm_("N",
                         "N",
                         &number_of_plane_waves,
                         &number_of_bands,
                         &basis_size,
                         cast_alpha,
                         cast_basis,
                         &leading_dimension,
                         cast_eigvec,
                         &leading_basis_dimension,
                         cast_beta,
                         cast_result,
                         &leading_dimension);

        enforce_gamma_g0_real(
            is_gamma_only, number_of_plane_waves, number_of_bands, rotated_scratch);

        for (int band_index = 0; band_index < number_of_bands; ++band_index) {
            quantum_system.set_eigenvalue(kpoint_index_state, band_index, eigenvalues[band_index]);
            Complex* band_data =
                quantum_system.wavefunction_band_data(kpoint_index_state, band_index);
            std::copy(rotated_scratch + band_index * number_of_plane_waves,
                      rotated_scratch + (band_index + 1) * number_of_plane_waves,
                      band_data);
        }
    }

    int build_correction_vectors(int basis_size,
                                 int number_of_plane_waves,
                                 int number_of_bands,
                                 int maximum_candidates,
                                 bool is_gamma_only,
                                 const Complex* eigenvector_matrix,
                                 int eigenvector_leading_dimension,
                                 const Real* eigenvalues,
                                 const std::vector<bool>& band_converged,
                                 const Real* kinetic_energy_squared,
                                 const Complex* hamiltonian_basis_vectors,
                                 const Complex* overlap_basis_vectors,
                                 Complex* correction_columns) override {
        /**
         * Gather the unconverged bands' eigenvector columns, compacted, so the
         * two GEMMs below write consecutive columns and the caller can hand
         * the block straight to orthogonalisation.
         */
        std::vector<int> source_bands;
        source_bands.reserve(static_cast<std::size_t>(number_of_bands));
        for (int band_index = 0; band_index < number_of_bands; ++band_index) {
            if (!band_converged[band_index] &&
                static_cast<int>(source_bands.size()) < maximum_candidates) {
                source_bands.push_back(band_index);
            }
        }
        int number_of_candidates = static_cast<int>(source_bands.size());
        if (number_of_candidates == 0) {
            return 0;
        }

        std::vector<Complex> packed_coefficients(static_cast<std::size_t>(basis_size) *
                                                     static_cast<std::size_t>(number_of_candidates),
                                                 Complex(0.0, 0.0));
        std::vector<Real> candidate_eigenvalues(static_cast<std::size_t>(number_of_candidates),
                                                0.0);
        for (int candidate_index = 0; candidate_index < number_of_candidates; ++candidate_index) {
            int band_index = source_bands[static_cast<std::size_t>(candidate_index)];
            candidate_eigenvalues[static_cast<std::size_t>(candidate_index)] =
                eigenvalues[band_index];
            std::memcpy(
                &packed_coefficients[static_cast<std::size_t>(candidate_index) * basis_size],
                eigenvector_matrix + static_cast<std::size_t>(band_index) *
                                         static_cast<std::size_t>(eigenvector_leading_dimension),
                static_cast<std::size_t>(basis_size) * sizeof(Complex));
        }

        /**
         * R = (SV)c scaled by -eps, then (HV)c accumulated on top. Under band
         * distribution each rank owns a slice of the candidate columns and
         * both GEMMs plus the scaling touch only that slice, so one allgather
         * at the end republishes the block.
         */
        int band_group_size = execution_policy_.band_distribution_size();
        bool distribute = band_group_size > 1 && number_of_candidates >= band_group_size;
        int first_column = 0;
        int local_columns = number_of_candidates;
        execution::BandPartition partition;
        if (distribute) {
            partition = execution::partition_bands(number_of_candidates,
                                                   band_group_size,
                                                   execution_policy_.band_distribution_rank_index(),
                                                   number_of_plane_waves);
            first_column = partition.local_first_band;
            local_columns = partition.local_number_of_bands;
        }

        if (local_columns > 0) {
            int leading_dimension = std::max(number_of_plane_waves, 1);
            Complex alpha_one(1.0, 0.0), beta_zero(0.0, 0.0), beta_one(1.0, 0.0);
            auto* cast_alpha = reinterpret_cast<std::complex<double>*>(&alpha_one);
            auto* cast_beta_zero = reinterpret_cast<std::complex<double>*>(&beta_zero);
            auto* cast_beta_one = reinterpret_cast<std::complex<double>*>(&beta_one);
            auto* cast_coefficients = reinterpret_cast<std::complex<double>*>(
                packed_coefficients.data() + static_cast<std::size_t>(first_column) * basis_size);
            auto* cast_destination = reinterpret_cast<std::complex<double>*>(
                correction_columns +
                static_cast<std::size_t>(first_column) * number_of_plane_waves);

            q3::blas::zgemm_("N",
                             "N",
                             &number_of_plane_waves,
                             &local_columns,
                             &basis_size,
                             cast_alpha,
                             reinterpret_cast<std::complex<double>*>(
                                 const_cast<Complex*>(overlap_basis_vectors)),
                             &leading_dimension,
                             cast_coefficients,
                             &basis_size,
                             cast_beta_zero,
                             cast_destination,
                             &leading_dimension);

            for (int local_index = 0; local_index < local_columns; ++local_index) {
                Real negative_eigenvalue =
                    -candidate_eigenvalues[static_cast<std::size_t>(first_column + local_index)];
                Complex* column =
                    correction_columns +
                    static_cast<std::size_t>(first_column + local_index) * number_of_plane_waves;
                for (int g_index = 0; g_index < number_of_plane_waves; ++g_index) {
                    column[g_index] *= negative_eigenvalue;
                }
            }

            q3::blas::zgemm_("N",
                             "N",
                             &number_of_plane_waves,
                             &local_columns,
                             &basis_size,
                             cast_alpha,
                             reinterpret_cast<std::complex<double>*>(
                                 const_cast<Complex*>(hamiltonian_basis_vectors)),
                             &leading_dimension,
                             cast_coefficients,
                             &basis_size,
                             cast_beta_one,
                             cast_destination,
                             &leading_dimension);

            apply_preconditioner_to_columns(local_columns,
                                            number_of_plane_waves,
                                            kinetic_energy_squared,
                                            candidate_eigenvalues.data() + first_column,
                                            correction_columns +
                                                static_cast<std::size_t>(first_column) *
                                                    number_of_plane_waves);
        }

        if (distribute) {
            execution_policy_.allgather_across_bands(
                correction_columns, partition.counts.data(), partition.displacements.data());
        }

        enforce_gamma_g0_real(
            is_gamma_only, number_of_plane_waves, number_of_candidates, correction_columns);
        return number_of_candidates;
    }

    /**
     * Teter-Payne-Allan preconditioning of a compact block of columns, applied
     * in place. The legacy per-band entry point below delegates to this.
     */
    void apply_preconditioner_to_columns(int number_of_columns,
                                         int number_of_plane_waves,
                                         const Real* kinetic_energy_squared,
                                         const Real* eigenvalues,
                                         Complex* columns) const {
        for (int column_index = 0; column_index < number_of_columns; ++column_index) {
            Complex* column = &columns[static_cast<std::size_t>(column_index) *
                                       static_cast<std::size_t>(number_of_plane_waves)];
            for (int g_index = 0; g_index < number_of_plane_waves; ++g_index) {
                Real value_x = kinetic_energy_squared[g_index] - eigenvalues[column_index];
                Real denominator =
                    0.5 * (1.0 + value_x + std::sqrt(1.0 + (value_x - 1.0) * (value_x - 1.0)));
                if (denominator < math::tolerances::preconditioner_denominator_floor) {
                    denominator = math::tolerances::preconditioner_denominator_floor;
                }
                column[g_index] /= denominator;
            }
        }
    }

    void apply_preconditioner(int number_of_bands,
                              int number_of_plane_waves,
                              const Real* kinetic_energy_squared,
                              const Real* eigenvalues,
                              const std::vector<bool>& band_converged,
                              Complex* residual_wavefunctions) override {
        for (int band_index = 0; band_index < number_of_bands; band_index++) {
            if (band_converged[band_index]) {
                continue;
            }
            Complex* residual = &residual_wavefunctions[band_index * number_of_plane_waves];
            for (int g_vector_index = 0; g_vector_index < number_of_plane_waves; g_vector_index++) {
                Real value_x = kinetic_energy_squared[g_vector_index] - eigenvalues[band_index];
                Real denominator =
                    0.5 * (1.0 + value_x + std::sqrt(1.0 + (value_x - 1.0) * (value_x - 1.0)));
                if (denominator < math::tolerances::preconditioner_denominator_floor) {
                    denominator = math::tolerances::preconditioner_denominator_floor;
                }
                residual[g_vector_index] /= denominator;
            }
        }
    }

  private:
    // ─── Gamma-only helpers ───────────────────────────────────────────

    /**
     * Zero the imaginary part at G = 0 -- the two-band transform needs the
     * origin real -- on the rank whose index 0 is the origin. Anywhere else
     * index 0 is an ordinary G whose imaginary part is data.
     */
    void enforce_gamma_g0_real(bool is_gamma_only,
                               int number_of_plane_waves,
                               int number_of_vectors,
                               Complex* vectors) const {
        if (!is_gamma_only || !gamma_origin_is_local_) {
            return;
        }
        for (int vector_index = 0; vector_index < number_of_vectors; ++vector_index) {
            Complex& g0 = vectors[vector_index * number_of_plane_waves];
            g0 = Complex(g0.real(), 0.0);
        }
    }

    Real gamma_aware_real_dot(bool is_gamma_only,
                              int number_of_plane_waves,
                              const std::complex<double>* vector_a,
                              const std::complex<double>* vector_b) const {
        if (!is_gamma_only) {
            return std::complex<double>(
                       execution_policy_.dot_product(number_of_plane_waves, vector_a, vector_b))
                .real();
        }
        /**
         * 2 Re over this rank's half sphere, less the origin where it lives,
         * then one reduction. The correction has to go in before the sum is
         * shared, or the ranks would not agree on the norm.
         */
        std::complex<double> local_sum(
            q3::blas::q3_complex_conjugate_dot_product(number_of_plane_waves, vector_a, vector_b));
        Real full_sphere = 2.0 * local_sum.real();
        if (gamma_origin_is_local_) {
            full_sphere -= (std::conj(vector_a[0]) * vector_b[0]).real();
        }
        execution_policy_.reduce_within_group(&full_sphere, 1);
        return full_sphere;
    }

    // ─── Build and diag (used by collapse) ───────────────────────────

    void build_subspace_matrices_and_diagonalize(int basis_size,
                                                 int number_of_plane_waves,
                                                 bool is_gamma_only,
                                                 const OverlapOperator* overlap_operator,
                                                 const Complex* basis_vectors,
                                                 const Complex* hamiltonian_basis_vectors,
                                                 const Complex* overlap_basis_vectors,
                                                 std::vector<Real>& eigenvalues,
                                                 std::vector<Complex>& eigenvector_matrix) const {
        int leading_dimension = std::max(number_of_plane_waves, 1);
        eigenvector_matrix.assign(basis_size * basis_size, Complex(0.0, 0.0));
        std::vector<Complex> overlap_matrix(basis_size * basis_size, Complex(0.0, 0.0));

        Complex alpha_one(1.0, 0.0), beta_zero(0.0, 0.0);
        auto* cast_basis =
            reinterpret_cast<std::complex<double>*>(const_cast<Complex*>(basis_vectors));
        auto* cast_h_basis = reinterpret_cast<std::complex<double>*>(
            const_cast<Complex*>(hamiltonian_basis_vectors));
        auto* cast_s_basis =
            reinterpret_cast<std::complex<double>*>(const_cast<Complex*>(overlap_basis_vectors));
        auto* cast_h_sub = reinterpret_cast<std::complex<double>*>(eigenvector_matrix.data());
        auto* cast_s_sub = reinterpret_cast<std::complex<double>*>(overlap_matrix.data());
        auto* cast_alpha = reinterpret_cast<std::complex<double>*>(&alpha_one);
        auto* cast_beta = reinterpret_cast<std::complex<double>*>(&beta_zero);

        q3::blas::zgemm_("C",
                         "N",
                         &basis_size,
                         &basis_size,
                         &number_of_plane_waves,
                         cast_alpha,
                         cast_basis,
                         &leading_dimension,
                         cast_h_basis,
                         &leading_dimension,
                         cast_beta,
                         cast_h_sub,
                         &basis_size);
        q3::blas::zgemm_("C",
                         "N",
                         &basis_size,
                         &basis_size,
                         &number_of_plane_waves,
                         cast_alpha,
                         cast_basis,
                         &leading_dimension,
                         cast_s_basis,
                         &leading_dimension,
                         cast_beta,
                         cast_s_sub,
                         &basis_size);

        if (is_gamma_only) {
            /**
             * The ZGEMMs summed the stored half sphere; the full-sphere value
             * is 2 Re of that with the origin, held once, removed from the
             * double count -- on the rank that holds it.
             */
            for (int column = 0; column < basis_size; ++column) {
                Complex h_right_g0 = hamiltonian_basis_vectors[column * number_of_plane_waves];
                Complex s_right_g0 = overlap_basis_vectors[column * number_of_plane_waves];
                for (int row = 0; row < basis_size; ++row) {
                    Complex left_g0 = basis_vectors[row * number_of_plane_waves];
                    Real h_origin =
                        gamma_origin_is_local_ ? (std::conj(left_g0) * h_right_g0).real() : 0.0;
                    Real s_origin =
                        gamma_origin_is_local_ ? (std::conj(left_g0) * s_right_g0).real() : 0.0;
                    int index = column * basis_size + row;
                    eigenvector_matrix[index] =
                        Complex(2.0 * eigenvector_matrix[index].real() - h_origin, 0.0);
                    overlap_matrix[index] =
                        Complex(2.0 * overlap_matrix[index].real() - s_origin, 0.0);
                }
            }
        }

        execution_policy_.reduce_within_group(eigenvector_matrix.data(), basis_size * basis_size);
        execution_policy_.reduce_within_group(overlap_matrix.data(), basis_size * basis_size);

        {
            int info;
            int diagonalization_type = 1;

            if (is_gamma_only) {
                std::vector<Real> hamiltonian_real(basis_size * basis_size);
                std::vector<Real> overlap_real(basis_size * basis_size);
                for (int element_index = 0; element_index < basis_size * basis_size;
                     ++element_index) {
                    hamiltonian_real[element_index] = eigenvector_matrix[element_index].real();
                    overlap_real[element_index] = overlap_matrix[element_index].real();
                }

                int work_size = std::max(1, 3 * basis_size - 1);
                std::vector<Real> work(work_size);
                Real optimal_work_size;
                int query_size = -1;
                q3::lapack::dsygv_(&diagonalization_type,
                                   "V",
                                   "U",
                                   &basis_size,
                                   hamiltonian_real.data(),
                                   &basis_size,
                                   overlap_real.data(),
                                   &basis_size,
                                   eigenvalues.data(),
                                   &optimal_work_size,
                                   &query_size,
                                   &info);
                work_size = static_cast<int>(optimal_work_size);
                work.resize(work_size);
                q3::lapack::dsygv_(&diagonalization_type,
                                   "V",
                                   "U",
                                   &basis_size,
                                   hamiltonian_real.data(),
                                   &basis_size,
                                   overlap_real.data(),
                                   &basis_size,
                                   eigenvalues.data(),
                                   work.data(),
                                   &work_size,
                                   &info);

                for (int element_index = 0; element_index < basis_size * basis_size;
                     ++element_index) {
                    eigenvector_matrix[element_index] =
                        Complex(hamiltonian_real[element_index], 0.0);
                }
            } else {
                int work_size = 2 * basis_size + basis_size * basis_size;
                std::vector<Complex> work(work_size);
                std::vector<Real> real_work(std::max(1, 3 * basis_size - 2));
                std::complex<double>* cast_work =
                    reinterpret_cast<std::complex<double>*>(work.data());
                q3::lapack::zhegv_(&diagonalization_type,
                                   "V",
                                   "U",
                                   &basis_size,
                                   cast_h_sub,
                                   &basis_size,
                                   cast_s_sub,
                                   &basis_size,
                                   eigenvalues.data(),
                                   cast_work,
                                   &work_size,
                                   real_work.data(),
                                   &info);
            }

            if (info != 0) {
                log::warn(
                    "LAPACK eigensolver returned info={} in Davidson collapse (basis_size={})",
                    info,
                    basis_size);
            }
        }
    }

    // ─── Execution context ──────────────────────────────────────────

    const execution::ExecutionPolicy& execution_policy_;
    bool gamma_origin_is_local_ = true;
    std::unique_ptr<SubspaceDiagonalizer> subspace_diagonalizer_;

    // ─── Pre-allocated workspaces ───────────────────────────────────

    mutable std::vector<Complex> diagonalize_overlap_copy_workspace_;
};

} // namespace q3
