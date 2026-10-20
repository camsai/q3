#pragma once

/**
 * GPUPlanewaveSubspaceManager — GPU implementation of subspace operations.
 *
 * Implements PlanewaveSubspaceManager using GPU BLAS for the large matrix
 * operations (subspace Hamiltonian build, Ritz vector rotation, orthogonalization).
 * Subspace diagonalization always runs on CPU via GPUEigensolver (which
 * internally uses LAPACK — subspace matrices are small N×N).
 *
 * Constructor takes GPUBLAS& and GPUEigensolver& for delegation, plus
 * the execution policy for band-group collectives.
 *
 * No CUDA headers. No Metal headers. Uses only vendor-neutral GPU abstractions.
 */

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstring>
#include <memory>
#include <vector>

#include "io/logger.hpp"
#include "layer_0_core/execution/gpu/device/init.hpp"
#include "layer_0_core/execution/gpu/device/internal/buffer.hpp"
#include "layer_0_core/execution/init.hpp"
#include "layer_0_core/math/linear_algebra/blas.hpp"
#include "layer_0_core/math/linear_algebra/gpu/blas/init.hpp"
#include "layer_0_core/math/tolerances.hpp"
#include "layer_0_core/types/matrix.hpp"
#include "layer_4_computation/diagonalizer/planewave/internal/hamiltonian_applicator/gpu_kernels.hpp"
#include "layer_4_computation/diagonalizer/planewave/internal/overlap/init.hpp"
#include "layer_4_computation/diagonalizer/planewave/internal/subspace_manager/init.hpp"
#include "layer_4_computation/diagonalizer/planewave/subspace/factory.hpp"

namespace q3 {

class GPUPlanewaveSubspaceManager : public PlanewaveSubspaceManager {
  public:
    /**
     * Construct with GPU math wrappers and execution policy.
     *
     * @param gpu_blas          GPU BLAS wrapper for GEMM operations.
     * @param execution_policy  Execution policy for band-group collectives.
     * @param device_backend    GPU device backend for buffer allocation.
     */
    GPUPlanewaveSubspaceManager(
        execution::gpu::blas::GPUBLAS& gpu_blas,
        const execution::ExecutionPolicy& execution_policy,
        std::shared_ptr<execution::gpu::device::GPUDeviceBackend> device_backend)
        : gpu_blas_{gpu_blas}
        , execution_policy_{execution_policy}
        , device_backend_{device_backend}
        , subspace_diagonalizer_{SubspaceDiagonalizerFactory::create(execution_policy)} {}

    /**
     * S-orthogonalize and normalize candidates using GPU BLAS.
     *
     * Uses GPU GEMM for computing overlap coefficients and applying
     * projections. Falls back to modified Gram-Schmidt if Cholesky fails.
     */
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

        // Apply overlap operator to candidates.
        overlap_operator->apply_batch(
            number_of_plane_waves,
            number_of_candidates,
            &basis_vectors[starting_vector_index * number_of_plane_waves],
            &overlap_basis_vectors[starting_vector_index * number_of_plane_waves]);

        int number_of_existing = starting_vector_index;

        // Project out existing basis via GPU GEMM.
        if (number_of_existing > 0 && !is_gamma_only) {
            Complex alpha_one(1.0, 0.0), beta_zero(0.0, 0.0);

            // Compute overlap coefficients: coeffs = V† · S·candidates
            std::vector<Complex> overlap_coefficients(number_of_existing * number_of_candidates,
                                                      Complex(0.0, 0.0));
            execution::gpu::device::GPUDeviceBuffer<Complex> device_overlap_coefficients(
                number_of_existing * number_of_candidates, device_backend_);

            gpu_blas_.gemm('C',
                           'N',
                           number_of_existing,
                           number_of_candidates,
                           number_of_plane_waves,
                           alpha_one,
                           basis_vectors,
                           leading_dimension,
                           &overlap_basis_vectors[starting_vector_index * number_of_plane_waves],
                           leading_dimension,
                           beta_zero,
                           device_overlap_coefficients.data(),
                           number_of_existing);

            device_backend_->synchronize();

            std::memcpy(overlap_coefficients.data(),
                        device_overlap_coefficients.data(),
                        overlap_coefficients.size() * sizeof(Complex));

            execution_policy_.reduce_within_group(overlap_coefficients.data(),
                                                  number_of_existing * number_of_candidates);

            std::memcpy(device_overlap_coefficients.data(),
                        overlap_coefficients.data(),
                        overlap_coefficients.size() * sizeof(Complex));

            // Subtract projections: candidates -= V · coeffs
            Complex alpha_neg_one(-1.0, 0.0), beta_one(1.0, 0.0);
            gpu_blas_.gemm('N',
                           'N',
                           number_of_plane_waves,
                           number_of_candidates,
                           number_of_existing,
                           alpha_neg_one,
                           basis_vectors,
                           leading_dimension,
                           device_overlap_coefficients.data(),
                           number_of_existing,
                           beta_one,
                           &basis_vectors[starting_vector_index * number_of_plane_waves],
                           leading_dimension);

            // Same for overlap vectors: S·candidates -= S·V · coeffs
            gpu_blas_.gemm('N',
                           'N',
                           number_of_plane_waves,
                           number_of_candidates,
                           number_of_existing,
                           alpha_neg_one,
                           overlap_basis_vectors,
                           leading_dimension,
                           device_overlap_coefficients.data(),
                           number_of_existing,
                           beta_one,
                           &overlap_basis_vectors[starting_vector_index * number_of_plane_waves],
                           leading_dimension);

            device_backend_->synchronize();
        } else if (number_of_existing > 0 && is_gamma_only) {
            // Gamma-only: project out existing basis using GPU DGEMM.
            int double_number_of_plane_waves = 2 * number_of_plane_waves;
            int double_leading_dimension = 2 * leading_dimension;
            double alpha_two = 2.0;
            double d_beta_zero = 0.0;

            auto* real_basis = reinterpret_cast<double*>(const_cast<Complex*>(basis_vectors));
            auto* real_candidates_sv = reinterpret_cast<double*>(
                &overlap_basis_vectors[starting_vector_index * number_of_plane_waves]);

            std::vector<double> overlap_coefficients(number_of_existing * number_of_candidates,
                                                     0.0);
            execution::gpu::device::GPUDeviceBuffer<double> device_overlap_coefficients(
                number_of_existing * number_of_candidates, device_backend_);

            // coeffs = V† · S·candidates
            gpu_blas_.dgemm('T',
                            'N',
                            number_of_existing,
                            number_of_candidates,
                            double_number_of_plane_waves,
                            alpha_two,
                            real_basis,
                            double_leading_dimension,
                            real_candidates_sv,
                            double_leading_dimension,
                            d_beta_zero,
                            device_overlap_coefficients.data(),
                            number_of_existing);

            device_backend_->synchronize();

            std::memcpy(overlap_coefficients.data(),
                        device_overlap_coefficients.data(),
                        overlap_coefficients.size() * sizeof(double));

            // Subtract the double-counted G=0 component, where this rank holds it.
            if (gamma_origin_is_local_) {
                for (int candidate_index = 0; candidate_index < number_of_candidates;
                     ++candidate_index) {
                    for (int basis_index = 0; basis_index < number_of_existing; ++basis_index) {
                        Complex basis_g0 = basis_vectors[basis_index * number_of_plane_waves];
                        Complex sv_g0 =
                            overlap_basis_vectors[(starting_vector_index + candidate_index) *
                                                  number_of_plane_waves];
                        overlap_coefficients[candidate_index * number_of_existing + basis_index] -=
                            (std::conj(basis_g0) * sv_g0).real();
                    }
                }
            }

            execution_policy_.reduce_within_group(overlap_coefficients.data(),
                                                  number_of_existing * number_of_candidates);

            std::memcpy(device_overlap_coefficients.data(),
                        overlap_coefficients.data(),
                        overlap_coefficients.size() * sizeof(double));

            auto* real_candidates = reinterpret_cast<double*>(
                &basis_vectors[starting_vector_index * number_of_plane_waves]);
            auto* real_overlap_basis = reinterpret_cast<double*>(overlap_basis_vectors);

            double alpha_neg_one = -1.0;
            double beta_one = 1.0;

            // candidates -= V · coeffs
            gpu_blas_.dgemm('N',
                            'N',
                            double_number_of_plane_waves,
                            number_of_candidates,
                            number_of_existing,
                            alpha_neg_one,
                            real_basis,
                            double_leading_dimension,
                            device_overlap_coefficients.data(),
                            number_of_existing,
                            beta_one,
                            real_candidates,
                            double_leading_dimension);

            // S·candidates -= S·V · coeffs
            gpu_blas_.dgemm('N',
                            'N',
                            double_number_of_plane_waves,
                            number_of_candidates,
                            number_of_existing,
                            alpha_neg_one,
                            real_overlap_basis,
                            double_leading_dimension,
                            device_overlap_coefficients.data(),
                            number_of_existing,
                            beta_one,
                            real_candidates_sv,
                            double_leading_dimension);

            device_backend_->synchronize();
        }

        // Cholesky QR for inter-candidate orthogonalization.
        if (!is_gamma_only) {
            auto* cast_candidates = reinterpret_cast<std::complex<double>*>(
                &basis_vectors[starting_vector_index * number_of_plane_waves]);
            auto* cast_candidates_sv = reinterpret_cast<std::complex<double>*>(
                &overlap_basis_vectors[starting_vector_index * number_of_plane_waves]);

            std::vector<Complex> cholesky_factor(number_of_candidates * number_of_candidates,
                                                 Complex(0.0, 0.0));
            execution::gpu::device::GPUDeviceBuffer<Complex> device_cholesky_factor(
                number_of_candidates * number_of_candidates, device_backend_);
            Complex alpha_one(1.0, 0.0), beta_zero(0.0, 0.0);

            gpu_blas_.gemm('C',
                           'N',
                           number_of_candidates,
                           number_of_candidates,
                           number_of_plane_waves,
                           alpha_one,
                           &basis_vectors[starting_vector_index * number_of_plane_waves],
                           leading_dimension,
                           &overlap_basis_vectors[starting_vector_index * number_of_plane_waves],
                           leading_dimension,
                           beta_zero,
                           device_cholesky_factor.data(),
                           number_of_candidates);

            device_backend_->synchronize();

            std::memcpy(cholesky_factor.data(),
                        device_cholesky_factor.data(),
                        cholesky_factor.size() * sizeof(Complex));

            execution_policy_.reduce_within_group(cholesky_factor.data(),
                                                  number_of_candidates * number_of_candidates);

            int info = 0;
            auto* cast_cholesky_factor =
                reinterpret_cast<std::complex<double>*>(cholesky_factor.data());
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
                return number_of_candidates;
            }
        } else {
            // Gamma-only Cholesky: same as serial.
            int double_number_of_plane_waves = 2 * number_of_plane_waves;
            int double_leading_dimension = 2 * leading_dimension;
            auto* real_candidates = reinterpret_cast<double*>(
                &basis_vectors[starting_vector_index * number_of_plane_waves]);
            auto* real_candidates_sv = reinterpret_cast<double*>(
                &overlap_basis_vectors[starting_vector_index * number_of_plane_waves]);

            std::vector<double> cholesky_factor(number_of_candidates * number_of_candidates, 0.0);
            execution::gpu::device::GPUDeviceBuffer<double> device_cholesky_factor(
                number_of_candidates * number_of_candidates, device_backend_);
            double alpha_two = 2.0;
            double beta_zero = 0.0;
            gpu_blas_.dgemm('T',
                            'N',
                            number_of_candidates,
                            number_of_candidates,
                            double_number_of_plane_waves,
                            alpha_two,
                            real_candidates,
                            double_leading_dimension,
                            real_candidates_sv,
                            double_leading_dimension,
                            beta_zero,
                            device_cholesky_factor.data(),
                            number_of_candidates);

            device_backend_->synchronize();

            std::memcpy(cholesky_factor.data(),
                        device_cholesky_factor.data(),
                        cholesky_factor.size() * sizeof(double));

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
                return number_of_candidates;
            }
        }

        // Fallback: Modified Gram-Schmidt (same as serial path).
        int accepted_count = 0;
        int one_increment = 1;
        int double_leading_dimension = 2 * leading_dimension;
        std::vector<double> batch_coefficients_buffer(number_of_candidates, 0.0);
        std::vector<Complex> z_batch_coefficients_buffer(number_of_candidates, Complex(0.0, 0.0));

        for (int candidate_index = 0; candidate_index < number_of_candidates; ++candidate_index) {
            int source_index = starting_vector_index + candidate_index;
            int destination_index = starting_vector_index + accepted_count;

            if (destination_index != source_index) {
                std::memcpy(&basis_vectors[destination_index * number_of_plane_waves],
                            &basis_vectors[source_index * number_of_plane_waves],
                            number_of_plane_waves * sizeof(Complex));
                std::memcpy(&overlap_basis_vectors[destination_index * number_of_plane_waves],
                            &overlap_basis_vectors[source_index * number_of_plane_waves],
                            number_of_plane_waves * sizeof(Complex));
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

            Real norm_squared = gamma_aware_real_dot(
                is_gamma_only, number_of_plane_waves, current_vector, current_overlap);

            if (norm_squared > math::tolerances::iterative_solver_norm_threshold) {
                double inverse_norm = 1.0 / std::sqrt(norm_squared);
                q3::blas::zdscal_(
                    &number_of_plane_waves, &inverse_norm, current_vector, &one_increment);
                q3::blas::zdscal_(
                    &number_of_plane_waves, &inverse_norm, current_overlap, &one_increment);
                accepted_count++;
            }
        }
        return accepted_count;
    }

    /**
     * Incrementally update subspace matrices via GPU GEMM.
     */
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

        if (is_gamma_only) {
            // Gamma-only: use GPU DGEMM.
            int double_number_of_plane_waves = 2 * number_of_plane_waves;
            int double_leading_dimension = 2 * leading_dimension;
            double alpha_two = 2.0;
            double d_beta_zero = 0.0;

            auto* real_basis = reinterpret_cast<const double*>(basis_vectors);
            auto* real_h_new = reinterpret_cast<const double*>(
                &hamiltonian_basis_vectors[start_column * number_of_plane_waves]);
            auto* real_s_new = reinterpret_cast<const double*>(
                &overlap_basis_vectors[start_column * number_of_plane_waves]);

            std::vector<double> h_block_real(static_cast<size_t>(total_basis_size) * new_columns,
                                             0.0);
            std::vector<double> s_block_real(static_cast<size_t>(total_basis_size) * new_columns,
                                             0.0);

            execution::gpu::device::GPUDeviceBuffer<double> device_h_block_real(
                total_basis_size * new_columns, device_backend_);
            execution::gpu::device::GPUDeviceBuffer<double> device_s_block_real(
                total_basis_size * new_columns, device_backend_);

            gpu_blas_.dgemm('T',
                            'N',
                            total_basis_size,
                            new_columns,
                            double_number_of_plane_waves,
                            alpha_two,
                            real_basis,
                            double_leading_dimension,
                            real_h_new,
                            double_leading_dimension,
                            d_beta_zero,
                            device_h_block_real.data(),
                            total_basis_size);

            gpu_blas_.dgemm('T',
                            'N',
                            total_basis_size,
                            new_columns,
                            double_number_of_plane_waves,
                            alpha_two,
                            real_basis,
                            double_leading_dimension,
                            real_s_new,
                            double_leading_dimension,
                            d_beta_zero,
                            device_s_block_real.data(),
                            total_basis_size);

            device_backend_->synchronize();

            std::memcpy(h_block_real.data(),
                        device_h_block_real.data(),
                        h_block_real.size() * sizeof(double));
            std::memcpy(s_block_real.data(),
                        device_s_block_real.data(),
                        s_block_real.size() * sizeof(double));

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
            // Non-gamma: use GPU GEMM.
            Complex alpha_one(1.0, 0.0), beta_zero(0.0, 0.0);

            gpu_blas_.gemm('C',
                           'N',
                           total_basis_size,
                           new_columns,
                           number_of_plane_waves,
                           alpha_one,
                           basis_vectors,
                           leading_dimension,
                           &hamiltonian_basis_vectors[start_column * number_of_plane_waves],
                           leading_dimension,
                           beta_zero,
                           &subspace_hamiltonian[start_column * maximum_basis_size],
                           maximum_basis_size);

            gpu_blas_.gemm('C',
                           'N',
                           total_basis_size,
                           new_columns,
                           number_of_plane_waves,
                           alpha_one,
                           basis_vectors,
                           leading_dimension,
                           &overlap_basis_vectors[start_column * number_of_plane_waves],
                           leading_dimension,
                           beta_zero,
                           &subspace_overlap[start_column * maximum_basis_size],
                           maximum_basis_size);

            device_backend_->synchronize();
        }

        int new_block_size = maximum_basis_size * new_columns;
        int new_block_offset = start_column * maximum_basis_size;
        execution_policy_.reduce_within_group(&subspace_hamiltonian[new_block_offset],
                                              new_block_size);
        execution_policy_.reduce_within_group(&subspace_overlap[new_block_offset], new_block_size);
    }

    /**
     * Diagonalize subspace via CPU LAPACK (small matrix — GPU not beneficial).
     */
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
            log::warn(
                "LAPACK eigensolver returned info={} in GPU Davidson subspace (basis_size={})",
                info,
                basis_size);
        }
        return info;
    }

    /**
     * Collapse subspace via GPU GEMM rotation.
     */
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
         * Same formulation as the CPU path: R = (HV)c - eps (SV)c, two GEMMs
         * writing straight into the free basis columns, so no Ritz vectors and
         * no residual array are held. The device path is single-process, so
         * there is no band partition to apply.
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

        execution::gpu::device::GPUDeviceBuffer<Complex> device_coefficients(
            static_cast<std::size_t>(basis_size) * static_cast<std::size_t>(number_of_candidates),
            device_backend_);
        std::vector<Real> candidate_eigenvalues(static_cast<std::size_t>(number_of_candidates),
                                                0.0);
        for (int candidate_index = 0; candidate_index < number_of_candidates; ++candidate_index) {
            int band_index = source_bands[static_cast<std::size_t>(candidate_index)];
            candidate_eigenvalues[static_cast<std::size_t>(candidate_index)] =
                eigenvalues[band_index];
            std::memcpy(
                device_coefficients.data() + static_cast<std::size_t>(candidate_index) * basis_size,
                eigenvector_matrix + static_cast<std::size_t>(band_index) *
                                         static_cast<std::size_t>(eigenvector_leading_dimension),
                static_cast<std::size_t>(basis_size) * sizeof(Complex));
        }

        int leading_dimension = std::max(number_of_plane_waves, 1);
        Complex alpha_one(1.0, 0.0), beta_zero(0.0, 0.0), beta_one(1.0, 0.0);

        gpu_blas_.gemm('N',
                       'N',
                       number_of_plane_waves,
                       number_of_candidates,
                       basis_size,
                       alpha_one,
                       overlap_basis_vectors,
                       leading_dimension,
                       device_coefficients.data(),
                       basis_size,
                       beta_zero,
                       correction_columns,
                       leading_dimension);
        device_backend_->synchronize();

        for (int candidate_index = 0; candidate_index < number_of_candidates; ++candidate_index) {
            Complex negative_eigenvalue(
                -candidate_eigenvalues[static_cast<std::size_t>(candidate_index)], 0.0);
            gpu_blas_.scale(number_of_plane_waves,
                            negative_eigenvalue,
                            correction_columns +
                                static_cast<std::size_t>(candidate_index) * number_of_plane_waves);
        }
        device_backend_->synchronize();

        gpu_blas_.gemm('N',
                       'N',
                       number_of_plane_waves,
                       number_of_candidates,
                       basis_size,
                       alpha_one,
                       hamiltonian_basis_vectors,
                       leading_dimension,
                       device_coefficients.data(),
                       basis_size,
                       beta_one,
                       correction_columns,
                       leading_dimension);
        device_backend_->synchronize();

        /**
         * The preconditioner kernel dereferences the eigenvalues on the
         * device, so stage them there rather than hand over a host vector.
         */
        execution::gpu::device::GPUDeviceBuffer<Real> device_candidate_eigenvalues(
            static_cast<std::size_t>(number_of_candidates), device_backend_);
        std::memcpy(device_candidate_eigenvalues.data(),
                    candidate_eigenvalues.data(),
                    static_cast<std::size_t>(number_of_candidates) * sizeof(Real));
        std::vector<bool> none_converged(static_cast<std::size_t>(number_of_candidates), false);
        apply_preconditioner(number_of_candidates,
                             number_of_plane_waves,
                             kinetic_energy_squared,
                             device_candidate_eigenvalues.data(),
                             none_converged,
                             correction_columns);

        enforce_gamma_g0_real(
            is_gamma_only, number_of_plane_waves, number_of_candidates, correction_columns);
        return number_of_candidates;
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
        std::vector<Complex> eigenvector_matrix(maximum_basis_size * maximum_basis_size);
        /**
         * As on the CPU path: the caller's subspace is complete through
         * basis_size, so diagonalise it; rebuild from the vectors only as the
         * fallback for the recovery collapse.
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
                                                    eigenvalues_expanded.data(),
                                                    eigenvector_matrix.data());
        }

        int leading_dimension = std::max(number_of_plane_waves, 1);
        Complex alpha_one(1.0, 0.0), beta_zero(0.0, 0.0);
        /**
         * As on the CPU path: only number_of_bands columns are rotated and
         * copied back, and the destination is the caller's scratch rather than
         * three more device buffers the size of the whole basis.
         */
        std::size_t collapsed_size = static_cast<std::size_t>(number_of_bands) *
                                     static_cast<std::size_t>(number_of_plane_waves);

        // Rotate via GPU GEMM.
        execution::gpu::device::GPUDeviceBuffer<Complex> device_eigenvector_matrix(
            basis_size * basis_size, device_backend_);
        std::memcpy(device_eigenvector_matrix.data(),
                    eigenvector_matrix.data(),
                    basis_size * basis_size * sizeof(Complex));

        /**
         * One scratch buffer serves all three rotations: each reads a
         * different array than it writes, and the overlap needs no rotation at
         * all when it aliases the basis.
         */
        auto rotate_into_scratch = [&](const Complex* source) {
            gpu_blas_.gemm('N',
                           'N',
                           number_of_plane_waves,
                           number_of_bands,
                           basis_size,
                           alpha_one,
                           source,
                           leading_dimension,
                           device_eigenvector_matrix.data(),
                           basis_size,
                           beta_zero,
                           rotation_scratch,
                           leading_dimension);
            device_backend_->synchronize();
        };

        rotate_into_scratch(basis_vectors);
        std::memcpy(basis_vectors, rotation_scratch, collapsed_size * sizeof(Complex));

        rotate_into_scratch(hamiltonian_basis_vectors);
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
            rotate_into_scratch(overlap_basis_vectors);
            std::memcpy(overlap_basis_vectors, rotation_scratch, collapsed_size * sizeof(Complex));
        }

        enforce_gamma_g0_real(is_gamma_only, number_of_plane_waves, number_of_bands, basis_vectors);
        enforce_gamma_g0_real(
            is_gamma_only, number_of_plane_waves, number_of_bands, hamiltonian_basis_vectors);
        enforce_gamma_g0_real(
            is_gamma_only, number_of_plane_waves, number_of_bands, overlap_basis_vectors);

        return number_of_bands;
    }

    /**
     * Extract eigenvectors using GPU GEMM rotation.
     */
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

        /**
         * The rotation lands in the caller's scratch, which on this path is a
         * device buffer already sized for it, so the extra device allocation
         * this used to make is gone. The host copy stays: band data is written
         * to the quantum system one band at a time from host memory.
         */
        std::vector<Complex> final_wavefunctions(static_cast<std::size_t>(number_of_bands) *
                                                 static_cast<std::size_t>(number_of_plane_waves));

        execution::gpu::device::GPUDeviceBuffer<Complex> device_eigenvector_matrix(
            basis_size * basis_size, device_backend_);
        std::memcpy(device_eigenvector_matrix.data(),
                    eigenvector_matrix,
                    basis_size * basis_size * sizeof(Complex));

        Complex alpha_one(1.0, 0.0), beta_zero(0.0, 0.0);

        gpu_blas_.gemm('N',
                       'N',
                       number_of_plane_waves,
                       number_of_bands,
                       basis_size,
                       alpha_one,
                       basis_vectors,
                       leading_dimension,
                       device_eigenvector_matrix.data(),
                       basis_size,
                       beta_zero,
                       rotated_scratch,
                       leading_dimension);

        device_backend_->synchronize();

        std::memcpy(final_wavefunctions.data(),
                    rotated_scratch,
                    final_wavefunctions.size() * sizeof(Complex));

        enforce_gamma_g0_real(
            is_gamma_only, number_of_plane_waves, number_of_bands, final_wavefunctions.data());

        for (int band_index = 0; band_index < number_of_bands; ++band_index) {
            quantum_system.set_eigenvalue(kpoint_index_state, band_index, eigenvalues[band_index]);
            Complex* band_data =
                quantum_system.wavefunction_band_data(kpoint_index_state, band_index);
            std::copy(&final_wavefunctions[band_index * number_of_plane_waves],
                      &final_wavefunctions[(band_index + 1) * number_of_plane_waves],
                      band_data);
        }
    }

    void apply_preconditioner(int number_of_bands,
                              int number_of_plane_waves,
                              const Real* kinetic_energy_squared,
                              const Real* eigenvalues,
                              const std::vector<bool>& band_converged,
                              Complex* residual_wavefunctions) override {
        // Copy band_converged to a temporary device buffer.
        execution::gpu::device::GPUDeviceBuffer<bool> device_band_converged(number_of_bands,
                                                                            device_backend_);
        /**
         * std::vector<bool> is bit-packed and has no contiguous bool storage,
         * so stage through a byte-per-flag vector for the device upload.
         */
        std::vector<char> host_converged(number_of_bands);
        for (int band = 0; band < number_of_bands; ++band) {
            host_converged[band] = band_converged[band] ? 1 : 0;
        }
        std::memcpy(
            device_band_converged.data(), host_converged.data(), number_of_bands * sizeof(bool));

        // Copy kinetic_energy_squared to a temporary device buffer.
        execution::gpu::device::GPUDeviceBuffer<Real> device_kinetic_energy_squared(
            number_of_plane_waves, device_backend_);
        std::memcpy(device_kinetic_energy_squared.data(),
                    kinetic_energy_squared,
                    number_of_plane_waves * sizeof(Real));

        diagonalizer::planewave::gpu::launch_apply_preconditioner(
            number_of_bands,
            number_of_plane_waves,
            device_kinetic_energy_squared.data(),
            eigenvalues,
            device_band_converged.data(),
            math::tolerances::preconditioner_denominator_floor,
            reinterpret_cast<std::complex<double>*>(residual_wavefunctions));
        device_backend_->synchronize();
    }

  private:
    // ─── Helpers ──────────────────────────────────────────────────────

    /**
     * Build full subspace matrices and diagonalize for collapse.
     */
    void build_subspace_matrices_and_diagonalize(int basis_size,
                                                 int number_of_plane_waves,
                                                 bool is_gamma_only,
                                                 const OverlapOperator* overlap_operator,
                                                 const Complex* basis_vectors,
                                                 const Complex* hamiltonian_basis_vectors,
                                                 const Complex* overlap_basis_vectors,
                                                 Real* eigenvalues,
                                                 Complex* eigenvector_matrix) {

        std::vector<Complex> subspace_hamiltonian(basis_size * basis_size, Complex(0.0, 0.0));
        std::vector<Complex> subspace_overlap(basis_size * basis_size, Complex(0.0, 0.0));

        int leading_dimension = std::max(number_of_plane_waves, 1);
        Complex alpha_one(1.0, 0.0), beta_zero(0.0, 0.0);

        if (!is_gamma_only) {
            execution::gpu::device::GPUDeviceBuffer<Complex> device_subspace_hamiltonian(
                basis_size * basis_size, device_backend_);
            execution::gpu::device::GPUDeviceBuffer<Complex> device_subspace_overlap(
                basis_size * basis_size, device_backend_);

            gpu_blas_.gemm('C',
                           'N',
                           basis_size,
                           basis_size,
                           number_of_plane_waves,
                           alpha_one,
                           basis_vectors,
                           leading_dimension,
                           hamiltonian_basis_vectors,
                           leading_dimension,
                           beta_zero,
                           device_subspace_hamiltonian.data(),
                           basis_size);

            gpu_blas_.gemm('C',
                           'N',
                           basis_size,
                           basis_size,
                           number_of_plane_waves,
                           alpha_one,
                           basis_vectors,
                           leading_dimension,
                           overlap_basis_vectors,
                           leading_dimension,
                           beta_zero,
                           device_subspace_overlap.data(),
                           basis_size);

            device_backend_->synchronize();

            std::memcpy(subspace_hamiltonian.data(),
                        device_subspace_hamiltonian.data(),
                        subspace_hamiltonian.size() * sizeof(Complex));
            std::memcpy(subspace_overlap.data(),
                        device_subspace_overlap.data(),
                        subspace_overlap.size() * sizeof(Complex));
        } else {
            // Gamma-only: use GPU DGEMM.
            int double_number_of_plane_waves = 2 * number_of_plane_waves;
            int double_leading_dimension = 2 * leading_dimension;
            double alpha_two = 2.0;
            double d_beta_zero = 0.0;

            auto* real_basis = reinterpret_cast<const double*>(basis_vectors);
            auto* real_h_basis = reinterpret_cast<const double*>(hamiltonian_basis_vectors);
            auto* real_s_basis = reinterpret_cast<const double*>(overlap_basis_vectors);

            std::vector<double> h_real(static_cast<size_t>(basis_size) * basis_size, 0.0);
            std::vector<double> s_real(static_cast<size_t>(basis_size) * basis_size, 0.0);

            execution::gpu::device::GPUDeviceBuffer<double> device_h_real(basis_size * basis_size,
                                                                          device_backend_);
            execution::gpu::device::GPUDeviceBuffer<double> device_s_real(basis_size * basis_size,
                                                                          device_backend_);

            gpu_blas_.dgemm('T',
                            'N',
                            basis_size,
                            basis_size,
                            double_number_of_plane_waves,
                            alpha_two,
                            real_basis,
                            double_leading_dimension,
                            real_h_basis,
                            double_leading_dimension,
                            d_beta_zero,
                            device_h_real.data(),
                            basis_size);

            gpu_blas_.dgemm('T',
                            'N',
                            basis_size,
                            basis_size,
                            double_number_of_plane_waves,
                            alpha_two,
                            real_basis,
                            double_leading_dimension,
                            real_s_basis,
                            double_leading_dimension,
                            d_beta_zero,
                            device_s_real.data(),
                            basis_size);

            device_backend_->synchronize();

            std::memcpy(h_real.data(), device_h_real.data(), h_real.size() * sizeof(double));
            std::memcpy(s_real.data(), device_s_real.data(), s_real.size() * sizeof(double));

            for (int col = 0; col < basis_size; ++col) {
                Complex h_right_g0 = hamiltonian_basis_vectors[col * number_of_plane_waves];
                Complex s_right_g0 = overlap_basis_vectors[col * number_of_plane_waves];
                for (int row = 0; row < basis_size; ++row) {
                    Complex left_g0 = basis_vectors[row * number_of_plane_waves];
                    int element_index = col * basis_size + row;
                    if (gamma_origin_is_local_) {
                        h_real[element_index] -= (std::conj(left_g0) * h_right_g0).real();
                        s_real[element_index] -= (std::conj(left_g0) * s_right_g0).real();
                    }
                    subspace_hamiltonian[element_index] = Complex(h_real[element_index], 0.0);
                    subspace_overlap[element_index] = Complex(s_real[element_index], 0.0);
                }
            }
        }

        execution_policy_.reduce_within_group(subspace_hamiltonian.data(), basis_size * basis_size);
        execution_policy_.reduce_within_group(subspace_overlap.data(), basis_size * basis_size);

        /**
         * See diagonalize_subspace() for the three cases: collective solvers are
         * entered by every rank, replicated vectors are solved redundantly by
         * every rank, and only a distributed non-collective solve is root-only
         * plus a broadcast.
         */
        const bool all_ranks_solve = subspace_diagonalizer_->is_collective();
        const bool vectors_are_replicated = execution_policy_.group_size() <= 1;
        const bool this_rank_solves =
            all_ranks_solve || vectors_are_replicated || execution_policy_.group_rank_index() == 0;
        int info = 0;
        if (this_rank_solves) {
            info = subspace_diagonalizer_->solve(basis_size,
                                                 subspace_hamiltonian.data(),
                                                 subspace_overlap.data(),
                                                 eigenvalues,
                                                 eigenvector_matrix,
                                                 is_gamma_only);
        }
        if (!all_ranks_solve && !vectors_are_replicated) {
            execution_policy_.broadcast_from_group_root(eigenvector_matrix,
                                                        basis_size * basis_size);
            execution_policy_.broadcast_from_group_root(eigenvalues, basis_size);
        }
        static_cast<void>(info);
    }

    /**
     * Enforce a real G=0 component for gamma-only calculations, on the rank
     * whose index 0 is the origin.
     */
    void enforce_gamma_g0_real(bool is_gamma_only,
                               int number_of_plane_waves,
                               int number_of_bands,
                               Complex* vectors) const {
        if (!is_gamma_only || !gamma_origin_is_local_)
            return;
        for (int band = 0; band < number_of_bands; ++band) {
            Complex& g0 = vectors[band * number_of_plane_waves];
            g0 = Complex(g0.real(), 0.0);
        }
    }

    /**
     * Compute gamma-aware real dot product.
     */
    Real gamma_aware_real_dot(bool is_gamma_only,
                              int number_of_plane_waves,
                              const std::complex<double>* vector_a,
                              const std::complex<double>* vector_b) const {
        if (is_gamma_only) {
            double result = 0.0;
            for (int g_index = 0; g_index < number_of_plane_waves; ++g_index) {
                double re_a = vector_a[g_index].real();
                double im_a = vector_a[g_index].imag();
                double re_b = vector_b[g_index].real();
                double im_b = vector_b[g_index].imag();
                result += 2.0 * (re_a * re_b + im_a * im_b);
            }
            // G=0 correction, where this rank holds the origin.
            if (gamma_origin_is_local_) {
                result -= (vector_a[0].real() * vector_b[0].real() +
                           vector_a[0].imag() * vector_b[0].imag());
            }
            return result;
        } else {
            double result = 0.0;
            for (int g_index = 0; g_index < number_of_plane_waves; ++g_index) {
                result += (std::conj(vector_a[g_index]) * vector_b[g_index]).real();
            }
            return result;
        }
    }

    // ─── Member data ──────────────────────────────────────────────────

    execution::gpu::blas::GPUBLAS& gpu_blas_;
    const execution::ExecutionPolicy& execution_policy_;
    bool gamma_origin_is_local_ = true;
    std::shared_ptr<execution::gpu::device::GPUDeviceBackend> device_backend_;
    std::unique_ptr<SubspaceDiagonalizer> subspace_diagonalizer_;

    // ─── Workspaces ───────────────────────────────────────────────────

    std::vector<Complex> diagonalize_overlap_copy_workspace_;
};

} // namespace q3
