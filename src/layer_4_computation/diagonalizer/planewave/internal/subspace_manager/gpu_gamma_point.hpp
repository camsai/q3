#pragma once

/**
 * GammaPointGPUPlanewaveSubspaceManager — the device subspace manager of a gamma-point
 * calculation.
 *
 * At the gamma point the bands are real in real space, so only half of the plane-wave
 * sphere is stored and every inner product is 2·Re(a†·b) over the stored half with the
 * origin -- held once, counted twice -- removed on the rank that holds it. The subspace
 * matrices are real for the same reason. This class overrides the steps of
 * GPUPlanewaveSubspaceManager that depend on that: the products run as real GEMM over
 * the coefficients viewed as pairs of doubles, the G = 0 coefficient is kept real, and
 * the subspace eigenproblem is always solved on the host.
 *
 * Chosen by create_gpu_planewave_subspace_manager().
 */

#include <algorithm>
#include <complex>
#include <cstddef>
#include <memory>
#include <vector>

#include "layer_0_core/execution/gpu/device/init.hpp"
#include "layer_0_core/execution/gpu/device/internal/buffer.hpp"
#include "layer_0_core/execution/init.hpp"
#include "layer_0_core/math/linear_algebra/blas.hpp"
#include "layer_0_core/math/linear_algebra/gpu/blas/init.hpp"
#include "layer_0_core/types/init.hpp"
#include "layer_3_representation/quantum_system/kohn_sham/planewave/init.hpp"
#include "layer_4_computation/diagonalizer/planewave/internal/overlap/init.hpp"
#include "layer_4_computation/diagonalizer/planewave/internal/subspace_manager/gpu.hpp"

namespace q3 {

/// Device subspace manager of a gamma-point calculation: real arithmetic over the half sphere.
class GammaPointGPUPlanewaveSubspaceManager : public GPUPlanewaveSubspaceManager {
  public:
    /**
     * Construct with GPU math wrappers and execution policy.
     *
     * @param gpu_blas          GPU BLAS wrapper for GEMM operations.
     * @param execution_policy  Execution policy for band-group collectives.
     * @param device_backend    GPU device backend for buffer allocation.
     */
    GammaPointGPUPlanewaveSubspaceManager(
        execution::gpu::blas::GPUBLAS& gpu_blas,
        const execution::ExecutionPolicy& execution_policy,
        std::shared_ptr<execution::gpu::device::GPUDeviceBackend> device_backend)
        : GPUPlanewaveSubspaceManager{gpu_blas, execution_policy, device_backend} {}

    /// Record whether this rank's plane-wave list begins with G = 0, whose term it then removes.
    void set_gamma_origin_is_local(bool origin_is_local) override {
        gamma_origin_is_local_ = origin_is_local;
    }

    /**
     * Diagonalize the real subspace problem on the host, whichever device eigensolver is
     * set: the device solver is a complex one.
     */
    int diagonalize_subspace(int basis_size,
                             int maximum_basis_size,
                             bool is_gamma_only,
                             const Complex* subspace_hamiltonian,
                             const Complex* subspace_overlap,
                             Real* eigenvalues,
                             Complex* eigenvector_matrix) override {
        return diagonalize_subspace_on_host(basis_size,
                                            maximum_basis_size,
                                            is_gamma_only,
                                            subspace_hamiltonian,
                                            subspace_overlap,
                                            eigenvalues,
                                            eigenvector_matrix);
    }

  protected:
    /**
     * Project the existing basis out of the candidates with real device GEMM, removing the
     * double-counted G = 0 term from the coefficients. Both blocks are updated here, so
     * `candidate_overlap_block` is not asked to follow.
     */
    void project_out_existing_basis(int starting_vector_index,
                                    int number_of_candidates,
                                    int number_of_plane_waves,
                                    GPUOverlapBlock& /*candidate_overlap_block*/,
                                    Complex* basis_vectors,
                                    Complex* overlap_basis_vectors) override {
        int leading_dimension = std::max(number_of_plane_waves, 1);
        int number_of_existing = starting_vector_index;

        // Gamma-only: project out existing basis using GPU DGEMM.
        int double_number_of_plane_waves = 2 * number_of_plane_waves;
        int double_leading_dimension = 2 * leading_dimension;
        double alpha_two = 2.0;
        double d_beta_zero = 0.0;

        auto* real_basis = reinterpret_cast<double*>(const_cast<Complex*>(basis_vectors));
        auto* real_candidates_sv = reinterpret_cast<double*>(
            &overlap_basis_vectors[starting_vector_index * number_of_plane_waves]);

        std::vector<double> overlap_coefficients(number_of_existing * number_of_candidates, 0.0);
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

        device_backend_->copy(overlap_coefficients.data(),
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

        device_backend_->copy(device_overlap_coefficients.data(),
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

    /**
     * Cholesky QR of the candidates under the gamma-point inner product: a real Gram
     * matrix, a real factorization on the host, and a real triangular solve of both blocks
     * on the device. Both blocks are updated here, so `candidate_overlap_block` is not
     * asked to follow.
     *
     * @return false, with the candidates untouched, when the Gram matrix is not positive
     *         definite.
     */
    bool orthonormalize_candidates_by_cholesky(int starting_vector_index,
                                               int number_of_candidates,
                                               int number_of_plane_waves,
                                               GPUOverlapBlock& /*candidate_overlap_block*/,
                                               Complex* basis_vectors,
                                               Complex* overlap_basis_vectors) override {
        int leading_dimension = std::max(number_of_plane_waves, 1);

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

        device_backend_->copy(cholesky_factor.data(),
                              device_cholesky_factor.data(),
                              cholesky_factor.size() * sizeof(double));

        if (gamma_origin_is_local_) {
            for (int column_index = 0; column_index < number_of_candidates; ++column_index) {
                for (int row_index = 0; row_index < number_of_candidates; ++row_index) {
                    Complex vi_g0 =
                        basis_vectors[(starting_vector_index + row_index) * number_of_plane_waves];
                    Complex svj_g0 = overlap_basis_vectors[(starting_vector_index + column_index) *
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
            // Apply R^-1 on the device, as in the complex branch above.
            device_backend_->copy(device_cholesky_factor.data(),
                                  cholesky_factor.data(),
                                  cholesky_factor.size() * sizeof(double));
            for (double* block : {real_candidates, real_candidates_sv}) {
                gpu_blas_.dtrsm('R',
                                'U',
                                'N',
                                'N',
                                double_number_of_plane_waves,
                                number_of_candidates,
                                1.0,
                                device_cholesky_factor.data(),
                                number_of_candidates,
                                block,
                                double_leading_dimension);
            }
            device_backend_->synchronize();
            return true;
        }
        return false;
    }

    /**
     * One step of the modified Gram-Schmidt fallback under the gamma-point inner product,
     * with real host BLAS over the coefficients viewed as pairs of doubles.
     */
    void project_candidate_against_accepted(int starting_vector_index,
                                            int destination_index,
                                            int number_of_batch_existing,
                                            int number_of_plane_waves,
                                            Complex* basis_vectors,
                                            Complex* overlap_basis_vectors) override {
        int leading_dimension = std::max(number_of_plane_waves, 1);
        int one_increment = 1;
        int double_leading_dimension = 2 * leading_dimension;
        std::vector<double> batch_coefficients_buffer(number_of_batch_existing, 0.0);
        auto* current_vector = reinterpret_cast<std::complex<double>*>(
            &basis_vectors[destination_index * number_of_plane_waves]);
        auto* current_overlap = reinterpret_cast<std::complex<double>*>(
            &overlap_basis_vectors[destination_index * number_of_plane_waves]);

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
            for (int basis_index = 0; basis_index < number_of_batch_existing; ++basis_index) {
                Complex basis_g0 =
                    basis_vectors[(starting_vector_index + basis_index) * number_of_plane_waves];
                Complex sv_g0 = overlap_basis_vectors[destination_index * number_of_plane_waves];
                batch_coefficients_buffer[basis_index] -= (std::conj(basis_g0) * sv_g0).real();
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
    }

    /**
     * The gamma-point real inner product: 2·Re(a†·b) over the stored half sphere, less the
     * G = 0 term where this rank holds the origin.
     */
    Real real_inner_product(int number_of_plane_waves,
                            const std::complex<double>* vector_a,
                            const std::complex<double>* vector_b) const override {
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
            result -=
                (vector_a[0].real() * vector_b[0].real() + vector_a[0].imag() * vector_b[0].imag());
        }
        return result;
    }

    /**
     * Fill the new columns of the subspace matrices with real device GEMM: the products
     * come down to the host, lose their double-counted G = 0 term, and are written into
     * the caller's matrices as real numbers.
     */
    void compute_new_subspace_matrix_columns(int start_column,
                                             int new_columns,
                                             int total_basis_size,
                                             int maximum_basis_size,
                                             int number_of_plane_waves,
                                             const Complex* basis_vectors,
                                             const Complex* hamiltonian_basis_vectors,
                                             const Complex* overlap_basis_vectors,
                                             Complex* subspace_hamiltonian,
                                             Complex* subspace_overlap) override {
        int leading_dimension = std::max(number_of_plane_waves, 1);

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

        std::vector<double> h_block_real(static_cast<size_t>(total_basis_size) * new_columns, 0.0);
        std::vector<double> s_block_real(static_cast<size_t>(total_basis_size) * new_columns, 0.0);

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

        device_backend_->copy(
            h_block_real.data(), device_h_block_real.data(), h_block_real.size() * sizeof(double));
        device_backend_->copy(
            s_block_real.data(), device_s_block_real.data(), s_block_real.size() * sizeof(double));

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
    }

    /**
     * Build the whole subspace matrices with real device GEMM, removing the double-counted
     * G = 0 term on the host.
     */
    void build_full_subspace_matrices(int basis_size,
                                      int number_of_plane_waves,
                                      const Complex* basis_vectors,
                                      const Complex* hamiltonian_basis_vectors,
                                      const Complex* overlap_basis_vectors,
                                      std::vector<Complex>& subspace_hamiltonian,
                                      std::vector<Complex>& subspace_overlap) override {
        int leading_dimension = std::max(number_of_plane_waves, 1);

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

        device_backend_->copy(h_real.data(), device_h_real.data(), h_real.size() * sizeof(double));
        device_backend_->copy(s_real.data(), device_s_real.data(), s_real.size() * sizeof(double));

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

    /**
     * Rotate the basis into `rotated_scratch` on the device, bring the bands to the host,
     * make their G = 0 coefficients real and store them and the eigenvalues in the
     * k-point state.
     */
    void store_rotated_bands(int basis_size,
                             int number_of_plane_waves,
                             int number_of_bands,
                             int kpoint_index_state,
                             const Complex* device_eigenvector_matrix,
                             const Complex* basis_vectors,
                             const Real* eigenvalues,
                             Complex* rotated_scratch,
                             PlanewaveKohnShamQuantumSystem& quantum_system) override {
        int leading_dimension = std::max(number_of_plane_waves, 1);
        Complex alpha_one(1.0, 0.0), beta_zero(0.0, 0.0);

        // Gamma point: the rotation lands in the caller's scratch, a device buffer sized for it.
        gpu_blas_.gemm('N',
                       'N',
                       number_of_plane_waves,
                       number_of_bands,
                       basis_size,
                       alpha_one,
                       basis_vectors,
                       leading_dimension,
                       device_eigenvector_matrix,
                       basis_size,
                       beta_zero,
                       rotated_scratch,
                       leading_dimension);

        device_backend_->synchronize();

        // The G = 0 coefficient is made real on the host before the bands are stored.
        std::vector<Complex> final_wavefunctions(static_cast<std::size_t>(number_of_bands) *
                                                 static_cast<std::size_t>(number_of_plane_waves));
        device_backend_->copy(final_wavefunctions.data(),
                              rotated_scratch,
                              final_wavefunctions.size() * sizeof(Complex));

        make_origin_coefficients_real(
            number_of_plane_waves, number_of_bands, final_wavefunctions.data());

        for (int band_index = 0; band_index < number_of_bands; ++band_index) {
            quantum_system.set_eigenvalue(kpoint_index_state, band_index, eigenvalues[band_index]);
            Complex* band_data =
                quantum_system.mutable_wavefunction_band_data(kpoint_index_state, band_index);
            std::copy(&final_wavefunctions[band_index * number_of_plane_waves],
                      &final_wavefunctions[(band_index + 1) * number_of_plane_waves],
                      band_data);
        }
    }

    /**
     * Enforce a real G=0 component for gamma-only calculations, on the rank
     * whose index 0 is the origin.
     */
    void make_origin_coefficients_real(int number_of_plane_waves,
                                       int number_of_bands,
                                       Complex* vectors) const override {
        if (!gamma_origin_is_local_)
            return;
        for (int band = 0; band < number_of_bands; ++band) {
            Complex& g0 = vectors[band * number_of_plane_waves];
            g0 = Complex(g0.real(), 0.0);
        }
    }

  private:
    /// Whether this rank's plane-wave list begins with G = 0; see set_gamma_origin_is_local().
    bool gamma_origin_is_local_ = true;
};

} // namespace q3
