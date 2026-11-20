#pragma once

/**
 * GPUPlanewaveSubspaceManager — GPU implementation of subspace operations.
 *
 * Implements PlanewaveSubspaceManager using GPU BLAS for the large matrix
 * operations (subspace Hamiltonian build, Ritz vector rotation, orthogonalization).
 * The subspace eigenproblem is solved on the host (the matrices are small N×N)
 * unless a device eigensolver is set.
 *
 * This class is the manager for the general complex representation: its inner
 * products are a†·b over the whole plane-wave list. The steps that depend on the
 * representation are protected virtuals, which GammaPointGPUPlanewaveSubspaceManager
 * overrides for the half-sphere storage of a gamma-point calculation. Which of the
 * two a run uses is decided once, in create_gpu_planewave_subspace_manager(); the
 * is_gamma_only arguments of the interface are not consulted here.
 *
 * No CUDA headers. No Metal headers. Uses only vendor-neutral GPU abstractions.
 */

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstring>
#include <memory>
#include <optional>
#include <stdexcept>
#include <vector>

#include "io/logger.hpp"
#include "layer_0_core/execution/gpu/device/init.hpp"
#include "layer_0_core/execution/gpu/device/internal/buffer.hpp"
#include "layer_0_core/execution/init.hpp"
#include "layer_0_core/math/linear_algebra/blas.hpp"
#include "layer_0_core/math/linear_algebra/gpu/blas/init.hpp"
#include "layer_0_core/math/linear_algebra/gpu/lapack/init.hpp"
#include "layer_0_core/math/tolerances.hpp"
#include "layer_0_core/profiling/stage_timer.hpp"
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
        , subspace_diagonalizer_{
              SubspaceDiagonalizerFactory::create_for_device_solver(execution_policy)} {}

    /**
     * Have the subspace eigenproblem solved by this device eigensolver, on the subspace
     * matrices where they are, in place of copying them to the host for LAPACK. Null, the
     * default, keeps the host solve. Used only where this rank holds the subspace whole:
     * ranks sharing a band group keep their host solvers, and so does the gamma-point
     * manager, whichever eigensolver is set. The eigensolver must outlive this manager.
     */
    void set_device_eigensolver(execution::gpu::lapack::GPUEigensolver* device_eigensolver) {
        device_eigensolver_ = device_eigensolver;
    }

    /**
     * Nothing to record: the complex representation counts every plane wave once, so its
     * inner products have no origin term to remove.
     */
    void set_gamma_origin_is_local(bool /*origin_is_local*/) override {}

    /**
     * S-orthogonalize and normalize candidates using GPU BLAS.
     *
     * Uses GPU GEMM for computing overlap coefficients and applying
     * projections. Falls back to modified Gram-Schmidt if Cholesky fails.
     */
    int orthogonalize_batch(int starting_vector_index,
                            int number_of_candidates,
                            int number_of_plane_waves,
                            bool /*is_gamma_only*/,
                            const OverlapOperator* overlap_operator,
                            Complex* basis_vectors,
                            Complex* overlap_basis_vectors) override {
        if (number_of_candidates == 0) {
            return 0;
        }

        /**
         * The candidates and S applied to them. What S is decides how the second block is
         * formed and kept in step with the first, and the operator's block knows: for S = 1
         * it is a device copy, where the operator's own apply is a host loop that migrated
         * both device-resident arrays to the host and back on every call. The gamma-point
         * path and the Gram-Schmidt fallback below still update both blocks themselves.
         */
        Complex* candidates = &basis_vectors[starting_vector_index * number_of_plane_waves];
        Complex* overlap_candidates =
            &overlap_basis_vectors[starting_vector_index * number_of_plane_waves];
        std::unique_ptr<GPUOverlapBlock> candidate_overlap_block =
            overlap_operator->create_gpu_overlap_block(*device_backend_,
                                                       number_of_plane_waves,
                                                       number_of_candidates,
                                                       candidates,
                                                       overlap_candidates);
        candidate_overlap_block->fill_from_vectors();

        int number_of_existing = starting_vector_index;

        // Project out existing basis via GPU GEMM.
        if (number_of_existing > 0) {
            project_out_existing_basis(starting_vector_index,
                                       number_of_candidates,
                                       number_of_plane_waves,
                                       *candidate_overlap_block,
                                       basis_vectors,
                                       overlap_basis_vectors);
        }

        // Cholesky QR for inter-candidate orthogonalization.
        if (orthonormalize_candidates_by_cholesky(starting_vector_index,
                                                  number_of_candidates,
                                                  number_of_plane_waves,
                                                  *candidate_overlap_block,
                                                  basis_vectors,
                                                  overlap_basis_vectors)) {
            return number_of_candidates;
        }

        // Fallback: Modified Gram-Schmidt (same as serial path).
        int accepted_count = 0;
        int one_increment = 1;

        for (int candidate_index = 0; candidate_index < number_of_candidates; ++candidate_index) {
            int source_index = starting_vector_index + candidate_index;
            int destination_index = starting_vector_index + accepted_count;

            if (destination_index != source_index) {
                device_backend_->copy(&basis_vectors[destination_index * number_of_plane_waves],
                                      &basis_vectors[source_index * number_of_plane_waves],
                                      number_of_plane_waves * sizeof(Complex));
                device_backend_->copy(
                    &overlap_basis_vectors[destination_index * number_of_plane_waves],
                    &overlap_basis_vectors[source_index * number_of_plane_waves],
                    number_of_plane_waves * sizeof(Complex));
            }

            auto* current_vector = reinterpret_cast<std::complex<double>*>(
                &basis_vectors[destination_index * number_of_plane_waves]);
            auto* current_overlap = reinterpret_cast<std::complex<double>*>(
                &overlap_basis_vectors[destination_index * number_of_plane_waves]);

            int number_of_batch_existing = destination_index - starting_vector_index;
            if (number_of_batch_existing > 0) {
                project_candidate_against_accepted(starting_vector_index,
                                                   destination_index,
                                                   number_of_batch_existing,
                                                   number_of_plane_waves,
                                                   basis_vectors,
                                                   overlap_basis_vectors);
            }

            Real norm_squared =
                real_inner_product(number_of_plane_waves, current_vector, current_overlap);

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
                                  bool /*is_gamma_only*/,
                                  const Complex* basis_vectors,
                                  const Complex* hamiltonian_basis_vectors,
                                  const Complex* overlap_basis_vectors,
                                  Complex* subspace_hamiltonian,
                                  Complex* subspace_overlap) override {
        compute_new_subspace_matrix_columns(start_column,
                                            new_columns,
                                            total_basis_size,
                                            maximum_basis_size,
                                            number_of_plane_waves,
                                            basis_vectors,
                                            hamiltonian_basis_vectors,
                                            overlap_basis_vectors,
                                            subspace_hamiltonian,
                                            subspace_overlap);

        int new_block_size = maximum_basis_size * new_columns;
        int new_block_offset = start_column * maximum_basis_size;
        execution_policy_.reduce_within_group(&subspace_hamiltonian[new_block_offset],
                                              new_block_size);
        execution_policy_.reduce_within_group(&subspace_overlap[new_block_offset], new_block_size);
    }

    /**
     * Diagonalize the subspace: on the device where a device eigensolver is set and this
     * rank holds the subspace whole, on the host (small matrix) otherwise.
     */
    int diagonalize_subspace(int basis_size,
                             int maximum_basis_size,
                             bool is_gamma_only,
                             const Complex* subspace_hamiltonian,
                             const Complex* subspace_overlap,
                             Real* eigenvalues,
                             Complex* eigenvector_matrix) override {
        if (device_eigensolver_ != nullptr && basis_size > 0 &&
            execution_policy_.group_size() <= 1) {
            return diagonalize_subspace_on_device(basis_size,
                                                  maximum_basis_size,
                                                  subspace_hamiltonian,
                                                  subspace_overlap,
                                                  eigenvalues,
                                                  eigenvector_matrix);
        }
        return diagonalize_subspace_on_host(basis_size,
                                            maximum_basis_size,
                                            is_gamma_only,
                                            subspace_hamiltonian,
                                            subspace_overlap,
                                            eigenvalues,
                                            eigenvector_matrix);
    }

  private:
    /**
     * The subspace eigenproblem on the device. The columns in use of the two subspace
     * matrices are copied, device to device, into workspaces the solver may overwrite (the
     * caller's matrices carry the next iteration's incremental update), and solved there with
     * the caller's leading dimension, so nothing is compacted on the device. What comes down
     * is the eigenvectors, in one copy, and the eigenvalues: half of what the host solve
     * brought down, and no host LAPACK.
     *
     * @return 0, or 1 if the device solver reported a failure, which the Davidson loop
     *         answers as it does a failed host solve: collapse the subspace and retry.
     */
    int diagonalize_subspace_on_device(int basis_size,
                                       int maximum_basis_size,
                                       const Complex* subspace_hamiltonian,
                                       const Complex* subspace_overlap,
                                       Real* eigenvalues,
                                       Complex* eigenvector_matrix) {
        const std::size_t staged_elements =
            static_cast<std::size_t>(basis_size) * maximum_basis_size;
        const std::size_t staged_bytes = staged_elements * sizeof(Complex);
        ensure_capacity(device_solve_hamiltonian_, staged_elements);
        ensure_capacity(device_solve_overlap_, staged_elements);
        ensure_capacity(device_solve_eigenvalues_, static_cast<std::size_t>(basis_size));

        std::optional<StageTimer> stage_timer(std::in_place, "davidson.subspace.device_solve");
        device_backend_->copy(device_solve_hamiltonian_.data(), subspace_hamiltonian, staged_bytes);
        device_backend_->copy(device_solve_overlap_.data(), subspace_overlap, staged_bytes);
        try {
            device_eigensolver_->generalized_hermitian_eigenvalue_solve(
                basis_size,
                device_solve_hamiltonian_.data(),
                maximum_basis_size,
                device_solve_overlap_.data(),
                maximum_basis_size,
                device_solve_eigenvalues_.data());
        } catch (const std::runtime_error& failure) {
            log::warn("Device eigensolver failed in GPU Davidson subspace (basis_size={}): {}",
                      basis_size,
                      failure.what());
            return 1;
        }
        device_backend_->synchronize();

        stage_timer.emplace("davidson.subspace.download");
        host_subspace_hamiltonian_.resize(staged_elements);
        device_backend_->copy(
            host_subspace_hamiltonian_.data(), device_solve_hamiltonian_.data(), staged_bytes);
        device_backend_->copy(eigenvalues,
                              device_solve_eigenvalues_.data(),
                              static_cast<std::size_t>(basis_size) * sizeof(Real));
        for (int column = 0; column < basis_size; ++column) {
            std::copy_n(host_subspace_hamiltonian_.data() +
                            static_cast<std::size_t>(column) * maximum_basis_size,
                        basis_size,
                        eigenvector_matrix + static_cast<std::size_t>(column) * basis_size);
        }
        return 0;
    }

    /// Grow `buffer` to hold `elements`; never shrink it.
    template <typename Element>
    void ensure_capacity(execution::gpu::device::GPUDeviceBuffer<Element>& buffer,
                         std::size_t elements) {
        if (buffer.size() < elements) {
            buffer = execution::gpu::device::GPUDeviceBuffer<Element>(elements, device_backend_);
        }
    }

  public:
    /**
     * Collapse subspace via GPU GEMM rotation.
     */
    int build_correction_vectors(int basis_size,
                                 int number_of_plane_waves,
                                 int number_of_bands,
                                 int maximum_candidates,
                                 bool /*is_gamma_only*/,
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
            device_backend_->copy(
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
        device_backend_->copy(device_candidate_eigenvalues.data(),
                              candidate_eigenvalues.data(),
                              static_cast<std::size_t>(number_of_candidates) * sizeof(Real));
        std::vector<bool> none_converged(static_cast<std::size_t>(number_of_candidates), false);
        apply_preconditioner(number_of_candidates,
                             number_of_plane_waves,
                             kinetic_energy_squared,
                             device_candidate_eigenvalues.data(),
                             none_converged,
                             correction_columns);

        make_origin_coefficients_real(
            number_of_plane_waves, number_of_candidates, correction_columns);
        return number_of_candidates;
    }

    int collapse_subspace(int basis_size,
                          int number_of_plane_waves,
                          int number_of_bands,
                          int maximum_basis_size,
                          bool is_gamma_only,
                          bool /*is_identity_overlap*/,
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
        device_backend_->copy(device_eigenvector_matrix.data(),
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

        auto rotate_block = [&](Complex* block) {
            rotate_into_scratch(block);
            device_backend_->copy(block, rotation_scratch, collapsed_size * sizeof(Complex));
        };

        rotate_block(basis_vectors);
        rotate_block(hamiltonian_basis_vectors);

        /**
         * S|V> follows the basis in the way its operator calls for: rotated like it for a
         * general S, copied from it for S = 1, and left alone where it is the basis itself.
         * The last is the only case in which nothing is done, and an identity overlap alone
         * is not that case: the CPU workspace aliases S|V> to |V> for it, but the device
         * workspace keeps S|V> in a buffer of its own. Skipping on "S = 1" left that buffer
         * unrotated after every collapse, so the next orthogonalisation and subspace update
         * read stale S|V>, and a run that collapsed often converged to the wrong answer --
         * 1.77 Ry low on the two-atom silicon deck at expansion factor 2, which collapses
         * every iteration. Factor 4 hid it: small decks converge before the basis fills.
         */
        overlap_operator
            ->create_gpu_overlap_block(*device_backend_,
                                       number_of_plane_waves,
                                       number_of_bands,
                                       basis_vectors,
                                       overlap_basis_vectors)
            ->follow(rotate_block);

        make_origin_coefficients_real(number_of_plane_waves, number_of_bands, basis_vectors);
        make_origin_coefficients_real(
            number_of_plane_waves, number_of_bands, hamiltonian_basis_vectors);
        make_origin_coefficients_real(
            number_of_plane_waves, number_of_bands, overlap_basis_vectors);

        return number_of_bands;
    }

    /**
     * Extract eigenvectors using GPU GEMM rotation.
     */
    void extract_eigenvectors(int basis_size,
                              int number_of_plane_waves,
                              int number_of_bands,
                              int kpoint_index_state,
                              bool /*is_gamma_only*/,
                              const Complex* eigenvector_matrix,
                              const Complex* basis_vectors,
                              const Real* eigenvalues,
                              Complex* rotated_scratch,
                              PlanewaveKohnShamQuantumSystem& quantum_system) override {
        execution::gpu::device::GPUDeviceBuffer<Complex> device_eigenvector_matrix(
            basis_size * basis_size, device_backend_);
        device_backend_->copy(device_eigenvector_matrix.data(),
                              eigenvector_matrix,
                              basis_size * basis_size * sizeof(Complex));

        store_rotated_bands(basis_size,
                            number_of_plane_waves,
                            number_of_bands,
                            kpoint_index_state,
                            device_eigenvector_matrix.data(),
                            basis_vectors,
                            eigenvalues,
                            rotated_scratch,
                            quantum_system);
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
        device_backend_->copy(
            device_band_converged.data(), host_converged.data(), number_of_bands * sizeof(bool));

        // Copy kinetic_energy_squared to a temporary device buffer.
        execution::gpu::device::GPUDeviceBuffer<Real> device_kinetic_energy_squared(
            number_of_plane_waves, device_backend_);
        device_backend_->copy(device_kinetic_energy_squared.data(),
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

  protected:
    // ─── Steps that depend on the representation ─────────────────────

    /**
     * Project the existing basis out of the candidates: candidates -= V·(V†·S·candidates),
     * and S·candidates with them, by device GEMM. `candidate_overlap_block` is the
     * candidates' S block, which follows the update in the way its operator calls for.
     */
    virtual void project_out_existing_basis(int starting_vector_index,
                                            int number_of_candidates,
                                            int number_of_plane_waves,
                                            GPUOverlapBlock& candidate_overlap_block,
                                            Complex* basis_vectors,
                                            Complex* overlap_basis_vectors) {
        int leading_dimension = std::max(number_of_plane_waves, 1);
        Complex* candidates = &basis_vectors[starting_vector_index * number_of_plane_waves];
        int number_of_existing = starting_vector_index;

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

        device_backend_->copy(overlap_coefficients.data(),
                              device_overlap_coefficients.data(),
                              overlap_coefficients.size() * sizeof(Complex));

        execution_policy_.reduce_within_group(overlap_coefficients.data(),
                                              number_of_existing * number_of_candidates);

        device_backend_->copy(device_overlap_coefficients.data(),
                              overlap_coefficients.data(),
                              overlap_coefficients.size() * sizeof(Complex));

        /**
         * Subtract projections: candidates -= V · coeffs, and S·candidates -= S·V · coeffs
         * with it. The basis the product is taken with is the block's own side: V for the
         * candidates, S·V for their overlap block.
         */
        Complex alpha_neg_one(-1.0, 0.0), beta_one(1.0, 0.0);
        const std::ptrdiff_t candidates_offset =
            static_cast<std::ptrdiff_t>(starting_vector_index) * number_of_plane_waves;
        auto subtract_projection = [&](Complex* block) {
            gpu_blas_.gemm('N',
                           'N',
                           number_of_plane_waves,
                           number_of_candidates,
                           number_of_existing,
                           alpha_neg_one,
                           block - candidates_offset,
                           leading_dimension,
                           device_overlap_coefficients.data(),
                           number_of_existing,
                           beta_one,
                           block,
                           leading_dimension);
        };
        subtract_projection(candidates);
        candidate_overlap_block.follow(subtract_projection);

        device_backend_->synchronize();
    }

    /**
     * S-orthonormalize the candidates among themselves by Cholesky QR: factor their Gram
     * matrix on the host and apply the inverse factor to both blocks on the device.
     *
     * @return false, with the candidates untouched, when the Gram matrix is not positive
     *         definite; the caller then falls back to modified Gram-Schmidt.
     */
    virtual bool orthonormalize_candidates_by_cholesky(int starting_vector_index,
                                                       int number_of_candidates,
                                                       int number_of_plane_waves,
                                                       GPUOverlapBlock& candidate_overlap_block,
                                                       Complex* basis_vectors,
                                                       Complex* overlap_basis_vectors) {
        int leading_dimension = std::max(number_of_plane_waves, 1);
        Complex* candidates = &basis_vectors[starting_vector_index * number_of_plane_waves];

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

        device_backend_->copy(cholesky_factor.data(),
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
            /**
             * Apply R^-1 on the device. The factor is small and goes up once; the candidate
             * blocks are device-resident and a host solve would migrate both of them.
             */
            device_backend_->copy(device_cholesky_factor.data(),
                                  cholesky_factor.data(),
                                  cholesky_factor.size() * sizeof(Complex));
            Complex z_alpha_one(1.0, 0.0);
            auto solve_block = [&](Complex* block) {
                gpu_blas_.trsm('R',
                               'U',
                               'N',
                               'N',
                               number_of_plane_waves,
                               number_of_candidates,
                               z_alpha_one,
                               device_cholesky_factor.data(),
                               number_of_candidates,
                               block,
                               leading_dimension);
            };
            solve_block(candidates);
            candidate_overlap_block.follow(solve_block);
            device_backend_->synchronize();
            return true;
        }
        return false;
    }

    /**
     * One step of the modified Gram-Schmidt fallback: project the candidate now held in
     * column `destination_index`, and its S block, against the `number_of_batch_existing`
     * candidates already accepted, which start at column `starting_vector_index`.
     */
    virtual void project_candidate_against_accepted(int starting_vector_index,
                                                    int destination_index,
                                                    int number_of_batch_existing,
                                                    int number_of_plane_waves,
                                                    Complex* basis_vectors,
                                                    Complex* overlap_basis_vectors) {
        int leading_dimension = std::max(number_of_plane_waves, 1);
        int one_increment = 1;
        std::vector<Complex> z_batch_coefficients_buffer(number_of_batch_existing,
                                                         Complex(0.0, 0.0));
        auto* current_vector = reinterpret_cast<std::complex<double>*>(
            &basis_vectors[destination_index * number_of_plane_waves]);
        auto* current_overlap = reinterpret_cast<std::complex<double>*>(
            &overlap_basis_vectors[destination_index * number_of_plane_waves]);

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

    /// Re(a†·b) over the plane-wave list: the real inner product of the complex representation.
    virtual Real real_inner_product(int number_of_plane_waves,
                                    const std::complex<double>* vector_a,
                                    const std::complex<double>* vector_b) const {
        double result = 0.0;
        for (int g_index = 0; g_index < number_of_plane_waves; ++g_index) {
            result += (std::conj(vector_a[g_index]) * vector_b[g_index]).real();
        }
        return result;
    }

    /**
     * Fill the new columns of the subspace matrices, V†·(H·V_new) and V†·(S·V_new), before
     * they are reduced across the ranks sharing the vectors. Here by device GEMM, straight
     * into the caller's device-resident matrices.
     */
    virtual void compute_new_subspace_matrix_columns(int start_column,
                                                     int new_columns,
                                                     int total_basis_size,
                                                     int maximum_basis_size,
                                                     int number_of_plane_waves,
                                                     const Complex* basis_vectors,
                                                     const Complex* hamiltonian_basis_vectors,
                                                     const Complex* overlap_basis_vectors,
                                                     Complex* subspace_hamiltonian,
                                                     Complex* subspace_overlap) {
        int leading_dimension = std::max(number_of_plane_waves, 1);

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

    /**
     * Build the whole subspace matrices V†·(H·V) and V†·(S·V) into host vectors of
     * basis_size squared, before they are reduced across the ranks sharing the vectors.
     * Used by the collapse when the caller's matrices cannot be diagonalized.
     */
    virtual void build_full_subspace_matrices(int basis_size,
                                              int number_of_plane_waves,
                                              const Complex* basis_vectors,
                                              const Complex* hamiltonian_basis_vectors,
                                              const Complex* overlap_basis_vectors,
                                              std::vector<Complex>& subspace_hamiltonian,
                                              std::vector<Complex>& subspace_overlap) {
        int leading_dimension = std::max(number_of_plane_waves, 1);
        Complex alpha_one(1.0, 0.0), beta_zero(0.0, 0.0);

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

        device_backend_->copy(subspace_hamiltonian.data(),
                              device_subspace_hamiltonian.data(),
                              subspace_hamiltonian.size() * sizeof(Complex));
        device_backend_->copy(subspace_overlap.data(),
                              device_subspace_overlap.data(),
                              subspace_overlap.size() * sizeof(Complex));
    }

    /**
     * Rotate the basis by the subspace eigenvectors, already on the device with leading
     * dimension basis_size, into the bands of the k-point state, and store the eigenvalues.
     * `rotated_scratch` is a device buffer of number_of_bands * number_of_plane_waves for a
     * representation that cannot write the bands in place; this one can and leaves it alone.
     */
    virtual void store_rotated_bands(int basis_size,
                                     int number_of_plane_waves,
                                     int number_of_bands,
                                     int kpoint_index_state,
                                     const Complex* device_eigenvector_matrix,
                                     const Complex* basis_vectors,
                                     const Real* eigenvalues,
                                     Complex* /*rotated_scratch*/,
                                     PlanewaveKohnShamQuantumSystem& quantum_system) {
        int leading_dimension = std::max(number_of_plane_waves, 1);
        Complex alpha_one(1.0, 0.0), beta_zero(0.0, 0.0);

        /**
         * The rotation is written straight into the k-point state's band block on the
         * device, where the next solve's seed and the density read it. Nothing comes
         * back to the host here; the host copy is refreshed if and when host code asks
         * for the bands.
         */
        gpu_blas_.gemm(
            'N',
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
            quantum_system.overwrite_device_band_block(kpoint_index_state, device_backend_),
            leading_dimension);
        device_backend_->synchronize();
        for (int band_index = 0; band_index < number_of_bands; ++band_index) {
            quantum_system.set_eigenvalue(kpoint_index_state, band_index, eigenvalues[band_index]);
        }
    }

    /**
     * Apply the representation's constraint on the G = 0 coefficient of each of
     * `number_of_bands` vectors. The complex representation has none.
     */
    virtual void make_origin_coefficients_real(int /*number_of_plane_waves*/,
                                               int /*number_of_bands*/,
                                               Complex* /*vectors*/) const {}

    /**
     * Solve the subspace eigenproblem on the host: both matrices come down from the device
     * and the subspace diagonalizer of this execution policy solves them. `is_gamma_only`
     * is handed to that diagonalizer unchanged.
     *
     * @return the solver's info code (0 = success).
     */
    int diagonalize_subspace_on_host(int basis_size,
                                     int maximum_basis_size,
                                     bool is_gamma_only,
                                     const Complex* subspace_hamiltonian,
                                     const Complex* subspace_overlap,
                                     Real* eigenvalues,
                                     Complex* eigenvector_matrix) {
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
        /**
         * The caller's matrices are device buffers, which the device updates
         * again right after this solve. Reading them element by element from
         * here pulled their pages to the host and the next update pushed them
         * back, on every iteration. The columns in use come over in one
         * explicit copy each instead, and the compaction reads host memory.
         */
        const std::size_t staged_elements =
            static_cast<std::size_t>(basis_size) * maximum_basis_size;
        std::optional<StageTimer> stage_timer(std::in_place, "davidson.subspace.download");
        host_subspace_hamiltonian_.resize(staged_elements);
        host_subspace_overlap_.resize(staged_elements);
        device_backend_->copy(host_subspace_hamiltonian_.data(),
                              subspace_hamiltonian,
                              staged_elements * sizeof(Complex));
        device_backend_->copy(
            host_subspace_overlap_.data(), subspace_overlap, staged_elements * sizeof(Complex));
        for (int column = 0; column < basis_size; ++column) {
            for (int row = 0; row < basis_size; ++row) {
                eigenvector_matrix[column * basis_size + row] =
                    host_subspace_hamiltonian_[column * maximum_basis_size + row];
                diagonalize_overlap_copy_workspace_[column * basis_size + row] =
                    host_subspace_overlap_[column * maximum_basis_size + row];
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
        stage_timer.emplace("davidson.subspace.host_solve");
        if (this_rank_solves) {
            info = subspace_diagonalizer_->solve(basis_size,
                                                 eigenvector_matrix,
                                                 diagonalize_overlap_copy_workspace_.data(),
                                                 eigenvalues,
                                                 eigenvector_matrix,
                                                 is_gamma_only);
        }
        stage_timer.reset();

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

    // ─── Member data shared with the representations ─────────────────

    /// Device BLAS for the products over the plane-wave dimension.
    execution::gpu::blas::GPUBLAS& gpu_blas_;
    /// Execution policy, for the reductions across the ranks sharing the vectors.
    const execution::ExecutionPolicy& execution_policy_;
    /// Device backend the buffers are allocated on and copied through.
    std::shared_ptr<execution::gpu::device::GPUDeviceBackend> device_backend_;

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

        build_full_subspace_matrices(basis_size,
                                     number_of_plane_waves,
                                     basis_vectors,
                                     hamiltonian_basis_vectors,
                                     overlap_basis_vectors,
                                     subspace_hamiltonian,
                                     subspace_overlap);

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

    // ─── Member data ──────────────────────────────────────────────────

    /**
     * Device workspaces of the device subspace solve, kept between calls and only ever
     * grown. Declared after the backend, which they are made on.
     */
    execution::gpu::device::GPUDeviceBuffer<Complex> device_solve_hamiltonian_{0, device_backend_};
    execution::gpu::device::GPUDeviceBuffer<Complex> device_solve_overlap_{0, device_backend_};
    execution::gpu::device::GPUDeviceBuffer<Real> device_solve_eigenvalues_{0, device_backend_};
    std::unique_ptr<SubspaceDiagonalizer> subspace_diagonalizer_;
    /// Null for a host solve of the subspace eigenproblem; see set_device_eigensolver().
    execution::gpu::lapack::GPUEigensolver* device_eigensolver_ = nullptr;

    // ─── Workspaces ───────────────────────────────────────────────────

    std::vector<Complex> diagonalize_overlap_copy_workspace_;

    /// Host copies of the caller's device-resident subspace matrices; see diagonalize_subspace().

    std::vector<Complex> host_subspace_hamiltonian_;

    std::vector<Complex> host_subspace_overlap_;
};

} // namespace q3
