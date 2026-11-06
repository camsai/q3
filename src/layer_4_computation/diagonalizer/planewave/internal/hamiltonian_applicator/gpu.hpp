#pragma once

/**
 * GPUPlanewaveHamiltonianApplicator — GPU implementation of H|ψ⟩ application.
 *
 * Implements PlanewaveHamiltonianApplicator using GPU math wrappers (GPUBLAS,
 * GPUFFTGrid) and device memory (GPUDeviceBuffer). All data transferred to
 * the GPU is stored in GPUDeviceBuffer members; all BLAS calls go through
 * the GPUBLAS interface.
 *
 * Supports both double precision (FP64) and single precision (FP32) paths.
 * The FP32 path is used by the mixed-precision Davidson loop when the
 * precision policy indicates that single precision is appropriate.
 *
 * Constructor takes the same planewave context as SerialPlanewaveHamiltonianApplicator,
 * plus GPUBLAS&, GPUFFTGrid&, and shared_ptr<GPUDeviceBackend>.
 *
 * No CUDA headers. No Metal headers. Uses only the vendor-neutral GPU abstractions.
 */

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "io/logger.hpp"
#include "layer_0_core/execution/gpu/device/init.hpp"
#include "layer_0_core/execution/gpu/device/internal/buffer.hpp"
#include "layer_0_core/math/linear_algebra/gpu/blas/init.hpp"
#include "layer_0_core/math/linear_algebra/gpu/lapack/init.hpp"
#include "layer_0_core/math/tolerances.hpp"
#include "layer_0_core/types/precision_cast.hpp"
#include "layer_0_core/types/type_aliases.hpp"
#include "layer_1_structure/crystal/crystal.hpp"
#include "layer_1_structure/reciprocal_space/kpoints_grid.hpp"
#include "layer_3_representation/basis/planewave/init.hpp"
#include "layer_3_representation/basis/planewave/internal/fft/fft_grid.hpp"
#include "layer_3_representation/basis/planewave/internal/fft/gpu/init.hpp"
#include "layer_3_representation/pseudopotential/planewave/internal/projector.hpp"
#include "layer_3_representation/quantum_system/kohn_sham/planewave/norm_conserving/init.hpp"
#include "layer_4_computation/diagonalizer/planewave/internal/hamiltonian_applicator/gpu_fft_batch.hpp"
#include "layer_4_computation/diagonalizer/planewave/internal/hamiltonian_applicator/gpu_kernels.hpp"
#include "layer_4_computation/diagonalizer/planewave/internal/hamiltonian_applicator/init.hpp"
#include "layer_4_computation/diagonalizer/planewave/internal/hamiltonian_applicator/planewave_potential_utils.hpp"

namespace q3 {

class GPUPlanewaveHamiltonianApplicator : public PlanewaveHamiltonianApplicator {
  public:
    /**
     * Construct with run-constant planewave context and GPU math wrappers.
     *
     * All references and the shared_ptr must outlive this object.
     *
     * @param crystal_structure              Crystal geometry.
     * @param planewave_basis                Planewave basis (G-vectors, FFT grids).
     * @param kpoint_grid                    K-point grid for Brillouin zone sampling.
     * @param pseudopotential_quantum_system Pseudopotential data (KB projectors, D-matrix).
     * @param gpu_blas                       GPU BLAS wrapper for matrix/vector operations.
     * @param gpu_fft_grid                   GPU FFT grid for potential application.
     * @param device_backend                 GPU device backend for memory allocation.
     */
    GPUPlanewaveHamiltonianApplicator(
        const Crystal& crystal_structure,
        const PlanewaveBasis& planewave_basis,
        const KPointsGrid& kpoint_grid,
        const NormConservingPlanewaveKohnShamQuantumSystem& pseudopotential_quantum_system,
        execution::gpu::blas::GPUBLAS& gpu_blas,
        gpu::GPUFFTGrid* gpu_fft_grid,
        std::shared_ptr<execution::gpu::device::GPUDeviceBackend> device_backend)
        : crystal_structure_{crystal_structure}
        , planewave_basis_{planewave_basis}
        , kpoint_grid_{kpoint_grid}
        , pseudopotential_quantum_system_{pseudopotential_quantum_system}
        , gpu_blas_{gpu_blas}
        , gpu_fft_grid_{gpu_fft_grid}
        , device_backend_{std::move(device_backend)}
        // Initialize empty GPU buffers — they will be resized as needed.
        , device_smooth_potential_(0, device_backend_)
        , device_smooth_potential_single_(0, device_backend_)
        , device_g_to_fft_mapping_(0, device_backend_)
        , device_kinetic_energy_squared_(0, device_backend_)
        , device_kinetic_energy_squared_single_(0, device_backend_)
        , device_projector_matrix_(0, device_backend_)
        , device_projector_matrix_single_(0, device_backend_)
        , device_dense_d_matrix_(0, device_backend_)
        , device_fft_workspace_(0, device_backend_)
        , device_fft_workspace_single_(0, device_backend_)
        , device_fft_batch_workspace_(0, device_backend_)
        , device_fft_batch_workspace_single_(0, device_backend_)
        , device_minus_g_to_fft_mapping_(0, device_backend_) {
        /**
         * Initialize the GPU FFT grid with the wave (smooth) grid dimensions
         * so that cuFFT plans are created before any FFT calls.
         */
        if (gpu_fft_grid_ != nullptr) {
            const auto& wave_fft = planewave_basis_.fft_wave();
            gpu_fft_grid_->init(wave_fft.grid_dimension_1(),
                                wave_fft.grid_dimension_2(),
                                wave_fft.grid_dimension_3());
        }
    }

    /**
     * Enable or disable mixed-precision mode.
     *
     * When enabled, apply_batch() runs the local potential FFT section in FP32
     * (scatter FP64→FP32, FFT, V*ψ, FFT, gather FP32→FP64). Kinetic energy
     * and nonlocal projectors remain in FP64. This gives most of the GPU FP32
     * speedup without restructuring the Davidson loop.
     */
    void set_use_mixed_precision(bool use_mixed_precision) {
        use_mixed_precision_ = use_mixed_precision;
        if (use_mixed_precision) {
            log::info("     GPU applicator: mixed-precision FP32 FFT enabled");
        }
    }

    /**
     * Update the effective potential for a new SCF iteration.
     *
     * Computes the smooth-grid potential on CPU (same as serial), then
     * copies the result to a GPUDeviceBuffer for GPU-side V(r)·ψ(r)
     * multiplication.
     */
    void update_potential(const std::vector<Real>& real_space_potential) override {
        // Compute smooth grid potential on CPU (reuses serial logic).
        cached_smooth_potential_ =
            compute_smooth_grid_potential(real_space_potential, planewave_basis_);

        int local_smooth_grid_points = planewave_basis_.fft_wave().local_grid_points();
        cached_scaled_potential_.resize(local_smooth_grid_points);
        Real inverse_total_points = 1.0 / planewave_basis_.fft_wave().total_grid_points();
        for (int index = 0; index < local_smooth_grid_points; ++index) {
            cached_scaled_potential_[index] =
                cached_smooth_potential_[index] * inverse_total_points;
        }

        // Copy scaled potential to GPU buffer.
        device_smooth_potential_ = execution::gpu::device::GPUDeviceBuffer<Real>(
            local_smooth_grid_points, device_backend_);
        std::memcpy(device_smooth_potential_.data(),
                    cached_scaled_potential_.data(),
                    local_smooth_grid_points * sizeof(Real));

        // Also prepare single-precision copy for mixed-precision path.
        device_smooth_potential_single_ = execution::gpu::device::GPUDeviceBuffer<RealSingle>(
            local_smooth_grid_points, device_backend_);
        precision::cast_real_double_to_single(local_smooth_grid_points,
                                              cached_scaled_potential_.data(),
                                              device_smooth_potential_single_.data());

        smooth_potential_cache_valid_ = true;
        cached_dense_d_matrix_.clear();
    }

    /**
     * Configure for a specific k-point.
     *
     * Copies G-to-FFT mapping, kinetic energy factors, and KB projectors
     * to GPUDeviceBuffers. The KB projectors are computed on CPU and then
     * transferred to the GPU.
     */
    void prepare_kpoint(int kpoint_index,
                        int number_of_plane_waves,
                        const std::vector<int>& g_vector_indices,
                        const std::vector<Real>& kinetic_energy_squared) override {
        current_kpoint_index_ = kpoint_index;
        current_number_of_plane_waves_ = number_of_plane_waves;

        // Compute G-to-FFT mapping on CPU (same as serial).
        precompute_g_to_fft_mapping(number_of_plane_waves, g_vector_indices, planewave_basis_);

        // Copy mapping to GPU.
        device_g_to_fft_mapping_ =
            execution::gpu::device::GPUDeviceBuffer<int>(number_of_plane_waves, device_backend_);
        std::memcpy(device_g_to_fft_mapping_.data(),
                    g_to_fft_mapping_.data(),
                    number_of_plane_waves * sizeof(int));

        if (planewave_basis_.is_gamma_only()) {
            device_minus_g_to_fft_mapping_ = execution::gpu::device::GPUDeviceBuffer<int>(
                number_of_plane_waves, device_backend_);
            std::memcpy(device_minus_g_to_fft_mapping_.data(),
                        minus_g_to_fft_mapping_.data(),
                        number_of_plane_waves * sizeof(int));
        } else {
            device_minus_g_to_fft_mapping_ =
                execution::gpu::device::GPUDeviceBuffer<int>(0, device_backend_);
        }

        // Copy kinetic energy to GPU (FP64).
        device_kinetic_energy_squared_ =
            execution::gpu::device::GPUDeviceBuffer<Real>(number_of_plane_waves, device_backend_);
        std::memcpy(device_kinetic_energy_squared_.data(),
                    kinetic_energy_squared.data(),
                    number_of_plane_waves * sizeof(Real));

        // Copy kinetic energy to GPU (FP32) for mixed-precision.
        device_kinetic_energy_squared_single_ = execution::gpu::device::GPUDeviceBuffer<RealSingle>(
            number_of_plane_waves, device_backend_);
        precision::cast_real_double_to_single(number_of_plane_waves,
                                              kinetic_energy_squared.data(),
                                              device_kinetic_energy_squared_single_.data());

        // Compute KB projectors on CPU.
        Vector3D kpoint_coordinates = kpoint_grid_[kpoint_index].coordinates;
        compute_kleinman_bylander_projectors(
            kpoint_coordinates,
            crystal_structure_,
            planewave_basis_,
            pseudopotential_quantum_system_.pseudopotential_basis().pseudopotential_set(),
            pseudopotential_quantum_system_.pseudopotential_basis().beta_function_table(),
            g_vector_indices,
            number_of_plane_waves,
            pseudopotential_quantum_system_.pseudopotential_basis().total_number_of_projectors(),
            cached_projector_matrix_);

        if (!cached_projector_matrix_.empty()) {
            int total_projectors = static_cast<int>(cached_projector_matrix_.rows());
            current_total_projectors_ = total_projectors;
            int projector_data_size = total_projectors * number_of_plane_waves;

            // Copy projectors to GPU (FP64).
            device_projector_matrix_ = execution::gpu::device::GPUDeviceBuffer<Complex>(
                projector_data_size, device_backend_);
            std::memcpy(device_projector_matrix_.data(),
                        cached_projector_matrix_.data(),
                        projector_data_size * sizeof(Complex));

            // Copy projectors to GPU (FP32) for mixed-precision.
            device_projector_matrix_single_ =
                execution::gpu::device::GPUDeviceBuffer<ComplexSingle>(projector_data_size,
                                                                       device_backend_);
            precision::cast_double_to_single(projector_data_size,
                                             cached_projector_matrix_.data(),
                                             device_projector_matrix_single_.data());

            // Build and copy dense D-matrix to GPU.
            build_dense_d_matrix(total_projectors);
        } else {
            current_total_projectors_ = 0;
        }
    }

    /**
     * Apply H|ψ⟩ in double precision (FP64) to a batch of vectors.
     *
     * Uses GPU BLAS for KB projector application (3× ZGEMM) and CPU-side
     * FFT for the local potential application. The kinetic energy is applied
     * element-wise on the host (unified memory).
     *
     * All input/output pointers must come from GPUDeviceBuffer (unified memory).
     */
    void apply_batch(const Complex* input_wavefunctions,
                     Complex* output_hamiltonian_wavefunctions,
                     int number_of_vectors) override {
        int number_of_plane_waves = current_number_of_plane_waves_;
        int leading_dimension = std::max(number_of_plane_waves, 1);
        int local_smooth_grid_points = planewave_basis_.fft_wave().local_grid_points();

        // Ensure FFT workspace is large enough.
        if (static_cast<int>(device_fft_workspace_.size()) < local_smooth_grid_points) {
            device_fft_workspace_ = execution::gpu::device::GPUDeviceBuffer<Complex>(
                local_smooth_grid_points, device_backend_);
        }

        // 1. Kinetic energy (element-wise on GPU).
        diagonalizer::planewave::gpu::launch_apply_kinetic_energy(
            number_of_vectors,
            number_of_plane_waves,
            device_kinetic_energy_squared_.data(),
            input_wavefunctions,
            output_hamiltonian_wavefunctions);

        // 2. Local potential via FFT.
        if (use_mixed_precision_) {
            // Mixed-precision path: scatter FP64→FP32, FFT in FP32, gather FP32→FP64.
            apply_local_potential_mixed_precision(input_wavefunctions,
                                                  output_hamiltonian_wavefunctions,
                                                  number_of_vectors,
                                                  number_of_plane_waves,
                                                  local_smooth_grid_points);
        } else {
            // Full FP64 path: FFT via planewave_basis_.fft_wave().
            apply_local_potential_double_precision(input_wavefunctions,
                                                   output_hamiltonian_wavefunctions,
                                                   number_of_vectors,
                                                   number_of_plane_waves,
                                                   local_smooth_grid_points);
        }

        // 3. Nonlocal KB projectors via GPU BLAS (3× GEMM, always FP64).
        if (current_total_projectors_ > 0) {
            apply_nonlocal_projectors_double(input_wavefunctions,
                                             output_hamiltonian_wavefunctions,
                                             number_of_plane_waves,
                                             number_of_vectors);
        }
        device_backend_->synchronize();
    }

    /**
     * Apply H|ψ⟩ in single precision (FP32) to a batch of vectors.
     *
     * Identical algorithm to apply_batch() but operates on ComplexSingle
     * arrays and uses FP32 BLAS/FFT calls. Used by the mixed-precision
     * Davidson loop for bulk iterations before switchover.
     */
    void apply_batch_single_precision(const ComplexSingle* input_wavefunctions,
                                      ComplexSingle* output_hamiltonian_wavefunctions,
                                      int number_of_vectors) {
        int number_of_plane_waves = current_number_of_plane_waves_;
        int local_smooth_grid_points = planewave_basis_.fft_wave().local_grid_points();
        bool is_gamma = planewave_basis_.is_gamma_only();

        // Ensure single-precision FFT workspace.
        if (static_cast<int>(device_fft_workspace_single_.size()) < local_smooth_grid_points) {
            device_fft_workspace_single_ = execution::gpu::device::GPUDeviceBuffer<ComplexSingle>(
                local_smooth_grid_points, device_backend_);
        }

        // 1. Kinetic energy in FP32 on GPU.
        diagonalizer::planewave::gpu::launch_apply_kinetic_energy_single(
            number_of_vectors,
            number_of_plane_waves,
            device_kinetic_energy_squared_single_.data(),
            input_wavefunctions,
            output_hamiltonian_wavefunctions);

        /**
         * 2. Local potential via batched FFT in FP32.
         * Scatter a pass of bands → batch inverse FFT → V·ψ → batch forward FFT → gather,
         * replacing 2×N single FFT calls with 2 batched launches per pass. A pass holds
         * every band unless their workspace would pass the element budget in
         * gpu_fft_batch.hpp (camsai/q3#4); every pass runs the FFT at the same batch size,
         * so one FFT plan serves them all.
         */
        const int bands_per_pass = diagonalizer::planewave::fft_batch_band_count(
            local_smooth_grid_points, number_of_vectors);
        const std::size_t batch_workspace_size =
            static_cast<std::size_t>(local_smooth_grid_points) * bands_per_pass;
        if (device_fft_batch_workspace_single_.size() < batch_workspace_size) {
            device_fft_batch_workspace_single_ =
                execution::gpu::device::GPUDeviceBuffer<ComplexSingle>(batch_workspace_size,
                                                                       device_backend_);
        }

        for (int first_band = 0; first_band < number_of_vectors; first_band += bands_per_pass) {
            const int bands_in_pass = std::min(bands_per_pass, number_of_vectors - first_band);
            const std::size_t band_offset =
                static_cast<std::size_t>(first_band) * number_of_plane_waves;

            // Scatter this pass's bands to contiguous FFT workspaces on GPU.
            diagonalizer::planewave::gpu::launch_scatter_g_to_fft_grid_single(
                bands_in_pass,
                number_of_plane_waves,
                local_smooth_grid_points,
                device_g_to_fft_mapping_.data(),
                is_gamma ? device_minus_g_to_fft_mapping_.data() : nullptr,
                is_gamma,
                input_wavefunctions + band_offset,
                device_fft_batch_workspace_single_.data());

            // Batched inverse FFT in FP32 (the whole pass in one kernel launch).
            gpu_fft_grid_->inverse_raw_single_batch(device_fft_batch_workspace_single_.data(),
                                                    bands_per_pass);

            // Multiply each band by potential in FP32 on GPU.
            diagonalizer::planewave::gpu::launch_multiply_potential_single(
                bands_in_pass,
                local_smooth_grid_points,
                device_smooth_potential_single_.data(),
                device_fft_batch_workspace_single_.data());

            // Batched forward FFT in FP32 (the whole pass in one kernel launch).
            gpu_fft_grid_->forward_single_batch(device_fft_batch_workspace_single_.data(),
                                                bands_per_pass);

            // Gather this pass's bands from FFT workspaces on GPU.
            diagonalizer::planewave::gpu::launch_gather_fft_grid_to_g_single(
                bands_in_pass,
                number_of_plane_waves,
                local_smooth_grid_points,
                device_g_to_fft_mapping_.data(),
                device_fft_batch_workspace_single_.data(),
                output_hamiltonian_wavefunctions + band_offset);
        }

        // 3. Nonlocal KB projectors via GPU BLAS in FP32 (3× CGEMM).
        if (current_total_projectors_ > 0) {
            apply_nonlocal_projectors_single(input_wavefunctions,
                                             output_hamiltonian_wavefunctions,
                                             number_of_plane_waves,
                                             number_of_vectors);
        }
        device_backend_->synchronize();
    }

    void invalidate_smooth_potential_cache() override {
        smooth_potential_cache_valid_ = false;
        cached_dense_d_matrix_.clear();
    }

    /**
     * Whether the smooth potential cache is currently valid.
     */
    [[nodiscard]] bool is_smooth_potential_cache_valid() const override {
        return smooth_potential_cache_valid_;
    }

    /**
     * Access the cached scaled potential (for serial fallback or debugging).
     */
    [[nodiscard]] const std::vector<Real>& cached_scaled_potential() const {
        return cached_scaled_potential_;
    }

    /**
     * Access the G-to-FFT mapping for the current k-point.
     */
    [[nodiscard]] const std::vector<int>& g_to_fft_mapping() const {
        return g_to_fft_mapping_;
    }

    /**
     * Access the minus-G-to-FFT mapping (for gamma-only scatter/gather).
     */
    [[nodiscard]] const std::vector<int>& minus_g_to_fft_mapping() const {
        return minus_g_to_fft_mapping_;
    }

    /**
     * Report whether the current k-point is gamma-only.
     *
     * This affects dot product computations in the subspace manager:
     * gamma-only uses 2×Re(z·z*) instead of complex zdotc.
     */
    [[nodiscard]] bool is_gamma_only_current() const override {
        return planewave_basis_.is_gamma_only();
    }

    /**
     * No-op: GPU applicator computes projectors internally in prepare_kpoint().
     */
    void set_cached_projectors(int /*kpoint_index*/,
                               const q3::Matrix<Complex>* /*projectors*/) override {}

    /**
     * Returns nullptr — GPU applicator manages projectors internally.
     */
    [[nodiscard]] const q3::Matrix<Complex>*
    get_cached_projectors(int /*kpoint_index*/) const override {
        return nullptr;
    }

  private:
    // ─── Local potential FFT application ──────────────────────────────

    /**
     * Apply local potential via FFT in full FP64 precision.
     *
     * When a GPU FFT grid is available, uses gpu_fft_grid_ for FP64 FFTs
     * (cuFFT Z2Z on CUDA). Falls back to planewave_basis's serial FFT grid
     * (CPU FFTW) when gpu_fft_grid_ is null.
     *
     * Pipeline: scatter ψ(G) → IFFT → V(r)·ψ(r) → FFT → gather H|ψ⟩(G)
     */
    void apply_local_potential_double_precision(const Complex* input_wavefunctions,
                                                Complex* output_hamiltonian_wavefunctions,
                                                int number_of_vectors,
                                                int number_of_plane_waves,
                                                int local_smooth_grid_points) {
        bool is_gamma = planewave_basis_.is_gamma_only();

        if (gpu_fft_grid_ != nullptr) {
            /**
             * GPU path: batched FP64 FFTs (cuFFT Z2Z on CUDA).
             * scatter a pass of bands → batch inverse FFT → V·ψ → batch forward FFT → gather.
             * A pass holds every band unless their workspace would pass the element budget
             * in gpu_fft_batch.hpp (camsai/q3#4); every pass runs the FFT at the same batch
             * size, so one FFT plan serves them all.
             */
            const int bands_per_pass = diagonalizer::planewave::fft_batch_band_count(
                local_smooth_grid_points, number_of_vectors);
            const std::size_t batch_workspace_size =
                static_cast<std::size_t>(local_smooth_grid_points) * bands_per_pass;
            if (device_fft_batch_workspace_.size() < batch_workspace_size) {
                device_fft_batch_workspace_ = execution::gpu::device::GPUDeviceBuffer<Complex>(
                    batch_workspace_size, device_backend_);
            }

            for (int first_band = 0; first_band < number_of_vectors; first_band += bands_per_pass) {
                const int bands_in_pass = std::min(bands_per_pass, number_of_vectors - first_band);
                const std::size_t band_offset =
                    static_cast<std::size_t>(first_band) * number_of_plane_waves;

                // Scatter this pass's bands to contiguous FFT workspaces on GPU.
                diagonalizer::planewave::gpu::launch_scatter_g_to_fft_grid(
                    bands_in_pass,
                    number_of_plane_waves,
                    local_smooth_grid_points,
                    device_g_to_fft_mapping_.data(),
                    is_gamma ? device_minus_g_to_fft_mapping_.data() : nullptr,
                    is_gamma,
                    input_wavefunctions + band_offset,
                    device_fft_batch_workspace_.data());

                // Batched inverse FFT in FP64 (the whole pass in one kernel launch).
                gpu_fft_grid_->inverse_transform_batch(device_fft_batch_workspace_.data(),
                                                       bands_per_pass);

                // Multiply each band by potential on GPU.
                diagonalizer::planewave::gpu::launch_multiply_potential(
                    bands_in_pass,
                    local_smooth_grid_points,
                    device_smooth_potential_.data(),
                    device_fft_batch_workspace_.data());

                // Batched forward FFT in FP64 (the whole pass in one kernel launch).
                gpu_fft_grid_->forward_transform_batch(device_fft_batch_workspace_.data(),
                                                       bands_per_pass);

                // Gather this pass's bands from FFT workspaces on GPU.
                diagonalizer::planewave::gpu::launch_gather_fft_grid_to_g(
                    bands_in_pass,
                    number_of_plane_waves,
                    local_smooth_grid_points,
                    device_g_to_fft_mapping_.data(),
                    device_fft_batch_workspace_.data(),
                    output_hamiltonian_wavefunctions + band_offset);
            }
        } else {
            // CPU fallback: use serial FFT grid (FFTW).
            const FFTGrid& fft_wave = planewave_basis_.fft_wave();
            Complex* fft_input = fft_wave.input_buffer();
            for (int vector_index = 0; vector_index < number_of_vectors; ++vector_index) {
                const Complex* psi = &input_wavefunctions[vector_index * number_of_plane_waves];
                Complex* hpsi =
                    &output_hamiltonian_wavefunctions[vector_index * number_of_plane_waves];

                // Scatter to FFT grid.
                std::memset(fft_input, 0, local_smooth_grid_points * sizeof(Complex));
                for (int g_index = 0; g_index < number_of_plane_waves; ++g_index) {
                    int fft_index = g_to_fft_mapping_[g_index];
                    if (fft_index >= 0) {
                        fft_input[fft_index] = psi[g_index];
                    }
                }
                // Gamma-only: fill -G with conj(ψ(G)).
                if (planewave_basis_.is_gamma_only()) {
                    for (int g_index = 1; g_index < number_of_plane_waves; ++g_index) {
                        int fft_minus = minus_g_to_fft_mapping_[g_index];
                        fft_input[fft_minus] = std::conj(psi[g_index]);
                    }
                }

                // Inverse FFT.
                fft_wave.execute_inverse_raw_on_buffers();
                Complex* fft_output = fft_wave.output_buffer();

                // Multiply by potential.
                for (int grid_index = 0; grid_index < local_smooth_grid_points; ++grid_index) {
                    fft_output[grid_index] *= cached_scaled_potential_[grid_index];
                }

                // Forward FFT.
                fft_wave.prepare_forward_from_output();
                fft_wave.execute_forward_on_buffers();
                fft_output = fft_wave.output_buffer();

                // Gather from FFT grid.
                for (int g_index = 0; g_index < number_of_plane_waves; ++g_index) {
                    int fft_index = g_to_fft_mapping_[g_index];
                    if (fft_index >= 0) {
                        hpsi[g_index] += fft_output[fft_index];
                    }
                }
            }
        }
    }

    /**
     * Apply local potential with mixed-precision FFT.
     *
     * The scatter/IFFT/V·ψ/FFT/gather pipeline runs in FP32. On Apple Silicon
     * (Accelerate backend), the FP32 FFT is done via upcast to FP64 through FFTW.
     * On CUDA, this would use native cuFFT FP32 (cufftExecC2C) for real speedup.
     *
     * Input/output wavefunctions remain FP64 — only the FFT workspace is FP32.
     * The accumulated hpsi contribution is cast back to FP64 before adding.
     */
    void apply_local_potential_mixed_precision(const Complex* input_wavefunctions,
                                               Complex* output_hamiltonian_wavefunctions,
                                               int number_of_vectors,
                                               int number_of_plane_waves,
                                               int local_smooth_grid_points) {

        // Ensure FP32 FFT workspace and potential are ready.
        if (static_cast<int>(device_fft_workspace_single_.size()) < local_smooth_grid_points) {
            device_fft_workspace_single_ = execution::gpu::device::GPUDeviceBuffer<ComplexSingle>(
                local_smooth_grid_points, device_backend_);
        }
        if (static_cast<int>(device_smooth_potential_single_.size()) < local_smooth_grid_points) {
            // Downcast the FP64 potential to FP32 (done once per SCF iteration).
            device_smooth_potential_single_ = execution::gpu::device::GPUDeviceBuffer<RealSingle>(
                local_smooth_grid_points, device_backend_);
            for (int grid_index = 0; grid_index < local_smooth_grid_points; ++grid_index) {
                device_smooth_potential_single_.data()[grid_index] =
                    static_cast<float>(cached_scaled_potential_[grid_index]);
            }
        }

        for (int vector_index = 0; vector_index < number_of_vectors; ++vector_index) {
            const Complex* psi = &input_wavefunctions[vector_index * number_of_plane_waves];
            Complex* hpsi = &output_hamiltonian_wavefunctions[vector_index * number_of_plane_waves];

            if (gpu_fft_grid_ != nullptr) {
                /**
                 * Native GPU FP32 FFT path.
                 * Scatter FP64 ψ → FP32 workspace on GPU (interleaved cast).
                 */
                diagonalizer::planewave::gpu::launch_scatter_g_to_fft_grid_mixed(
                    1, // Note: apply_local_potential_mixed_precision operates in a loop over
                       // vector_index, so number_of_vectors = 1
                    number_of_plane_waves,
                    local_smooth_grid_points,
                    device_g_to_fft_mapping_.data(),
                    planewave_basis_.is_gamma_only() ? device_minus_g_to_fft_mapping_.data()
                                                     : nullptr,
                    planewave_basis_.is_gamma_only(),
                    psi,
                    device_fft_workspace_single_.data());

                // Inverse FFT in FP32.
                gpu_fft_grid_->inverse_raw_single(device_fft_workspace_single_.data());

                // Multiply by FP32 potential on GPU.
                diagonalizer::planewave::gpu::launch_multiply_potential_single(
                    1,
                    local_smooth_grid_points,
                    device_smooth_potential_single_.data(),
                    device_fft_workspace_single_.data());

                // Forward FFT in FP32.
                gpu_fft_grid_->forward_single(device_fft_workspace_single_.data());

                // Gather FP32 → FP64 on GPU.
                diagonalizer::planewave::gpu::launch_gather_fft_grid_to_g_mixed(
                    1,
                    number_of_plane_waves,
                    local_smooth_grid_points,
                    device_g_to_fft_mapping_.data(),
                    device_fft_workspace_single_.data(),
                    hpsi);
            } else {
                /**
                 * Fallback: no GPU FFT grid. Use planewave_basis FFTW (FP64)
                 * with FP32 rounding at scatter and potential multiply.
                 */
                const FFTGrid& fft_wave = planewave_basis_.fft_wave();
                Complex* fft_buffer = fft_wave.input_buffer();

                // Scatter with FP32 rounding.
                std::memset(fft_buffer, 0, local_smooth_grid_points * sizeof(Complex));
                for (int g_index = 0; g_index < number_of_plane_waves; ++g_index) {
                    int fft_index = g_to_fft_mapping_[g_index];
                    if (fft_index >= 0) {
                        float real_part = static_cast<float>(psi[g_index].real());
                        float imag_part = static_cast<float>(psi[g_index].imag());
                        fft_buffer[fft_index] = Complex(real_part, imag_part);
                    }
                }
                // Gamma-only: fill -G with conj(ψ(G)) with FP32 rounding.
                if (planewave_basis_.is_gamma_only()) {
                    for (int g_index = 1; g_index < number_of_plane_waves; ++g_index) {
                        int fft_minus = minus_g_to_fft_mapping_[g_index];
                        float real_part = static_cast<float>(psi[g_index].real());
                        float imag_part = static_cast<float>(psi[g_index].imag());
                        fft_buffer[fft_minus] = Complex(real_part, -imag_part);
                    }
                }

                // Inverse FFT (FFTW FP64).
                fft_wave.execute_inverse_raw_on_buffers();
                Complex* fft_output = fft_wave.output_buffer();

                // Multiply by potential in FP32 precision.
                for (int grid_index = 0; grid_index < local_smooth_grid_points; ++grid_index) {
                    float potential_value = device_smooth_potential_single_.data()[grid_index];
                    float real_result =
                        static_cast<float>(fft_output[grid_index].real()) * potential_value;
                    float imag_result =
                        static_cast<float>(fft_output[grid_index].imag()) * potential_value;
                    fft_output[grid_index] = Complex(real_result, imag_result);
                }

                // Forward FFT (FFTW FP64).
                fft_wave.prepare_forward_from_output();
                fft_wave.execute_forward_on_buffers();
                fft_output = fft_wave.output_buffer();

                // Gather back to FP64.
                for (int g_index = 0; g_index < number_of_plane_waves; ++g_index) {
                    int fft_index = g_to_fft_mapping_[g_index];
                    if (fft_index >= 0) {
                        hpsi[g_index] += fft_output[fft_index];
                    }
                }
            }
        }
    }

    /**
     * Apply nonlocal KB projectors via 3× ZGEMM on GPU BLAS.
     *
     * Step 1: overlap = β† · ψ  (projector overlap)
     * Step 2: scaled = D · overlap  (D-matrix application)
     * Step 3: hψ += β · scaled  (accumulate into H|ψ⟩)
     */
    void apply_nonlocal_projectors_double(const Complex* input_wavefunctions,
                                          Complex* output_hamiltonian_wavefunctions,
                                          int number_of_plane_waves,
                                          int number_of_vectors) {

        int leading_dimension = std::max(number_of_plane_waves, 1);
        int total_projectors = current_total_projectors_;

        // Allocate GPU buffers for intermediate results.
        execution::gpu::device::GPUDeviceBuffer<Complex> overlap_buffer(
            total_projectors * number_of_vectors, device_backend_);
        execution::gpu::device::GPUDeviceBuffer<Complex> scaled_overlap_buffer(
            total_projectors * number_of_vectors, device_backend_);

        Complex alpha_one(1.0, 0.0);
        Complex beta_zero(0.0, 0.0);
        Complex beta_one(1.0, 0.0);

        // Step 1: overlap = β† · ψ
        if (planewave_basis_.is_gamma_only()) {
            /**
             * Gamma-only: use DGEMM with α=2 on interleaved real/imag parts.
             * The wavefunction only stores +G half. The full overlap is:
             *   ⟨β|ψ⟩ = 2·Re(Σ_{G>0} β*(G)·ψ(G)) + Re(β*(0)·ψ(0))
             * DGEMM on interleaved data gives 2·Re(β†·ψ), then subtract
             * the double-counted G=0 contribution.
             */
            int double_number_of_plane_waves = 2 * number_of_plane_waves;
            int double_leading_dimension = 2 * leading_dimension;

            // Use a real-valued temporary for the DGEMM output in unified memory.
            execution::gpu::device::GPUDeviceBuffer<double> device_overlap_real(
                static_cast<size_t>(total_projectors) * number_of_vectors, device_backend_);

            const double* real_projectors =
                reinterpret_cast<const double*>(device_projector_matrix_.data());
            const double* real_wavefunctions = reinterpret_cast<const double*>(input_wavefunctions);

            gpu_blas_.dgemm('T',
                            'N',
                            total_projectors,
                            number_of_vectors,
                            double_number_of_plane_waves,
                            2.0,
                            real_projectors,
                            double_leading_dimension,
                            real_wavefunctions,
                            double_leading_dimension,
                            0.0,
                            device_overlap_real.data(),
                            total_projectors);

            device_backend_->synchronize();

            /**
             * G=0 correction: subtract the double-counted G=0 term and copy to Complex.
             * Matrix is row-major (nproj × npw), so element (proj, g=0) is at
             * offset proj * npw in the flat array.
             */
            for (int vector_index = 0; vector_index < number_of_vectors; ++vector_index) {
                const Complex* psi_g0 = &input_wavefunctions[vector_index * number_of_plane_waves];
                for (int projector_index = 0; projector_index < total_projectors;
                     ++projector_index) {
                    Complex beta_g0 =
                        device_projector_matrix_.data()[projector_index * number_of_plane_waves];
                    int index = projector_index + vector_index * total_projectors;
                    double value =
                        device_overlap_real.data()[index] - (std::conj(beta_g0) * psi_g0[0]).real();
                    overlap_buffer.data()[index] = Complex(value, 0.0);
                }
            }
        } else {
            // Non-gamma: standard ZGEMM.
            gpu_blas_.gemm('C',
                           'N',
                           total_projectors,
                           number_of_vectors,
                           number_of_plane_waves,
                           alpha_one,
                           device_projector_matrix_.data(),
                           leading_dimension,
                           input_wavefunctions,
                           leading_dimension,
                           beta_zero,
                           overlap_buffer.data(),
                           total_projectors);
        }

        // Step 2: scaled = D · overlap
        gpu_blas_.gemm('N',
                       'N',
                       total_projectors,
                       number_of_vectors,
                       total_projectors,
                       alpha_one,
                       device_dense_d_matrix_.data(),
                       total_projectors,
                       overlap_buffer.data(),
                       total_projectors,
                       beta_zero,
                       scaled_overlap_buffer.data(),
                       total_projectors);

        // Step 3: hψ += β · scaled
        gpu_blas_.gemm('N',
                       'N',
                       number_of_plane_waves,
                       number_of_vectors,
                       total_projectors,
                       alpha_one,
                       device_projector_matrix_.data(),
                       leading_dimension,
                       scaled_overlap_buffer.data(),
                       total_projectors,
                       beta_one,
                       output_hamiltonian_wavefunctions,
                       leading_dimension);
    }

    // ─── KB projector application in FP32 ─────────────────────────────

    /**
     * Apply nonlocal KB projectors via 3× CGEMM on GPU BLAS (single precision).
     */
    void apply_nonlocal_projectors_single(const ComplexSingle* input_wavefunctions,
                                          ComplexSingle* output_hamiltonian_wavefunctions,
                                          int number_of_plane_waves,
                                          int number_of_vectors) {

        int leading_dimension = std::max(number_of_plane_waves, 1);
        int total_projectors = current_total_projectors_;

        // Allocate GPU buffers for intermediate results in FP32.
        execution::gpu::device::GPUDeviceBuffer<ComplexSingle> overlap_buffer(
            total_projectors * number_of_vectors, device_backend_);
        execution::gpu::device::GPUDeviceBuffer<ComplexSingle> scaled_overlap_buffer(
            total_projectors * number_of_vectors, device_backend_);

        // Build single-precision D-matrix in a temporary buffer.
        execution::gpu::device::GPUDeviceBuffer<ComplexSingle> device_d_matrix_single(
            total_projectors * total_projectors, device_backend_);
        precision::cast_double_to_single(total_projectors * total_projectors,
                                         device_dense_d_matrix_.data(),
                                         device_d_matrix_single.data());

        ComplexSingle alpha_one(1.0f, 0.0f);
        ComplexSingle beta_zero(0.0f, 0.0f);
        ComplexSingle beta_one(1.0f, 0.0f);

        // Step 1: overlap = β† · ψ (FP32)
        gpu_blas_.gemm_single('C',
                              'N',
                              total_projectors,
                              number_of_vectors,
                              number_of_plane_waves,
                              alpha_one,
                              device_projector_matrix_single_.data(),
                              leading_dimension,
                              input_wavefunctions,
                              leading_dimension,
                              beta_zero,
                              overlap_buffer.data(),
                              total_projectors);

        // Step 2: scaled = D · overlap (FP32)
        gpu_blas_.gemm_single('N',
                              'N',
                              total_projectors,
                              number_of_vectors,
                              total_projectors,
                              alpha_one,
                              device_d_matrix_single.data(),
                              total_projectors,
                              overlap_buffer.data(),
                              total_projectors,
                              beta_zero,
                              scaled_overlap_buffer.data(),
                              total_projectors);

        // Step 3: hψ += β · scaled (FP32)
        gpu_blas_.gemm_single('N',
                              'N',
                              number_of_plane_waves,
                              number_of_vectors,
                              total_projectors,
                              alpha_one,
                              device_projector_matrix_single_.data(),
                              leading_dimension,
                              scaled_overlap_buffer.data(),
                              total_projectors,
                              beta_one,
                              output_hamiltonian_wavefunctions,
                              leading_dimension);
    }

    // ─── D-matrix construction ────────────────────────────────────────

    /**
     * Build and cache the dense block-diagonal D-matrix on GPU.
     */
    void build_dense_d_matrix(int total_projectors) {
        cached_dense_d_matrix_.assign(static_cast<size_t>(total_projectors) * total_projectors,
                                      Complex(0.0, 0.0));

        int starting_projector_index = 0;
        for (int atom_index = 0; atom_index < crystal_structure_.number_of_atoms(); ++atom_index) {
            int species_index = crystal_structure_.species_type_index(atom_index);
            int projectors_for_atom = pseudopotential_quantum_system_.pseudopotential_basis()
                                          .number_of_projector_channels(species_index);

            for (int projector_a = 0; projector_a < projectors_for_atom; ++projector_a) {
                for (int projector_b = 0; projector_b < projectors_for_atom; ++projector_b) {
                    Real d_value = pseudopotential_quantum_system_.d_matrix_element(
                        atom_index, species_index, projector_a, projector_b, spin_channel_);
                    if (std::abs(d_value) > math::tolerances::d_matrix_element_threshold) {
                        int row = starting_projector_index + projector_a;
                        int col = starting_projector_index + projector_b;
                        cached_dense_d_matrix_[static_cast<size_t>(row) +
                                               static_cast<size_t>(col) * total_projectors] =
                            Complex(d_value, 0.0);
                    }
                }
            }
            starting_projector_index += projectors_for_atom;
        }

        // Copy to GPU.
        device_dense_d_matrix_ = execution::gpu::device::GPUDeviceBuffer<Complex>(
            total_projectors * total_projectors, device_backend_);
        std::memcpy(device_dense_d_matrix_.data(),
                    cached_dense_d_matrix_.data(),
                    total_projectors * total_projectors * sizeof(Complex));
    }

    // ─── Smooth potential computation (CPU-side) ──────────────────────

    /**
     * Restrict the effective potential from the dense grid to the smooth grid.
     *
     * Uses the shared utility for core FFT + G-space truncation.
     * No MPI reduce — GPU applicator is single-process only (asserted).
     */
    std::vector<Real>
    compute_smooth_grid_potential(const std::vector<Real>& dense_real_space_potential,
                                  const PlanewaveBasis& planewave_basis) const {
        /**
         * GPU path currently supports single-process only. Multi-rank GPU requires
         * NCCL or MPI-aware unified memory, which is not yet implemented.
         */
        if (planewave_basis.fft_rho().local_grid_points() !=
            planewave_basis.fft_rho().total_grid_points()) {
            throw std::runtime_error(
                "The GPU Hamiltonian applicator does not support band-distributed MPI: this rank "
                "holds " +
                std::to_string(planewave_basis.fft_rho().local_grid_points()) + " of " +
                std::to_string(planewave_basis.fft_rho().total_grid_points()) +
                " density grid points, and the device path assumes the whole grid. Run one rank "
                "per k-point pool, or use the CPU path. This was an assert, which a release "
                "build compiles out.");
        }

        // Core: dense FFT -> G-space truncation -> smooth V(G)
        auto full_smooth_potential_g =
            compute_smooth_grid_potential_from_dense(dense_real_space_potential, planewave_basis);

        // Inverse FFT to real space (no MPI reduce needed — single process)
        return inverse_fft_smooth_potential(full_smooth_potential_g, planewave_basis);
    }

    // ─── G-vector to FFT mapping ──────────────────────────────────────

    /**
     * Pre-compute G-vector to smooth FFT grid index mapping.
     *
     * Uses the shared core mapping. GPU path does not support gamma-only.
     */
    void precompute_g_to_fft_mapping(int number_of_plane_waves,
                                     const std::vector<int>& g_vector_indices,
                                     const PlanewaveBasis& planewave_basis) {
        // Core G -> smooth FFT index mapping (shared with serial applicator)
        compute_g_to_fft_index_mapping(
            number_of_plane_waves, g_vector_indices, planewave_basis, g_to_fft_mapping_);

        // For gamma-only: pre-compute the -G FFT grid index for each G-vector.
        if (planewave_basis.is_gamma_only()) {
            minus_g_to_fft_mapping_.resize(number_of_plane_waves);
            for (int g_vector_index = 0; g_vector_index < number_of_plane_waves; g_vector_index++) {
                const auto& g_vector =
                    planewave_basis.g_vectors()[g_vector_indices[g_vector_index]];
                int miller_index_1 = static_cast<int>(std::round(g_vector.miller.x));
                int miller_index_2 = static_cast<int>(std::round(g_vector.miller.y));
                int miller_index_3 = static_cast<int>(std::round(g_vector.miller.z));
                minus_g_to_fft_mapping_[g_vector_index] =
                    planewave_basis.fft_wave().g_vector_to_fft_grid_index(
                        -miller_index_1, -miller_index_2, -miller_index_3);
            }
        } else {
            minus_g_to_fft_mapping_.clear();
        }
    }

    // ─── Run-constant context ─────────────────────────────────────────

    const Crystal& crystal_structure_;
    const PlanewaveBasis& planewave_basis_;
    const KPointsGrid& kpoint_grid_;
    const NormConservingPlanewaveKohnShamQuantumSystem& pseudopotential_quantum_system_;
    execution::gpu::blas::GPUBLAS& gpu_blas_;
    gpu::GPUFFTGrid* gpu_fft_grid_;
    std::shared_ptr<execution::gpu::device::GPUDeviceBackend> device_backend_;
    bool use_mixed_precision_ = false;

    // ─── Per-k-point state ────────────────────────────────────────────

    int current_kpoint_index_ = 0;
    int current_number_of_plane_waves_ = 0;
    int current_total_projectors_ = 0;

    // ─── CPU-side cached data ─────────────────────────────────────────

    std::vector<Real> cached_smooth_potential_;
    std::vector<Real> cached_scaled_potential_;
    bool smooth_potential_cache_valid_ = false;
    std::vector<Complex> cached_dense_d_matrix_;
    Matrix<Complex> cached_projector_matrix_;

    // ─── FFT mappings ─────────────────────────────────────────────────

    std::vector<int> g_to_fft_mapping_;
    std::vector<int> minus_g_to_fft_mapping_;

    // ─── GPU device buffers ───────────────────────────────────────────

    // Smooth potential (FP64 and FP32).
    execution::gpu::device::GPUDeviceBuffer<Real> device_smooth_potential_;
    execution::gpu::device::GPUDeviceBuffer<RealSingle> device_smooth_potential_single_;

    // G-to-FFT index mapping.
    execution::gpu::device::GPUDeviceBuffer<int> device_g_to_fft_mapping_;
    execution::gpu::device::GPUDeviceBuffer<int> device_minus_g_to_fft_mapping_;

    // Kinetic energy factors (FP64 and FP32).
    execution::gpu::device::GPUDeviceBuffer<Real> device_kinetic_energy_squared_;
    execution::gpu::device::GPUDeviceBuffer<RealSingle> device_kinetic_energy_squared_single_;

    // KB projector matrix (FP64 and FP32).
    execution::gpu::device::GPUDeviceBuffer<Complex> device_projector_matrix_;
    execution::gpu::device::GPUDeviceBuffer<ComplexSingle> device_projector_matrix_single_;

    // Dense block-diagonal D-matrix.
    execution::gpu::device::GPUDeviceBuffer<Complex> device_dense_d_matrix_;

    // FFT workspace (FP64 and FP32).
    execution::gpu::device::GPUDeviceBuffer<Complex> device_fft_workspace_;
    execution::gpu::device::GPUDeviceBuffer<ComplexSingle> device_fft_workspace_single_;

    // Batched FFT workspace: number_of_vectors × local_smooth_grid_points contiguous grids.
    execution::gpu::device::GPUDeviceBuffer<Complex> device_fft_batch_workspace_;
    execution::gpu::device::GPUDeviceBuffer<ComplexSingle> device_fft_batch_workspace_single_;
};

} // namespace q3
