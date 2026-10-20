#pragma once

/**
 * PlanewaveGPUExecutionPolicy — GPU implementation of PlanewaveExecutionPolicy.
 *
 * Extends PlanewaveExecutionPolicy to provide GPU-accelerated band-group
 * vector operations by delegating BLAS calls to GPUBLAS. Pool-level
 * operations (reduce_charge_density, reduce_scalar, etc.) remain no-ops
 * in this single-process GPU policy.
 *
 * For GPU+MPI, see PlanewaveGPUMPIExecutionPolicy (future).
 *
 * No CUDA headers. No Metal headers. Vendor-neutral.
 */

#include "layer_0_core/execution/gpu/device/init.hpp"
#include "layer_0_core/execution/gpu/init.hpp"
#include "layer_0_core/math/linear_algebra/gpu/blas/init.hpp"
#include "layer_3_representation/basis/planewave/internal/fft/fft_grid.hpp"
#include "layer_3_representation/basis/planewave/internal/fft/gpu/init.hpp"
#include "layer_4_computation/iterator/kohn_sham/planewave/execution/init.hpp"

#include <cstring>
#include <memory>
#include <string>

namespace q3 {
namespace execution {

class PlanewaveGPUExecutionPolicy : public gpu::GPUExecutionPolicy,
                                    public PlanewaveExecutionPolicy {
  public:
    /**
     * Construct a planewave GPU execution policy.
     *
     * @param device_backend  Shared pointer to the GPU device backend.
     * @param gpu_blas        Reference to the GPU BLAS wrapper for vector operations.
     *                        Must outlive this policy.
     */
    PlanewaveGPUExecutionPolicy(std::shared_ptr<gpu::device::GPUDeviceBackend> device_backend,
                                gpu::blas::GPUBLAS& gpu_blas)
        : GPUExecutionPolicy(std::move(device_backend)), gpu_blas_(gpu_blas) {}

    // ── Band-group vector operations (delegated to GPUBLAS) ──────────────

    /**
     * Compute the inner product x† · y via GPUBLAS.
     */
    ComplexValue dot_product(int number_of_elements,
                             const ComplexValue* vector_x,
                             const ComplexValue* vector_y) const override {
        return gpu_blas_.dot_product(number_of_elements, vector_x, vector_y);
    }

    /**
     * Scaled vector addition via GPUBLAS: vector_y += scaling_factor * vector_x.
     */
    void add_scaled_vector(int number_of_elements,
                           ComplexValue scaling_factor,
                           const ComplexValue* vector_x,
                           ComplexValue* vector_y) const override {
        gpu_blas_.axpy(number_of_elements, scaling_factor, vector_x, vector_y);
    }

    /**
     * Euclidean norm via GPUBLAS.
     */
    Real norm(int number_of_elements, const ComplexValue* vector_x) const override {
        return gpu_blas_.norm(number_of_elements, vector_x);
    }

    // ── Planewave pool-level operations (no-ops in single-process GPU) ───

    int number_of_pools() const override {
        return 1;
    }
    bool one_process_owns_complete_state() const override {
        return true;
    }
    /// The single GPU process holds every k-point whole.
    bool each_process_owns_whole_kpoints() const override {
        return true;
    }
    /// Identity: the single GPU process holds the global k-point grid in order.
    int global_kpoint_index(int local_kpoint_index) const override {
        return local_kpoint_index;
    }

    void reduce_charge_density(std::vector<std::vector<Real>>& /*charge_density*/) const override {}

    void reduce_scalar(Real& /*value*/) const override {}
    void reduce_vector(std::vector<Real>& /*values*/) const override {}
    void reduce_scalar_max(int& /*value*/) const override {}
    void reduce_scalar_max(Real& /*value*/) const override {}
    void reduce_scalar_min(Real& /*value*/) const override {}
    void reduce_band_group_scalar_min(int& /*value*/) const override {}

    void reduce_within_pool_scalar(Real& /*value*/) const override {}
    void reduce_within_pool_vector(std::vector<Real>& /*values*/) const override {}

    /**
     * Identity: return the global grid unchanged.
     */
    KPointsGrid local_kpoint_grid(const KPointsGrid& global_kpoint_grid) const override {
        return global_kpoint_grid;
    }

    std::vector<Real>
    assemble_global_kpoint_state_data(const std::vector<Real>& local_data,
                                      int /*local_number_of_physical_kpoints*/,
                                      int /*number_of_spin_components*/,
                                      int /*number_of_values_per_state*/) const override {
        return local_data;
    }

    /**
     * Create a GPU FFT grid instance.
     *
     * Intentionally returns nullptr: vendor GPU FFT grids (`GPUFFTGrid`,
     * cuFFT/hipFFT/SYCL-backed) exist but are constructed by the GPU entry
     * points (main_gpu*.cpp) and handed to the basis directly, not created
     * through this policy hook. The caller must check for null and fall
     * back to a serial FFT grid.
     */
    std::unique_ptr<FFTGrid> create_fft_grid() const override {
        return nullptr;
    }

    /**
     * Access the GPU BLAS wrapper.
     */
    gpu::blas::GPUBLAS& gpu_blas() {
        return gpu_blas_;
    }

    /**
     * Access the GPU BLAS wrapper (const).
     */
    const gpu::blas::GPUBLAS& gpu_blas() const {
        return gpu_blas_;
    }

  private:
    gpu::blas::GPUBLAS& gpu_blas_;
};

} // namespace execution
} // namespace q3
