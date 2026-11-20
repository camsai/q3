#pragma once

/**
 * Factory for the device subspace manager.
 *
 * The plane-wave representation decides which manager a run needs: the general complex
 * one, or the gamma-point one, whose inner products run over half the sphere in real
 * arithmetic. That choice is made here, once, so that neither manager tests for it.
 */

#include <memory>

#include "layer_0_core/execution/gpu/device/init.hpp"
#include "layer_0_core/execution/init.hpp"
#include "layer_0_core/math/linear_algebra/gpu/blas/init.hpp"
#include "layer_4_computation/diagonalizer/planewave/internal/subspace_manager/gpu.hpp"
#include "layer_4_computation/diagonalizer/planewave/internal/subspace_manager/gpu_gamma_point.hpp"

namespace q3 {

/**
 * Create the device subspace manager for a plane-wave representation.
 *
 * @param is_gamma_only     Whether the basis is the half sphere of a gamma-point calculation.
 * @param gpu_blas          GPU BLAS wrapper for GEMM operations; must outlive the manager.
 * @param execution_policy  Execution policy for band-group collectives; must outlive the manager.
 * @param device_backend    GPU device backend for buffer allocation.
 * @return GammaPointGPUPlanewaveSubspaceManager when `is_gamma_only`, and
 *         GPUPlanewaveSubspaceManager, the complex manager, otherwise.
 */
inline std::unique_ptr<GPUPlanewaveSubspaceManager> create_gpu_planewave_subspace_manager(
    bool is_gamma_only,
    execution::gpu::blas::GPUBLAS& gpu_blas,
    const execution::ExecutionPolicy& execution_policy,
    std::shared_ptr<execution::gpu::device::GPUDeviceBackend> device_backend) {
    if (is_gamma_only) {
        return std::make_unique<GammaPointGPUPlanewaveSubspaceManager>(
            gpu_blas, execution_policy, device_backend);
    }
    return std::make_unique<GPUPlanewaveSubspaceManager>(
        gpu_blas, execution_policy, device_backend);
}

} // namespace q3
