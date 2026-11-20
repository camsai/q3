#pragma once

/**
 * Factory for the starting-rotation provider of a device diagonalizer.
 *
 * The device applicator is known to reproduce the initializer's host Hamiltonian for one
 * case: the standard eigenproblem (identity overlap; with an overlap operator the initializer
 * applies S itself and takes its D from elsewhere than the device applicator does before the
 * first potential update), no Hubbard term, not the gamma-point representation, and vectors
 * this rank holds whole (the device projections are not reduced over a band group). That
 * case gets the device provider; every other run keeps the caller's host solver. The choice
 * is made here, once per run, so that the diagonalizer tests none of it.
 */

#include <functional>
#include <memory>
#include <utility>

#include "layer_0_core/execution/gpu/device/init.hpp"
#include "layer_0_core/math/linear_algebra/gpu/blas/init.hpp"
#include "layer_4_computation/diagonalizer/planewave/internal/hamiltonian_applicator/init.hpp"
#include "layer_4_computation/diagonalizer/planewave/internal/hamiltonian_applicator/starting_subspace_solver_provider.hpp"
#include "layer_4_computation/diagonalizer/planewave/internal/hamiltonian_applicator/starting_subspace_solver_provider_gpu.hpp"
#include "layer_4_computation/diagonalizer/planewave/internal/hamiltonian_applicator/starting_subspace_solver_provider_serial.hpp"

namespace q3 {

/// What decides who does a run's starting rotation; see the file comment.
struct StartingRotationContext {
    /// Whether a DFT+U term has been given to the diagonalizer.
    bool has_hubbard_term = false;
    /// Whether the basis is the half sphere of a gamma-point calculation.
    bool is_gamma_only = false;
    /// Ranks sharing one band group.
    int band_group_size = 1;
    /// Whether the overlap operator of the run is S = 1.
    bool overlap_is_identity = true;
};

/**
 * Create the starting-rotation provider for a device diagonalizer.
 *
 * @param context         The run's representation, band group, overlap and Hubbard term.
 * @param applicator      Returns the device applicator, creating it if need be; called only
 *                        when the device provider is chosen, so that a run kept on the host
 *                        does not make the applicator before its first solve. The applicator
 *                        must outlive the provider.
 * @param gpu_blas        GPU BLAS wrapper; must outlive the provider.
 * @param device_backend  GPU device backend for the solver's buffers.
 * @return GPUStartingSubspaceSolverProvider in the case named in the file comment, and
 *         SerialStartingSubspaceSolverProvider otherwise.
 */
inline std::unique_ptr<StartingSubspaceSolverProvider> create_gpu_starting_subspace_solver_provider(
    const StartingRotationContext& context,
    const std::function<PlanewaveHamiltonianApplicator&()>& applicator,
    const execution::gpu::blas::GPUBLAS& gpu_blas,
    std::shared_ptr<execution::gpu::device::GPUDeviceBackend> device_backend) {
    if (context.has_hubbard_term || context.is_gamma_only || context.band_group_size > 1 ||
        !context.overlap_is_identity) {
        return std::make_unique<SerialStartingSubspaceSolverProvider>();
    }
    return std::make_unique<GPUStartingSubspaceSolverProvider>(
        applicator(), gpu_blas, std::move(device_backend));
}

} // namespace q3
