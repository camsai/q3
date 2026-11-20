#pragma once

/**
 * GPUStartingSubspaceSolverProvider — the starting rotation on the device applicator.
 *
 * At each k-point the applicator is pointed at the k-point's spin channel, given the
 * potential if the one it holds has gone stale, and prepared for the k-point's plane waves;
 * one GPUStartingSubspaceSolver around it is made at the first k-point and re-targeted at
 * each one after, so its device buffers are only ever grown.
 */

#include <memory>
#include <utility>
#include <vector>

#include "layer_0_core/execution/gpu/device/init.hpp"
#include "layer_0_core/math/linear_algebra/gpu/blas/init.hpp"
#include "layer_0_core/types/init.hpp"
#include "layer_1_structure/reciprocal_space/kpoints_grid.hpp"
#include "layer_3_representation/wavefunction/planewave/starting_subspace_solver.hpp"
#include "layer_4_computation/diagonalizer/planewave/internal/hamiltonian_applicator/init.hpp"
#include "layer_4_computation/diagonalizer/planewave/internal/hamiltonian_applicator/starting_subspace_solver_gpu.hpp"
#include "layer_4_computation/diagonalizer/planewave/internal/hamiltonian_applicator/starting_subspace_solver_provider.hpp"

namespace q3 {

/// Prepares the device applicator for each k-point and hands out a device solver around it.
class GPUStartingSubspaceSolverProvider : public StartingSubspaceSolverProvider {
  public:
    /// `applicator` and `gpu_blas` must outlive this object.
    GPUStartingSubspaceSolverProvider(
        PlanewaveHamiltonianApplicator& applicator,
        const execution::gpu::blas::GPUBLAS& gpu_blas,
        std::shared_ptr<execution::gpu::device::GPUDeviceBackend> device_backend)
        : applicator_{applicator}
        , gpu_blas_{gpu_blas}
        , device_backend_{std::move(device_backend)} {}

    /// The device solver, with the applicator set to this k-point's spin channel and plane waves.
    StartingSubspaceSolver& prepare(int kpoint_index_state,
                                    int kpoint_index,
                                    int number_of_plane_waves,
                                    const std::vector<int>& g_vector_indices,
                                    const std::vector<Real>& kinetic_energy_squared,
                                    const std::vector<Real>& real_space_potential,
                                    const KPointsGrid& kpoint_grid,
                                    StartingSubspaceSolver& /*serial_solver*/) override {
        applicator_.set_spin_channel(kpoint_grid.number_of_kpoints() > 0
                                         ? kpoint_index_state / kpoint_grid.number_of_kpoints()
                                         : 0);
        if (!applicator_.is_smooth_potential_cache_valid()) {
            applicator_.update_potential(real_space_potential);
        }
        applicator_.prepare_kpoint(
            kpoint_index, number_of_plane_waves, g_vector_indices, kinetic_energy_squared);
        if (solver_ == nullptr) {
            solver_ = std::make_unique<GPUStartingSubspaceSolver>(
                applicator_, gpu_blas_, device_backend_, number_of_plane_waves);
        } else {
            solver_->set_number_of_plane_waves(number_of_plane_waves);
        }
        return *solver_;
    }

  private:
    PlanewaveHamiltonianApplicator& applicator_;
    const execution::gpu::blas::GPUBLAS& gpu_blas_;
    std::shared_ptr<execution::gpu::device::GPUDeviceBackend> device_backend_;
    /// Made at the first k-point, re-targeted after; see the class comment.
    std::unique_ptr<GPUStartingSubspaceSolver> solver_;
};

} // namespace q3
