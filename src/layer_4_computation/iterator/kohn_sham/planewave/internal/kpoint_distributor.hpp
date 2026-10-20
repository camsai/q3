#pragma once

/**
 * K-point distributor for pool-based parallelism.
 *
 * Given the global IBZ k-point grid and the pool topology (pool index and
 * total number of pools), produces the local subset of k-points assigned
 * to this pool using a round-robin distribution.
 *
 * Round-robin assignment: pool p owns global k-point indices
 *   { p, p + number_of_pools, p + 2*number_of_pools, ... }
 *
 * Serial by construction — no MPI headers. Used by PlanewaveMPIExecutionPolicy
 * to compute each pool's k-point subset; trivially usable in single-process
 * builds (one pool owning everything).
 */

#include "layer_1_structure/reciprocal_space/kpoints_grid.hpp"
#include <stdexcept>
#include <vector>

namespace q3 {
namespace parallel {

class KPointDistributor {
  public:
    /**
     * Distribute k-points across pools.
     *
     * @param global_kpoint_grid   The full IBZ k-point grid.
     * @param pool_index           Zero-based index of this pool.
     * @param number_of_pools      Total number of pools.
     */
    KPointDistributor(const KPointsGrid& global_kpoint_grid, int pool_index, int number_of_pools)
        : pool_index_(pool_index), number_of_pools_(number_of_pools) {
        if (number_of_pools < 1) {
            throw std::invalid_argument("KPointDistributor: number_of_pools must be at least 1.");
        }
        if (pool_index < 0 || pool_index >= number_of_pools) {
            throw std::invalid_argument(
                "KPointDistributor: pool_index must be in [0, number_of_pools).");
        }

        const int total_kpoints = global_kpoint_grid.number_of_kpoints();

        // Round-robin: pool p owns global indices { p, p+N, p+2N, ... }
        for (int global_index = pool_index; global_index < total_kpoints;
             global_index += number_of_pools) {
            local_kpoint_grid_.add_kpoint(global_kpoint_grid[global_index]);
            local_to_global_kpoint_map_.push_back(global_index);
        }

        /**
         * Preserve the original grid metadata (dimensions and shifts) so that
         * downstream gamma-only detection (grid == 1×1×1, no shift) still works
         * on the distributed local grid.
         */
        local_kpoint_grid_.set_grid_metadata(global_kpoint_grid.grid_dimensions(),
                                             global_kpoint_grid.grid_shifts());
    }

    /**
     * The local k-point grid assigned to this pool.
     *
     * Weights are preserved from the global grid. The sum of weights
     * across all pools equals the sum of global weights.
     */
    [[nodiscard]] const KPointsGrid& local_kpoint_grid() const {
        return local_kpoint_grid_;
    }

    /**
     * Maps local k-point index → global k-point index.
     *
     * Used to reassemble eigenvalues from all pools into the global ordering
     * expected by results.json.
     */
    [[nodiscard]] const std::vector<int>& local_to_global_kpoint_map() const {
        return local_to_global_kpoint_map_;
    }

    [[nodiscard]] int number_of_local_kpoints() const {
        return static_cast<int>(local_to_global_kpoint_map_.size());
    }

    [[nodiscard]] int pool_index() const {
        return pool_index_;
    }
    [[nodiscard]] int number_of_pools() const {
        return number_of_pools_;
    }

  private:
    KPointsGrid local_kpoint_grid_;
    std::vector<int> local_to_global_kpoint_map_;
    int pool_index_;
    int number_of_pools_;
};

} // namespace parallel
} // namespace q3
