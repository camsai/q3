#pragma once

/**
 * PlanewaveSerialExecutionPolicy — serial implementation of PlanewaveExecutionPolicy.
 *
 * Inherits band-group ops from SerialExecutionPolicy.
 * All pool-level operations are no-ops: in a single-process run every process
 * already holds the complete charge density and k-point grid.
 */

#include "layer_0_core/execution/serial.hpp"
#include "layer_3_representation/basis/planewave/internal/fft/fft_grid_serial.hpp"
#include "layer_4_computation/iterator/kohn_sham/planewave/execution/init.hpp"

namespace q3 {
namespace execution {

class PlanewaveSerialExecutionPolicy : public SerialExecutionPolicy,
                                       public PlanewaveExecutionPolicy {
  public:
    /**
     * Band-group ops inherited from SerialExecutionPolicy:
     *   dot_product, add_scaled_vector, norm
     */

    /**
     * No-op: single process already holds the full charge density.
     */
    int number_of_pools() const override {
        return 1;
    }
    bool one_process_owns_complete_state() const override {
        return true;
    }
    /// The single process holds every k-point whole.
    bool each_process_owns_whole_kpoints() const override {
        return true;
    }
    /// Identity: the single process holds the global k-point grid in order.
    int global_kpoint_index(int local_kpoint_index) const override {
        return local_kpoint_index;
    }
    void reduce_charge_density(std::vector<std::vector<Real>>& /*charge_density*/) const override {}

    /**
     * No-op: single process already holds the full scalar value.
     */
    void reduce_scalar(Real& /*value*/) const override {}
    void reduce_vector(std::vector<Real>& /*values*/) const override {}
    void reduce_scalar_max(int& /*value*/) const override {}
    void reduce_scalar_max(Real& /*value*/) const override {}
    void reduce_scalar_min(Real& /*value*/) const override {}
    void reduce_band_group_scalar_min(int& /*value*/) const override {}

    /**
     * No-op: single process owns all grid slabs (no slab decomposition).
     */
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

    std::unique_ptr<FFTGrid> create_fft_grid() const override {
        auto grid = std::make_unique<SerialFFTGrid>();
        if (bands_are_transformed_on_a_device_) {
            grid->honor_planning_time_limit();
            grid->limit_planning_time(device_run_planning_seconds_per_plan);
        }
        return grid;
    }

    /**
     * Say that this run transforms its bands on a device, so that the host grids are made
     * for what is left for them. The density grid's transforms are then the only host
     * transforms that matter, and they are made one at a time: each gets every thread, where
     * on a grid of single-threaded slots it used one core. And neither grid is worth an
     * exhaustive planning search, which cost such a run 0.9 s of setup: both are planned
     * under a time cap. Not the default, because a threaded plan rounds differently from a
     * single-threaded one and a host run's results are held bit for bit.
     */
    void set_bands_are_transformed_on_a_device() {
        bands_are_transformed_on_a_device_ = true;
    }

    /// The density grid: threaded as a whole on a device run, and otherwise as any other grid.
    std::unique_ptr<FFTGrid> create_density_fft_grid() const override {
        auto density_grid = std::make_unique<SerialFFTGrid>();
        if (bands_are_transformed_on_a_device_) {
            density_grid->give_every_thread_to_each_transform();
        }
        return density_grid;
    }

    /// Planner time cap, per plan, for the host grids of a run that transforms on a device.
    static constexpr double device_run_planning_seconds_per_plan = 0.02;

  private:
    bool bands_are_transformed_on_a_device_ = false;
};

} // namespace execution
} // namespace q3
