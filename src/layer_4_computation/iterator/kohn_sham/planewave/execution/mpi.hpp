#pragma once

/**
 * PlanewaveMPIExecutionPolicy — MPI implementation of PlanewaveExecutionPolicy.
 *
 * Extends MPIExecutionPolicy (which owns the MPI lifecycle and the process
 * group communicator) with k-point pool collective operations specific to the
 * planewave representation:
 *   - Pool communicator split (each pool owns a subset of k-points)
 *   - reduce_charge_density: Allreduce density across pools via pool communicator
 *   - reduce_scalar:         Allreduce scalars (band energy, smearing entropy) across pools
 *   - local_kpoint_grid:     Round-robin k-point distribution to this pool
 *
 * This is the only file in planewave/execution/ that includes <mpi.h>.
 *
 * Topology mapping: each k-point pool is one band group — the layer-0 process
 * group doubles as the pool. Pools partition the k-point grid; ranks within a
 * pool share each diagonalization (band parallelism via G-vector slabs).
 *
 * Lifecycle (owned by this object, RAII):
 *   Constructor: MPIExecutionPolicy ctor → MPI_Init (if needed), group comm = world
 *   configure_topology: MPI_Comm_split → group (pool) + inter-pool communicators
 *   Destructor:  free inter-pool comm → MPIExecutionPolicy dtor → MPI_Finalize
 */

#include "layer_0_core/execution/mpi.hpp"
#ifdef Q3_WITH_FFTW_MPI
#include "layer_3_representation/basis/planewave/internal/fft/fft_grid_mpi.hpp"
#endif
#include "layer_3_representation/basis/planewave/internal/fft/fft_grid_sticks.hpp"
#ifdef Q3_WITH_FFTW_MPI
#endif
#include "layer_3_representation/basis/planewave/internal/fft/fft_grid_serial.hpp"
#include "layer_4_computation/iterator/kohn_sham/planewave/execution/init.hpp"
#include "layer_4_computation/iterator/kohn_sham/planewave/internal/kpoint_distributor.hpp"

#include <algorithm>
#include <exception>
#include <mpi.h>
#include <stdexcept>

namespace q3 {
namespace execution {

class PlanewaveMPIExecutionPolicy : public MPIExecutionPolicy, public PlanewaveExecutionPolicy {
  public:
    /**
     * Initialize MPI (via MPIExecutionPolicy), then split into k-point pools.
     *
     * Each process starts with 1 pool (all ranks in one band group) until
     * configure_topology() is called. Quantum ESPRESSO's initial default is also
     * a single pool, but note it does not stop there: when the rank count is
     * large relative to the FFT z-dimension and more than one k-point exists, it
     * raises the pool count on its own. Q3 never does -- the pool count is
     * whatever the caller asked for.
     */
    PlanewaveMPIExecutionPolicy() {
        number_of_pools_ = 1;
    }

    /**
     * Configure the parallel topology from the user-requested pool count.
     *
     * All pool topology logic lives here so that callers (main, factory) need
     * no knowledge of what a "pool" is or how ranks are distributed.
     *
     * @param requested_number_of_pools  Number of k-point pools requested by the
     *                                   user. Pass 0 to default to 1 pool (all
     *                                   ranks share one band group — matching QE's
     *                                   default band/FFT parallelism behaviour).
     * @throws std::invalid_argument if world_size is not divisible by the pool count.
     */
    void configure_topology(int requested_number_of_pools) {
        number_of_pools_was_requested_ = (requested_number_of_pools != 0);
        int number_of_pools = (requested_number_of_pools == 0) ? 1 : requested_number_of_pools;

        if (world_size() % number_of_pools != 0) {
            throw std::invalid_argument(
                "Number of MPI ranks (" + std::to_string(world_size()) +
                ") is not divisible by the requested number of k-point pools (" +
                std::to_string(number_of_pools) +
                "). Each pool must have the same number of ranks. "
                "Choose a number of pools that divides evenly into the rank count.");
        }

        configure_group_topology(number_of_pools);
        number_of_pools_ = number_of_pools;

        /**
         * Create inter-pool communicator: ranks with the same group_rank_index_
         * work on the same grid slabs across different pools (k-point parallel groups).
         */
        if (inter_pool_communicator_ != MPI_COMM_NULL) {
            MPI_Comm_free(&inter_pool_communicator_);
        }
        MPI_Comm_split(
            MPI_COMM_WORLD, group_rank_index_, world_rank_index(), &inter_pool_communicator_);
    }

    /**
     * Reconfigure the parallel topology after initial setup.
     *
     * Called by the factory when it needs to clamp the pool count to the number
     * of available IBZ k-points (e.g., when --pools 4 is requested but only 2
     * k-points exist). Delegates to configure_topology with the new count.
     *
     * @param new_number_of_pools  Clamped pool count (must be <= original count).
     */
    void reconfigure_topology(int new_number_of_pools) override {
        configure_topology(new_number_of_pools);
    }

    /**
     * Free the inter-pool communicator, unless an exception is in flight.
     *
     * MPI_Comm_free is collective, and this destructor runs ahead of
     * ~MPIExecutionPolicy() on the same object, so it declines for the same
     * reason and must decline independently: a base class skipping its
     * teardown does nothing for a derived one that goes first. Open MPI
     * happens to complete this call without synchronizing, which is why the
     * hang reproduced in ~MPIExecutionPolicy()'s barrier rather than here, but
     * the standard makes it collective and an implementation is free to make
     * it block -- so this cannot be left to rest on which MPI is installed.
     */
    ~PlanewaveMPIExecutionPolicy() override {
        if (std::uncaught_exceptions() > 0) {
            return;
        }
        int finalized = 0;
        MPI_Finalized(&finalized);
        if (!finalized && inter_pool_communicator_ != MPI_COMM_NULL) {
            MPI_Comm_free(&inter_pool_communicator_);
        }
    }

    // Non-copyable
    PlanewaveMPIExecutionPolicy(const PlanewaveMPIExecutionPolicy&) = delete;
    PlanewaveMPIExecutionPolicy& operator=(const PlanewaveMPIExecutionPolicy&) = delete;

    /**
     * ── Collective vector ops: inherited from MPIExecutionPolicy ──────────────
     * dot_product, add_scaled_vector, norm — reduce over the group communicator
     */

    void disable_band_distribution() override {
        MPIExecutionPolicy::disable_group_distribution();
    }

    // ── Pool-level ops ────────────────────────────────────────────────────────

    /**
     * Allreduce spin-resolved charge density across all k-point pools.
     *
     * Uses the inter-pool communicator (grouped by group_rank_index_) so that
     * each slab of the real-space rho grid is summed across matching band-rank
     * positions in different pools: rank i of pool 0 reduces with rank i of
     * pool 1, etc. This correctly accumulates k-point contributions while
     * keeping the slab decomposition intact.
     *
     * When group_size > 1, band ranks within the same pool hold non-overlapping
     * slab portions of the rho grid at different LOCAL indices. These must NOT
     * be Allreduced within the pool (that would add incompatible grid points).
     * Each band rank's slab is already the correct contribution for that
     * portion of the grid.
     *
     * Note: single-pool band parallelism (number_of_pools_==1, group_size_>1)
     * is not supported by this Allreduce design. It would require MPI_Allgather
     * to assemble the full density, which requires resizing charge_density_real_
     * throughout the SCF. That architectural change is deferred.
     */
    void reduce_charge_density(std::vector<std::vector<Real>>& charge_density) const override {
        MPI_Comm comm =
            (inter_pool_communicator_ != MPI_COMM_NULL) ? inter_pool_communicator_ : MPI_COMM_SELF;
        /**
         * A zero-length channel still enters the Allreduce. MPI accepts count 0
         * and never dereferences the buffer, whereas skipping the call on one
         * rank would leave its peers blocked forever. Sizes are uniform here by
         * construction -- rank i of every pool owns the same slab of identical
         * grids -- but a collective must not depend on that holding.
         */
        for (auto& spin_channel : charge_density) {
            MPI_Allreduce(MPI_IN_PLACE,
                          spin_channel.data(),
                          static_cast<int>(spin_channel.size()),
                          MPI_DOUBLE,
                          MPI_SUM,
                          comm);
        }
    }

    /**
     * Allreduce a scalar k-point sum across all pools.
     *
     * Uses the inter-pool communicator so that only one rank per pool
     * participates, avoiding double-counting when group_size > 1.
     */
    void reduce_scalar(Real& value) const override {
        MPI_Comm comm =
            (inter_pool_communicator_ != MPI_COMM_NULL) ? inter_pool_communicator_ : MPI_COMM_WORLD;
        MPI_Allreduce(MPI_IN_PLACE, &value, 1, MPI_DOUBLE, MPI_SUM, comm);
    }

    /**
     * Allreduce a k-point-local vector across all pools.
     *
     * Uses the inter-pool communicator for the same reason as reduce_scalar.
     */
    void reduce_vector(std::vector<Real>& values) const override {
        // No early return on an empty vector: see reduce_charge_density().
        MPI_Comm comm =
            (inter_pool_communicator_ != MPI_COMM_NULL) ? inter_pool_communicator_ : MPI_COMM_WORLD;
        MPI_Allreduce(MPI_IN_PLACE,
                      values.data(),
                      static_cast<int>(values.size()),
                      MPI_DOUBLE,
                      MPI_SUM,
                      comm);
    }

    /**
     * Allreduce an integer scalar across all pools using MPI_MAX.
     *
     * Used for summary statistics (e.g. max_plane_waves_per_kpoint) where each
     * pool holds only its local k-points and the global maximum is required.
     */
    void reduce_scalar_max(int& value) const override {
        MPI_Comm comm =
            (inter_pool_communicator_ != MPI_COMM_NULL) ? inter_pool_communicator_ : MPI_COMM_WORLD;
        MPI_Allreduce(MPI_IN_PLACE, &value, 1, MPI_INT, MPI_MAX, comm);
    }

    /**
     * Allreduce a Real scalar across all pools using MPI_MAX.
     *
     * Used for global eigenvalue range (HOMO) in Fermi energy calculation.
     */
    void reduce_scalar_max(Real& value) const override {
        MPI_Comm comm =
            (inter_pool_communicator_ != MPI_COMM_NULL) ? inter_pool_communicator_ : MPI_COMM_WORLD;
        MPI_Allreduce(MPI_IN_PLACE, &value, 1, MPI_DOUBLE, MPI_MAX, comm);
    }

    /**
     * Allreduce a Real scalar across all pools using MPI_MIN.
     *
     * Used for global eigenvalue range (LUMO) in Fermi energy calculation.
     */
    void reduce_scalar_min(Real& value) const override {
        MPI_Comm comm =
            (inter_pool_communicator_ != MPI_COMM_NULL) ? inter_pool_communicator_ : MPI_COMM_WORLD;
        MPI_Allreduce(MPI_IN_PLACE, &value, 1, MPI_DOUBLE, MPI_MIN, comm);
    }

    /**
     * Allreduce an integer scalar within the band group using MPI_MIN.
     *
     * Uses the band communicator so that ranks within the same pool can
     * collectively determine the minimum plane-wave count.
     */
    void reduce_band_group_scalar_min(int& value) const override {
        MPI_Allreduce(MPI_IN_PLACE, &value, 1, MPI_INT, MPI_MIN, get_group_communicator());
    }

    /**
     * Allreduce a real-space slab integral scalar within this pool.
     *
     * Uses the band communicator (within-pool) so that all slab contributions
     * from ranks in the same pool are combined. No-op when group_size == 1.
     */
    void reduce_within_pool_scalar(Real& value) const override {
        if (group_size_ <= 1)
            return;
        MPI_Allreduce(MPI_IN_PLACE, &value, 1, MPI_DOUBLE, MPI_SUM, get_group_communicator());
    }

    /**
     * Allreduce a real-space slab integral vector within this pool.
     *
     * Uses the band communicator for the same reason as reduce_within_pool_scalar.
     */
    void reduce_within_pool_vector(std::vector<Real>& values) const override {
        /**
         * group_size_ is identical on every rank of the group, so returning on it
         * is safe; the vector's local emptiness is not, and is no longer tested.
         */
        if (group_size_ <= 1)
            return;
        MPI_Allreduce(MPI_IN_PLACE,
                      values.data(),
                      static_cast<int>(values.size()),
                      MPI_DOUBLE,
                      MPI_SUM,
                      get_group_communicator());
    }

    /**
     * Distribute k-points round-robin across pools; return this pool's subset.
     */
    KPointsGrid local_kpoint_grid(const KPointsGrid& global_kpoint_grid) const override {
        parallel::KPointDistributor distributor(
            global_kpoint_grid, group_index(), number_of_pools_);
        return distributor.local_kpoint_grid();
    }

    std::vector<Real>
    assemble_global_kpoint_state_data(const std::vector<Real>& local_data,
                                      int local_number_of_physical_kpoints,
                                      int number_of_spin_components,
                                      int number_of_values_per_state) const override {
        if (local_number_of_physical_kpoints < 0 || number_of_spin_components < 1 ||
            number_of_values_per_state < 1) {
            throw std::invalid_argument("Global k-point assembly received invalid dimensions.");
        }
        const std::size_t expected_local_size =
            static_cast<std::size_t>(local_number_of_physical_kpoints) * number_of_spin_components *
            number_of_values_per_state;
        if (local_data.size() != expected_local_size) {
            throw std::invalid_argument(
                "Global k-point assembly data size does not match its dimensions.");
        }

        int root_local_number_of_physical_kpoints =
            (group_rank_index_ == 0) ? local_number_of_physical_kpoints : 0;
        int global_number_of_physical_kpoints = 0;
        MPI_Allreduce(&root_local_number_of_physical_kpoints,
                      &global_number_of_physical_kpoints,
                      1,
                      MPI_INT,
                      MPI_SUM,
                      MPI_COMM_WORLD);

        std::vector<Real> global_data(static_cast<std::size_t>(global_number_of_physical_kpoints) *
                                          number_of_spin_components * number_of_values_per_state,
                                      0.0);
        if (group_rank_index_ == 0) {
            for (int spin_index = 0; spin_index < number_of_spin_components; ++spin_index) {
                for (int local_kpoint_index = 0;
                     local_kpoint_index < local_number_of_physical_kpoints;
                     ++local_kpoint_index) {
                    int local_state_index =
                        spin_index * local_number_of_physical_kpoints + local_kpoint_index;
                    int global_state_index = spin_index * global_number_of_physical_kpoints +
                                             global_kpoint_index(local_kpoint_index);
                    std::copy_n(local_data.begin() + static_cast<std::size_t>(local_state_index) *
                                                         number_of_values_per_state,
                                number_of_values_per_state,
                                global_data.begin() + static_cast<std::size_t>(global_state_index) *
                                                          number_of_values_per_state);
                }
            }
        }
        if (!global_data.empty()) {
            MPI_Allreduce(MPI_IN_PLACE,
                          global_data.data(),
                          static_cast<int>(global_data.size()),
                          MPI_DOUBLE,
                          MPI_SUM,
                          MPI_COMM_WORLD);
        }
        return global_data;
    }

    std::unique_ptr<FFTGrid> create_fft_grid() const override {
        /**
         * When running with a single rank per band group (np=1), use the serial
         * FFT grid which supports FFTW threading (OMP_NUM_THREADS). This avoids
         * MPI FFT overhead and enables hybrid thread parallelism for FFTs.
         */
        if (group_size() <= 1) {
            return std::make_unique<SerialFFTGrid>();
        }
        /**
         * Ranks sharing a pool distribute reciprocal space by stick. The FFTW-MPI
         * slab grid it replaces split a plane from its mirror, which is what
         * forced gamma-only off under MPI, and divided unevenly enough to leave
         * ranks empty at eight. The stick grid needs only serial FFTW plus MPI.
         */
        return std::make_unique<FFTGridSticks>(get_group_communicator());
    }

    // ── Pool topology accessors ───────────────────────────────────────────────

    int pool_index() const {
        return group_index();
    }
    bool number_of_pools_was_requested() const override {
        return number_of_pools_was_requested_;
    }

    int number_of_pools() const override {
        return number_of_pools_;
    }
    bool one_process_owns_complete_state() const override {
        return world_size() == 1;
    }

    /**
     * One rank per pool. Decided from the topology, not from group_size():
     * a pool whose coefficient distribution was switched off reports a group
     * of one while its ranks still divide the FFT grid.
     */
    bool each_process_owns_whole_kpoints() const override {
        return world_size() == number_of_pools_;
    }

    /**
     * Pool p holds global k-points p, p + P, p + 2P, ... -- the round-robin
     * KPointDistributor deals out in local_kpoint_grid().
     */
    int global_kpoint_index(int local_kpoint_index) const override {
        return group_index() + local_kpoint_index * number_of_pools_;
    }

  private:
    int number_of_pools_ = 1;
    bool number_of_pools_was_requested_ = false;
    MPI_Comm inter_pool_communicator_ = MPI_COMM_NULL;
};

} // namespace execution
} // namespace q3
