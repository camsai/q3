#pragma once

/**
 * PlanewaveExecutionPolicy — abstract interface for planewave-layer collective operations.
 *
 * Extends QuantumSystemExecutionPolicy (band groups over layer-0 process
 * groups: dot_product, add_scaled_vector, norm, band_size) with k-point pool
 * collective operations specific to the planewave representation.
 *
 * Two categories of collective operations:
 *
 * 1. Across-pool reductions (inter-pool communicator)
 *    - reduce_scalar:            k-point energy sums (band energy, smearing entropy)
 *    - reduce_vector:            k-point force/stress sums (nonlocal, kinetic)
 *
 * 2. Within-pool reductions (band communicator)
 *    - reduce_within_pool_scalar: real-space slab integrals (XC energy, DCC, delta-eSCF)
 *    - reduce_within_pool_vector: real-space slab force/stress (local, XC, core-correction)
 *
 * With 1-rank-per-pool (current default), the band communicator is trivial so
 * reduce_within_pool_* are no-ops. With multiple ranks per pool the FFTW-MPI slab
 * decomposition means each rank only owns a subset of grid points and both categories
 * are required for correctness.
 *
 * This interface has no knowledge of MPI or CUDA — implementations live in
 * planewave/execution/serial.hpp and planewave/execution/mpi.hpp.
 */

#include "generated/q3/results/runtime_log/execution.hpp"
#include "layer_0_core/execution/threading.hpp"
#include "layer_0_core/profiling/counters.hpp"
#include "layer_0_core/types/init.hpp"
#include "layer_1_structure/reciprocal_space/kpoints_grid.hpp"
#include "layer_2_quantum_system/execution/init.hpp"
#include "layer_3_representation/basis/planewave/internal/fft/fft_grid_factory.hpp"

#include <cstdint>
#include <memory>
#include <vector>

namespace q3 {
namespace execution {

class PlanewaveExecutionPolicy : public virtual QuantumSystemExecutionPolicy,
                                 public FFTGridFactory {
  public:
    ~PlanewaveExecutionPolicy() override = default;

    /**
     * Disable G-vector distribution, making band-group reductions into no-ops.
     *
     * Called when slab-based PW distribution yields zero plane waves for some
     * ranks. Default is a no-op (serial policy never distributes G-vectors).
     */
    virtual void disable_band_distribution() {}

    /**
     * Reconfigure the k-point pool topology after initial setup.
     *
     * Called by the factory when the requested pool count exceeds the number of
     * available IBZ k-points. The implementation clamps the pool count to
     * new_number_of_pools and reconfigures all MPI communicators accordingly.
     *
     * The default implementation is a no-op: the serial policy always has
     * exactly one pool so clamping is never needed there.
     *
     * @param new_number_of_pools  New (clamped) pool count.
     */
    virtual void reconfigure_topology(int /*new_number_of_pools*/) {}

    /**
     * Number of k-point pools in the current topology.
     *
     * Returns 1 in serial mode. In MPI mode returns the number of pools
     * configured by configure_topology(). Used by the factory to validate
     * that no pool is left empty (which would cause MPI collective mismatches).
     */
    virtual int number_of_pools() const = 0;

    /**
     * Whether the pool count came from the caller rather than a default.
     *
     * The factory needs the distinction: an explicit choice is honoured as
     * given, while an unrequested one is chosen from the k-point count. Serial
     * runs report true, since their single pool is not a fallback.
     */
    virtual bool number_of_pools_was_requested() const {
        return true;
    }

    /**
     * Whether one process owns every k-point and every G-vector coefficient.
     *
     * Writers without a gather implementation may serialize state only when
     * this is true. It is stronger than checking the current group size because
     * a small-basis fallback can disable coefficient distribution without
     * turning a multi-process FFT grid into a serial one.
     */
    virtual bool one_process_owns_complete_state() const = 0;

    /**
     * Whether every process holds the whole of each k-point it works on: all
     * of that k-point's plane-wave coefficients, and the complete density grid.
     *
     * True exactly when each k-point pool is a single process. Pools still
     * divide the k-points between them, so this is weaker than
     * one_process_owns_complete_state(), but it is enough for a writer whose
     * output splits by k-point: each process writes its own k-points under
     * their global_kpoint_index(), and what pools reduce -- the density, the
     * potential built from it -- is already whole on every process. Ranks
     * sharing a pool split both the coefficients and the grid, and nothing
     * gathers them yet.
     */
    virtual bool each_process_owns_whole_kpoints() const = 0;

    /**
     * The position in the global k-point grid of this pool's k-point
     * `local_kpoint_index`, as ordered in local_kpoint_grid().
     *
     * Identity in serial; under pools, the inverse of the round-robin
     * distribution.
     */
    virtual int global_kpoint_index(int local_kpoint_index) const = 0;

    virtual void reduce_charge_density(std::vector<std::vector<Real>>& charge_density) const = 0;

    /**
     * ── Across-pool reductions (inter-pool communicator) ──────────────────────
     *
     * Use for k-point-local quantities that must be summed across pools.
     * With 1-rank-per-pool these happen over MPI_COMM_WORLD; with multiple
     * ranks per pool they use the inter-pool communicator (one rank per pool).
     */

    /**
     * Reduce a scalar k-point sum across all pools (MPI_SUM).
     *
     * Used for band_energy and smearing_entropy after local k-point accumulation.
     *
     * @param value  In-out: local k-point contribution, replaced by global sum.
     */
    virtual void reduce_scalar(Real& value) const = 0;

    /**
     * Reduce a vector of k-point-local values across all pools (MPI_SUM).
     *
     * Used for nonlocal forces and kinetic/nonlocal stress contributions.
     *
     * @param values  In-out: local k-point contributions, replaced by global sum.
     */
    virtual void reduce_vector(std::vector<Real>& values) const = 0;

    /**
     * Reduce a scalar across all pools using MPI_MAX.
     *
     * Used for summary statistics such as max_plane_waves_per_kpoint where
     * each pool holds only its local k-points and the global maximum is needed.
     *
     * @param value  In-out: local maximum, replaced by global maximum.
     */
    virtual void reduce_scalar_max(int& value) const = 0;

    /**
     * Reduce a Real scalar across all pools using MPI_MAX.
     *
     * Used for global eigenvalue range (e.g., HOMO across pools).
     *
     * @param value  In-out: local maximum, replaced by global maximum.
     */
    virtual void reduce_scalar_max(Real& value) const = 0;

    /**
     * Reduce a Real scalar across all pools using MPI_MIN.
     *
     * Used for global eigenvalue range (e.g., LUMO across pools).
     *
     * @param value  In-out: local minimum, replaced by global minimum.
     */
    virtual void reduce_scalar_min(Real& value) const = 0;

    /**
     * Reduce an integer scalar within the band group using MPI_MIN.
     *
     * Used during initialization to check if any rank in the pool received
     * zero plane waves from slab distribution.
     *
     * @param value  In-out: local value, replaced by band-group minimum.
     */
    virtual void reduce_band_group_scalar_min(int& value) const = 0;

    /**
     * ── Within-pool reductions (band communicator) ────────────────────────────
     *
     * Use for real-space slab integrals. With 1-rank-per-pool these are no-ops.
     * With multiple ranks per pool, FFTW-MPI distributes the real-space grid
     * across the band communicator; each rank integrates only its local slabs
     * and these methods sum the partial results within the pool.
     */

    /**
     * Reduce a real-space slab integral scalar within this pool (MPI_SUM).
     *
     * Used for XC energy, double-counting correction, and delta-eSCF correction.
     *
     * @param value  In-out: partial slab integral, replaced by pool-total.
     */
    virtual void reduce_within_pool_scalar(Real& value) const = 0;

    /**
     * Reduce a real-space slab integral vector within this pool (MPI_SUM).
     *
     * Used for local forces, XC stress, local-PP stress, and core-correction
     * forces/stress computed from real-space grid slabs.
     *
     * @param values  In-out: partial slab contributions, replaced by pool-total.
     */
    virtual void reduce_within_pool_vector(std::vector<Real>& values) const = 0;

    /**
     * create_fft_grid() is inherited (pure) from FFTGridFactory — the layer-3
     * interface through which PlanewaveFFTPair obtains grids without an
     * upward dependency on this class.
     */

    /**
     * Compute the local k-point grid assigned to this process's pool.
     *
     * Serial: returns the global grid unchanged (identity).
     * MPI:    distributes k-points round-robin across pools.
     *
     * @param global_kpoint_grid  The full IBZ k-point grid.
     * @return                    The local subset owned by this pool.
     */
    virtual KPointsGrid local_kpoint_grid(const KPointsGrid& global_kpoint_grid) const = 0;

    /**
     * Assemble fixed-width data for local k-point states into global order.
     *
     * Local state order is spin-major, with physical k-points ordered as they
     * appear in local_kpoint_grid(). The returned data uses the same spin-major
     * convention over the complete global grid. Serial policies return the
     * input unchanged; MPI policies gather one contribution per pool.
     */
    virtual std::vector<Real>
    assemble_global_kpoint_state_data(const std::vector<Real>& local_data,
                                      int local_number_of_physical_kpoints,
                                      int number_of_spin_components,
                                      int number_of_values_per_state) const = 0;

    /**
     * What the run executed on -- processes, k-point pools, ranks per pool,
     * the threads each process may use, whether reciprocal space was split,
     * and the peak resident memory of the largest rank -- as results.json's
     * runtime_log.execution, so a results file says which policy produced it.
     *
     * Collective under MPI: every rank must call it, for the memory reduction.
     */
    [[nodiscard]] generated::ResultsExecution execution_record() const {
        generated::ResultsExecution record;
        record.number_of_processes = number_of_processes();
        record.number_of_threads_per_process = threading::maximum_threads();
        record.number_of_kpoint_pools = number_of_pools();
        record.band_group_size = record.number_of_processes / record.number_of_kpoint_pools;
        record.reciprocal_space_vectors_are_distributed = band_size() > 1;
        Real peak_resident_megabytes =
            static_cast<Real>(ProfilingCounters::instance().peak_resident_bytes_now()) / 1048576.0;
        reduce_scalar_max(peak_resident_megabytes);
        record.peak_resident_memory_bytes =
            static_cast<int64_t>(static_cast<std::uint64_t>(peak_resident_megabytes * 1048576.0));
        return record;
    }
};

} // namespace execution
} // namespace q3
