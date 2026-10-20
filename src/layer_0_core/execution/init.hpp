#pragma once

/**
 * ExecutionPolicy — abstract interface for process-group collective operations.
 *
 * Defines the contract consumed by DavidsonDiagonalizer for vector operations
 * on distributed coefficient vectors. Implementations provide serial (BLAS only)
 * or distributed (BLAS + MPI_Allreduce over the group communicator) variants.
 *
 * This interface is representation-agnostic: it operates on raw complex vectors
 * and has no knowledge of k-points, plane waves, bands, or any physical basis.
 * What a process group *means* (e.g. a band group of processes diagonalizing
 * the same k-point together) is assigned by higher layers — see
 * layer_2_quantum_system/execution/init.hpp.
 */

#include "layer_0_core/types/init.hpp"

#include <algorithm>
#include <vector>

#include <complex>

namespace q3 {
namespace execution {

/**
 * A contiguous, near-equal division of bands across the ranks sharing them.
 *
 * `counts` and `displacements` are in complex elements (bands multiplied by
 * elements_per_band) so they can be handed straight to
 * ExecutionPolicy::allgather_across_bands. The remainder is spread one unit per
 * rank over the lowest ranks rather than dumped on the last one, which keeps the
 * longest local block as short as possible -- the batch is only finished when the
 * slowest rank is.
 *
 * `band_granularity` is the number of bands that must stay together on one rank.
 * It exists for the gamma-only Hamiltonian, which packs two real wavefunctions
 * into the real and imaginary parts of a single complex transform and so walks
 * bands in pairs. Splitting 130 bands as 65 and 65 hands both ranks an odd count,
 * which breaks that pairing and makes each rank compute a different result from
 * the serial one. Passing 2 keeps every rank's block even, so the pairing -- and
 * the arithmetic -- match serial exactly.
 */
struct BandPartition {
    std::vector<int> counts;
    std::vector<int> displacements;
    int local_first_band = 0;
    int local_number_of_bands = 0;
};

inline BandPartition partition_bands(int number_of_bands,
                                     int number_of_ranks,
                                     int rank_index,
                                     int elements_per_band,
                                     int band_granularity = 1) {
    BandPartition partition;
    partition.counts.assign(number_of_ranks, 0);
    partition.displacements.assign(number_of_ranks, 0);
    int granularity = std::max(band_granularity, 1);
    /**
     * Partition whole units, then convert back to bands. The final unit may be
     * short when the band count is not a multiple of the granularity, so every
     * block is clamped to what is actually left.
     */
    int number_of_units = (number_of_bands + granularity - 1) / granularity;
    int base_units = number_of_units / number_of_ranks;
    int remainder_units = number_of_units % number_of_ranks;
    int first_band = 0;
    for (int rank = 0; rank < number_of_ranks; rank++) {
        int units_for_rank = base_units + (rank < remainder_units ? 1 : 0);
        int bands_for_rank =
            std::min(units_for_rank * granularity, std::max(number_of_bands - first_band, 0));
        partition.displacements[rank] = first_band * elements_per_band;
        partition.counts[rank] = bands_for_rank * elements_per_band;
        if (rank == rank_index) {
            partition.local_first_band = first_band;
            partition.local_number_of_bands = bands_for_rank;
        }
        first_band += bands_for_rank;
    }
    return partition;
}

class ExecutionPolicy {
  public:
    using ComplexValue = Complex;

    virtual ~ExecutionPolicy() = default;

    /**
     * Compute the inner product x† · y, optionally reduced across the process group.
     *
     * @param number_of_elements  Length of both vectors.
     * @param vector_x            Left operand (conjugated).
     * @param vector_y            Right operand.
     * @return                    The (globally summed) complex inner product.
     */
    virtual ComplexValue dot_product(int number_of_elements,
                                     const ComplexValue* vector_x,
                                     const ComplexValue* vector_y) const = 0;

    /**
     * Scaled vector addition in-place: vector_y += scaling_factor * vector_x.
     *
     * @param number_of_elements  Length of both vectors.
     * @param scaling_factor      Complex scaling coefficient.
     * @param vector_x            Source vector.
     * @param vector_y            Destination vector (modified in-place).
     */
    virtual void add_scaled_vector(int number_of_elements,
                                   ComplexValue scaling_factor,
                                   const ComplexValue* vector_x,
                                   ComplexValue* vector_y) const = 0;

    /**
     * Compute the Euclidean norm, optionally reduced across the process group.
     *
     * @param number_of_elements  Length of the vector.
     * @param vector_x            Input vector.
     * @return                    The real-valued (globally reduced) norm.
     */
    virtual Real norm(int number_of_elements, const ComplexValue* vector_x) const = 0;

    /**
     * Check if this process is the root process (rank 0 in the world communicator).
     *
     * Used by entry points to ensure only one process writes output files or
     * logs non-essential information.
     */
    virtual bool is_root() const = 0;

    /**
     * Index of this process within its process group.
     *
     * 0 in serial mode. NOT necessarily 0 when group_size() == 1: a group whose
     * data distribution was disabled at run time (see
     * MPIExecutionPolicy::disable_group_distribution) reports group_size() == 1
     * while keeping its pool-local rank index, because the pool identity is
     * still needed. Use group_size() <= 1 to ask "is my data replicated"; use
     * group_rank_index() == 0 only to pick one representative per pool.
     */
    virtual int group_rank_index() const = 0;

    /**
     * Number of processes in this process group.
     * Always 1 in serial mode or when each group contains a single process.
     */
    virtual int group_size() const = 0;

    /**
     * Number of processes in the complete execution topology.
     */
    virtual int number_of_processes() const = 0;

    /**
     * Sum a rank-local element count across the process group.
     *
     * Distributed vector algorithms use this to distinguish the local slab
     * length from the dimension of the complete vector space.
     */
    virtual int global_number_of_elements(int local_number_of_elements) const = 0;

    /**
     * Allreduce an array of complex values within the process group.
     *
     * Used after ZGEMM-based inner products over the distributed vector
     * dimension (e.g., subspace matrices, orthogonalization overlaps)
     * when coefficient vectors are distributed across group ranks.
     * No-op when group_size == 1.
     *
     * @param data            In-out array of complex values.
     * @param number_of_elements  Number of complex elements to reduce.
     */
    virtual void reduce_within_group(ComplexValue* data, int number_of_elements) const = 0;

    /**
     * Allreduce an array of real values within the process group.
     *
     * Same as the complex variant but for real-valued arrays.
     *
     * @param data            In-out array of real values.
     * @param number_of_elements  Number of real elements to reduce.
     */
    virtual void reduce_within_group(Real* data, int number_of_elements) const = 0;

    /**
     * Allgather rank-local slab data into a full-size buffer on every rank.
     *
     * Gathers each group rank's contiguous local slice into a contiguous
     * full array on every rank (concatenated in rank order).
     * No-op when group_size == 1.
     *
     * @param local_data        Rank-local slab data.
     * @param local_count       Number of real elements in the local slab.
     * @param full_data         Output: full-grid data (all slabs concatenated).
     * @param total_count       Total number of real elements across all ranks.
     */
    virtual void allgather_within_group(const Real* local_data,
                                        int local_count,
                                        Real* full_data,
                                        int total_count) const = 0;

    /**
     * Reduce group-distributed partial results across the process group.
     *
     * Used when different ranks compute disjoint contributions to the same
     * output array. Unlike reduce_within_group
     * (which is a no-op when the input vectors are replicated), this always performs
     * the reduction when group_size > 1. No-op in serial.
     *
     * @param data              In-out array of complex values.
     * @param number_of_elements  Number of complex elements to reduce.
     */
    virtual void reduce_group_distributed_result(ComplexValue* /*data*/,
                                                 int /*number_of_elements*/) const {}

    /**
     * ── Band distribution ─────────────────────────────────────────────────────
     *
     * A second, mutually exclusive way to divide work within a k-point pool.
     * Where group distribution splits the reciprocal-space coefficients of every
     * band across ranks, band distribution splits the *bands* and leaves each
     * rank holding complete coefficient vectors.
     *
     * The two cannot both be active: a rank either owns all of G for some bands
     * or some of G for all bands. When bands are distributed, group distribution
     * is switched off (group_size() reports 1), so every reduce_within_group()
     * correctly becomes a no-op -- the coefficients each rank contracts over are
     * whole, not partial. What replaces those reductions is an allgather over
     * the band axis, once per batch, rather than a reduction per inner product.
     *
     * Every method below is defaulted to the replicated, single-rank answer, so
     * a policy that does not distribute bands inherits correct behaviour.
     */

    /**
     * Number of ranks sharing the band split. 1 when bands are not distributed.
     */
    virtual int band_distribution_size() const {
        return 1;
    }

    /**
     * This rank's index within the band split. 0 when bands are not distributed.
     */
    virtual int band_distribution_rank_index() const {
        return 0;
    }

    /**
     * Complete an array whose local band block is already written in place.
     *
     * Each rank has filled its own contiguous block of `data`; afterwards every
     * rank holds all blocks. `counts` and `displacements` are in complex
     * elements and indexed by band-distribution rank, so a caller hands over the
     * same partition it used to divide the work.
     *
     * The in-place contract is deliberate. Gathering from a sub-range of the
     * destination into that same destination aliases the send and receive
     * buffers, which MPI forbids; requiring the block to be in place already is
     * what lets the implementation say so with MPI_IN_PLACE instead.
     * No-op when bands are not distributed -- the one block is the whole array.
     */
    virtual void allgather_across_bands(ComplexValue* /*data*/,
                                        const int* /*counts*/,
                                        const int* /*displacements*/) const {}

    /**
     * Sum an array across the band split, so that a quantity accumulated from
     * disjoint band ranges (the charge density above all) becomes whole.
     * No-op when bands are not distributed.
     */
    virtual void reduce_across_bands(Real* /*data*/, int /*number_of_elements*/) const {}

    /**
     * Republish an array from the first rank of the band split to the rest.
     *
     * This exists because "every rank recomputes the same thing from the same
     * data, so every rank agrees" is not true in practice. Vendor BLAS picks its
     * blocking and summation order from what the machine looks like at the time,
     * so two processes running identical arithmetic on identical inputs can
     * differ in the last bits -- measured here at a relative 1e-15 on the first
     * expansion. Davidson then amplifies that: orthogonalisation rejects
     * candidates on a threshold, so a last-bit difference changes how many
     * vectors a rank accepts, the ranks stop agreeing on the size of the next
     * batch, and the collective that follows fails outright.
     *
     * Broadcasting the redundantly computed result makes one rank's arithmetic
     * authoritative, which restores exactly the agreement the design assumed.
     * No-op when bands are not distributed.
     */
    virtual void broadcast_across_bands(ComplexValue* /*data*/, int /*number_of_elements*/) const {}

    /// Integer overload, for the counts that decide collective sizes.
    virtual void broadcast_across_bands(int* /*data*/, int /*number_of_elements*/) const {}

    /// Real overload, for eigenvalues -- the numbers convergence decisions are made from.
    virtual void broadcast_across_bands(Real* /*data*/, int /*number_of_elements*/) const {}

    /**
     * Broadcast complex data from the group root (rank 0) to all ranks in the group.
     *
     * Used to avoid replicated LAPACK computation: rank 0 computes, then
     * broadcasts the result. No-op when group_size == 1.
     *
     * @param data              In-out array of complex values.
     * @param number_of_elements  Number of complex elements to broadcast.
     */
    virtual void broadcast_from_group_root(ComplexValue* /*data*/,
                                           int /*number_of_elements*/) const {}

    /**
     * Broadcast real data from the group root (rank 0) to all ranks in the group.
     *
     * @param data              In-out array of real values.
     * @param number_of_elements  Number of real elements to broadcast.
     */
    virtual void broadcast_from_group_root(Real* /*data*/, int /*number_of_elements*/) const {}
};

} // namespace execution
} // namespace q3
