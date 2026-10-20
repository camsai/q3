#pragma once

/**
 * QuantumSystemExecutionPolicy — introduces band semantics on top of the
 * representation-agnostic ExecutionPolicy.
 *
 * Layer 0 provides collective operations over an anonymous process group; it
 * deliberately knows nothing about bands, k-points, or any physical basis.
 * This interface fixes the meaning of that group at the quantum-system level:
 * a *band group* is the set of processes that diagonalize the same quantum
 * states together, sharing the work of one diagonalization. The layer-0
 * process group IS the band group — dot_product, norm, reduce_within_group,
 * and the group broadcast/allgather primitives all operate over it.
 *
 * Representation-specific topology (e.g. k-point pools and G-vector slab
 * distribution in the planewave representation) is layered on further down,
 * in layer_4_computation/iterator/kohn_sham/planewave/execution/init.hpp.
 *
 * No MPI headers — the band accessors are thin, band-named views of the
 * layer-0 group topology.
 */

#include "layer_0_core/execution/init.hpp"

namespace q3 {
namespace execution {

class QuantumSystemExecutionPolicy : public virtual ExecutionPolicy {
  public:
    ~QuantumSystemExecutionPolicy() override = default;

    /**
     * Number of processes collaborating within this band group.
     *
     * 1 means each process diagonalizes its quantum states alone (no band
     * parallelism); >1 means the diagonalization work of each quantum state
     * is shared across the group.
     */
    int band_size() const {
        return group_size();
    }

    /**
     * Index of this process within its band group (0 = band root).
     */
    int band_rank_index() const {
        return group_rank_index();
    }
};

} // namespace execution
} // namespace q3
