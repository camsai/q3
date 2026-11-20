#pragma once

/**
 * ProjectorSetProvider — where an eigensolver component gets a k-point's
 * Kleinman-Bylander projector set.
 *
 * From the store it was handed, which is the SCF iterator's and the normal
 * case: the set is then the one every other consumer reads, built once. A
 * component driven without an iterator, as the unit tests drive the solvers,
 * was handed no store; the provider then keeps one set of its own and builds
 * it for the k-point and the plane waves it is asked about, on every request,
 * since nothing tells it whether those changed.
 *
 * Either way the projectors come out of a ProjectorSetBuilder, so the solvers
 * hold no projector computation of their own.
 */

#include <vector>

#include "layer_1_structure/crystal/crystal.hpp"
#include "layer_1_structure/reciprocal_space/kpoints_grid.hpp"
#include "layer_3_representation/basis/planewave/init.hpp"
#include "layer_3_representation/basis/planewave/pseudopotential/init.hpp"
#include "layer_3_representation/pseudopotential/planewave/kleinman_bylander_projector_set.hpp"
#include "layer_3_representation/pseudopotential/planewave/kleinman_bylander_projector_set_store.hpp"
#include "layer_3_representation/pseudopotential/planewave/projector_set_builder_serial.hpp"

namespace q3 {

/// Hands a solver component the projector set of a k-point; see the file comment.
class ProjectorSetProvider {
  public:
    /// Read the sets from this store from now on; null returns to the provider's own set.
    void set_store(KleinmanBylanderProjectorSetStore* projector_set_store) {
        projector_set_store_ = projector_set_store;
    }

    /// The store the sets come from, or null when the provider builds its own.
    [[nodiscard]] KleinmanBylanderProjectorSetStore* store() const {
        return projector_set_store_;
    }

    /**
     * The projector set of a k-point. With a store, the store's set, built if need be. Without
     * one, the provider's own set, rebuilt here for the plane waves given.
     */
    [[nodiscard]] const KleinmanBylanderProjectorSet&
    projector_set(int kpoint_index,
                  const Crystal& crystal_structure,
                  const KPointsGrid& kpoint_grid,
                  const PlanewaveBasis& planewave_basis,
                  const NormConservingPlanewaveBasis& pseudopotential_basis,
                  const std::vector<int>& g_vector_indices,
                  int number_of_plane_waves) {
        if (projector_set_store_ != nullptr) {
            return projector_set_store_->at(kpoint_index);
        }
        SerialProjectorSetBuilder builder(
            crystal_structure, planewave_basis, pseudopotential_basis);
        builder.build(kpoint_grid[kpoint_index].coordinates,
                      g_vector_indices,
                      number_of_plane_waves,
                      own_projector_set_);
        return own_projector_set_;
    }

  private:
    KleinmanBylanderProjectorSetStore* projector_set_store_ = nullptr;
    KleinmanBylanderProjectorSet own_projector_set_;
};

} // namespace q3
