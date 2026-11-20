#pragma once

/**
 * SerialProjectorSetBuilder — builds Kleinman-Bylander projector sets on the host.
 *
 * Wraps compute_kleinman_bylander_projectors(): tabulated radial functions,
 * real spherical harmonics and structure factors, combined per atom on the
 * host. Holds references to the crystal, the plane-wave basis and the
 * pseudopotential basis it was made for, which must outlive it.
 */

#include <vector>

#include "layer_0_core/types/init.hpp"
#include "layer_1_structure/crystal/crystal.hpp"
#include "layer_3_representation/basis/planewave/init.hpp"
#include "layer_3_representation/basis/planewave/pseudopotential/init.hpp"
#include "layer_3_representation/pseudopotential/planewave/internal/projector.hpp"
#include "layer_3_representation/pseudopotential/planewave/kleinman_bylander_projector_set.hpp"
#include "layer_3_representation/pseudopotential/planewave/projector_set_builder.hpp"

namespace q3 {

/// ProjectorSetBuilder that computes the projectors on the host; see the file comment.
class SerialProjectorSetBuilder : public ProjectorSetBuilder {
  public:
    SerialProjectorSetBuilder(const Crystal& crystal_structure,
                              const PlanewaveBasis& planewave_basis,
                              const NormConservingPlanewaveBasis& pseudopotential_basis)
        : crystal_structure_(crystal_structure)
        , planewave_basis_(planewave_basis)
        , pseudopotential_basis_(pseudopotential_basis) {}

    /// See ProjectorSetBuilder::build(); one projector per atom, beta function and m.
    void build(const Vector3D& kpoint_crystal_coordinates,
               const std::vector<int>& g_vector_indices,
               int number_of_plane_waves,
               KleinmanBylanderProjectorSet& projector_set) const override {
        const int total_number_of_projectors = pseudopotential_basis_.total_number_of_projectors();
        Complex* projectors = projector_set.begin_build(kpoint_crystal_coordinates,
                                                        crystal_structure_,
                                                        total_number_of_projectors,
                                                        number_of_plane_waves);
        compute_kleinman_bylander_projectors(kpoint_crystal_coordinates,
                                             crystal_structure_,
                                             planewave_basis_,
                                             pseudopotential_basis_.pseudopotential_set(),
                                             pseudopotential_basis_.beta_function_table(),
                                             g_vector_indices,
                                             number_of_plane_waves,
                                             projectors);
    }

  private:
    const Crystal& crystal_structure_;
    const PlanewaveBasis& planewave_basis_;
    const NormConservingPlanewaveBasis& pseudopotential_basis_;
};

} // namespace q3
