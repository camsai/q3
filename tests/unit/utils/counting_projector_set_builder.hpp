#pragma once

#include <vector>

#include "layer_3_representation/pseudopotential/planewave/kleinman_bylander_projector_set.hpp"
#include "layer_3_representation/pseudopotential/planewave/projector_set_builder.hpp"
#include "layer_3_representation/pseudopotential/planewave/projector_set_builder_serial.hpp"

namespace q3::test_support {

/// A host builder that counts its builds in a counter the test owns.
class CountingProjectorSetBuilder : public ProjectorSetBuilder {
  public:
    CountingProjectorSetBuilder(const Crystal& crystal_structure,
                                const PlanewaveBasis& planewave_basis,
                                const NormConservingPlanewaveBasis& pseudopotential_basis,
                                int& number_of_builds)
        : host_builder_(crystal_structure, planewave_basis, pseudopotential_basis)
        , number_of_builds_(number_of_builds) {}

    void build(const Vector3D& kpoint_crystal_coordinates,
               const std::vector<int>& g_vector_indices,
               int number_of_plane_waves,
               KleinmanBylanderProjectorSet& projector_set) const override {
        ++number_of_builds_;
        host_builder_.build(
            kpoint_crystal_coordinates, g_vector_indices, number_of_plane_waves, projector_set);
    }

  private:
    SerialProjectorSetBuilder host_builder_;
    int& number_of_builds_;
};

} // namespace q3::test_support
