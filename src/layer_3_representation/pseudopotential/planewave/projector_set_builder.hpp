#pragma once

/**
 * ProjectorSetBuilder — how a KleinmanBylanderProjectorSet gets its projectors.
 *
 * The one way a projector set is built. An implementation is bound to a
 * crystal, a plane-wave basis and a set of pseudopotentials when it is made,
 * and fills a set for any k-point of that system. SerialProjectorSetBuilder
 * computes on the host; a builder that computes on a device is a second
 * implementation of this interface, not a second code path through the
 * consumers.
 */

#include <vector>

#include "layer_0_core/types/init.hpp"
#include "layer_3_representation/pseudopotential/planewave/kleinman_bylander_projector_set.hpp"

namespace q3 {

/// Interface: fills a KleinmanBylanderProjectorSet for one k-point; see the file comment.
class ProjectorSetBuilder {
  public:
    virtual ~ProjectorSetBuilder() = default;

    /**
     * Build <k+G|β> for every projector of every atom at one k-point, replacing whatever the
     * set held.
     *
     * @param kpoint_crystal_coordinates  The k-point, in crystal coordinates.
     * @param g_vector_indices            The k-point's plane waves, as indices into the basis's
     *                                    G-vectors.
     * @param number_of_plane_waves       How many of them the set covers.
     * @param projector_set               The set to build.
     */
    virtual void build(const Vector3D& kpoint_crystal_coordinates,
                       const std::vector<int>& g_vector_indices,
                       int number_of_plane_waves,
                       KleinmanBylanderProjectorSet& projector_set) const = 0;
};

} // namespace q3
