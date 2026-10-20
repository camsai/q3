#pragma once

/**
 * SubspaceSelection — how the N states the gauge is built from are chosen
 * out of the band window.
 *
 * Two answers exist, and they are two classes rather than a mode the builder
 * branches on. IsolatedManifoldSelection takes the
 * window as it stands: the bands ARE the states, and the choice is the
 * projection gauge on them. ProjectabilityDisentanglementSelection
 * classifies the bands and iterates a smooth subspace out of a wider window,
 * which is what an entangled manifold needs. Both hand back the same thing
 * -- an orthonormal frame per k-point, in the projection gauge -- so the
 * gauge choice that follows does not know which one ran.
 *
 * Mirrors OverlapOperator: an interface, one class per physical context, and
 * a factory (factory.hpp) that dispatches on the settings.
 */

#include <cstddef>
#include <vector>

#include "layer_0_core/math/linear_algebra/dense_matrix.hpp"
#include "layer_0_core/types/init.hpp"
#include "layer_4_computation/interpolator/method/wannier/construction_inputs.hpp"
#include "layer_4_computation/interpolator/method/wannier/internal/band_classification.hpp"
#include "layer_4_computation/interpolator/method/wannier/report/init.hpp"
#include "layer_4_computation/interpolator/method/wannier/subspace/wannier_subspace.hpp"

namespace q3 {

class SubspaceSelection {
  public:
    virtual ~SubspaceSelection() = default;

    [[nodiscard]] virtual WannierSubspace choose(const WannierConstructionInputs& inputs) const = 0;
};

} // namespace q3
