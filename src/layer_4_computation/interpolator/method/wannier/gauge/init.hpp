#pragma once

/**
 * GaugeChoice — how the gauge inside the chosen subspace is fixed.
 *
 * The subspace selection hands over frames in the projection gauge, exact at
 * the mesh points whatever else it does. A gauge choice is a unitary
 * rotation of each frame: it changes nothing at the mesh points and
 * everything between them, which is where an interpolator is actually used.
 * ProjectionGaugeChoice keeps the frames as they are and evaluates their
 * spread where it is defined; MaximallyLocalizedGaugeChoice minimises the
 * Marzari-Vanderbilt spread over the rotations. Both fill the same report
 * section, because the projection gauge has a spread too and a driver
 * comparing two models wants it either way.
 *
 * Mirrors OverlapOperator: an interface, one class per physical context, and
 * a factory (factory.hpp) that dispatches on the policy and the record.
 */

#include <vector>

#include "layer_0_core/math/linear_algebra/dense_matrix.hpp"
#include "layer_0_core/types/init.hpp"
#include "layer_4_computation/interpolator/method/wannier/construction_inputs.hpp"
#include "layer_4_computation/interpolator/method/wannier/gauge/wannier_gauge.hpp"
#include "layer_4_computation/interpolator/method/wannier/internal/bloch_overlaps.hpp"
#include "layer_4_computation/interpolator/method/wannier/report/init.hpp"
#include "layer_4_computation/interpolator/method/wannier/subspace/init.hpp"

namespace q3 {

class GaugeChoice {
  public:
    virtual ~GaugeChoice() = default;

    [[nodiscard]] virtual WannierGauge choose(const WannierConstructionInputs& inputs,
                                              const WannierSubspace& subspace) const = 0;
};

} // namespace q3
