#pragma once

/**
 * QuantumStateInterpolatorConstruction — what the evaluation-side factory
 * produces: the interpolator, and the method's report of how its model was
 * built.
 *
 * One report member per method, as InterpolationRecord carries one payload
 * per method; a method that was not the record's leaves its member at its
 * defaults. The report is returned rather than written through a pointer a
 * caller may or may not pass, so there is no "no report" mode to branch on:
 * a caller that wants the interpolator alone takes `interpolator` and drops
 * the rest.
 */

#include <memory>

#include "layer_4_computation/interpolator/method/wannier/report/init.hpp"
#include "layer_4_computation/interpolator/target/quantum_state/init.hpp"

namespace q3 {

struct [[nodiscard]] QuantumStateInterpolatorConstruction {
    std::unique_ptr<QuantumStateInterpolator> interpolator;
    WannierConstructionReport wannier;
};

} // namespace q3
