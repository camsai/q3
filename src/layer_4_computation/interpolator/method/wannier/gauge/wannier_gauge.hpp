#pragma once

/**
 * WannierGauge — what a GaugeChoice hands on: the frames after the choice --
 * C(k), kept x N with orthonormal columns -- and their spread, which is its
 * section of the construction report.
 */

#include <vector>

#include "layer_0_core/math/linear_algebra/dense_matrix.hpp"
#include "layer_0_core/types/init.hpp"
#include "layer_4_computation/interpolator/method/wannier/report/spread_report.hpp"

namespace q3 {

struct [[nodiscard]] WannierGauge {
    std::vector<DenseMatrix<Complex>> frames;
    SpreadReport spread;
};

} // namespace q3
