#pragma once

/**
 * WannierSubspace — the chosen subspace, as a SubspaceSelection hands it to
 * the gauge choice, with its section of the construction report.
 *
 * `frames[k]` is C(k), kept x N with orthonormal columns: the subspace at k
 * in the projection gauge, its rows in the order of
 * `classification.kpoints[k].kept_bands`. An isolated manifold keeps every
 * band and its frame is the projection gauge itself; a disentangled one
 * keeps a subset and its frame is V(k) U(k), the iteration's subspace times
 * the projection gauge inside it.
 */

#include <vector>

#include "layer_0_core/math/linear_algebra/dense_matrix.hpp"
#include "layer_0_core/types/init.hpp"
#include "layer_4_computation/interpolator/method/wannier/internal/band_classification.hpp"
#include "layer_4_computation/interpolator/method/wannier/report/disentanglement_report.hpp"

namespace q3 {

struct [[nodiscard]] WannierSubspace {
    BandClassification classification;
    std::vector<DenseMatrix<Complex>> frames;
    DisentanglementReport report;
};

} // namespace q3
