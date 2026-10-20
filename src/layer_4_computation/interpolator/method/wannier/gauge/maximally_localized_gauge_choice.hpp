#pragma once

/**
 * MaximallyLocalizedGaugeChoice — the Marzari-Vanderbilt spread minimised
 * over the gauge after the subspace is fixed (SpreadMinimizer), the frames
 * rotated by the unitaries it found.
 *
 * Two things a record can lack make this impossible rather than merely
 * poor, and both are refused before anything is iterated. The spread is
 * defined through the neighbour overlaps M_mn(k,b), and a record without
 * them has nothing to minimise. And the centres are finite-difference Berry
 * phases, which need k+b and k-b to be distinct: on a mesh with a two-point
 * axis they coincide, and a minimiser run there reports a spread bounded by
 * the mesh rather than by the physics -- a number that looks like an answer.
 * require_supported() states both, so the factory can ask before the
 * subspace is chosen and the class can insist when it is called directly.
 */

#include <array>
#include <cstddef>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "layer_0_core/logging/init.hpp"
#include "layer_0_core/math/linear_algebra/dense_matrix.hpp"
#include "layer_0_core/math/linear_algebra/dense_matrix_product.hpp"
#include "layer_0_core/math/linear_algebra/matrix_operation.hpp"
#include "layer_0_core/types/init.hpp"
#include "layer_4_computation/interpolator/method/wannier/construction_inputs.hpp"
#include "layer_4_computation/interpolator/method/wannier/diagnostic.hpp"
#include "layer_4_computation/interpolator/method/wannier/diagnostic_error.hpp"
#include "layer_4_computation/interpolator/method/wannier/gauge/init.hpp"
#include "layer_4_computation/interpolator/method/wannier/internal/bloch_overlaps.hpp"
#include "layer_4_computation/interpolator/method/wannier/internal/phase_branch/guided_phase_branch.hpp"
#include "layer_4_computation/interpolator/method/wannier/internal/spread_functional.hpp"
#include "layer_4_computation/interpolator/method/wannier/internal/spread_minimizer.hpp"
#include "layer_4_computation/interpolator/method/wannier/subspace/init.hpp"

namespace q3 {

class MaximallyLocalizedGaugeChoice : public GaugeChoice {
  public:
    explicit MaximallyLocalizedGaugeChoice(SpreadMinimizationSettings settings)
        : settings_(settings) {}

    /// Throws WannierDiagnosticError, with the way out named, when the record cannot be localised.
    static void require_supported(const WannierConstructionInputs& inputs) {
        if (!inputs.has_overlaps()) {
            throw WannierDiagnosticError(
                WannierDiagnostic::localization_unsupported,
                "MaximallyLocalizedGaugeChoice: localization needs the neighbour overlaps "
                "M_mn(k,b), and this record carries none. The spread is defined through them; "
                "without them there is nothing to minimise. Recapture the record with Q3, or ask "
                "for control.interpolation.localization: none and accept the projection gauge.");
        }
        const std::optional<int> two_point_direction = inputs.two_point_axis_direction();
        if (two_point_direction) {
            const std::array<int, 3>& grid = inputs.record().kpoint_grid_dimensions;
            const IntegerVector3D& step = inputs.stencil().grid_step(*two_point_direction);
            throw WannierDiagnosticError(
                WannierDiagnostic::two_point_axis,
                "MaximallyLocalizedGaugeChoice: the k-mesh is " + std::to_string(grid[0]) + "x" +
                    std::to_string(grid[1]) + "x" + std::to_string(grid[2]) +
                    ", and the neighbour direction (" + std::to_string(step.x) + ", " +
                    std::to_string(step.y) + ", " + std::to_string(step.z) +
                    ") reaches the same k-point as its opposite. The Wannier centres are "
                    "finite-difference Berry phases, which need k+b and k-b to be distinct, so "
                    "localization cannot produce meaningful centres or spreads on this mesh. Use "
                    "at least three k-points along every periodic direction, or set "
                    "control.interpolation.localization: none and accept the projection gauge.");
        }
    }

    [[nodiscard]] WannierGauge choose(const WannierConstructionInputs& inputs,
                                      const WannierSubspace& subspace) const override {
        require_supported(inputs);
        const WannierInputs& record = inputs.record();
        const bool has_trial_centers = record.has_trial_orbital_centers();
        if (!has_trial_centers) {
            log::info("localization: the record carries no trial-orbital centres (older format "
                      "or imported), so the projection gauge's own centres guide the phase "
                      "branches; recapture with Q3 to guide by the trial orbitals instead");
        }
        const BlochOverlaps overlaps =
            inputs.overlaps_in_gauge(subspace.classification, subspace.frames);
        /**
         * The guiding centres are taken once and then held for the whole run.
         *
         * The trial-orbital positions when the record carries them, which is
         * Wannier90's choice and the right one; the projection gauge's own
         * centres only as a fallback. The difference is not academic. On a
         * rattled four-atom Si3C cell the fallback started at Omega = 186 A^2
         * and stalled at 170, with centres 5 A from any atom, while Wannier90
         * on the same matrices started at 36.7 and finished at 25.9: with the
         * phase branches inconsistent across k, the Berry-phase centres of the
         * starting gauge are already garbage, and guiding to garbage keeps it.
         */
        const SpreadFunctional functional(
            inputs.stencil(),
            inputs.number_of_wannier_functions(),
            std::make_shared<const GuidedPhaseBranch>(
                has_trial_centers
                    ? record.trial_orbital_centers_bohr
                    : SpreadFunctional(inputs.stencil(), inputs.number_of_wannier_functions())
                          .evaluate(overlaps)
                          .centers_bohr));
        const SpreadMinimizationResult localized =
            SpreadMinimizer(inputs.stencil(), settings_).run(overlaps, functional);

        WannierGauge gauge;
        gauge.frames.reserve(subspace.frames.size());
        for (std::size_t kpoint_index = 0; kpoint_index < subspace.frames.size(); kpoint_index++) {
            gauge.frames.push_back(multiply(MatrixOperation::None,
                                            subspace.frames[kpoint_index],
                                            MatrixOperation::None,
                                            localized.gauges[kpoint_index]));
        }
        gauge.spread = SpreadReport::evaluated_as(localized.terms,
                                                  has_trial_centers,
                                                  record.trial_orbital_centers_bohr,
                                                  localized.initial_minimizable_bohr_squared);
        SpreadReport& spread = gauge.spread;
        spread.localized = true;
        spread.localization_iterations = localized.iterations;
        spread.localization_converged = localized.stationary;
        spread.localization_gradient_norm = localized.gradient_norm;
        return gauge;
    }

  private:
    SpreadMinimizationSettings settings_;
};

} // namespace q3
