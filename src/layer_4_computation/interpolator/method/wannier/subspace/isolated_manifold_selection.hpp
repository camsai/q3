#pragma once

/**
 * IsolatedManifoldSelection — the band window as it stands: the bands ARE
 * the N states, and the frame at each k-point is their projection gauge
 * U(k) = A (A^dagger A)^(-1/2).
 *
 * That gauge is defined only when there are exactly as many bands as Wannier
 * functions. With more bands, U^dagger E U is a Rayleigh quotient in a
 * subspace chosen independently at each k, meaningful only after
 * disentanglement has made that choice smooth across the mesh. Such inputs
 * are refused rather than interpolated, with the two ways out named.
 *
 * The window is described as a classification that keeps every band and
 * freezes none. That is not a formality: it lets the gauge choice and the
 * Fourier step run as one code path for both selections, and a second path
 * is where the two cases drift apart.
 */

#include <stdexcept>
#include <string>

#include "layer_0_core/math/linear_algebra/dense_matrix.hpp"
#include "layer_0_core/types/init.hpp"
#include "layer_4_computation/interpolator/method/wannier/construction_inputs.hpp"
#include "layer_4_computation/interpolator/method/wannier/internal/band_classifier.hpp"
#include "layer_4_computation/interpolator/method/wannier/internal/projection_gauge.hpp"
#include "layer_4_computation/interpolator/method/wannier/subspace/init.hpp"

namespace q3 {

class IsolatedManifoldSelection : public SubspaceSelection {
  public:
    [[nodiscard]] WannierSubspace choose(const WannierConstructionInputs& inputs) const override {
        const WannierInputs& record = inputs.record();
        if (record.number_of_bands != record.number_of_trial_orbitals) {
            throw std::invalid_argument(
                "IsolatedManifoldSelection: the inputs hold " +
                std::to_string(record.number_of_bands) + " bands and " +
                std::to_string(record.number_of_trial_orbitals) +
                " Wannier functions. The projection gauge on its own is defined only for an "
                "isolated manifold with as many bands as Wannier functions. Either recompute with "
                "number_of_bands = " +
                std::to_string(record.number_of_trial_orbitals) + ", choose " +
                std::to_string(record.number_of_bands) +
                " trial orbitals, or set control.interpolation.disentanglement: projectability "
                "to choose the subspace from this wider window.");
        }
        WannierSubspace subspace;
        subspace.classification =
            BandClassification::every_band_kept(inputs.number_of_kpoints(), record.number_of_bands);
        subspace.frames = ProjectionGauge().compute_all(
            inputs.projections_restricted_to(subspace.classification));
        return subspace;
    }
};

} // namespace q3
