#pragma once

/**
 * ProjectionGaugeChoice — the projection gauge as it stands. Exact at the
 * mesh points and only as good as the trial orbitals between them; the
 * choice for a record that cannot support localisation, and for a caller
 * that refused it.
 *
 * Its spread is still evaluated where it is defined: per-function spreads
 * at the gauge the model actually uses are what a driver reads to judge it.
 * The guiding centres only fix the branch of each phase; with the trial
 * orbitals to guide by the centres come out beside them, and without, on
 * the principal branch.
 */

#include "layer_4_computation/interpolator/method/wannier/construction_inputs.hpp"
#include "layer_4_computation/interpolator/method/wannier/gauge/init.hpp"
#include "layer_4_computation/interpolator/method/wannier/internal/phase_branch/guided_phase_branch.hpp"
#include "layer_4_computation/interpolator/method/wannier/internal/phase_branch/init.hpp"
#include "layer_4_computation/interpolator/method/wannier/internal/phase_branch/principal_phase_branch.hpp"
#include "layer_4_computation/interpolator/method/wannier/internal/spread_functional.hpp"
#include "layer_4_computation/interpolator/method/wannier/subspace/init.hpp"

namespace q3 {

class ProjectionGaugeChoice : public GaugeChoice {
  public:
    [[nodiscard]] WannierGauge choose(const WannierConstructionInputs& inputs,
                                      const WannierSubspace& subspace) const override {
        WannierGauge gauge;
        gauge.frames = subspace.frames;
        if (!inputs.spread_is_defined()) {
            /**
             * Omega_I is the one spread term a disentangled model has even on
             * a mesh where the centres are undefined; the subspace iteration
             * measured it, so it is reported.
             */
            gauge.spread.gauge_invariant_bohr_squared =
                subspace.report.gauge_invariant_spread_bohr_squared;
            return gauge;
        }
        const WannierInputs& record = inputs.record();
        const bool guided = record.has_trial_orbital_centers();
        const std::shared_ptr<const PhaseBranch> branch =
            guided ? std::shared_ptr<const PhaseBranch>(std::make_shared<const GuidedPhaseBranch>(
                         record.trial_orbital_centers_bohr))
                   : std::make_shared<const PrincipalPhaseBranch>();
        const SpreadTerms terms =
            SpreadFunctional(inputs.stencil(), inputs.number_of_wannier_functions(), branch)
                .evaluate(inputs.overlaps_in_gauge(subspace.classification, subspace.frames));
        gauge.spread = SpreadReport::evaluated_as(
            terms, guided, record.trial_orbital_centers_bohr, terms.minimizable_bohr_squared());
        return gauge;
    }
};

} // namespace q3
