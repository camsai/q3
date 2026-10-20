#pragma once

/**
 * WannierConstructionReport — what the construction produces beside the
 * Hamiltonian: the numbers a driver needs to decide whether to keep the
 * model.
 *
 * One section per strategy. The subspace selection fills `disentanglement`,
 * the gauge choice fills `spread`, and the builder measures the decay of the
 * H(R) it assembled. A section a strategy did not run says so in its own
 * flag rather than by a sentinel value, so a reader never mistakes a zero
 * for a measurement.
 */

#include <string>

#include "layer_0_core/types/init.hpp"
#include "layer_3_representation/local_orbital/hamiltonian_decay.hpp"
#include "layer_4_computation/interpolator/method/wannier/report/disentanglement_report.hpp"
#include "layer_4_computation/interpolator/method/wannier/report/spread_report.hpp"

namespace q3 {

struct [[nodiscard]] WannierConstructionReport {
    DisentanglementReport disentanglement;
    SpreadReport spread;
    /// How far H(R) has decayed by the edge of its Wigner-Seitz domain.
    HamiltonianDecay hamiltonian_decay;

    /**
     * "wannier", then what was done to it, so the model says how it was
     * built -- including when the localisation did not finish. A gauge that
     * stalled is not maximally localised, and a description that says it is
     * turns a warning in the log into a claim in the results file.
     */
    [[nodiscard]] std::string description() const {
        std::string text = "wannier (";
        text += disentanglement.performed ? "projectability disentanglement" : "projection gauge";
        if (spread.localized) {
            text +=
                spread.localization_converged ? ", maximally localized" : ", localization stalled";
        }
        text += ")";
        return text;
    }
};

} // namespace q3
