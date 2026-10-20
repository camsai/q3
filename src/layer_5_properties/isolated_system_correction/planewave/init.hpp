#pragma once

/**
 * IsolatedSystemCorrectionCalculator — abstract interface for isolated-system
 * (finite-size electrostatic) correction schemes.
 *
 * A charged or dipolar system computed in periodic boundary conditions
 * carries a spurious electrostatic interaction with its own periodic images;
 * a correction scheme estimates and removes it from the total energy after
 * SCF convergence. Concrete schemes (Makov-Payne and Martyna-Tuckerman
 * today; point-countercharge and slab/ESM-style corrections are the natural
 * future additions, mirroring QE's assume_isolated options) implement this
 * interface and are selected by name through
 * create_isolated_system_correction_calculator() (factory.hpp), so
 * orchestrators never branch on the scheme themselves -- "none" resolves to
 * NullIsolatedSystemCorrectionCalculator (null.hpp), which reports "nothing
 * applied", removing the need for any if/else at the call sites.
 *
 * The result is deliberately scheme-agnostic: a list of labeled additive
 * energy terms rather than fixed fields, because the breakdown differs per
 * scheme (Makov-Payne has a monopole and a dipole+quadrupole term;
 * Martyna-Tuckerman reports a single reciprocal-space term, see
 * MartynaTuckermanCorrectionCalculator -- its docstring also explains why
 * this is the post-SCF evaluation of MT, distinct from QE's self-consistent
 * assume_isolated='m-t'). Every term already carries the sign such that
 *   corrected_total_energy = total_energy + sum(terms).
 */

#include <string>
#include <utility>
#include <vector>

#include "layer_0_core/types/init.hpp"
#include "layer_4_computation/state/kohn_sham/planewave/norm_conserving/init.hpp"

namespace q3 {

/**
 * Outcome of one correction scheme's evaluation.
 *
 * `applied` is false only for the Null scheme ("none"); reporting code keys
 * off it to decide whether to emit anything. `terms_in_ry` are the labeled
 * additive contributions (labels are snake_case and become the
 * `<label>_ry` keys in results.json).
 */
struct IsolatedSystemCorrectionResult {
    bool applied = false;
    std::string method = "none";
    std::vector<std::pair<std::string, Real>> terms_in_ry;

    [[nodiscard]] Real total_in_ry() const {
        Real total = 0.0;
        for (const auto& [label, value] : terms_in_ry) {
            total += value;
        }
        return total;
    }
};

class IsolatedSystemCorrectionCalculator {
  public:
    virtual ~IsolatedSystemCorrectionCalculator() = default;

    /**
     * Evaluate the correction on a converged SCF state. Schemes that have
     * geometric preconditions (Makov-Payne: cubic lattice) throw
     * std::runtime_error before touching any density data when violated.
     */
    [[nodiscard]] virtual IsolatedSystemCorrectionResult
    compute(const NormConservingPlanewaveKohnShamSCFState& scf_state) const = 0;

    /// Scheme name as accepted by the input ("none", "makov-payne", ...).
    [[nodiscard]] virtual std::string name() const = 0;
};

} // namespace q3
