#pragma once

/**
 * WannierDiagnosticError — a failure of the Wannier construction that a
 * driver can act on: the prose a person reads, the WannierDiagnostic a driver
 * routes on, and -- for a disentanglement out of iterations -- the spreads it
 * reached, as numbers.
 *
 * A DiagnosticError, so the utility prints the code -- and the spreads, when
 * there are any -- on lines of their own after the prose without knowing
 * that it was this method that failed; the code and the spreads as values
 * are there for the catch that knows to ask.
 */

#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "layer_0_core/diagnostic_error.hpp"
#include "layer_0_core/math/convergence_window.hpp"
#include "layer_0_core/types/type_aliases.hpp"
#include "layer_4_computation/interpolator/method/wannier/diagnostic.hpp"

namespace q3 {

class WannierDiagnosticError : public DiagnosticError {
  public:
    WannierDiagnosticError(WannierDiagnostic diagnostic,
                           const std::string& message,
                           std::vector<Real> spreads_bohr_squared = {})
        : DiagnosticError(message)
        , diagnostic_(diagnostic)
        , spreads_bohr_squared_(std::move(spreads_bohr_squared)) {}

    [[nodiscard]] WannierDiagnostic diagnostic() const {
        return diagnostic_;
    }

    /**
     * For disentanglement_unconverged: the last gauge-invariant spreads the
     * iteration reached, oldest first, in bohr squared -- the numbers the
     * retry ladder loosens its tolerance from. Empty for every other code.
     */
    [[nodiscard]] const std::vector<Real>& spreads_bohr_squared() const {
        return spreads_bohr_squared_;
    }

    /// The spreads as "a, b, c" with six decimals each, the form the prose names them in.
    [[nodiscard]] std::string spreads_text() const {
        return ConvergenceWindow::history_text(spreads_bohr_squared_);
    }

    /// `diagnostic: <code>`, and the spreads on a line of their own when there are any.
    [[nodiscard]] std::vector<std::string> diagnostic_lines() const override {
        std::vector<std::string> lines{"diagnostic: " + wannier_diagnostic_name(diagnostic_)};
        if (!spreads_bohr_squared_.empty()) {
            lines.push_back("diagnostic_spreads_bohr_squared: " + spreads_text());
        }
        return lines;
    }

  private:
    WannierDiagnostic diagnostic_;
    std::vector<Real> spreads_bohr_squared_;
};

} // namespace q3
