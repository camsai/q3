#pragma once

/**
 * KerkerPreconditioner — the average (global) Thomas-Fermi screening filter.
 *
 *   K(G) = G² / (G² + q0²)
 *
 * Damps the long-wavelength (small-G) components of the residual, where the
 * Coulomb kernel amplifies a small density error into a large potential error
 * and drives charge sloshing. As G → ∞ the filter tends to 1 and leaves the
 * short-wavelength residual alone. Kerker, PRB 23, 3082 (1981).
 *
 * The screening wavevector q0 is a single number for the whole cell, which
 * makes the operator diagonal in G: one multiply per component, no transforms.
 */

#include <algorithm>
#include <string>
#include <vector>

#include "layer_4_computation/iterator/internal/mixing/preconditioner/init.hpp"

namespace q3 {

/// Diagonal G²/(G²+q0²) residual filter.
class KerkerPreconditioner : public Preconditioner {
  public:
    /**
     * @param g_squared             |G|² per G-vector, in bohr⁻².
     * @param q0_squared            screening wavevector squared, in bohr⁻².
     * @param first_nonzero_g_index index of the first G ≠ 0 component; every
     *                              component below it is left untouched.
     */
    KerkerPreconditioner(const std::vector<Real>& g_squared,
                         Real q0_squared,
                         int first_nonzero_g_index)
        : first_nonzero_g_index_(first_nonzero_g_index) {
        weights_.assign(g_squared.size(), 1.0);
        for (std::size_t g_index = static_cast<std::size_t>(first_nonzero_g_index);
             g_index < g_squared.size();
             ++g_index) {
            weights_[g_index] = g_squared[g_index] / (g_squared[g_index] + q0_squared);
        }
    }

    [[nodiscard]] std::string name() const override {
        return "kerker";
    }

    void apply(std::vector<Complex>& residual_g) const override {
        std::size_t count = std::min(residual_g.size(), weights_.size());
        for (std::size_t g_index = static_cast<std::size_t>(first_nonzero_g_index_);
             g_index < count;
             ++g_index) {
            residual_g[g_index] *= weights_[g_index];
        }
    }

    /// The filter values, exposed for the local-TF inner preconditioner and for tests.
    [[nodiscard]] const std::vector<Real>& weights() const {
        return weights_;
    }

  private:
    std::vector<Real> weights_;
    int first_nonzero_g_index_{0};
};

} // namespace q3
