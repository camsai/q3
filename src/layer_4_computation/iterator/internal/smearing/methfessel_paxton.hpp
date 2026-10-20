#pragma once

/**
 * MethfesselPaxtonSmearing — Methfessel-Paxton occupation smearing of order N >= 1.
 *
 * Methfessel & Paxton, Phys. Rev. B 40, 3616 (1989). The smearing delta function is
 * expanded in even Hermite polynomials,
 *
 *     δ_N(x) = Σ_{n=0..N} A_n H_{2n}(x) exp(-x²),   A_n = (-1)^n / (n! 4^n √π),
 *
 * which integrates to the occupation (x = (ε - E_F)/σ, occupied for x < 0)
 *
 *     f_N(x) = ½ erfc(x) + Σ_{n=1..N} A_n H_{2n-1}(x) exp(-x²),
 *
 * and the generalized-entropy term of the free energy involves only the highest order,
 * S_N(x) = ½ A_N H_{2N}(x) exp(-x²). Order N makes the free energy converge to the
 * σ → 0 total energy as O(σ^(2N+2)). Q3's sign convention for entropy_weight is −S_N,
 * matching GaussianSmearing (the N = 0 case).
 *
 * Order 1 keeps its original closed form so existing order-1 results stay bit-identical;
 * higher orders use the Hermite three-term recurrence H_{k+1} = 2x H_k − 2k H_{k−1}.
 */

#include <cmath>
#include <stdexcept>
#include <string>

#include "layer_4_computation/iterator/internal/smearing/init.hpp"

namespace q3 {

/// Methfessel-Paxton smearing of order N >= 1 (default 1).
class MethfesselPaxtonSmearing : public SmearingScheme {
  public:
    explicit MethfesselPaxtonSmearing(Real smearing_width = 0.0, int order = 1)
        : SmearingScheme(smearing_width), order_(order) {
        if (order_ < 1) {
            throw std::invalid_argument("Methfessel-Paxton order must be >= 1, got " +
                                        std::to_string(order_));
        }
    }

    [[nodiscard]] Real occupation(Real x) const override {
        const Real sqrtpi_inv = 1.0 / std::sqrt(constants::pi);
        if (order_ == 1) {
            // f_1(x) = (1/2)erfc(x) - x*exp(-x^2) / (2*sqrt(pi))
            return 0.5 * std::erfc(x) - 0.5 * sqrtpi_inv * x * std::exp(-x * x);
        }
        const Real gaussian = std::exp(-x * x);
        Real occupation_value = 0.5 * std::erfc(x);
        Real hermite_previous = 1.0;    // H_0
        Real hermite_current = 2.0 * x; // H_1
        int hermite_degree = 1;
        Real coefficient = sqrtpi_inv; // A_0
        for (int n = 1; n <= order_; ++n) {
            coefficient *= -1.0 / (4.0 * n);
            occupation_value += coefficient * hermite_current * gaussian; // A_n H_{2n-1}
            advance_hermite(x, hermite_previous, hermite_current, hermite_degree);
            advance_hermite(x, hermite_previous, hermite_current, hermite_degree);
        }
        return occupation_value;
    }

    /**
     * −S_N(x) = −½ A_N H_{2N}(x) exp(-x²). For N = 1 this is
     * (2x² − 1) exp(-x²) / (4√π), the original closed form.
     */
    [[nodiscard]] Real entropy_weight(Real x) const override {
        const Real sqrtpi_inv = 1.0 / std::sqrt(constants::pi);
        if (order_ == 1) {
            return 0.25 * sqrtpi_inv * (2.0 * x * x - 1.0) * std::exp(-x * x);
        }
        Real hermite_previous = 1.0;
        Real hermite_current = 2.0 * x;
        int hermite_degree = 1;
        while (hermite_degree < 2 * order_) {
            advance_hermite(x, hermite_previous, hermite_current, hermite_degree);
        }
        Real coefficient = sqrtpi_inv;
        for (int n = 1; n <= order_; ++n) {
            coefficient *= -1.0 / (4.0 * n);
        }
        return -0.5 * coefficient * hermite_current * std::exp(-x * x);
    }

    [[nodiscard]] std::string name() const override {
        return "methfessel-paxton";
    }

    /// Expansion order N.
    [[nodiscard]] int order() const {
        return order_;
    }

  private:
    /// Advance (H_{k-1}, H_k) to (H_k, H_{k+1}) via H_{k+1} = 2x H_k − 2k H_{k−1}.
    static void advance_hermite(Real x, Real& previous, Real& current, int& degree) {
        const Real next = 2.0 * x * current - 2.0 * degree * previous;
        previous = current;
        current = next;
        ++degree;
    }

    int order_;
};

} // namespace q3
