#pragma once

/**
 * Exchange-correlation functional interface.
 *
 * Supports both built-in functionals (PZ LDA) and LibXC-based functionals
 * (600+ LDA/GGA functionals).
 * Supports both unpolarized (number_of_spin_components=1) and spin-polarized
 * (number_of_spin_components=2) calculations.
 */

#include <cassert>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "layer_0_core/types/init.hpp"

namespace q3 {

/// Abstract base class for XC functionals.
class XCFunctional {
  public:
    virtual ~XCFunctional() = default;

    /// Compute XC potential and energy density (LDA interface, unpolarized).
    virtual void compute(const std::vector<Real>& density,
                         std::vector<Real>& xc_potential,
                         std::vector<Real>& xc_energy_density) const = 0;

    /**
     * Compute XC potential and energy density (LDA interface, spin-polarized).
     * density_up/density_down are spin-up/spin-down densities.
     * xc_potential_up/xc_potential_down are the spin-resolved XC potentials.
     * xc_energy_density is the XC energy density (per unit volume, evaluated at total density).
     */
    virtual void compute_spin(const std::vector<Real>& density_up,
                              const std::vector<Real>& density_down,
                              std::vector<Real>& xc_potential_up,
                              std::vector<Real>& xc_potential_down,
                              std::vector<Real>& xc_energy_density) const {
        /**
         * This default body is refused rather than reached: a spin-polarized calculation
         * using a functional that did not override compute_spin() would silently
         * produce wrong physics (LDA-unpolarized result for a spin-polarized input).
         */
        throw std::logic_error(
            "compute_spin() not overridden: this exchange-correlation functional has no "
            "spin-polarized form, but the run is spin-polarized. The unpolarized body would "
            "return an unpolarized result for a polarized density -- wrong physics that "
            "converges and prints a plausible number. This was an assert, which a release "
            "build compiles out.");
        (void)density_up;
        (void)density_down;
        (void)xc_potential_up;
        (void)xc_potential_down;
        (void)xc_energy_density;
    }

    /// Whether this is a GGA functional (needs gradients).
    [[nodiscard]] virtual bool requires_density_gradient() const {
        return false;
    }

    /**
     * The spin mode this functional was constructed for: 1 or 2.
     *
     * A LibXC handle is initialized once for a fixed mode and cannot be used in
     * the other — a polarized handle driven through `compute()` misreads the
     * interleaved (ρ↑, ρ↓) array it expects, and returns numbers that are
     * finite, plausible and wrong. Callers that pick between the unpolarized
     * and spin-polarized routines must branch on this rather than on how many
     * densities they happen to hold.
     */
    [[nodiscard]] virtual int number_of_spin_components() const {
        return 1;
    }

    /**
     * Compute XC potential and energy density (GGA interface, unpolarized).
     * gradient_density_squared[i] = |∇ρ|² at grid point i.
     */
    virtual void compute_gradient_corrected(const std::vector<Real>& density,
                                            const std::vector<Real>& gradient_density_squared,
                                            std::vector<Real>& xc_potential,
                                            std::vector<Real>& gradient_correction_density,
                                            std::vector<Real>& xc_energy_density) const {
        /**
         * This default body is refused rather than reached: a GGA calculation using a
         * functional that did not override compute_gradient_corrected() would silently fall back
         * to LDA, producing numerically wrong results with no error.
         */
        throw std::logic_error(
            "compute_gradient_corrected() not overridden: this exchange-correlation functional "
            "has no gradient-corrected form, but the run selected a GGA. Falling through would "
            "silently evaluate the functional at LDA level. This was an assert, which a release "
            "build compiles out.");
        (void)density;
        (void)gradient_density_squared;
        (void)xc_potential;
        (void)gradient_correction_density;
        (void)xc_energy_density;
    }

    /**
     * Compute XC potential and energy density (GGA interface, spin-polarized).
     * grho2_uu = |∇ρ↑|², grho2_ud = ∇ρ↑·∇ρ↓, grho2_dd = |∇ρ↓|²
     */
    virtual void compute_gradient_corrected_spin(const std::vector<Real>& density_up,
                                                 const std::vector<Real>& density_down,
                                                 const std::vector<Real>& grho2_uu,
                                                 const std::vector<Real>& grho2_ud,
                                                 const std::vector<Real>& grho2_dd,
                                                 std::vector<Real>& xc_potential_up,
                                                 std::vector<Real>& xc_potential_down,
                                                 std::vector<Real>& gradient_correction_up_up,
                                                 std::vector<Real>& gradient_correction_up_down,
                                                 std::vector<Real>& gradient_correction_down_down,
                                                 std::vector<Real>& xc_energy_density) const {
        /**
         * This default body is refused rather than reached: a spin-polarized GGA calculation
         * using a functional that did not override compute_gradient_corrected_spin() would silently
         * produce wrong physics.
         */
        throw std::logic_error(
            "compute_gradient_corrected_spin() not overridden: this exchange-correlation "
            "functional has no spin-polarized gradient-corrected form, but the run is both. "
            "This was an assert, which a release build compiles out.");
        (void)density_up;
        (void)density_down;
        (void)grho2_uu;
        (void)grho2_ud;
        (void)grho2_dd;
        (void)xc_potential_up;
        (void)xc_potential_down;
        (void)gradient_correction_up_up;
        (void)gradient_correction_up_down;
        (void)gradient_correction_down_down;
        (void)xc_energy_density;
    }

    /**
     * Compute XC kernel f_xc(r) = d²E_xc/dρ² for DFPT linear response.
     * Default: compute via numerical differentiation of V_xc.
     */
    virtual void
    compute_exchange_correlation_kernel(const std::vector<Real>& density,
                                        std::vector<Real>& exchange_correlation_kernel) const {
        int number_of_grid_points = static_cast<int>(density.size());
        exchange_correlation_kernel.resize(number_of_grid_points);
        constexpr Real delta = 1e-6;
        std::vector<Real> density_plus(number_of_grid_points), density_minus(number_of_grid_points);
        for (int grid_index = 0; grid_index < number_of_grid_points; grid_index++) {
            density_plus[grid_index] = std::max(density[grid_index] + delta, delta);
            density_minus[grid_index] = std::max(density[grid_index] - delta, delta);
        }
        std::vector<Real> vxc_p(number_of_grid_points), vxc_m(number_of_grid_points),
            exc_p(number_of_grid_points), exc_m(number_of_grid_points);
        compute(density_plus, vxc_p, exc_p);
        compute(density_minus, vxc_m, exc_m);
        for (int grid_index = 0; grid_index < number_of_grid_points; grid_index++) {
            exchange_correlation_kernel[grid_index] =
                (vxc_p[grid_index] - vxc_m[grid_index]) /
                (density_plus[grid_index] - density_minus[grid_index]);
        }
    }

    /**
     * Compute spin-resolved XC kernel for LSDA.
     */
    virtual void compute_fxc_spin(const std::vector<Real>& density_up,
                                  const std::vector<Real>& density_down,
                                  std::vector<Real>& fxc_uu,
                                  std::vector<Real>& fxc_ud,
                                  std::vector<Real>& fxc_dd) const {
        int number_of_grid_points = static_cast<int>(density_up.size());
        fxc_uu.resize(number_of_grid_points);
        fxc_ud.resize(number_of_grid_points);
        fxc_dd.resize(number_of_grid_points);
        constexpr Real delta = 1e-6;

        std::vector<Real> density_up_plus(number_of_grid_points),
            density_up_minus(number_of_grid_points);
        for (int grid_index = 0; grid_index < number_of_grid_points; grid_index++) {
            density_up_plus[grid_index] = std::max(density_up[grid_index] + delta, delta);
            density_up_minus[grid_index] = std::max(density_up[grid_index] - delta, delta);
        }
        std::vector<Real> xc_potential_up_plus(number_of_grid_points),
            xc_potential_up_minus(number_of_grid_points),
            xc_potential_down_plus(number_of_grid_points),
            xc_potential_down_minus(number_of_grid_points),
            xc_energy_density(number_of_grid_points);
        compute_spin(density_up_plus,
                     density_down,
                     xc_potential_up_plus,
                     xc_potential_down_plus,
                     xc_energy_density);
        compute_spin(density_up_minus,
                     density_down,
                     xc_potential_up_minus,
                     xc_potential_down_minus,
                     xc_energy_density);
        for (int grid_index = 0; grid_index < number_of_grid_points; grid_index++) {
            fxc_uu[grid_index] =
                (xc_potential_up_plus[grid_index] - xc_potential_up_minus[grid_index]) /
                (density_up_plus[grid_index] - density_up_minus[grid_index]);
            fxc_ud[grid_index] =
                (xc_potential_down_plus[grid_index] - xc_potential_down_minus[grid_index]) /
                (density_up_plus[grid_index] - density_up_minus[grid_index]);
        }

        std::vector<Real> density_down_plus(number_of_grid_points),
            density_down_minus(number_of_grid_points);
        for (int grid_index = 0; grid_index < number_of_grid_points; grid_index++) {
            density_down_plus[grid_index] = std::max(density_down[grid_index] + delta, delta);
            density_down_minus[grid_index] = std::max(density_down[grid_index] - delta, delta);
        }
        compute_spin(density_up,
                     density_down_plus,
                     xc_potential_up_plus,
                     xc_potential_down_plus,
                     xc_energy_density);
        compute_spin(density_up,
                     density_down_minus,
                     xc_potential_up_minus,
                     xc_potential_down_minus,
                     xc_energy_density);
        for (int grid_index = 0; grid_index < number_of_grid_points; grid_index++) {
            fxc_dd[grid_index] =
                (xc_potential_down_plus[grid_index] - xc_potential_down_minus[grid_index]) /
                (density_down_plus[grid_index] - density_down_minus[grid_index]);
        }
    }

    [[nodiscard]] virtual std::string name() const = 0;
};

} // namespace q3
