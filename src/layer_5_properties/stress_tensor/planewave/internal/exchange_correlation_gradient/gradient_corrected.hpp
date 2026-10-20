#pragma once

/**
 * GradientCorrectedExchangeCorrelationGradientStressCalculator — the GGA
 * ∇ρ⊗∇ρ stress term (Nielsen-Martin), unpolarized and LSDA.
 *
 * The numerical stream is the one the pre-decomposition
 * `requires_density_gradient()` branch produced -- same FFTs in the same
 * order, same accumulation loops -- so stress tensors are bit-identical to
 * what ExchangeCorrelationStressTensorCalculator::compute() used to emit.
 * Densities come exclusively through ExchangeCorrelationStressDensitySource,
 * which is how the NLCC core-charge subclassing keeps working unchanged.
 */

#include <complex>
#include <vector>

#include "layer_0_core/types/init.hpp"
#include "layer_1_structure/crystal/crystal.hpp"
#include "layer_3_representation/basis/planewave/init.hpp"
#include "layer_4_computation/effective_potential/internal/exchange_correlation/init.hpp"
#include "layer_4_computation/state/kohn_sham/planewave/norm_conserving/init.hpp"
#include "layer_5_properties/stress_tensor/planewave/contributions.hpp"
#include "layer_5_properties/stress_tensor/planewave/internal/exchange_correlation_gradient/init.hpp"

namespace q3 {

class GradientCorrectedExchangeCorrelationGradientStressCalculator
    : public ExchangeCorrelationGradientStressCalculator {
  public:
    GradientCorrectedExchangeCorrelationGradientStressCalculator(
        const Crystal& crystal_structure,
        const PlanewaveBasis& planewave_basis,
        const NormConservingPlanewaveKohnShamSCFState& scf_state,
        const ExchangeCorrelationStressDensitySource& density_source,
        StressTensorContributions& result)
        : crystal_structure_(crystal_structure)
        , planewave_basis_(planewave_basis)
        , scf_state_(scf_state)
        , density_source_(density_source)
        , result_(result) {}

    void add_gradient_stress() override {
        const XCFunctional& exchange_correlation_functional = (*scf_state_.exchange_correlation);
        if (scf_state_.number_of_spin_components() == 2) {
            compute_gga_gradient_stress_polarized(exchange_correlation_functional);
        } else {
            std::vector<Complex> total_charge_density_g =
                density_source_.build_total_charge_density_g();
            compute_gga_gradient_stress(exchange_correlation_functional, total_charge_density_g);
        }
    }

  private:
    /// ∇_direction ρ(r), computed as F⁻¹[i·G_direction·ρ(G)].
    [[nodiscard]] std::vector<Real>
    compute_gradient_component(const std::vector<Complex>& density_g, int direction) const {
        std::vector<Complex> gradient_g(planewave_basis_.number_of_g_vectors());
        for (int g_vector_index = 0; g_vector_index < planewave_basis_.number_of_g_vectors();
             g_vector_index++) {
            Real gradient_component =
                planewave_basis_.g_vectors()[g_vector_index].g_cart[direction] *
                crystal_structure_.two_pi_over_lattice_parameter();
            gradient_g[g_vector_index] =
                Complex(0.0, 1.0) * gradient_component * density_g[g_vector_index];
        }
        std::vector<Real> gradient_r;
        planewave_basis_.rho_g_to_r(gradient_g, gradient_r);
        return gradient_r;
    }

    void compute_gga_gradient_stress(const XCFunctional& exchange_correlation_functional,
                                     const std::vector<Complex>& total_charge_density_g) {
        const int number_of_real_space_points = planewave_basis_.fft_rho().total_grid_points();

        std::vector<Real> gradient_x = compute_gradient_component(total_charge_density_g, 0);
        std::vector<Real> gradient_y = compute_gradient_component(total_charge_density_g, 1);
        std::vector<Real> gradient_z = compute_gradient_component(total_charge_density_g, 2);

        std::vector<Real> total_charge_density_r(number_of_real_space_points);
        planewave_basis_.rho_g_to_r(total_charge_density_g, total_charge_density_r);

        std::vector<Real> gradient_charge_density_squared(number_of_real_space_points);
        for (int point_index = 0; point_index < number_of_real_space_points; point_index++) {
            gradient_charge_density_squared[point_index] =
                gradient_x[point_index] * gradient_x[point_index] +
                gradient_y[point_index] * gradient_y[point_index] +
                gradient_z[point_index] * gradient_z[point_index];
        }

        std::vector<Real> gradient_charge_density_vxc_scratch, gradient_correction_coefficients,
            exc_scratch;
        exchange_correlation_functional.compute_gradient_corrected(
            total_charge_density_r,
            gradient_charge_density_squared,
            gradient_charge_density_vxc_scratch,
            gradient_correction_coefficients,
            exc_scratch);

        Real gc11 = 0, gc21 = 0, gc22 = 0, gc31 = 0, gc32 = 0, gc33 = 0;
        int local_grid_points = planewave_basis_.fft_rho().local_grid_points();
        for (int point_index = 0; point_index < local_grid_points; point_index++) {
            Real v2xc = 2.0 * gradient_correction_coefficients[point_index];
            gc11 += gradient_x[point_index] * gradient_x[point_index] * v2xc;
            gc21 += gradient_y[point_index] * gradient_x[point_index] * v2xc;
            gc22 += gradient_y[point_index] * gradient_y[point_index] * v2xc;
            gc31 += gradient_z[point_index] * gradient_x[point_index] * v2xc;
            gc32 += gradient_z[point_index] * gradient_y[point_index] * v2xc;
            gc33 += gradient_z[point_index] * gradient_z[point_index] * v2xc;
        }
        Real inverse_total_points = 1.0 / planewave_basis_.fft_rho().total_grid_points();
        result_.exchange_correlation[0][0] += gc11 * inverse_total_points;
        result_.exchange_correlation[1][0] += gc21 * inverse_total_points;
        result_.exchange_correlation[1][1] += gc22 * inverse_total_points;
        result_.exchange_correlation[2][0] += gc31 * inverse_total_points;
        result_.exchange_correlation[2][1] += gc32 * inverse_total_points;
        result_.exchange_correlation[2][2] += gc33 * inverse_total_points;

        result_.exchange_correlation[0][1] = result_.exchange_correlation[1][0];
        result_.exchange_correlation[0][2] = result_.exchange_correlation[2][0];
        result_.exchange_correlation[1][2] = result_.exchange_correlation[2][1];
    }

    /**
     * Spin-polarized (LSDA) GGA gradient stress.
     *
     * Generalizes compute_gga_gradient_stress() to the three-sigma polarized
     * functional derivative (σ↑↑, σ↑↓, σ↓↓), following the same up/down
     * weighting (2× same-spin, 1× cross-spin) used to build the polarized
     * XC potential's gradient term in EffectivePotentialCalculator — the
     * standard Nielsen-Martin GGA stress extended to two spin channels.
     * The unpolarized path must not be called here: the XCFunctional
     * held by scf_state_ is configured XC_POLARIZED for LSDA, and libxc's
     * unpolarized entry points write 2× the buffer size that call expects,
     * which is a heap-buffer-overflow (this is what used to crash here).
     */
    void
    compute_gga_gradient_stress_polarized(const XCFunctional& exchange_correlation_functional) {
        const int number_of_real_space_points = planewave_basis_.fft_rho().total_grid_points();
        const int local_grid_points = planewave_basis_.fft_rho().local_grid_points();

        std::vector<Complex> density_up_g = density_source_.build_spin_charge_density_g(0);
        std::vector<Complex> density_down_g = density_source_.build_spin_charge_density_g(1);

        std::vector<Real> gradient_x_up = compute_gradient_component(density_up_g, 0);
        std::vector<Real> gradient_y_up = compute_gradient_component(density_up_g, 1);
        std::vector<Real> gradient_z_up = compute_gradient_component(density_up_g, 2);
        std::vector<Real> gradient_x_down = compute_gradient_component(density_down_g, 0);
        std::vector<Real> gradient_y_down = compute_gradient_component(density_down_g, 1);
        std::vector<Real> gradient_z_down = compute_gradient_component(density_down_g, 2);

        std::vector<Real> density_up_r(number_of_real_space_points);
        std::vector<Real> density_down_r(number_of_real_space_points);
        planewave_basis_.rho_g_to_r(density_up_g, density_up_r);
        planewave_basis_.rho_g_to_r(density_down_g, density_down_r);

        std::vector<Real> gradient_squared_up_up(number_of_real_space_points);
        std::vector<Real> gradient_squared_up_down(number_of_real_space_points);
        std::vector<Real> gradient_squared_down_down(number_of_real_space_points);
        for (int point_index = 0; point_index < number_of_real_space_points; point_index++) {
            gradient_squared_up_up[point_index] =
                gradient_x_up[point_index] * gradient_x_up[point_index] +
                gradient_y_up[point_index] * gradient_y_up[point_index] +
                gradient_z_up[point_index] * gradient_z_up[point_index];
            gradient_squared_down_down[point_index] =
                gradient_x_down[point_index] * gradient_x_down[point_index] +
                gradient_y_down[point_index] * gradient_y_down[point_index] +
                gradient_z_down[point_index] * gradient_z_down[point_index];
            gradient_squared_up_down[point_index] =
                gradient_x_up[point_index] * gradient_x_down[point_index] +
                gradient_y_up[point_index] * gradient_y_down[point_index] +
                gradient_z_up[point_index] * gradient_z_down[point_index];
        }

        std::vector<Real> xc_potential_up_scratch, xc_potential_down_scratch;
        std::vector<Real> vsigma_up_up, vsigma_up_down, vsigma_down_down, exc_scratch;
        exchange_correlation_functional.compute_gradient_corrected_spin(density_up_r,
                                                                        density_down_r,
                                                                        gradient_squared_up_up,
                                                                        gradient_squared_up_down,
                                                                        gradient_squared_down_down,
                                                                        xc_potential_up_scratch,
                                                                        xc_potential_down_scratch,
                                                                        vsigma_up_up,
                                                                        vsigma_up_down,
                                                                        vsigma_down_down,
                                                                        exc_scratch);

        Real gc11 = 0, gc21 = 0, gc22 = 0, gc31 = 0, gc32 = 0, gc33 = 0;
        for (int point_index = 0; point_index < local_grid_points; point_index++) {
            Real v2xc_up_up = 2.0 * vsigma_up_up[point_index];
            Real v2xc_down_down = 2.0 * vsigma_down_down[point_index];
            Real v2xc_up_down = vsigma_up_down[point_index];

            Real gx_up = gradient_x_up[point_index], gx_down = gradient_x_down[point_index];
            Real gy_up = gradient_y_up[point_index], gy_down = gradient_y_down[point_index];
            Real gz_up = gradient_z_up[point_index], gz_down = gradient_z_down[point_index];

            gc11 += v2xc_up_up * gx_up * gx_up + v2xc_down_down * gx_down * gx_down +
                    v2xc_up_down * (gx_up * gx_down + gx_down * gx_up);
            gc21 += v2xc_up_up * gy_up * gx_up + v2xc_down_down * gy_down * gx_down +
                    v2xc_up_down * (gy_up * gx_down + gy_down * gx_up);
            gc22 += v2xc_up_up * gy_up * gy_up + v2xc_down_down * gy_down * gy_down +
                    v2xc_up_down * (gy_up * gy_down + gy_down * gy_up);
            gc31 += v2xc_up_up * gz_up * gx_up + v2xc_down_down * gz_down * gx_down +
                    v2xc_up_down * (gz_up * gx_down + gz_down * gx_up);
            gc32 += v2xc_up_up * gz_up * gy_up + v2xc_down_down * gz_down * gy_down +
                    v2xc_up_down * (gz_up * gy_down + gz_down * gy_up);
            gc33 += v2xc_up_up * gz_up * gz_up + v2xc_down_down * gz_down * gz_down +
                    v2xc_up_down * (gz_up * gz_down + gz_down * gz_up);
        }
        Real inverse_total_points = 1.0 / planewave_basis_.fft_rho().total_grid_points();
        result_.exchange_correlation[0][0] += gc11 * inverse_total_points;
        result_.exchange_correlation[1][0] += gc21 * inverse_total_points;
        result_.exchange_correlation[1][1] += gc22 * inverse_total_points;
        result_.exchange_correlation[2][0] += gc31 * inverse_total_points;
        result_.exchange_correlation[2][1] += gc32 * inverse_total_points;
        result_.exchange_correlation[2][2] += gc33 * inverse_total_points;

        result_.exchange_correlation[0][1] = result_.exchange_correlation[1][0];
        result_.exchange_correlation[0][2] = result_.exchange_correlation[2][0];
        result_.exchange_correlation[1][2] = result_.exchange_correlation[2][1];
    }

    const Crystal& crystal_structure_;
    const PlanewaveBasis& planewave_basis_;
    const NormConservingPlanewaveKohnShamSCFState& scf_state_;
    const ExchangeCorrelationStressDensitySource& density_source_;
    StressTensorContributions& result_;
};

} // namespace q3
