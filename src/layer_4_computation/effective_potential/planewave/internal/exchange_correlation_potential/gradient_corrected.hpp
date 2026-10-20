#pragma once

/**
 * GradientCorrectedExchangeCorrelationPotentialCalculator — V_xc for
 * functionals that also depend on |∇ρ|² (GGA).
 *
 * The White-Bird evaluation the branchy predecessor of this class ran inline:
 * build ∇ρ in real space from ρ(G) (three inverse FFTs per spin channel),
 * evaluate the functional's density and σ = ∇ρ·∇ρ derivatives, then fold the
 * σ-derivative back into the potential as -∇·(2 v_σ ∇ρ) via one more FFT
 * round trip. The spin-polarized path carries the three-sigma generalization
 * (σ↑↑, σ↑↓, σ↓↓) with the standard 2×-same-spin / 1×-cross-spin weighting.
 *
 * The numerical stream is the one the pre-decomposition code produced -- same
 * FFTs in the same order, same loop bodies -- so a run's results are
 * bit-identical to what the `requires_density_gradient()` branch used to
 * compute.
 */

#include <complex>
#include <vector>

#include "layer_0_core/types/init.hpp"
#include "layer_1_structure/crystal/crystal.hpp"
#include "layer_3_representation/basis/planewave/init.hpp"
#include "layer_4_computation/effective_potential/internal/exchange_correlation/init.hpp"
#include "layer_4_computation/effective_potential/planewave/internal/exchange_correlation_potential/init.hpp"

namespace q3 {

class GradientCorrectedExchangeCorrelationPotentialCalculator
    : public ExchangeCorrelationPotentialCalculator {
  public:
    GradientCorrectedExchangeCorrelationPotentialCalculator(
        const PlanewaveBasis& planewave_basis,
        const Crystal& crystal_structure,
        const XCFunctional& exchange_correlation)
        : planewave_basis_(planewave_basis)
        , crystal_structure_(crystal_structure)
        , exchange_correlation_(exchange_correlation) {}

    void
    evaluate_unpolarized(const std::vector<Real>& exchange_correlation_density,
                         const std::vector<Complex>& exchange_correlation_density_g,
                         std::vector<Real>& exchange_correlation_potential,
                         std::vector<Real>& exchange_correlation_energy_density) const override {
        int local_grid_points = planewave_basis_.fft_rho().local_grid_points();

        std::vector<Real> gradient_x;
        std::vector<Real> gradient_y;
        std::vector<Real> gradient_z;
        compute_density_gradient(
            exchange_correlation_density_g, gradient_x, gradient_y, gradient_z);

        std::vector<Real> gradient_squared(local_grid_points);
        for (int grid_index = 0; grid_index < local_grid_points; ++grid_index) {
            gradient_squared[grid_index] = gradient_x[grid_index] * gradient_x[grid_index] +
                                           gradient_y[grid_index] * gradient_y[grid_index] +
                                           gradient_z[grid_index] * gradient_z[grid_index];
        }

        std::vector<Real> kinetic_density;
        exchange_correlation_.compute_gradient_corrected(exchange_correlation_density,
                                                         gradient_squared,
                                                         exchange_correlation_potential,
                                                         kinetic_density,
                                                         exchange_correlation_energy_density);

        std::vector<Real> field_x(local_grid_points);
        std::vector<Real> field_y(local_grid_points);
        std::vector<Real> field_z(local_grid_points);
        for (int grid_index = 0; grid_index < local_grid_points; ++grid_index) {
            Real twice_kinetic_density = 2.0 * kinetic_density[grid_index];
            field_x[grid_index] = twice_kinetic_density * gradient_x[grid_index];
            field_y[grid_index] = twice_kinetic_density * gradient_y[grid_index];
            field_z[grid_index] = twice_kinetic_density * gradient_z[grid_index];
        }
        subtract_gradient_divergence(field_x, field_y, field_z, exchange_correlation_potential);
    }

    void
    evaluate_spin_polarized(const std::vector<Real>& exchange_correlation_density_up,
                            const std::vector<Real>& exchange_correlation_density_down,
                            const std::vector<Complex>& exchange_correlation_density_up_g,
                            const std::vector<Complex>& exchange_correlation_density_down_g,
                            std::vector<Real>& exchange_correlation_potential_up,
                            std::vector<Real>& exchange_correlation_potential_down,
                            std::vector<Real>& exchange_correlation_energy_density) const override {
        int local_grid_points = planewave_basis_.fft_rho().local_grid_points();

        std::vector<std::vector<Real>> gradient_x(2);
        std::vector<std::vector<Real>> gradient_y(2);
        std::vector<std::vector<Real>> gradient_z(2);
        for (int spin_index = 0; spin_index < 2; ++spin_index) {
            const std::vector<Complex>& density_g = (spin_index == 0)
                                                        ? exchange_correlation_density_up_g
                                                        : exchange_correlation_density_down_g;
            compute_density_gradient(
                density_g, gradient_x[spin_index], gradient_y[spin_index], gradient_z[spin_index]);
        }

        std::vector<Real> gradient_squared_up_up(local_grid_points);
        std::vector<Real> gradient_squared_up_down(local_grid_points);
        std::vector<Real> gradient_squared_down_down(local_grid_points);
        for (int grid_index = 0; grid_index < local_grid_points; ++grid_index) {
            gradient_squared_up_up[grid_index] =
                gradient_x[0][grid_index] * gradient_x[0][grid_index] +
                gradient_y[0][grid_index] * gradient_y[0][grid_index] +
                gradient_z[0][grid_index] * gradient_z[0][grid_index];
            gradient_squared_down_down[grid_index] =
                gradient_x[1][grid_index] * gradient_x[1][grid_index] +
                gradient_y[1][grid_index] * gradient_y[1][grid_index] +
                gradient_z[1][grid_index] * gradient_z[1][grid_index];
            gradient_squared_up_down[grid_index] =
                gradient_x[0][grid_index] * gradient_x[1][grid_index] +
                gradient_y[0][grid_index] * gradient_y[1][grid_index] +
                gradient_z[0][grid_index] * gradient_z[1][grid_index];
        }

        std::vector<Real> kinetic_density_up_up;
        std::vector<Real> kinetic_density_up_down;
        std::vector<Real> kinetic_density_down_down;
        exchange_correlation_.compute_gradient_corrected_spin(exchange_correlation_density_up,
                                                              exchange_correlation_density_down,
                                                              gradient_squared_up_up,
                                                              gradient_squared_up_down,
                                                              gradient_squared_down_down,
                                                              exchange_correlation_potential_up,
                                                              exchange_correlation_potential_down,
                                                              kinetic_density_up_up,
                                                              kinetic_density_up_down,
                                                              kinetic_density_down_down,
                                                              exchange_correlation_energy_density);

        for (int spin_index = 0; spin_index < 2; ++spin_index) {
            auto& kinetic_density_same =
                (spin_index == 0) ? kinetic_density_up_up : kinetic_density_down_down;
            auto& gradient_x_same = gradient_x[spin_index];
            auto& gradient_y_same = gradient_y[spin_index];
            auto& gradient_z_same = gradient_z[spin_index];
            auto& gradient_x_other = gradient_x[1 - spin_index];
            auto& gradient_y_other = gradient_y[1 - spin_index];
            auto& gradient_z_other = gradient_z[1 - spin_index];

            std::vector<Real> field_x(local_grid_points);
            std::vector<Real> field_y(local_grid_points);
            std::vector<Real> field_z(local_grid_points);
            for (int grid_index = 0; grid_index < local_grid_points; ++grid_index) {
                field_x[grid_index] =
                    2.0 * kinetic_density_same[grid_index] * gradient_x_same[grid_index] +
                    kinetic_density_up_down[grid_index] * gradient_x_other[grid_index];
                field_y[grid_index] =
                    2.0 * kinetic_density_same[grid_index] * gradient_y_same[grid_index] +
                    kinetic_density_up_down[grid_index] * gradient_y_other[grid_index];
                field_z[grid_index] =
                    2.0 * kinetic_density_same[grid_index] * gradient_z_same[grid_index] +
                    kinetic_density_up_down[grid_index] * gradient_z_other[grid_index];
            }

            auto& exchange_correlation_potential_spin = (spin_index == 0)
                                                            ? exchange_correlation_potential_up
                                                            : exchange_correlation_potential_down;
            subtract_gradient_divergence(
                field_x, field_y, field_z, exchange_correlation_potential_spin);
        }
    }

  private:
    /// Real-space Cartesian gradient of a density given in reciprocal space.
    void compute_density_gradient(const std::vector<Complex>& density_g,
                                  std::vector<Real>& gradient_x,
                                  std::vector<Real>& gradient_y,
                                  std::vector<Real>& gradient_z) const {
        int number_of_g_vectors = planewave_basis_.number_of_g_vectors();
        Real two_pi_over_lattice_parameter = crystal_structure_.two_pi_over_lattice_parameter();
        std::vector<Complex> gradient_gx(number_of_g_vectors);
        std::vector<Complex> gradient_gy(number_of_g_vectors);
        std::vector<Complex> gradient_gz(number_of_g_vectors);
        for (int g_vector_index = 0; g_vector_index < number_of_g_vectors; ++g_vector_index) {
            Complex imaginary_rho_g = Complex(0.0, 1.0) * density_g[g_vector_index];
            Vector3D g_cartesian =
                planewave_basis_.g_vectors()[g_vector_index].g_cart * two_pi_over_lattice_parameter;
            gradient_gx[g_vector_index] = imaginary_rho_g * g_cartesian.x;
            gradient_gy[g_vector_index] = imaginary_rho_g * g_cartesian.y;
            gradient_gz[g_vector_index] = imaginary_rho_g * g_cartesian.z;
        }
        planewave_basis_.rho_g_to_r(gradient_gx, gradient_x);
        planewave_basis_.rho_g_to_r(gradient_gy, gradient_y);
        planewave_basis_.rho_g_to_r(gradient_gz, gradient_z);
    }

    /// Subtract the divergence of the gradient-correction vector field from the potential.
    void subtract_gradient_divergence(const std::vector<Real>& field_x,
                                      const std::vector<Real>& field_y,
                                      const std::vector<Real>& field_z,
                                      std::vector<Real>& potential) const {
        int number_of_g_vectors = planewave_basis_.number_of_g_vectors();
        int local_grid_points = planewave_basis_.fft_rho().local_grid_points();
        Real two_pi_over_lattice_parameter = crystal_structure_.two_pi_over_lattice_parameter();

        std::vector<Complex> field_gx;
        std::vector<Complex> field_gy;
        std::vector<Complex> field_gz;
        planewave_basis_.rho_r_to_g(field_x, field_gx);
        planewave_basis_.rho_r_to_g(field_y, field_gy);
        planewave_basis_.rho_r_to_g(field_z, field_gz);

        std::vector<Complex> divergence_g(number_of_g_vectors);
        for (int g_vector_index = 0; g_vector_index < number_of_g_vectors; ++g_vector_index) {
            Vector3D g_cartesian =
                planewave_basis_.g_vectors()[g_vector_index].g_cart * two_pi_over_lattice_parameter;
            divergence_g[g_vector_index] =
                Complex(0.0, 1.0) * (g_cartesian.x * field_gx[g_vector_index] +
                                     g_cartesian.y * field_gy[g_vector_index] +
                                     g_cartesian.z * field_gz[g_vector_index]);
        }
        std::vector<Real> divergence_real;
        planewave_basis_.rho_g_to_r(divergence_g, divergence_real);
        for (int grid_index = 0; grid_index < local_grid_points; ++grid_index) {
            potential[grid_index] -= divergence_real[grid_index];
        }
    }

    const PlanewaveBasis& planewave_basis_;
    const Crystal& crystal_structure_;
    const XCFunctional& exchange_correlation_;
};

} // namespace q3
