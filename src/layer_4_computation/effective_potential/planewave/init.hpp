#pragma once

/**
 * PlanewaveEffectivePotentialCalculator — Hartree + exchange-correlation + ionic local
 * potential assembly for a plane-wave Kohn-Sham representation.
 *
 * Given the spin-resolved charge density and the ionic local potential, this
 * calculator produces the real-space effective potential per spin and its
 * reciprocal-space projections, together with the Hartree and exchange-correlation
 * energy terms. It holds no mutable state; the SCF iterator owns the result and
 * copies it into its own members/energies.
 *
 * The exchange-correlation density is constructed by the virtual method
 * build_exchange_correlation_density(), which subclasses override to include
 * additional contributions (e.g., NLCC core charge from pseudopotentials).
 */

#include <cmath>
#include <complex>
#include <memory>
#include <vector>

#include "layer_0_core/types/constants.hpp"
#include "layer_0_core/types/init.hpp"
#include "layer_1_structure/crystal/crystal.hpp"
#include "layer_3_representation/basis/planewave/init.hpp"
#include "layer_4_computation/effective_potential/internal/exchange_correlation/init.hpp"
#include "layer_4_computation/effective_potential/planewave/internal/exchange_correlation_potential/factory.hpp"
#include "layer_4_computation/effective_potential/planewave/internal/exchange_correlation_potential/init.hpp"
#include "layer_4_computation/effective_potential/planewave/result.hpp"
#include "layer_4_computation/iterator/kohn_sham/planewave/execution/init.hpp"

namespace q3 {

class PlanewaveEffectivePotentialCalculator {
  public:
    PlanewaveEffectivePotentialCalculator(
        const PlanewaveBasis& planewave_basis,
        const Crystal& crystal_structure,
        const XCFunctional& exchange_correlation,
        const execution::PlanewaveExecutionPolicy& execution_policy)
        : planewave_basis_(planewave_basis)
        , crystal_structure_(crystal_structure)
        , exchange_correlation_(exchange_correlation)
        , execution_policy_(execution_policy)
        , exchange_correlation_potential_calculator_(
              create_exchange_correlation_potential_calculator(
                  planewave_basis, crystal_structure, exchange_correlation)) {}

    virtual ~PlanewaveEffectivePotentialCalculator() = default;

    EffectivePotentialResult compute(const std::vector<std::vector<Real>>& charge_density_real,
                                     int number_of_spin_components,
                                     const std::vector<Complex>& local_ionic_potential_g) const {
        EffectivePotentialResult output;
        int total_grid_points = planewave_basis_.fft_rho().total_grid_points();
        int local_grid_points = planewave_basis_.fft_rho().local_grid_points();

        std::vector<Real> total_charge_density_real(local_grid_points, 0.0);
        for (int spin_index = 0; spin_index < number_of_spin_components; ++spin_index) {
            for (int grid_index = 0; grid_index < local_grid_points; ++grid_index) {
                total_charge_density_real[grid_index] +=
                    charge_density_real[spin_index][grid_index];
            }
        }
        planewave_basis_.rho_r_to_g(total_charge_density_real, output.charge_density_g);

        std::vector<Real> hartree_potential_real = compute_hartree_potential(output);

        std::vector<std::vector<Real>> exchange_correlation_density;
        std::vector<std::vector<Complex>> exchange_correlation_density_g;
        build_exchange_correlation_density(charge_density_real,
                                           number_of_spin_components,
                                           exchange_correlation_density,
                                           exchange_correlation_density_g);

        std::vector<std::vector<Real>> exchange_correlation_potential_by_spin(
            number_of_spin_components, std::vector<Real>(local_grid_points));
        if (number_of_spin_components == 1) {
            compute_xc_potential_unpolarized(exchange_correlation_density,
                                             exchange_correlation_density_g,
                                             charge_density_real,
                                             exchange_correlation_potential_by_spin[0],
                                             output);
        } else {
            compute_xc_potential_spin_polarized(exchange_correlation_density,
                                                exchange_correlation_density_g,
                                                charge_density_real,
                                                exchange_correlation_potential_by_spin[0],
                                                exchange_correlation_potential_by_spin[1],
                                                output);
        }

        /**
         * Reduce XC energy and XC potential energy within this pool.
         * Both are real-space slab integrals (over local_grid_points) and each rank
         * in the band communicator holds only its slab of the grid. With 1-rank-per-pool
         * this is a no-op; with multiple ranks per pool the partial sums must be combined.
         */
        execution_policy_.reduce_within_pool_scalar(output.exchange_correlation_energy);
        execution_policy_.reduce_within_pool_scalar(output.exchange_correlation_potential_energy);

        assemble_effective_potential(hartree_potential_real,
                                     exchange_correlation_potential_by_spin,
                                     number_of_spin_components,
                                     local_ionic_potential_g,
                                     output);
        return output;
    }

  protected:
    const PlanewaveBasis& planewave_basis_;
    const Crystal& crystal_structure_;
    const XCFunctional& exchange_correlation_;
    const execution::PlanewaveExecutionPolicy& execution_policy_;
    /// V_xc evaluation strategy (local-density vs gradient-corrected), fixed at construction.
    std::unique_ptr<ExchangeCorrelationPotentialCalculator>
        exchange_correlation_potential_calculator_;
    mutable std::vector<Real> cached_local_ionic_potential_real_;

    /**
     * Build the density seen by the XC functional.
     * Base implementation copies the valence charge density directly.
     * Subclasses override to include additional contributions (e.g., NLCC core charge).
     */
    virtual void build_exchange_correlation_density(
        const std::vector<std::vector<Real>>& charge_density_real,
        int number_of_spin_components,
        std::vector<std::vector<Real>>& exchange_correlation_density,
        std::vector<std::vector<Complex>>& exchange_correlation_density_g) const {
        int number_of_g_vectors = planewave_basis_.number_of_g_vectors();
        int local_grid_points = planewave_basis_.fft_rho().local_grid_points();

        exchange_correlation_density.assign(number_of_spin_components,
                                            std::vector<Real>(local_grid_points));
        exchange_correlation_density_g.assign(number_of_spin_components,
                                              std::vector<Complex>(number_of_g_vectors));

        for (int spin_index = 0; spin_index < number_of_spin_components; ++spin_index) {
            for (int grid_index = 0; grid_index < local_grid_points; ++grid_index) {
                exchange_correlation_density[spin_index][grid_index] =
                    charge_density_real[spin_index][grid_index];
            }

            planewave_basis_.rho_r_to_g(charge_density_real[spin_index],
                                        exchange_correlation_density_g[spin_index]);
        }
    }

  private:
    std::vector<Real> compute_hartree_potential(EffectivePotentialResult& output) const {
        int number_of_g_vectors = planewave_basis_.number_of_g_vectors();
        int first_nonzero_g_index = planewave_basis_.first_nonzero_g_index();
        Real cell_volume = crystal_structure_.cell_volume();
        Real two_pi_over_lattice_parameter_squared =
            crystal_structure_.two_pi_over_lattice_parameter_squared();

        std::vector<Complex> hartree_potential_g(number_of_g_vectors, Complex(0.0, 0.0));
        Real hartree_energy_local = 0.0;
        bool is_gamma_only = planewave_basis_.is_gamma_only();
        for (int g_vector_index = first_nonzero_g_index; g_vector_index < number_of_g_vectors;
             ++g_vector_index) {
            Real g_squared = planewave_basis_.g_vectors()[g_vector_index].magnitude_squared *
                             two_pi_over_lattice_parameter_squared;
            Real prefactor =
                constants::electron_charge_squared_rydberg * constants::four_pi / g_squared;
            hartree_potential_g[g_vector_index] =
                output.charge_density_g[g_vector_index] * prefactor;
            /**
             * For gamma-only half-sphere: double the energy contribution since
             * each G≠0 represents both G and -G, and |ρ(-G)|² = |ρ(G)|².
             */
            Real gamma_factor = is_gamma_only ? 2.0 : 1.0;
            hartree_energy_local +=
                std::norm(output.charge_density_g[g_vector_index]) * prefactor * gamma_factor;
        }
        output.hartree_energy = hartree_energy_local * 0.5 * cell_volume;

        std::vector<Real> hartree_potential_real;
        planewave_basis_.rho_g_to_r(hartree_potential_g, hartree_potential_real);
        return hartree_potential_real;
    }

    void compute_xc_potential_unpolarized(
        const std::vector<std::vector<Real>>& exchange_correlation_density,
        const std::vector<std::vector<Complex>>& exchange_correlation_density_g,
        const std::vector<std::vector<Real>>& charge_density_real,
        std::vector<Real>& exchange_correlation_potential,
        EffectivePotentialResult& output) const {
        int total_grid_points = planewave_basis_.fft_rho().total_grid_points();
        int local_grid_points = planewave_basis_.fft_rho().local_grid_points();
        Real cell_volume = crystal_structure_.cell_volume();

        exchange_correlation_potential.assign(local_grid_points, 0.0);
        std::vector<Real> exchange_correlation_energy_density(local_grid_points);

        /**
         * Whether this evaluation needs density gradients was decided once, at
         * construction, by create_exchange_correlation_potential_calculator().
         */
        exchange_correlation_potential_calculator_->evaluate_unpolarized(
            exchange_correlation_density[0],
            exchange_correlation_density_g[0],
            exchange_correlation_potential,
            exchange_correlation_energy_density);

        output.exchange_correlation_energy = 0.0;
        output.exchange_correlation_potential_energy = 0.0;
        for (int grid_index = 0; grid_index < local_grid_points; ++grid_index) {
            output.exchange_correlation_energy += exchange_correlation_energy_density[grid_index] *
                                                  exchange_correlation_density[0][grid_index];
            output.exchange_correlation_potential_energy +=
                exchange_correlation_potential[grid_index] * charge_density_real[0][grid_index];
        }
        output.exchange_correlation_energy *= cell_volume / total_grid_points;
        output.exchange_correlation_potential_energy *= cell_volume / total_grid_points;
    }

    void compute_xc_potential_spin_polarized(
        const std::vector<std::vector<Real>>& exchange_correlation_density,
        const std::vector<std::vector<Complex>>& exchange_correlation_density_g,
        const std::vector<std::vector<Real>>& charge_density_real,
        std::vector<Real>& exchange_correlation_potential_up,
        std::vector<Real>& exchange_correlation_potential_down,
        EffectivePotentialResult& output) const {
        int total_grid_points = planewave_basis_.fft_rho().total_grid_points();
        int local_grid_points = planewave_basis_.fft_rho().local_grid_points();
        Real cell_volume = crystal_structure_.cell_volume();

        exchange_correlation_potential_up.assign(local_grid_points, 0.0);
        exchange_correlation_potential_down.assign(local_grid_points, 0.0);
        std::vector<Real> exchange_correlation_energy_density(local_grid_points);

        /**
         * Whether this evaluation needs density gradients was decided once, at
         * construction, by create_exchange_correlation_potential_calculator().
         */
        exchange_correlation_potential_calculator_->evaluate_spin_polarized(
            exchange_correlation_density[0],
            exchange_correlation_density[1],
            exchange_correlation_density_g[0],
            exchange_correlation_density_g[1],
            exchange_correlation_potential_up,
            exchange_correlation_potential_down,
            exchange_correlation_energy_density);

        output.exchange_correlation_energy = 0.0;
        output.exchange_correlation_potential_energy = 0.0;
        for (int grid_index = 0; grid_index < local_grid_points; ++grid_index) {
            Real total_density = exchange_correlation_density[0][grid_index] +
                                 exchange_correlation_density[1][grid_index];
            output.exchange_correlation_energy +=
                exchange_correlation_energy_density[grid_index] * total_density;
            output.exchange_correlation_potential_energy +=
                exchange_correlation_potential_up[grid_index] * charge_density_real[0][grid_index] +
                exchange_correlation_potential_down[grid_index] *
                    charge_density_real[1][grid_index];
        }
        output.exchange_correlation_energy *= cell_volume / total_grid_points;
        output.exchange_correlation_potential_energy *= cell_volume / total_grid_points;
    }

    /**
     * Combine Hartree, exchange-correlation and ionic local potentials into the
     * real-space effective potential per spin and its reciprocal-space projections.
     */
    void assemble_effective_potential(
        const std::vector<Real>& hartree_potential_real,
        const std::vector<std::vector<Real>>& exchange_correlation_potential_by_spin,
        int number_of_spin_components,
        const std::vector<Complex>& local_ionic_potential_g,
        EffectivePotentialResult& output) const {
        int grid_points = planewave_basis_.fft_rho().total_grid_points();

        // Cache the local ionic potential FFT (constant during SCF)
        if (cached_local_ionic_potential_real_.empty()) {
            planewave_basis_.rho_g_to_r(local_ionic_potential_g,
                                        cached_local_ionic_potential_real_);
        }
        int local_grid_points = planewave_basis_.fft_rho().local_grid_points();
        const auto& local_potential_real_space = cached_local_ionic_potential_real_;

        output.real_space_potential.resize(number_of_spin_components);
        for (int spin_index = 0; spin_index < number_of_spin_components; ++spin_index) {
            output.real_space_potential[spin_index].resize(local_grid_points);
            for (int grid_index = 0; grid_index < local_grid_points; ++grid_index) {
                output.real_space_potential[spin_index][grid_index] =
                    hartree_potential_real[grid_index] +
                    exchange_correlation_potential_by_spin[spin_index][grid_index] +
                    local_potential_real_space[grid_index];
            }
        }

        std::vector<Real> spin_averaged_potential(local_grid_points, 0.0);
        for (int spin_index = 0; spin_index < number_of_spin_components; ++spin_index) {
            for (int grid_index = 0; grid_index < local_grid_points; ++grid_index) {
                spin_averaged_potential[grid_index] +=
                    output.real_space_potential[spin_index][grid_index] / number_of_spin_components;
            }
        }
        planewave_basis_.rho_r_to_g(spin_averaged_potential, output.total_potential_g);

        /**
         * Each channel's own potential, transformed separately. Kept alongside
         * the average rather than replacing it: the average is still what the
         * norm-conserving path and the checkpoint format expect, and it is the
         * correct thing for an unpolarized cell.
         */
        output.total_potential_g_by_spin.resize(number_of_spin_components);
        for (int spin_index = 0; spin_index < number_of_spin_components; ++spin_index) {
            planewave_basis_.rho_r_to_g(output.real_space_potential[spin_index],
                                        output.total_potential_g_by_spin[spin_index]);
        }

        std::vector<Real> exchange_correlation_potential_average(local_grid_points, 0.0);
        for (int grid_index = 0; grid_index < local_grid_points; ++grid_index) {
            exchange_correlation_potential_average[grid_index] =
                spin_averaged_potential[grid_index] - hartree_potential_real[grid_index] -
                local_potential_real_space[grid_index];
        }
        planewave_basis_.rho_r_to_g(exchange_correlation_potential_average,
                                    output.exchange_correlation_potential_g);
    }
};

} // namespace q3
