#pragma once

/**
 * Exchange-correlation stress tensor contribution (base class).
 *
 * Computes the diagonal term (from the difference between XC energy and
 * potential energy), which is the complete XC stress for a local functional,
 * then delegates any gradient (GGA) term to the strategy resolved by
 * create_exchange_correlation_gradient_stress_calculator() — the Null object
 * for local functionals — so compute() never branches on the functional's
 * capabilities.
 *
 * The densities the gradient term consumes are provided through the
 * ExchangeCorrelationStressDensitySource interface this class implements;
 * subclasses override the two builders (e.g., to include NLCC core charge
 * from pseudopotentials), keeping the pseudopotential axis independent of
 * the functional axis.
 *
 * This base class has no pseudopotential dependency.
 */

#include <vector>

#include "layer_1_structure/crystal/crystal.hpp"
#include "layer_3_representation/basis/planewave/init.hpp"
#include "layer_4_computation/effective_potential/internal/exchange_correlation/init.hpp"
#include "layer_4_computation/state/kohn_sham/planewave/norm_conserving/init.hpp"
#include "layer_5_properties/stress_tensor/planewave/contributions.hpp"
#include "layer_5_properties/stress_tensor/planewave/internal/exchange_correlation_gradient/factory.hpp"
#include "layer_5_properties/stress_tensor/planewave/internal/exchange_correlation_gradient/init.hpp"

namespace q3 {

class ExchangeCorrelationStressTensorCalculator : public ExchangeCorrelationStressDensitySource {
  public:
    ExchangeCorrelationStressTensorCalculator(
        const Crystal& crystal_structure,
        const PlanewaveBasis& planewave_basis,
        const NormConservingPlanewaveKohnShamSCFState& scf_state,
        StressTensorContributions& result)
        : crystal_structure_(crystal_structure)
        , planewave_basis_(planewave_basis)
        , scf_state_(scf_state)
        , result_(result) {}

    virtual ~ExchangeCorrelationStressTensorCalculator() = default;

    void compute() {
        Real exchange_correlation_energy = scf_state_.energies().exchange_correlation_energy;
        Real exchange_correlation_potential_energy =
            scf_state_.energies().exchange_correlation_potential_energy;
        Real exchange_correlation_diagonal =
            -(exchange_correlation_energy - exchange_correlation_potential_energy) /
            crystal_structure_.cell_volume();
        for (int direction_a = 0; direction_a < 3; direction_a++) {
            result_.exchange_correlation[direction_a][direction_a] = exchange_correlation_diagonal;
        }

        /**
         * Any gradient (GGA) term is the strategy's business: local
         * functionals resolve to the Null object, so there is no branching on
         * the functional's capabilities here.
         */
        create_exchange_correlation_gradient_stress_calculator(*scf_state_.exchange_correlation,
                                                               crystal_structure_,
                                                               planewave_basis_,
                                                               scf_state_,
                                                               *this,
                                                               result_)
            ->add_gradient_stress();
    }

  protected:
    /**
     * Build the total charge density in G-space for GGA gradient corrections.
     * Base implementation returns the valence charge density from the SCF.
     * Subclasses override to include additional contributions (e.g., NLCC core charge).
     * Declared public on ExchangeCorrelationStressDensitySource, which is how the
     * gradient-stress strategy reaches them; protected here keeps them out of this
     * class's own public surface.
     */
    [[nodiscard]] std::vector<Complex> build_total_charge_density_g() const override {
        int number_of_g_vectors = planewave_basis_.number_of_g_vectors();
        std::vector<Complex> total_charge_density_g(number_of_g_vectors);
        for (int g_vector_index = 0; g_vector_index < number_of_g_vectors; g_vector_index++) {
            total_charge_density_g[g_vector_index] = scf_state_.charge_density_g[g_vector_index];
        }
        return total_charge_density_g;
    }

    /**
     * Build the per-spin charge density in G-space for spin-polarized (LSDA) GGA
     * gradient corrections. Base implementation FFTs the valence spin channel from
     * the SCF state. Subclasses override to split additional contributions evenly
     * across spins (e.g., NLCC core charge, which carries no net moment).
     */
    [[nodiscard]] std::vector<Complex> build_spin_charge_density_g(int spin_index) const override {
        std::vector<Complex> density_g;
        planewave_basis_.rho_r_to_g(scf_state_.charge_density_real[spin_index], density_g);
        return density_g;
    }

    const Crystal& crystal_structure_;
    const PlanewaveBasis& planewave_basis_;
    const NormConservingPlanewaveKohnShamSCFState& scf_state_;
    StressTensorContributions& result_;
};

} // namespace q3
