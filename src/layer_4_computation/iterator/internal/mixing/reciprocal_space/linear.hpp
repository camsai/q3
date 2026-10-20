#pragma once

/**
 * LinearMixer — simple linear (plain) charge density mixer.
 *
 * Implements the simplest possible mixing scheme:
 *   ρ_new = ρ_in + α · (ρ_out - ρ_in)
 *
 * Slower per-iteration than Broyden, but unconditionally stable for small
 * enough alpha, and it converges reliably for metallic systems where Broyden
 * may diverge due to charge sloshing.
 *
 * Kerker preconditioning (Thomas-Fermi screening) can be enabled by
 * supplying preconditioning_weights through the MixerEnvironment. When
 * enabled, the search direction is damped at small G:
 *   ρ_new = ρ_in + α · K(G) · (ρ_out - ρ_in),  K(G) = G²/(G²+q0²)
 */

#include <string>
#include <vector>

#include "layer_4_computation/iterator/internal/mixing/reciprocal_space/init.hpp"

namespace q3 {

/**
 * Simple linear density mixer operating in G-space with optional Kerker preconditioning.
 *
 * Convergence is guaranteed for sufficiently small alpha, unlike Broyden which
 * can diverge for difficult metallic systems.
 *
 * Not to be confused with mixing_mode='plain' in a Quantum ESPRESSO input:
 * there, 'plain' only turns the preconditioner off and the Broyden history
 * still runs, which is why io/input/parser/qe.hpp maps it to this code's
 * Broyden mixer rather than to this class.
 */
class LinearMixer : public ReciprocalSpaceMixer {
  public:
    explicit LinearMixer(Real mixing_parameter = 0.4) : ReciprocalSpaceMixer(mixing_parameter, 0) {}

    /**
     * Perform one linear mixing step in G-space.
     *
     * ρ_new = ρ_in + α · K(G) · (ρ_out - ρ_in)
     *
     * where K is the configured residual preconditioner (Kerker or local-TF if
     * enabled, the identity otherwise).
     */
    void mix_in_reciprocal_space(std::vector<std::vector<Complex>>& rho_in_g,
                                 const std::vector<std::vector<Complex>>& rho_out_g,
                                 int number_of_spin_components) override {
        int number_of_spin = static_cast<int>(rho_in_g.size());
        int number_of_g_vectors = static_cast<int>(rho_in_g[0].size());
        int smooth_g_vector_count =
            (number_of_smooth_g_vectors_ > 0 && number_of_smooth_g_vectors_ < number_of_g_vectors)
                ? number_of_smooth_g_vectors_
                : number_of_g_vectors;
        /**
         * Compute last_residual_norm_ BEFORE mixing using the pre-mix residual,
         * in the same shared two-channel metric Broyden reports -- see
         * ReciprocalSpaceMixer::compute_metric_dot_product().
         */
        std::vector<std::vector<Complex>> raw_residual(number_of_spin);
        for (int spin_index = 0; spin_index < number_of_spin; spin_index++) {
            raw_residual[spin_index].resize(smooth_g_vector_count);
            for (int g_index = 0; g_index < smooth_g_vector_count; g_index++) {
                raw_residual[spin_index][g_index] =
                    rho_out_g[spin_index][g_index] - rho_in_g[spin_index][g_index];
            }
        }
        last_residual_norm_ = compute_metric_dot_product(
            raw_residual, raw_residual, number_of_spin_components, smooth_g_vector_count);

        /**
         * Apply linear mixing: ρ_new = ρ_in + α · K(G) · (ρ_out - ρ_in)
         * K(G) = G²/(G²+q0²) for Kerker, 1 otherwise.
         * Only mix smooth G here. Hard G above smooth_g_vector_count are mixed
         * separately, by the hard-G loop in ReciprocalSpaceMixer::mix().
         */
        for (int spin_index = 0; spin_index < number_of_spin; spin_index++) {
            /**
             * Charge channel: G=0 is fixed by charge conservation, and the
             * preconditioner screens the Coulomb response. Magnetization channel
             * (spin 1 in the (total, m) representation): the moment is not
             * conserved, so its G=0 component must be mixed, and no screening
             * applies (no Coulomb kernel for m).
             */
            bool is_charge_channel = (spin_index == 0);
            int start_g_index = is_charge_channel ? first_nonzero_g_index_ : 0;

            /**
             * The preconditioner is an operator, so the residual is formed first
             * and preconditioned as a whole rather than scaled per G on the fly.
             * It is built over the full G range even though only the smooth
             * range is mixed here -- a non-diagonal preconditioner couples G
             * components, and truncating its input to the smooth range would
             * solve a different problem. For a diagonal preconditioner this is
             * exactly the previous per-G multiply.
             */
            std::vector<Complex> residual(rho_in_g[spin_index].size());
            for (std::size_t g_index = 0; g_index < residual.size(); ++g_index) {
                residual[g_index] = rho_out_g[spin_index][g_index] - rho_in_g[spin_index][g_index];
            }
            if (is_charge_channel && preconditioner_ != nullptr) {
                preconditioner_->apply(residual);
            }

            for (int g_index = start_g_index; g_index < smooth_g_vector_count; g_index++) {
                rho_in_g[spin_index][g_index] += mixing_parameter_ * residual[g_index];
            }
        }
    }

    [[nodiscard]] std::string name() const override {
        return "linear";
    }
};

} // namespace q3
