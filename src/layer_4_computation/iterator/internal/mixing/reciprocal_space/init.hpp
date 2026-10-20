#pragma once

/**
 * ReciprocalSpaceMixer — base class for mixers operating in G-space.
 *
 * Transparently wraps the real-space mix() interface: densities are FFT'd
 * to G-space, mixed via the virtual mix_in_reciprocal_space() hook, then
 * transformed back. Concrete implementation: BroydenMixer.
 */

#include <memory>
#include <utility>
#include <vector>

#include "layer_4_computation/iterator/internal/mixing/init.hpp"

namespace q3 {

/**
 * Mixer operating on G-space charge density vectors with a precomputed metric.
 *
 * The unified real-space mix() transforms densities to reciprocal space, defers
 * to the concrete mix_in_reciprocal_space() update, and transforms back, so the
 * caller never has to know the mixer works in G-space.
 */
class ReciprocalSpaceMixer : public Mixer {
  public:
    using Mixer::Mixer;

    void initialize(const MixerEnvironment& environment) override {
        to_reciprocal_space_ = environment.to_reciprocal_space;
        to_real_space_ = environment.to_real_space;
        charge_metric_weights_ = environment.charge_metric_weights;
        preconditioner_ = environment.preconditioner;
        magnetization_metric_weight_ = environment.magnetization_metric_weight;
        first_nonzero_g_index_ = environment.first_nonzero_g_index;
        is_gamma_only_ = environment.is_gamma_only;
        number_of_smooth_g_vectors_ = environment.number_of_smooth_g_vectors;
    }

    void mix(const std::vector<std::vector<Real>>& input_density,
             std::vector<std::vector<Real>>& output_density,
             int number_of_spin_components) override {
        /**
         * A density-dependent preconditioner needs the TOTAL real-space density,
         * and it needs it here: mix() receives (ρ↑, ρ↓), and the rotation to
         * (total, magnetization) happens further down in G-space, so the sum
         * over spin channels has to be done explicitly before the transforms.
         */
        if (preconditioner_ != nullptr) {
            preconditioner_->update(
                sum_over_spin_channels(input_density, number_of_spin_components));
        }

        std::vector<std::vector<Complex>> input_density_g(number_of_spin_components);
        std::vector<std::vector<Complex>> output_density_g(number_of_spin_components);
        for (int spin_index = 0; spin_index < number_of_spin_components; ++spin_index) {
            to_reciprocal_space_(input_density[spin_index], input_density_g[spin_index]);
            to_reciprocal_space_(output_density[spin_index], output_density_g[spin_index]);
        }

        /**
         * Spin-polarized densities are mixed in the (total, magnetization)
         * representation, the one that diagonalizes the mixing metric: channel
         * 0 = ρ↑+ρ↓ carries the whole Coulomb response, so it gets the 1/G²
         * metric and Kerker preconditioning; channel 1 = ρ↑−ρ↓ has no Coulomb
         * kernel, so it gets the flat metric and no preconditioning. The
         * per-channel metric code in the concrete mixers assumes this layout.
         */
        if (number_of_spin_components == 2) {
            convert_up_down_to_total_magnetization(input_density_g);
            convert_up_down_to_total_magnetization(output_density_g);
        }

        mix_in_reciprocal_space(input_density_g, output_density_g, number_of_spin_components);

        /**
         * For dual-grid (ultrasoft/PAW) calculations, the hard G-vectors (above the smooth
         * cutoff) carry the augmentation charges. They sit outside the Broyden subspace, so
         * they need an update of their own; plain linear mixing at the configured
         * mixing_beta is the cheapest stable one. Q3 was replacing them at 100% (beta=1.0),
         * which caused augmentation oscillations and Broyden divergence.
         * Fix: linear-mix hard G with the same mixing_parameter_ used for the smooth region.
         */
        if (number_of_smooth_g_vectors_ > 0) {
            for (int spin_index = 0; spin_index < number_of_spin_components; ++spin_index) {
                int number_of_g_vectors = static_cast<int>(input_density_g[spin_index].size());
                for (int g_index = number_of_smooth_g_vectors_; g_index < number_of_g_vectors;
                     ++g_index) {
                    input_density_g[spin_index][g_index] +=
                        mixing_parameter_ * (output_density_g[spin_index][g_index] -
                                             input_density_g[spin_index][g_index]);
                }
            }
        }

        if (number_of_spin_components == 2) {
            convert_total_magnetization_to_up_down(input_density_g);
        }

        for (int spin_index = 0; spin_index < number_of_spin_components; ++spin_index) {
            to_real_space_(input_density_g[spin_index], output_density[spin_index]);
        }
    }

    /// Total real-space density ρ↑+ρ↓ (a copy of channel 0 when unpolarized).
    static std::vector<Real>
    sum_over_spin_channels(const std::vector<std::vector<Real>>& density_real,
                           int number_of_spin_components) {
        std::vector<Real> total = density_real[0];
        for (int spin_index = 1; spin_index < number_of_spin_components; ++spin_index) {
            for (std::size_t index = 0; index < total.size(); ++index) {
                total[index] += density_real[spin_index][index];
            }
        }
        return total;
    }

    /// In-place (ρ↑, ρ↓) → (ρ↑+ρ↓, ρ↑−ρ↓).
    static void
    convert_up_down_to_total_magnetization(std::vector<std::vector<Complex>>& density_g) {
        for (std::size_t g_index = 0; g_index < density_g[0].size(); ++g_index) {
            Complex up = density_g[0][g_index];
            Complex down = density_g[1][g_index];
            density_g[0][g_index] = up + down;
            density_g[1][g_index] = up - down;
        }
    }

    /// In-place (ρ↑+ρ↓, ρ↑−ρ↓) → (ρ↑, ρ↓).
    static void
    convert_total_magnetization_to_up_down(std::vector<std::vector<Complex>>& density_g) {
        for (std::size_t g_index = 0; g_index < density_g[0].size(); ++g_index) {
            Complex total = density_g[0][g_index];
            Complex magnetization = density_g[1][g_index];
            density_g[0][g_index] = 0.5 * (total + magnetization);
            density_g[1][g_index] = 0.5 * (total - magnetization);
        }
    }

    /**
     * Perform one mixing step in G-space.
     * @param input_density_g  input density in G-space (modified in-place with the mixed result).
     * @param output_density_g output density in G-space.
     * @param number_of_spin_components number of spin components.
     */
    virtual void mix_in_reciprocal_space(std::vector<std::vector<Complex>>& input_density_g,
                                         const std::vector<std::vector<Complex>>& output_density_g,
                                         int number_of_spin_components) = 0;

  protected:
    /**
     * Weighted inner product ⟨ρ_a|ρ_b⟩ in the two-channel mixing metric -- the same
     * metric the SCF reports its convergence estimate in, so that a conv_thr value
     * carried in from a plane-wave input means the same thing here as there:
     *
     *   channel 0 (total charge):  Σ_{G≠0} w_charge(G) ρ_a*(G) ρ_b(G),
     *                              w_charge = Hartree kernel ∝ 1/G²
     *   channel 1 (magnetization): Σ_{all G, incl. G=0} w_m ρ_a*(G) ρ_b(G),
     *                              w_m = flat metric (no Coulomb kernel)
     *
     * On gamma-only half-sphere grids every G≠0 term is doubled (ρ(-G) = ρ*(G));
     * the G=0 magnetization term is stored once and is not doubled.
     *
     * @param g_vector_count  Limit the G range (e.g. smooth-only mixing);
     *                        pass a negative value to use the full vectors.
     */
    Real compute_metric_dot_product(const std::vector<std::vector<Complex>>& rho_a,
                                    const std::vector<std::vector<Complex>>& rho_b,
                                    int number_of_spin_components,
                                    int g_vector_count = -1) const {
        int g_count = (g_vector_count >= 0) ? g_vector_count : static_cast<int>(rho_a[0].size());
        Real gamma_factor = is_gamma_only_ ? 2.0 : 1.0;
        Real result = 0.0;

        /**
         * Total charge channel: Hartree 1/G² metric, G=0 excluded (fixed by
         * charge conservation, and the kernel diverges there).
         */
        Real charge_result = 0.0;
        for (int g_index = first_nonzero_g_index_; g_index < g_count; g_index++) {
            charge_result += charge_metric_weights_[g_index] *
                             std::real(std::conj(rho_a[0][g_index]) * rho_b[0][g_index]);
        }
        result += charge_result * gamma_factor;

        /**
         * Magnetization channels: flat metric including G=0 (the total moment
         * is not conserved and must contribute to the metric).
         */
        for (int spin_index = 1; spin_index < number_of_spin_components; spin_index++) {
            for (int g_index = 0; g_index < g_count; g_index++) {
                Real g_factor = (g_index < first_nonzero_g_index_) ? 1.0 : gamma_factor;
                result +=
                    magnetization_metric_weight_ * g_factor *
                    std::real(std::conj(rho_a[spin_index][g_index]) * rho_b[spin_index][g_index]);
            }
        }

        return result;
    }

    std::function<void(const std::vector<Real>&, std::vector<Complex>&)> to_reciprocal_space_;
    std::function<void(const std::vector<Complex>&, std::vector<Real>&)> to_real_space_;
    std::vector<Real> charge_metric_weights_;
    std::shared_ptr<Preconditioner> preconditioner_;
    Real magnetization_metric_weight_{0.0};
    int first_nonzero_g_index_{0};
    bool is_gamma_only_{false};
    int number_of_smooth_g_vectors_{0};
};

} // namespace q3
