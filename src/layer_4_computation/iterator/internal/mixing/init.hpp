#pragma once

/**
 * Mixer — abstract base class for charge-density mixing algorithms.
 *
 * Defines the interface for mixing input and output densities during
 * the self-consistent field cycle. Concrete implementations (Anderson,
 * Broyden) live in the real_space/ and reciprocal_space/ subdirectories.
 */

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "layer_0_core/types/init.hpp"
#include "layer_4_computation/iterator/internal/mixing/preconditioner/init.hpp"

namespace q3 {

/**
 * Everything a mixer might need from the surrounding representation.
 *
 * Real-space mixers ignore the reciprocal-space members; reciprocal-space
 * mixers use the transforms and metric weights. Passing a single struct lets
 * the iterator initialize any mixer without knowing its space.
 */
struct MixerEnvironment {
    std::function<void(const std::vector<Real>&, std::vector<Complex>&)> to_reciprocal_space;
    std::function<void(const std::vector<Complex>&, std::vector<Real>&)> to_real_space;
    std::vector<Real> charge_metric_weights;
    /**
     * Residual preconditioner, applied to the total-charge channel only. Never
     * null: "no preconditioning" is a NoPreconditioner, not an absent pointer,
     * so mixers call it unconditionally.
     */
    std::shared_ptr<Preconditioner> preconditioner;
    Real magnetization_metric_weight = 0.0;
    int first_nonzero_g_index = 0;
    bool is_gamma_only = false;
    int number_of_smooth_g_vectors = 0; // For dual-grid mixing (QE-style)
    std::function<void(std::vector<Real>&)> global_reduce;
};

/// Abstract base class for all charge density mixers.
class Mixer {
  public:
    Mixer(Real mixing_parameter, int history_depth)
        : mixing_parameter_(mixing_parameter), history_depth_(history_depth) {}
    virtual ~Mixer() = default;

    [[nodiscard]] virtual std::string name() const = 0;
    virtual void reset() {}

    /// Supply the representation context the mixer needs before the first mix().
    virtual void initialize(const MixerEnvironment&) {}

    /**
     * Mix one self-consistent field step using real-space densities.
     *
     * @param input_density            density used to build the potential (rho_in), per spin.
     * @param output_density           density from the bands (rho_out) on entry; overwritten
     *                                 with the mixed next-iteration density on exit, per spin.
     * @param number_of_spin_components number of spin channels.
     *
     * Each mixer performs whatever internal transforms it needs (e.g. FFT to
     * reciprocal space) so callers never branch on the mixer's space.
     */
    virtual void mix(const std::vector<std::vector<Real>>& input_density,
                     std::vector<std::vector<Real>>& output_density,
                     int number_of_spin_components) = 0;

    [[nodiscard]] Real mixing_parameter() const {
        return mixing_parameter_;
    }
    [[nodiscard]] int history_depth() const {
        return history_depth_;
    }

    /**
     * Return the density residual norm (dr2) computed during the last mix() call.
     *
     * Computing dr2 inside mix() avoids the FFT round-trip noise that occurs when
     * the SCF loop independently FFTs real-space densities back to G-space.
     * Returns -1.0 if not yet computed (callers should fall back to their own dr2).
     */
    [[nodiscard]] Real last_residual_norm() const {
        return last_residual_norm_;
    }

  protected:
    Real mixing_parameter_;
    int history_depth_;
    Real last_residual_norm_{-1.0};
};

} // namespace q3
