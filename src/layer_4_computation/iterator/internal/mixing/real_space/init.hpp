#pragma once

/**
 * RealSpaceMixer — base class for mixers that operate on real-space densities.
 *
 * Dispatches per-spin mixing to the `mix_one_component()` virtual hook.
 * The concrete implementation (AndersonMixer) lives in anderson.hpp.
 */

#include <functional>
#include <vector>

#include "layer_4_computation/iterator/internal/mixing/init.hpp"

namespace q3 {

/// Mixer operating on real-space charge density vectors.
class RealSpaceMixer : public Mixer {
  public:
    using Mixer::Mixer;

    void initialize(const MixerEnvironment& environment) override {
        global_reduce_ = environment.global_reduce;
    }

    void mix(const std::vector<std::vector<Real>>& input_density,
             std::vector<std::vector<Real>>& output_density,
             int number_of_spin_components) override {
        for (int spin_index = 0; spin_index < number_of_spin_components; ++spin_index) {
            output_density[spin_index] =
                mix_one_component(input_density[spin_index], output_density[spin_index]);
        }
    }

    /**
     * Mix one spin component to produce the next input density.
     * @param input_density  density used to build the potential (rho_in).
     * @param output_density density from diagonalization/sum_band (rho_out).
     * @return mixed density for the next iteration.
     */
    virtual std::vector<Real> mix_one_component(const std::vector<Real>& input_density,
                                                const std::vector<Real>& output_density) = 0;

    using Preconditioner = std::function<void(std::vector<Real>&)>;
    void set_preconditioner(Preconditioner preconditioner) {
        preconditioner_ = std::move(preconditioner);
    }
    void set_global_reduce(std::function<void(std::vector<Real>&)> global_reduce) {
        global_reduce_ = std::move(global_reduce);
    }

  protected:
    Preconditioner preconditioner_;
    std::function<void(std::vector<Real>&)> global_reduce_;
};

} // namespace q3
