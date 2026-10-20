#pragma once

/**
 * MixerFactory — creates charge-density mixer instances by name.
 *
 * Supported mixers:
 *   - "broyden"  G-space quasi-Newton (default). Best for insulators and semiconductors.
 *   - "linear"   Simple linear (plain) mixing in G-space. Unconditionally stable;
 *                recommended for metallic US systems where Broyden may diverge.
 *   - "anderson" Real-space Pulay/DIIS mixing.
 *
 * All mixers accept mixing_parameter and history_depth (unused for linear).
 */

#include <memory>
#include <stdexcept>
#include <string>

#include "layer_4_computation/iterator/internal/mixing/real_space/anderson.hpp"
#include "layer_4_computation/iterator/internal/mixing/reciprocal_space/broyden.hpp"
#include "layer_4_computation/iterator/internal/mixing/reciprocal_space/linear.hpp"

namespace q3 {
namespace MixerFactory {

/// Create a charge density mixer by name ("broyden", "linear", or "anderson").
inline std::unique_ptr<Mixer>
create(const std::string& mode, Real mixing_parameter = 0.7, int history_depth = 8) {
    if (mode == "broyden") {
        return std::make_unique<BroydenMixer>(mixing_parameter, history_depth);
    }
    if (mode == "linear") {
        return std::make_unique<LinearMixer>(mixing_parameter);
    }
    if (mode == "anderson") {
        return std::make_unique<AndersonMixer>(mixing_parameter, history_depth);
    }
    throw std::runtime_error("Unknown mixer mode: '" + mode +
                             "'. Supported values: 'broyden', 'linear', 'anderson'.");
}

} // namespace MixerFactory
} // namespace q3
