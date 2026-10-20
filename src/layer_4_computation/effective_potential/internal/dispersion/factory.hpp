#pragma once

/**
 * Factory for DispersionCalculator implementations.
 *
 * Resolves the input's dispersion-correction name (InputData::
 * dispersion_correction / YAML `model.dispersion`) to a concrete scheme,
 * mirroring the mixing and smearing factories. "none" (or empty) resolves to
 * NullDispersionCalculator so call sites need no branching; unknown names
 * fail loudly rather than silently running without the correction the user
 * asked for. Adding a scheme = one new class implementing
 * DispersionCalculator + one case here.
 */

#include <memory>
#include <stdexcept>
#include <string>

#include "layer_1_structure/crystal/crystal.hpp"
#include "layer_4_computation/effective_potential/internal/dispersion/d3bj.hpp"
#include "layer_4_computation/effective_potential/internal/dispersion/init.hpp"
#include "layer_4_computation/effective_potential/internal/dispersion/null.hpp"

namespace q3 {

[[nodiscard]] inline std::unique_ptr<DispersionCalculator>
create_dispersion_calculator(const std::string& dispersion_correction,
                             const Crystal& crystal_structure) {
    if (dispersion_correction.empty() || dispersion_correction == "none") {
        return std::make_unique<NullDispersionCalculator>(crystal_structure);
    }
    if (dispersion_correction == "d3bj") {
        return std::make_unique<D3BJDispersionCalculator>(crystal_structure);
    }
    throw std::runtime_error("Unknown dispersion correction '" + dispersion_correction +
                             "'. Supported: none, d3bj");
}

} // namespace q3
