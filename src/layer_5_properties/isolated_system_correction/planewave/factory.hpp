#pragma once

/**
 * Factory for IsolatedSystemCorrectionCalculator implementations.
 *
 * Resolves the input's correction name (InputData::isolated_system_correction
 * / YAML `control.isolated_system_correction`) to a concrete scheme,
 * mirroring create_dispersion_calculator(). "none" (or empty) resolves to
 * NullIsolatedSystemCorrectionCalculator so call sites need no branching;
 * unknown names fail loudly rather than silently running without the
 * correction the user asked for (the YAML parser already validates the value,
 * so a throw here means a programmatic caller bypassed parsing). Adding a
 * scheme = one new class implementing IsolatedSystemCorrectionCalculator +
 * one case here.
 *
 * "martyna-tuckerman" resolves to MartynaTuckermanCorrectionCalculator, the
 * post-SCF (non-self-consistent) evaluation of the Martyna-Tuckerman
 * reciprocal-space correction -- see that class's docstring for the
 * important distinction from QE's self-consistent `assume_isolated='m-t'`.
 * Unlike Makov-Payne it has no lattice-type gate (valid for any Bravais
 * lattice, with a documented, warning-only locality heuristic instead).
 */

#include <memory>
#include <stdexcept>
#include <string>

#include "layer_5_properties/isolated_system_correction/planewave/init.hpp"
#include "layer_5_properties/isolated_system_correction/planewave/makov_payne.hpp"
#include "layer_5_properties/isolated_system_correction/planewave/martyna_tuckerman.hpp"
#include "layer_5_properties/isolated_system_correction/planewave/null.hpp"

namespace q3 {

[[nodiscard]] inline std::unique_ptr<IsolatedSystemCorrectionCalculator>
create_isolated_system_correction_calculator(const std::string& isolated_system_correction) {
    if (isolated_system_correction.empty() || isolated_system_correction == "none") {
        return std::make_unique<NullIsolatedSystemCorrectionCalculator>();
    }
    if (isolated_system_correction == "makov-payne") {
        return std::make_unique<MakovPayneCorrectionCalculator>();
    }
    if (isolated_system_correction == "martyna-tuckerman") {
        return std::make_unique<MartynaTuckermanCorrectionCalculator>();
    }
    throw std::runtime_error("Unknown isolated_system_correction '" + isolated_system_correction +
                             "'. Supported: none, makov-payne, martyna-tuckerman");
}

} // namespace q3
