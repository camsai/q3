#pragma once

/**
 * create_smearing — factory function for occupation smearing schemes.
 *
 * Supported schemes: "fixed"/"none" (insulating, step function),
 * "gaussian"/"gauss" (metallic, erfc smearing with configurable width),
 * "methfessel-paxton"/"m-p"/"mp" (Hermite correction of order
 * `methfessel_paxton_order`, default 1; ignored by the other schemes),
 * "marzari-vanderbilt"/"cold"/"m-v"/"mv" (cold smearing), and
 * "fermi-dirac"/"f-d"/"fd" (physical finite-T occupation). Aliases match
 * Quantum ESPRESSO's `smearing` namelist values so QE input files translate
 * directly without a separate name-mapping layer.
 */

#include <memory>
#include <stdexcept>
#include <string>

#include "layer_4_computation/iterator/internal/smearing/fermi_dirac.hpp"
#include "layer_4_computation/iterator/internal/smearing/fixed.hpp"
#include "layer_4_computation/iterator/internal/smearing/gaussian.hpp"
#include "layer_4_computation/iterator/internal/smearing/marzari_vanderbilt.hpp"
#include "layer_4_computation/iterator/internal/smearing/methfessel_paxton.hpp"

namespace q3 {

/// Factory for smearing schemes.
inline std::unique_ptr<SmearingScheme> create_smearing(const std::string& name,
                                                       Real smearing_width = 0.0,
                                                       int methfessel_paxton_order = 1) {
    if (name == "fixed" || name == "none" || name.empty()) {
        return std::make_unique<FixedOccupation>();
    }
    if (name == "gaussian" || name == "gauss") {
        return std::make_unique<GaussianSmearing>(smearing_width);
    }
    if (name == "methfessel-paxton" || name == "m-p" || name == "mp") {
        return std::make_unique<MethfesselPaxtonSmearing>(smearing_width, methfessel_paxton_order);
    }
    if (name == "marzari-vanderbilt" || name == "cold" || name == "m-v" || name == "mv") {
        return std::make_unique<MarzariVanderbiltSmearing>(smearing_width);
    }
    if (name == "fermi-dirac" || name == "f-d" || name == "fd") {
        return std::make_unique<FermiDiracSmearing>(smearing_width);
    }
    throw std::runtime_error(
        "Unknown smearing: " + name +
        ". Options: fixed, gaussian, methfessel-paxton, marzari-vanderbilt, fermi-dirac");
}

} // namespace q3
