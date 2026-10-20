#pragma once

/**
 * The pseudopotential file formats this build can read.
 *
 * Core seeds the registry with UPF on first use. A build-time add-on registers
 * its formats from its registration function, which every entry point runs
 * before it reads input (io/addon_registration.hpp). Formats are consulted in
 * registration order, so core's UPF is always asked first.
 */

#include <memory>

#include "io/pseudopotential/pseudopotential_format.hpp"
#include "io/pseudopotential/upf_pseudopotential_format.hpp"
#include "utils/registry.hpp"

namespace q3 {

/// The registry PseudopotentialFactory::read() chooses a format from.
inline Registry<PseudopotentialFormat>& pseudopotential_format_registry() {
    static Registry<PseudopotentialFormat> registry = [] {
        Registry<PseudopotentialFormat> seeded;
        seeded.add(std::make_unique<UPFPseudopotentialFormat>());
        return seeded;
    }();
    return registry;
}

} // namespace q3
