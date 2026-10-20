#pragma once

/**
 * The PAW one-center conventions this build can compute in.
 *
 * Keyed by the name a dataset records in PawPseudopotential::one_center_convention.
 * Core seeds the registry with "absolute", QE's convention and the one UPF-PAW
 * datasets follow. A build-time add-on that reads datasets generated for
 * another convention registers that convention from its registration function,
 * which every entry point runs before it reads input (io/addon_registration.hpp).
 */

#include <memory>
#include <string>
#include <vector>

#include "layer_4_computation/paw/one_center/convention/init.hpp"
#include "utils/named_factory.hpp"
#include "utils/registry.hpp"

namespace q3 {
namespace paw_one_center {

/// The registry create_one_center_convention() builds conventions from.
inline Registry<NamedFactory<PawOneCenterConvention>>& one_center_convention_registry() {
    static Registry<NamedFactory<PawOneCenterConvention>> registry = [] {
        Registry<NamedFactory<PawOneCenterConvention>> seeded;
        seeded.add(std::make_unique<NamedFactory<PawOneCenterConvention>>(
            std::vector<std::string>{"absolute"},
            [] { return std::make_unique<PawOneCenterConvention>(); }));
        return seeded;
    }();
    return registry;
}

} // namespace paw_one_center
} // namespace q3
