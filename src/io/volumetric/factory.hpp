#pragma once

/**
 * Factory for volumetric field writers.
 *
 * Writers are registered by format name. Core registers "cube" (Gaussian
 * cube, the default) and "xsf" (XCrySDen); a build-time add-on may register
 * more from its registration function, which every entry point runs before it
 * reads input (io/addon_registration.hpp).
 */

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "io/volumetric/cube.hpp"
#include "io/volumetric/writer.hpp"
#include "io/volumetric/xsf.hpp"
#include "utils/named_factory.hpp"
#include "utils/registry.hpp"

namespace q3 {

/// The registry create_volumetric_field_writer() builds writers from.
inline Registry<NamedFactory<VolumetricFieldWriter>>& volumetric_field_writer_registry() {
    static Registry<NamedFactory<VolumetricFieldWriter>> registry = [] {
        Registry<NamedFactory<VolumetricFieldWriter>> seeded;
        seeded.add(std::make_unique<NamedFactory<VolumetricFieldWriter>>(
            std::vector<std::string>{"cube"},
            [] { return std::make_unique<CubeVolumetricFieldWriter>(); }));
        seeded.add(std::make_unique<NamedFactory<VolumetricFieldWriter>>(
            std::vector<std::string>{"xsf"},
            [] { return std::make_unique<XsfVolumetricFieldWriter>(); }));
        return seeded;
    }();
    return registry;
}

inline std::unique_ptr<VolumetricFieldWriter>
create_volumetric_field_writer(const std::string& format) {
    const auto& registry = volumetric_field_writer_registry();
    if (const auto* factory = registry.find(format))
        return factory->create();
    throw std::runtime_error("Unknown volumetric format '" + format +
                             "'. Supported: " + registry.canonical_names(", "));
}

} // namespace q3
