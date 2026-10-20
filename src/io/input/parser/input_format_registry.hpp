#pragma once

/**
 * The input formats this build can read, and how a path is matched to one.
 *
 * Core seeds the registry with YAML and Quantum ESPRESSO inputs on first use.
 * A build-time add-on registers its formats from its registration function,
 * which every entry point runs before it reads input (io/addon_registration.hpp),
 * so detection, `--format` and the usage text all see the same list.
 */

#include <filesystem>
#include <memory>

#include "io/input/parser/input_format.hpp"
#include "io/input/parser/qe_input_format.hpp"
#include "io/input/parser/yaml_input_format.hpp"
#include "utils/registry.hpp"

namespace q3 {

/// The registry every entry point reads input formats from.
inline Registry<InputFormat>& input_format_registry() {
    static Registry<InputFormat> registry = [] {
        Registry<InputFormat> seeded;
        seeded.add(std::make_unique<YAMLInputFormat>());
        seeded.add(std::make_unique<QEInputFormat>());
        return seeded;
    }();
    return registry;
}

/**
 * The format that claims `path`, or nullptr when none does.
 *
 * Every format that recognises the path is a candidate, and the one with the
 * highest detection_priority() wins. Ties go to the format registered first,
 * so detection does not depend on anything but the registry's contents.
 */
inline const InputFormat* detect_input_format(const std::filesystem::path& path) {
    const InputFormat* detected = nullptr;
    for (const auto& format : input_format_registry().entries()) {
        if (!format->recognizes(path))
            continue;
        if (detected == nullptr || format->detection_priority() > detected->detection_priority())
            detected = format.get();
    }
    return detected;
}

/**
 * The format whose input a run given no input path should read from
 * `directory`, or nullptr when no format claims it.
 */
inline const InputFormat* format_claiming_directory(const std::filesystem::path& directory) {
    for (const auto& format : input_format_registry().entries()) {
        if (format->claims_working_directory(directory))
            return format.get();
    }
    return nullptr;
}

} // namespace q3
