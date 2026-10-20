#pragma once

/**
 * Hand a finished run back to the input format it was read from.
 *
 * Some input formats come with conventions for what a finished run leaves
 * beside its results, and the pipelines built around them read those files.
 * The format that parsed the input knows which files they are, so every entry
 * point calls this once, after results.json is written, and the format writes
 * its own (InputFormat::write_companion_outputs). A run read from a format
 * with no such convention, or assembled in code, writes nothing more.
 */

#include "io/input/data.hpp"
#include "io/input/parser/input_format_registry.hpp"
#include "layer_4_computation/flavors/result.hpp"

namespace q3 {

/// Write the companion outputs of the format `input` was read from, if it has any.
inline void write_companion_outputs(const InputData& input, const FlavorResult& result) {
    if (const InputFormat* format = input_format_registry().find(input.source_format))
        format->write_companion_outputs(input, result);
}

} // namespace q3
