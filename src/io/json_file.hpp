#pragma once

/**
 * write_json_file — a JSON document to a path, the way every results file of
 * the engine and its utility is written: the parent directories created,
 * four-space indents, a trailing newline, and the path logged.
 */

#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

#include <nlohmann/json.hpp>

#include "io/logger.hpp"

namespace q3 {

inline void write_json_file(const std::string& path, const nlohmann::json& document) {
    const std::filesystem::path file_path(path);
    if (file_path.has_parent_path()) {
        std::filesystem::create_directories(file_path.parent_path());
    }
    std::ofstream stream(file_path);
    if (!stream.is_open()) {
        throw std::runtime_error("write_json_file: cannot open '" + path + "' for writing");
    }
    stream << document.dump(4) << "\n";
    log::info("     Results written to: {}", path);
}

} // namespace q3
