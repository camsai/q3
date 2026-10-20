#pragma once

/**
 * I/O utility functions.
 *
 * General-purpose file and path utilities used across the I/O subsystem:
 * path resolution, JSON file reading.
 */

#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

#include <nlohmann/json.hpp>

namespace q3 {
namespace detail {

/**
 * Resolve a path relative to a given directory.
 * Returns the path unchanged if it is already absolute.
 */
inline std::string resolve_path(const std::string& base_directory, const std::string& path) {
    std::filesystem::path path_object(path);
    if (path_object.is_absolute())
        return path;
    return (std::filesystem::path(base_directory) / path_object).string();
}

/**
 * Read a JSON file and return the parsed object.
 */
inline nlohmann::json read_json_file(const std::string& path) {
    std::ifstream file_stream(path);
    if (!file_stream.is_open())
        throw std::runtime_error("Cannot open JSON file: " + path);
    return nlohmann::json::parse(file_stream);
}

} // namespace detail
} // namespace q3
