#pragma once

/**
 * back_up_existing_file — move a file aside before it is overwritten, so a
 * second run into the same output directory keeps the first run's file.
 *
 * `<path>` is renamed to `<path>-<number>`, the lowest number from 1 not
 * already taken: si_results.json goes to si_results.json-1, then
 * si_results.json-2, and so on -- the naming rupy's safe_path_generator uses.
 */

#include <filesystem>
#include <optional>
#include <string>

#include "io/logger.hpp"

namespace q3 {

/// The name `path` is backed up under as backup number `backup_number`: `<path>-<number>`.
[[nodiscard]] inline std::filesystem::path backup_path(const std::filesystem::path& path,
                                                       int backup_number) {
    return std::filesystem::path(path.string() + "-" + std::to_string(backup_number));
}

/**
 * Move `path` to its first unused backup name if it exists, and say where.
 * Returns the backup's path, or nothing when there was no file to keep.
 */
inline std::optional<std::filesystem::path>
back_up_existing_file(const std::filesystem::path& path) {
    if (!std::filesystem::is_regular_file(path)) {
        return std::nullopt;
    }
    int backup_number = 1;
    while (std::filesystem::exists(backup_path(path, backup_number))) {
        ++backup_number;
    }
    const std::filesystem::path backup = backup_path(path, backup_number);
    std::filesystem::rename(path, backup);
    log::info("     Previous results kept as: {}", backup.string());
    return backup;
}

} // namespace q3
