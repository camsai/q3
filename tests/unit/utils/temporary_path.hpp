#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <random>
#include <string>

namespace q3::test_support {

inline std::string make_temporary_path(const std::string& filename_ending = "") {
    static std::atomic<std::uint64_t> sequence_number{0};
    static std::random_device entropy_source;

    const auto timestamp = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    const auto sequence = sequence_number.fetch_add(1, std::memory_order_relaxed);
    const auto entropy = entropy_source();
    const auto filename = "q3_test_" + std::to_string(timestamp) + "_" + std::to_string(sequence) +
                          "_" + std::to_string(entropy) + filename_ending;
    return (std::filesystem::temp_directory_path() / filename).string();
}

} // namespace q3::test_support
