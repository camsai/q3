#pragma once

// @generated from interpolation/results/quality/spread.json — DO NOT EDIT

#include <cstdint>
#include <vector>

#include <nlohmann/json.hpp>

namespace q3::generated {

/**
 * Statistics of the per-function spreads; present when a spread was evaluated.
 */
struct InterpolationResultsQualitySpread {
    double median_bohr_squared{};
    // The population variance of the spreads across functions: the published no-reference criterion.
    double variance_bohr_squared{};
    double maximum_bohr_squared{};
    // Zero-based indices of functions above five times the median.
    std::vector<int64_t> outliers;
    // Zero-based indices of functions whose spread exceeds ten square angstrom, whatever the others do.
    std::vector<int64_t> above_ten_angstrom_squared;
};

inline void to_json(nlohmann::json& json_output, const InterpolationResultsQualitySpread& data) {
    json_output = nlohmann::json::object();
    json_output["median_bohr_squared"] = data.median_bohr_squared;
    json_output["variance_bohr_squared"] = data.variance_bohr_squared;
    json_output["maximum_bohr_squared"] = data.maximum_bohr_squared;
    json_output["outliers"] = data.outliers;
    json_output["above_ten_angstrom_squared"] = data.above_ten_angstrom_squared;
}

inline void from_json(const nlohmann::json& json_input, InterpolationResultsQualitySpread& data) {
    json_input.at("median_bohr_squared").get_to(data.median_bohr_squared);
    json_input.at("variance_bohr_squared").get_to(data.variance_bohr_squared);
    json_input.at("maximum_bohr_squared").get_to(data.maximum_bohr_squared);
    json_input.at("outliers").get_to(data.outliers);
    json_input.at("above_ten_angstrom_squared").get_to(data.above_ten_angstrom_squared);
}

} // namespace q3::generated
