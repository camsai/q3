#pragma once

// @generated from interpolation/results/construction/disentanglement/bands_excluded_from_freeze.json — DO NOT EDIT

#include <cstdint>

#include <nlohmann/json.hpp>

namespace q3::generated {

/**
 */
struct InterpolationResultsExcludedBand {
    // The band, one-based as the manifest counts them.
    int64_t band{};
    // At how many k-points it was excluded.
    int64_t kpoints{};
};

inline void to_json(nlohmann::json& json_output, const InterpolationResultsExcludedBand& data) {
    json_output = nlohmann::json::object();
    json_output["band"] = data.band;
    json_output["kpoints"] = data.kpoints;
}

inline void from_json(const nlohmann::json& json_input, InterpolationResultsExcludedBand& data) {
    json_input.at("band").get_to(data.band);
    json_input.at("kpoints").get_to(data.kpoints);
}

} // namespace q3::generated
