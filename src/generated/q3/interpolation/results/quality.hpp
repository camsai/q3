#pragma once

// @generated from interpolation/results/quality.json — DO NOT EDIT

#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "generated/q3/interpolation/results/quality/band_distance.hpp"
#include "generated/q3/interpolation/results/quality/centers.hpp"
#include "generated/q3/interpolation/results/quality/hamiltonian_decay.hpp"
#include "generated/q3/interpolation/results/quality/ladder.hpp"
#include "generated/q3/interpolation/results/quality/spread.hpp"

namespace q3::generated {

/**
 * What the model says about itself with no reference calculation, and — added afterwards by q3.workflows.wannier, which runs the calculation they need — the band distance against a direct one and the verdict it earned.
 */
struct InterpolationResultsQuality {
    // Statistics of the per-function spreads; present when a spread was evaluated.
    std::optional<InterpolationResultsQualitySpread> spread;
    // The Wannier centres against the atoms of the manifest's material, over periodic images; present when a spread was evaluated and the manifest names atoms.
    std::optional<InterpolationResultsQualityCenters> centers;
    // How far H(R) has decayed by the edge of its Wigner-Seitz domain. A Bloch sum that has not decayed by its last term is the direct cause of error between mesh points.
    InterpolationResultsHamiltonianDecay hamiltonian_decay;
    // Repeated from spread.localization_converged when the gauge was minimised, so the block a driver judges by is complete on its own.
    std::optional<bool> localization_converged;
    // Added by q3.workflows.wannier: "accept" or "reject", the verdict of the retry ladder on this model.
    std::optional<std::string> status;
    // Added by q3.workflows.wannier: why the verdict was reached, in words.
    std::optional<std::string> reason;
    // Added by q3.workflows.wannier: every evaluation the retry ladder made of the saved record, in order, ending with the one this file describes.
    std::optional<std::vector<InterpolationResultsLadderAttempt>> ladder;
    // Added by q3.workflows.wannier: the band distance of this model against a direct calculation on the reference k-points.
    std::optional<InterpolationResultsBandDistance> band_distance;
};

inline void to_json(nlohmann::json& json_output, const InterpolationResultsQuality& data) {
    json_output = nlohmann::json::object();
    if (data.spread) {
        json_output["spread"] = *data.spread;
    }
    if (data.centers) {
        json_output["centers"] = *data.centers;
    }
    json_output["hamiltonian_decay"] = data.hamiltonian_decay;
    if (data.localization_converged) {
        json_output["localization_converged"] = *data.localization_converged;
    }
    if (data.status) {
        json_output["status"] = *data.status;
    }
    if (data.reason) {
        json_output["reason"] = *data.reason;
    }
    if (data.ladder) {
        json_output["ladder"] = *data.ladder;
    }
    if (data.band_distance) {
        json_output["band_distance"] = *data.band_distance;
    }
}

inline void from_json(const nlohmann::json& json_input, InterpolationResultsQuality& data) {
    if (json_input.contains("spread")) {
        data.spread = json_input.at("spread").get<InterpolationResultsQualitySpread>();
    }
    if (json_input.contains("centers")) {
        data.centers = json_input.at("centers").get<InterpolationResultsQualityCenters>();
    }
    json_input.at("hamiltonian_decay").get_to(data.hamiltonian_decay);
    if (json_input.contains("localization_converged")) {
        data.localization_converged = json_input.at("localization_converged").get<bool>();
    }
    if (json_input.contains("status")) {
        data.status = json_input.at("status").get<std::string>();
    }
    if (json_input.contains("reason")) {
        data.reason = json_input.at("reason").get<std::string>();
    }
    if (json_input.contains("ladder")) {
        data.ladder = json_input.at("ladder").get<std::vector<InterpolationResultsLadderAttempt>>();
    }
    if (json_input.contains("band_distance")) {
        data.band_distance = json_input.at("band_distance").get<InterpolationResultsBandDistance>();
    }
}

} // namespace q3::generated
