#pragma once

// @generated from interpolation/results/quality/hamiltonian_decay.json — DO NOT EDIT

#include <cstdint>

#include <nlohmann/json.hpp>

namespace q3::generated {

/**
 * How far H(R) has decayed by the edge of its Wigner-Seitz domain. A Bloch sum that has not decayed by its last term is the direct cause of error between mesh points.
 */
struct InterpolationResultsHamiltonianDecay {
    // |R| of the outermost shell of lattice vectors.
    double outermost_shell_distance_bohr{};
    // How many lattice vectors share that distance.
    int64_t outermost_shell_size{};
    // The largest Bloch-sum term w_R |H_mn(R)| on that shell.
    double outermost_shell_maximum_ry{};
    // The largest term that varies with k: any R away from the origin, or m != n on site, so a rigid shift of the energy zero cannot move it.
    double largest_hopping_ry{};
    // outermost_shell_maximum_ry / largest_hopping_ry; zero when there is nothing to divide by.
    double ratio{};
};

inline void to_json(nlohmann::json& json_output, const InterpolationResultsHamiltonianDecay& data) {
    json_output = nlohmann::json::object();
    json_output["outermost_shell_distance_bohr"] = data.outermost_shell_distance_bohr;
    json_output["outermost_shell_size"] = data.outermost_shell_size;
    json_output["outermost_shell_maximum_ry"] = data.outermost_shell_maximum_ry;
    json_output["largest_hopping_ry"] = data.largest_hopping_ry;
    json_output["ratio"] = data.ratio;
}

inline void from_json(const nlohmann::json& json_input, InterpolationResultsHamiltonianDecay& data) {
    json_input.at("outermost_shell_distance_bohr").get_to(data.outermost_shell_distance_bohr);
    json_input.at("outermost_shell_size").get_to(data.outermost_shell_size);
    json_input.at("outermost_shell_maximum_ry").get_to(data.outermost_shell_maximum_ry);
    json_input.at("largest_hopping_ry").get_to(data.largest_hopping_ry);
    json_input.at("ratio").get_to(data.ratio);
}

} // namespace q3::generated
