#pragma once

// @generated from interpolation/results/properties.json — DO NOT EDIT

#include <optional>

#include <nlohmann/json.hpp>

#include "generated/esse/properties/non_scalar/density_of_states.hpp"
#include "generated/esse/properties/non_scalar/eigenvalues_at_kpoints.hpp"
#include "generated/esse/properties/scalar/fermi_energy.hpp"

namespace q3::generated {

/**
 * What the model gave at the k-points it was evaluated at, grouped as the SCF results file groups its own: the Fermi level, the eigenvalues and the density of states, each carrying its unit in its own `units` field.
 */
struct InterpolationResultsProperties {
    FermiEnergy fermi_energy;
    // the band energies (and occupations) at each k-point of a calculation, in one energy unit
    std::optional<EigenvaluesAtKpoints> eigenvalues_at_kpoints;
    std::optional<DensityOfStates> density_of_states;
};

inline void to_json(nlohmann::json& json_output, const InterpolationResultsProperties& data) {
    json_output = nlohmann::json::object();
    json_output["fermi_energy"] = data.fermi_energy;
    if (data.eigenvalues_at_kpoints) {
        json_output["eigenvalues_at_kpoints"] = *data.eigenvalues_at_kpoints;
    }
    if (data.density_of_states) {
        json_output["density_of_states"] = *data.density_of_states;
    }
}

inline void from_json(const nlohmann::json& json_input, InterpolationResultsProperties& data) {
    json_input.at("fermi_energy").get_to(data.fermi_energy);
    if (json_input.contains("eigenvalues_at_kpoints")) {
        data.eigenvalues_at_kpoints = json_input.at("eigenvalues_at_kpoints").get<EigenvaluesAtKpoints>();
    }
    if (json_input.contains("density_of_states")) {
        data.density_of_states = json_input.at("density_of_states").get<DensityOfStates>();
    }
}

} // namespace q3::generated
