#pragma once

// @generated from results.json — DO NOT EDIT

#include <optional>

#include <nlohmann/json.hpp>

#include "generated/q3/results/properties.hpp"
#include "generated/q3/results/runtime_log.hpp"
#include "generated/q3/results/structure.hpp"

namespace q3::generated {

/**
 * Q3 results.json — the single machine-readable record of a finished run, filled by every flavor as generated::Results (src/io/output/results.hpp), written as <prefix>_results.json and returned verbatim by the Python bindings, grouped by domain: structure (the material), provenance (how the result was produced), properties (the physical properties, ESSE records under properties.records) and runtime_log (how the calculation ran). Every object is closed (additionalProperties: false), so these schemas are the complete list of keys a reader may encounter, and a key added to the writer without being added here fails the regression suite. Every quantity carries its unit in its own units field; no key carries a unit in its name.
 *
 * Schema: results.json
 */
struct Results {
    // The structure the run used.
    std::optional<ResultsStructure> structure;
    // How this result was produced: engine build, the model/method/precision asked for, and the pseudopotential identities. Deliberately left open — it is metadata about the run rather than part of the physics record.
    std::optional<nlohmann::json> provenance;
    // The physical properties the run computed: every one ESSE has a property schema for as that ESSE record under records, and beside them the quantities ESSE has no property for yet, each an ESSE schema or one built from ESSE pieces carrying its own units.
    ResultsProperties properties;
    // How the calculation ran: which flavor, whether it converged and in how many iterations, the parallel topology and where the time went, and which derivatives are incomplete.
    ResultsRuntimeLog runtime_log;
};

inline void to_json(nlohmann::json& json_output, const Results& data) {
    json_output = nlohmann::json::object();
    if (data.structure) {
        json_output["structure"] = *data.structure;
    }
    if (data.provenance) {
        json_output["provenance"] = *data.provenance;
    }
    json_output["properties"] = data.properties;
    json_output["runtime_log"] = data.runtime_log;
}

inline void from_json(const nlohmann::json& json_input, Results& data) {
    if (json_input.contains("structure")) {
        data.structure = json_input.at("structure").get<ResultsStructure>();
    }
    if (json_input.contains("provenance")) {
        data.provenance = json_input.at("provenance").get<nlohmann::json>();
    }
    json_input.at("properties").get_to(data.properties);
    json_input.at("runtime_log").get_to(data.runtime_log);
}

} // namespace q3::generated
