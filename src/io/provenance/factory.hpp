#pragma once

/**
 * create_provenance_record — pick the provenance record a Q3 run produces.
 *
 * The records under provenance/ each describe one level of what a run was:
 * a quantum system that was solved (QuantumSystemProvenanceRecord), solved in a
 * plane-wave basis (PlanewaveProvenanceRecord), whose potentials are
 * pseudopotentials (PseudopotentialProvenanceRecord). Which of them a given run
 * needs is a question about the run, and this is the single place that answers
 * it -- the same shape as create_input_parser() and create_mean_field_exporter().
 *
 * Today the answer is unconditional, because a Q3 calculation is a plane-wave
 * pseudopotential calculation and there is nothing yet to choose between. That
 * is deliberately still expressed as a factory rather than a hard-coded
 * construction at the call sites: when a second representation arrives, the
 * choice is made here, once, and Flavor::run(), the bindings and every other
 * caller keep working through build_provenance_json() without knowing that a
 * choice exists at all.
 *
 * A record borrows the InputData it reports on rather than copying it, so it
 * must not outlive that input -- which is why the common case has a function of
 * its own that builds the JSON and lets the record go.
 */

#include <memory>

#include <nlohmann/json.hpp>

#include "io/input/data.hpp"
#include "io/provenance/init.hpp"
#include "io/provenance/planewave/init.hpp"
#include "io/provenance/planewave/pseudopotential.hpp"

namespace q3 {

/// The provenance record for a run of this input. Borrows `input`.
[[nodiscard]] inline std::unique_ptr<QuantumSystemProvenanceRecord>
create_provenance_record(const InputData& input) {
    return std::make_unique<PseudopotentialProvenanceRecord>(input);
}

/// Build the provenance block for a completed run, from the input it consumed.
[[nodiscard]] inline nlohmann::json build_provenance_json(const InputData& input) {
    return create_provenance_record(input)->to_json();
}

} // namespace q3
