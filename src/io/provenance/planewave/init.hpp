#pragma once

/**
 * PlanewaveProvenanceRecord — a provenance record for a quantum system that was
 * solved in a plane-wave basis.
 *
 * The basis set is the first thing about a run that is not a fact about the
 * quantum system itself: an energy cutoff means nothing without plane waves to
 * bound. So the two cutoffs enter the record here rather than in
 * QuantumSystemProvenanceRecord, and a future record for a basis Q3 does not
 * have yet inherits everything above this without inheriting them.
 *
 * The datasets that fill that basis are one level further down, in
 * planewave/pseudopotential.hpp: a plane-wave calculation need not be a
 * pseudopotential one.
 *
 * Both cutoffs are reported in Rydberg, with the suffix that says so -- see the
 * units note in provenance/init.hpp.
 */

#include <nlohmann/json.hpp>

#include "io/provenance/init.hpp"

namespace q3 {

class PlanewaveProvenanceRecord : public QuantumSystemProvenanceRecord {
  public:
    using QuantumSystemProvenanceRecord::QuantumSystemProvenanceRecord;

  protected:
    void add_representation_method_details(nlohmann::json& method) const override {
        method["wavefunction_cutoff_rydberg"] = input().wavefunction_energy_cutoff;
        method["density_cutoff_rydberg"] = input().charge_density_energy_cutoff;
    }
};

} // namespace q3
