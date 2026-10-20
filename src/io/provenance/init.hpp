#pragma once

/**
 * QuantumSystemProvenanceRecord — what a results file says about how it was
 * produced, for the quantum system that was solved.
 *
 * A total energy without its provenance is a number, not a measurement: it
 * cannot be reproduced, compared against another engine, or registered, because
 * nothing in it says which code, which physical model, or which numerics
 * produced it. This record is what makes a results file self-describing.
 *
 * What is here is what every quantum system has, whatever it is represented in:
 * who computed it (generator), which Hamiltonian was solved (model), the
 * numerics that solved it (method), and the convergence that was asked for
 * (precision). What the run *achieved* is not here -- convergence and results
 * are the rest of the results file.
 *
 * How the system was *represented* -- plane waves and pseudopotentials, or
 * whatever else a future representation brings -- is deliberately not known at
 * this level: a basis-set cutoff or a pseudopotential dataset is a fact about a
 * representation, not about a quantum system, and a record that names one here
 * would have to name every other one here too. A representation reports itself
 * by deriving and overriding add_representation_method_details():
 *
 *   QuantumSystemProvenanceRecord        generator, model, k-points, precision
 *     -> PlanewaveProvenanceRecord       + the basis-set cutoffs
 *          -> PseudopotentialProvenanceRecord  + the per-species datasets
 *
 * (provenance/init.hpp, provenance/planewave/init.hpp,
 * provenance/planewave/pseudopotential.hpp respectively; provenance/factory.hpp
 * picks the one a given run needs.) Adding a representation therefore means
 * adding a subclass beside those, not another branch here.
 *
 * Field names follow the provenance schema of the registry that consumes these
 * files, so its reader maps rather than guesses. Two deliberate divergences,
 * both about units: that schema states energies in electronvolt, while Q3 works
 * and quotes in Rydberg. Rather than convert here -- which would report a
 * threshold the user never typed -- the fields that would differ carry an
 * explicit `_rydberg` suffix where the consumer's carry `_electronvolt`. A
 * reader cannot then take a Rydberg number for an electronvolt one, which a
 * shared name would have invited.
 *
 * The record is built once per run, after the calculation, from the same
 * InputData the calculation consumed -- which it borrows rather than copies, so
 * a record must not outlive the input it reports on.
 */

#include <string>

#include <nlohmann/json.hpp>

#include "io/input/data.hpp"
#include "layer_0_core/addon_identity.hpp"
#include "layer_0_core/version.hpp"

namespace q3 {

/// Map the number of spin components to the consumer schema's spin treatment.
[[nodiscard]] inline std::string spin_treatment_name(int number_of_spin_components) {
    if (number_of_spin_components == 4) {
        return "non_collinear";
    }
    if (number_of_spin_components == 2) {
        return "collinear";
    }
    return "unpolarized";
}

class QuantumSystemProvenanceRecord {
  public:
    explicit QuantumSystemProvenanceRecord(const InputData& input) : input_(input) {}
    virtual ~QuantumSystemProvenanceRecord() = default;

    /// The provenance block, grouped the way the consumer's record is.
    [[nodiscard]] nlohmann::json to_json() const {
        nlohmann::json provenance;
        provenance["generator"] = generator_json();
        provenance["model"] = model_json();
        provenance["method"] = method_json();
        provenance["precision"] = precision_json();
        if (input_.is_relaxation_step) {
            provenance["relaxation"] = {
                {"force_threshold_ry_per_bohr", input_.relax_force_threshold},
                {"energy_threshold_ry", input_.relax_energy_threshold},
                {"maximum_steps", input_.relax_maximum_steps}};
        }
        return provenance;
    }

  protected:
    /// The input the calculation consumed, for a subclass to report on in turn.
    [[nodiscard]] const InputData& input() const {
        return input_;
    }

    /**
     * Hook: add what the concrete representation contributes to "method".
     *
     * The base record describes a quantum system that has been solved; it
     * cannot say what a basis set or a dataset was, because at this level there
     * is none. A subclass that knows one adds it here.
     */
    virtual void add_representation_method_details(nlohmann::json& /*method*/) const {}

  private:
    [[nodiscard]] nlohmann::json generator_json() const {
        nlohmann::json generator;
        generator["engine_name"] = "q3";
        generator["engine_version"] = version;
        generator["engine_commit"] = commit;
        /**
         * The build-time add-ons compiled in, since a result can depend on one;
         * an empty list for a core-only build, so absence is stated rather
         * than inferred from a missing key.
         */
        nlohmann::json addons = nlohmann::json::array();
        for (const AddonIdentity& addon : active_addons())
            addons.push_back({{"name", addon.name}, {"version", addon.version}});
        generator["addons"] = addons;
        return generator;
    }

    [[nodiscard]] nlohmann::json model_json() const {
        nlohmann::json model;
        model["exchange_correlation_functional"] = input_.exchange_correlation_functional;
        model["spin_treatment"] = spin_treatment_name(input_.number_of_spin_components);
        if (input_.dispersion_correction != "none") {
            model["dispersion_correction"] = input_.dispersion_correction;
        }
        if (!input_.hubbard.empty()) {
            nlohmann::json hubbard_by_species = nlohmann::json::object();
            for (const HubbardSpecification& specification : input_.hubbard) {
                hubbard_by_species[specification.species_symbol + "-" +
                                   specification.orbital_label] = specification.u_rydberg;
            }
            model["hubbard_u_rydberg_by_species"] = hubbard_by_species;
        }
        return model;
    }

    [[nodiscard]] nlohmann::json method_json() const {
        nlohmann::json method;
        method["kpoint_grid"] = {input_.kpoint_grid_dimension_1,
                                 input_.kpoint_grid_dimension_2,
                                 input_.kpoint_grid_dimension_3};
        method["kpoint_shift"] = {
            input_.kpoint_grid_shift_1, input_.kpoint_grid_shift_2, input_.kpoint_grid_shift_3};
        const bool uses_smearing = (input_.smearing != "none");
        method["electronic_occupation"] = uses_smearing ? "smearing" : "fixed";
        if (uses_smearing) {
            method["smearing_kind"] = input_.smearing;
            method["smearing_width_rydberg"] = input_.smearing_width;
        }
        add_representation_method_details(method);
        return method;
    }

    [[nodiscard]] nlohmann::json precision_json() const {
        nlohmann::json precision;
        precision["energy_convergence_threshold_rydberg"] = input_.convergence_threshold;
        precision["maximum_self_consistent_field_iterations"] = input_.maximum_iterations;
        return precision;
    }

    const InputData& input_;
};

} // namespace q3
