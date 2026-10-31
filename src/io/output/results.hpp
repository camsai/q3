#pragma once

/**
 * The results of a finished flavor run, as the struct generated from
 * schemas/q3/results.json (`generated::Results`).
 *
 * A flavor fills that struct directly: its ESSE records under
 * properties.records, the Q3 blocks beside them and the runtime log. There is
 * no second, hand-written copy of the results to convert from. What is here is
 * the little the generated struct cannot say for itself:
 *
 * - properties.records is a list of ESSE records of different types (the
 *   schema's oneOf), so the generated struct holds them as JSON;
 *   find_record() and set_record() read and write one by its generated type,
 *   as q3.io.results' record() does in Python;
 * - Q3's vectors and matrices, copied into the shapes ESSE lists them in;
 * - writing the file.
 */

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "generated/esse/core/reusable/energy.hpp"
#include "generated/esse/properties/non_scalar/stress_tensor.hpp"
#include "generated/esse/properties/non_scalar/total_energy_contributions.hpp"
#include "generated/esse/properties/scalar/fermi_energy.hpp"
#include "generated/esse/properties/scalar/total_energy.hpp"
#include "generated/esse/properties/structural/atomic_forces.hpp"
#include "generated/q3/results.hpp"
#include "io/json_file.hpp"
#include "layer_0_core/execution/threading.hpp"
#include "layer_0_core/types/init.hpp"
#include "layer_4_computation/state/internal/energies/kohn_sham/planewave/init.hpp"

namespace q3 {

/**
 * The results of a `flavor` run before it has computed anything: the runtime
 * log names the flavor, and the execution is one process with every thread
 * the run may use, until the execution policy's execution_record() states the real topology.
 */
[[nodiscard]] inline generated::Results new_results(const std::string& flavor) {
    generated::Results results;
    results.runtime_log.flavor = flavor;
    generated::ResultsExecution& execution = results.runtime_log.execution;
    execution.number_of_processes = 1;
    execution.number_of_threads_per_process = execution::threading::maximum_threads();
    execution.number_of_kpoint_pools = 1;
    execution.band_group_size = 1;
    return results;
}

/// The `name` every record of the generated ESSE type `Record` carries.
template <typename Record> [[nodiscard]] std::string record_name() {
    return nlohmann::json(Record{}.name).template get<std::string>();
}

/// The record of type `Record` the results list, or nothing when they list none.
template <typename Record>
[[nodiscard]] std::optional<Record> find_record(const generated::Results& results) {
    const std::string name = record_name<Record>();
    for (const auto& record : results.properties.records) {
        if (record.value("name", "") == name) {
            return record.template get<Record>();
        }
    }
    return std::nullopt;
}

/// List `record` in the results, in place of any record of the same name.
template <typename Record> void set_record(generated::Results& results, const Record& record) {
    const nlohmann::json value = record;
    const std::string name = record_name<Record>();
    auto& records = results.properties.records;
    for (auto& existing : records) {
        if (existing.value("name", "") == name) {
            existing = value;
            return;
        }
    }
    records.push_back(value);
}

/// A vector as ESSE lists one: [x, y, z].
[[nodiscard]] inline std::vector<double> to_vector(const Vector3D& vector) {
    return {vector.x, vector.y, vector.z};
}

/// A 3x3 matrix as ESSE lists one: three rows of three.
[[nodiscard]] inline std::vector<std::vector<double>> to_matrix(const Real (&matrix)[3][3]) {
    std::vector<std::vector<double>> rows;
    for (const auto& row : matrix) {
        rows.push_back({row[0], row[1], row[2]});
    }
    return rows;
}

/// One of Q3's energies that ESSE has no property for, as an ESSE energy in rydberg.
[[nodiscard]] inline generated::Energy energy(const std::string& name, Real value_in_ry) {
    return {name, generated::EnergyUnitsEnum::RY, value_in_ry};
}

/// Forces in Ry/bohr as ESSE's atomic_forces, atoms by 1-based id: the total or one contribution.
[[nodiscard]] inline generated::AtomicForces
atomic_forces(const std::vector<Vector3D>& values_in_ry_per_bohr) {
    generated::AtomicForces forces;
    forces.name = generated::AtomicForcesNameEnum::ATOMIC_FORCES;
    forces.units = generated::AtomicForcesUnitsEnum::RY_A_U;
    for (size_t atom_index = 0; atom_index < values_in_ry_per_bohr.size(); atom_index++) {
        forces.values.push_back(
            {to_vector(values_in_ry_per_bohr[atom_index]), static_cast<int64_t>(atom_index) + 1});
    }
    return forces;
}

/// A stress in kbar as ESSE's stress_tensor: the total or one contribution to it.
[[nodiscard]] inline generated::StressTensor stress_tensor(const Real (&values_in_kbar)[3][3]) {
    return {to_matrix(values_in_kbar),
            generated::StressTensorNameEnum::STRESS_TENSOR,
            generated::StressTensorUnitsEnum::KBAR};
}

/**
 * The run's energies: ESSE's total_energy record (`total_energy_in_ry`) and
 * total_energy_contributions record (the terms ESSE has keys for), and beside
 * them in energy_contributions the terms it has none for yet. The double
 * counting is always listed; hubbard (DFT+U, Dudarev), dispersion (already
 * inside the total) and one_center (PAW) only when nonzero. A flavor with no
 * energy of its own passes zeros, as a non-self-consistent pass does.
 */
inline void set_energy_records(generated::Results& results,
                               Real total_energy_in_ry,
                               const PlanewaveKohnShamEnergies& energies) {
    set_record(results,
               generated::TotalEnergy{generated::TotalEnergyNameEnum::TOTAL_ENERGY,
                                      generated::TotalEnergyUnitsEnum::RY,
                                      total_energy_in_ry});

    generated::TotalEnergyContributions contributions;
    contributions.name = generated::TotalEnergyContributionsNameEnum::TOTAL_ENERGY_CONTRIBUTIONS;
    contributions.units = generated::TotalEnergyContributionsUnitsEnum::RY;
    contributions.eigenvalues = generated::TotalEnergyContributionsEigenvalues{
        generated::TotalEnergyContributionsEigenvaluesNameEnum::EIGENVALUES, energies.band_energy};
    contributions.hartree = generated::TotalEnergyContributionsHartree{
        generated::TotalEnergyContributionsHartreeNameEnum::HARTREE, energies.hartree_energy};
    contributions.exchange_correlation = generated::TotalEnergyContributionsExchangeCorrelation{
        generated::TotalEnergyContributionsExchangeCorrelationNameEnum::EXCHANGE_CORRELATION,
        energies.exchange_correlation_energy};
    contributions.ewald = generated::TotalEnergyContributionsEwald{
        generated::TotalEnergyContributionsEwaldNameEnum::EWALD, energies.ewald_energy};
    contributions.temperature_entropy = generated::TotalEnergyContributionsTemperatureEntropy{
        generated::TotalEnergyContributionsTemperatureEntropyNameEnum::TEMPERATURE_ENTROPY,
        energies.smearing_entropy_contribution};
    set_record(results, contributions);

    generated::ResultsEnergyContributions& terms = results.properties.energy_contributions;
    terms.double_counting = energy("double_counting", energies.double_counting_correction);
    if (energies.hubbard_energy_contribution() != 0.0) {
        terms.hubbard = energy("hubbard", energies.hubbard_energy_contribution());
    }
    if (energies.dispersion_energy() != 0.0) {
        terms.dispersion = energy("dispersion", energies.dispersion_energy());
    }
    if (energies.one_center_energy_contribution() != 0.0) {
        terms.one_center = energy("one_center", energies.one_center_energy_contribution());
    }
}

/// ESSE's fermi_energy record, in rydberg.
inline void set_fermi_energy(generated::Results& results, Real fermi_energy_in_ry) {
    set_record(results,
               generated::FermiEnergy{generated::FermiEnergyNameEnum::FERMI_ENERGY,
                                      generated::FermiEnergyUnitsEnum::RY,
                                      fermi_energy_in_ry});
}

/**
 * The results as the text of results.json. write_results_json() writes the
 * same text, so the file a run leaves and the string the Python bindings
 * return cannot drift apart.
 */
[[nodiscard]] inline std::string results_json_string(const generated::Results& results) {
    return nlohmann::json(results).dump(4) + "\n";
}

/// Write the results to `output_path` as results.json.
inline void write_results_json(const generated::Results& results, const std::string& output_path) {
    write_json_file(output_path, results);
}

} // namespace q3
