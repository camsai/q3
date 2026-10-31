#pragma once

// @generated from results/properties.json — DO NOT EDIT

#include <map>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "generated/esse/properties/non_scalar/stress_tensor.hpp"
#include "generated/esse/properties/structural/atomic_forces.hpp"
#include "generated/q3/results/properties/energy_contributions.hpp"
#include "generated/q3/results/properties/hubbard_occupations.hpp"
#include "generated/q3/results/properties/isolated_system_correction.hpp"
#include "generated/q3/results/properties/orbital_projections.hpp"

namespace q3::generated {

/**
 * The physical properties the run computed: every one ESSE has a property schema for as that ESSE record under records, and beside them the quantities ESSE has no property for yet, each an ESSE schema or one built from ESSE pieces carrying its own units.
 */
struct ResultsProperties {
    // Every result ESSE has a property schema for, as that ESSE record and nowhere else in this file, each validated against the ESSE schema itself: total_energy, total_energy_contributions (eigenvalues, hartree, exchange_correlation, ewald, temperatureEntropy), fermi_energy and eigenvalues_at_kpoints always; atomic_forces (ids 1-based, as in the material basis) whenever forces were computed, and stress_tensor and pressure whenever the stress was, a known-missing term in either being named in runtime_log.incomplete; band_edge_valence_max, the highest eigenvalue at or below the Fermi energy, and band_edge_conduction_min, the lowest above it, each when such a level exists and with its k-point when the run lists k-point coordinates; band_gaps, as a 'direct' and an 'indirect' entry, when both band edges exist and no level is fractionally occupied; magnetization_total, the integral of rho_up - rho_down over the cell, and magnetization_absolute, the integral of its absolute value, for a spin-polarized run (the total alone is zero both for a converged antiferromagnet and for one that collapsed to non-magnetic); convergence_electronic, the total energy after each SCF iteration as one row, where the flavor records that trajectory. eigenvalues_at_kpoints lists one entry per k-point state with one spin channel: a spin-polarized run lists every k-point twice, the spin-up states first and then the spin-down ones, each at half the k-point's weight; weights sum to 2.0 over the k-points, the spin degeneracy included, and occupations are fractions in [0, 1]. A quantity ESSE has no property schema for yet is one of the blocks beside these records, carrying its own units, until the schema is added to ESSE (AGENTS.md, 'Output schemas').
    std::vector<nlohmann::json> records;
    // Energy terms Q3 reports that ESSE's total_energy_contributions has no key for yet, each an ESSE energy in rydberg; the terms ESSE does have are in that record under properties. hubbard (DFT+U, Dudarev), dispersion (already inside total_energy) and one_center (PAW) are present only when nonzero.
    ResultsEnergyContributions energy_contributions;
    // The charged/isolated-cell correction, when one was applied: the scheme's own labelled terms (Makov-Payne: first_order, second_order; Martyna-Tuckerman: energy), all signed so that corrected_total_energy = total_energy + total.
    std::optional<ResultsIsolatedSystemCorrection> isolated_system_correction;
    // Tr[n] per Hubbard manifold, per spin channel — the electron count the manifold's projectors see. Absent when no Hubbard correction is active.
    std::optional<std::vector<ResultsPropertiesHubbardOccupationsItem>> hubbard_occupations;
    // The per-atom force broken down by physical mechanism ('local', 'ewald', 'nonlocal', 'core_correction', 'augmentation', 'dispersion', 'hubbard'). A total that disagrees with a reference says only that something is wrong; the term that disagrees says what.
    std::optional<std::map<std::string, AtomicForces>> force_contributions;
    // The stress tensor broken down by physical mechanism, same names as the force breakdown plus 'kinetic', 'hartree' and 'exchange_correlation'.
    std::optional<std::map<std::string, StressTensor>> stress_contributions;
    // Present only when a non-self-consistent run was asked for projections.
    std::optional<ResultsOrbitalProjections> orbital_projections;
};

inline void to_json(nlohmann::json& json_output, const ResultsProperties& data) {
    json_output = nlohmann::json::object();
    json_output["records"] = data.records;
    json_output["energy_contributions"] = data.energy_contributions;
    if (data.isolated_system_correction) {
        json_output["isolated_system_correction"] = *data.isolated_system_correction;
    }
    if (data.hubbard_occupations) {
        json_output["hubbard_occupations"] = *data.hubbard_occupations;
    }
    if (data.force_contributions) {
        json_output["force_contributions"] = *data.force_contributions;
    }
    if (data.stress_contributions) {
        json_output["stress_contributions"] = *data.stress_contributions;
    }
    if (data.orbital_projections) {
        json_output["orbital_projections"] = *data.orbital_projections;
    }
}

inline void from_json(const nlohmann::json& json_input, ResultsProperties& data) {
    json_input.at("records").get_to(data.records);
    json_input.at("energy_contributions").get_to(data.energy_contributions);
    if (json_input.contains("isolated_system_correction")) {
        data.isolated_system_correction = json_input.at("isolated_system_correction").get<ResultsIsolatedSystemCorrection>();
    }
    if (json_input.contains("hubbard_occupations")) {
        data.hubbard_occupations = json_input.at("hubbard_occupations").get<std::vector<ResultsPropertiesHubbardOccupationsItem>>();
    }
    if (json_input.contains("force_contributions")) {
        data.force_contributions = json_input.at("force_contributions").get<std::map<std::string, AtomicForces>>();
    }
    if (json_input.contains("stress_contributions")) {
        data.stress_contributions = json_input.at("stress_contributions").get<std::map<std::string, StressTensor>>();
    }
    if (json_input.contains("orbital_projections")) {
        data.orbital_projections = json_input.at("orbital_projections").get<ResultsOrbitalProjections>();
    }
}

} // namespace q3::generated
