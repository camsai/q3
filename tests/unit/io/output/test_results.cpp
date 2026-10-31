#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>

#include <nlohmann/json.hpp>

#include "io/output/results.hpp"

using namespace q3;

namespace {

const nlohmann::json* find_property(const nlohmann::json& serialized, const std::string& name) {
    for (const auto& property : serialized["properties"]["records"]) {
        if (property["name"] == name) {
            return &property;
        }
    }
    return nullptr;
}

/// Energies with every term set, as a converged DFT+U run with dispersion reports them.
struct EveryEnergyTerm : PlanewaveKohnShamEnergies {
    [[nodiscard]] Real dispersion_energy() const override {
        return -0.0421;
    }
    [[nodiscard]] Real hubbard_energy_contribution() const override {
        return 0.125;
    }
};

} // namespace

/**
 * ── Units ────────────────────────────────────────────────────────────
 *
 * Every quantity carries its unit in its own "units" field, the way ESSE's
 * records do, and no key names one (issue #314): a reader checks the unit
 * where it reads the value. Both serialization entry points
 * (write_results_json, results_json_string) must agree, since the string is
 * what nanobind returns directly to Python.
 */

namespace {

/// Every key in the document, at any depth, that ends in a unit suffix.
void collect_unit_suffixed_keys(const nlohmann::json& node, std::vector<std::string>& found) {
    if (node.is_object()) {
        for (const auto& [key, value] : node.items()) {
            for (const char* suffix : {"_ry", "_kbar", "_ry_per_bohr", "_bohr_magneton", "_ev"}) {
                const std::string unit_suffix(suffix);
                if (key.size() > unit_suffix.size() &&
                    key.compare(key.size() - unit_suffix.size(), unit_suffix.size(), unit_suffix) ==
                        0) {
                    found.push_back(key);
                }
            }
            collect_unit_suffixed_keys(value, found);
        }
    } else if (node.is_array()) {
        for (const auto& value : node) {
            collect_unit_suffixed_keys(value, found);
        }
    }
}

generated::Results results_with_every_block() {
    generated::Results results = new_results("self_consistent_field");
    EveryEnergyTerm energies;
    energies.band_energy = 1.25;
    set_energy_records(results, -15.8547, energies);
    set_fermi_energy(results, 0.1);
    set_record(results, atomic_forces({{0.01, 0.0, 0.0}}));
    results.properties.force_contributions = std::map<std::string, generated::AtomicForces>{
        {"local", atomic_forces({{0.01, 0.0, 0.0}})}};
    const Real tensor_in_kbar[3][3] = {};
    set_record(results, stress_tensor(tensor_in_kbar));
    results.properties.stress_contributions =
        std::map<std::string, generated::StressTensor>{{"kinetic", stress_tensor(tensor_in_kbar)}};
    generated::ResultsIsolatedSystemCorrection correction;
    correction.method = "makov_payne";
    correction.terms["first_order"] = energy("first_order", -0.01);
    correction.total = energy("total", -0.01);
    correction.corrected_total_energy = energy("corrected_total_energy", -15.8647);
    results.properties.isolated_system_correction = correction;
    return results;
}

} // namespace

TEST(ResultsUnits, NoKeyNamesAUnit) {
    const nlohmann::json parsed =
        nlohmann::json::parse(results_json_string(results_with_every_block()));

    std::vector<std::string> unit_suffixed_keys;
    collect_unit_suffixed_keys(parsed, unit_suffixed_keys);
    EXPECT_TRUE(unit_suffixed_keys.empty())
        << "first unit-suffixed key: " << unit_suffixed_keys.front();
    EXPECT_FALSE(parsed.contains("units"));
}

TEST(ResultsUnits, EveryQuantityStatesItsOwnUnit) {
    const nlohmann::json parsed =
        nlohmann::json::parse(results_json_string(results_with_every_block()));

    EXPECT_EQ(parsed["properties"]["energy_contributions"]["double_counting"]["units"], "Ry");
    EXPECT_EQ(parsed["properties"]["energy_contributions"]["hubbard"]["units"], "Ry");
    EXPECT_EQ(parsed["properties"]["force_contributions"]["local"]["units"], "Ry/a.u.");
    EXPECT_EQ(parsed["properties"]["stress_contributions"]["kinetic"]["units"], "kbar");
    EXPECT_EQ(parsed["properties"]["isolated_system_correction"]["total"]["units"], "Ry");
    for (const auto& record : parsed["properties"]["records"]) {
        EXPECT_TRUE(record.contains("units")) << record["name"];
    }
}

TEST(ResultsUnits, WriteResultsJsonAndResultsJsonStringAgree) {
    const generated::Results results = results_with_every_block();

    const auto temporary_path =
        std::filesystem::temp_directory_path() / "q3_results_units_agreement.json";
    write_results_json(results, temporary_path.string());
    std::ifstream file_stream(temporary_path);
    nlohmann::json from_file;
    file_stream >> from_file;
    std::filesystem::remove(temporary_path);

    nlohmann::json from_string = nlohmann::json::parse(results_json_string(results));

    EXPECT_EQ(from_file, from_string);
}

TEST(ResultsUnits, WriteResultsJsonCreatesMissingOutputDirectory) {
    const generated::Results results = new_results("self_consistent_field");

    const auto output_directory =
        std::filesystem::temp_directory_path() / "q3_results_missing_directory";
    const auto output_path = output_directory / "nested" / "results.json";
    std::filesystem::remove_all(output_directory);

    write_results_json(results, output_path.string());

    EXPECT_TRUE(std::filesystem::is_regular_file(output_path));
    std::filesystem::remove_all(output_directory);
}

/**
 * ── A run before it has computed anything ────────────────────────────
 */

TEST(NewResults, NamesTheFlavorAndLeavesTheIterationCountOut) {
    /**
     * Only the flavors that iterate to self-consistency have a count, so the
     * others must leave the key out rather than emit a zero.
     */
    const nlohmann::json serialized =
        nlohmann::json::parse(results_json_string(new_results("non_self_consistent_field")));

    EXPECT_EQ(serialized["runtime_log"]["flavor"], "non_self_consistent_field");
    EXPECT_FALSE(serialized["runtime_log"].contains("self_consistent_field_iterations"));
    EXPECT_FALSE(serialized["runtime_log"].contains("density_residuals"));
}

/**
 * A single-process run's speed depends on its thread count as much as an MPI run's
 * does on its ranks; the default is what OMP_NUM_THREADS gives this process.
 */
TEST(NewResults, RecordsTheThreadsEachProcessMayUse) {
    const generated::Results results = new_results("self_consistent_field");

    EXPECT_EQ(results.runtime_log.execution.number_of_threads_per_process,
              execution::threading::maximum_threads());
    EXPECT_EQ(results.runtime_log.execution.number_of_processes, 1);
    EXPECT_EQ(results.runtime_log.execution.number_of_kpoint_pools, 1);
    EXPECT_EQ(results.runtime_log.execution.band_group_size, 1);
}

/**
 * ── Records by their generated type ─────────────────────────────────
 */

TEST(ResultsRecords, FindRecordReadsBackWhatSetRecordListed) {
    generated::Results results = new_results("self_consistent_field");
    EXPECT_FALSE(find_record<generated::FermiEnergy>(results).has_value());

    set_fermi_energy(results, 0.45);

    const auto fermi_energy = find_record<generated::FermiEnergy>(results);
    ASSERT_TRUE(fermi_energy.has_value());
    EXPECT_DOUBLE_EQ(fermi_energy->value, 0.45);
}

TEST(ResultsRecords, SetRecordReplacesARecordOfTheSameName) {
    generated::Results results = new_results("self_consistent_field");
    set_fermi_energy(results, 0.45);
    set_fermi_energy(results, 0.5);

    ASSERT_EQ(results.properties.records.size(), 1u);
    EXPECT_DOUBLE_EQ(find_record<generated::FermiEnergy>(results)->value, 0.5);
}

/**
 * ── Energies ─────────────────────────────────────────────────────────
 *
 * Issue #314: results ESSE has a property schema for are written as that
 * ESSE record, under "properties", and nowhere else, so a consumer that reads
 * ESSE needs no Q3-specific mapping. tests/regression/test_results_schema.py
 * validates the records against ESSE itself; these pin what the writer puts
 * in them.
 */

TEST(ResultsEnergies, EnergiesAreListedInEsseShape) {
    generated::Results results = new_results("self_consistent_field");
    PlanewaveKohnShamEnergies energies;
    energies.band_energy = 1.25;
    energies.hartree_energy = 1.1;
    energies.exchange_correlation_energy = -4.8;
    energies.ewald_energy = -16.9;
    energies.smearing_entropy_contribution = -0.002;
    set_energy_records(results, -15.85, energies);
    set_fermi_energy(results, 0.45);

    const nlohmann::json serialized = nlohmann::json::parse(results_json_string(results));

    const nlohmann::json* total_energy = find_property(serialized, "total_energy");
    ASSERT_NE(total_energy, nullptr);
    EXPECT_EQ((*total_energy)["units"], "Ry");
    EXPECT_DOUBLE_EQ((*total_energy)["value"], -15.85);

    const nlohmann::json* contributions = find_property(serialized, "total_energy_contributions");
    ASSERT_NE(contributions, nullptr);
    EXPECT_EQ((*contributions)["units"], "Ry");
    EXPECT_DOUBLE_EQ((*contributions)["eigenvalues"]["value"], 1.25);
    EXPECT_DOUBLE_EQ((*contributions)["hartree"]["value"], 1.1);
    EXPECT_DOUBLE_EQ((*contributions)["exchange_correlation"]["value"], -4.8);
    EXPECT_DOUBLE_EQ((*contributions)["ewald"]["value"], -16.9);
    EXPECT_DOUBLE_EQ((*contributions)["temperatureEntropy"]["value"], -0.002);

    const nlohmann::json* fermi_energy = find_property(serialized, "fermi_energy");
    ASSERT_NE(fermi_energy, nullptr);
    EXPECT_DOUBLE_EQ((*fermi_energy)["value"], 0.45);
}

TEST(ResultsEnergies, NonzeroHubbardAndDispersionTermsAreListed) {
    generated::Results results = new_results("self_consistent_field");
    set_energy_records(results, -15.0, EveryEnergyTerm{});

    const nlohmann::json serialized = nlohmann::json::parse(results_json_string(results));

    const nlohmann::json& terms = serialized["properties"]["energy_contributions"];
    EXPECT_DOUBLE_EQ(terms["hubbard"]["value"], 0.125);
    EXPECT_DOUBLE_EQ(terms["dispersion"]["value"], -0.0421);
    EXPECT_FALSE(terms.contains("one_center"));
}

TEST(ResultsEnergies, TermsThatDidNotRunAreOmitted) {
    generated::Results results = new_results("self_consistent_field");
    set_energy_records(results, -15.0, PlanewaveKohnShamEnergies{});

    const nlohmann::json serialized = nlohmann::json::parse(results_json_string(results));

    const nlohmann::json& terms = serialized["properties"]["energy_contributions"];
    EXPECT_TRUE(terms.contains("double_counting"));
    EXPECT_FALSE(terms.contains("hubbard"));
    EXPECT_FALSE(terms.contains("dispersion"));
    EXPECT_FALSE(terms.contains("one_center"));
}

/**
 * ── Forces and stress in ESSE's shapes ───────────────────────────────
 */

TEST(ResultsForces, ForcesAreKeyedByTheOneBasedBasisIds) {
    const nlohmann::json forces = atomic_forces({{0.01, 0.0, -0.02}, {-0.01, 0.0, 0.02}});

    EXPECT_EQ(forces["name"], "atomic_forces");
    EXPECT_EQ(forces["units"], "Ry/a.u.");
    ASSERT_EQ(forces["values"].size(), 2u);
    EXPECT_EQ(forces["values"][0]["id"], 1);
    EXPECT_EQ(forces["values"][1]["id"], 2);
    EXPECT_DOUBLE_EQ(forces["values"][1]["value"][2], 0.02);
}

TEST(ResultsStress, TheTensorIsListedRowByRow) {
    const Real tensor_in_kbar[3][3] = {
        {1.0, 2.0, 3.0},
        {2.0, 4.0, 5.0},
        {3.0, 5.0, 6.0},
    };

    const nlohmann::json expected = {
        {"name", "stress_tensor"},
        {"units", "kbar"},
        {"value", {{1.0, 2.0, 3.0}, {2.0, 4.0, 5.0}, {3.0, 5.0, 6.0}}},
    };
    EXPECT_EQ(nlohmann::json(stress_tensor(tensor_in_kbar)), expected);
}

TEST(ResultsRecords, UncomputedDerivativesAreNotListed) {
    /**
     * A run that computed no forces or stress (NSCF, or a band-parallel
     * topology that skips the stress) lists no record, rather than an empty
     * force list or the zero tensor a reader would take as measured.
     */
    generated::Results results = new_results("non_self_consistent_field");
    set_energy_records(results, 0.0, PlanewaveKohnShamEnergies{});

    const nlohmann::json serialized = nlohmann::json::parse(results_json_string(results));

    EXPECT_EQ(find_property(serialized, "atomic_forces"), nullptr);
    EXPECT_EQ(find_property(serialized, "stress_tensor"), nullptr);
    EXPECT_EQ(find_property(serialized, "pressure"), nullptr);
    EXPECT_FALSE(serialized["properties"].contains("force_contributions"));
    EXPECT_FALSE(serialized["properties"].contains("stress_contributions"));
}

TEST(ResultsRecords, NoResultIsWrittenTwice) {
    const nlohmann::json serialized =
        nlohmann::json::parse(results_json_string(results_with_every_block()));

    for (const char* flat_key : {"total_energy_ry",
                                 "band_energy_ry",
                                 "hartree_energy_ry",
                                 "exchange_correlation_energy_ry",
                                 "ewald_energy_ry",
                                 "smearing_entropy_contribution_ry",
                                 "fermi_energy_ry",
                                 "forces_ry_per_bohr",
                                 "stress_kbar",
                                 "pressure_kbar",
                                 "band_gap_ry"}) {
        EXPECT_FALSE(serialized.contains(flat_key)) << flat_key;
    }
}
