#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <sstream>

#include <nlohmann/json.hpp>

#include "layer_4_computation/flavors/result.hpp"

using namespace q3;

/**
 * ── Units field ──────────────────────────────────────────────────────
 *
 * TODOs.md §3.1 "Add units fields": every serialized JSON output must be
 * self-describing about the units its values are in, rather than relying
 * solely on unit-suffixed key names (total_energy_ry, etc.). Both
 * serialization entry points (write_results_json, to_json_string) must
 * agree, since to_json_string's output is what nanobind returns directly
 * to Python.
 */

TEST(FlavorResultUnits, ToJsonStringIncludesUnitsObject) {
    FlavorResult result;
    result.flavor = "self_consistent_field";
    result.total_energy_in_ry = -15.8547;

    nlohmann::json parsed = nlohmann::json::parse(result.to_json_string());
    ASSERT_TRUE(parsed.contains("units"));
    EXPECT_EQ(parsed["units"]["energy"], "Ry");
    EXPECT_EQ(parsed["units"]["forces"], "Ry/bohr");
    EXPECT_EQ(parsed["units"]["stress"], "kbar");
    EXPECT_EQ(parsed["units"]["eigenvalues"], "Ry");
}

TEST(FlavorResultUnits, WriteResultsJsonIncludesUnitsObject) {
    FlavorResult result;
    result.flavor = "self_consistent_field";

    const auto temporary_path =
        std::filesystem::temp_directory_path() / "q3_flavor_result_units.json";
    result.write_results_json(temporary_path.string());

    std::ifstream file_stream(temporary_path);
    nlohmann::json parsed;
    file_stream >> parsed;
    std::filesystem::remove(temporary_path);

    ASSERT_TRUE(parsed.contains("units"));
    EXPECT_EQ(parsed["units"]["energy"], "Ry");
    EXPECT_EQ(parsed["units"]["forces"], "Ry/bohr");
    EXPECT_EQ(parsed["units"]["stress"], "kbar");
    EXPECT_EQ(parsed["units"]["eigenvalues"], "Ry");
}

TEST(FlavorResultUnits, WriteResultsJsonAndToJsonStringAgreeOnUnits) {
    /**
     * Both serialization entry points must stay in sync (they are two
     * hand-synced copies of the same schema -- see TODOs.md's separate
     * "consolidate duplicate energy serialization" item, not addressed here).
     */
    FlavorResult result;
    result.flavor = "self_consistent_field";
    result.total_energy_in_ry = -337.5;

    const auto temporary_path =
        std::filesystem::temp_directory_path() / "q3_flavor_result_units_agreement.json";
    result.write_results_json(temporary_path.string());
    std::ifstream file_stream(temporary_path);
    nlohmann::json from_file;
    file_stream >> from_file;
    std::filesystem::remove(temporary_path);

    nlohmann::json from_string = nlohmann::json::parse(result.to_json_string());

    EXPECT_EQ(from_file["units"], from_string["units"]);
}

TEST(FlavorResultUnits, WriteResultsJsonCreatesMissingOutputDirectory) {
    FlavorResult result;
    result.flavor = "self_consistent_field";

    const auto output_directory =
        std::filesystem::temp_directory_path() / "q3_flavor_result_missing_directory";
    const auto output_path = output_directory / "nested" / "results.json";
    std::filesystem::remove_all(output_directory);

    result.write_results_json(output_path.string());

    EXPECT_TRUE(std::filesystem::is_regular_file(output_path));
    std::filesystem::remove_all(output_directory);
}

TEST(FlavorResultEnergies, SerializersIncludeNonzeroHubbardEnergy) {
    FlavorResult result;
    result.flavor = "self_consistent_field";
    result.hubbard_energy_in_ry = 0.125;

    const auto output_path =
        std::filesystem::temp_directory_path() / "q3_flavor_result_hubbard_energy.json";
    result.write_results_json(output_path.string());

    std::ifstream file_stream(output_path);
    nlohmann::json from_file;
    file_stream >> from_file;
    std::filesystem::remove(output_path);
    const nlohmann::json from_string = nlohmann::json::parse(result.to_json_string());

    EXPECT_DOUBLE_EQ(from_file["hubbard_energy_ry"], 0.125);
    EXPECT_DOUBLE_EQ(from_string["hubbard_energy_ry"], 0.125);
}

/**
 * ── SCF iteration count ──────────────────────────────────────────────
 *
 * A result that did not converge has to say how far it got: before the count
 * was serialized, the Python layer reported every failed run as having stopped
 * "after 0 iterations". Only the flavors that iterate to self-consistency have
 * a count, so the others must leave the key out rather than emit a zero.
 */

TEST(FlavorResultConvergence, SerializersIncludeTheIterationCount) {
    FlavorResult result;
    result.flavor = "self_consistent_field";
    result.converged = false;
    result.number_of_self_consistent_field_iterations = 100;

    const auto output_path =
        std::filesystem::temp_directory_path() / "q3_flavor_result_iteration_count.json";
    result.write_results_json(output_path.string());
    std::ifstream file_stream(output_path);
    nlohmann::json from_file;
    file_stream >> from_file;
    std::filesystem::remove(output_path);
    const nlohmann::json from_string = nlohmann::json::parse(result.to_json_string());

    EXPECT_EQ(from_file["self_consistent_field_iterations"], 100);
    EXPECT_EQ(from_string["self_consistent_field_iterations"], 100);
}

TEST(FlavorResultConvergence, IterationCountIsOmittedWhenTheFlavorHasNone) {
    FlavorResult result;
    result.flavor = "non_self_consistent_field";
    result.converged = true;

    const nlohmann::json serialized = nlohmann::json::parse(result.to_json_string());

    EXPECT_FALSE(serialized.contains("self_consistent_field_iterations"));
}

TEST(FlavorResultExecution, SerializersIncludeActualTopology) {
    FlavorResult result;
    result.number_of_processes = 4;
    result.number_of_kpoint_pools = 2;
    result.band_group_size = 2;
    result.reciprocal_space_vectors_are_distributed = true;

    const nlohmann::json serialized = nlohmann::json::parse(result.to_json_string());

    EXPECT_EQ(serialized["execution"]["number_of_processes"], 4);
    EXPECT_EQ(serialized["execution"]["number_of_kpoint_pools"], 2);
    EXPECT_EQ(serialized["execution"]["band_group_size"], 2);
    EXPECT_EQ(serialized["execution"]["reciprocal_space_vectors_are_distributed"], true);
}

TEST(FlavorResultStress, SerializersIncludeFullContributionTensors) {
    FlavorResult result;
    Real ewald_stress[3][3] = {
        {1.0, 2.0, 3.0},
        {2.0, 4.0, 5.0},
        {3.0, 5.0, 6.0},
    };
    result.add_stress_contribution("ewald", ewald_stress);

    const auto output_path =
        std::filesystem::temp_directory_path() / "q3_flavor_result_stress_contributions.json";
    result.write_results_json(output_path.string());
    std::ifstream file_stream(output_path);
    nlohmann::json from_file;
    file_stream >> from_file;
    std::filesystem::remove(output_path);
    const nlohmann::json from_string = nlohmann::json::parse(result.to_json_string());

    const nlohmann::json expected = {
        {"ewald", {{1.0, 2.0, 3.0}, {2.0, 4.0, 5.0}, {3.0, 5.0, 6.0}}},
    };
    EXPECT_EQ(from_file["stress_contributions_kbar"], expected);
    EXPECT_EQ(from_string["stress_contributions_kbar"], expected);
}

/**
 * ── The quantities that used to exist only on the console ────────────
 *
 * Dispersion energy, the per-atom force breakdown, the scalar pressure, both
 * magnetizations, DFT+U Tr[n] and the band edges were printed by the run and
 * nowhere else, so anything that wanted them had to regular-expression them
 * back out of stdout. They are keys now; these tests pin the shapes that
 * schemas/q3/results.json describes, and the omit-rather-than-zero rule that
 * lets a reader tell "not applicable" from "came out at zero".
 */

TEST(FlavorResultEnergies, SerializersIncludeNonzeroDispersionEnergy) {
    FlavorResult result;
    result.dispersion_energy_in_ry = -0.0421;

    const nlohmann::json serialized = nlohmann::json::parse(result.to_json_string());

    EXPECT_DOUBLE_EQ(serialized["dispersion_energy_ry"], -0.0421);
}

TEST(FlavorResultEnergies, DispersionEnergyIsOmittedWhenNoSchemeRan) {
    FlavorResult result;

    const nlohmann::json serialized = nlohmann::json::parse(result.to_json_string());

    EXPECT_FALSE(serialized.contains("dispersion_energy_ry"));
}

TEST(FlavorResultForces, SerializersIncludeThePerAtomContributionBreakdown) {
    FlavorResult result;
    result.forces_in_ry_per_bohr = {{0.5, 0.0, 0.0}, {-0.5, 0.0, 0.0}};
    result.add_force_contribution("ewald", {{0.3, 0.0, 0.0}, {-0.3, 0.0, 0.0}});
    result.add_force_contribution("nonlocal", {{0.2, 0.0, 0.0}, {-0.2, 0.0, 0.0}});

    const nlohmann::json serialized = nlohmann::json::parse(result.to_json_string());

    const nlohmann::json expected = {
        {"ewald", {{0.3, 0.0, 0.0}, {-0.3, 0.0, 0.0}}},
        {"nonlocal", {{0.2, 0.0, 0.0}, {-0.2, 0.0, 0.0}}},
    };
    EXPECT_EQ(serialized["force_contributions_ry_per_bohr"], expected);
}

TEST(FlavorResultForces, ContributionBreakdownIsOmittedWhenNoForcesWereComputed) {
    FlavorResult result;

    const nlohmann::json serialized = nlohmann::json::parse(result.to_json_string());

    EXPECT_FALSE(serialized.contains("force_contributions_ry_per_bohr"));
}

TEST(FlavorResultStress, SerializersIncludeTheScalarPressure) {
    FlavorResult result;
    result.pressure_in_kbar = -17.25;

    const nlohmann::json serialized = nlohmann::json::parse(result.to_json_string());

    EXPECT_DOUBLE_EQ(serialized["pressure_kbar"], -17.25);
    EXPECT_EQ(serialized["units"]["pressure"], "kbar");
}

TEST(FlavorResultMagnetization, SerializersIncludeBothMomentsWhenSpinPolarized) {
    FlavorResult result;
    result.magnetization = Magnetization{0.0, 2.6};

    const nlohmann::json serialized = nlohmann::json::parse(result.to_json_string());

    EXPECT_DOUBLE_EQ(serialized["total_magnetization_bohr_magneton"], 0.0);
    EXPECT_DOUBLE_EQ(serialized["absolute_magnetization_bohr_magneton"], 2.6);
    EXPECT_EQ(serialized["units"]["magnetization"], "uB");
}

TEST(FlavorResultMagnetization, MomentsAreOmittedRatherThanEmittedAsZero) {
    /**
     * A zero total is what a converged antiferromagnet reports, so emitting
     * zeros for a spin-unpolarized run would make the two indistinguishable.
     */
    FlavorResult result;

    const nlohmann::json serialized = nlohmann::json::parse(result.to_json_string());

    EXPECT_FALSE(serialized.contains("total_magnetization_bohr_magneton"));
    EXPECT_FALSE(serialized.contains("absolute_magnetization_bohr_magneton"));
}

TEST(FlavorResultHubbard, SerializersIncludeTheOccupationTracePerManifold) {
    FlavorResult result;
    result.hubbard_occupations.push_back({0, {4.8, 4.03}});

    const nlohmann::json serialized = nlohmann::json::parse(result.to_json_string());

    ASSERT_EQ(serialized["hubbard_occupations"].size(), 1u);
    EXPECT_EQ(serialized["hubbard_occupations"][0]["atom_index"], 0);
    EXPECT_DOUBLE_EQ(serialized["hubbard_occupations"][0]["occupation_trace"][0], 4.8);
    EXPECT_DOUBLE_EQ(serialized["hubbard_occupations"][0]["occupation_trace"][1], 4.03);
}

TEST(FlavorResultBandEdges, EdgesAndGapAreTakenAroundTheFermiEnergy) {
    FlavorResult result;
    result.fermi_energy_in_ry = 0.0;
    result.eigenvalues_in_ry = {{{-1.0, -0.4, 0.6, 1.2}}, {{-0.9, -0.3, 0.8, 1.5}}};

    const nlohmann::json serialized = nlohmann::json::parse(result.to_json_string());

    EXPECT_DOUBLE_EQ(serialized["valence_band_maximum_ry"], -0.3);
    EXPECT_DOUBLE_EQ(serialized["conduction_band_minimum_ry"], 0.6);
    EXPECT_DOUBLE_EQ(serialized["band_gap_ry"], 0.9);
}

TEST(FlavorResultBandEdges, NoGapIsReportedWithoutAConductionEdge) {
    /**
     * A run asked for exactly its occupied manifold has no unoccupied level to
     * measure to; reporting a gap anyway would invent the half that was never
     * computed.
     */
    FlavorResult result;
    result.fermi_energy_in_ry = 0.0;
    result.eigenvalues_in_ry = {{{-1.0, -0.4}}};

    const nlohmann::json serialized = nlohmann::json::parse(result.to_json_string());

    EXPECT_DOUBLE_EQ(serialized["valence_band_maximum_ry"], -0.4);
    EXPECT_FALSE(serialized.contains("conduction_band_minimum_ry"));
    EXPECT_FALSE(serialized.contains("band_gap_ry"));
}

TEST(FlavorResultBandEdges, NothingIsReportedWithoutEigenvalues) {
    FlavorResult result;

    const nlohmann::json serialized = nlohmann::json::parse(result.to_json_string());

    EXPECT_FALSE(serialized.contains("valence_band_maximum_ry"));
    EXPECT_FALSE(serialized.contains("conduction_band_minimum_ry"));
    EXPECT_FALSE(serialized.contains("band_gap_ry"));
}
