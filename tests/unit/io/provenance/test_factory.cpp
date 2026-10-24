#include "io/provenance/factory.hpp"

#include <gtest/gtest.h>
#include <memory>
#include <string>

using namespace q3;

namespace {

/// A minimal silicon input, enough for every provenance field to have a source.
InputData make_silicon_input() {
    InputData input;
    input.exchange_correlation_functional = "PBE";
    input.number_of_spin_components = 1;
    input.wavefunction_energy_cutoff = 48.0;
    input.charge_density_energy_cutoff = 192.0;
    input.kpoint_grid_dimension_1 = 2;
    input.kpoint_grid_dimension_2 = 2;
    input.kpoint_grid_dimension_3 = 2;
    input.kpoint_grid_shift_1 = 1;
    input.kpoint_grid_shift_2 = 1;
    input.kpoint_grid_shift_3 = 1;
    input.convergence_threshold = 1.0e-8;
    input.maximum_iterations = 100;
    input.pseudopotential_directory = std::string(EXAMPLES_DIR) + "/data/pseudopotentials/nc/";

    AtomSpecies silicon;
    silicon.symbol = "Si";
    silicon.mass = 28.0855;
    silicon.pseudopotential_file = "Si.upf";
    input.species.push_back(silicon);
    input.number_of_species = 1;

    return input;
}

} // namespace

/// The engine identifies itself, using the spellings the consuming registry reads.
TEST(Provenance, RecordsTheEngineIdentity) {
    const nlohmann::json provenance = build_provenance_json(make_silicon_input());

    ASSERT_TRUE(provenance.contains("generator"));
    EXPECT_EQ(provenance["generator"]["engine_name"], "q3");
    EXPECT_EQ(provenance["generator"]["engine_version"], std::string(version));
    EXPECT_EQ(provenance["generator"]["engine_commit"], std::string(commit));
}

/**
 * The BLAS/LAPACK the build linked travels with every result: a run's speed can hinge on
 * it, and a slow run reported from another machine said nothing about it.
 */
TEST(Provenance, RecordsTheLinearAlgebraLibrariesTheBuildLinked) {
    const nlohmann::json provenance = build_provenance_json(make_silicon_input());

    ASSERT_TRUE(provenance["generator"].contains("linear_algebra_libraries"));
    EXPECT_EQ(provenance["generator"]["linear_algebra_libraries"],
              std::string(linear_algebra_libraries));
    EXPECT_FALSE(std::string(linear_algebra_libraries).empty())
        << "CMake passes the LAPACK libraries it found; an empty string means it did not";
}

/// The numerics that were asked for, in Q3's own units, named so no reader can mistake them.
TEST(Provenance, EchoesTheRequestedModelAndMethod) {
    const nlohmann::json provenance = build_provenance_json(make_silicon_input());

    EXPECT_EQ(provenance["model"]["exchange_correlation_functional"], "PBE");
    EXPECT_EQ(provenance["model"]["spin_treatment"], "unpolarized");

    EXPECT_DOUBLE_EQ(provenance["method"]["wavefunction_cutoff_rydberg"].get<double>(), 48.0);
    EXPECT_DOUBLE_EQ(provenance["method"]["density_cutoff_rydberg"].get<double>(), 192.0);
    EXPECT_EQ(provenance["method"]["kpoint_grid"], nlohmann::json({2, 2, 2}));
    EXPECT_EQ(provenance["method"]["kpoint_shift"], nlohmann::json({1, 1, 1}));
    EXPECT_EQ(provenance["method"]["electronic_occupation"], "fixed");

    EXPECT_DOUBLE_EQ(provenance["precision"]["energy_convergence_threshold_rydberg"].get<double>(),
                     1.0e-8);
    EXPECT_EQ(provenance["precision"]["maximum_self_consistent_field_iterations"], 100);
}

/// Smearing is reported only when it was used, and with the width that was requested.
TEST(Provenance, ReportsSmearingOnlyWhenItIsInUse) {
    InputData input = make_silicon_input();
    input.smearing = "gaussian";
    input.smearing_width = 0.01;

    const nlohmann::json method = build_provenance_json(input)["method"];
    EXPECT_EQ(method["electronic_occupation"], "smearing");
    EXPECT_EQ(method["smearing_kind"], "gaussian");
    EXPECT_DOUBLE_EQ(method["smearing_width_rydberg"].get<double>(), 0.01);

    const nlohmann::json fixed_method = build_provenance_json(make_silicon_input())["method"];
    EXPECT_FALSE(fixed_method.contains("smearing_kind"));
    EXPECT_FALSE(fixed_method.contains("smearing_width_rydberg"));
}

/**
 * A Q3 run is a plane-wave pseudopotential calculation, and the factory is the
 * one place that says so: the record it hands back reports through the full
 * chain -- the system, its basis, and its datasets -- when it is used through
 * the base class, which is how every caller uses it.
 */
TEST(Provenance, BuildsThePlanewavePseudopotentialRecordForAQ3Run) {
    const InputData input = make_silicon_input();
    const std::unique_ptr<QuantumSystemProvenanceRecord> record = create_provenance_record(input);
    ASSERT_NE(record, nullptr);

    const nlohmann::json method = record->to_json()["method"];
    EXPECT_DOUBLE_EQ(method["wavefunction_cutoff_rydberg"].get<double>(), 48.0);
    ASSERT_TRUE(method.contains("pseudopotentials"));
    EXPECT_EQ(method["pseudopotentials"].size(), 1u);
}
