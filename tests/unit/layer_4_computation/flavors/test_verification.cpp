#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <string>

#include <nlohmann/json.hpp>

#include "io/input/data.hpp"
#include "layer_4_computation/flavors/factory.hpp"
#include "layer_4_computation/flavors/result.hpp"
#include "layer_4_computation/flavors/verification.hpp"

using namespace q3;

/**
 * ── Verification flavor ──────────────────────────────────────────────
 *
 * Restarts an SCF run from a submitted charge density and reports the energy
 * of every iteration, so a third party can check that the density really is
 * the converged one rather than taking the published total energy on trust.
 *
 * The physics of a restart needs a real SCF run and lives in the integration
 * suite. What is covered here is everything a caller depends on before the
 * physics starts: that the flavor is reachable by name, that it refuses inputs
 * it cannot honour, and that the trajectory reaches results.json under the key
 * consumers read.
 */

namespace {

InputData make_verification_input(const std::string& output_directory, int maximum_iterations) {
    InputData input;
    input.output_directory = output_directory;
    input.prefix = "silicon";
    input.maximum_iterations = maximum_iterations;
    return input;
}

} // namespace

TEST(VerificationFlavorFactory, IsReachableByName) {
    auto flavor = create_flavor("verification");

    ASSERT_NE(flavor, nullptr);
    EXPECT_NE(dynamic_cast<VerificationFlavor*>(flavor.get()), nullptr);
}

TEST(VerificationFlavorFactory, UnsupportedTypeNamesVerificationAsAnOption) {
    try {
        create_flavor("restart");
        FAIL() << "an unsupported calculation type must be refused";
    } catch (const std::runtime_error& refusal) {
        EXPECT_NE(std::string(refusal.what()).find("verification"), std::string::npos);
    }
}

TEST(VerificationFlavor, RefusesAnIterationBudgetOfOne) {
    /**
     * One energy cannot show whether a density moved, so accepting the input
     * would produce a report that always says "stable" -- the failure mode the
     * flavor exists to catch.
     */
    VerificationFlavor flavor;
    auto input = make_verification_input("/nonexistent", 1);

    try {
        flavor.run(input);
        FAIL() << "a single-iteration budget must be refused";
    } catch (const std::runtime_error& refusal) {
        EXPECT_NE(std::string(refusal.what()).find("at least 2 iterations"), std::string::npos);
    }
}

TEST(VerificationFlavor, SaysWhereItLookedWhenTheCheckpointIsMissing) {
    VerificationFlavor flavor;
    auto missing_directory =
        (std::filesystem::temp_directory_path() / "q3-verification-absent").string();
    auto input = make_verification_input(missing_directory, 3);

    try {
        flavor.run(input);
        FAIL() << "a missing checkpoint must be refused";
    } catch (const std::runtime_error& refusal) {
        const std::string message = refusal.what();
        EXPECT_NE(message.find("effective-potential.q3cp"), std::string::npos);
        EXPECT_NE(message.find(missing_directory), std::string::npos);
    }
}

/**
 * ── Trajectory serialization ─────────────────────────────────────────
 *
 * The key is `scf_energies_ry`, which is what the Q3 Ledger restart verifier
 * reads. Both serialization entry points must agree, since to_json_string() is
 * what the Python binding returns and write_results_json() is what a
 * command-line run leaves on disk.
 */

TEST(FlavorResultScfEnergies, ToJsonStringCarriesTheTrajectory) {
    FlavorResult result;
    result.flavor = "verification";
    result.scf_energies_in_ry = {-15.8547, -15.8547, -15.8547};

    nlohmann::json parsed = nlohmann::json::parse(result.to_json_string());

    ASSERT_TRUE(parsed.contains("scf_energies_ry"));
    EXPECT_EQ(parsed["scf_energies_ry"].size(), 3u);
    EXPECT_DOUBLE_EQ(parsed["scf_energies_ry"][0].get<Real>(), -15.8547);
}

TEST(FlavorResultScfEnergies, WriteResultsJsonCarriesTheTrajectory) {
    FlavorResult result;
    result.flavor = "verification";
    result.scf_energies_in_ry = {-15.1, -15.7};

    auto output_path = (std::filesystem::temp_directory_path() / "q3-verification-results.json");
    result.write_results_json(output_path.string());
    std::ifstream input_file(output_path);
    nlohmann::json parsed = nlohmann::json::parse(input_file);
    input_file.close();
    std::filesystem::remove(output_path);

    ASSERT_TRUE(parsed.contains("scf_energies_ry"));
    EXPECT_EQ(parsed["scf_energies_ry"].size(), 2u);
}

TEST(FlavorResultScfEnergies, IsAbsentWhenNoTrajectoryWasRecorded) {
    /**
     * Absent rather than empty, so a consumer can tell a flavor that does not
     * report a trajectory from one that reported a flat one -- the difference
     * between "not checked" and "checked and stable".
     */
    FlavorResult result;
    result.flavor = "self_consistent_field";

    nlohmann::json parsed = nlohmann::json::parse(result.to_json_string());

    EXPECT_FALSE(parsed.contains("scf_energies_ry"));
}
