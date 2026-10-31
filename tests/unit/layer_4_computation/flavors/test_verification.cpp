#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <string>

#include <nlohmann/json.hpp>

#include "generated/esse/properties/workflow/convergence_electronic.hpp"

#include "io/input/data.hpp"
#include "io/output/results.hpp"
#include "layer_4_computation/flavors/factory.hpp"
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

TEST(VerificationFlavorFactory, RefusesAnEntryPointsExecutionPolicy) {
    /**
     * The MPI and GPU binaries hand every flavor their iterator factory; a
     * verification would drop it and run serially, so it is refused instead.
     */
    IteratorFactory entry_point_factory = [](const InputData& input) {
        return create_iterator(input);
    };
    try {
        create_flavor("verification", entry_point_factory);
        FAIL() << "a verification must not silently drop the entry point's setup";
    } catch (const std::runtime_error& refusal) {
        EXPECT_NE(std::string(refusal.what()).find("serial q3"), std::string::npos)
            << refusal.what();
    }
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
 * The trajectory is ESSE's convergence_electronic record under "properties",
 * one row of total energies in Ry, which is what the Q3 Ledger restart
 * verifier reads. Both serialization entry points must agree, since results_json_string() is
 * what the Python binding returns and write_results_json() is what a
 * command-line run leaves on disk.
 */

namespace {

nlohmann::json convergence_record(const nlohmann::json& results) {
    for (const auto& record : results["properties"]["records"]) {
        if (record["name"] == "convergence_electronic") {
            return record;
        }
    }
    return nullptr;
}

/// Results as the verification flavor lists its trajectory: one convergence_electronic row.
generated::Results results_with_trajectory(const std::vector<double>& energies_in_ry) {
    generated::Results results = new_results("verification");
    set_record(results,
               generated::ConvergenceElectronic{
                   generated::ConvergenceElectronicNameEnum::CONVERGENCE_ELECTRONIC,
                   generated::ConvergenceElectronicUnitsEnum::RY,
                   {energies_in_ry}});
    return results;
}

} // namespace

TEST(ResultsTrajectory, ResultsJsonStringCarriesTheTrajectory) {
    const generated::Results results = results_with_trajectory({-15.8547, -15.8547, -15.8547});

    nlohmann::json parsed = nlohmann::json::parse(results_json_string(results));

    const nlohmann::json trajectory = convergence_record(parsed);
    ASSERT_FALSE(trajectory.is_null());
    EXPECT_EQ(trajectory["units"], "Ry");
    EXPECT_EQ(trajectory["data"][0].size(), 3u);
    EXPECT_DOUBLE_EQ(trajectory["data"][0][0].get<Real>(), -15.8547);
}

TEST(ResultsTrajectory, WriteResultsJsonCarriesTheTrajectory) {
    const generated::Results results = results_with_trajectory({-15.1, -15.7});

    auto output_path = (std::filesystem::temp_directory_path() / "q3-verification-results.json");
    write_results_json(results, output_path.string());
    std::ifstream input_file(output_path);
    nlohmann::json parsed = nlohmann::json::parse(input_file);
    input_file.close();
    std::filesystem::remove(output_path);

    const nlohmann::json trajectory = convergence_record(parsed);
    ASSERT_FALSE(trajectory.is_null());
    EXPECT_EQ(trajectory["data"][0].size(), 2u);
}

TEST(ResultsTrajectory, IsAbsentWhenNoTrajectoryWasRecorded) {
    /**
     * Absent rather than empty, so a consumer can tell a flavor that does not
     * report a trajectory from one that reported a flat one -- the difference
     * between "not checked" and "checked and stable".
     */
    const generated::Results results = new_results("self_consistent_field");

    nlohmann::json parsed = nlohmann::json::parse(results_json_string(results));

    EXPECT_TRUE(convergence_record(parsed).is_null());
}
