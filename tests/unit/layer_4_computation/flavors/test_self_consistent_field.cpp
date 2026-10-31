#include <gtest/gtest.h>

#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "layer_4_computation/flavors/self_consistent_field.hpp"

using namespace q3;

/**
 * ── What an SCF reports ──────────────────────────────────────────────
 *
 * Dispersion energy, the per-atom force breakdown, the scalar pressure, both
 * magnetizations, DFT+U Tr[n] and the trajectory were printed by the run and
 * nowhere else, so anything that wanted them had to regular-expression them
 * back out of stdout. SelfConsistentFieldFlavor lists them now, through the
 * helpers below, which the bindings' SCFSolver shares; these tests pin the
 * shapes that schemas/q3/results.json describes, and the omit-rather-than-zero
 * rule that lets a reader tell "not applicable" from "came out at zero".
 */

namespace {

const nlohmann::json* find_property(const nlohmann::json& serialized, const std::string& name) {
    for (const auto& property : serialized["properties"]["records"]) {
        if (property["name"] == name) {
            return &property;
        }
    }
    return nullptr;
}

/// One SCF iteration as the iterator's history records it.
struct Iteration {
    Real total_energy_in_ry;
    Real density_residual;
};

/// A Hubbard manifold, as far as the occupation report reads one.
struct Manifold {
    int atom_index;
};

nlohmann::json serialized(const generated::Results& results) {
    return nlohmann::json::parse(results_json_string(results));
}

} // namespace

TEST(SelfConsistentFieldRecords, TheTrajectoryIsAnEsseConvergenceRecordWithItsResiduals) {
    generated::Results results = new_results("self_consistent_field");
    SelfConsistentFieldFlavor::record_convergence(
        results, std::vector<Iteration>{{-15.0, 1.0e-2}, {-15.7, 1.0e-5}, {-15.8, 1.0e-9}});

    const nlohmann::json parsed = serialized(results);

    const nlohmann::json* convergence = find_property(parsed, "convergence_electronic");
    ASSERT_NE(convergence, nullptr);
    EXPECT_EQ((*convergence)["units"], "Ry");
    ASSERT_EQ((*convergence)["data"].size(), 1u);
    EXPECT_DOUBLE_EQ((*convergence)["data"][0][2], -15.8);
    ASSERT_EQ(parsed["runtime_log"]["density_residuals"].size(), 3u);
    EXPECT_DOUBLE_EQ(parsed["runtime_log"]["density_residuals"][2], 1.0e-9);
}

TEST(SelfConsistentFieldRecords, NoConvergenceRecordWithoutATrajectory) {
    generated::Results results = new_results("self_consistent_field");
    SelfConsistentFieldFlavor::record_convergence(results, std::vector<Iteration>{});

    const nlohmann::json parsed = serialized(results);

    EXPECT_EQ(find_property(parsed, "convergence_electronic"), nullptr);
    EXPECT_FALSE(parsed["runtime_log"].contains("density_residuals"));
}

TEST(SelfConsistentFieldRecords, BothMomentsAreListedWhenSpinPolarized) {
    generated::Results results = new_results("self_consistent_field");
    SelfConsistentFieldFlavor::record_magnetization(results, Magnetization{0.0, 2.6});

    const nlohmann::json parsed = serialized(results);

    const nlohmann::json* total = find_property(parsed, "magnetization_total");
    const nlohmann::json* absolute = find_property(parsed, "magnetization_absolute");
    ASSERT_NE(total, nullptr);
    ASSERT_NE(absolute, nullptr);
    EXPECT_DOUBLE_EQ((*total)["value"], 0.0);
    EXPECT_DOUBLE_EQ((*absolute)["value"], 2.6);
    EXPECT_EQ((*total)["units"], "uB");
}

TEST(SelfConsistentFieldRecords, MomentsAreOmittedRatherThanEmittedAsZero) {
    /**
     * A zero total is what a converged antiferromagnet reports, so emitting
     * zeros for a spin-unpolarized run would make the two indistinguishable.
     */
    generated::Results results = new_results("self_consistent_field");
    SelfConsistentFieldFlavor::record_magnetization(results, std::nullopt);

    const nlohmann::json parsed = serialized(results);

    EXPECT_EQ(find_property(parsed, "magnetization_total"), nullptr);
    EXPECT_EQ(find_property(parsed, "magnetization_absolute"), nullptr);
}

TEST(SelfConsistentFieldRecords, TheOccupationTraceIsListedPerManifold) {
    generated::Results results = new_results("self_consistent_field");
    SelfConsistentFieldFlavor::record_hubbard_occupations(
        results, {{4.8, 4.03}}, std::vector<Manifold>{{0}});

    const nlohmann::json parsed = serialized(results);

    const nlohmann::json& occupations = parsed["properties"]["hubbard_occupations"];
    ASSERT_EQ(occupations.size(), 1u);
    EXPECT_EQ(occupations[0]["atom_index"], 0);
    EXPECT_DOUBLE_EQ(occupations[0]["occupation_trace"][0], 4.8);
    EXPECT_DOUBLE_EQ(occupations[0]["occupation_trace"][1], 4.03);
}

TEST(SelfConsistentFieldRecords, NoOccupationsWithoutHubbardManifolds) {
    generated::Results results = new_results("self_consistent_field");
    SelfConsistentFieldFlavor::record_hubbard_occupations(results, {}, std::vector<Manifold>{});

    EXPECT_FALSE(serialized(results)["properties"].contains("hubbard_occupations"));
}

TEST(SelfConsistentFieldRecords, ForcesAreListedWithTheirPerAtomBreakdown) {
    generated::Results results = new_results("self_consistent_field");
    AtomicForcesContributions contributions;
    contributions.ewald = {{0.3, 0.0, 0.0}, {-0.3, 0.0, 0.0}};
    contributions.nonlocal = {{0.2, 0.0, 0.0}, {-0.2, 0.0, 0.0}};
    SelfConsistentFieldFlavor::record_forces(
        results, {{0.5, 0.0, 0.0}, {-0.5, 0.0, 0.0}}, contributions);

    const nlohmann::json parsed = serialized(results);

    const nlohmann::json* forces = find_property(parsed, "atomic_forces");
    ASSERT_NE(forces, nullptr);
    EXPECT_DOUBLE_EQ((*forces)["values"][1]["value"][0], -0.5);
    const nlohmann::json expected_ewald = {
        {"name", "atomic_forces"},
        {"units", "Ry/a.u."},
        {"values",
         {{{"id", 1}, {"value", {0.3, 0.0, 0.0}}}, {{"id", 2}, {"value", {-0.3, 0.0, 0.0}}}}},
    };
    EXPECT_EQ(parsed["properties"]["force_contributions"]["ewald"], expected_ewald);
    EXPECT_EQ(parsed["properties"]["force_contributions"]["nonlocal"]["values"][0]["value"][0],
              0.2);
}

TEST(SelfConsistentFieldRecords, ComputedStressIsListedWithItsPressureAndBreakdown) {
    generated::Results results = new_results("self_consistent_field");
    StressTensorTotal total;
    StressTensorContributions contributions;
    for (int axis = 0; axis < 3; axis++) {
        total.values[axis][axis] = -5.0;
        contributions.kinetic[axis][axis] = -5.0;
    }
    total.pressure = -5.0;
    SelfConsistentFieldFlavor::record_stress(results, total, contributions);

    const nlohmann::json parsed = serialized(results);

    const nlohmann::json* stress = find_property(parsed, "stress_tensor");
    ASSERT_NE(stress, nullptr);
    EXPECT_EQ((*stress)["units"], "kbar");
    EXPECT_DOUBLE_EQ((*stress)["value"][2][2], -5.0);
    const nlohmann::json* pressure = find_property(parsed, "pressure");
    ASSERT_NE(pressure, nullptr);
    EXPECT_DOUBLE_EQ((*pressure)["value"], -5.0);
    EXPECT_EQ((*pressure)["units"], "kbar");
    EXPECT_DOUBLE_EQ(parsed["properties"]["stress_contributions"]["kinetic"]["value"][1][1], -5.0);
    EXPECT_TRUE(parsed["properties"]["stress_contributions"].contains("local_pseudopotential"));
}

TEST(SelfConsistentFieldRecords, EveryReasonTheStressIsIncompleteIsKept) {
    generated::Results results = new_results("self_consistent_field");
    SelfConsistentFieldFlavor::record_incomplete_stress(results, "no DFT+U stress term");
    SelfConsistentFieldFlavor::record_incomplete_stress(results, "distributed band group");

    const nlohmann::json parsed = serialized(results);

    EXPECT_EQ(parsed["runtime_log"]["incomplete"]["stress"],
              "no DFT+U stress term; distributed band group");
    EXPECT_FALSE(parsed["runtime_log"]["incomplete"].contains("forces"));
}
