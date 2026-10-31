#include <gtest/gtest.h>

#include <optional>
#include <vector>

#include "layer_5_properties/band_structure/band_edges.hpp"

using namespace q3;

namespace {

/// One k-point state of an eigenvalues_at_kpoints record: one channel, at `kpoint` when given.
generated::EigenvaluesAtKpointsEigenvaluesAtKPoint
kpoint_state(std::vector<double> energies_in_ry,
             std::optional<std::vector<double>> kpoint = std::nullopt,
             std::optional<std::vector<double>> occupations = std::nullopt) {
    generated::EigenvaluesAtKpointsBandEnergies channel;
    channel.energies = std::move(energies_in_ry);
    channel.occupations = std::move(occupations);
    generated::EigenvaluesAtKpointsEigenvaluesAtKPoint state;
    state.kpoint = std::move(kpoint);
    state.eigenvalues = {channel};
    return state;
}

/// Results listing these k-point states and a Fermi energy of zero, with their band edges added.
generated::Results
results_with_band_edges(std::vector<generated::EigenvaluesAtKpointsEigenvaluesAtKPoint> states) {
    generated::Results results = new_results("self_consistent_field");
    generated::EigenvaluesAtKpoints eigenvalues;
    eigenvalues.name = generated::EigenvaluesAtKpointsNameEnum::EIGENVALUES_AT_KPOINTS;
    eigenvalues.units = generated::EigenvaluesAtKpointsUnitsEnum::RY;
    eigenvalues.values = std::move(states);
    set_record(results, eigenvalues);
    set_fermi_energy(results, 0.0);
    add_band_edge_records(results);
    return results;
}

} // namespace

TEST(BandEdges, EdgesAndGapAreTakenAroundTheFermiEnergy) {
    const generated::Results results = results_with_band_edges(
        {kpoint_state({-1.0, -0.4, 0.6, 1.2}), kpoint_state({-0.9, -0.3, 0.8, 1.5})});

    const auto valence = find_record<generated::BandEdgeValenceMax>(results);
    const auto conduction = find_record<generated::BandEdgeConductionMin>(results);
    ASSERT_TRUE(valence.has_value());
    ASSERT_TRUE(conduction.has_value());
    EXPECT_DOUBLE_EQ(valence->value, -0.3);
    EXPECT_DOUBLE_EQ(conduction->value, 0.6);
    EXPECT_EQ(nlohmann::json(*valence)["units"], "Ry");
    EXPECT_FALSE(valence->kpoint.has_value());
    const auto gaps = find_record<generated::BandGaps>(results);
    ASSERT_TRUE(gaps.has_value());
    EXPECT_DOUBLE_EQ(gaps->values.back().value, 0.9);
}

TEST(BandEdges, EachEdgeCarriesTheKPointItLiesAt) {
    const generated::Results results =
        results_with_band_edges({kpoint_state({-1.0, -0.4, 0.6, 1.2}, {{0.0, 0.0, 0.0}}),
                                 kpoint_state({-0.9, -0.3, 0.8, 1.5}, {{0.5, 0.0, 0.0}})});

    const auto valence = find_record<generated::BandEdgeValenceMax>(results);
    const auto conduction = find_record<generated::BandEdgeConductionMin>(results);
    ASSERT_TRUE(valence && valence->kpoint);
    ASSERT_TRUE(conduction && conduction->kpoint);
    EXPECT_DOUBLE_EQ((*valence->kpoint)[0], 0.5);
    EXPECT_DOUBLE_EQ((*conduction->kpoint)[0], 0.0);
}

TEST(BandEdges, NoGapIsReportedWithoutAConductionEdge) {
    /**
     * A run asked for exactly its occupied manifold has no unoccupied level to
     * measure to; reporting a gap anyway would invent the half that was never
     * computed.
     */
    const generated::Results results = results_with_band_edges({kpoint_state({-1.0, -0.4})});

    const auto valence = find_record<generated::BandEdgeValenceMax>(results);
    ASSERT_TRUE(valence.has_value());
    EXPECT_DOUBLE_EQ(valence->value, -0.4);
    EXPECT_FALSE(find_record<generated::BandEdgeConductionMin>(results).has_value());
    EXPECT_FALSE(find_record<generated::BandGaps>(results).has_value());
}

TEST(BandEdges, NothingIsAddedWithoutEigenvalues) {
    generated::Results results = new_results("verification");
    set_fermi_energy(results, 0.0);

    add_band_edge_records(results);

    EXPECT_FALSE(find_record<generated::BandEdgeValenceMax>(results).has_value());
    EXPECT_FALSE(find_record<generated::BandEdgeConductionMin>(results).has_value());
    EXPECT_FALSE(find_record<generated::BandGaps>(results).has_value());
}

TEST(BandEdges, BandGapsSeparateTheDirectFromTheFundamentalGap) {
    /**
     * Valence maximum at the second k-point, conduction minimum at the first:
     * the fundamental ("indirect") gap is 0.6 − (−0.3) = 0.9, and the smallest
     * vertical gap is at the first k-point, 0.6 − (−0.4) = 1.0, not the
     * second's 0.8 − (−0.3) = 1.1.
     */
    const generated::Results results =
        results_with_band_edges({kpoint_state({-1.0, -0.4, 0.6, 1.2}, {{0.0, 0.0, 0.0}}),
                                 kpoint_state({-0.9, -0.3, 0.8, 1.5}, {{0.5, 0.0, 0.0}})});

    const auto gaps = find_record<generated::BandGaps>(results);
    ASSERT_TRUE(gaps.has_value());
    const nlohmann::json serialized = *gaps;
    ASSERT_EQ(serialized["values"].size(), 2u);
    const nlohmann::json& direct = serialized["values"][0];
    const nlohmann::json& indirect = serialized["values"][1];
    EXPECT_EQ(direct["type"], "direct");
    EXPECT_DOUBLE_EQ(direct["value"], 1.0);
    EXPECT_DOUBLE_EQ(direct["kpointValence"][0], 0.0);
    EXPECT_DOUBLE_EQ(direct["kpointConduction"][0], 0.0);
    EXPECT_EQ(indirect["type"], "indirect");
    EXPECT_DOUBLE_EQ(indirect["value"], 0.9);
    EXPECT_DOUBLE_EQ(indirect["kpointValence"][0], 0.5);
    EXPECT_DOUBLE_EQ(indirect["kpointConduction"][0], 0.0);
}

TEST(BandEdges, NoBandGapsForAFractionallyOccupiedMetal) {
    /**
     * Both band edges are still reported; the ESSE record, which would state
     * the level spacing between them as the material's gap, is left out.
     */
    const generated::Results results = results_with_band_edges(
        {kpoint_state({-0.4, -0.01, 0.01, 0.6}, std::nullopt, {{1.0, 0.7, 0.3, 0.0}})});

    EXPECT_TRUE(find_record<generated::BandEdgeConductionMin>(results).has_value());
    EXPECT_FALSE(find_record<generated::BandGaps>(results).has_value());
}
