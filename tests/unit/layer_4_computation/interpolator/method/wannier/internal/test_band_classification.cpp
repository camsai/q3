#include <gtest/gtest.h>

#include <vector>

#include "layer_4_computation/interpolator/method/wannier/internal/band_classification.hpp"

using namespace q3;

TEST(BandClassification, EveryBandKeptIsTheClassificationOfAnIsolatedManifold) {
    const BandClassification classification = BandClassification::every_band_kept(3, 4);
    ASSERT_EQ(classification.kpoints.size(), 3u);
    for (const KPointBandClassification& entry : classification.kpoints) {
        EXPECT_EQ(entry.number_kept(), 4);
        EXPECT_EQ(entry.number_frozen(), 0);
        EXPECT_EQ(entry.kept_bands, (std::vector<int>{0, 1, 2, 3}));
        EXPECT_EQ(entry.roles, std::vector<BandRole>(4, BandRole::Candidate));
    }
    EXPECT_FALSE(classification.summary.metallic);
    EXPECT_TRUE(classification.summary.bands_excluded_from_freeze.empty());
}
