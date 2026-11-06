#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <utility>

#include "layer_4_computation/diagonalizer/planewave/internal/hamiltonian_applicator/gpu_fft_batch.hpp"

using q3::diagonalizer::planewave::fft_batch_band_count;
using q3::diagonalizer::planewave::maximum_fft_batch_workspace_elements;

namespace {

/// Passes needed to cover `vectors` bands, `per_pass` at a time.
int passes_for(int vectors, int per_pass) {
    return (vectors + per_pass - 1) / per_pass;
}

} // namespace

TEST(FFTBatchBandCount, AllBandsInOnePassWhenTheyFit) {
    EXPECT_EQ(fft_batch_band_count(48 * 48 * 48, 16), 16);
    // 144³ × 616 is below 2³¹ but above the budget, so it is split too.
    EXPECT_LT(fft_batch_band_count(144 * 144 * 144, 616), 616);
}

TEST(FFTBatchBandCount, TheReportedOverflowIsSplitBelowTheBudget) {
    // camsai/q3#4: 144³ × 800 and 162³ × 600 overflow a 32-bit element count.
    for (auto [grid, vectors] :
         {std::pair{144 * 144 * 144, 800}, std::pair{162 * 162 * 162, 600}}) {
        ASSERT_GT(std::int64_t{grid} * vectors, std::numeric_limits<int>::max());
        int per_pass = fft_batch_band_count(grid, vectors);
        EXPECT_GE(per_pass, 1);
        EXPECT_LE(std::int64_t{grid} * per_pass, maximum_fft_batch_workspace_elements);
        EXPECT_GE(per_pass * passes_for(vectors, per_pass), vectors);
    }
}

TEST(FFTBatchBandCount, BandsAreSharedEvenlyOverTheFewestPasses) {
    // 1000 bands, 300 fit: four passes, so 250 each rather than 300+300+300+100.
    EXPECT_EQ(fft_batch_band_count(10, 1000, 3000), 250);
    // Seven bands, three fit: three passes of three, three and one.
    EXPECT_EQ(fft_batch_band_count(10, 7, 30), 3);
}

TEST(FFTBatchBandCount, AGridLargerThanTheBudgetStillGetsOneBand) {
    EXPECT_EQ(fft_batch_band_count(1000, 5, 10), 1);
}

TEST(FFTBatchBandCount, NoBandsMeansNoPasses) {
    EXPECT_EQ(fft_batch_band_count(1000, 0), 0);
}
