#include "layer_0_core/profiling/memory.hpp"
#include <gtest/gtest.h>
#include <vector>

using namespace q3;

// ─── Sampling ───────────────────────────────────────────────────────────────

TEST(ResidentMemorySample, SampleReportsSomethingOnSupportedPlatforms) {
    ResidentMemorySample sample = ResidentMemorySample::sample();
#if defined(__linux__) || defined(__APPLE__)
    EXPECT_TRUE(sample.available);
    EXPECT_GT(sample.current_bytes, 0u);
    EXPECT_GE(sample.peak_bytes, sample.current_bytes);
#else
    EXPECT_FALSE(sample.available);
#endif
}

/**
 * The point of the sampler is to notice an allocation. A quarter-gibibyte
 * buffer, touched so the pages are resident, has to move the reading.
 *
 * The buffer is summed into a volatile sink because an optimizing build is
 * otherwise free to delete a write-only allocation, which is exactly what it
 * did the first time this test was written.
 */
TEST(ResidentMemorySample, PeakRisesAfterALargeResidentAllocation) {
    ResidentMemorySample before = ResidentMemorySample::sample();
    if (!before.available) {
        GTEST_SKIP() << "no resident-memory source on this platform";
    }
    const std::size_t allocation_bytes = 256u * 1024u * 1024u;
    static volatile unsigned long long sink = 0;
    {
        std::vector<unsigned char> buffer(allocation_bytes, 1u);
        for (std::size_t offset = 0; offset < allocation_bytes; offset += 4096) {
            buffer[offset] = static_cast<unsigned char>(offset);
        }
        unsigned long long total = 0;
        for (std::size_t offset = 0; offset < allocation_bytes; offset += 4096) {
            total += buffer[offset];
        }
        sink = total;
        ResidentMemorySample during = ResidentMemorySample::sample();
        EXPECT_GT(during.current_bytes, before.current_bytes + allocation_bytes / 2);
        EXPECT_GE(during.peak_bytes, during.current_bytes);
    }
    ResidentMemorySample after = ResidentMemorySample::sample();
    EXPECT_GE(after.peak_bytes, before.peak_bytes);
}

// ─── Formatting ─────────────────────────────────────────────────────────────

TEST(ResidentMemorySample, GibibyteConversionUsesBinaryUnits) {
    EXPECT_DOUBLE_EQ(ResidentMemorySample::to_gibibytes(1073741824u), 1.0);
    EXPECT_DOUBLE_EQ(ResidentMemorySample::to_gibibytes(0u), 0.0);
}

TEST(ResidentMemorySample, FormatCarriesTwoDecimalsAndAUnit) {
    EXPECT_EQ(ResidentMemorySample::format_gibibytes(1073741824u), "1.00 GiB");
    EXPECT_EQ(ResidentMemorySample::format_gibibytes(0u), "0.00 GiB");
}
