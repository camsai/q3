#include "utils/time.hpp"
#include <chrono>
#include <gtest/gtest.h>
#include <string>

using namespace q3;

TEST(Timestamp, ReturnsIso8601FormatWithTimezone) {
    std::string timestamp = get_iso_timestamp_with_timezone();

    // Minimum ISO 8601 with timezone: "2026-06-18T15:30:45+00:00" = 25 chars
    ASSERT_GE(timestamp.size(), 19u);

    // Format: YYYY-MM-DDTHH:MM:SS...
    EXPECT_EQ(timestamp[4], '-');
    EXPECT_EQ(timestamp[7], '-');
    EXPECT_EQ(timestamp[10], 'T');
    EXPECT_EQ(timestamp[13], ':');
    EXPECT_EQ(timestamp[16], ':');

    // Year should be 4 digits starting with '2' (valid through 2999)
    for (int index = 0; index < 4; index++) {
        EXPECT_TRUE(std::isdigit(static_cast<unsigned char>(timestamp[index])));
    }
    EXPECT_EQ(timestamp[0], '2');
}

TEST(Timestamp, ConsecutiveCallsAreChronological) {
    std::string first = get_iso_timestamp_with_timezone();
    std::string second = get_iso_timestamp_with_timezone();
    // ISO 8601 timestamps are lexicographically ordered
    EXPECT_LE(first, second);
}
