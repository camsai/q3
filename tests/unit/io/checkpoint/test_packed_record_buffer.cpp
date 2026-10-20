#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>

#include "io/checkpoint/packed_record_buffer.hpp"

using namespace q3;

TEST(PackedRecordBuffer, MixedWidthFieldsAreLaidOutWithoutPadding) {
    PackedRecordBuffer buffer;
    buffer.append(static_cast<int32_t>(7));
    buffer.append(2.5);
    buffer.append(static_cast<int32_t>(-3));

    ASSERT_EQ(buffer.size(), 4u + 8u + 4u);
    int32_t first = 0;
    double second = 0.0;
    int32_t third = 0;
    std::memcpy(&first, buffer.data(), 4);
    std::memcpy(&second, buffer.data() + 4, 8);
    std::memcpy(&third, buffer.data() + 12, 4);
    EXPECT_EQ(first, 7);
    EXPECT_DOUBLE_EQ(second, 2.5);
    EXPECT_EQ(third, -3);
}

TEST(PackedRecordBuffer, EmptyBufferHasZeroSize) {
    PackedRecordBuffer buffer;
    EXPECT_EQ(buffer.size(), 0u);
}
