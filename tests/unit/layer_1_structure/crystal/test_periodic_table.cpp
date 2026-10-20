#include <gtest/gtest.h>

#include "layer_1_structure/crystal/periodic_table.hpp"

using namespace q3;

// ─── PeriodicTable Tests ────────────────────────────────────────────────────

TEST(PeriodicTable, HydrogenReturnsAtomicNumber1) {
    EXPECT_EQ(PeriodicTable::atomic_number_by_element_symbol("H"), 1);
}

TEST(PeriodicTable, HeliumReturnsAtomicNumber2) {
    EXPECT_EQ(PeriodicTable::atomic_number_by_element_symbol("He"), 2);
}

TEST(PeriodicTable, SiliconReturnsAtomicNumber14) {
    EXPECT_EQ(PeriodicTable::atomic_number_by_element_symbol("Si"), 14);
}

TEST(PeriodicTable, CopperReturnsAtomicNumber29) {
    EXPECT_EQ(PeriodicTable::atomic_number_by_element_symbol("Cu"), 29);
}

TEST(PeriodicTable, IronReturnsAtomicNumber26) {
    EXPECT_EQ(PeriodicTable::atomic_number_by_element_symbol("Fe"), 26);
}

TEST(PeriodicTable, GoldReturnsAtomicNumber79) {
    EXPECT_EQ(PeriodicTable::atomic_number_by_element_symbol("Au"), 79);
}

TEST(PeriodicTable, OganessonReturnsAtomicNumber118) {
    EXPECT_EQ(PeriodicTable::atomic_number_by_element_symbol("Og"), 118);
}

TEST(PeriodicTable, TrailingDigitsAreStripped) {
    EXPECT_EQ(PeriodicTable::atomic_number_by_element_symbol("Si1"), 14);
}

TEST(PeriodicTable, UnknownElementReturnsZero) {
    EXPECT_EQ(PeriodicTable::atomic_number_by_element_symbol("Xx"), 0);
}

TEST(PeriodicTable, EmptyStringReturnsZero) {
    EXPECT_EQ(PeriodicTable::atomic_number_by_element_symbol(""), 0);
}
