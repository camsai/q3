#include <gtest/gtest.h>

#include "io/addon_registration.hpp"

int main(int argc, char** argv) {
    q3::addons::register_all();
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
