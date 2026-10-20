#include <gtest/gtest.h>

#include <stdexcept>
#include <string>

#include "io/volumetric/factory.hpp"

using namespace q3;

TEST(VolumetricFieldWriterFactory, EachFormatNamesItsOwnFile) {
    EXPECT_EQ(create_volumetric_field_writer("cube")->default_file_name("si_density"),
              "si_density.cube");
    EXPECT_EQ(create_volumetric_field_writer("xsf")->default_file_name("si_density"),
              "si_density.xsf");
}

TEST(VolumetricFieldWriterFactory, AnUnknownFormatIsRefused) {
    EXPECT_THROW(create_volumetric_field_writer("vtk"), std::runtime_error);
}
