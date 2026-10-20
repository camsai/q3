#include <gtest/gtest.h>

#include <memory>
#include <stdexcept>

#include "io/interpolation/factory.hpp"

using namespace q3;

TEST(InterpolationExporterFactory, NoneAndEmptyResolveToTheNullExporter) {
    EXPECT_EQ(create_interpolation_exporter("none")->name(), "none");
    EXPECT_EQ(create_interpolation_exporter("")->name(), "none");
}

TEST(InterpolationExporterFactory, Wannier90ResolvesByName) {
    EXPECT_EQ(create_interpolation_exporter("wannier90")->name(), "wannier90");
}

TEST(InterpolationExporterFactory, RefusesAnUnknownFormat) {
    EXPECT_THROW(create_interpolation_exporter("wannier95"), std::runtime_error);
}

TEST(InterpolationExporterFactory, TheNullExporterWritesNothing) {
    /// No record content is needed for a no-op, and none may be required.
    InterpolationRecord record;
    EXPECT_NO_THROW(create_interpolation_exporter("none")->export_to_directory(
        record, "/nonexistent/directory", "si"));
}
