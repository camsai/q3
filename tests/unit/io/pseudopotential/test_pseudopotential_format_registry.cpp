/**
 * The pseudopotential-format registry and the factory that reads through it:
 * UPF first, the format chosen from the file's content, and a clear refusal
 * when no registered format recognises a file.
 */

#include "io/pseudopotential/factory.hpp"
#include "io/pseudopotential/pseudopotential_format_registry.hpp"
#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>

using namespace q3;

namespace {

/// A format standing in for one an add-on registers: it claims files opening with "WIDGET".
class WidgetPseudopotentialFormat : public PseudopotentialFormat {
  public:
    [[nodiscard]] std::vector<std::string> names() const override {
        return {"registry_test_widget"};
    }
    [[nodiscard]] bool recognizes(const std::string& first_line) const override {
        return first_line.rfind("WIDGET", 0) == 0;
    }
    [[nodiscard]] int detection_priority() const override {
        return 1000;
    }
    [[nodiscard]] std::unique_ptr<Pseudopotential> read(const std::string& /*filename*/,
                                                        int dataset_index) const override {
        auto pseudopotential = std::make_unique<Pseudopotential>();
        pseudopotential->element = "W" + std::to_string(dataset_index);
        return pseudopotential;
    }
};

std::filesystem::path file_holding(const std::string& name, const std::string& first_line) {
    const auto path = std::filesystem::temp_directory_path() / name;
    std::ofstream(path) << first_line << "\n";
    return path;
}

} // namespace

TEST(PseudopotentialFormatRegistryTest, UpfIsRegisteredFirst) {
    const auto& entries = pseudopotential_format_registry().entries();
    ASSERT_FALSE(entries.empty());
    EXPECT_EQ(entries.front()->names().front(), "upf");
}

TEST(PseudopotentialFormatRegistryTest, TheFactoryHandsAFileToTheFormatThatRecognisesIt) {
    if (pseudopotential_format_registry().find("registry_test_widget") == nullptr)
        pseudopotential_format_registry().add(std::make_unique<WidgetPseudopotentialFormat>());
    const auto path = file_holding("q3_widget_dataset", "WIDGET dataset");
    EXPECT_EQ(PseudopotentialFactory::read(path.string(), 2)->element, "W2");
    std::filesystem::remove(path);
}

TEST(PseudopotentialFormatRegistryTest, AMissingFileIsRefused) {
    EXPECT_THROW(PseudopotentialFactory::read("/nonexistent/path/to/dataset"), std::runtime_error);
}
