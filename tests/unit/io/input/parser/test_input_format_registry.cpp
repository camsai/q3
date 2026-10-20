/**
 * The input-format registry: the formats core seeds it with, how a path is
 * matched to one by priority, and how a run given no input path finds one.
 */

#include "io/input/parser/factory.hpp"
#include "io/input/parser/input_format_registry.hpp"
#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>

using namespace q3;

namespace {

/**
 * A format standing in for one an add-on registers: it claims files named
 * exactly `widget.cfg`, and directories holding one, at the highest priority.
 */
class WidgetInputFormat : public InputFormat {
  public:
    [[nodiscard]] std::vector<std::string> names() const override {
        return {"registry_test_widget"};
    }
    [[nodiscard]] std::string display_name() const override {
        return "a widget configuration";
    }
    [[nodiscard]] int detection_priority() const override {
        return 1000;
    }
    [[nodiscard]] bool recognizes(const std::filesystem::path& path) const override {
        return path.filename() == "widget.cfg";
    }
    [[nodiscard]] bool
    claims_working_directory(const std::filesystem::path& directory) const override {
        return std::filesystem::exists(directory / "widget.cfg");
    }
    [[nodiscard]] std::unique_ptr<InputParser> create_parser() const override {
        return std::make_unique<YAMLInputParser>();
    }
};

void register_widget_format_once() {
    if (input_format_registry().find("registry_test_widget") == nullptr)
        input_format_registry().add(std::make_unique<WidgetInputFormat>());
}

} // namespace

TEST(InputFormatRegistryTest, CoreSeedsYamlAndQuantumEspresso) {
    EXPECT_NE(input_format_registry().find("yaml"), nullptr);
    EXPECT_NE(input_format_registry().find("yml"), nullptr);
    EXPECT_NE(input_format_registry().find("qe"), nullptr);
    EXPECT_NE(input_format_registry().find("espresso"), nullptr);
}

TEST(InputFormatRegistryTest, ExtensionOutranksTheYamlFallback) {
    EXPECT_EQ(detect_input_format("scf.in")->names().front(), "qe");
    EXPECT_EQ(detect_input_format("scf.qe")->names().front(), "qe");
    EXPECT_EQ(detect_input_format("scf.yml")->names().front(), "yaml");
    EXPECT_EQ(detect_input_format("anything.json")->names().front(), "yaml");
}

TEST(InputFormatRegistryTest, AHigherPriorityFormatClaimsThePathsItRecognises) {
    register_widget_format_once();
    EXPECT_EQ(detect_input_format("widget.cfg")->names().front(), "registry_test_widget");
    EXPECT_EQ(detect_input_format("scf.in")->names().front(), "qe");
}

TEST(InputFormatRegistryTest, NoFormatClaimsAnUnrecognisedDirectory) {
    const auto directory = std::filesystem::temp_directory_path() / "q3_no_input_format_here";
    std::filesystem::create_directories(directory);
    EXPECT_EQ(detect_input_format(directory), nullptr);
    EXPECT_THROW(input_format_for(directory.string()), std::runtime_error);
    std::filesystem::remove_all(directory);
}

TEST(InputFormatRegistryTest, AnUnknownFormatNameIsRefusedWithTheRegisteredNames) {
    try {
        input_format_for("scf.yml", "no_such_format");
        FAIL() << "an unknown format name must be refused";
    } catch (const std::runtime_error& error) {
        EXPECT_NE(std::string(error.what()).find("yaml"), std::string::npos);
        EXPECT_NE(std::string(error.what()).find("qe"), std::string::npos);
    }
}

TEST(InputFormatRegistryTest, AFormatClaimingTheDirectoryAnswersAnArgumentLessRun) {
    register_widget_format_once();
    const auto directory = std::filesystem::temp_directory_path() / "q3_widget_directory";
    std::filesystem::create_directories(directory);
    std::ofstream(directory / "widget.cfg").put('\n');
    const InputFormat* format = format_claiming_directory(directory);
    ASSERT_NE(format, nullptr);
    EXPECT_EQ(format->display_name(), "a widget configuration");
    std::filesystem::remove_all(directory);
}
