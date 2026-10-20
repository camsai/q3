/**
 * write_companion_outputs(): a finished run goes back to the format it was
 * read from, and to no other.
 */

#include "io/input/parser/input_format_registry.hpp"
#include "io/output/companion_outputs.hpp"
#include <gtest/gtest.h>

#include <memory>
#include <string>

using namespace q3;

namespace {

int widget_companion_output_calls = 0;

/**
 * A format standing in for one with a companion-output convention: it counts the runs it is
 * handed.
 */
class CompanionWidgetInputFormat : public InputFormat {
  public:
    [[nodiscard]] std::vector<std::string> names() const override {
        return {"companion_test_widget"};
    }
    [[nodiscard]] std::string display_name() const override {
        return "a companion widget";
    }
    [[nodiscard]] int detection_priority() const override {
        return -1;
    }
    [[nodiscard]] bool recognizes(const std::filesystem::path& /*path*/) const override {
        return false;
    }
    [[nodiscard]] std::unique_ptr<InputParser> create_parser() const override {
        return std::make_unique<YAMLInputParser>();
    }
    void write_companion_outputs(const InputData& /*input*/,
                                 const FlavorResult& /*result*/) const override {
        widget_companion_output_calls++;
    }
};

} // namespace

TEST(CompanionOutputsTest, OnlyTheFormatTheRunWasReadFromWritesItsOutputs) {
    if (input_format_registry().find("companion_test_widget") == nullptr)
        input_format_registry().add(std::make_unique<CompanionWidgetInputFormat>());
    widget_companion_output_calls = 0;
    const FlavorResult result;

    InputData read_from_widget;
    read_from_widget.source_format = "companion_test_widget";
    write_companion_outputs(read_from_widget, result);
    EXPECT_EQ(widget_companion_output_calls, 1);

    InputData read_from_yaml;
    read_from_yaml.source_format = "yaml";
    write_companion_outputs(read_from_yaml, result);
    EXPECT_EQ(widget_companion_output_calls, 1);
}

TEST(CompanionOutputsTest, ARunAssembledInCodeWritesNoCompanionOutputs) {
    const InputData assembled;
    EXPECT_TRUE(assembled.source_format.empty());
    EXPECT_NO_THROW(write_companion_outputs(assembled, FlavorResult{}));
}
