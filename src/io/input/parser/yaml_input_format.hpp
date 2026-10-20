#pragma once

/**
 * YAMLInputFormat — Q3's own YAML input, and the fallback format.
 *
 * Any path that is not a directory is a YAML input unless a more specific
 * format claims it, which is how inputs have always been read: a file with no
 * recognised extension goes to the YAML parser, and a parse error there says
 * what is wrong with it.
 */

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "io/input/parser/input_format.hpp"
#include "io/input/parser/yaml.hpp"

namespace q3 {

/// Q3's own YAML input, the fallback for any file nothing more specific claims.
class YAMLInputFormat : public InputFormat {
  public:
    /// Ranked lowest: the fallback answers only when nothing more specific does.
    static constexpr int fallback_detection_priority = 0;

    /// "yaml", and the "yml" extension spelling.
    [[nodiscard]] std::vector<std::string> names() const override {
        return {"yaml", "yml"};
    }

    /// How a log line names a YAML input.
    [[nodiscard]] std::string display_name() const override {
        return "a Q3 YAML input";
    }

    /// The lowest rank, fallback_detection_priority.
    [[nodiscard]] int detection_priority() const override {
        return fallback_detection_priority;
    }

    /// Any path that is not a directory.
    [[nodiscard]] bool recognizes(const std::filesystem::path& path) const override {
        return !std::filesystem::is_directory(path);
    }

    /// A YAMLInputParser.
    [[nodiscard]] std::unique_ptr<InputParser> create_parser() const override {
        return std::make_unique<YAMLInputParser>();
    }
};

} // namespace q3
