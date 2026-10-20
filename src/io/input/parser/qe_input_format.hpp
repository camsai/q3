#pragma once

/**
 * QEInputFormat — a Quantum ESPRESSO pw.x namelist input.
 *
 * Recognised by its extension, `.in` or `.qe`, the two spellings pw.x inputs
 * are conventionally given.
 */

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "io/input/parser/input_format.hpp"
#include "io/input/parser/qe.hpp"

namespace q3 {

/// A pw.x namelist input, recognised by its `.in` or `.qe` extension.
class QEInputFormat : public InputFormat {
  public:
    /// Ranked above the YAML fallback, below formats recognised by name or contents.
    static constexpr int extension_detection_priority = 100;

    /// "qe", and "espresso".
    [[nodiscard]] std::vector<std::string> names() const override {
        return {"qe", "espresso"};
    }

    /// How a log line names a pw.x input.
    [[nodiscard]] std::string display_name() const override {
        return "a Quantum ESPRESSO input";
    }

    /// extension_detection_priority: above the fallback, below name-based formats.
    [[nodiscard]] int detection_priority() const override {
        return extension_detection_priority;
    }

    /// A file, not a directory, whose extension is `.in` or `.qe`.
    [[nodiscard]] bool recognizes(const std::filesystem::path& path) const override {
        if (std::filesystem::is_directory(path))
            return false;
        const std::string extension = path.extension().string();
        return extension == ".in" || extension == ".qe";
    }

    /// A QEInputParser.
    [[nodiscard]] std::unique_ptr<InputParser> create_parser() const override {
        return std::make_unique<QEInputParser>();
    }
};

} // namespace q3
