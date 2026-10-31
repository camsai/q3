#pragma once

/**
 * InputFormat — one input format a run can be read from.
 *
 * Formats are registered by name with the input-format registry
 * (input_format_registry.hpp). Core registers Q3's YAML and Quantum ESPRESSO
 * namelist inputs, and a build-time add-on may register more. An entry point
 * asks the format everything it needs to know, so no entry point and no
 * factory has to name one:
 *
 *  - the names `--format` accepts, and how a log line names an input;
 *  - whether a path is an input of this format, consulted in descending
 *    detection_priority() when no `--format` is given;
 *  - whether a directory holds one that a run given no input path should read;
 *  - the parser that reads it;
 *  - any files the format's own conventions expect beside the results once
 *    the run has finished.
 */

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "io/input/data.hpp"
#include "io/input/parser/init.hpp"

namespace q3 {

namespace generated {
struct Results;
} // namespace generated

/// One input format a run can be read from; see the file comment.
class InputFormat {
  public:
    virtual ~InputFormat() = default;

    /**
     * The names `--format` accepts. The first is canonical: parse_input()
     * records it as InputData::source_format, and usage text prints it.
     */
    [[nodiscard]] virtual std::vector<std::string> names() const = 0;

    /// How a log line names an input of this format, e.g. "a Quantum ESPRESSO input".
    [[nodiscard]] virtual std::string display_name() const = 0;

    /**
     * Rank among the formats that recognise a path, highest first. A format
     * that recognises paths by file name or directory contents ranks above
     * one that goes by extension, and the fallback ranks lowest, so a path is
     * claimed by the most specific rule that matches it.
     */
    [[nodiscard]] virtual int detection_priority() const = 0;

    /// Whether `path` names an input of this format, judged without parsing it.
    [[nodiscard]] virtual bool recognizes(const std::filesystem::path& path) const = 0;

    /**
     * Whether `directory` holds an input of this format that a run given no
     * input path should read. False for a format whose inputs are always
     * named explicitly.
     */
    [[nodiscard]] virtual bool
    claims_working_directory(const std::filesystem::path& /*directory*/) const {
        return false;
    }

    /// A parser for inputs of this format.
    [[nodiscard]] virtual std::unique_ptr<InputParser> create_parser() const = 0;

    /**
     * Write the files this format's conventions expect beside the results of
     * a finished run that was read from it. Writes nothing for a format with
     * no such convention.
     */
    virtual void write_companion_outputs(const InputData& /*input*/,
                                         const generated::Results& /*results*/) const {}
};

} // namespace q3
