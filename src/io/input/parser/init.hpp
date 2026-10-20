#pragma once

/**
 * InputParser — abstract interface for input file parsers.
 *
 * Defines the contract for parsing different input formats (YAML, QE)
 * into a common InputData struct. Concrete implementations:
 *   yaml.hpp  — YAMLInputParser
 *   qe.hpp   — QEInputParser
 *
 * Use create_input_parser() from factory.hpp to obtain the right parser.
 */

#include <string>

#include "io/input/data.hpp"

namespace q3 {

class InputParser {
  public:
    virtual ~InputParser() = default;

    /**
     * Parse the given input file and return the populated InputData.
     */
    virtual InputData parse(const std::string& filename) const = 0;
};

} // namespace q3
