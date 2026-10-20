#pragma once

/**
 * DiagnosticError — a failure that names itself to a driver: the prose a
 * person reads, and beside it the lines a program reads.
 *
 * An entry point prints `Error: <what()>` for any exception. For one of these
 * it then prints each of diagnostic_lines() on a line of its own, `key:
 * value`, the first of them `diagnostic: <code>` -- so a driver routes on the
 * code while a reader still gets the sentence, and the entry point needs to
 * know nothing about what failed or which lines that failure has to add.
 *
 * A std::runtime_error, so every catch of the base class still sees the prose.
 */

#include <stdexcept>
#include <string>
#include <vector>

namespace q3 {

class DiagnosticError : public std::runtime_error {
  public:
    using std::runtime_error::runtime_error;

    /// The lines an entry point prints after the prose, in order; never empty.
    [[nodiscard]] virtual std::vector<std::string> diagnostic_lines() const = 0;
};

} // namespace q3
