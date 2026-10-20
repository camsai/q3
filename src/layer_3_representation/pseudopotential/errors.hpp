#pragma once

/**
 * Pseudopotential-related error types.
 */

#include <stdexcept>

namespace q3 {

/**
 * Thrown when an input requests a pseudopotential type that is recognized
 * but not implemented (e.g. PAW). Distinct from std::runtime_error so that
 * callers and tests can tell "unsupported type" apart from a malformed file.
 */
class UnsupportedPseudopotentialTypeError : public std::runtime_error {
  public:
    using std::runtime_error::runtime_error;
};

} // namespace q3
