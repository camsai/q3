#pragma once

/**
 * LocalizationPolicy — how the gauge inside the chosen subspace is fixed; the
 * manifest's `localization: none | auto | maximal`, in that order.
 *
 * The spread is defined through the neighbour overlaps M_mn(k,b), and a
 * record imported from another code's projections and eigenvalues alone
 * carries none. `Preferred` localises when the record can support it and
 * keeps the projection gauge when it cannot; `Required` makes that an error.
 * The manifest's default is `auto`, a preference, because defaulting to a
 * demand would make every such record an error for asking for the default.
 */

#include <stdexcept>
#include <string>

namespace q3 {

enum class LocalizationPolicy { None, Preferred, Required };

/// The policy a manifest's `localization` word names; any other word is refused.
[[nodiscard]] inline LocalizationPolicy localization_policy_from(const std::string& word) {
    if (word == "none") {
        return LocalizationPolicy::None;
    }
    if (word == "auto") {
        return LocalizationPolicy::Preferred;
    }
    if (word == "maximal") {
        return LocalizationPolicy::Required;
    }
    throw std::invalid_argument("localization_policy_from: unknown localization '" + word +
                                "'. Supported: none, auto, maximal");
}

} // namespace q3
