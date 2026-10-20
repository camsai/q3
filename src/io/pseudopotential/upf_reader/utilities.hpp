#pragma once

/**
 * Shared UPF parsing utilities.
 *
 * Low-level helpers used by both UPF v1 and v2 readers for string manipulation,
 * numeric extraction, XML attribute parsing, and post-read validation.
 */

#include <cctype>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "io/logger.hpp"
#include "layer_0_core/math/integration/simpson.hpp"
#include "layer_3_representation/pseudopotential/init.hpp"

namespace q3 {
namespace upf_utilities {

/// Remove leading and trailing whitespace from a string.
inline std::string trim(const std::string& input) {
    size_t first_non_whitespace = input.find_first_not_of(" \t\r\n");
    if (first_non_whitespace == std::string::npos)
        return "";
    size_t last_non_whitespace = input.find_last_not_of(" \t\r\n");
    return input.substr(first_non_whitespace, last_non_whitespace - first_non_whitespace + 1);
}

/**
 * Canonical kind name for a UPF `pseudo_type` header value.
 *
 * `Pseudopotential::pseudopotential_type` is documented to hold "NC", "US" or
 * "PAW", and the rest of the tree relies on that: the reader factory picks the
 * subclass to construct by comparing against those spellings, and
 * `pseudopotential_kind_name` renders them into the provenance record. The v2
 * reader, though, stored the header attribute verbatim, so whether a dataset
 * got its augmentation apparatus depended on how its generator happened to
 * spell the field.
 *
 * The spellings are not uniform in practice. The UPF header documents "NC",
 * "SL", "US" and "PAW", but generators also emit "USPP" for an ultrasoft
 * dataset — a UPF v2 carbon USPP written by the `atomic` code reports
 * `pseudo_type="USPP"`, and against the exact-match test that read as "not
 * ultrasoft". The file was then built as a plain norm-conserving
 * `Pseudopotential`, silently dropping every Q_ij it carried, and the first
 * code to ask for its augmentation data aborted on the `as_ultrasoft` cast.
 * Matching on a canonical form rather than one blessed spelling is what keeps
 * that from being a per-generator lottery.
 *
 * An unrecognized value is returned trimmed but otherwise untouched, so it
 * still lands on the norm-conserving construction path exactly as before and
 * the original label survives for diagnostics.
 */
inline std::string canonical_pseudopotential_type(const std::string& raw_type) {
    std::string token = trim(raw_type);
    std::string upper_token = token;
    for (char& character : upper_token)
        character = static_cast<char>(std::toupper(static_cast<unsigned char>(character)));

    if (upper_token == "PAW")
        return "PAW";
    if (upper_token == "US" || upper_token == "USPP")
        return "US";
    // "SL" is the semilocal member of the norm-conserving family.
    if (upper_token == "NC" || upper_token == "SL")
        return "NC";
    return token;
}

/// Read a specified number of real values from a stream.
inline std::vector<Real> read_reals(std::ifstream& file_stream, int count) {
    std::vector<Real> values;
    values.reserve(count);
    Real value;
    while (static_cast<int>(values.size()) < count && file_stream >> value) {
        values.push_back(value);
    }
    file_stream.clear(); // Clear failbit so getline works after hitting non-numeric text
    return values;
}

/// Advance the stream until a line containing the given tag is found.
inline bool find_tag(std::ifstream& file_stream, const std::string& tag) {
    std::string line;
    while (std::getline(file_stream, line)) {
        if (line.find(tag) != std::string::npos)
            return true;
    }
    return false;
}

/// Extract an XML attribute value from a block of text (e.g. attr="value" or attr='value').
inline std::string xml_attribute(const std::string& text, const std::string& attribute_name) {
    std::string patterns[] = {attribute_name + "=\"", attribute_name + "='"};
    for (auto& pattern : patterns) {
        auto position = text.find(pattern);
        if (position != std::string::npos) {
            position += pattern.size();
            char delimiter = pattern.back();
            auto end_position = text.find(delimiter, position);
            if (end_position != std::string::npos) {
                return text.substr(position, end_position - position);
            }
        }
    }
    return "";
}

/**
 * Build the error a malformed required attribute produces.
 *
 * Named separately so the "missing" and "unparseable" paths report the same
 * way: both are the same problem from a user's point of view — this file
 * cannot be read, and here is the field to look at.
 */
inline std::runtime_error malformed_attribute_error(const std::string& attribute_name,
                                                    const std::string& source_description,
                                                    const std::string& expected_kind,
                                                    const std::string& found_value) {
    std::string message = "Malformed UPF file " + source_description +
                          ": required PP_HEADER "
                          "attribute '" +
                          attribute_name + "' ";
    if (found_value.empty()) {
        message += "is missing";
    } else {
        message += "is not " + expected_kind + " (found \"" + found_value + "\")";
    }
    return std::runtime_error(message);
}

/**
 * Read a required numeric attribute from a UPF header tag.
 *
 * xml_attribute() returns an empty string for an attribute that is not present,
 * and handing that to the standard conversions yields "stoi: no conversion" —
 * an error naming neither the file nor the field, which is no help to someone
 * holding a pseudopotential that will not load. These two helpers name both.
 *
 * Trailing characters are rejected rather than ignored: std::stoi("20x0")
 * happily returns 20, and a mesh size silently truncated from a corrupt
 * attribute is worse than a refusal.
 *
 * Use only for attributes a file genuinely cannot omit. Optional attributes
 * (number_of_wfc, core_correction) stay tolerant of absence at their call sites.
 */
inline int required_integer_attribute(const std::string& text,
                                      const std::string& attribute_name,
                                      const std::string& source_description) {
    std::string value = trim(xml_attribute(text, attribute_name));
    if (value.empty())
        throw malformed_attribute_error(attribute_name, source_description, "an integer", "");

    std::size_t characters_consumed = 0;
    int parsed_value = 0;
    try {
        parsed_value = std::stoi(value, &characters_consumed);
    } catch (const std::exception&) {
        throw malformed_attribute_error(attribute_name, source_description, "an integer", value);
    }
    if (!trim(value.substr(characters_consumed)).empty())
        throw malformed_attribute_error(attribute_name, source_description, "an integer", value);
    return parsed_value;
}

/// Read a required real-valued attribute from a UPF header tag.
inline Real required_real_attribute(const std::string& text,
                                    const std::string& attribute_name,
                                    const std::string& source_description) {
    std::string value = trim(xml_attribute(text, attribute_name));
    if (value.empty())
        throw malformed_attribute_error(attribute_name, source_description, "a number", "");

    std::size_t characters_consumed = 0;
    Real parsed_value = 0.0;
    try {
        parsed_value = std::stod(value, &characters_consumed);
    } catch (const std::exception&) {
        throw malformed_attribute_error(attribute_name, source_description, "a number", value);
    }
    if (!trim(value.substr(characters_consumed)).empty())
        throw malformed_attribute_error(attribute_name, source_description, "a number", value);
    return parsed_value;
}

/// Extract numeric data from an XML section body, skipping lines that contain XML tags.
inline std::vector<Real> xml_data(const std::string& section) {
    std::vector<Real> values;
    std::istringstream stream(section);
    std::string line;
    while (std::getline(stream, line)) {
        // Skip XML tags (lines containing < or >)
        if (line.find('<') != std::string::npos)
            continue;
        std::istringstream line_stream(line);
        Real value;
        while (line_stream >> value)
            values.push_back(value);
    }
    return values;
}

/// Validate atomic wavefunction norms and mark zero-norm wavefunctions.
inline void check_atomic_wavefunction_norm(Pseudopotential& pseudopotential) {
    constexpr Real zero_norm_threshold = 1.0e-10;
    for (int wavefunction_index = 0;
         wavefunction_index < static_cast<int>(pseudopotential.wavefunctions_atomic.size());
         wavefunction_index++) {
        auto& wavefunction = pseudopotential.wavefunctions_atomic[wavefunction_index];
        // Calculate norm of the smooth part: ∫ chi(r)^2 dr
        std::vector<Real> integrand(pseudopotential.mesh_size);
        for (int radial_index = 0; radial_index < pseudopotential.mesh_size; radial_index++) {
            integrand[radial_index] =
                wavefunction.radial_values[radial_index] * wavefunction.radial_values[radial_index];
        }
        Real norm = simpson_rule(
            integrand, pseudopotential.radial_grid_derivative, pseudopotential.mesh_size);
        if (norm < zero_norm_threshold) {
            log::warn("     WARNING: atomic wfc # {} for atom type {} has zero norm",
                      wavefunction_index + 1,
                      pseudopotential.element);
            wavefunction.occupations = -1.0e-10; // set to a small negative value to skip it
        }
    }
}

} // namespace upf_utilities
} // namespace q3
