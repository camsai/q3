#pragma once

/**
 * UPFPseudopotentialFormat — the Unified Pseudopotential Format, v1 and v2.
 *
 * Every UPF file opens with an XML tag (`<UPF version=...>` for v2,
 * `<PP_INFO>` for v1), which is how it is recognised. A UPF file holds one
 * dataset, so the dataset index is ignored.
 */

#include <memory>
#include <string>
#include <vector>

#include "io/pseudopotential/pseudopotential_format.hpp"
#include "io/pseudopotential/upf_reader.hpp"

namespace q3 {

/// UPF v1 and v2, recognised by the XML tag every UPF file opens with.
class UPFPseudopotentialFormat : public PseudopotentialFormat {
  public:
    /// Ranked as a definite signature: every UPF file opens with an XML tag.
    static constexpr int signature_detection_priority = 100;

    /// "upf".
    [[nodiscard]] std::vector<std::string> names() const override {
        return {"upf"};
    }

    /// A first line holding an XML tag.
    [[nodiscard]] bool recognizes(const std::string& first_line) const override {
        return first_line.find('<') != std::string::npos;
    }

    /// signature_detection_priority.
    [[nodiscard]] int detection_priority() const override {
        return signature_detection_priority;
    }

    /// The file's one dataset, read by UPFReader.
    [[nodiscard]] std::unique_ptr<Pseudopotential> read(const std::string& filename,
                                                        int /*dataset_index*/) const override {
        return UPFReader::read(filename);
    }
};

} // namespace q3
