#pragma once

/**
 * PseudopotentialFormat — one file format pseudopotential datasets are read from.
 *
 * Formats are registered with the pseudopotential-format registry
 * (pseudopotential_format_registry.hpp). Core registers UPF, and a build-time
 * add-on may register more. PseudopotentialFactory::read() shows each
 * registered format the file's first line and hands the file to the
 * recognising format with the highest detection_priority(), so a dataset is
 * read by what it contains rather than by what it is called.
 */

#include <memory>
#include <string>
#include <vector>

#include "layer_3_representation/pseudopotential/init.hpp"

namespace q3 {

/// One file format pseudopotential datasets are read from; see the file comment.
class PseudopotentialFormat {
  public:
    virtual ~PseudopotentialFormat() = default;

    /// The names this format answers to; the first is canonical and appears in error text.
    [[nodiscard]] virtual std::vector<std::string> names() const = 0;

    /// Whether a file whose first line is `first_line` holds a dataset of this format.
    [[nodiscard]] virtual bool recognizes(const std::string& first_line) const = 0;

    /**
     * Rank among the formats that recognise a file, highest first. A format
     * recognised by a definite signature ranks above one that claims whatever
     * the others do not, so the broad claim never hides a specific one.
     */
    [[nodiscard]] virtual int detection_priority() const = 0;

    /**
     * Read the dataset at `dataset_index` from `filename`.
     *
     * The index counts datasets within one file, for a format that can hold
     * several; a format holding one dataset per file ignores it.
     */
    [[nodiscard]] virtual std::unique_ptr<Pseudopotential> read(const std::string& filename,
                                                                int dataset_index) const = 0;
};

} // namespace q3
