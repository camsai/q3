#pragma once

/**
 * PseudopotentialFactory — read a dataset in whichever registered format it is.
 *
 * This is what lets an ordinary Q3 input name any dataset file the build can
 * read: the format is chosen from the file itself, never from its name. Every
 * registered format (pseudopotential_format_registry.hpp) is shown the first
 * line, and of those that recognise it, the one with the highest
 * detection_priority() reads the file, ties going to the format registered
 * first. Sniffing content
 * rather than trusting a name matters because dataset files are routinely
 * renamed per species, and a file handed to the wrong reader fails far from
 * the cause, or worse, parses.
 *
 * The format readers keep their own narrower entry points (UPFReader::read for
 * UPF), so a caller that means one format still gets that format or an error,
 * never another format by surprise.
 */

#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>

#include "io/pseudopotential/pseudopotential_format_registry.hpp"
#include "layer_3_representation/pseudopotential/init.hpp"

namespace q3 {

/// Reads a dataset in whichever registered format its first line names.
class PseudopotentialFactory {
  public:
    /**
     * Read the dataset at `dataset_index` from `filename`, in the
     * highest-ranked registered format that recognises the file's first line.
     */
    static std::unique_ptr<Pseudopotential> read(const std::string& filename,
                                                 int dataset_index = 0) {
        std::ifstream file_stream(filename);
        if (!file_stream.is_open())
            throw std::runtime_error("Cannot open PP file: " + filename);
        std::string first_line;
        std::getline(file_stream, first_line);
        file_stream.close();

        const auto& registry = pseudopotential_format_registry();
        const PseudopotentialFormat* chosen = nullptr;
        for (const auto& format : registry.entries()) {
            if (!format->recognizes(first_line))
                continue;
            if (chosen == nullptr || format->detection_priority() > chosen->detection_priority())
                chosen = format.get();
        }
        if (chosen != nullptr)
            return chosen->read(filename, dataset_index);
        throw std::runtime_error(
            "No registered pseudopotential format recognises " + filename +
            "; the formats this build reads are: " + registry.canonical_names(", "));
    }
};

} // namespace q3
