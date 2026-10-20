#pragma once

/**
 * UPFReader — read a dataset from a UPF v1 or v2 file.
 *
 * Detects the format version and delegates to the appropriate readers:
 *
 *   v1: UPFv1Reader → (if US) UltrasoftUPFv1Reader
 *   v2: UPFv2Reader → (if US or PAW) UltrasoftUPFv2Reader → (if PAW) PawUPFv2Reader
 *
 * The readers are organized by version (v1/, v2/) and pseudopotential type
 * (core, ultrasoft, paw), each in its own header file.
 *
 * A PAW dataset reads through the ultrasoft chain first because it carries the
 * ultrasoft augmentation apparatus verbatim; PawUPFv2Reader then adds the
 * one-center data.
 *
 * UPF only. A file of unknown format goes through PseudopotentialFactory::read()
 * (factory.hpp), which picks the reader from the file's contents.
 */

#include <fstream>
#include <iterator>
#include <memory>
#include <string>
#include <vector>

#include "io/pseudopotential/upf_reader/utilities.hpp"
#include "io/pseudopotential/upf_reader/v1/init.hpp"
#include "io/pseudopotential/upf_reader/v1/ultrasoft.hpp"
#include "io/pseudopotential/upf_reader/v2/init.hpp"
#include "io/pseudopotential/upf_reader/v2/paw.hpp"
#include "io/pseudopotential/upf_reader/v2/ultrasoft.hpp"
#include "layer_3_representation/pseudopotential/init.hpp"
#include "layer_3_representation/pseudopotential/paw/init.hpp"
#include "layer_3_representation/pseudopotential/ultrasoft/init.hpp"

namespace q3 {

/// Reads one dataset from a UPF v1 or v2 file.
class UPFReader {
  public:
    static std::unique_ptr<Pseudopotential> read(const std::string& filename) {
        std::ifstream file_stream(filename);
        if (!file_stream.is_open())
            throw std::runtime_error("Cannot open PP file: " + filename);

        std::string first_line;
        std::getline(file_stream, first_line);

        /**
         * Determine whether this is an ultrasoft PP by reading the header type first.
         * We need to peek at the pseudopotential type to decide which subclass to create.
         */
        bool is_ultrasoft_type = false;
        bool is_paw_type = false;
        std::string pseudopotential_type_string;

        if (first_line.find("version=\"2") != std::string::npos) {
            // UPF v2: read file content, check header for pseudo_type
            file_stream.seekg(0);
            std::string content((std::istreambuf_iterator<char>(file_stream)),
                                std::istreambuf_iterator<char>());
            file_stream.close();

            auto header_position = content.find("<PP_HEADER");
            if (header_position != std::string::npos) {
                auto end_position = content.find("/>", header_position);
                if (end_position == std::string::npos)
                    end_position = content.find(">", header_position);
                std::string header_tag =
                    content.substr(header_position, end_position - header_position + 2);
                pseudopotential_type_string = upf_utilities::canonical_pseudopotential_type(
                    upf_utilities::xml_attribute(header_tag, "pseudo_type"));
            }

            is_paw_type = (pseudopotential_type_string == "PAW");
            is_ultrasoft_type = (pseudopotential_type_string == "US");

            // Create the appropriate subclass
            std::unique_ptr<Pseudopotential> pseudopotential;
            if (is_paw_type) {
                pseudopotential = std::make_unique<PawPseudopotential>();
            } else if (is_ultrasoft_type) {
                pseudopotential = std::make_unique<UltrasoftPseudopotential>();
            } else {
                pseudopotential = std::make_unique<Pseudopotential>();
            }

            UPFv2Reader(*pseudopotential, content, filename).read();
            if (is_ultrasoft_type || is_paw_type) {
                UltrasoftUPFv2Reader(static_cast<UltrasoftPseudopotential&>(*pseudopotential),
                                     content)
                    .read();
            }
            if (is_paw_type) {
                PawUPFv2Reader(
                    static_cast<PawPseudopotential&>(*pseudopotential), content, filename)
                    .read();
            }

            upf_utilities::check_atomic_wavefunction_norm(*pseudopotential);
            return pseudopotential;
        } else {
            // UPF v1: sequential line-by-line parsing
            file_stream.close();
            std::ifstream v1_stream(filename);

            /**
             * For v1, we need to read the header first to determine the type.
             * Read a temporary PP to get the type, then re-read with the correct class.
             * Since v1 readers mutate the passed reference, we read into a temporary first.
             */
            auto temporary_pseudopotential = std::make_unique<Pseudopotential>();
            UPFv1Reader(*temporary_pseudopotential, v1_stream).read();

            is_ultrasoft_type = (temporary_pseudopotential->pseudopotential_type == "US");

            std::unique_ptr<Pseudopotential> pseudopotential;
            if (is_ultrasoft_type) {
                // Create UltrasoftPseudopotential and copy base fields
                auto ultrasoft_pseudopotential = std::make_unique<UltrasoftPseudopotential>();
                // Copy all base class fields
                static_cast<Pseudopotential&>(*ultrasoft_pseudopotential) =
                    std::move(*temporary_pseudopotential);
                pseudopotential = std::move(ultrasoft_pseudopotential);
            } else {
                pseudopotential = std::move(temporary_pseudopotential);
            }

            if (is_ultrasoft_type) {
                std::ifstream us_stream(filename);
                Pseudopotential dummy_for_seek;
                UPFv1Reader(dummy_for_seek, us_stream).read();
                UltrasoftUPFv1Reader(static_cast<UltrasoftPseudopotential&>(*pseudopotential),
                                     us_stream)
                    .read();
            }

            upf_utilities::check_atomic_wavefunction_norm(*pseudopotential);
            return pseudopotential;
        }
    }
};

} // namespace q3
