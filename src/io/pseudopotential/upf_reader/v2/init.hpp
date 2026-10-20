#pragma once

/**
 * UPF v2 core reader.
 *
 * Reads the common (norm-conserving) sections of a UPF v2 format file:
 * PP_HEADER (XML attributes), PP_MESH/PP_R/PP_RAB, PP_NLCC, PP_LOCAL,
 * PP_NONLOCAL (PP_BETA.N projectors + PP_DIJ), PP_PSWFC (PP_CHI.N), PP_RHOATOM.
 *
 * UPF v2 uses XML-like formatting where section data is embedded between
 * opening and closing tags, and header fields are XML attributes.
 * The entire file content is read into memory for string-based parsing.
 *
 * After this reader runs, the Pseudopotential struct has all fields populated
 * that are common to both norm-conserving and ultrasoft pseudopotential_set.
 * Ultrasoft-specific sections (PP_AUGMENTATION) are handled by UltrasoftUPFv2Reader.
 */

#include <sstream>
#include <stdexcept>
#include <string>

#include "io/logger.hpp"
#include "io/pseudopotential/upf_reader/utilities.hpp"
#include "layer_3_representation/pseudopotential/init.hpp"

namespace q3 {

class UPFv2Reader {
  public:
    /**
     * @param source_description  File path this content came from, used to name
     *                            the file in errors about malformed fields.
     */
    UPFv2Reader(Pseudopotential& pseudopotential,
                const std::string& file_content,
                const std::string& source_description)
        : pseudopotential_(pseudopotential)
        , content_(file_content)
        , source_description_(source_description) {}

    void read() {
        read_header();
        read_mesh();
        read_core_charge_density();
        read_local_potential();
        read_beta_projectors();
        read_d_matrix();
        read_pseudo_wavefunctions();
        read_atomic_charge_density();
    }

  private:
    void read_header() {
        using namespace upf_utilities;
        auto position = content_.find("<PP_HEADER");
        if (position == std::string::npos)
            throw std::runtime_error("No PP_HEADER found in UPF v2 file");
        auto end_position = content_.find("/>", position);
        if (end_position == std::string::npos)
            end_position = content_.find(">", position);
        std::string header_tag = content_.substr(position, end_position - position + 2);

        pseudopotential_.element = xml_attribute(header_tag, "element");
        // Trim whitespace from element
        while (!pseudopotential_.element.empty() && pseudopotential_.element.back() == ' ')
            pseudopotential_.element.pop_back();
        while (!pseudopotential_.element.empty() && pseudopotential_.element.front() == ' ')
            pseudopotential_.element.erase(pseudopotential_.element.begin());

        pseudopotential_.pseudopotential_type =
            canonical_pseudopotential_type(xml_attribute(header_tag, "pseudo_type"));
        pseudopotential_.valence_charge =
            required_real_attribute(header_tag, "z_valence", source_description_);
        pseudopotential_.mesh_size =
            required_integer_attribute(header_tag, "mesh_size", source_description_);
        pseudopotential_.number_of_projectors =
            required_integer_attribute(header_tag, "number_of_proj", source_description_);
        pseudopotential_.maximum_angular_momentum =
            required_integer_attribute(header_tag, "l_max", source_description_);

        std::string core_correction = xml_attribute(header_tag, "core_correction");
        pseudopotential_.has_nonlinear_core_correction =
            (core_correction == "T" || core_correction == "t" || core_correction == "true" ||
             core_correction == "True");

        pseudopotential_.xc_functional = xml_attribute(header_tag, "functional");

        std::string number_of_wavefunctions_string = xml_attribute(header_tag, "number_of_wfc");
        if (!number_of_wavefunctions_string.empty())
            pseudopotential_.number_of_wavefunctions = std::stoi(number_of_wavefunctions_string);

        log::info("     UPFv2: {} ({}) Z={} mesh={} num_beta={} max_l={} has_nlcc={}",
                  pseudopotential_.element,
                  pseudopotential_.pseudopotential_type,
                  pseudopotential_.valence_charge,
                  pseudopotential_.mesh_size,
                  pseudopotential_.number_of_projectors,
                  pseudopotential_.maximum_angular_momentum,
                  pseudopotential_.has_nonlinear_core_correction);
    }

    void read_mesh() {
        using namespace upf_utilities;
        // Read mesh attribute from PP_MESH tag if present
        auto mesh_position = content_.find("<PP_MESH");
        if (mesh_position != std::string::npos) {
            auto tag_end = content_.find(">", mesh_position);
            std::string tag_line = content_.substr(mesh_position, tag_end - mesh_position);
            std::string mesh_attribute = xml_attribute(tag_line, "mesh");
            if (!mesh_attribute.empty()) {
                pseudopotential_.mesh_size = std::stoi(mesh_attribute);
            }
        }

        // Read r data from PP_R tag
        auto position = content_.find("<PP_R ");
        if (position == std::string::npos)
            position = content_.find("<PP_R>");
        if (position != std::string::npos) {
            auto end_position = content_.find("</PP_R>", position);
            std::string section = content_.substr(position, end_position - position);
            pseudopotential_.radial_grid = xml_data(section);
        }

        // Read rab data from PP_RAB tag
        position = content_.find("<PP_RAB ");
        if (position == std::string::npos)
            position = content_.find("<PP_RAB>");
        if (position != std::string::npos) {
            auto end_position = content_.find("</PP_RAB>", position);
            std::string section = content_.substr(position, end_position - position);
            pseudopotential_.radial_grid_derivative = xml_data(section);
        }
    }

    void read_core_charge_density() {
        using namespace upf_utilities;
        if (!pseudopotential_.has_nonlinear_core_correction)
            return;
        auto position = content_.find("<PP_NLCC");
        if (position != std::string::npos) {
            auto end_position = content_.find("</PP_NLCC>", position);
            std::string section = content_.substr(position, end_position - position);
            pseudopotential_.charge_core.radial_values = xml_data(section);
        }
    }

    void read_local_potential() {
        using namespace upf_utilities;
        auto position = content_.find("<PP_LOCAL");
        if (position != std::string::npos) {
            auto end_position = content_.find("</PP_LOCAL>", position);
            std::string section = content_.substr(position, end_position - position);
            pseudopotential_.potential_local.radial_values = xml_data(section);
        }
    }

    void read_beta_projectors() {
        using namespace upf_utilities;
        pseudopotential_.projectors.resize(pseudopotential_.number_of_projectors);
        for (int beta_index = 0; beta_index < pseudopotential_.number_of_projectors; beta_index++) {
            std::string tag = "<PP_BETA." + std::to_string(beta_index + 1);
            auto position = content_.find(tag);
            if (position == std::string::npos) {
                throw std::runtime_error("Cannot find " + tag + " in UPF v2 file");
            }

            // Read attributes from the opening tag
            auto tag_end = content_.find(">", position);
            std::string tag_line = content_.substr(position, tag_end - position);

            std::string angular_momentum_string = xml_attribute(tag_line, "angular_momentum");
            pseudopotential_.projectors[beta_index].angular_momentum =
                angular_momentum_string.empty() ? 0 : std::stoi(angular_momentum_string);

            std::string cutoff_index_string = xml_attribute(tag_line, "cutoff_radius_index");
            pseudopotential_.projectors[beta_index].number_of_radial_cutoff_points =
                cutoff_index_string.empty() ? pseudopotential_.mesh_size
                                            : std::stoi(cutoff_index_string);

            // Read beta data
            auto close_position =
                content_.find("</PP_BETA." + std::to_string(beta_index + 1) + ">", position);
            if (close_position == std::string::npos) {
                // Try generic </PP_BETA>
                close_position = content_.find("</PP_BETA>", position);
            }
            std::string section = content_.substr(tag_end + 1, close_position - tag_end - 1);
            pseudopotential_.projectors[beta_index].radial_values = xml_data(section);

            // Ensure beta has mesh_size entries (pad with zeros if needed)
            pseudopotential_.projectors[beta_index].radial_values.resize(pseudopotential_.mesh_size,
                                                                         0.0);
        }
    }

    void read_d_matrix() {
        using namespace upf_utilities;
        auto position = content_.find("<PP_DIJ");
        if (position != std::string::npos) {
            auto end_position = content_.find("</PP_DIJ>", position);
            std::string section = content_.substr(position, end_position - position);
            std::vector<Real> dij_flat = xml_data(section);

            pseudopotential_.d_ion_matrix.resize(
                pseudopotential_.number_of_projectors,
                std::vector<Real>(pseudopotential_.number_of_projectors, 0.0));
            int flat_index = 0;
            for (int row = 0; row < pseudopotential_.number_of_projectors &&
                              flat_index < static_cast<int>(dij_flat.size());
                 row++) {
                for (int column = 0; column < pseudopotential_.number_of_projectors &&
                                     flat_index < static_cast<int>(dij_flat.size());
                     column++) {
                    pseudopotential_.d_ion_matrix[row][column] = dij_flat[flat_index++];
                }
            }
        }
    }

    void read_pseudo_wavefunctions() {
        using namespace upf_utilities;
        auto position = content_.find("<PP_PSWFC>");
        if (position == std::string::npos)
            position = content_.find("<PP_PSWFC ");
        if (position == std::string::npos)
            return;

        auto end_pswfc = content_.find("</PP_PSWFC>", position);
        if (end_pswfc == std::string::npos)
            return;

        pseudopotential_.wavefunctions_atomic.resize(pseudopotential_.number_of_wavefunctions);
        for (int wavefunction_index = 0;
             wavefunction_index < pseudopotential_.number_of_wavefunctions;
             wavefunction_index++) {
            std::string tag = "<PP_CHI." + std::to_string(wavefunction_index + 1);
            auto wavefunction_position = content_.find(tag, position);
            if (wavefunction_position == std::string::npos || wavefunction_position >= end_pswfc)
                continue;

            auto wavefunction_tag_end = content_.find(">", wavefunction_position);
            std::string tag_line = content_.substr(wavefunction_position,
                                                   wavefunction_tag_end - wavefunction_position);

            pseudopotential_.wavefunctions_atomic[wavefunction_index].label =
                xml_attribute(tag_line, "label");
            std::string angular_momentum_string = xml_attribute(tag_line, " l");
            if (angular_momentum_string.empty())
                angular_momentum_string = xml_attribute(tag_line, "\nl");
            if (angular_momentum_string.empty())
                angular_momentum_string = xml_attribute(tag_line, "\tl");
            pseudopotential_.wavefunctions_atomic[wavefunction_index].angular_momentum =
                angular_momentum_string.empty() ? 0 : std::stoi(angular_momentum_string);
            std::string occupation_string = xml_attribute(tag_line, "occupation");
            pseudopotential_.wavefunctions_atomic[wavefunction_index].occupations =
                occupation_string.empty() ? 0.0 : std::stod(occupation_string);

            std::string close_tag = "</PP_CHI." + std::to_string(wavefunction_index + 1) + ">";
            auto wavefunction_end = content_.find(close_tag, wavefunction_position);
            if (wavefunction_end == std::string::npos)
                wavefunction_end = content_.find("</PP_CHI>", wavefunction_position);

            if (wavefunction_end != std::string::npos) {
                std::string section = content_.substr(wavefunction_tag_end + 1,
                                                      wavefunction_end - wavefunction_tag_end - 1);
                pseudopotential_.wavefunctions_atomic[wavefunction_index].radial_values =
                    xml_data(section);
                pseudopotential_.wavefunctions_atomic[wavefunction_index].radial_values.resize(
                    pseudopotential_.mesh_size, 0.0);
            }
        }
    }

    void read_atomic_charge_density() {
        using namespace upf_utilities;
        auto position = content_.find("<PP_RHOATOM");
        if (position != std::string::npos) {
            auto end_position = content_.find("</PP_RHOATOM>", position);
            std::string section = content_.substr(position, end_position - position);
            pseudopotential_.charge_atomic.radial_values = xml_data(section);
        }
    }

    Pseudopotential& pseudopotential_;
    const std::string& content_;
    std::string source_description_;
};

} // namespace q3
