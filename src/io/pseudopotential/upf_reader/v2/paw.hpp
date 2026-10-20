#pragma once

/**
 * UPF v2 PAW reader.
 *
 * Reads the PAW-specific sections of a UPF v2 file:
 *
 *   PP_FULL_WFC   → PP_AEWFC.N (all-electron partial waves) and
 *                   PP_PSWFC.N (pseudo partial waves)
 *   PP_PAW        → PP_OCCUPATIONS, PP_AE_NLCC (all-electron core density),
 *                   PP_AE_VLOC (all-electron local potential)
 *   PP_MULTIPOLES → multipole moments of the augmentation channels
 *   PP_AUGMENTATION attributes → shape, cutoff radius and index, l_max
 *
 * Must run after UPFv2Reader and UltrasoftUPFv2Reader have populated the base
 * and augmentation fields: a PAW dataset carries the ultrasoft augmentation
 * apparatus verbatim, so PP_Q and PP_QIJL are read by the ultrasoft reader.
 *
 * Note the tag collision this reader has to navigate: PP_PSWFC.N inside
 * PP_FULL_WFC are the pseudo partial waves, while the top-level PP_PSWFC block
 * holds PP_CHI.N atomic wavefunctions. Searches here are bounded to the
 * PP_FULL_WFC block so the two never mix.
 */

#include <string>

#include "io/logger.hpp"
#include "io/pseudopotential/upf_reader/utilities.hpp"
#include "layer_3_representation/pseudopotential/paw/init.hpp"

namespace q3 {

class PawUPFv2Reader {
  public:
    /**
     * @param source_description  File path this content came from, used to name
     *                            the file in errors about missing PAW sections.
     */
    PawUPFv2Reader(PawPseudopotential& pseudopotential,
                   const std::string& file_content,
                   const std::string& source_description)
        : pseudopotential_(pseudopotential)
        , content_(file_content)
        , source_description_(source_description) {}

    void read() {
        read_augmentation_metadata();
        read_partial_waves();
        read_paw_block();
        read_multipole_moments();
        log_summary();
    }

  private:
    void read_augmentation_metadata() {
        using namespace upf_utilities;
        auto position = content_.find("<PP_AUGMENTATION");
        if (position == std::string::npos)
            throw std::runtime_error("Malformed PAW UPF file " + source_description_ +
                                     ": no PP_AUGMENTATION section");

        auto tag_end = content_.find(">", position);
        std::string tag_line = content_.substr(position, tag_end - position);

        pseudopotential_.augmentation_shape = trim(xml_attribute(tag_line, "shape"));

        std::string cutoff_radius = xml_attribute(tag_line, "cutoff_r");
        if (!cutoff_radius.empty())
            pseudopotential_.augmentation_cutoff_radius = std::stod(cutoff_radius);

        /**
         * cutoff_r_index and iraug carry the same grid index; prefer the former
         * and fall back to the latter, which is what older writers emit.
         */
        std::string cutoff_index = xml_attribute(tag_line, "cutoff_r_index");
        if (cutoff_index.empty())
            cutoff_index = xml_attribute(tag_line, "iraug");
        if (!cutoff_index.empty())
            pseudopotential_.augmentation_cutoff_radius_index = std::stoi(cutoff_index);

        std::string maximum_angular_momentum_augmentation = xml_attribute(tag_line, "lmax_aug");
        if (maximum_angular_momentum_augmentation.empty())
            maximum_angular_momentum_augmentation = xml_attribute(tag_line, "l_max_aug");
        if (!maximum_angular_momentum_augmentation.empty())
            pseudopotential_.maximum_angular_momentum_augmentation =
                std::stoi(maximum_angular_momentum_augmentation);

        /**
         * nqlc is the number of angular-momentum channels, 2·l_max + 1. Derive it
         * from l_max when the attribute is absent so the two never disagree.
         */
        std::string number_of_channels = xml_attribute(tag_line, "nqlc");
        pseudopotential_.number_of_augmentation_angular_momenta =
            number_of_channels.empty() ? 2 * pseudopotential_.maximum_angular_momentum + 1
                                       : std::stoi(number_of_channels);
    }

    void read_partial_waves() {
        using namespace upf_utilities;
        auto block_start = content_.find("<PP_FULL_WFC");
        if (block_start == std::string::npos)
            throw std::runtime_error(
                "Malformed PAW UPF file " + source_description_ +
                ": no PP_FULL_WFC section, so the all-electron and pseudo partial waves "
                "the one-center terms are built from are absent");
        auto block_end = content_.find("</PP_FULL_WFC>", block_start);
        if (block_end == std::string::npos)
            throw std::runtime_error("Malformed PAW UPF file " + source_description_ +
                                     ": unterminated PP_FULL_WFC section");

        int number_of_channels = pseudopotential_.number_of_projectors;
        pseudopotential_.partial_waves_all_electron.resize(number_of_channels);
        pseudopotential_.partial_waves_pseudo.resize(number_of_channels);

        for (int channel_index = 0; channel_index < number_of_channels; channel_index++) {
            read_partial_wave("PP_AEWFC",
                              channel_index,
                              block_start,
                              block_end,
                              pseudopotential_.partial_waves_all_electron[channel_index]);
            read_partial_wave("PP_PSWFC",
                              channel_index,
                              block_start,
                              block_end,
                              pseudopotential_.partial_waves_pseudo[channel_index]);
        }
    }

    void read_partial_wave(const std::string& tag_prefix,
                           int channel_index,
                           size_t block_start,
                           size_t block_end,
                           PartialWave& partial_wave) {
        using namespace upf_utilities;
        std::string tag = "<" + tag_prefix + "." + std::to_string(channel_index + 1);
        auto position = content_.find(tag, block_start);
        if (position == std::string::npos || position >= block_end)
            throw std::runtime_error("Malformed PAW UPF file " + source_description_ +
                                     ": missing " + tag.substr(1) + " inside PP_FULL_WFC");

        auto tag_end = content_.find(">", position);
        std::string tag_line = content_.substr(position, tag_end - position);

        std::string angular_momentum = xml_attribute(tag_line, " l");
        if (angular_momentum.empty())
            angular_momentum = xml_attribute(tag_line, "\nl");
        partial_wave.angular_momentum = angular_momentum.empty() ? 0 : std::stoi(angular_momentum);

        std::string close_tag = "</" + tag_prefix + "." + std::to_string(channel_index + 1) + ">";
        auto close_position = content_.find(close_tag, position);
        if (close_position == std::string::npos)
            throw std::runtime_error("Malformed PAW UPF file " + source_description_ +
                                     ": unterminated " + tag.substr(1));

        std::string section = content_.substr(tag_end + 1, close_position - tag_end - 1);
        partial_wave.radial_values = xml_data(section);
        partial_wave.radial_values.resize(pseudopotential_.mesh_size, 0.0);
    }

    void read_paw_block() {
        using namespace upf_utilities;
        auto block_start = content_.find("<PP_PAW");
        if (block_start == std::string::npos)
            throw std::runtime_error("Malformed PAW UPF file " + source_description_ +
                                     ": no PP_PAW section");

        auto tag_end = content_.find(">", block_start);
        std::string tag_line = content_.substr(block_start, tag_end - block_start);
        std::string core_energy = xml_attribute(tag_line, "core_energy");
        if (!core_energy.empty()) {
            pseudopotential_.core_energy = std::stod(core_energy);
            pseudopotential_.has_core_energy = true;
        }

        pseudopotential_.channel_occupations = read_section("PP_OCCUPATIONS", block_start);
        pseudopotential_.channel_occupations.resize(pseudopotential_.number_of_projectors, 0.0);

        pseudopotential_.charge_core_all_electron.radial_values =
            read_section("PP_AE_NLCC", block_start);
        pseudopotential_.charge_core_all_electron.radial_values.resize(pseudopotential_.mesh_size,
                                                                       0.0);

        pseudopotential_.potential_local_all_electron.radial_values =
            read_section("PP_AE_VLOC", block_start);
        pseudopotential_.potential_local_all_electron.radial_values.resize(
            pseudopotential_.mesh_size, 0.0);
    }

    void read_multipole_moments() {
        using namespace upf_utilities;
        int number_of_channels = pseudopotential_.number_of_projectors;
        int number_of_angular_momenta = pseudopotential_.number_of_augmentation_angular_momenta;

        pseudopotential_.multipole_moments.assign(
            number_of_angular_momenta,
            std::vector<std::vector<Real>>(number_of_channels,
                                           std::vector<Real>(number_of_channels, 0.0)));

        std::vector<Real> flat = read_section("PP_MULTIPOLES", 0);
        if (flat.empty())
            return;

        /**
         * Stored angular-momentum-major: index = L·nbeta² + i·nbeta + j. Verified
         * against both fixture datasets — the leading nbeta² block reproduces
         * PP_Q exactly, and every entry matches ∫(φ_i φ_j − φ̃_i φ̃_j) r^L dr on
         * the channels the triangle condition allows.
         */
        auto expected_size = static_cast<size_t>(number_of_angular_momenta) * number_of_channels *
                             number_of_channels;
        if (flat.size() < expected_size)
            throw std::runtime_error("Malformed PAW UPF file " + source_description_ +
                                     ": PP_MULTIPOLES holds " + std::to_string(flat.size()) +
                                     " values, expected " + std::to_string(expected_size));

        for (int angular_momentum = 0; angular_momentum < number_of_angular_momenta;
             angular_momentum++) {
            for (int channel_1 = 0; channel_1 < number_of_channels; channel_1++) {
                for (int channel_2 = 0; channel_2 < number_of_channels; channel_2++) {
                    size_t flat_index = static_cast<size_t>(angular_momentum) * number_of_channels *
                                            number_of_channels +
                                        static_cast<size_t>(channel_1) * number_of_channels +
                                        channel_2;
                    pseudopotential_.multipole_moments[angular_momentum][channel_1][channel_2] =
                        flat[flat_index];
                }
            }
        }
    }

    /// Read the numeric body of a section, searching from a starting offset.
    std::vector<Real> read_section(const std::string& tag_name, size_t search_from) {
        using namespace upf_utilities;
        auto position = content_.find("<" + tag_name + " ", search_from);
        if (position == std::string::npos)
            position = content_.find("<" + tag_name + ">", search_from);
        if (position == std::string::npos)
            return {};

        auto tag_end = content_.find(">", position);
        auto close_position = content_.find("</" + tag_name + ">", tag_end);
        if (close_position == std::string::npos)
            return {};

        return xml_data(content_.substr(tag_end + 1, close_position - tag_end - 1));
    }

    void log_summary() {
        log::info("     UPFv2 PAW: {} channels, aug shape={} r_aug={:.4f} bohr (index {}) l_max={}",
                  pseudopotential_.number_of_projectors,
                  pseudopotential_.augmentation_shape,
                  pseudopotential_.augmentation_cutoff_radius,
                  pseudopotential_.augmentation_cutoff_radius_index,
                  pseudopotential_.maximum_angular_momentum_augmentation);
    }

    PawPseudopotential& pseudopotential_;
    const std::string& content_;
    std::string source_description_;
};

} // namespace q3
