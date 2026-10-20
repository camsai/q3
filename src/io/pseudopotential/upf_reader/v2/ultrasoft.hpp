#pragma once

/**
 * UPF v2 ultrasoft reader.
 *
 * Reads the ultrasoft-specific sections of a UPF v2 format file:
 * PP_Q (precomputed qqq integrals), PP_QIJL (L-resolved augmentation charges
 * on the radial grid), PP_QIJ.N (fallback total augmentation charges),
 * and computes missing qqq integrals by numerical integration if needed.
 *
 * Must be called after UPFv2Reader has populated the base fields.
 */

#include <cmath>
#include <sstream>
#include <string>
#include <vector>

#include "io/logger.hpp"
#include "io/pseudopotential/upf_reader/utilities.hpp"
#include "layer_0_core/math/integration/simpson.hpp"
#include "layer_3_representation/pseudopotential/ultrasoft/init.hpp"

namespace q3 {

class UltrasoftUPFv2Reader {
  public:
    UltrasoftUPFv2Reader(UltrasoftPseudopotential& pseudopotential, const std::string& file_content)
        : pseudopotential_(pseudopotential), content_(file_content) {}

    void read() {
        int number_of_betas = pseudopotential_.number_of_projectors;
        int number_of_pairs = number_of_betas * number_of_betas;
        pseudopotential_.augmentation_function.resize(number_of_pairs);
        pseudopotential_.augmentation_overlap_integrals.resize(
            number_of_betas, std::vector<Real>(number_of_betas, 0.0));
        pseudopotential_.number_of_q_coefficients =
            0; // UPF v2 uses tabulated Q on mesh, no polynomial coefficients

        read_precomputed_qqq();
        read_l_resolved_augmentation();
        read_total_augmentation_fallback();
        compute_missing_qqq_integrals();
    }

  private:
    void read_precomputed_qqq() {
        using namespace upf_utilities;
        int number_of_betas = pseudopotential_.number_of_projectors;

        auto position = content_.find("<PP_Q>");
        if (position == std::string::npos)
            position = content_.find("<PP_Q ");
        if (position == std::string::npos)
            return;

        auto end_position = content_.find("</PP_Q>", position);
        if (end_position == std::string::npos)
            return;

        auto tag_end = content_.find(">", position);
        std::string section = content_.substr(tag_end + 1, end_position - tag_end - 1);
        std::vector<Real> qqq_flat = xml_data(section);
        if (static_cast<int>(qqq_flat.size()) >= number_of_betas * number_of_betas) {
            for (int row = 0; row < number_of_betas; row++) {
                for (int column = 0; column < number_of_betas; column++) {
                    pseudopotential_.augmentation_overlap_integrals[row][column] =
                        qqq_flat[row * number_of_betas + column];
                }
            }
        }
    }

    void read_l_resolved_augmentation() {
        using namespace upf_utilities;
        int number_of_betas = pseudopotential_.number_of_projectors;
        pseudopotential_.number_of_q_angular_momenta =
            2 * pseudopotential_.maximum_angular_momentum + 1;
        pseudopotential_.augmentation_function_by_angular_momentum.resize(number_of_betas *
                                                                          number_of_betas);
        for (auto& entry : pseudopotential_.augmentation_function_by_angular_momentum)
            entry.resize(pseudopotential_.number_of_q_angular_momenta);

        size_t search_position = 0;
        while ((search_position = content_.find("<PP_QIJL", search_position)) !=
               std::string::npos) {
            auto tag_end = content_.find(">", search_position);
            std::string tag_line = content_.substr(search_position, tag_end - search_position);

            std::string angular_momentum_string = xml_attribute(tag_line, "angular_momentum");
            std::string first_index_string = xml_attribute(tag_line, "first_index");
            std::string second_index_string = xml_attribute(tag_line, "second_index");
            if (first_index_string.empty() || second_index_string.empty() ||
                angular_momentum_string.empty()) {
                search_position = tag_end;
                continue;
            }
            int beta_index_1 = std::stoi(first_index_string) - 1; // Convert 1-based to 0-based
            int beta_index_2 = std::stoi(second_index_string) - 1;
            int angular_momentum = std::stoi(angular_momentum_string);

            if (beta_index_1 < 0 || beta_index_1 >= number_of_betas || beta_index_2 < 0 ||
                beta_index_2 >= number_of_betas || angular_momentum < 0 ||
                angular_momentum >= pseudopotential_.number_of_q_angular_momenta) {
                search_position = tag_end;
                continue;
            }

            // Extract the tag name for closing tag (e.g., PP_QIJL.1.1.0)
            auto space_position = tag_line.find(" ");
            std::string tag_name = tag_line.substr(1, space_position - 1);
            std::string close_tag = "</" + tag_name + ">";

            auto close_position = content_.find(close_tag, search_position);
            if (close_position != std::string::npos) {
                std::string section = content_.substr(tag_end + 1, close_position - tag_end - 1);
                int pair_index = beta_index_1 * number_of_betas + beta_index_2;
                int symmetric_pair_index = beta_index_2 * number_of_betas + beta_index_1;
                auto data = xml_data(section);
                data.resize(pseudopotential_.mesh_size, 0.0);
                pseudopotential_
                    .augmentation_function_by_angular_momentum[pair_index][angular_momentum] = data;
                pseudopotential_.augmentation_function_by_angular_momentum[symmetric_pair_index]
                                                                          [angular_momentum] = data;

                // Also populate legacy qfunc with L=0 component
                if (angular_momentum == 0) {
                    pseudopotential_.augmentation_function[pair_index] = data;
                    pseudopotential_.augmentation_function[symmetric_pair_index] = data;
                }
            }

            search_position = tag_end;
        }
    }

    void read_total_augmentation_fallback() {
        using namespace upf_utilities;
        int number_of_betas = pseudopotential_.number_of_projectors;

        /**
         * Fall back: PP_QIJ blocks (the `q_with_l="false"` format: one
         * un-decomposed radial function per (i, j) projector pair, used
         * instead of the L-resolved PP_QIJL blocks read above).
         *
         * The block for pair (i, j) is *not* reliably named `<PP_QIJ.N>`
         * with N a linear/composite index — real files in the wild name it
         * `<PP_QIJ.i.j>` with i, j the
         * 1-based `first_index`/`second_index`, and separately carry the
         * linear index as a `composite_index` attribute. Guessing the tag
         * name from a computed index and substring-searching for it — the
         * previous approach — matches the wrong block for almost every pair
         * (searching for "<PP_QIJ.2" finds "<PP_QIJ.2.2" before it ever
         * reaches "<PP_QIJ.1.2"), silently cross-wiring one projector pair's
         * augmentation charge onto another and breaking charge conservation.
         * Reading `first_index`/`second_index` as attributes off whatever
         * tag is actually there — the same approach `read_l_resolved_augmentation`
         * already uses for PP_QIJL — sidesteps the naming convention entirely.
         */
        /**
         * The trailing period in the search string is what keeps this off
         * `<PP_QIJL...>` tags (already handled above): a real PP_QIJL tag has
         * 'L' where this pattern requires '.', so `find` can never land on
         * one -- no separate exclusion check is needed.
         */
        size_t search_position = 0;
        while ((search_position = content_.find("<PP_QIJ.", search_position)) !=
               std::string::npos) {
            auto tag_end = content_.find(">", search_position);
            std::string tag_line = content_.substr(search_position, tag_end - search_position);

            std::string first_index_string = xml_attribute(tag_line, "first_index");
            std::string second_index_string = xml_attribute(tag_line, "second_index");
            /**
             * Last-resort tolerance of a malformed file, and loudly so.
             *
             * The bug this reader exists to fix was a *silent* mis-association
             * of augmentation data; skipping a block we cannot interpret
             * without saying so is the same failure with a different cause, so
             * every skip below names the tag and the reason.
             */
            if (first_index_string.empty() || second_index_string.empty()) {
                log::warn("     PP: skipping augmentation block '{}' -- it carries no "
                          "first_index/second_index attributes to place it by",
                          tag_line);
                search_position = tag_end;
                continue;
            }
            int beta_index_1 = std::stoi(first_index_string) - 1; // Convert 1-based to 0-based
            int beta_index_2 = std::stoi(second_index_string) - 1;

            if (beta_index_1 < 0 || beta_index_1 >= number_of_betas || beta_index_2 < 0 ||
                beta_index_2 >= number_of_betas) {
                search_position = tag_end;
                continue;
            }

            int pair_index = beta_index_1 * number_of_betas + beta_index_2;
            int symmetric_pair_index = beta_index_2 * number_of_betas + beta_index_1;

            // Extract the tag name for the matching closing tag (e.g. PP_QIJ.1.2)
            auto space_position = tag_line.find(" ");
            std::string tag_name = tag_line.substr(1, space_position - 1);
            std::string close_tag = "</" + tag_name + ">";
            auto close_position = content_.find(close_tag, search_position);
            if (close_position == std::string::npos) {
                search_position = tag_end;
                continue;
            }

            if (pseudopotential_.augmentation_function[pair_index].empty()) {
                // Not already loaded from PP_QIJL
                std::string section = content_.substr(tag_end + 1, close_position - tag_end - 1);
                pseudopotential_.augmentation_function[pair_index] = xml_data(section);
                pseudopotential_.augmentation_function[pair_index].resize(
                    pseudopotential_.mesh_size, 0.0);
                pseudopotential_.augmentation_function[symmetric_pair_index] =
                    pseudopotential_.augmentation_function[pair_index];
            }

            // Get Q_int from attributes if PP_Q was not found
            if (pseudopotential_.augmentation_overlap_integrals[beta_index_1][beta_index_2] ==
                0.0) {
                std::string q_integral_string = xml_attribute(tag_line, "q_int");
                if (!q_integral_string.empty()) {
                    pseudopotential_.augmentation_overlap_integrals[beta_index_1][beta_index_2] =
                        std::stod(q_integral_string);
                    pseudopotential_.augmentation_overlap_integrals[beta_index_2][beta_index_1] =
                        pseudopotential_.augmentation_overlap_integrals[beta_index_1][beta_index_2];
                }
            }

            search_position = tag_end;
        }
    }

    void compute_missing_qqq_integrals() {
        int number_of_betas = pseudopotential_.number_of_projectors;

        // Last resort: compute qqq by integrating Q(r) if still zero
        for (int beta_index_1 = 0; beta_index_1 < number_of_betas; beta_index_1++) {
            for (int beta_index_2 = beta_index_1; beta_index_2 < number_of_betas; beta_index_2++) {
                if (pseudopotential_.augmentation_overlap_integrals[beta_index_1][beta_index_2] ==
                        0.0 &&
                    !pseudopotential_
                         .augmentation_function[beta_index_1 * number_of_betas + beta_index_2]
                         .empty()) {
                    /**
                     * qfunc from PP_QIJL already contains r² factor,
                     * so qqq = ∫ qfunc * radial_grid_derivative dr (no extra r²)
                     */
                    Real integral = simpson_rule(
                        pseudopotential_
                            .augmentation_function[beta_index_1 * number_of_betas + beta_index_2],
                        pseudopotential_.radial_grid_derivative,
                        pseudopotential_.mesh_size);
                    pseudopotential_.augmentation_overlap_integrals[beta_index_1][beta_index_2] =
                        integral;
                    pseudopotential_.augmentation_overlap_integrals[beta_index_2][beta_index_1] =
                        integral;
                }
            }
        }
    }

    UltrasoftPseudopotential& pseudopotential_;
    const std::string& content_;
};

} // namespace q3
