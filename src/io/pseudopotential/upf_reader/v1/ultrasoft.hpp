#pragma once

/**
 * UPF v1 ultrasoft reader.
 *
 * Reads the ultrasoft-specific sections of a UPF v1 format file:
 * PP_QIJ (augmentation charges Q_ij), PP_RINNER (the L-dependent inner radii
 * r_L), PP_QFCOEF (polynomial coefficients), and rebuilds the L-resolved
 * augmentation functions from the tabulated Q_ij and those coefficients.
 *
 * That rebuild is what the record layout means rather than an extra step.
 * PP_QIJ tabulates Q_ij(r) correctly only outside r_L; inside r_L it carries
 * the generator's minimal-L form only, and PP_QFCOEF exists so the correct
 * form can be recovered for each admissible L. The file states the rule on
 * its own nqf line: "If not zero, Qij's inside rinner are computed using
 * qfcoef's". See Vanderbilt, Phys. Rev. B 41, 7892 (1990) for the underlying
 * decomposition of Q_ij into total-angular-momentum channels.
 *
 * Must be called after UPFv1Reader has populated the base fields.
 */

#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "io/logger.hpp"
#include "io/pseudopotential/upf_reader/utilities.hpp"
#include "layer_3_representation/pseudopotential/ultrasoft/init.hpp"

namespace q3 {

class UltrasoftUPFv1Reader {
  public:
    UltrasoftUPFv1Reader(UltrasoftPseudopotential& pseudopotential, std::ifstream& file_stream)
        : pseudopotential_(pseudopotential), file_stream_(file_stream) {}

    void read() {
        log::info("     PP {}: Ultrasoft pseudopotential", pseudopotential_.element);

        read_qij_header();
        read_rinner();
        read_qij_data();
        reconstruct_l_resolved_qfuncl();
    }

  private:
    void read_qij_header() {
        using namespace upf_utilities;
        file_stream_.clear();
        file_stream_.seekg(0);
        if (!find_tag(file_stream_, "<PP_QIJ>"))
            throw std::runtime_error("PP_QIJ section not found for ultrasoft PP");

        std::string header_line;
        std::getline(file_stream_, header_line);
        {
            std::istringstream stream(header_line);
            stream >> pseudopotential_.number_of_q_coefficients;
        }

        int number_of_betas = pseudopotential_.number_of_projectors;
        pseudopotential_.augmentation_function.resize(number_of_betas * number_of_betas);
        pseudopotential_.augmentation_overlap_integrals.resize(
            number_of_betas, std::vector<Real>(number_of_betas, 0.0));
        pseudopotential_.augmentation_coefficients.resize(number_of_betas * number_of_betas);
        pseudopotential_.inner_cutoff_radii.resize(
            2 * pseudopotential_.maximum_angular_momentum + 1, 0.0);
    }

    void read_rinner() {
        using namespace upf_utilities;
        if (pseudopotential_.number_of_q_coefficients <= 0)
            return;

        file_stream_.clear();
        file_stream_.seekg(0);
        if (find_tag(file_stream_, "<PP_RINNER>")) {
            int number_of_rinner_entries = 2 * pseudopotential_.maximum_angular_momentum + 1;
            for (int angular_momentum = 0; angular_momentum < number_of_rinner_entries;
                 angular_momentum++) {
                int angular_momentum_index;
                Real inner_radius;
                file_stream_ >> angular_momentum_index >> inner_radius;
                pseudopotential_.inner_cutoff_radii[angular_momentum] = inner_radius;
            }
            file_stream_.clear();
        }
    }

    void read_qij_data() {
        using namespace upf_utilities;
        /**
         * Read all lines from the QIJ section, stripping XML tags and marking QFCOEF boundaries.
         * This collects the raw data for sequential parsing of Q_ij pairs.
         */
        file_stream_.clear();
        file_stream_.seekg(0);
        find_tag(file_stream_, "<PP_QIJ>");

        std::vector<std::string> qij_lines;
        {
            std::string line;
            std::getline(file_stream_, line); // skip nqf line
            bool inside_rinner_section = false;
            while (std::getline(file_stream_, line)) {
                if (line.find("</PP_QIJ>") != std::string::npos)
                    break;
                if (line.find("<PP_RINNER>") != std::string::npos) {
                    inside_rinner_section = true;
                    continue;
                }
                if (line.find("</PP_RINNER>") != std::string::npos) {
                    inside_rinner_section = false;
                    continue;
                }
                if (inside_rinner_section)
                    continue;
                if (line.find("<PP_QFCOEF>") != std::string::npos) {
                    qij_lines.push_back("__QFCOEF_BEGIN__");
                    continue;
                }
                if (line.find("</PP_QFCOEF>") != std::string::npos) {
                    qij_lines.push_back("__QFCOEF_END__");
                    continue;
                }
                qij_lines.push_back(line);
            }
        }

        // Parse the collected lines sequentially
        size_t line_index = 0;
        auto next_non_empty_line = [&]() -> std::string {
            while (line_index < qij_lines.size()) {
                std::string trimmed = trim(qij_lines[line_index++]);
                if (!trimmed.empty())
                    return trimmed;
            }
            return "";
        };
        auto read_reals_from_lines = [&](int count) -> std::vector<Real> {
            std::vector<Real> values;
            values.reserve(count);
            while (static_cast<int>(values.size()) < count && line_index < qij_lines.size()) {
                const std::string& raw = qij_lines[line_index];
                if (raw == "__QFCOEF_BEGIN__" || raw == "__QFCOEF_END__") {
                    line_index++;
                    continue;
                }
                std::istringstream stream(raw);
                Real value;
                while (static_cast<int>(values.size()) < count && stream >> value) {
                    values.push_back(value);
                }
                line_index++;
            }
            return values;
        };

        int number_of_betas = pseudopotential_.number_of_projectors;
        for (int beta_index_1 = 0; beta_index_1 < number_of_betas; beta_index_1++) {
            for (int beta_index_2 = beta_index_1; beta_index_2 < number_of_betas; beta_index_2++) {
                // Header line: i, j, l(j) (1-based indices) — not used
                next_non_empty_line();

                // Q_int line
                std::string q_integral_line = next_non_empty_line();
                {
                    std::istringstream stream(q_integral_line);
                    stream >>
                        pseudopotential_.augmentation_overlap_integrals[beta_index_1][beta_index_2];
                    pseudopotential_.augmentation_overlap_integrals[beta_index_2][beta_index_1] =
                        pseudopotential_.augmentation_overlap_integrals[beta_index_1][beta_index_2];
                }

                int pair_index = beta_index_1 * number_of_betas + beta_index_2;
                int symmetric_pair_index = beta_index_2 * number_of_betas + beta_index_1;

                // Q_ij(r) on the radial grid
                pseudopotential_.augmentation_function[pair_index] =
                    read_reals_from_lines(pseudopotential_.mesh_size);
                pseudopotential_.augmentation_function[symmetric_pair_index] =
                    pseudopotential_.augmentation_function[pair_index];

                // Q function polynomial coefficients (inside PP_QFCOEF markers)
                if (pseudopotential_.number_of_q_coefficients > 0) {
                    // Skip __QFCOEF_BEGIN__ marker
                    while (line_index < qij_lines.size() &&
                           qij_lines[line_index] == "__QFCOEF_BEGIN__")
                        line_index++;

                    pseudopotential_.augmentation_coefficients[pair_index].resize(
                        2 * pseudopotential_.maximum_angular_momentum + 1);
                    for (int angular_momentum = 0;
                         angular_momentum < 2 * pseudopotential_.maximum_angular_momentum + 1;
                         angular_momentum++) {
                        pseudopotential_.augmentation_coefficients[pair_index][angular_momentum] =
                            read_reals_from_lines(pseudopotential_.number_of_q_coefficients);
                    }
                    pseudopotential_.augmentation_coefficients[symmetric_pair_index] =
                        pseudopotential_.augmentation_coefficients[pair_index];

                    // Skip __QFCOEF_END__ marker
                    while (line_index < qij_lines.size() &&
                           qij_lines[line_index] == "__QFCOEF_END__")
                        line_index++;
                }
            }
        }

        log::info("     PP {}: {} Q_ij pairs read, nqf={}",
                  pseudopotential_.element,
                  number_of_betas * (number_of_betas + 1) / 2,
                  pseudopotential_.number_of_q_coefficients);
    }

    void reconstruct_l_resolved_qfuncl() {
        /**
         * Decompose the tabulated Q_ij(r) into its L-resolved channels.
         *
         * For each (nb1, nb2) pair and each L the triangle rule admits
         * (L = |l1-l2|, |l1-l2|+2, ... , l1+l2):
         *    1) Copy the tabulated Q_ij(r) into channel L. Outside r_L that
         *       function is already correct and does not depend on L.
         *    2) If rinner[L] > 0, replace the points inside r_L with the
         *       polynomial PP_QFCOEF stores for that channel.
         *
         * A file with nqf = 0 carries no PP_QFCOEF block, so there is nothing
         * to decompose with: this returns early and leaves the per-L array
         * empty. That is not a missing input downstream —
         * charge_density_augmentation.hpp reuses the single tabulated Q_ij(r)
         * at *every* admissible L for such files, which is where their L > 0
         * augmentation comes from.
         */
        int number_of_betas = pseudopotential_.number_of_projectors;
        pseudopotential_.number_of_q_angular_momenta =
            2 * pseudopotential_.maximum_angular_momentum + 1;

        if (pseudopotential_.number_of_q_coefficients <= 0)
            return;

        pseudopotential_.augmentation_function_by_angular_momentum.resize(number_of_betas *
                                                                          number_of_betas);
        for (auto& entry : pseudopotential_.augmentation_function_by_angular_momentum)
            entry.resize(pseudopotential_.number_of_q_angular_momenta);

        for (int beta_index_1 = 0; beta_index_1 < number_of_betas; beta_index_1++) {
            for (int beta_index_2 = beta_index_1; beta_index_2 < number_of_betas; beta_index_2++) {
                int pair_index = beta_index_1 * number_of_betas + beta_index_2;
                int symmetric_pair_index = beta_index_2 * number_of_betas + beta_index_1;
                int angular_momentum_1 = pseudopotential_.projectors[beta_index_1].angular_momentum;
                int angular_momentum_2 = pseudopotential_.projectors[beta_index_2].angular_momentum;

                // Skip if total qfunc not available for this pair
                if (pair_index >= static_cast<int>(pseudopotential_.augmentation_function.size()) ||
                    pseudopotential_.augmentation_function[pair_index].empty())
                    continue;

                // Copy total qfunc to each valid L channel
                for (int angular_momentum = std::abs(angular_momentum_1 - angular_momentum_2);
                     angular_momentum <= angular_momentum_1 + angular_momentum_2;
                     angular_momentum += 2) {
                    if (angular_momentum >= pseudopotential_.number_of_q_angular_momenta)
                        continue;
                    pseudopotential_
                        .augmentation_function_by_angular_momentum[pair_index][angular_momentum] =
                        pseudopotential_.augmentation_function[pair_index];
                    if (symmetric_pair_index != pair_index)
                        pseudopotential_
                            .augmentation_function_by_angular_momentum[symmetric_pair_index]
                                                                      [angular_momentum] =
                            pseudopotential_.augmentation_function[pair_index];
                }

                // Replace the points inside r_L with the PP_QFCOEF polynomial
                if (pair_index <
                        static_cast<int>(pseudopotential_.augmentation_coefficients.size()) &&
                    !pseudopotential_.augmentation_coefficients[pair_index].empty()) {
                    for (int angular_momentum = std::abs(angular_momentum_1 - angular_momentum_2);
                         angular_momentum <= angular_momentum_1 + angular_momentum_2;
                         angular_momentum += 2) {
                        if (angular_momentum >= pseudopotential_.number_of_q_angular_momenta)
                            continue;
                        if (angular_momentum >=
                                static_cast<int>(pseudopotential_.inner_cutoff_radii.size()) ||
                            pseudopotential_.inner_cutoff_radii[angular_momentum] <= 0.0)
                            continue;
                        if (angular_momentum >=
                            static_cast<int>(
                                pseudopotential_.augmentation_coefficients[pair_index].size()))
                            continue;
                        if (static_cast<int>(
                                pseudopotential_
                                    .augmentation_coefficients[pair_index][angular_momentum]
                                    .size()) < pseudopotential_.number_of_q_coefficients)
                            continue;

                        // Find last grid point inside rinner[L]
                        int last_inner_point = 0;
                        for (int radial_index = 0; radial_index < pseudopotential_.mesh_size;
                             radial_index++) {
                            if (pseudopotential_.radial_grid[radial_index] <
                                pseudopotential_.inner_cutoff_radii[angular_momentum]) {
                                last_inner_point = radial_index;
                            }
                        }

                        /**
                         * The polynomial PP_QFCOEF stores for this channel:
                         *
                         *     Q_L(r) = r^(L+2) * Σ_i qfcoef[i] * r^(2*(i-1))
                         *
                         * The exponent is L+2, not L: r^L is the multipole's
                         * leading behaviour and the extra r^2 is this
                         * format's storage convention, since the tabulated
                         * radial functions are r*psi(r) and a product of two
                         * of them carries r^2. Consumers integrate these
                         * against j_L(qr) with no further r^2 — see
                         * charge_density_augmentation.hpp.
                         */
                        const auto& coefficients =
                            pseudopotential_
                                .augmentation_coefficients[pair_index][angular_momentum];
                        for (int radial_index = 0; radial_index <= last_inner_point;
                             radial_index++) {
                            Real radius_squared = pseudopotential_.radial_grid[radial_index] *
                                                  pseudopotential_.radial_grid[radial_index];
                            Real polynomial_value = coefficients[0];
                            for (int coefficient_index = 1;
                                 coefficient_index < pseudopotential_.number_of_q_coefficients;
                                 coefficient_index++) {
                                polynomial_value +=
                                    coefficients[coefficient_index] *
                                    std::pow(radius_squared, static_cast<Real>(coefficient_index));
                            }
                            // Multiply by r^(L+2) to get Q_L(r) on the grid
                            Real q_value = polynomial_value *
                                           std::pow(pseudopotential_.radial_grid[radial_index],
                                                    angular_momentum + 2);
                            pseudopotential_.augmentation_function_by_angular_momentum
                                [pair_index][angular_momentum][radial_index] = q_value;
                            if (symmetric_pair_index != pair_index)
                                pseudopotential_.augmentation_function_by_angular_momentum
                                    [symmetric_pair_index][angular_momentum][radial_index] =
                                    q_value;
                        }
                    }
                }
            }
        }
        log::info("     PP {}: L-resolved qfuncl reconstructed from qfcoef (nqlc={})",
                  pseudopotential_.element,
                  pseudopotential_.number_of_q_angular_momenta);
    }

    UltrasoftPseudopotential& pseudopotential_;
    std::ifstream& file_stream_;
};

} // namespace q3
