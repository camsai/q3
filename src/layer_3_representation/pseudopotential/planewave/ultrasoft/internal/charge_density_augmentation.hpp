#pragma once

/**
 * AugmentationQTable — precomputed augmentation charge Q_ij(q, L) radial
 * interpolation tables for ultrasoft pseudopotential_set.
 *
 * Owns the Bessel-transformed Q radial tables and provides interpolation
 * accessors used by the D-matrix, augmentation charge, and stress routines.
 */

#include <algorithm>
#include <cmath>
#include <vector>

#include "layer_0_core/math/interpolation/lagrange.hpp"
#include "layer_0_core/math/spherical/harmonics.hpp"
#include "layer_0_core/types/init.hpp"
#include "layer_1_structure/crystal/crystal.hpp"
#include "layer_3_representation/pseudopotential/set.hpp"
#include "layer_3_representation/pseudopotential/ultrasoft/init.hpp"
#include "layer_3_representation/pseudopotential/ultrasoft/internal/charge_density_augmentation.hpp"

namespace q3 {

class AugmentationQTable : public AugmentationChargeTable {
  public:
    AugmentationQTable() = default;
    ~AugmentationQTable() override = default;

    /**
     * Initialize augmentation Q tables for all ultrasoft species.
     *
     * @param pseudopotential_set   The pseudopotential set (one per species type).
     * @param mesh_sizes         Truncated radial mesh size per species.
     * @param number_of_q_points Number of points in the q-grid.
     * @param q_spacing          Spacing between q-grid points.
     * @param cell_volume        Unit cell volume in Bohr³.
     */
    void init(const PseudopotentialSet& pseudopotential_set,
              const std::vector<int>& mesh_sizes,
              int number_of_q_points,
              Real q_spacing,
              Real cell_volume) {
        number_of_q_points_ = number_of_q_points;
        q_spacing_ = q_spacing;
        cell_volume_ = cell_volume;

        int number_of_species_types = static_cast<int>(pseudopotential_set.size());

        determine_maximum_angular_momentum_q(pseudopotential_set, number_of_species_types);
        build_all_tables(pseudopotential_set, mesh_sizes, number_of_species_types);
    }

    /**
     * Interpolate the L-resolved Q radial function at a given q value.
     */
    [[nodiscard]] Real interpolate_qrad(int species_type_index,
                                        int projector_pair_index,
                                        int angular_momentum,
                                        Real q_value) const override {
        if (species_type_index >= static_cast<int>(augmentation_q_radial_table_.size()) ||
            projector_pair_index >=
                static_cast<int>(augmentation_q_radial_table_[species_type_index].size()) ||
            angular_momentum >=
                static_cast<int>(
                    augmentation_q_radial_table_[species_type_index][projector_pair_index]
                        .size()) ||
            augmentation_q_radial_table_[species_type_index][projector_pair_index][angular_momentum]
                .empty()) {
            return 0.0;
        }
        return lagrange_interpolation_4point(
            augmentation_q_radial_table_[species_type_index][projector_pair_index]
                                        [angular_momentum],
            q_value,
            q_spacing_,
            number_of_q_points_);
    }

    /**
     * Interpolate the derivative dQ_rad/dq of the L-resolved Q radial function.
     */
    [[nodiscard]] Real interpolate_qrad_derivative(int species_type_index,
                                                   int projector_pair_index,
                                                   int angular_momentum,
                                                   Real q_value) const override {
        if (species_type_index >= static_cast<int>(augmentation_q_radial_table_.size()) ||
            projector_pair_index >=
                static_cast<int>(augmentation_q_radial_table_[species_type_index].size()) ||
            angular_momentum >=
                static_cast<int>(
                    augmentation_q_radial_table_[species_type_index][projector_pair_index]
                        .size()) ||
            augmentation_q_radial_table_[species_type_index][projector_pair_index][angular_momentum]
                .empty()) {
            return 0.0;
        }
        return lagrange_interpolation_4point_derivative(
            augmentation_q_radial_table_[species_type_index][projector_pair_index]
                                        [angular_momentum],
            q_value,
            q_spacing_,
            number_of_q_points_);
    }

    /**
     * Interpolate the total (L=0) Q function at a given q value.
     */
    [[nodiscard]] Real interpolate_qfunc(int species_type_index,
                                         int projector_pair_index,
                                         Real q_value) const override {
        if (q_function_table_.empty() || q_function_table_[species_type_index].empty())
            return 0.0;
        return lagrange_interpolation_4point(
            q_function_table_[species_type_index][projector_pair_index],
            q_value,
            q_spacing_,
            number_of_q_points_);
    }

    [[nodiscard]] int maximum_angular_momentum_q() const override {
        return maximum_angular_momentum_q_;
    }

  private:
    /// Determine the maximum angular momentum for augmentation charges.
    void determine_maximum_angular_momentum_q(const PseudopotentialSet& pseudopotential_set,
                                              int number_of_species_types) {
        maximum_angular_momentum_q_ = 0;
        for (int species_type_index = 0; species_type_index < number_of_species_types;
             species_type_index++) {
            /**
             * Capability, not type. This table is built whenever *any* species
             * needs augmentation, so a mixed cell brings its norm-conserving
             * species through here too, and `as_ultrasoft` is a dynamic_cast
             * whose null check is an assert compiled out in Release. A species
             * without augmentation charges contributes no Q angular momenta.
             */
            if (!pseudopotential_set[species_type_index].has_augmentation_charges()) {
                continue;
            }
            const auto& ultrasoft_pseudopotential =
                pseudopotential_set.as_ultrasoft(species_type_index);
            if (ultrasoft_pseudopotential.number_of_q_angular_momenta > 0) {
                maximum_angular_momentum_q_ =
                    std::max(maximum_angular_momentum_q_,
                             ultrasoft_pseudopotential.number_of_q_angular_momenta - 1);
            }
        }
        if (maximum_angular_momentum_q_ == 0) {
            for (int species_type_index = 0; species_type_index < number_of_species_types;
                 species_type_index++) {
                maximum_angular_momentum_q_ =
                    std::max(maximum_angular_momentum_q_,
                             2 * pseudopotential_set[species_type_index].maximum_angular_momentum);
            }
        }
    }

    /// Build augmentation Q radial interpolation tables for all species.
    void build_all_tables(const PseudopotentialSet& pseudopotential_set,
                          const std::vector<int>& mesh_sizes,
                          int number_of_species_types) {
        augmentation_q_radial_table_.resize(number_of_species_types);
        q_function_table_.resize(number_of_species_types);
        for (int species_type_index = 0; species_type_index < number_of_species_types;
             species_type_index++) {
            /**
             * Same per-species gate as determine_maximum_angular_momentum_q: a
             * norm-conserving species in a mixed cell has no augmentation
             * tables to build, and asking for it via `as_ultrasoft` is the
             * Release null dereference. Its table entries stay empty, which is
             * what precompute_q_functions_g's own gate already expects.
             */
            if (!pseudopotential_set[species_type_index].has_augmentation_charges()) {
                continue;
            }
            build_tables_for_species(pseudopotential_set.as_ultrasoft(species_type_index),
                                     mesh_sizes[species_type_index],
                                     species_type_index);
        }
    }

    /**
     * Build augmentation Q tables for a single ultrasoft species.
     * j_L(q*r) depends on (L, q) and on nothing about the projector pair, so it
     * is evaluated once per (L, q) and reused across every pair the triangle
     * rule admits. That reuse is what puts L and q outside the pair loop; the
     * alternative -- pair outermost -- would make the table writes contiguous
     * but would repeat a whole-mesh Bessel recurrence per pair.
     */
    void build_tables_for_species(const UltrasoftPseudopotential& pseudopotential,
                                  int mesh_size,
                                  int species_type_index) {
        int number_of_projectors = pseudopotential.number_of_projectors;
        int number_of_q_angular_momenta = pseudopotential.number_of_q_angular_momenta > 0
                                              ? pseudopotential.number_of_q_angular_momenta
                                              : (2 * pseudopotential.maximum_angular_momentum + 1);

        augmentation_q_radial_table_[species_type_index].resize(number_of_projectors *
                                                                number_of_projectors);
        q_function_table_[species_type_index].resize(number_of_projectors * number_of_projectors);

        // Pre-allocate all tables
        for (int beta_index_i = 0; beta_index_i < number_of_projectors; beta_index_i++) {
            for (int beta_index_j = beta_index_i; beta_index_j < number_of_projectors;
                 beta_index_j++) {
                int projector_pair_index = beta_index_i * number_of_projectors + beta_index_j;
                int reverse_projector_pair_index =
                    beta_index_j * number_of_projectors + beta_index_i;
                augmentation_q_radial_table_[species_type_index][projector_pair_index].resize(
                    number_of_q_angular_momenta);
                augmentation_q_radial_table_[species_type_index][reverse_projector_pair_index]
                    .resize(number_of_q_angular_momenta);
                for (int angular_momentum = 0; angular_momentum < number_of_q_angular_momenta;
                     angular_momentum++) {
                    augmentation_q_radial_table_[species_type_index][projector_pair_index]
                                                [angular_momentum]
                                                    .assign(number_of_q_points_, 0.0);
                    augmentation_q_radial_table_[species_type_index][reverse_projector_pair_index]
                                                [angular_momentum]
                                                    .assign(number_of_q_points_, 0.0);
                }
            }
        }

        // Pre-collect active projector pairs per angular momentum channel
        struct ProjectorPairInfo {
            int beta_index_i;
            int beta_index_j;
            int projector_pair_index;
            int reverse_projector_pair_index;
            const std::vector<Real>* augmentation_function_data;
        };

        Real prefactor = 4.0 * constants::pi / cell_volume_;

        // Precompute bessel buffer
        std::vector<Real> bessel_values(mesh_size);
        std::vector<Real> integrand(mesh_size);

        /**
         * L outermost: the Bessel factor is shared by every pair at this (L, q),
         * and the surviving-pair list built below is itself L-dependent.
         */
        for (int angular_momentum = 0; angular_momentum < number_of_q_angular_momenta;
             angular_momentum++) {

            // Collect all projector pairs active for this angular momentum
            std::vector<ProjectorPairInfo> active_pairs;
            for (int beta_index_i = 0; beta_index_i < number_of_projectors; beta_index_i++) {
                for (int beta_index_j = beta_index_i; beta_index_j < number_of_projectors;
                     beta_index_j++) {
                    int angular_momentum_i =
                        pseudopotential.projectors[beta_index_i].angular_momentum;
                    int angular_momentum_j =
                        pseudopotential.projectors[beta_index_j].angular_momentum;

                    // Triangle rule check
                    if (angular_momentum < std::abs(angular_momentum_i - angular_momentum_j) ||
                        angular_momentum > angular_momentum_i + angular_momentum_j ||
                        (angular_momentum + angular_momentum_i + angular_momentum_j) % 2 != 0) {
                        continue;
                    }

                    int projector_pair_index = beta_index_i * number_of_projectors + beta_index_j;
                    int reverse_projector_pair_index =
                        beta_index_j * number_of_projectors + beta_index_i;

                    // Find the appropriate Q function data
                    const std::vector<Real>* augmentation_function_data = nullptr;
                    if (!pseudopotential.augmentation_function_by_angular_momentum.empty() &&
                        projector_pair_index <
                            static_cast<int>(
                                pseudopotential.augmentation_function_by_angular_momentum.size()) &&
                        angular_momentum <
                            static_cast<int>(
                                pseudopotential
                                    .augmentation_function_by_angular_momentum[projector_pair_index]
                                    .size()) &&
                        !pseudopotential
                             .augmentation_function_by_angular_momentum[projector_pair_index]
                                                                       [angular_momentum]
                             .empty()) {
                        augmentation_function_data =
                            &pseudopotential
                                 .augmentation_function_by_angular_momentum[projector_pair_index]
                                                                           [angular_momentum];
                    } else if (projector_pair_index <
                                   static_cast<int>(pseudopotential.augmentation_function.size()) &&
                               !pseudopotential.augmentation_function[projector_pair_index]
                                    .empty()) {
                        /**
                         * `q_with_l = false` pseudopotentials carry a single
                         * un-decomposed Q_ij(r) per projector pair rather than
                         * one per L. The established convention for these is to
                         * reuse that same radial function at *every* L the
                         * triangle rule admits for the pair, not just L=0:
                         * the flat function is the pair's whole augmentation
                         * content, so each allowed multipole draws on it.
                         * Restricting this fallback to L=0 silently dropped the
                         * L>0 augmentation for such files (Q_ij(G=0) — total
                         * charge — is unaffected, since a pure L>0 term
                         * integrates to zero, but the anisotropic shape it
                         * contributes to the Hartree/XC energy is not).
                         */
                        augmentation_function_data =
                            &pseudopotential.augmentation_function[projector_pair_index];
                    }

                    if (!augmentation_function_data ||
                        static_cast<int>(augmentation_function_data->size()) < mesh_size) {
                        continue;
                    }

                    active_pairs.push_back({beta_index_i,
                                            beta_index_j,
                                            projector_pair_index,
                                            reverse_projector_pair_index,
                                            augmentation_function_data});
                }
            }

            if (active_pairs.empty())
                continue;

            // For each q-point, compute bessel values ONCE and integrate all active pairs
            for (int q_index = 0; q_index < number_of_q_points_; q_index++) {
                Real q_value = q_index * q_spacing_;

                // Compute spherical Bessel function for all radial points at this (l, q)
                spherical_bessel_batch(angular_momentum,
                                       q_value,
                                       pseudopotential.radial_grid.data(),
                                       mesh_size,
                                       bessel_values.data());

                // Integrate against each active projector pair's Q function
                for (const auto& pair : active_pairs) {
                    Real sum = 0.0;
                    const auto& q_func = *pair.augmentation_function_data;
                    for (int radial_index = 0; radial_index < mesh_size; radial_index++) {
                        Real integrand_value = q_func[radial_index] * bessel_values[radial_index] *
                                               pseudopotential.radial_grid_derivative[radial_index];
                        Real weight;
                        if (radial_index == 0 || radial_index == mesh_size - 1)
                            weight = 1.0;
                        else if (radial_index % 2 == 1)
                            weight = 4.0;
                        else
                            weight = 2.0;
                        sum += weight * integrand_value;
                    }
                    Real table_value = sum / 3.0 * prefactor;
                    augmentation_q_radial_table_[species_type_index][pair.projector_pair_index]
                                                [angular_momentum][q_index] = table_value;
                    augmentation_q_radial_table_[species_type_index]
                                                [pair.reverse_projector_pair_index]
                                                [angular_momentum][q_index] = table_value;
                }
            }
        }

        // Store l=0 table as the total Q function table
        for (int beta_index_i = 0; beta_index_i < number_of_projectors; beta_index_i++) {
            for (int beta_index_j = beta_index_i; beta_index_j < number_of_projectors;
                 beta_index_j++) {
                int projector_pair_index = beta_index_i * number_of_projectors + beta_index_j;
                int reverse_projector_pair_index =
                    beta_index_j * number_of_projectors + beta_index_i;

                if (!augmentation_q_radial_table_[species_type_index][projector_pair_index]
                         .empty() &&
                    !augmentation_q_radial_table_[species_type_index][projector_pair_index][0]
                         .empty()) {
                    q_function_table_[species_type_index][projector_pair_index] =
                        augmentation_q_radial_table_[species_type_index][projector_pair_index][0];
                    q_function_table_[species_type_index][reverse_projector_pair_index] =
                        augmentation_q_radial_table_[species_type_index][projector_pair_index][0];
                } else {
                    q_function_table_[species_type_index][projector_pair_index].assign(
                        number_of_q_points_, 0.0);
                    q_function_table_[species_type_index][reverse_projector_pair_index].assign(
                        number_of_q_points_, 0.0);
                }
            }
        }
    }

    /// Q radial tables: [species][projector_pair][angular_momentum][q_index]
    std::vector<std::vector<std::vector<std::vector<Real>>>> augmentation_q_radial_table_;

    /// Total Q function tables (L=0): [species][projector_pair][q_index]
    std::vector<std::vector<std::vector<Real>>> q_function_table_;

    int maximum_angular_momentum_q_{0};
    int number_of_q_points_{};
    Real q_spacing_{};
    Real cell_volume_{};
};

} // namespace q3
