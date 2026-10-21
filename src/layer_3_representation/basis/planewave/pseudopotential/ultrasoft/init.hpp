#pragma once

/**
 * UltrasoftPlanewaveBasis — ultrasoft augmentation data in the plane-wave basis.
 *
 * Extends NormConservingPlanewaveBasis with static augmentation data:
 * - AugmentationQTable (tabulated Q_ij(q) radial functions)
 * - SphericalHarmonicsTable (precomputed Y_lm for G-vectors)
 * - GauntCoefficientTable (Gaunt coefficients for L-resolved augmentation)
 *
 * Also provides augmentation operations that only depend on static data:
 * - add_augmentation_charges()
 * - compute_q_g_single()
 * - interpolate_qfunc()
 * - augmentation_overlap_integral()
 *
 * The effective D-matrix (dynamic, recomputed each SCF step) lives in the
 * SCF iterator, not here.
 */

#include <algorithm>
#include <cmath>
#include <complex>
#include <stdexcept>
#include <string>
#include <vector>

#include "layer_0_core/logging/init.hpp"
#include "layer_0_core/math/phase_factor.hpp"
#include "layer_0_core/math/spherical/coefficients.hpp"
#include "layer_0_core/math/spherical/harmonics.hpp"
#include "layer_0_core/types/init.hpp"
#include "layer_0_core/types/matrix.hpp"
#include "layer_1_structure/crystal/crystal.hpp"
#include "layer_3_representation/basis/planewave/internal/spherical_harmonics_table.hpp"
#include "layer_3_representation/basis/planewave/pseudopotential/init.hpp"
#include "layer_3_representation/pseudopotential/planewave/ultrasoft/internal/charge_density_augmentation.hpp"

namespace q3 {

class UltrasoftPlanewaveBasis : public NormConservingPlanewaveBasis {
  public:
    UltrasoftPlanewaveBasis() = default;
    ~UltrasoftPlanewaveBasis() override = default;

    void init(const Crystal& crystal_structure,
              const PlanewaveBasis& planewave_basis,
              const PseudopotentialSet& pseudopotential_set) override {
        NormConservingPlanewaveBasis::init(crystal_structure, planewave_basis, pseudopotential_set);
        init_ultrasoft(crystal_structure, planewave_basis);
    }

    [[nodiscard]] bool is_ultrasoft() const override {
        return true;
    }

    // ─── Augmentation table access ──────────────────────────────────

    /**
     * Access the augmentation Q-function table.
     */
    [[nodiscard]] const AugmentationQTable& augmentation_q_table() const {
        return augmentation_q_table_;
    }

    /**
     * Access the precomputed spherical harmonics table.
     */
    [[nodiscard]] const SphericalHarmonicsTable& spherical_harmonics_table() const {
        return spherical_harmonics_table_;
    }

    /**
     * Access the Gaunt coefficient table.
     */
    [[nodiscard]] const GauntCoefficientTable& gaunt_table() const {
        return gaunt_table_;
    }

    // ─── Augmentation operations ────────────────────────────────────

    /**
     * Interpolate augmentation function Q_{ij}(|q|) from the radial table.
     * @param species_type_index    Species index
     * @param projector_pair_index  Linearized (i,j) pair index for the Q function
     * @param q_value               Wavevector magnitude in atomic units
     */
    [[nodiscard]] Real
    interpolate_qfunc(int species_type_index, int projector_pair_index, Real q_value) const {
        return augmentation_q_table_.interpolate_qfunc(
            species_type_index, projector_pair_index, q_value);
    }

    /**
     * Augmentation overlap integral q_{ij} = ∫ Q_ij(r) dr, or zero when the
     * species carries no augmentation charges.
     *
     * The gate belongs here, not in each caller. This class is instantiated
     * whenever *any* species needs augmentation, so a mixed cell — one
     * ultrasoft or PAW species alongside one norm-conserving species — brings
     * its norm-conserving species through every method on it. `as_ultrasoft`
     * is a `dynamic_cast` whose null check is an `assert`, so asking a
     * norm-conserving species for its q_ij aborts a Debug build and
     * dereferences null in the Release builds users actually run.
     *
     * q_ij = 0 is the physically correct answer, not merely a safe one: a
     * norm-conserving dataset has no augmentation charges, so it contributes
     * nothing to S = 1 + Σ q_ij |β_i⟩⟨β_j|. Its projectors still participate
     * in the nonlocal potential; only the augmentation term vanishes.
     *
     * Callers used to carry this test themselves and mostly did not — the
     * whole-system `has_ultrasoft()` flag reads as if it settled the question
     * per species, and it does not. `hubbard_augmentation_overlap` and the
     * force kernel in `forces/.../nonlocal.hpp` grew their own copies for this
     * exact reason; a mixed US+NC cell still aborted in the wavefunction
     * initializer's overlap, which had none.
     *
     * @param species_type  Species index
     * @param projector_a   Beta projector index i (radial)
     * @param projector_b   Beta projector index j (radial)
     */
    [[nodiscard]] Real
    augmentation_overlap_integral(int species_type, int projector_a, int projector_b) const {
        if (!pseudopotential_set_[species_type].has_augmentation_charges()) {
            return 0.0;
        }
        return pseudopotential_set_.as_ultrasoft(species_type)
            .augmentation_overlap_integrals[projector_a][projector_b];
    }

    /**
     * Access a pre-computed Q_ij(G) value.
     *
     * @param species_type_index    Species index
     * @param channel_pair_index    Linearized channel pair: channel_i * n_channels + channel_j
     * @param g_index               G-vector index
     * @param number_of_g_vectors   Total number of G-vectors (stride for flat indexing)
     * @return The pre-computed Q_ij(G) complex value
     */
    [[nodiscard]] Complex q_function_g_value(int species_type_index,
                                             int channel_pair_index,
                                             int g_index,
                                             int number_of_g_vectors) const {
        return q_functions_g_[species_type_index]
                             [channel_pair_index * number_of_g_vectors + g_index];
    }

    /**
     * Access the raw Q(G) data pointer for a given species and channel pair.
     *
     * Returns a pointer to number_of_g_vectors contiguous Complex values.
     * Used for BLAS-accelerated dot products in newd and addusdens.
     */
    [[nodiscard]] const Complex* q_function_g_data(int species_type_index,
                                                   int channel_pair_index,
                                                   int number_of_g_vectors) const {
        return &q_functions_g_[species_type_index][channel_pair_index * number_of_g_vectors];
    }

    /**
     * Check whether Q(G) table is available for a species.
     */
    [[nodiscard]] bool has_q_functions_g(int species_type_index) const {
        return species_type_index < static_cast<int>(q_functions_g_.size()) &&
               !q_functions_g_[species_type_index].empty();
    }

    /**
     * Add ultrasoft augmentation charges to the charge density in G-space.
     *
     * Uses the pre-computed Q_ij(G) table for O(atoms × pairs × G) cost,
     * eliminating the per-G interpolation that dominated the previous implementation.
     *
     * @param crystal_structure  Crystal geometry
     * @param projector_overlap_coefficients  ⟨ψ|β⟩⟨β|ψ⟩ summed over bands
     * @param charge_density_g  Charge density in G-space (modified in place)
     */
    void add_augmentation_charges(
        const Crystal& crystal_structure,
        const Matrix<Real>& projector_overlap_coefficients,
        std::vector<Complex>& charge_density_g,
        const std::vector<std::vector<Complex>>* cached_structure_factors = nullptr) const {

        int number_of_g_vectors = planewave_basis_ptr_->number_of_g_vectors();

        for (int atom_index = 0; atom_index < crystal_structure.number_of_atoms(); atom_index++) {
            int species_type_index = crystal_structure.species_type_index(atom_index);

            /**
             * Skip species that have no Q(G) table, exactly as the augmentation
             * force kernel in `forces/.../ultrasoft/internal/augmentation.hpp`
             * already does — the two assemble the same A_I(G) and must agree on
             * which atoms contribute to it.
             *
             * In a mixed cell `precompute_q_functions_g` leaves the entry for a
             * norm-conserving species empty, but ρ_ij is accumulated over
             * *every* atom, so its ⟨ψ|β_i⟩⟨β_j|ψ⟩ are nonzero and this loop
             * reached `q_function_g_data` for it anyway. That takes the address
             * of an element of an empty vector and then reads
             * `number_of_g_vectors` complex values through it — out of bounds on
             * every SCF iteration, silently corrupting the density rather than
             * failing outright. A norm-conserving species has no augmentation
             * charge to add, so contributing nothing is also the right answer.
             */
            if (!has_q_functions_g(species_type_index)) {
                continue;
            }

            int number_of_projector_channels =
                projector_mapping_table_.number_of_channels(species_type_index);

            /**
             * Accumulate ρ_ij-weighted Q_ij(G) over all channel pairs.
             * Exploit Q_ij(G) = Q_ji(G) symmetry: loop only over upper triangle.
             * For i < j: use (ρ_ij + ρ_ji) * Q_ij(G) in a single pass.
             * For i == j: use ρ_ii * Q_ii(G) directly.
             */
            std::vector<Complex> augmentation_g(number_of_g_vectors, Complex(0.0, 0.0));

            for (int channel_index_i = 0; channel_index_i < number_of_projector_channels;
                 ++channel_index_i) {
                // Diagonal: i == j
                {
                    int channel_pair_index =
                        channel_index_i * number_of_projector_channels + channel_index_i;
                    Real diagonal_projector_overlap_coefficient =
                        projector_overlap_coefficients(atom_index, channel_pair_index);
                    if (std::abs(diagonal_projector_overlap_coefficient) >= 1e-15) {
                        const Complex* q_g_data = q_function_g_data(
                            species_type_index, channel_pair_index, number_of_g_vectors);
                        for (int g_index = 0; g_index < number_of_g_vectors; ++g_index) {
                            augmentation_g[g_index] +=
                                diagonal_projector_overlap_coefficient * q_g_data[g_index];
                        }
                    }
                }

                // Off-diagonal: j > i, use Q_ij = Q_ji symmetry
                for (int channel_index_j = channel_index_i + 1;
                     channel_index_j < number_of_projector_channels;
                     ++channel_index_j) {
                    int pair_ij = channel_index_i * number_of_projector_channels + channel_index_j;
                    int pair_ji = channel_index_j * number_of_projector_channels + channel_index_i;
                    Real combined_projector_overlap_coefficient =
                        projector_overlap_coefficients(atom_index, pair_ij) +
                        projector_overlap_coefficients(atom_index, pair_ji);
                    if (std::abs(combined_projector_overlap_coefficient) < 1e-15)
                        continue;

                    const Complex* q_g_data =
                        q_function_g_data(species_type_index, pair_ij, number_of_g_vectors);
                    for (int g_index = 0; g_index < number_of_g_vectors; ++g_index) {
                        augmentation_g[g_index] +=
                            combined_projector_overlap_coefficient * q_g_data[g_index];
                    }
                }
            }

            // Multiply by structure factor and add to charge density
            if (cached_structure_factors != nullptr) {
                const auto& structure_factor = (*cached_structure_factors)[atom_index];
                for (int g_index = 0; g_index < number_of_g_vectors; ++g_index) {
                    charge_density_g[g_index] +=
                        augmentation_g[g_index] * structure_factor[g_index];
                }
            } else {
                Vector3D atomic_position =
                    crystal_structure.convert_from_crystal_to_cartesian(atom_index);
                for (int g_index = 0; g_index < number_of_g_vectors; ++g_index) {
                    const auto& g_vector = planewave_basis_ptr_->g_vectors()[g_index];
                    Real phase_angle = constants::two_pi * g_vector.g_cart.dot(atomic_position);
                    charge_density_g[g_index] +=
                        augmentation_g[g_index] * compute_phase_factor(phase_angle);
                }
            }
        }
    }

    /**
     * Compute the augmentation Q-function Q_ij(G) for a single G-vector.
     * @param species_type_index  Species index
     * @param projector_index_i   Projector channel index i
     * @param projector_index_j   Projector channel index j
     * @param g_cartesian         G-vector in Cartesian reciprocal units (2π/a)
     * @param two_pi_over_lattice_parameter    Conversion factor 2π/alat
     * @return Complex Q_ij(G) value
     */
    [[nodiscard]] Complex compute_q_g_single(int species_type_index,
                                             int projector_index_i,
                                             int projector_index_j,
                                             const Vector3D& g_cartesian,
                                             Real two_pi_over_lattice_parameter) const {
        /**
         * Same per-species gate as augmentation_overlap_integral: Q_ij(G) is
         * identically zero for a species with no augmentation charges, and
         * reaching for its (nonexistent) table is the Release null dereference.
         */
        if (!pseudopotential_set_[species_type_index].has_augmentation_charges()) {
            return 0.0;
        }

        Real g_norm = g_cartesian.norm() * two_pi_over_lattice_parameter;
        if (g_norm < math::tolerances::zero_vector_magnitude_threshold) {
            if (projector_mapping_table_.combined_lm_index(species_type_index, projector_index_i) ==
                projector_mapping_table_.combined_lm_index(species_type_index, projector_index_j)) {
                int beta_index_i =
                    projector_mapping_table_.to_beta_index(species_type_index, projector_index_i);
                int beta_index_j =
                    projector_mapping_table_.to_beta_index(species_type_index, projector_index_j);
                return pseudopotential_set_.as_ultrasoft(species_type_index)
                    .augmentation_overlap_integrals[beta_index_i][beta_index_j];
            }
            return 0.0;
        }

        int beta_index_i =
            projector_mapping_table_.to_beta_index(species_type_index, projector_index_i);
        int beta_index_j =
            projector_mapping_table_.to_beta_index(species_type_index, projector_index_j);
        int combined_lm_index_i =
            projector_mapping_table_.combined_lm_index(species_type_index, projector_index_i);
        int combined_lm_index_j =
            projector_mapping_table_.combined_lm_index(species_type_index, projector_index_j);
        int angular_momentum_i =
            projector_mapping_table_.angular_momentum(species_type_index, projector_index_i);
        int angular_momentum_j =
            projector_mapping_table_.angular_momentum(species_type_index, projector_index_j);
        int number_of_projectors = pseudopotential_set_[species_type_index].number_of_projectors;
        int projector_pair_index = beta_index_i * number_of_projectors + beta_index_j;

        int maximum_angular_momentum_q_plus_one =
            augmentation_q_table_.maximum_angular_momentum_q() + 1;
        int number_of_lm_projectors =
            maximum_angular_momentum_q_plus_one * maximum_angular_momentum_q_plus_one;
        if (combined_lm_index_i >= number_of_lm_projectors ||
            combined_lm_index_j >= number_of_lm_projectors)
            return 0.0;
        if (gaunt_table_
                .number_of_nonzero_product_indices[combined_lm_index_i][combined_lm_index_j] == 0)
            return 0.0;

        Complex sum_integral(0.0, 0.0);
        for (int gaunt_index = 0;
             gaunt_index < gaunt_table_.number_of_nonzero_product_indices[combined_lm_index_i]
                                                                         [combined_lm_index_j];
             gaunt_index++) {
            int product_lm_index =
                gaunt_table_
                    .nonzero_product_indices[combined_lm_index_i][combined_lm_index_j][gaunt_index];
            Real gaunt_coefficient_value =
                gaunt_table_
                    .coefficients[product_lm_index][combined_lm_index_i][combined_lm_index_j];
            int angular_momentum =
                static_cast<int>(std::sqrt(static_cast<double>(product_lm_index)));
            if (angular_momentum < std::abs(angular_momentum_i - angular_momentum_j) ||
                angular_momentum > angular_momentum_i + angular_momentum_j ||
                (angular_momentum + angular_momentum_i + angular_momentum_j) % 2 != 0)
                continue;

            Real qr_val = augmentation_q_table_.interpolate_qrad(
                species_type_index, projector_pair_index, angular_momentum, g_norm);
            int offset_in_shell = product_lm_index - angular_momentum * angular_momentum;
            int magnetic_projection = (offset_in_shell == 0)
                                          ? 0
                                          : ((offset_in_shell % 2 == 1) ? (offset_in_shell + 1) / 2
                                                                        : -(offset_in_shell / 2));
            Real spherical_harmonic_value =
                compute_real_spherical_harmonic(angular_momentum, magnetic_projection, g_cartesian);
            Real integrand_contribution = qr_val * spherical_harmonic_value;

            switch (angular_momentum % 4) {
                case 0:
                    sum_integral += gaunt_coefficient_value * integrand_contribution;
                    break;
                case 1:
                    sum_integral += Complex(0.0, -gaunt_coefficient_value * integrand_contribution);
                    break;
                case 2:
                    sum_integral -= gaunt_coefficient_value * integrand_contribution;
                    break;
                case 3:
                    sum_integral += Complex(0.0, gaunt_coefficient_value * integrand_contribution);
                    break;
            }
        }
        return sum_integral;
    }

  protected:
    void init_ultrasoft(const Crystal& crystal_structure, const PlanewaveBasis& planewave_basis) {
        cell_volume_ = crystal_structure.cell_volume();

        augmentation_q_table_.init(pseudopotential_set_,
                                   mesh_sizes_,
                                   beta_function_table_->number_of_q_points(),
                                   beta_function_table_->q_spacing(),
                                   cell_volume_);

        spherical_harmonics_table_.init(
            crystal_structure, planewave_basis, augmentation_q_table_.maximum_angular_momentum_q());

        log::info("     Ultrasoft Q functions tabulated (L-resolved, lmaxq={})",
                  augmentation_q_table_.maximum_angular_momentum_q());

        gaunt_table_ = generate_gaunt_coefficients(projector_degree_for_gaunt_table(
            pseudopotential_set_, augmentation_q_table_.maximum_angular_momentum_q()));

        precompute_q_functions_g(planewave_basis);
    }

    /**
     * The projector degree the Gaunt table is to be built for, checked
     * against the product degree its consumers will index.
     *
     * `generate_gaunt_coefficients` expands products of *projector*
     * harmonics, so its argument is the projector degree and it evaluates
     * harmonics up to twice that. What used to be passed instead was
     * `maximum_angular_momentum_q()`, which is the maximum L of the Q
     * expansion and is already twice the projector degree -- so the table was
     * sized for a product degree of 8 on any d-channel dataset, its
     * collocation asked for harmonics at degrees 5 through 8, and
     * `compute_real_spherical_harmonic` answered every one of them with 0.0.
     *
     * Nothing observed it: the damaged entries are the ones no consumer
     * indexes. Consumers reach product degree `lmaxq` and no further, and
     * every entry at or below that sum was correct. The table was simply
     * about nine times larger than anything reads, with the excess wrong.
     *
     * Passing the projector degree makes the collocation exact over exactly
     * the range consumers use, and -- because it no longer doubles an already
     * doubled degree -- lets the harmonic refusal above degree 4 be added
     * without rejecting every ultrasoft and PAW run.
     *
     * The relation `lmaxq <= 2 * projector_degree` is what makes the smaller
     * table still cover every index a consumer forms. It holds for every
     * dataset Q3 reads, because `lmaxq` is derived from the projector
     * channels; a dataset that declared otherwise would index past the end of
     * the table, so it is checked rather than assumed.
     */
    [[nodiscard]] static int
    projector_degree_for_gaunt_table(const PseudopotentialSet& pseudopotential_set,
                                     int maximum_angular_momentum_q) {
        int projector_degree = 0;
        for (int species_type_index = 0; species_type_index < pseudopotential_set.size();
             species_type_index++) {
            projector_degree = std::max(
                projector_degree, pseudopotential_set[species_type_index].maximum_angular_momentum);
        }
        if (maximum_angular_momentum_q > 2 * projector_degree) {
            throw std::invalid_argument(
                "Gaunt table: the augmentation charges declare angular momenta up to L = " +
                std::to_string(maximum_angular_momentum_q) +
                ", above the 2 * l_max = " + std::to_string(2 * projector_degree) +
                " that the projector channels of this dataset can produce. The table is "
                "built from the projector degree, so such an expansion would be indexed "
                "past its end.");
        }
        return projector_degree;
    }

    /**
     * Pre-compute Q_ij(G) for all G-vectors and all channel pairs per species.
     *
     * For each species s with N_ch channels and each pair (i,j):
     *   Q_ij(G) = Σ_{lm} gaunt(lm, lm_i, lm_j) * qrad(s, beta_pair, l, |G|)
     *             * Ylm(Ĝ) * (-i)^l
     *
     * Stored as q_functions_g_[species][pair_linear_index][g_index]
     * where pair_linear_index = channel_i * n_channels + channel_j.
     *
     * Memory: n_species × n_channels² × n_g_vectors × sizeof(Complex)
     */
    void precompute_q_functions_g(const PlanewaveBasis& planewave_basis) {
        int number_of_species = static_cast<int>(pseudopotential_set_.size());
        int number_of_g_vectors = planewave_basis.number_of_g_vectors();
        const auto& g_vector_magnitudes = planewave_basis.g_vector_magnitudes();

        int maximum_angular_momentum_q_plus_one =
            augmentation_q_table_.maximum_angular_momentum_q() + 1;
        int number_of_lm_projectors =
            maximum_angular_momentum_q_plus_one * maximum_angular_momentum_q_plus_one;

        q_functions_g_.resize(number_of_species);

        for (int species_type_index = 0; species_type_index < number_of_species;
             ++species_type_index) {
            /**
             * Capability, not type — a PAW species has Q_ij^L too (synthesized
             * for datasets that ship only the augmentation charges) and needs the
             * same Q(G) table.
             */
            if (!pseudopotential_set_[species_type_index].has_augmentation_charges()) {
                continue;
            }

            int number_of_projector_channels =
                projector_mapping_table_.number_of_channels(species_type_index);
            int number_of_beta_functions =
                pseudopotential_set_[species_type_index].number_of_projectors;
            int total_pairs = number_of_projector_channels * number_of_projector_channels;

            q_functions_g_[species_type_index].assign(total_pairs * number_of_g_vectors,
                                                      Complex(0.0, 0.0));

            // Pre-compute valid Gaunt terms for each channel pair
            struct GauntTermForPair {
                int product_lm_index;
                int angular_momentum;
                int projector_pair_index;
                Real gaunt_coefficient_value;
            };

            struct ChannelPairInfo {
                int linear_pair_index;
                std::vector<GauntTermForPair> gaunt_terms;
            };

            std::vector<ChannelPairInfo> active_pairs;

            for (int channel_index_i = 0; channel_index_i < number_of_projector_channels;
                 ++channel_index_i) {
                for (int channel_index_j = 0; channel_index_j < number_of_projector_channels;
                     ++channel_index_j) {
                    int combined_lm_index_i = projector_mapping_table_.combined_lm_index(
                        species_type_index, channel_index_i);
                    int combined_lm_index_j = projector_mapping_table_.combined_lm_index(
                        species_type_index, channel_index_j);
                    int angular_momentum_i = projector_mapping_table_.angular_momentum(
                        species_type_index, channel_index_i);
                    int angular_momentum_j = projector_mapping_table_.angular_momentum(
                        species_type_index, channel_index_j);
                    int beta_index_i =
                        projector_mapping_table_.to_beta_index(species_type_index, channel_index_i);
                    int beta_index_j =
                        projector_mapping_table_.to_beta_index(species_type_index, channel_index_j);
                    int projector_pair_index =
                        beta_index_i * number_of_beta_functions + beta_index_j;

                    if (combined_lm_index_i >= number_of_lm_projectors ||
                        combined_lm_index_j >= number_of_lm_projectors) {
                        continue;
                    }
                    if (gaunt_table_.number_of_nonzero_product_indices[combined_lm_index_i]
                                                                      [combined_lm_index_j] == 0) {
                        continue;
                    }

                    std::vector<GauntTermForPair> terms;
                    for (int gaunt_index = 0;
                         gaunt_index <
                         gaunt_table_.number_of_nonzero_product_indices[combined_lm_index_i]
                                                                       [combined_lm_index_j];
                         ++gaunt_index) {
                        int product_lm_index =
                            gaunt_table_.nonzero_product_indices[combined_lm_index_i]
                                                                [combined_lm_index_j][gaunt_index];
                        Real gaunt_coefficient_value =
                            gaunt_table_.coefficients[product_lm_index][combined_lm_index_i]
                                                     [combined_lm_index_j];
                        int angular_momentum =
                            static_cast<int>(std::sqrt(static_cast<double>(product_lm_index)));
                        if (angular_momentum < std::abs(angular_momentum_i - angular_momentum_j) ||
                            angular_momentum > angular_momentum_i + angular_momentum_j ||
                            (angular_momentum + angular_momentum_i + angular_momentum_j) % 2 != 0) {
                            continue;
                        }
                        terms.push_back({product_lm_index,
                                         angular_momentum,
                                         projector_pair_index,
                                         gaunt_coefficient_value});
                    }

                    if (!terms.empty()) {
                        int linear_pair_index =
                            channel_index_i * number_of_projector_channels + channel_index_j;
                        active_pairs.push_back({linear_pair_index, std::move(terms)});
                    }
                }
            }

            // Fill the Q(G) table for all G-vectors and active channel pairs
            for (int g_index = 0; g_index < number_of_g_vectors; ++g_index) {
                Real g_magnitude = g_vector_magnitudes[g_index];

                for (const auto& pair : active_pairs) {
                    Real q_g_real = 0.0;
                    Real q_g_imag = 0.0;

                    for (const auto& term : pair.gaunt_terms) {
                        Real qr_val =
                            augmentation_q_table_.interpolate_qrad(species_type_index,
                                                                   term.projector_pair_index,
                                                                   term.angular_momentum,
                                                                   g_magnitude);
                        Real spherical_harmonic_value =
                            spherical_harmonics_table_.value(g_index, term.product_lm_index);
                        Real integrand_contribution = qr_val * spherical_harmonic_value;

                        switch (term.angular_momentum % 4) {
                            case 0:
                                q_g_real += term.gaunt_coefficient_value * integrand_contribution;
                                break;
                            case 1:
                                q_g_imag -= term.gaunt_coefficient_value * integrand_contribution;
                                break;
                            case 2:
                                q_g_real -= term.gaunt_coefficient_value * integrand_contribution;
                                break;
                            case 3:
                                q_g_imag += term.gaunt_coefficient_value * integrand_contribution;
                                break;
                        }
                    }

                    q_functions_g_[species_type_index]
                                  [pair.linear_pair_index * number_of_g_vectors + g_index] =
                                      Complex(q_g_real, q_g_imag);
                }
            }

            log::info(
                "     Pre-computed Q(G) table for species {} ({} channel pairs, {} G-vectors)",
                species_type_index,
                static_cast<int>(active_pairs.size()),
                number_of_g_vectors);
        }
    }

  private:
    AugmentationQTable augmentation_q_table_;
    SphericalHarmonicsTable spherical_harmonics_table_;
    GauntCoefficientTable gaunt_table_;

    /**
     * Pre-computed Q_ij(G) table.
     *
     * Indexed as: q_functions_g_[species_type][pair_index * n_g + g_index]
     * where pair_index = channel_i * n_channels + channel_j.
     * Flat layout for cache efficiency during G-vector inner loops.
     */
    std::vector<std::vector<Complex>> q_functions_g_;
};

} // namespace q3
