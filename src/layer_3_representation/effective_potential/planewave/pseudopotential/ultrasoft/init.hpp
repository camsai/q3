#pragma once

/**
 * UltrasoftPlanewaveEffectivePotential — non-local effective potential
 * for ultrasoft pseudopotentials (the D_eff matrix).
 *
 * D_eff_ij = D_ion_ij + ∫ conj(V_eff(G)) * Q_ij(G) * e^{-iG·τ} dG
 *
 * This is the non-local counterpart to PlanewaveEffectivePotential (local V_eff).
 * It is recomputed each SCF step from the current V_eff via the pre-computed Q(G) table.
 */

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "layer_0_core/math/linear_algebra/blas.hpp"
#include "layer_0_core/math/phase_factor.hpp"
#include "layer_0_core/types/init.hpp"
#include "layer_1_structure/crystal/crystal.hpp"
#include "layer_3_representation/basis/planewave/init.hpp"
#include "layer_3_representation/basis/planewave/pseudopotential/ultrasoft/init.hpp"
#include "layer_3_representation/effective_potential/planewave/pseudopotential/init.hpp"
#include "layer_3_representation/pseudopotential/set.hpp"

namespace q3 {
using namespace q3::blas;

class UltrasoftPlanewaveEffectivePotential : public NormConservingPlanewaveEffectivePotential {
  public:
    UltrasoftPlanewaveEffectivePotential() = default;
    using NormConservingPlanewaveEffectivePotential::NormConservingPlanewaveEffectivePotential;

    /**
     * Compute the effective D-matrix for all atoms from the current V_eff(G).
     *
     * Uses the pre-computed Q_ij(G) table from the ultrasoft basis to avoid
     * per-G interpolation. The computation reduces to:
     *   D_eff(i,j) = D_ion(i,j) + Re(Σ_G conj(V(G)) * Q_ij(G) * e^{-iG·τ}) * Ω
     *
     * @param crystal_structure         Crystal geometry
     * @param planewave_basis           Plane-wave basis (G-vectors)
     * @param ultrasoft_pseudopotential_basis  Ultrasoft basis (pre-computed Q(G), projector
     * mappings)
     * @param total_potential_g         Total effective potential in G-space
     */
    /**
     * Compute D for every spin channel.
     *
     * D_ij = D_ion + ∫v_eff·Q_ij, and v_eff is spin-dependent, so D is too.
     * Calling this with one potential reproduces the unpolarized result
     * exactly; calling it with two gives each channel its own D, which is what
     * a magnetic system needs and what a shared, spin-averaged D silently
     * denies it.
     */
    void compute_by_spin(const Crystal& crystal_structure,
                         const PlanewaveBasis& planewave_basis,
                         const UltrasoftPlanewaveBasis& ultrasoft_pseudopotential_basis,
                         const std::vector<std::vector<Complex>>& total_potential_g_by_spin) {
        int number_of_spins = static_cast<int>(total_potential_g_by_spin.size());
        effective_d_matrix_by_spin_.resize(number_of_spins);
        for (int spin = 0; spin < number_of_spins; spin++) {
            compute(crystal_structure,
                    planewave_basis,
                    ultrasoft_pseudopotential_basis,
                    total_potential_g_by_spin[spin]);
            effective_d_matrix_by_spin_[spin] = effective_d_matrix_;
        }
        /**
         * Leave `effective_d_matrix_` holding channel 0 so anything still
         * reading the spin-free accessor sees a real D rather than whatever the
         * last loop iteration left behind.
         */
        if (number_of_spins > 0)
            effective_d_matrix_ = effective_d_matrix_by_spin_[0];
    }

    /**
     * Add per-atom terms to one spin channel's D (PAW's one-center term).
     *
     * Refuses a channel that was never computed rather than returning quietly.
     * A silent no-op would leave that channel's Hamiltonian carrying the other
     * channel's one-center D — via `element()`'s fallback — while the
     * double-counting term still subtracted its own, so the SCF would converge
     * to a stationary point of something that is not the energy it prints.
     */
    void add_per_atom_d_terms_for_spin(int spin, const std::vector<std::vector<Real>>& terms) {
        if (spin >= static_cast<int>(effective_d_matrix_by_spin_.size()))
            throw std::runtime_error(
                "one-center D terms were computed for more spin channels than the nonlocal "
                "effective potential holds; the two must be built from the same spin count");
        std::swap(effective_d_matrix_, effective_d_matrix_by_spin_[spin]);
        add_per_atom_d_terms(terms);
        std::swap(effective_d_matrix_, effective_d_matrix_by_spin_[spin]);
        if (spin == 0)
            effective_d_matrix_ = effective_d_matrix_by_spin_[0];
    }

    void compute(const Crystal& crystal_structure,
                 const PlanewaveBasis& planewave_basis,
                 const UltrasoftPlanewaveBasis& ultrasoft_pseudopotential_basis,
                 const std::vector<Complex>& total_potential_g) {
        int number_of_atoms = crystal_structure.number_of_atoms();
        int number_of_g_vectors = planewave_basis.number_of_g_vectors();
        Real cell_volume = crystal_structure.cell_volume();
        const auto& pseudopotential_set = ultrasoft_pseudopotential_basis.pseudopotential_set();
        const auto& projector_mapping = ultrasoft_pseudopotential_basis.projector_mapping_table();
        int number_of_species = static_cast<int>(pseudopotential_set.size());

        /**
         * Phase 1: For each US species, precompute aux_ij(G) = conj(V(G)) * Q_ij(G)
         * This is done once per species, not once per atom.
         *
         * Correct formula: D_ij = ∫ Q(r−R)·V(r) dr = Σ_G Q(G)·conj(V(G))·e^{−iGR}·Ω
         * [For real V: V(−G)=V*(G), so the integral picks up V*(G).]
         * Taking conj(Q) instead of conj(V) gives the same result only when
         * Im(V(G)) = 0 — true for symmetric structures but NOT for displaced atoms.
         */
        struct SpeciesAuxData {
            int number_of_projectors;
            // aux_data[pair_index * n_g + g_index] for upper-triangle pairs
            std::vector<Complex> v_conj_times_q;
            // Linear pair indices for the upper triangle
            std::vector<std::pair<int, int>> upper_triangle_pairs;
        };
        std::vector<SpeciesAuxData> species_aux(number_of_species);

        for (int species_index = 0; species_index < number_of_species; ++species_index) {
            int number_of_projectors = projector_mapping.number_of_channels(species_index);
            species_aux[species_index].number_of_projectors = number_of_projectors;

            /**
             * A species contributes ∫v_eff·Q_ij here exactly when it has
             * augmentation charges — which PAW does. Skipping on `!=
             * ultrasoft` would leave a PAW atom's D at D_ion alone, a large
             * and entirely silent error.
             */
            if (!pseudopotential_set[species_index].has_augmentation_charges()) {
                continue;
            }

            auto& aux = species_aux[species_index];
            // Collect upper-triangle pairs
            for (int projector_index_i = 0; projector_index_i < number_of_projectors;
                 ++projector_index_i) {
                for (int projector_index_j = projector_index_i;
                     projector_index_j < number_of_projectors;
                     ++projector_index_j) {
                    aux.upper_triangle_pairs.emplace_back(projector_index_i, projector_index_j);
                }
            }

            int number_of_pairs = static_cast<int>(aux.upper_triangle_pairs.size());
            aux.v_conj_times_q.resize(static_cast<size_t>(number_of_pairs) * number_of_g_vectors);

            for (int pair_index = 0; pair_index < number_of_pairs; ++pair_index) {
                auto [proj_i, proj_j] = aux.upper_triangle_pairs[pair_index];
                int channel_pair_index = proj_i * number_of_projectors + proj_j;
                const Complex* q_g_data = ultrasoft_pseudopotential_basis.q_function_g_data(
                    species_index, channel_pair_index, number_of_g_vectors);

                Complex* dest = aux.v_conj_times_q.data() +
                                static_cast<size_t>(pair_index) * number_of_g_vectors;
                for (int g_index = 0; g_index < number_of_g_vectors; ++g_index) {
                    /**
                     * D_ij = ∫ Q(r−R)·V(r) dr = Σ_G Q(G)·V*(G)·e^{−iGR}·Ω
                     * Must conjugate V, NOT Q (Q may also be complex due to (−i)^l factor).
                     */
                    dest[g_index] = std::conj(total_potential_g[g_index]) * q_g_data[g_index];
                }
            }
        }

        /**
         * Phase 2: Compute D_eff for all atoms using batched BLAS zgemm per species.
         * Instead of 54 individual zgemv calls, we group atoms by species and
         * compute all integrals in a single zgemm: result = aux^T × sf_matrix.
         * aux is (n_g × n_pairs), sf_matrix is (n_g × n_atoms_of_species),
         * result is (n_pairs × n_atoms_of_species).
         */
        effective_d_matrix_.resize(number_of_atoms);

        // First pass: initialize D-ion for all atoms
        for (int atom_index = 0; atom_index < number_of_atoms; ++atom_index) {
            int species_type_index = crystal_structure.species_type_index(atom_index);
            const auto& aux = species_aux[species_type_index];
            int number_of_projectors = aux.number_of_projectors;
            effective_d_matrix_[atom_index].assign(number_of_projectors * number_of_projectors,
                                                   0.0);

            for (int projector_index_i = 0; projector_index_i < number_of_projectors;
                 ++projector_index_i) {
                for (int projector_index_j = 0; projector_index_j < number_of_projectors;
                     ++projector_index_j) {
                    int combined_lm_i =
                        projector_mapping.combined_lm_index(species_type_index, projector_index_i);
                    int combined_lm_j =
                        projector_mapping.combined_lm_index(species_type_index, projector_index_j);
                    if (combined_lm_i == combined_lm_j) {
                        int projector_beta_index_i =
                            projector_mapping.to_beta_index(species_type_index, projector_index_i);
                        int projector_beta_index_j =
                            projector_mapping.to_beta_index(species_type_index, projector_index_j);
                        effective_d_matrix_[atom_index][projector_index_i * number_of_projectors +
                                                        projector_index_j] =
                            pseudopotential_set[species_type_index]
                                .d_ion_matrix[projector_beta_index_i][projector_beta_index_j];
                    }
                }
            }
        }

        // Cache structure factors if not already done
        if (cached_structure_factors_.empty()) {
            precompute_structure_factors(crystal_structure, planewave_basis);
        }

        // Second pass: batch zgemm per species
        for (int species_index = 0; species_index < number_of_species; ++species_index) {
            if (!pseudopotential_set[species_index].has_augmentation_charges()) {
                continue;
            }

            const auto& aux = species_aux[species_index];
            int number_of_pairs = static_cast<int>(aux.upper_triangle_pairs.size());
            int number_of_projectors = aux.number_of_projectors;

            // Collect atom indices for this species
            std::vector<int> species_atom_indices;
            for (int atom_index = 0; atom_index < number_of_atoms; ++atom_index) {
                if (crystal_structure.species_type_index(atom_index) == species_index) {
                    species_atom_indices.push_back(atom_index);
                }
            }
            int number_of_atoms_of_species = static_cast<int>(species_atom_indices.size());

            // Build contiguous structure factor matrix: (n_g × n_atoms_of_species) column-major
            std::vector<Complex> structure_factor_matrix(static_cast<size_t>(number_of_g_vectors) *
                                                         number_of_atoms_of_species);
            for (int local_atom = 0; local_atom < number_of_atoms_of_species; ++local_atom) {
                int atom_index = species_atom_indices[local_atom];
                const auto& structure_factor = cached_structure_factors_[atom_index];
                std::copy(structure_factor.begin(),
                          structure_factor.end(),
                          structure_factor_matrix.begin() +
                              static_cast<size_t>(local_atom) * number_of_g_vectors);
            }

            /**
             * zgemm: result = aux^T × sf_matrix
             * aux is (n_g × n_pairs) column-major, sf_matrix is (n_g × n_atoms) column-major
             * result = aux^T × sf = (n_pairs × n_g) × (n_g × n_atoms) = (n_pairs × n_atoms)
             */
            std::vector<Complex> all_integrals(static_cast<size_t>(number_of_pairs) *
                                               number_of_atoms_of_species);
            Complex blas_alpha(1.0, 0.0);
            Complex blas_beta(0.0, 0.0);
            zgemm_("T",
                   "N",
                   &number_of_pairs,
                   &number_of_atoms_of_species,
                   &number_of_g_vectors,
                   &blas_alpha,
                   aux.v_conj_times_q.data(),
                   &number_of_g_vectors,
                   structure_factor_matrix.data(),
                   &number_of_g_vectors,
                   &blas_beta,
                   all_integrals.data(),
                   &number_of_pairs);

            /**
             * Unpack results into per-atom D-matrices
             * For gamma-only: multiply by 2 (half-sphere → full sphere), then
             * subtract the G=0 contribution once (it was double-counted).
             */
            bool is_gamma_only = planewave_basis.is_gamma_only();
            Real gamma_scale = is_gamma_only ? 2.0 : 1.0;

            /**
             * If gamma-only, precompute the G=0 correction:
             * G=0 term = Re(conj(V(0))*Q_ij(0)*sf(0)) = Re(conj(V(0))*Q_ij(0)*1)
             * sf(G=0) = exp(i·0·τ) = 1 for all atoms.
             * At G=0, V(0) is real and Q(0) is real (l=0 only), so conj(V(0))=V(0).
             * This correction is the same for all atoms of this species.
             */
            std::vector<Real> g0_correction;
            if (is_gamma_only && planewave_basis.first_nonzero_g_index() == 1) {
                g0_correction.resize(number_of_pairs, 0.0);
                for (int pair_index = 0; pair_index < number_of_pairs; ++pair_index) {
                    // v_conj_times_q[pair * ngm + 0] = conj(V(0)) * Q(0) at G=0
                    Complex g0_val =
                        aux.v_conj_times_q[static_cast<size_t>(pair_index) * number_of_g_vectors];
                    g0_correction[pair_index] = std::real(g0_val) * cell_volume;
                }
            }

            for (int local_atom = 0; local_atom < number_of_atoms_of_species; ++local_atom) {
                int atom_index = species_atom_indices[local_atom];
                const Complex* integrals =
                    all_integrals.data() + static_cast<size_t>(local_atom) * number_of_pairs;
                for (int pair_index = 0; pair_index < number_of_pairs; ++pair_index) {
                    auto [proj_i, proj_j] = aux.upper_triangle_pairs[pair_index];
                    Real contribution =
                        std::real(integrals[pair_index]) * cell_volume * gamma_scale;
                    // Subtract the G=0 double-count for gamma-only
                    if (is_gamma_only && !g0_correction.empty()) {
                        contribution -= g0_correction[pair_index];
                    }
                    effective_d_matrix_[atom_index][proj_i * number_of_projectors + proj_j] +=
                        contribution;
                    if (proj_i != proj_j) {
                        effective_d_matrix_[atom_index][proj_j * number_of_projectors + proj_i] +=
                            contribution;
                    }
                }
            }
        }
    }

    /**
     * Add per-atom terms that a layered flavor contributes to D.
     *
     * PAW's D is this D plus a one-center term:
     *
     *     D_ij = D⁰_ij  +  ∫ v_eff Q_ij  +  (D¹ − D̃¹)_Hxc
     *            └─ dataset ─┘ └─ here ─┘    └─ passed in ─┘
     *
     * The first two are what compute() built and are never recomputed for
     * PAW — the kinetic and core-reference pieces baked into `PP_DIJ`/`DION`
     * at generation time would be double-counted if they were. So the
     * one-center term is a decoration of this result, and this is the seam it
     * arrives through: the correction depends on the on-site occupancies,
     * which only the SCF loop knows, so it is passed in rather than computed
     * here.
     *
     * Expressed as a method rather than a subclass override deliberately. The
     * iterators hold this type by value, so a `PawPlanewaveEffectivePotential`
     * could not actually be substituted for it without pointer-ifying every
     * holder — a subclass that cannot be substituted is worse than a named
     * operation on the object that owns D.
     *
     * Call after compute(), once per SCF step. An empty entry means "no terms
     * for this atom", so a mixed PAW/ultrasoft cell needs no special case.
     */
    void add_per_atom_d_terms(const std::vector<std::vector<Real>>& terms_by_atom) {
        int number_of_atoms = static_cast<int>(effective_d_matrix_.size());
        for (int atom_index = 0; atom_index < number_of_atoms; ++atom_index) {
            const auto& terms = terms_by_atom[atom_index];
            if (terms.empty()) {
                continue;
            }
            auto& destination = effective_d_matrix_[atom_index];
            /**
             * A size mismatch means the caller's channel expansion disagrees
             * with the projector mapping table's, which would otherwise write
             * plausible numbers into the wrong elements. There is no
             * partial-overlap reading of these matrices that is meaningful, so
             * refusing is the only safe response.
             */
            if (terms.size() != destination.size()) {
                throw std::runtime_error("per-atom D terms do not match the projector channel "
                                         "count for atom " +
                                         std::to_string(atom_index));
            }
            for (size_t index = 0; index < destination.size(); ++index) {
                destination[index] += terms[index];
            }
        }
    }

    /**
     * Read an element of the effective D-matrix.
     */
    [[nodiscard]] Real
    element(int atom_index, int projector_index_i, int projector_index_j, int spin = 0) const {
        /**
         * Falls back to the spin-free matrix when no per-spin one was computed,
         * which is every unpolarized run and every path that still calls the
         * single-potential `compute`.
         */
        const auto& matrix = (spin < static_cast<int>(effective_d_matrix_by_spin_.size()) &&
                              !effective_d_matrix_by_spin_[spin].empty())
                                 ? effective_d_matrix_by_spin_[spin]
                                 : effective_d_matrix_;
        if (matrix.empty()) {
            return 0.0;
        }
        if (atom_index < static_cast<int>(matrix.size())) {
            int number_of_projectors_squared = static_cast<int>(matrix[atom_index].size());
            int number_of_projectors = static_cast<int>(
                std::round(std::sqrt(static_cast<double>(number_of_projectors_squared))));
            if (projector_index_i < number_of_projectors &&
                projector_index_j < number_of_projectors)
                return matrix[atom_index]
                             [projector_index_i * number_of_projectors + projector_index_j];
        }
        return 0.0;
    }

    [[nodiscard]] bool is_initialized() const {
        return !effective_d_matrix_.empty();
    }

    /**
     * Precompute and cache structure factors for all atoms.
     * These are constant during SCF since atomic positions don't change.
     */
    void precompute_structure_factors(const Crystal& crystal_structure,
                                      const PlanewaveBasis& planewave_basis) {
        int number_of_atoms = crystal_structure.number_of_atoms();
        int number_of_g_vectors = planewave_basis.number_of_g_vectors();
        cached_structure_factors_.resize(number_of_atoms);
        for (int atom_index = 0; atom_index < number_of_atoms; ++atom_index) {
            cached_structure_factors_[atom_index].resize(number_of_g_vectors);
            Vector3D atomic_position =
                crystal_structure.convert_from_crystal_to_cartesian(atom_index);
            for (int g_index = 0; g_index < number_of_g_vectors; ++g_index) {
                const auto& g_vector = planewave_basis.g_vectors()[g_index];
                Real phase_angle = constants::two_pi * g_vector.g_cart.dot(atomic_position);
                cached_structure_factors_[atom_index][g_index] = compute_phase_factor(phase_angle);
            }
        }
    }

    /**
     * Return the cached structure factors (computed on first call to compute()).
     * Useful for sharing with augmentation charge code.
     */
    [[nodiscard]] const std::vector<std::vector<Complex>>& cached_structure_factors() const {
        return cached_structure_factors_;
    }

  protected:
    /**
     * Effective D-matrix: per-atom × (projector_channels²).
     *
     * Protected rather than private because PAW's D is this D plus a
     * one-center term — the seam named in the A3 addendum. A subclass adds to
     * it in place; nothing else may.
     */
    std::vector<std::vector<Real>> effective_d_matrix_;

    /// One D per spin channel; empty when the spin-free `compute` was used.
    std::vector<std::vector<std::vector<Real>>> effective_d_matrix_by_spin_;

  private:
    /// Cached structure factors: per-atom × G-vectors (constant during SCF).
    std::vector<std::vector<Complex>> cached_structure_factors_;
};

} // namespace q3
