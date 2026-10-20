#pragma once

/**
 * DFT+U (Hubbard, Dudarev) Hellmann-Feynman force contribution.
 *
 * Computes the analytic gradient of E_U = (U/2)Tr[n(1-n)] with respect to atomic positions,
 * using the same "atomic" (non-orthogonalized) projector convention as the energy/potential
 * (effective_potential/internal/hubbard/hubbard.hpp). Structurally identical to
 * NonlocalPseudopotentialForceCalculator (../internal/nonlocal.hpp): both are D-matrix-weighted
 * sums over pairs of position-dependent G-space projectors (Kleinman-Bylander beta functions
 * there, atomic orbitals here), differentiated via the same -i(k+G) structure-factor-derivative
 * trick and collapsed into the same "2*Re[...]" form by the D-matrix's Hermiticity -- see
 * hubbard.hpp's accumulate_hubbard_force_gradient() for the full derivation.
 *
 * The atomic-orbital projectors are rebuilt fresh here (not reused from the SCF iterator's
 * cache), mirroring both NonlocalPseudopotentialForceCalculator's own KB projector rebuild and
 * PlanewaveWavefunctionProjectionCalculator's (projections/planewave/init.hpp) identical
 * pattern for the same atomic orbitals in a different post-SCF context.
 */

#include <algorithm>
#include <stdexcept>
#include <vector>

#include "layer_0_core/logging/init.hpp"
#include "layer_0_core/math/tolerances.hpp"
#include "layer_0_core/types/init.hpp"
#include "layer_0_core/types/matrix.hpp"
#include "layer_1_structure/crystal/crystal.hpp"
#include "layer_1_structure/reciprocal_space/kpoints_grid.hpp"
#include "layer_3_representation/basis/planewave/init.hpp"
#include "layer_3_representation/pseudopotential/planewave/internal/atomic_orbital_builder.hpp"
#include "layer_3_representation/pseudopotential/planewave/internal/lowdin_orthogonalization.hpp"
#include "layer_3_representation/pseudopotential/planewave/internal/projector.hpp"
#include "layer_3_representation/pseudopotential/planewave/internal/wavefunction_atomic.hpp"
#include "layer_3_representation/quantum_system/kohn_sham/planewave/ultrasoft/init.hpp"
#include "layer_4_computation/diagonalizer/planewave/internal/overlap/pseudopotential/factory.hpp"
#include "layer_4_computation/effective_potential/internal/hubbard/hubbard.hpp"
#include "layer_4_computation/state/kohn_sham/planewave/norm_conserving/init.hpp"

namespace q3 {

class HubbardForceCalculator {
  public:
    HubbardForceCalculator(const Crystal& crystal_structure,
                           const PlanewaveBasis& planewave_basis,
                           const KPointsGrid& kpoint_grid,
                           const NormConservingPlanewaveKohnShamSCFState& scf_state)
        : crystal_structure_(crystal_structure)
        , planewave_basis_(planewave_basis)
        , kpoint_grid_(kpoint_grid)
        , scf_state_(scf_state) {}

    [[nodiscard]] std::vector<Vector3D> compute() const {
        int number_of_atoms = crystal_structure_.number_of_atoms();
        std::vector<Vector3D> hubbard_force(number_of_atoms, {0.0, 0.0, 0.0});

        const auto& manifolds = scf_state_.hubbard_manifolds;
        if (manifolds.empty()) {
            return hubbard_force;
        }
        /**
         * Ortho-atomic projectors are a joint, system-wide transform of every
         * atomic orbital, so their gradient carries a term from orbitals that
         * belong to neither the manifold nor the displaced atom, and is
         * assembled separately.
         */
        if (scf_state_.hubbard_projector_type == "ortho-atomic") {
            return compute_with_orthogonal_projectors();
        }

        /**
         * Ultrasoft and PAW project through the overlap operator, so their
         * gradient carries dS/dR as well as dphi/dR and is assembled
         * separately.
         */
        if (scf_state_.quantum_system().has_ultrasoft()) {
            return compute_with_overlap_operator();
        }

        const auto& pseudopotential_set =
            scf_state_.quantum_system().pseudopotential_basis().pseudopotential_set();

        constexpr Real q_spacing = 0.01;
        constexpr int number_of_q_points = 900;
        auto atomic_wavefunction_table = tabulate_atomic_wavefunctions_in_q_space(
            pseudopotential_set, crystal_structure_.cell_volume(), q_spacing, number_of_q_points);

        size_t number_of_manifolds = manifolds.size();
        std::vector<int> manifold_row_offsets(number_of_manifolds);
        int total_hubbard_orbitals = 0;
        for (size_t manifold_index = 0; manifold_index < number_of_manifolds; manifold_index++) {
            manifold_row_offsets[manifold_index] = total_hubbard_orbitals;
            total_hubbard_orbitals +=
                static_cast<int>(manifolds[manifold_index].orbital_indices.size());
        }

        int number_of_kpoints = kpoint_grid_.number_of_kpoints();
        int number_of_bands = scf_state_.number_of_bands();
        int number_of_kpoint_states = number_of_kpoints * scf_state_.number_of_spin_components();
        const Matrix3x3& reciprocal_lattice_vectors =
            crystal_structure_.reciprocal_lattice_vectors();

        for (int kpoint_index_state = 0; kpoint_index_state < number_of_kpoint_states;
             kpoint_index_state++) {
            int spin_index = kpoint_index_state / number_of_kpoints;
            int kpoint_index = kpoint_index_state % number_of_kpoints;
            int number_of_plane_waves = scf_state_.number_of_plane_waves(kpoint_index);
            const auto& plane_wave_indices = scf_state_.g_vector_indices_at_kpoint(kpoint_index);
            Real kpoint_weight = kpoint_grid_[kpoint_index].weight;
            bool is_gamma_only = planewave_basis_.is_gamma_only();
            int g_zero_index = is_gamma_only ? find_g_zero_plane_wave_index(planewave_basis_,
                                                                            plane_wave_indices,
                                                                            number_of_plane_waves)
                                             : -1;

            Vector3D kpoint_vector = kpoint_grid_[kpoint_index].coordinates;
            Vector3D kpoint_cartesian;
            for (int direction = 0; direction < 3; direction++) {
                kpoint_cartesian[direction] =
                    kpoint_vector.x * reciprocal_lattice_vectors[0][direction] +
                    kpoint_vector.y * reciprocal_lattice_vectors[1][direction] +
                    kpoint_vector.z * reciprocal_lattice_vectors[2][direction];
            }

            std::vector<std::vector<Complex>> all_atomic_orbitals;
            build_atomic_orbitals_g_space(kpoint_index,
                                          number_of_plane_waves,
                                          plane_wave_indices,
                                          crystal_structure_,
                                          kpoint_grid_,
                                          planewave_basis_,
                                          pseudopotential_set,
                                          atomic_wavefunction_table,
                                          q_spacing,
                                          number_of_q_points,
                                          all_atomic_orbitals);

            /**
             * -i(k+G) per direction -- identical construction to
             * NonlocalPseudopotentialForceCalculator's own minus_i_k_plus_g.
             */
            std::vector<std::vector<Complex>> minus_i_k_plus_g(
                3, std::vector<Complex>(number_of_plane_waves));
            for (int g_index = 0; g_index < number_of_plane_waves; g_index++) {
                const auto& g_vector = planewave_basis_.g_vectors()[plane_wave_indices[g_index]];
                for (int direction = 0; direction < 3; direction++) {
                    Real k_plus_g_direction =
                        (g_vector.g_cart[direction] + kpoint_cartesian[direction]) *
                        crystal_structure_.two_pi_over_lattice_parameter();
                    minus_i_k_plus_g[direction][g_index] = Complex(0.0, -k_plus_g_direction);
                }
            }

            std::vector<Real> band_occupations(number_of_bands);
            for (int band_index = 0; band_index < number_of_bands; band_index++) {
                band_occupations[band_index] =
                    scf_state_.occupation(kpoint_index_state, band_index);
            }

            /**
             * ── Batched <phi_m|psi> and <phi_m|-i(k+G)|psi> ───────────────────────────────
             * These four overlap sets are the whole cost of this calculator. Each is one
             * ZGEMM over every Hubbard orbital and occupied band at once -- see
             * compute_hubbard_projector_overlaps()'s docstring -- structurally identical to
             * NonlocalPseudopotentialForceCalculator's own "4 overlaps per band" batching
             * (../internal/nonlocal.hpp), which this calculator already mirrors for the
             * physics. Both operands are packed once per k-point and shared across
             * manifolds, so neither the psi reads nor the orbital packing repeat per
             * Hubbard atom.
             *
             * Only occupied bands are contracted (as before -- the scalar loop skipped the
             * rest), and the results are scattered back into the same per-manifold
             * `overlaps` matrices, leaving the MPI reduction and the gradient accumulation
             * below untouched.
             */
            std::vector<int> occupied_band_indices;
            for (int band_index = 0; band_index < number_of_bands; band_index++) {
                if (band_occupations[band_index] >= math::tolerances::occupation_threshold) {
                    occupied_band_indices.push_back(band_index);
                }
            }
            int number_of_occupied_bands = static_cast<int>(occupied_band_indices.size());

            /**
             * Hubbard orbitals packed column-major, (number_of_plane_waves x
             * total_hubbard_orbitals), column `row_offset + m` holding manifold m's orbital.
             */
            std::vector<Complex> packed_orbitals(static_cast<size_t>(number_of_plane_waves) *
                                                 total_hubbard_orbitals);
            for (size_t manifold_index = 0; manifold_index < number_of_manifolds;
                 manifold_index++) {
                const auto& manifold = manifolds[manifold_index];
                int row_offset = manifold_row_offsets[manifold_index];
                for (size_t m = 0; m < manifold.orbital_indices.size(); m++) {
                    const auto& orbital =
                        all_atomic_orbitals[static_cast<size_t>(manifold.orbital_indices[m])];
                    std::copy(orbital.begin(),
                              orbital.begin() + number_of_plane_waves,
                              packed_orbitals.begin() +
                                  (static_cast<size_t>(row_offset) + m) * number_of_plane_waves);
                }
            }

            /**
             * batched_overlaps[c] is column-major (total_hubbard_orbitals x
             * number_of_occupied_bands): component 0 is <phi|psi>, components 1..3 are
             * <phi|-i(k+G)_direction|psi>.
             */
            std::vector<std::vector<Complex>> batched_overlaps(4);
            {
                std::vector<Complex> modified_wavefunction(
                    static_cast<size_t>(number_of_plane_waves) * number_of_occupied_bands);
                for (int component = 0; component < 4; component++) {
                    for (int occupied_band_index = 0;
                         occupied_band_index < number_of_occupied_bands;
                         occupied_band_index++) {
                        int band_index = occupied_band_indices[occupied_band_index];
                        size_t offset =
                            static_cast<size_t>(occupied_band_index) * number_of_plane_waves;
                        for (int g_index = 0; g_index < number_of_plane_waves; g_index++) {
                            Complex psi = scf_state_.wavefunction_coefficient(
                                kpoint_index_state, band_index, g_index);
                            modified_wavefunction[offset + g_index] =
                                (component == 0) ? psi
                                                 : psi * minus_i_k_plus_g[component - 1][g_index];
                        }
                    }
                    compute_hubbard_projector_overlaps(
                        number_of_plane_waves,
                        total_hubbard_orbitals,
                        number_of_occupied_bands,
                        packed_orbitals.data(),
                        modified_wavefunction.data(),
                        batched_overlaps[static_cast<size_t>(component)]);

                    /**
                     * Gamma-only stores one G of each +-G pair, so each of
                     * these sums is half of what it stands for. Component 0 is
                     * <phi_m|psi> and needs the G = 0 term the doubling would
                     * count twice removed; components 1-3 carry a factor of
                     * (k+G)_alpha = G_alpha, which vanishes at G = 0, so they
                     * take the doubling alone. Applying the correction to one
                     * and not the other is precisely the asymmetry that made
                     * the KB nonlocal force skew its components unevenly
                     * (KNOWN-ISSUES.md, 2026-08-08).
                     */
                    if (is_gamma_only) {
                        convert_gamma_only_projector_overlaps_to_full_sphere(
                            number_of_plane_waves,
                            total_hubbard_orbitals,
                            number_of_occupied_bands,
                            packed_orbitals.data(),
                            modified_wavefunction.data(),
                            g_zero_index,
                            /*band_functions_carry_g_component=*/component != 0,
                            batched_overlaps[static_cast<size_t>(component)]);
                    }
                }
            }

            for (size_t manifold_index = 0; manifold_index < number_of_manifolds;
                 manifold_index++) {
                const auto& manifold = manifolds[manifold_index];
                int manifold_size = static_cast<int>(manifold.orbital_indices.size());
                int row_offset = manifold_row_offsets[manifold_index];

                Matrix<Complex> overlaps(static_cast<size_t>(number_of_bands),
                                         static_cast<size_t>(manifold_size));
                std::vector<Matrix<Complex>> direction_derivative_overlaps(
                    3,
                    Matrix<Complex>(static_cast<size_t>(number_of_bands),
                                    static_cast<size_t>(manifold_size)));

                /**
                 * Scatter this manifold's row block out of the batched ZGEMM results.
                 * Rows of the unoccupied bands stay zero, exactly as the scalar loop's
                 * `continue` left them.
                 */
                for (int occupied_band_index = 0; occupied_band_index < number_of_occupied_bands;
                     occupied_band_index++) {
                    size_t band_index =
                        static_cast<size_t>(occupied_band_indices[occupied_band_index]);
                    size_t column_offset =
                        static_cast<size_t>(occupied_band_index) * total_hubbard_orbitals;
                    for (int m = 0; m < manifold_size; m++) {
                        size_t packed_index = column_offset + static_cast<size_t>(row_offset) +
                                              static_cast<size_t>(m);
                        overlaps(band_index, static_cast<size_t>(m)) =
                            batched_overlaps[0][packed_index];
                        for (int direction = 0; direction < 3; direction++) {
                            direction_derivative_overlaps[direction](band_index,
                                                                     static_cast<size_t>(m)) =
                                batched_overlaps[static_cast<size_t>(direction) + 1][packed_index];
                        }
                    }
                }

                /**
                 * Band-parallel (G-vector-distributed) MPI: the g_index loop above only
                 * covers this rank's local plane-wave subset -- reduce across the
                 * band-parallel group before use, mirroring both the Hamiltonian
                 * applicator's <phi_m|psi> reduction for H|psi> and this same fix applied to
                 * the occupation-matrix accumulation
                 * (iterator/kohn_sham/planewave/pseudopotential/init.hpp). No-op when
                 * band_size == 1.
                 */
                if (scf_state_.execution_policy != nullptr) {
                    scf_state_.execution_policy->reduce_within_group(
                        overlaps.data(), number_of_bands * manifold_size);
                    for (int direction = 0; direction < 3; direction++) {
                        scf_state_.execution_policy->reduce_within_group(
                            direction_derivative_overlaps[direction].data(),
                            number_of_bands * manifold_size);
                    }
                }

                /**
                 * Extract this manifold's manifold_size x manifold_size block (row-major,
                 * matching accumulate_hubbard_force_gradient's expectation) from the combined
                 * block-diagonal buffer (column-major, matching the applicator's BLAS
                 * convention -- see accumulate_hubbard_occupation_and_potential's own comment).
                 */
                const auto& combined_d_matrix = scf_state_.hubbard_d_matrix_per_spin[spin_index];
                std::vector<Complex> manifold_d_matrix(static_cast<size_t>(manifold_size) *
                                                       manifold_size);
                for (int m = 0; m < manifold_size; m++) {
                    for (int m_prime = 0; m_prime < manifold_size; m_prime++) {
                        int global_row = row_offset + m;
                        int global_column = row_offset + m_prime;
                        manifold_d_matrix[static_cast<size_t>(m * manifold_size + m_prime)] =
                            combined_d_matrix[static_cast<size_t>(global_row) +
                                              static_cast<size_t>(global_column) *
                                                  total_hubbard_orbitals];
                    }
                }

                for (int direction = 0; direction < 3; direction++) {
                    Real force_contribution =
                        accumulate_hubbard_force_gradient(number_of_bands,
                                                          manifold_size,
                                                          overlaps,
                                                          direction_derivative_overlaps[direction],
                                                          band_occupations,
                                                          kpoint_weight,
                                                          manifold_d_matrix);
                    hubbard_force[manifold.atom_index][direction] += force_contribution;
                }
            }
        }

        return hubbard_force;
    }

  private:
    /**
     * The DFT+U force when the projector is the Löwdin-orthogonalized orbital.
     *
     * ── What is being differentiated ──────────────────────────────────────
     *
     * The ortho-atomic projector is not a property of its own atom. The whole
     * system-wide atomic-orbital set {phi_J} is orthogonalized jointly,
     *
     *     phi^ortho_I = sum_J (O^-1/2)_JI phi_J,   O_IJ = <phi_I|phi_J>,
     *
     * so the projection whose occupation DFT+U is built from,
     *
     *     P_Iv = <phi^ortho_I|psi_v> = sum_J (O^-1/2)_IJ <phi_J|psi_v>,
     *
     * depends on the position of *every* atom in the cell through O. That is
     * the term the atomic path (compute() above) has no analogue of: there a
     * Hubbard force appears only on a Hubbard atom, here displacing any atom
     * exerts one on every manifold. It is Pulay-like in origin -- a
     * position-dependent basis -- but it is not the usual Pulay term, because
     * it is the *projector* basis that moves, not the wavefunction basis.
     *
     * Differentiating, with alpha the displaced atom and its own orbitals the
     * only ones with a non-zero dphi/dtau,
     *
     *     dP_Iv/dtau = sum_J d(O^-1/2)_IJ <phi_J|psi_v>
     *                + sum_{J in alpha} (O^-1/2)_IJ <dphi_J/dtau|psi_v>
     *
     * (using that O^-1/2 and its derivative are both Hermitian, so the
     * conjugate of the (J,I) element is the (I,J) element). dO/dtau follows
     * from the same product rule,
     *
     *     dO_IJ/dtau = [J in alpha] T_IJ - [I in alpha] T_IJ,
     *     T_IJ = <phi_I|-i(k+G)|phi_J>,
     *
     * which is antisymmetric enough to vanish identically when I and J both
     * belong to alpha -- rigidly translating an atom cannot change the
     * overlaps among its own orbitals. d(O^-1/2) then comes from dO and O's
     * eigendecomposition in closed form (lowdin_inverse_square_root_derivative),
     * which evaluates Eq. (32) of Phys. Rev. B 102, 235159 (2020).
     *
     * ── Sign convention ───────────────────────────────────────────────────
     *
     * `accumulate_hubbard_force_gradient` is shared with the norm-conserving
     * atomic path, which hands it <phi_m|-i(k+G)|psi>, i.e. **-dP/dtau** (see
     * compute_with_overlap_operator's own note). The same quantity is passed
     * here:
     *
     *     -dP_Iv/dtau = -sum_J d(O^-1/2)_IJ <phi_J|psi_v>
     *                 + sum_{J in alpha} (O^-1/2)_IJ T^psi_Jv
     *
     * with T^psi_Jv = <phi_J|-i(k+G)|psi_v>, using
     * <dphi_J/dtau|psi> = -T^psi_Jv. Freeze O^-1/2 at the identity and drop
     * its derivative and this collapses to T^psi_Iv on the manifold's own
     * atom and to zero everywhere else -- exactly what the atomic path passes.
     * Pinning the sign by reduction to an already-validated path is safer
     * than re-deriving it from the energy.
     *
     * For ultrasoft and PAW, O is the generalized <phi|S|phi> metric. The
     * additional branch below differentiates both S occurrences explicitly;
     * the norm-conserving algebra remains the simpler S=1 specialization.
     */
    [[nodiscard]] std::vector<Vector3D> compute_with_orthogonal_projectors() const {
        int number_of_atoms = crystal_structure_.number_of_atoms();
        std::vector<Vector3D> hubbard_force(number_of_atoms, {0.0, 0.0, 0.0});

        const auto& manifolds = scf_state_.hubbard_manifolds;
        const auto& pseudopotential_basis = scf_state_.quantum_system().pseudopotential_basis();
        const auto& pseudopotential_set = pseudopotential_basis.pseudopotential_set();
        bool has_augmented_overlap = scf_state_.quantum_system().has_ultrasoft();

        constexpr Real q_spacing = 0.01;
        constexpr int number_of_q_points = 900;
        auto atomic_wavefunction_table = tabulate_atomic_wavefunctions_in_q_space(
            pseudopotential_set, crystal_structure_.cell_volume(), q_spacing, number_of_q_points);

        /**
         * Where each atom's own atomic orbitals sit in the system-wide list.
         * enumerate_atomic_orbitals is atom-major, so each atom owns one
         * contiguous run, and displacing that atom is the only thing that gives
         * the orbitals in it a non-zero dphi/dtau. An atom contributing no
         * orbitals is left with first > last, making every loop over its range
         * empty.
         */
        auto orbital_labels = enumerate_atomic_orbitals(crystal_structure_, pseudopotential_set);
        int number_of_atomic_orbitals = static_cast<int>(orbital_labels.size());
        std::vector<int> atom_orbital_first(number_of_atoms, number_of_atomic_orbitals);
        std::vector<int> atom_orbital_last(number_of_atoms, 0);
        for (int orbital = 0; orbital < number_of_atomic_orbitals; orbital++) {
            int atom_index = orbital_labels[static_cast<size_t>(orbital)].atom_index;
            atom_orbital_first[atom_index] = std::min(atom_orbital_first[atom_index], orbital);
            atom_orbital_last[atom_index] = std::max(atom_orbital_last[atom_index], orbital + 1);
        }

        int total_projectors = pseudopotential_basis.total_number_of_projectors();
        std::vector<int> projector_offset_by_atom(number_of_atoms, 0);
        std::vector<int> projector_channels_by_atom(number_of_atoms, 0);
        {
            int projector_offset = 0;
            for (int atom_index = 0; atom_index < number_of_atoms; atom_index++) {
                int species_index = crystal_structure_.species_type_index(atom_index);
                int number_of_projector_channels =
                    pseudopotential_basis.number_of_projector_channels(species_index);
                projector_offset_by_atom[atom_index] = projector_offset;
                projector_channels_by_atom[atom_index] = number_of_projector_channels;
                projector_offset += number_of_projector_channels;
            }
        }
        auto augmentation_overlap = [&](int atom_index, int channel_i, int channel_j) -> Real {
            return hubbard_augmentation_overlap(
                static_cast<const UltrasoftPlanewaveKohnShamQuantumSystem&>(
                    scf_state_.quantum_system())
                    .ultrasoft_basis(),
                pseudopotential_basis,
                crystal_structure_.species_type_index(atom_index),
                channel_i,
                channel_j);
        };

        size_t number_of_manifolds = manifolds.size();
        std::vector<int> manifold_row_offsets(number_of_manifolds);
        int total_hubbard_orbitals = 0;
        for (size_t manifold_index = 0; manifold_index < number_of_manifolds; manifold_index++) {
            manifold_row_offsets[manifold_index] = total_hubbard_orbitals;
            total_hubbard_orbitals +=
                static_cast<int>(manifolds[manifold_index].orbital_indices.size());
        }

        int number_of_kpoints = kpoint_grid_.number_of_kpoints();
        int number_of_bands = scf_state_.number_of_bands();
        int number_of_kpoint_states = number_of_kpoints * scf_state_.number_of_spin_components();
        const Matrix3x3& reciprocal_lattice_vectors =
            crystal_structure_.reciprocal_lattice_vectors();

        for (int kpoint_index_state = 0; kpoint_index_state < number_of_kpoint_states;
             kpoint_index_state++) {
            int spin_index = kpoint_index_state / number_of_kpoints;
            int kpoint_index = kpoint_index_state % number_of_kpoints;
            int number_of_plane_waves = scf_state_.number_of_plane_waves(kpoint_index);
            const auto& plane_wave_indices = scf_state_.g_vector_indices_at_kpoint(kpoint_index);
            Real kpoint_weight = kpoint_grid_[kpoint_index].weight;
            bool is_gamma_only = planewave_basis_.is_gamma_only();
            int g_zero_index = is_gamma_only ? find_g_zero_plane_wave_index(planewave_basis_,
                                                                            plane_wave_indices,
                                                                            number_of_plane_waves)
                                             : -1;

            Vector3D kpoint_cartesian;
            {
                Vector3D kpoint_vector = kpoint_grid_[kpoint_index].coordinates;
                for (int direction = 0; direction < 3; direction++)
                    kpoint_cartesian[direction] =
                        kpoint_vector.x * reciprocal_lattice_vectors[0][direction] +
                        kpoint_vector.y * reciprocal_lattice_vectors[1][direction] +
                        kpoint_vector.z * reciprocal_lattice_vectors[2][direction];
            }

            std::vector<std::vector<Complex>> all_atomic_orbitals;
            build_atomic_orbitals_g_space(kpoint_index,
                                          number_of_plane_waves,
                                          plane_wave_indices,
                                          crystal_structure_,
                                          kpoint_grid_,
                                          planewave_basis_,
                                          pseudopotential_set,
                                          atomic_wavefunction_table,
                                          q_spacing,
                                          number_of_q_points,
                                          all_atomic_orbitals);

            std::vector<std::vector<Complex>> minus_i_k_plus_g(
                3, std::vector<Complex>(number_of_plane_waves));
            for (int g_index = 0; g_index < number_of_plane_waves; g_index++) {
                const auto& g_vector = planewave_basis_.g_vectors()[plane_wave_indices[g_index]];
                for (int direction = 0; direction < 3; direction++)
                    minus_i_k_plus_g[direction][g_index] =
                        Complex(0.0,
                                -(g_vector.g_cart[direction] + kpoint_cartesian[direction]) *
                                    crystal_structure_.two_pi_over_lattice_parameter());
            }

            std::vector<Real> band_occupations(number_of_bands);
            std::vector<int> occupied_band_indices;
            for (int band_index = 0; band_index < number_of_bands; band_index++) {
                band_occupations[band_index] =
                    scf_state_.occupation(kpoint_index_state, band_index);
                if (band_occupations[band_index] >= math::tolerances::occupation_threshold)
                    occupied_band_indices.push_back(band_index);
            }
            int number_of_occupied_bands = static_cast<int>(occupied_band_indices.size());

            /**
             * Column-major (number_of_plane_waves x N) operands, the layout
             * compute_hubbard_projector_overlaps' ZGEMM expects, plus their
             * -i(k+G)-weighted counterparts. Unlike the atomic path this needs
             * the FULL orbital set, not just the Hubbard manifolds' columns:
             * O and its derivative run over every atomic orbital in the cell.
             */
            std::vector<Complex> packed_orbitals(static_cast<size_t>(number_of_plane_waves) *
                                                 number_of_atomic_orbitals);
            for (int orbital = 0; orbital < number_of_atomic_orbitals; orbital++)
                std::copy(all_atomic_orbitals[static_cast<size_t>(orbital)].begin(),
                          all_atomic_orbitals[static_cast<size_t>(orbital)].begin() +
                              number_of_plane_waves,
                          packed_orbitals.begin() +
                              static_cast<size_t>(orbital) * number_of_plane_waves);
            std::vector<Complex> packed_bands(static_cast<size_t>(number_of_plane_waves) *
                                              number_of_occupied_bands);
            for (int occupied_band_index = 0; occupied_band_index < number_of_occupied_bands;
                 occupied_band_index++)
                for (int g_index = 0; g_index < number_of_plane_waves; g_index++)
                    packed_bands[static_cast<size_t>(occupied_band_index) * number_of_plane_waves +
                                 g_index] =
                        scf_state_.wavefunction_coefficient(
                            kpoint_index_state,
                            occupied_band_indices[static_cast<size_t>(occupied_band_index)],
                            g_index);

            auto weight_by_direction = [&](const std::vector<Complex>& source, int direction) {
                std::vector<Complex> weighted(source.size());
                size_t columns = source.size() / static_cast<size_t>(number_of_plane_waves);
                for (size_t column = 0; column < columns; column++)
                    for (int g_index = 0; g_index < number_of_plane_waves; g_index++) {
                        size_t offset = column * number_of_plane_waves + g_index;
                        weighted[offset] = source[offset] * minus_i_k_plus_g[direction][g_index];
                    }
                return weighted;
            };

            /**
             * Every contraction is a partial sum under band (G-vector)
             * parallelism and is reduced before use, the same reason the SCF
             * reduces the occupation matrix at its own G-sum. Gamma-only
             * contractions are first reconstructed from the stored half sphere;
             * derivative operands carry a G component and therefore no G=0 term.
             */
            auto contract = [&](const Complex* left,
                                int number_of_left,
                                const Complex* right,
                                int number_of_right,
                                bool right_carries_g_component = false) {
                std::vector<Complex> result;
                compute_hubbard_projector_overlaps(
                    number_of_plane_waves, number_of_left, number_of_right, left, right, result);
                if (is_gamma_only)
                    convert_gamma_only_projector_overlaps_to_full_sphere(number_of_plane_waves,
                                                                         number_of_left,
                                                                         number_of_right,
                                                                         left,
                                                                         right,
                                                                         g_zero_index,
                                                                         right_carries_g_component,
                                                                         result);
                if (scf_state_.execution_policy != nullptr)
                    scf_state_.execution_policy->reduce_within_group(
                        result.data(), static_cast<int>(result.size()));
                return result;
            };

            if (has_augmented_overlap) {
                if (scf_state_.execution_policy == nullptr) {
                    throw std::runtime_error(
                        "Ortho-atomic ultrasoft Hubbard forces require an execution policy");
                }

                Matrix<Complex> kleinman_bylander_projectors;
                compute_kleinman_bylander_projectors(kpoint_grid_[kpoint_index].coordinates,
                                                     crystal_structure_,
                                                     planewave_basis_,
                                                     pseudopotential_set,
                                                     pseudopotential_basis.beta_function_table(),
                                                     plane_wave_indices,
                                                     number_of_plane_waves,
                                                     total_projectors,
                                                     kleinman_bylander_projectors);
                auto overlap_operator = create_overlap_operator(crystal_structure_,
                                                                scf_state_.quantum_system(),
                                                                kleinman_bylander_projectors,
                                                                *scf_state_.execution_policy);

                std::vector<Complex> overlap_applied_orbitals(packed_orbitals.size());
                overlap_operator->apply_batch(number_of_plane_waves,
                                              number_of_atomic_orbitals,
                                              packed_orbitals.data(),
                                              overlap_applied_orbitals.data());
                std::vector<Complex> overlap_applied_bands(packed_bands.size());
                overlap_operator->apply_batch(number_of_plane_waves,
                                              number_of_occupied_bands,
                                              packed_bands.data(),
                                              overlap_applied_bands.data());

                auto overlap_matrix = contract(packed_orbitals.data(),
                                               number_of_atomic_orbitals,
                                               overlap_applied_orbitals.data(),
                                               number_of_atomic_orbitals);
                auto transform =
                    compute_lowdin_transform(overlap_matrix, number_of_atomic_orbitals);
                auto atomic_band_overlaps = contract(packed_orbitals.data(),
                                                     number_of_atomic_orbitals,
                                                     overlap_applied_bands.data(),
                                                     number_of_occupied_bands);
                auto element = [number_of_atomic_orbitals](int row, int column) {
                    return static_cast<size_t>(row + column * number_of_atomic_orbitals);
                };

                // P_Av = sum_I (O^-1/2)_AI <phi_I|S|psi_v>.
                std::vector<Complex> projection(static_cast<size_t>(number_of_atomic_orbitals) *
                                                    number_of_occupied_bands,
                                                Complex(0.0, 0.0));
                for (int occupied_band_index = 0; occupied_band_index < number_of_occupied_bands;
                     occupied_band_index++) {
                    for (int orbital = 0; orbital < number_of_atomic_orbitals; orbital++) {
                        for (int inner_orbital = 0; inner_orbital < number_of_atomic_orbitals;
                             inner_orbital++) {
                            projection[element(orbital, occupied_band_index)] +=
                                transform.inverse_square_root[element(orbital, inner_orbital)] *
                                atomic_band_overlaps[element(inner_orbital, occupied_band_index)];
                        }
                    }
                }

                /**
                 * Apply dS/dR to a column-major vector block. Both beta overlaps are
                 * completed across the band group before the local G-space result is
                 * assembled, matching the production overlap operator's collective order.
                 */
                auto apply_overlap_derivative = [&](const Complex* input_vectors,
                                                    int number_of_vectors,
                                                    int displaced_atom,
                                                    int direction) {
                    std::vector<Complex> derivative_projectors(
                        static_cast<size_t>(number_of_plane_waves) * total_projectors,
                        Complex(0.0, 0.0));
                    int displaced_projector_offset = projector_offset_by_atom[displaced_atom];
                    int displaced_projector_channels = projector_channels_by_atom[displaced_atom];
                    for (int channel = 0; channel < displaced_projector_channels; channel++) {
                        int projector_index = displaced_projector_offset + channel;
                        for (int g_index = 0; g_index < number_of_plane_waves; g_index++) {
                            derivative_projectors[static_cast<size_t>(projector_index) *
                                                      number_of_plane_waves +
                                                  g_index] =
                                kleinman_bylander_projectors[projector_index][g_index] *
                                minus_i_k_plus_g[direction][g_index];
                        }
                    }

                    auto projector_overlaps = contract(kleinman_bylander_projectors.data(),
                                                       total_projectors,
                                                       input_vectors,
                                                       number_of_vectors);
                    auto derivative_projector_overlaps = contract(derivative_projectors.data(),
                                                                  total_projectors,
                                                                  input_vectors,
                                                                  number_of_vectors);
                    std::vector<Complex> result(static_cast<size_t>(number_of_plane_waves) *
                                                    number_of_vectors,
                                                Complex(0.0, 0.0));
                    for (int channel_row = 0; channel_row < displaced_projector_channels;
                         channel_row++) {
                        int projector_row = displaced_projector_offset + channel_row;
                        for (int channel_column = 0; channel_column < displaced_projector_channels;
                             channel_column++) {
                            Real overlap_value =
                                augmentation_overlap(displaced_atom, channel_row, channel_column);
                            if (std::abs(overlap_value) <
                                math::tolerances::d_matrix_element_threshold) {
                                continue;
                            }
                            int projector_column = displaced_projector_offset + channel_column;
                            for (int vector_index = 0; vector_index < number_of_vectors;
                                 vector_index++) {
                                size_t overlap_index =
                                    static_cast<size_t>(projector_column) +
                                    static_cast<size_t>(vector_index) * total_projectors;
                                Complex derivative_projector_coefficient =
                                    overlap_value * projector_overlaps[overlap_index];
                                Complex projector_coefficient =
                                    overlap_value * derivative_projector_overlaps[overlap_index];
                                for (int g_index = 0; g_index < number_of_plane_waves; g_index++) {
                                    size_t result_index =
                                        static_cast<size_t>(vector_index) * number_of_plane_waves +
                                        g_index;
                                    result[result_index] +=
                                        derivative_projectors[static_cast<size_t>(projector_row) *
                                                                  number_of_plane_waves +
                                                              g_index] *
                                            derivative_projector_coefficient +
                                        kleinman_bylander_projectors[projector_row][g_index] *
                                            projector_coefficient;
                                }
                            }
                        }
                    }
                    return result;
                };

                for (int displaced_atom = 0; displaced_atom < number_of_atoms; displaced_atom++) {
                    int displaced_first = atom_orbital_first[displaced_atom];
                    int displaced_last = atom_orbital_last[displaced_atom];

                    for (int direction = 0; direction < 3; direction++) {
                        std::vector<Complex> orbital_derivatives(packed_orbitals.size(),
                                                                 Complex(0.0, 0.0));
                        for (int orbital = displaced_first; orbital < displaced_last; orbital++) {
                            for (int g_index = 0; g_index < number_of_plane_waves; g_index++) {
                                size_t orbital_index =
                                    static_cast<size_t>(orbital) * number_of_plane_waves + g_index;
                                orbital_derivatives[orbital_index] =
                                    packed_orbitals[orbital_index] *
                                    minus_i_k_plus_g[direction][g_index];
                            }
                        }

                        std::vector<Complex> overlap_applied_orbital_derivatives(
                            orbital_derivatives.size());
                        overlap_operator->apply_batch(number_of_plane_waves,
                                                      number_of_atomic_orbitals,
                                                      orbital_derivatives.data(),
                                                      overlap_applied_orbital_derivatives.data());
                        auto overlap_derivative_on_orbitals =
                            apply_overlap_derivative(packed_orbitals.data(),
                                                     number_of_atomic_orbitals,
                                                     displaced_atom,
                                                     direction);
                        auto overlap_derivative_on_bands =
                            apply_overlap_derivative(packed_bands.data(),
                                                     number_of_occupied_bands,
                                                     displaced_atom,
                                                     direction);

                        auto left_orbital_metric_derivative =
                            contract(orbital_derivatives.data(),
                                     number_of_atomic_orbitals,
                                     overlap_applied_orbitals.data(),
                                     number_of_atomic_orbitals);
                        auto overlap_metric_derivative =
                            contract(packed_orbitals.data(),
                                     number_of_atomic_orbitals,
                                     overlap_derivative_on_orbitals.data(),
                                     number_of_atomic_orbitals);
                        auto right_orbital_metric_derivative =
                            contract(packed_orbitals.data(),
                                     number_of_atomic_orbitals,
                                     overlap_applied_orbital_derivatives.data(),
                                     number_of_atomic_orbitals);
                        std::vector<Complex> metric_derivative(overlap_matrix.size());
                        for (size_t matrix_index = 0; matrix_index < metric_derivative.size();
                             matrix_index++) {
                            metric_derivative[matrix_index] =
                                left_orbital_metric_derivative[matrix_index] +
                                overlap_metric_derivative[matrix_index] +
                                right_orbital_metric_derivative[matrix_index];
                        }
                        for (int column = 0; column < number_of_atomic_orbitals; column++) {
                            for (int row = 0; row <= column; row++) {
                                Complex hermitian_value =
                                    0.5 * (metric_derivative[element(row, column)] +
                                           std::conj(metric_derivative[element(column, row)]));
                                metric_derivative[element(row, column)] = hermitian_value;
                                metric_derivative[element(column, row)] =
                                    std::conj(hermitian_value);
                            }
                        }
                        auto inverse_square_root_derivative =
                            lowdin_inverse_square_root_derivative(transform, metric_derivative);

                        auto left_orbital_projection_derivative =
                            contract(orbital_derivatives.data(),
                                     number_of_atomic_orbitals,
                                     overlap_applied_bands.data(),
                                     number_of_occupied_bands);
                        auto overlap_projection_derivative =
                            contract(packed_orbitals.data(),
                                     number_of_atomic_orbitals,
                                     overlap_derivative_on_bands.data(),
                                     number_of_occupied_bands);
                        std::vector<Complex> raw_projection_derivative(atomic_band_overlaps.size());
                        for (size_t overlap_index = 0;
                             overlap_index < raw_projection_derivative.size();
                             overlap_index++) {
                            raw_projection_derivative[overlap_index] =
                                left_orbital_projection_derivative[overlap_index] +
                                overlap_projection_derivative[overlap_index];
                        }

                        std::vector<Complex> negative_projection_derivative(projection.size(),
                                                                            Complex(0.0, 0.0));
                        for (int occupied_band_index = 0;
                             occupied_band_index < number_of_occupied_bands;
                             occupied_band_index++) {
                            for (int orbital = 0; orbital < number_of_atomic_orbitals; orbital++) {
                                for (int inner_orbital = 0;
                                     inner_orbital < number_of_atomic_orbitals;
                                     inner_orbital++) {
                                    negative_projection_derivative[element(orbital,
                                                                           occupied_band_index)] -=
                                        inverse_square_root_derivative[element(orbital,
                                                                               inner_orbital)] *
                                            atomic_band_overlaps[element(inner_orbital,
                                                                         occupied_band_index)] +
                                        transform.inverse_square_root[element(orbital,
                                                                              inner_orbital)] *
                                            raw_projection_derivative[element(inner_orbital,
                                                                              occupied_band_index)];
                                }
                            }
                        }

                        for (size_t manifold_index = 0; manifold_index < number_of_manifolds;
                             manifold_index++) {
                            const auto& manifold = manifolds[manifold_index];
                            int manifold_size = static_cast<int>(manifold.orbital_indices.size());
                            int row_offset = manifold_row_offsets[manifold_index];
                            Matrix<Complex> manifold_overlaps(static_cast<size_t>(number_of_bands),
                                                              static_cast<size_t>(manifold_size));
                            Matrix<Complex> derivative_overlaps(
                                static_cast<size_t>(number_of_bands),
                                static_cast<size_t>(manifold_size));
                            for (int manifold_orbital = 0; manifold_orbital < manifold_size;
                                 manifold_orbital++) {
                                int orbital =
                                    manifold.orbital_indices[static_cast<size_t>(manifold_orbital)];
                                for (int occupied_band_index = 0;
                                     occupied_band_index < number_of_occupied_bands;
                                     occupied_band_index++) {
                                    size_t band_index = static_cast<size_t>(
                                        occupied_band_indices[static_cast<size_t>(
                                            occupied_band_index)]);
                                    manifold_overlaps(band_index,
                                                      static_cast<size_t>(manifold_orbital)) =
                                        projection[element(orbital, occupied_band_index)];
                                    derivative_overlaps(band_index,
                                                        static_cast<size_t>(manifold_orbital)) =
                                        negative_projection_derivative[element(
                                            orbital, occupied_band_index)];
                                }
                            }

                            const auto& combined_d_matrix =
                                scf_state_.hubbard_d_matrix_per_spin[spin_index];
                            std::vector<Complex> manifold_d_matrix(
                                static_cast<size_t>(manifold_size) * manifold_size);
                            for (int row = 0; row < manifold_size; row++) {
                                for (int column = 0; column < manifold_size; column++) {
                                    manifold_d_matrix[static_cast<size_t>(row * manifold_size +
                                                                          column)] =
                                        combined_d_matrix[static_cast<size_t>(row_offset + row) +
                                                          static_cast<size_t>(row_offset + column) *
                                                              total_hubbard_orbitals];
                                }
                            }

                            hubbard_force[displaced_atom][direction] +=
                                accumulate_hubbard_force_gradient(number_of_bands,
                                                                  manifold_size,
                                                                  manifold_overlaps,
                                                                  derivative_overlaps,
                                                                  band_occupations,
                                                                  kpoint_weight,
                                                                  manifold_d_matrix);
                        }
                    }
                }
                continue;
            }

            auto overlap_matrix = contract(packed_orbitals.data(),
                                           number_of_atomic_orbitals,
                                           packed_orbitals.data(),
                                           number_of_atomic_orbitals);
            auto transform = compute_lowdin_transform(overlap_matrix, number_of_atomic_orbitals);
            auto atomic_band_overlaps = contract(packed_orbitals.data(),
                                                 number_of_atomic_orbitals,
                                                 packed_bands.data(),
                                                 number_of_occupied_bands);
            std::vector<std::vector<Complex>> orbital_derivative_overlaps(3);
            std::vector<std::vector<Complex>> band_derivative_overlaps(3);
            for (int direction = 0; direction < 3; direction++) {
                auto weighted_orbitals = weight_by_direction(packed_orbitals, direction);
                auto weighted_bands = weight_by_direction(packed_bands, direction);
                orbital_derivative_overlaps[static_cast<size_t>(direction)] =
                    contract(packed_orbitals.data(),
                             number_of_atomic_orbitals,
                             weighted_orbitals.data(),
                             number_of_atomic_orbitals,
                             /*right_carries_g_component=*/true);
                band_derivative_overlaps[static_cast<size_t>(direction)] =
                    contract(packed_orbitals.data(),
                             number_of_atomic_orbitals,
                             weighted_bands.data(),
                             number_of_occupied_bands,
                             /*right_carries_g_component=*/true);
            }

            auto element = [number_of_atomic_orbitals](int row, int column) {
                return static_cast<size_t>(row + column * number_of_atomic_orbitals);
            };

            // P_Iv = sum_J (O^-1/2)_IJ <phi_J|psi_v>.
            std::vector<Complex> projection(static_cast<size_t>(number_of_atomic_orbitals) *
                                                number_of_occupied_bands,
                                            Complex(0.0, 0.0));
            for (int occupied_band_index = 0; occupied_band_index < number_of_occupied_bands;
                 occupied_band_index++)
                for (int orbital = 0; orbital < number_of_atomic_orbitals; orbital++) {
                    Complex accumulated(0.0, 0.0);
                    for (int inner = 0; inner < number_of_atomic_orbitals; inner++)
                        accumulated += transform.inverse_square_root[element(orbital, inner)] *
                                       atomic_band_overlaps[element(inner, occupied_band_index)];
                    projection[element(orbital, occupied_band_index)] = accumulated;
                }

            for (int displaced_atom = 0; displaced_atom < number_of_atoms; displaced_atom++) {
                int displaced_first = atom_orbital_first[displaced_atom];
                int displaced_last = atom_orbital_last[displaced_atom];
                auto belongs_to_displaced_atom = [&](int orbital) {
                    return orbital >= displaced_first && orbital < displaced_last;
                };

                for (int direction = 0; direction < 3; direction++) {
                    const auto& orbital_derivative =
                        orbital_derivative_overlaps[static_cast<size_t>(direction)];
                    const auto& band_derivative =
                        band_derivative_overlaps[static_cast<size_t>(direction)];

                    std::vector<Complex> overlap_derivative(
                        static_cast<size_t>(number_of_atomic_orbitals) * number_of_atomic_orbitals,
                        Complex(0.0, 0.0));
                    for (int column = 0; column < number_of_atomic_orbitals; column++)
                        for (int row = 0; row < number_of_atomic_orbitals; row++) {
                            bool row_moves = belongs_to_displaced_atom(row);
                            bool column_moves = belongs_to_displaced_atom(column);
                            if (row_moves == column_moves)
                                continue;
                            Complex value = orbital_derivative[element(row, column)];
                            overlap_derivative[element(row, column)] =
                                column_moves ? value : -value;
                        }
                    auto inverse_square_root_derivative =
                        lowdin_inverse_square_root_derivative(transform, overlap_derivative);

                    for (size_t manifold_index = 0; manifold_index < number_of_manifolds;
                         manifold_index++) {
                        const auto& manifold = manifolds[manifold_index];
                        int manifold_size = static_cast<int>(manifold.orbital_indices.size());
                        int row_offset = manifold_row_offsets[manifold_index];

                        Matrix<Complex> manifold_overlaps(static_cast<size_t>(number_of_bands),
                                                          static_cast<size_t>(manifold_size));
                        Matrix<Complex> derivative_overlaps(static_cast<size_t>(number_of_bands),
                                                            static_cast<size_t>(manifold_size));
                        for (int m = 0; m < manifold_size; m++) {
                            int orbital = manifold.orbital_indices[static_cast<size_t>(m)];
                            for (int occupied_band_index = 0;
                                 occupied_band_index < number_of_occupied_bands;
                                 occupied_band_index++) {
                                Complex transform_term(0.0, 0.0);
                                for (int inner = 0; inner < number_of_atomic_orbitals; inner++)
                                    transform_term +=
                                        inverse_square_root_derivative[element(orbital, inner)] *
                                        atomic_band_overlaps[element(inner, occupied_band_index)];
                                Complex own_orbital_term(0.0, 0.0);
                                for (int inner = displaced_first; inner < displaced_last; inner++)
                                    own_orbital_term +=
                                        transform.inverse_square_root[element(orbital, inner)] *
                                        band_derivative[element(inner, occupied_band_index)];
                                size_t band_row =
                                    static_cast<size_t>(occupied_band_indices[static_cast<size_t>(
                                        occupied_band_index)]);
                                manifold_overlaps(band_row, static_cast<size_t>(m)) =
                                    projection[element(orbital, occupied_band_index)];
                                derivative_overlaps(band_row, static_cast<size_t>(m)) =
                                    own_orbital_term - transform_term;
                            }
                        }

                        const auto& combined_d_matrix =
                            scf_state_.hubbard_d_matrix_per_spin[spin_index];
                        std::vector<Complex> manifold_d_matrix(static_cast<size_t>(manifold_size) *
                                                               manifold_size);
                        for (int m = 0; m < manifold_size; m++)
                            for (int m_prime = 0; m_prime < manifold_size; m_prime++)
                                manifold_d_matrix[static_cast<size_t>(m * manifold_size +
                                                                      m_prime)] =
                                    combined_d_matrix[static_cast<size_t>(row_offset + m) +
                                                      static_cast<size_t>(row_offset + m_prime) *
                                                          total_hubbard_orbitals];

                        hubbard_force[displaced_atom][direction] +=
                            accumulate_hubbard_force_gradient(number_of_bands,
                                                              manifold_size,
                                                              manifold_overlaps,
                                                              derivative_overlaps,
                                                              band_occupations,
                                                              kpoint_weight,
                                                              manifold_d_matrix);
                    }
                }
            }
        }

        return hubbard_force;
    }

    /**
     * The DFT+U force when the projector is S|phi> rather than |phi>.
     *
     * ── What is being differentiated ──────────────────────────────────────
     *
     * For ultrasoft and PAW the occupation is built from
     *
     *     P_m = <phi_m|S|psi>,   S = 1 + sum_ij q_ij |beta_i><beta_j|
     *
     * so displacing an atom moves three things: the Hubbard orbital itself
     * (only when the displaced atom *is* the Hubbard atom, since these
     * orbitals are localized), and both beta projectors inside S (whenever the
     * displaced atom carries projectors, Hubbard or not). That last part is
     * why this cannot reuse the norm-conserving loop: there, a Hubbard force
     * only ever appears on a Hubbard atom; here, displacing an ordinary atom
     * exerts one too.
     *
     * S|phi> is never built. Expanding it,
     *
     *     P_m = <phi_m|psi> + sum_ij q_ij <phi_m|beta_i><beta_j|psi>
     *
     * needs only overlaps this file already forms plus <phi|beta>, which is
     * cheaper than carrying an augmented projector around and cannot drift
     * from the SCF's own definition.
     *
     * ── Sign convention ───────────────────────────────────────────────────
     *
     * `accumulate_hubbard_force_gradient` is shared with the norm-conserving
     * path, which hands it <phi_m|-i(k+G)|psi>. Writing T[X]_m for
     * <phi_m|-i(k+G)|X>, and using dphi/dtau = -i(k+G)phi (so
     * <dphi_m|X> = -T[X]_m) and the same for beta, the derivative of P is
     *
     *     dP_m/dtau = -delta_a ( T[psi]_m + sum_ij q_ij T[beta_i]_m <beta_j|psi> )
     *               + sum_{ij in alpha} q_ij ( T[beta_i]_m <beta_j|psi>
     *                                          - <phi_m|beta_i> Bd_j )
     *
     * with delta_a = 1 when the displaced atom owns the manifold and
     * Bd_j = <beta_j|-i(k+G)|psi>. What the contraction wants is **-dP/dtau**:
     * that is exactly T[psi]_m when q_ij vanishes, which is what the
     * norm-conserving path passes today. Deriving the sign by reduction to an
     * already-validated path is safer than re-deriving it from the energy.
     *
     * The k in (k+G) cancels between the two halves of dn, so using (k+G)
     * throughout — as the norm-conserving path does — rather than QE's bare G
     * for the orbital term gives the same force.
     */
    [[nodiscard]] std::vector<Vector3D> compute_with_overlap_operator() const {
        int number_of_atoms = crystal_structure_.number_of_atoms();
        std::vector<Vector3D> hubbard_force(number_of_atoms, {0.0, 0.0, 0.0});

        const auto& manifolds = scf_state_.hubbard_manifolds;
        const auto& basis = scf_state_.quantum_system().pseudopotential_basis();
        const auto& pseudopotential_set = basis.pseudopotential_set();
        const auto& ultrasoft_basis =
            static_cast<const UltrasoftPlanewaveKohnShamQuantumSystem&>(scf_state_.quantum_system())
                .ultrasoft_basis();

        int total_projectors = basis.total_number_of_projectors();
        if (total_projectors == 0)
            return hubbard_force;

        constexpr Real q_spacing = 0.01;
        constexpr int number_of_q_points = 900;
        auto atomic_wavefunction_table = tabulate_atomic_wavefunctions_in_q_space(
            pseudopotential_set, crystal_structure_.cell_volume(), q_spacing, number_of_q_points);

        size_t number_of_manifolds = manifolds.size();
        std::vector<int> manifold_row_offsets(number_of_manifolds);
        int total_hubbard_orbitals = 0;
        for (size_t manifold_index = 0; manifold_index < number_of_manifolds; manifold_index++) {
            manifold_row_offsets[manifold_index] = total_hubbard_orbitals;
            total_hubbard_orbitals +=
                static_cast<int>(manifolds[manifold_index].orbital_indices.size());
        }

        /**
         * Which atom each projector belongs to, and the q_ij block of each.
         * q_ij is diagonal in the combined lm index, the same rule
         * `augment_hubbard_projectors_with_overlap` applies when it builds
         * S|phi> during the SCF — the two must agree or the force differentiates
         * a different S than the energy used.
         */
        std::vector<int> projector_atom(total_projectors, 0);
        std::vector<int> projector_offset_by_atom(number_of_atoms, 0);
        std::vector<int> channels_by_atom(number_of_atoms, 0);
        {
            int offset = 0;
            for (int atom_index = 0; atom_index < number_of_atoms; atom_index++) {
                int species_index = crystal_structure_.species_type_index(atom_index);
                int number_of_channels = basis.number_of_projector_channels(species_index);
                projector_offset_by_atom[atom_index] = offset;
                channels_by_atom[atom_index] = number_of_channels;
                for (int channel = 0; channel < number_of_channels; channel++)
                    projector_atom[offset + channel] = atom_index;
                offset += number_of_channels;
            }
        }

        auto augmentation_charge = [&](int atom_index, int channel_i, int channel_j) -> Real {
            return hubbard_augmentation_overlap(ultrasoft_basis,
                                                basis,
                                                crystal_structure_.species_type_index(atom_index),
                                                channel_i,
                                                channel_j);
        };

        int number_of_kpoints = kpoint_grid_.number_of_kpoints();
        int number_of_bands = scf_state_.number_of_bands();
        int number_of_kpoint_states = number_of_kpoints * scf_state_.number_of_spin_components();
        const Matrix3x3& reciprocal_lattice_vectors =
            crystal_structure_.reciprocal_lattice_vectors();

        for (int kpoint_index_state = 0; kpoint_index_state < number_of_kpoint_states;
             kpoint_index_state++) {
            int spin_index = kpoint_index_state / number_of_kpoints;
            int kpoint_index = kpoint_index_state % number_of_kpoints;
            int number_of_plane_waves = scf_state_.number_of_plane_waves(kpoint_index);
            const auto& plane_wave_indices = scf_state_.g_vector_indices_at_kpoint(kpoint_index);
            Real kpoint_weight = kpoint_grid_[kpoint_index].weight;
            bool is_gamma_only = planewave_basis_.is_gamma_only();
            int g_zero_index = is_gamma_only ? find_g_zero_plane_wave_index(planewave_basis_,
                                                                            plane_wave_indices,
                                                                            number_of_plane_waves)
                                             : -1;

            Vector3D kpoint_cartesian;
            {
                Vector3D kpoint_vector = kpoint_grid_[kpoint_index].coordinates;
                for (int direction = 0; direction < 3; direction++)
                    kpoint_cartesian[direction] =
                        kpoint_vector.x * reciprocal_lattice_vectors[0][direction] +
                        kpoint_vector.y * reciprocal_lattice_vectors[1][direction] +
                        kpoint_vector.z * reciprocal_lattice_vectors[2][direction];
            }

            std::vector<std::vector<Complex>> all_atomic_orbitals;
            build_atomic_orbitals_g_space(kpoint_index,
                                          number_of_plane_waves,
                                          plane_wave_indices,
                                          crystal_structure_,
                                          kpoint_grid_,
                                          planewave_basis_,
                                          pseudopotential_set,
                                          atomic_wavefunction_table,
                                          q_spacing,
                                          number_of_q_points,
                                          all_atomic_orbitals);

            Matrix<Complex> beta_projectors;
            compute_kleinman_bylander_projectors(kpoint_grid_[kpoint_index].coordinates,
                                                 crystal_structure_,
                                                 planewave_basis_,
                                                 pseudopotential_set,
                                                 basis.beta_function_table(),
                                                 plane_wave_indices,
                                                 number_of_plane_waves,
                                                 total_projectors,
                                                 beta_projectors);

            std::vector<std::vector<Complex>> minus_i_k_plus_g(
                3, std::vector<Complex>(number_of_plane_waves));
            for (int g_index = 0; g_index < number_of_plane_waves; g_index++) {
                const auto& g_vector = planewave_basis_.g_vectors()[plane_wave_indices[g_index]];
                for (int direction = 0; direction < 3; direction++)
                    minus_i_k_plus_g[direction][g_index] =
                        Complex(0.0,
                                -(g_vector.g_cart[direction] + kpoint_cartesian[direction]) *
                                    crystal_structure_.two_pi_over_lattice_parameter());
            }

            std::vector<Real> band_occupations(number_of_bands);
            for (int band_index = 0; band_index < number_of_bands; band_index++)
                band_occupations[band_index] =
                    scf_state_.occupation(kpoint_index_state, band_index);

            /**
             * Every overlap this assembly needs, formed once per k-point.
             * Indexed [row * columns + column] throughout.
             *
             * Each is a partial sum under band (G-vector) parallelism and is
             * reduced before use — the same reason the occupation matrix and
             * S|phi> are reduced at their own G-sums rather than afterwards.
             *
             * Gamma-only sums half the G sphere, and every operand here is real
             * in real space, so each contraction is completed to its full-sphere
             * (and real) value on the spot. `modifier` decides the G = 0 rule:
             * a plain overlap has G = 0 in the half sphere and the doubling
             * counts it twice, while a modified one carries a factor of
             * (k+G)_alpha = G_alpha that vanishes there. Every consumer below
             * multiplies these together, so completing them individually is what
             * makes the products right.
             */
            auto contract = [&](const Complex* left_rows,
                                int number_of_left,
                                size_t left_stride,
                                const Complex* right_columns,
                                int number_of_right,
                                size_t right_stride,
                                const Complex* modifier) {
                std::vector<Complex> result(static_cast<size_t>(number_of_left) * number_of_right,
                                            Complex(0.0, 0.0));
                for (int left = 0; left < number_of_left; left++) {
                    const Complex* left_row = left_rows + static_cast<size_t>(left) * left_stride;
                    for (int right = 0; right < number_of_right; right++) {
                        const Complex* right_column =
                            right_columns + static_cast<size_t>(right) * right_stride;
                        Complex accumulated(0.0, 0.0);
                        for (int g_index = 0; g_index < number_of_plane_waves; g_index++) {
                            Complex value = std::conj(left_row[g_index]) * right_column[g_index];
                            accumulated += modifier ? value * modifier[g_index] : value;
                        }
                        if (is_gamma_only) {
                            Real g_zero_product = (modifier == nullptr && g_zero_index >= 0)
                                                      ? (std::conj(left_row[g_zero_index]) *
                                                         right_column[g_zero_index])
                                                            .real()
                                                      : 0.0;
                            accumulated =
                                gamma_only_full_sphere_overlap(accumulated, g_zero_product);
                        }
                        result[static_cast<size_t>(left) * number_of_right + right] = accumulated;
                    }
                }
                if (scf_state_.execution_policy != nullptr)
                    scf_state_.execution_policy->reduce_within_group(
                        result.data(), static_cast<int>(result.size()));
                return result;
            };

            // Hubbard orbitals and wavefunctions packed contiguously per row/column.
            std::vector<Complex> orbitals(static_cast<size_t>(total_hubbard_orbitals) *
                                          number_of_plane_waves);
            for (size_t manifold_index = 0; manifold_index < number_of_manifolds;
                 manifold_index++) {
                const auto& manifold = manifolds[manifold_index];
                int row_offset = manifold_row_offsets[manifold_index];
                for (size_t m = 0; m < manifold.orbital_indices.size(); m++) {
                    const auto& orbital =
                        all_atomic_orbitals[static_cast<size_t>(manifold.orbital_indices[m])];
                    std::copy(orbital.begin(),
                              orbital.begin() + number_of_plane_waves,
                              orbitals.begin() +
                                  (static_cast<size_t>(row_offset) + m) * number_of_plane_waves);
                }
            }
            std::vector<Complex> wavefunctions(static_cast<size_t>(number_of_bands) *
                                               number_of_plane_waves);
            for (int band_index = 0; band_index < number_of_bands; band_index++)
                for (int g_index = 0; g_index < number_of_plane_waves; g_index++)
                    wavefunctions[static_cast<size_t>(band_index) * number_of_plane_waves +
                                  g_index] = scf_state_.wavefunction_coefficient(kpoint_index_state,
                                                                                 band_index,
                                                                                 g_index);

            auto phi_psi = contract(orbitals.data(),
                                    total_hubbard_orbitals,
                                    number_of_plane_waves,
                                    wavefunctions.data(),
                                    number_of_bands,
                                    number_of_plane_waves,
                                    nullptr);
            auto phi_beta = contract(orbitals.data(),
                                     total_hubbard_orbitals,
                                     number_of_plane_waves,
                                     beta_projectors.data(),
                                     total_projectors,
                                     number_of_plane_waves,
                                     nullptr);
            auto beta_psi = contract(beta_projectors.data(),
                                     total_projectors,
                                     number_of_plane_waves,
                                     wavefunctions.data(),
                                     number_of_bands,
                                     number_of_plane_waves,
                                     nullptr);
            std::vector<std::vector<Complex>> phi_derivative_psi(3), phi_derivative_beta(3),
                beta_derivative_psi(3);
            for (int direction = 0; direction < 3; direction++) {
                phi_derivative_psi[direction] = contract(orbitals.data(),
                                                         total_hubbard_orbitals,
                                                         number_of_plane_waves,
                                                         wavefunctions.data(),
                                                         number_of_bands,
                                                         number_of_plane_waves,
                                                         minus_i_k_plus_g[direction].data());
                phi_derivative_beta[direction] = contract(orbitals.data(),
                                                          total_hubbard_orbitals,
                                                          number_of_plane_waves,
                                                          beta_projectors.data(),
                                                          total_projectors,
                                                          number_of_plane_waves,
                                                          minus_i_k_plus_g[direction].data());
                beta_derivative_psi[direction] = contract(beta_projectors.data(),
                                                          total_projectors,
                                                          number_of_plane_waves,
                                                          wavefunctions.data(),
                                                          number_of_bands,
                                                          number_of_plane_waves,
                                                          minus_i_k_plus_g[direction].data());
            }

            auto orbital_band = [&](int orbital, int band) {
                return static_cast<size_t>(orbital) * number_of_bands + band;
            };
            auto orbital_projector = [&](int orbital, int projector) {
                return static_cast<size_t>(orbital) * total_projectors + projector;
            };
            auto projector_band = [&](int projector, int band) {
                return static_cast<size_t>(projector) * number_of_bands + band;
            };

            /**
             * P_m = <phi_m|S|psi>, and the delta_a bracket -- the part of
             * -dP/dtau that appears only on the manifold's own atom. Both sum
             * q_ij over *every* atom's projectors, because S does.
             */
            std::vector<Complex> projection = phi_psi;
            std::vector<std::vector<Complex>> own_atom_term(3, phi_derivative_psi[0]);
            for (int direction = 0; direction < 3; direction++)
                own_atom_term[direction] = phi_derivative_psi[direction];

            for (int atom_index = 0; atom_index < number_of_atoms; atom_index++) {
                int offset = projector_offset_by_atom[atom_index];
                for (int channel_i = 0; channel_i < channels_by_atom[atom_index]; channel_i++) {
                    for (int channel_j = 0; channel_j < channels_by_atom[atom_index]; channel_j++) {
                        Real q_value = augmentation_charge(atom_index, channel_i, channel_j);
                        if (std::abs(q_value) < math::tolerances::d_matrix_element_threshold)
                            continue;
                        int projector_i = offset + channel_i;
                        int projector_j = offset + channel_j;
                        for (int orbital = 0; orbital < total_hubbard_orbitals; orbital++) {
                            Complex phi_beta_i = phi_beta[orbital_projector(orbital, projector_i)];
                            for (int band = 0; band < number_of_bands; band++) {
                                Complex beta_j_psi = beta_psi[projector_band(projector_j, band)];
                                projection[orbital_band(orbital, band)] +=
                                    q_value * phi_beta_i * beta_j_psi;
                                for (int direction = 0; direction < 3; direction++)
                                    own_atom_term[direction][orbital_band(orbital, band)] +=
                                        q_value *
                                        phi_derivative_beta[direction][orbital_projector(
                                            orbital, projector_i)] *
                                        beta_j_psi;
                            }
                        }
                    }
                }
            }

            Matrix<Complex> overlaps(static_cast<size_t>(number_of_bands),
                                     static_cast<size_t>(total_hubbard_orbitals));
            for (int orbital = 0; orbital < total_hubbard_orbitals; orbital++)
                for (int band = 0; band < number_of_bands; band++)
                    overlaps(static_cast<size_t>(band), static_cast<size_t>(orbital)) =
                        projection[orbital_band(orbital, band)];

            /**
             * One displaced atom at a time. The dS/dtau part is non-zero for
             * every atom carrying projectors, which is exactly the term the
             * norm-conserving path has no analogue of.
             */
            for (int displaced_atom = 0; displaced_atom < number_of_atoms; displaced_atom++) {
                int offset = projector_offset_by_atom[displaced_atom];
                int number_of_channels = channels_by_atom[displaced_atom];

                for (int direction = 0; direction < 3; direction++) {
                    std::vector<Complex> derivative(static_cast<size_t>(total_hubbard_orbitals) *
                                                        number_of_bands,
                                                    Complex(0.0, 0.0));

                    for (int channel_i = 0; channel_i < number_of_channels; channel_i++) {
                        for (int channel_j = 0; channel_j < number_of_channels; channel_j++) {
                            Real q_value =
                                augmentation_charge(displaced_atom, channel_i, channel_j);
                            if (std::abs(q_value) < math::tolerances::d_matrix_element_threshold)
                                continue;
                            int projector_i = offset + channel_i;
                            int projector_j = offset + channel_j;
                            for (int orbital = 0; orbital < total_hubbard_orbitals; orbital++) {
                                Complex phi_beta_i =
                                    phi_beta[orbital_projector(orbital, projector_i)];
                                Complex phi_derivative_beta_i =
                                    phi_derivative_beta[direction]
                                                       [orbital_projector(orbital, projector_i)];
                                for (int band = 0; band < number_of_bands; band++) {
                                    Complex beta_j_psi =
                                        beta_psi[projector_band(projector_j, band)];
                                    Complex beta_j_derivative_psi =
                                        beta_derivative_psi[direction]
                                                           [projector_band(projector_j, band)];
                                    derivative[orbital_band(orbital, band)] -=
                                        q_value * (phi_derivative_beta_i * beta_j_psi -
                                                   phi_beta_i * beta_j_derivative_psi);
                                }
                            }
                        }
                    }

                    for (size_t manifold_index = 0; manifold_index < number_of_manifolds;
                         manifold_index++) {
                        const auto& manifold = manifolds[manifold_index];
                        int manifold_size = static_cast<int>(manifold.orbital_indices.size());
                        int row_offset = manifold_row_offsets[manifold_index];
                        bool displaced_owns_manifold = manifold.atom_index == displaced_atom;

                        Matrix<Complex> derivative_overlaps(static_cast<size_t>(number_of_bands),
                                                            static_cast<size_t>(manifold_size));
                        for (int m = 0; m < manifold_size; m++) {
                            int orbital = row_offset + m;
                            for (int band = 0; band < number_of_bands; band++) {
                                Complex value = derivative[orbital_band(orbital, band)];
                                if (displaced_owns_manifold)
                                    value += own_atom_term[direction][orbital_band(orbital, band)];
                                derivative_overlaps(static_cast<size_t>(band),
                                                    static_cast<size_t>(m)) = value;
                            }
                        }

                        const auto& combined_d_matrix =
                            scf_state_.hubbard_d_matrix_per_spin[spin_index];
                        std::vector<Complex> manifold_d_matrix(static_cast<size_t>(manifold_size) *
                                                               manifold_size);
                        for (int m = 0; m < manifold_size; m++)
                            for (int m_prime = 0; m_prime < manifold_size; m_prime++)
                                manifold_d_matrix[static_cast<size_t>(m * manifold_size +
                                                                      m_prime)] =
                                    combined_d_matrix[static_cast<size_t>(row_offset + m) +
                                                      static_cast<size_t>(row_offset + m_prime) *
                                                          total_hubbard_orbitals];

                        Matrix<Complex> manifold_overlaps(static_cast<size_t>(number_of_bands),
                                                          static_cast<size_t>(manifold_size));
                        for (int m = 0; m < manifold_size; m++)
                            for (int band = 0; band < number_of_bands; band++)
                                manifold_overlaps(static_cast<size_t>(band),
                                                  static_cast<size_t>(m)) =
                                    overlaps(static_cast<size_t>(band),
                                             static_cast<size_t>(row_offset + m));

                        hubbard_force[displaced_atom][direction] +=
                            accumulate_hubbard_force_gradient(number_of_bands,
                                                              manifold_size,
                                                              manifold_overlaps,
                                                              derivative_overlaps,
                                                              band_occupations,
                                                              kpoint_weight,
                                                              manifold_d_matrix);
                    }
                }
            }
        }

        return hubbard_force;
    }

    const Crystal& crystal_structure_;
    const PlanewaveBasis& planewave_basis_;
    const KPointsGrid& kpoint_grid_;
    const NormConservingPlanewaveKohnShamSCFState& scf_state_;
};

} // namespace q3
