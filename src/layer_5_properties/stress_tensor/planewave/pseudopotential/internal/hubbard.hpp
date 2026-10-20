#pragma once

/**
 * DFT+U (Hubbard) stress tensor contribution, in the rotationally invariant simplified
 * form of Dudarev et al., Phys. Rev. B 57, 1505 (1998).
 *
 * The strain analogue of HubbardForceCalculator
 * (forces/planewave/pseudopotential/internal/hubbard.hpp): the same D-matrix-weighted
 * contraction of dn/d(parameter), with the parameter being the strain rather than an
 * atomic position. E_U = (U/2)Tr[n(1-n)] depends on strain only through the projection
 *
 *     P_m = <phi_m|S|psi_v>,   S = 1 + sum_ij q_ij |beta_i><beta_j|
 *
 * with psi and the occupations held fixed, so
 *
 *     dP_m/deps_ab = <dphi_m/deps_ab|S|psi> + <phi_m|dS/deps_ab|psi>
 *     dS/deps_ab   = sum_ij q_ij ( |dbeta_i/deps_ab><beta_j| + |beta_i><dbeta_j/deps_ab| )
 *
 * q_ij = int Q_ij(r)dr is a pure number and carries no strain derivative. Both phi and
 * beta have the form A(k+G) = f(|k+G|) Y_lm(k+G) S_I(k+G) i^l / sqrt(Omega), whose strain
 * derivative is
 *
 *     dA/deps_ab = -[ f'(q) (k+G)_a (k+G)_b / q  Y_lm  +  f(q) dY_lm/dq_a (k+G)_b ]
 *                  - delta_ab A / 2
 *
 * the last term from 1/sqrt(Omega) (dOmega/deps_ab = Omega delta_ab), and the structure
 * factor contributing nothing because (k+G).tau is strain-invariant. Quantum ESPRESSO's
 * Hubbard stress builds the same object from the same three pieces: an angular term
 * weighted by -(k+G)_b, a radial term weighted by -(k+G)_a (k+G)_b/|k+G|, and -1/2 times
 * the orbital itself on the diagonal (a == b).
 *
 * ── Sign and normalization ────────────────────────────────────────────────
 *
 * Pinned by reduction to the already-validated nonlocal stress rather than re-derived
 * from the energy. NonlocalPseudopotentialStressTensorCalculator (internal/nonlocal.hpp)
 * accumulates, for E_nl = sum_kv w sum_ij D_ij <psi|beta_i><beta_j|psi>,
 *
 *     sigma_ab = (1/Omega) sum D_ij [ 2Re(conj(becp_i)(bessel_j + ylm_j))
 *                                     + delta_ab Re(conj(becp_i) becp_j) ]
 *
 * which is identically -(1/Omega) dE_nl/deps_ab once dbeta/deps is substituted in the
 * form above -- the factor 2 on the derivative overlaps and the factor 1 on the diagonal
 * are what the -[bessel + ylm] - delta_ab/2 split produces. So Q3's convention is
 * sigma = -(1/Omega) dE/deps, the same sign and normalization Quantum ESPRESSO reports
 * for its own Hubbard stress tensor, and this calculator writes the Hubbard term the
 * same way: form dP/deps, hand it to the
 * shared `accumulate_hubbard_force_gradient` -- which returns
 * sum_v w f 2Re[sum_mm' D_mm' conj(X_m) P_m'], i.e. dE_U/dlambda when X = dP/dlambda --
 * and multiply by -1/Omega. Setting q_ij = 0 collapses dP to <dphi/deps|psi> and the
 * whole expression to the norm-conserving limit, where dS/deps vanishes identically.
 *
 * ── Time reversal ─────────────────────────────────────────────────────────
 *
 * `accumulate_hubbard_force_gradient` contracts 2*Re[...], which *is* the time-reversal
 * average of dn, matching how the occupation matrix itself is built
 * (symmetrize_hubbard_occupation_matrices_under_time_reversal). Using it rather than a
 * private contraction is what keeps dn/deps consistent with n, the inconsistency that
 * produced a 0.3% force error on the symmetry-reduced wedge (KNOWN-ISSUES.md,
 * 2026-08-09).
 *
 * The full 3x3 strain tensor is accumulated, not just its upper triangle. Of the three
 * pieces above only the dY_lm/dq_a (k+G)_b term is asymmetric in (a,b), and summed over
 * a symmetry-reduced k-set that asymmetry survives, so sigma_ab and sigma_ba are
 * genuinely different numbers whose average is the stress. `symmetrize_stress_tensor`
 * therefore averages rather than mirrors, cancelling the antisymmetric half instead of
 * substituting one triangle for it -- unlike the nonlocal calculator's same-named
 * helper, which fills the upper triangle only and mirrors it down.
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

#include "layer_0_core/math/interpolation/lagrange.hpp"
#include "layer_0_core/math/phase_factor.hpp"
#include "layer_0_core/math/spherical/harmonics.hpp"
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
#include "layer_5_properties/stress_tensor/planewave/contributions.hpp"

namespace q3 {

class HubbardStressTensorCalculator {
  public:
    HubbardStressTensorCalculator(const Crystal& crystal_structure,
                                  const PlanewaveBasis& planewave_basis,
                                  const KPointsGrid& kpoint_grid,
                                  const NormConservingPlanewaveKohnShamSCFState& scf_state,
                                  StressTensorContributions& result)
        : crystal_structure_(crystal_structure)
        , planewave_basis_(planewave_basis)
        , kpoint_grid_(kpoint_grid)
        , scf_state_(scf_state)
        , result_(result) {}

    /**
     * Whether a Hubbard stress term exists for this run's projector and dataset choice.
     *
     * Ortho-atomic projectors are a joint, system-wide Lowdin transform of every atomic
     * orbital in the cell, so their strain derivative carries a d(O^-1/2)/deps term over
     * orbitals belonging to no Hubbard manifold at all. The norm-conserving path evaluates
     * that term with `lowdin_inverse_square_root_derivative`. Ultrasoft and PAW use the same
     * matrix-function derivative with the generalized <phi|S|phi> metric and its complete
     * orbital-plus-overlap strain derivative.
     */
    [[nodiscard]] static bool supports(const NormConservingPlanewaveKohnShamSCFState& scf_state) {
        return scf_state.hubbard_manifolds.empty() ||
               scf_state.hubbard_projector_type == "atomic" ||
               scf_state.hubbard_projector_type == "ortho-atomic";
    }

    void compute() {
        const auto& manifolds = scf_state_.hubbard_manifolds;
        if (manifolds.empty() || !supports(scf_state_))
            return;
        if (scf_state_.hubbard_projector_type == "ortho-atomic") {
            compute_with_orthogonal_projectors();
            return;
        }

        const auto& basis = scf_state_.quantum_system().pseudopotential_basis();
        const auto& pseudopotential_set = basis.pseudopotential_set();

        /**
         * dS/deps is identically zero without augmentation charges, so the whole
         * beta-projector half of the assembly is switched off by giving it zero
         * projectors rather than by branching through it. `ultrasoft_basis()` is only
         * reachable once has_ultrasoft() is true.
         */
        bool has_overlap_operator = scf_state_.quantum_system().has_ultrasoft();
        int total_projectors = has_overlap_operator ? basis.total_number_of_projectors() : 0;

        constexpr Real q_spacing = 0.01;
        constexpr int number_of_q_points = 900;
        auto atomic_wavefunction_table = tabulate_atomic_wavefunctions_in_q_space(
            pseudopotential_set, crystal_structure_.cell_volume(), q_spacing, number_of_q_points);

        auto orbital_descriptors = enumerate_atomic_orbital_descriptors();

        size_t number_of_manifolds = manifolds.size();
        std::vector<int> manifold_row_offsets(number_of_manifolds);
        int total_hubbard_orbitals = 0;
        for (size_t manifold_index = 0; manifold_index < number_of_manifolds; manifold_index++) {
            manifold_row_offsets[manifold_index] = total_hubbard_orbitals;
            total_hubbard_orbitals +=
                static_cast<int>(manifolds[manifold_index].orbital_indices.size());
        }
        if (total_hubbard_orbitals == 0)
            return;

        // The descriptor of each packed Hubbard orbital, in the packed order.
        std::vector<AtomicOrbitalDescriptor> hubbard_descriptors(
            static_cast<size_t>(total_hubbard_orbitals));
        for (size_t manifold_index = 0; manifold_index < number_of_manifolds; manifold_index++) {
            const auto& manifold = manifolds[manifold_index];
            int row_offset = manifold_row_offsets[manifold_index];
            for (size_t m = 0; m < manifold.orbital_indices.size(); m++)
                hubbard_descriptors[static_cast<size_t>(row_offset) + m] =
                    orbital_descriptors[static_cast<size_t>(manifold.orbital_indices[m])];
        }

        int number_of_atoms = crystal_structure_.number_of_atoms();
        std::vector<int> projector_offset_by_atom(number_of_atoms, 0);
        std::vector<int> channels_by_atom(number_of_atoms, 0);
        // Left all-zero for norm-conserving, which makes every q_ij loop below empty.
        if (total_projectors > 0) {
            int offset = 0;
            for (int atom_index = 0; atom_index < number_of_atoms; atom_index++) {
                int species_index = crystal_structure_.species_type_index(atom_index);
                int number_of_channels = basis.number_of_projector_channels(species_index);
                projector_offset_by_atom[atom_index] = offset;
                channels_by_atom[atom_index] = number_of_channels;
                offset += number_of_channels;
            }
        }

        auto augmentation_charge = [&](int atom_index, int channel_i, int channel_j) -> Real {
            return hubbard_augmentation_overlap(
                static_cast<const UltrasoftPlanewaveKohnShamQuantumSystem&>(
                    scf_state_.quantum_system())
                    .ultrasoft_basis(),
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
        Real inverse_cell_volume = 1.0 / crystal_structure_.cell_volume();

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

            PlaneWaveGeometry geometry = build_plane_wave_geometry(
                kpoint_cartesian, plane_wave_indices, number_of_plane_waves);

            std::vector<Real> band_occupations(number_of_bands);
            std::vector<int> occupied_band_indices;
            for (int band_index = 0; band_index < number_of_bands; band_index++) {
                band_occupations[band_index] =
                    scf_state_.occupation(kpoint_index_state, band_index);
                if (band_occupations[band_index] >= math::tolerances::occupation_threshold)
                    occupied_band_indices.push_back(band_index);
            }
            int number_of_occupied_bands = static_cast<int>(occupied_band_indices.size());
            if (number_of_occupied_bands == 0)
                continue;

            // Column-major (number_of_plane_waves x N) operands, the layout the ZGEMM wants.
            std::vector<Complex> wavefunctions(static_cast<size_t>(number_of_plane_waves) *
                                               number_of_occupied_bands);
            for (int occupied_index = 0; occupied_index < number_of_occupied_bands;
                 occupied_index++)
                for (int g_index = 0; g_index < number_of_plane_waves; g_index++)
                    wavefunctions[static_cast<size_t>(occupied_index) * number_of_plane_waves +
                                  g_index] =
                        scf_state_.wavefunction_coefficient(
                            kpoint_index_state,
                            occupied_band_indices[static_cast<size_t>(occupied_index)],
                            g_index);

            std::vector<Complex> orbitals;
            std::vector<Complex> orbital_radial_derivative;
            std::array<std::vector<Complex>, 3> orbital_angular_derivative;
            build_atomic_orbital_strain_parts(kpoint_index,
                                              plane_wave_indices,
                                              number_of_plane_waves,
                                              geometry,
                                              hubbard_descriptors,
                                              atomic_wavefunction_table,
                                              q_spacing,
                                              number_of_q_points,
                                              orbitals,
                                              orbital_radial_derivative,
                                              orbital_angular_derivative);

            std::vector<Complex> beta_projectors;
            std::vector<Complex> beta_radial_derivative;
            std::array<std::vector<Complex>, 3> beta_angular_derivative;
            build_beta_strain_parts(kpoint_index,
                                    plane_wave_indices,
                                    number_of_plane_waves,
                                    total_projectors,
                                    geometry,
                                    beta_projectors,
                                    beta_radial_derivative,
                                    beta_angular_derivative);

            auto contract = [&](const Complex* left,
                                int number_of_left,
                                const Complex* right,
                                int number_of_right) {
                std::vector<Complex> result;
                compute_hubbard_projector_overlaps(
                    number_of_plane_waves, number_of_left, number_of_right, left, right, result);
                if (is_gamma_only)
                    complete_gamma_only_overlaps(number_of_plane_waves,
                                                 number_of_left,
                                                 number_of_right,
                                                 left,
                                                 right,
                                                 g_zero_index,
                                                 result);
                if (scf_state_.execution_policy != nullptr)
                    scf_state_.execution_policy->reduce_within_group(
                        result.data(), static_cast<int>(result.size()));
                return result;
            };

            // Column-major throughout: element (left, right) lives at left + right*number_of_left.
            auto orbital_band = [&](int orbital, int band) {
                return static_cast<size_t>(orbital) +
                       static_cast<size_t>(band) * total_hubbard_orbitals;
            };
            auto orbital_projector = [&](int orbital, int projector) {
                return static_cast<size_t>(orbital) +
                       static_cast<size_t>(projector) * total_hubbard_orbitals;
            };
            auto projector_band = [&](int projector, int band) {
                return static_cast<size_t>(projector) +
                       static_cast<size_t>(band) * total_projectors;
            };

            auto phi_psi = contract(orbitals.data(),
                                    total_hubbard_orbitals,
                                    wavefunctions.data(),
                                    number_of_occupied_bands);
            auto phi_beta = contract(
                orbitals.data(), total_hubbard_orbitals, beta_projectors.data(), total_projectors);
            auto beta_psi = contract(beta_projectors.data(),
                                     total_projectors,
                                     wavefunctions.data(),
                                     number_of_occupied_bands);

            // P_m = <phi_m|S|psi_v>, exactly the projection the occupation matrix is built from.
            std::vector<Complex> projection = phi_psi;
            accumulate_augmentation(number_of_atoms,
                                    channels_by_atom,
                                    projector_offset_by_atom,
                                    augmentation_charge,
                                    total_hubbard_orbitals,
                                    number_of_occupied_bands,
                                    phi_beta,
                                    beta_psi,
                                    orbital_projector,
                                    projector_band,
                                    orbital_band,
                                    projection);

            std::vector<Complex> strained_orbitals(orbitals.size());
            std::vector<Complex> strained_betas(beta_projectors.size());

            for (int direction_a = 0; direction_a < 3; direction_a++) {
                for (int direction_b = 0; direction_b < 3; direction_b++) {
                    apply_strain_derivative(number_of_plane_waves,
                                            total_hubbard_orbitals,
                                            geometry,
                                            direction_a,
                                            direction_b,
                                            orbitals,
                                            orbital_radial_derivative,
                                            orbital_angular_derivative[direction_a],
                                            strained_orbitals);
                    apply_strain_derivative(number_of_plane_waves,
                                            total_projectors,
                                            geometry,
                                            direction_a,
                                            direction_b,
                                            beta_projectors,
                                            beta_radial_derivative,
                                            beta_angular_derivative[direction_a],
                                            strained_betas);

                    /**
                     * dP_m/deps = <dphi_m|S|psi> + <phi_m|dS/deps|psi>, the same two
                     * pieces Quantum ESPRESSO assembles separately and adds.
                     * <dphi_m|S|psi> is expanded rather than built through an explicit
                     * S|psi>, exactly as the force does, so it cannot drift from the
                     * SCF's own definition of S.
                     */
                    auto derivative_projection = contract(strained_orbitals.data(),
                                                          total_hubbard_orbitals,
                                                          wavefunctions.data(),
                                                          number_of_occupied_bands);
                    auto strained_orbital_beta = contract(strained_orbitals.data(),
                                                          total_hubbard_orbitals,
                                                          beta_projectors.data(),
                                                          total_projectors);
                    auto orbital_strained_beta = contract(orbitals.data(),
                                                          total_hubbard_orbitals,
                                                          strained_betas.data(),
                                                          total_projectors);
                    auto strained_beta_psi = contract(strained_betas.data(),
                                                      total_projectors,
                                                      wavefunctions.data(),
                                                      number_of_occupied_bands);

                    accumulate_augmentation(number_of_atoms,
                                            channels_by_atom,
                                            projector_offset_by_atom,
                                            augmentation_charge,
                                            total_hubbard_orbitals,
                                            number_of_occupied_bands,
                                            strained_orbital_beta,
                                            beta_psi,
                                            orbital_projector,
                                            projector_band,
                                            orbital_band,
                                            derivative_projection);
                    accumulate_augmentation(number_of_atoms,
                                            channels_by_atom,
                                            projector_offset_by_atom,
                                            augmentation_charge,
                                            total_hubbard_orbitals,
                                            number_of_occupied_bands,
                                            orbital_strained_beta,
                                            beta_psi,
                                            orbital_projector,
                                            projector_band,
                                            orbital_band,
                                            derivative_projection);
                    accumulate_augmentation(number_of_atoms,
                                            channels_by_atom,
                                            projector_offset_by_atom,
                                            augmentation_charge,
                                            total_hubbard_orbitals,
                                            number_of_occupied_bands,
                                            phi_beta,
                                            strained_beta_psi,
                                            orbital_projector,
                                            projector_band,
                                            orbital_band,
                                            derivative_projection);

                    for (size_t manifold_index = 0; manifold_index < number_of_manifolds;
                         manifold_index++) {
                        const auto& manifold = manifolds[manifold_index];
                        int manifold_size = static_cast<int>(manifold.orbital_indices.size());
                        int row_offset = manifold_row_offsets[manifold_index];

                        Matrix<Complex> manifold_overlaps(static_cast<size_t>(number_of_bands),
                                                          static_cast<size_t>(manifold_size));
                        Matrix<Complex> derivative_overlaps(static_cast<size_t>(number_of_bands),
                                                            static_cast<size_t>(manifold_size));
                        for (int m = 0; m < manifold_size; m++)
                            for (int occupied_index = 0; occupied_index < number_of_occupied_bands;
                                 occupied_index++) {
                                size_t band_row = static_cast<size_t>(
                                    occupied_band_indices[static_cast<size_t>(occupied_index)]);
                                manifold_overlaps(band_row, static_cast<size_t>(m)) =
                                    projection[orbital_band(row_offset + m, occupied_index)];
                                derivative_overlaps(band_row, static_cast<size_t>(m)) =
                                    derivative_projection[orbital_band(row_offset + m,
                                                                       occupied_index)];
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

                        /**
                         * The accumulator returns dE_U/dlambda for a projection derivative
                         * handed to it; sigma = -(1/Omega) dE/deps.
                         */
                        result_.hubbard[direction_a][direction_b] -=
                            inverse_cell_volume *
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

        symmetrize_stress_tensor(result_.hubbard);
    }

  private:
    /// One atomic orbital's place in the pseudopotential tables and in the cell.
    struct AtomicOrbitalDescriptor {
        int global_orbital_index{}; // row of build_atomic_orbitals_g_space' output
        int atom_index{};
        int species_type_index{};
        int wavefunction_index{}; // row of tabulate_atomic_wavefunctions_in_q_space
        int angular_momentum{};
        int combined_lm_index{}; // l*l + m, the Y_lm storage order
    };

    /// Per-G geometry shared by every radial and angular derivative below.
    struct PlaneWaveGeometry {
        std::vector<std::array<Real, 3>> reciprocal_units;    ///< k+G in units of 2pi/a
        std::vector<Real> magnitude_squared_reciprocal_units; ///< |k+G|^2 in units of (2pi/a)^2
        std::vector<std::array<Real, 3>> cartesian;           ///< k+G in bohr^-1
        std::vector<Real> magnitude_bohr;                     ///< |k+G| in bohr^-1
        std::vector<Real> inverse_magnitude_bohr;             ///< 1/|k+G|, zero at k+G = 0
        std::vector<std::vector<Real>> spherical_harmonics;   ///< [g][lm]
        std::array<std::vector<std::vector<Real>>, 3>
            spherical_harmonic_derivatives;                  ///< [a][g][lm]
        std::vector<std::vector<Complex>> structure_factors; ///< [atom][g], exp(i(k+G).tau)
    };

    /**
     * Compute ortho-atomic Hubbard stress.
     *
     * The SCF projector is the system-wide Lowdin transform
     *
     *     |orthogonal_I> = sum_J |atomic_J> (O^-1/2)_JI,
     *     O_IJ = <atomic_I|S|atomic_J>.
     *
     * A homogeneous strain moves every atomic orbital, so both terms in
     *
     *     dP_I = sum_J d(O^-1/2)_IJ <atomic_J|psi>
     *          + sum_J (O^-1/2)_IJ <d atomic_J|psi>
     *
     * run over the complete cell-wide orbital set. The overlap derivative is assembled from
     * the same analytic orbital derivatives used by the atomic stress path, reduced before
     * diagonalization, and handed to the shared Lowdin matrix-function derivative. Only after
     * the complete derivative is formed are the Hubbard-manifold rows selected.
     */
    void compute_with_orthogonal_projectors() {
        const auto& manifolds = scf_state_.hubbard_manifolds;
        const auto& pseudopotential_basis = scf_state_.quantum_system().pseudopotential_basis();
        const auto& pseudopotential_set = pseudopotential_basis.pseudopotential_set();
        bool has_augmented_overlap = scf_state_.quantum_system().has_ultrasoft();

        constexpr Real q_spacing = 0.01;
        constexpr int number_of_q_points = 900;
        auto atomic_wavefunction_table = tabulate_atomic_wavefunctions_in_q_space(
            pseudopotential_set, crystal_structure_.cell_volume(), q_spacing, number_of_q_points);

        auto atomic_orbital_descriptors = enumerate_atomic_orbital_descriptors();
        int number_of_atomic_orbitals = static_cast<int>(atomic_orbital_descriptors.size());

        size_t number_of_manifolds = manifolds.size();
        std::vector<int> manifold_row_offsets(number_of_manifolds);
        int total_hubbard_orbitals = 0;
        for (size_t manifold_index = 0; manifold_index < number_of_manifolds; manifold_index++) {
            manifold_row_offsets[manifold_index] = total_hubbard_orbitals;
            total_hubbard_orbitals +=
                static_cast<int>(manifolds[manifold_index].orbital_indices.size());
        }
        if (number_of_atomic_orbitals == 0 || total_hubbard_orbitals == 0)
            return;

        int number_of_atoms = crystal_structure_.number_of_atoms();
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
        auto augmentation_overlap = [&](int atom_index, int channel_row, int channel_column) {
            return hubbard_augmentation_overlap(
                static_cast<const UltrasoftPlanewaveKohnShamQuantumSystem&>(
                    scf_state_.quantum_system())
                    .ultrasoft_basis(),
                pseudopotential_basis,
                crystal_structure_.species_type_index(atom_index),
                channel_row,
                channel_column);
        };

        int number_of_kpoints = kpoint_grid_.number_of_kpoints();
        int number_of_bands = scf_state_.number_of_bands();
        int number_of_kpoint_states = number_of_kpoints * scf_state_.number_of_spin_components();
        const Matrix3x3& reciprocal_lattice_vectors =
            crystal_structure_.reciprocal_lattice_vectors();
        Real inverse_cell_volume = 1.0 / crystal_structure_.cell_volume();

        auto orbital_matrix_element = [number_of_atomic_orbitals](int row, int column) {
            return static_cast<size_t>(row + column * number_of_atomic_orbitals);
        };

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
            Vector3D kpoint_vector = kpoint_grid_[kpoint_index].coordinates;
            for (int direction = 0; direction < 3; direction++)
                kpoint_cartesian[direction] =
                    kpoint_vector.x * reciprocal_lattice_vectors[0][direction] +
                    kpoint_vector.y * reciprocal_lattice_vectors[1][direction] +
                    kpoint_vector.z * reciprocal_lattice_vectors[2][direction];

            PlaneWaveGeometry geometry = build_plane_wave_geometry(
                kpoint_cartesian, plane_wave_indices, number_of_plane_waves);

            std::vector<Real> band_occupations(number_of_bands);
            std::vector<int> occupied_band_indices;
            for (int band_index = 0; band_index < number_of_bands; band_index++) {
                band_occupations[band_index] =
                    scf_state_.occupation(kpoint_index_state, band_index);
                if (band_occupations[band_index] >= math::tolerances::occupation_threshold)
                    occupied_band_indices.push_back(band_index);
            }
            int number_of_occupied_bands = static_cast<int>(occupied_band_indices.size());
            if (number_of_occupied_bands == 0)
                continue;

            std::vector<Complex> wavefunctions(static_cast<size_t>(number_of_plane_waves) *
                                               number_of_occupied_bands);
            for (int occupied_band_index = 0; occupied_band_index < number_of_occupied_bands;
                 occupied_band_index++)
                for (int g_index = 0; g_index < number_of_plane_waves; g_index++)
                    wavefunctions[static_cast<size_t>(occupied_band_index) * number_of_plane_waves +
                                  g_index] =
                        scf_state_.wavefunction_coefficient(
                            kpoint_index_state,
                            occupied_band_indices[static_cast<size_t>(occupied_band_index)],
                            g_index);

            std::vector<Complex> atomic_orbitals;
            std::vector<Complex> atomic_orbital_radial_derivative;
            std::array<std::vector<Complex>, 3> atomic_orbital_angular_derivative;
            build_atomic_orbital_strain_parts(kpoint_index,
                                              plane_wave_indices,
                                              number_of_plane_waves,
                                              geometry,
                                              atomic_orbital_descriptors,
                                              atomic_wavefunction_table,
                                              q_spacing,
                                              number_of_q_points,
                                              atomic_orbitals,
                                              atomic_orbital_radial_derivative,
                                              atomic_orbital_angular_derivative);

            std::vector<Complex> beta_projectors;
            std::vector<Complex> beta_radial_derivative;
            std::array<std::vector<Complex>, 3> beta_angular_derivative;
            if (has_augmented_overlap) {
                build_beta_strain_parts(kpoint_index,
                                        plane_wave_indices,
                                        number_of_plane_waves,
                                        total_projectors,
                                        geometry,
                                        beta_projectors,
                                        beta_radial_derivative,
                                        beta_angular_derivative);
            }

            auto contract = [&](const Complex* left,
                                int number_of_left,
                                const Complex* right,
                                int number_of_right) {
                std::vector<Complex> contraction;
                compute_hubbard_projector_overlaps(number_of_plane_waves,
                                                   number_of_left,
                                                   number_of_right,
                                                   left,
                                                   right,
                                                   contraction);
                if (is_gamma_only)
                    complete_gamma_only_overlaps(number_of_plane_waves,
                                                 number_of_left,
                                                 number_of_right,
                                                 left,
                                                 right,
                                                 g_zero_index,
                                                 contraction);
                if (scf_state_.execution_policy != nullptr)
                    scf_state_.execution_policy->reduce_within_group(
                        contraction.data(), static_cast<int>(contraction.size()));
                return contraction;
            };

            if (has_augmented_overlap) {
                if (scf_state_.execution_policy == nullptr) {
                    throw std::runtime_error(
                        "Ortho-atomic ultrasoft Hubbard stress requires an execution policy");
                }

                Matrix<Complex> canonical_beta_projectors(
                    static_cast<size_t>(total_projectors),
                    static_cast<size_t>(number_of_plane_waves));
                std::copy(beta_projectors.begin(),
                          beta_projectors.end(),
                          canonical_beta_projectors.data());
                auto overlap_operator = create_overlap_operator(crystal_structure_,
                                                                scf_state_.quantum_system(),
                                                                canonical_beta_projectors,
                                                                *scf_state_.execution_policy);

                std::vector<Complex> overlap_applied_orbitals(atomic_orbitals.size());
                overlap_operator->apply_batch(number_of_plane_waves,
                                              number_of_atomic_orbitals,
                                              atomic_orbitals.data(),
                                              overlap_applied_orbitals.data());
                std::vector<Complex> overlap_applied_wavefunctions(wavefunctions.size());
                overlap_operator->apply_batch(number_of_plane_waves,
                                              number_of_occupied_bands,
                                              wavefunctions.data(),
                                              overlap_applied_wavefunctions.data());

                auto overlap_matrix = contract(atomic_orbitals.data(),
                                               number_of_atomic_orbitals,
                                               overlap_applied_orbitals.data(),
                                               number_of_atomic_orbitals);
                auto lowdin_transform =
                    compute_lowdin_transform(overlap_matrix, number_of_atomic_orbitals);
                auto atomic_band_overlaps = contract(atomic_orbitals.data(),
                                                     number_of_atomic_orbitals,
                                                     overlap_applied_wavefunctions.data(),
                                                     number_of_occupied_bands);

                std::vector<Complex> projection(static_cast<size_t>(number_of_atomic_orbitals) *
                                                    number_of_occupied_bands,
                                                Complex(0.0, 0.0));
                for (int occupied_band_index = 0; occupied_band_index < number_of_occupied_bands;
                     occupied_band_index++) {
                    for (int orbital_index = 0; orbital_index < number_of_atomic_orbitals;
                         orbital_index++) {
                        for (int inner_orbital_index = 0;
                             inner_orbital_index < number_of_atomic_orbitals;
                             inner_orbital_index++) {
                            projection[orbital_matrix_element(orbital_index,
                                                              occupied_band_index)] +=
                                lowdin_transform.inverse_square_root[orbital_matrix_element(
                                    orbital_index, inner_orbital_index)] *
                                atomic_band_overlaps[orbital_matrix_element(inner_orbital_index,
                                                                            occupied_band_index)];
                        }
                    }
                }

                auto apply_overlap_derivative = [&](const Complex* input_vectors,
                                                    int number_of_vectors,
                                                    const std::vector<Complex>&
                                                        strained_beta_projectors) {
                    auto projector_overlaps = contract(
                        beta_projectors.data(), total_projectors, input_vectors, number_of_vectors);
                    auto strained_projector_overlaps = contract(strained_beta_projectors.data(),
                                                                total_projectors,
                                                                input_vectors,
                                                                number_of_vectors);
                    std::vector<Complex> result(static_cast<size_t>(number_of_plane_waves) *
                                                    number_of_vectors,
                                                Complex(0.0, 0.0));
                    for (int atom_index = 0; atom_index < number_of_atoms; atom_index++) {
                        int projector_offset = projector_offset_by_atom[atom_index];
                        int number_of_projector_channels = projector_channels_by_atom[atom_index];
                        for (int channel_row = 0; channel_row < number_of_projector_channels;
                             channel_row++) {
                            int projector_row = projector_offset + channel_row;
                            for (int channel_column = 0;
                                 channel_column < number_of_projector_channels;
                                 channel_column++) {
                                Real overlap_value =
                                    augmentation_overlap(atom_index, channel_row, channel_column);
                                if (std::abs(overlap_value) <
                                    math::tolerances::d_matrix_element_threshold) {
                                    continue;
                                }
                                int projector_column = projector_offset + channel_column;
                                for (int vector_index = 0; vector_index < number_of_vectors;
                                     vector_index++) {
                                    size_t projector_overlap_index =
                                        static_cast<size_t>(projector_column) +
                                        static_cast<size_t>(vector_index) * total_projectors;
                                    Complex strained_projector_coefficient =
                                        overlap_value * projector_overlaps[projector_overlap_index];
                                    Complex projector_coefficient =
                                        overlap_value *
                                        strained_projector_overlaps[projector_overlap_index];
                                    for (int g_index = 0; g_index < number_of_plane_waves;
                                         g_index++) {
                                        size_t result_index = static_cast<size_t>(vector_index) *
                                                                  number_of_plane_waves +
                                                              g_index;
                                        result[result_index] +=
                                            strained_beta_projectors[static_cast<size_t>(
                                                                         projector_row) *
                                                                         number_of_plane_waves +
                                                                     g_index] *
                                                strained_projector_coefficient +
                                            beta_projectors[static_cast<size_t>(projector_row) *
                                                                number_of_plane_waves +
                                                            g_index] *
                                                projector_coefficient;
                                    }
                                }
                            }
                        }
                    }
                    return result;
                };

                std::vector<Complex> strained_atomic_orbitals(atomic_orbitals.size());
                std::vector<Complex> strained_beta_projectors(beta_projectors.size());
                for (int direction_a = 0; direction_a < 3; direction_a++) {
                    for (int direction_b = 0; direction_b < 3; direction_b++) {
                        apply_strain_derivative(number_of_plane_waves,
                                                number_of_atomic_orbitals,
                                                geometry,
                                                direction_a,
                                                direction_b,
                                                atomic_orbitals,
                                                atomic_orbital_radial_derivative,
                                                atomic_orbital_angular_derivative[direction_a],
                                                strained_atomic_orbitals);
                        apply_strain_derivative(number_of_plane_waves,
                                                total_projectors,
                                                geometry,
                                                direction_a,
                                                direction_b,
                                                beta_projectors,
                                                beta_radial_derivative,
                                                beta_angular_derivative[direction_a],
                                                strained_beta_projectors);

                        std::vector<Complex> overlap_applied_strained_orbitals(
                            strained_atomic_orbitals.size());
                        overlap_operator->apply_batch(number_of_plane_waves,
                                                      number_of_atomic_orbitals,
                                                      strained_atomic_orbitals.data(),
                                                      overlap_applied_strained_orbitals.data());
                        auto overlap_derivative_on_orbitals =
                            apply_overlap_derivative(atomic_orbitals.data(),
                                                     number_of_atomic_orbitals,
                                                     strained_beta_projectors);
                        auto overlap_derivative_on_wavefunctions =
                            apply_overlap_derivative(wavefunctions.data(),
                                                     number_of_occupied_bands,
                                                     strained_beta_projectors);

                        auto left_orbital_metric_derivative =
                            contract(strained_atomic_orbitals.data(),
                                     number_of_atomic_orbitals,
                                     overlap_applied_orbitals.data(),
                                     number_of_atomic_orbitals);
                        auto overlap_metric_derivative =
                            contract(atomic_orbitals.data(),
                                     number_of_atomic_orbitals,
                                     overlap_derivative_on_orbitals.data(),
                                     number_of_atomic_orbitals);
                        auto right_orbital_metric_derivative =
                            contract(atomic_orbitals.data(),
                                     number_of_atomic_orbitals,
                                     overlap_applied_strained_orbitals.data(),
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
                                    0.5 *
                                    (metric_derivative[orbital_matrix_element(row, column)] +
                                     std::conj(
                                         metric_derivative[orbital_matrix_element(column, row)]));
                                metric_derivative[orbital_matrix_element(row, column)] =
                                    hermitian_value;
                                metric_derivative[orbital_matrix_element(column, row)] =
                                    std::conj(hermitian_value);
                            }
                        }
                        auto inverse_square_root_derivative = lowdin_inverse_square_root_derivative(
                            lowdin_transform, metric_derivative);

                        auto left_orbital_projection_derivative =
                            contract(strained_atomic_orbitals.data(),
                                     number_of_atomic_orbitals,
                                     overlap_applied_wavefunctions.data(),
                                     number_of_occupied_bands);
                        auto overlap_projection_derivative =
                            contract(atomic_orbitals.data(),
                                     number_of_atomic_orbitals,
                                     overlap_derivative_on_wavefunctions.data(),
                                     number_of_occupied_bands);
                        std::vector<Complex> raw_projection_derivative(atomic_band_overlaps.size());
                        for (size_t overlap_index = 0;
                             overlap_index < raw_projection_derivative.size();
                             overlap_index++) {
                            raw_projection_derivative[overlap_index] =
                                left_orbital_projection_derivative[overlap_index] +
                                overlap_projection_derivative[overlap_index];
                        }

                        std::vector<Complex> projection_derivative(projection.size(),
                                                                   Complex(0.0, 0.0));
                        for (int occupied_band_index = 0;
                             occupied_band_index < number_of_occupied_bands;
                             occupied_band_index++) {
                            for (int orbital_index = 0; orbital_index < number_of_atomic_orbitals;
                                 orbital_index++) {
                                for (int inner_orbital_index = 0;
                                     inner_orbital_index < number_of_atomic_orbitals;
                                     inner_orbital_index++) {
                                    projection_derivative[orbital_matrix_element(
                                        orbital_index, occupied_band_index)] +=
                                        inverse_square_root_derivative[orbital_matrix_element(
                                            orbital_index, inner_orbital_index)] *
                                            atomic_band_overlaps[orbital_matrix_element(
                                                inner_orbital_index, occupied_band_index)] +
                                        lowdin_transform.inverse_square_root[orbital_matrix_element(
                                            orbital_index, inner_orbital_index)] *
                                            raw_projection_derivative[orbital_matrix_element(
                                                inner_orbital_index, occupied_band_index)];
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
                            for (int orbital_within_manifold = 0;
                                 orbital_within_manifold < manifold_size;
                                 orbital_within_manifold++) {
                                int orbital_index = manifold.orbital_indices[static_cast<size_t>(
                                    orbital_within_manifold)];
                                for (int occupied_band_index = 0;
                                     occupied_band_index < number_of_occupied_bands;
                                     occupied_band_index++) {
                                    size_t band_row = static_cast<size_t>(
                                        occupied_band_indices[static_cast<size_t>(
                                            occupied_band_index)]);
                                    manifold_overlaps(
                                        band_row, static_cast<size_t>(orbital_within_manifold)) =
                                        projection[orbital_matrix_element(orbital_index,
                                                                          occupied_band_index)];
                                    derivative_overlaps(
                                        band_row, static_cast<size_t>(orbital_within_manifold)) =
                                        projection_derivative[orbital_matrix_element(
                                            orbital_index, occupied_band_index)];
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

                            result_.hubbard[direction_a][direction_b] -=
                                inverse_cell_volume *
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

            auto overlap_matrix = contract(atomic_orbitals.data(),
                                           number_of_atomic_orbitals,
                                           atomic_orbitals.data(),
                                           number_of_atomic_orbitals);
            auto lowdin_transform =
                compute_lowdin_transform(overlap_matrix, number_of_atomic_orbitals);
            auto atomic_band_overlaps = contract(atomic_orbitals.data(),
                                                 number_of_atomic_orbitals,
                                                 wavefunctions.data(),
                                                 number_of_occupied_bands);

            std::vector<Complex> projection(static_cast<size_t>(number_of_atomic_orbitals) *
                                                number_of_occupied_bands,
                                            Complex(0.0, 0.0));
            for (int occupied_band_index = 0; occupied_band_index < number_of_occupied_bands;
                 occupied_band_index++)
                for (int orbital_index = 0; orbital_index < number_of_atomic_orbitals;
                     orbital_index++)
                    for (int inner_orbital_index = 0;
                         inner_orbital_index < number_of_atomic_orbitals;
                         inner_orbital_index++)
                        projection[orbital_matrix_element(orbital_index, occupied_band_index)] +=
                            lowdin_transform.inverse_square_root[orbital_matrix_element(
                                orbital_index, inner_orbital_index)] *
                            atomic_band_overlaps[orbital_matrix_element(inner_orbital_index,
                                                                        occupied_band_index)];

            std::vector<Complex> strained_atomic_orbitals(atomic_orbitals.size());
            for (int direction_a = 0; direction_a < 3; direction_a++) {
                for (int direction_b = 0; direction_b < 3; direction_b++) {
                    apply_strain_derivative(number_of_plane_waves,
                                            number_of_atomic_orbitals,
                                            geometry,
                                            direction_a,
                                            direction_b,
                                            atomic_orbitals,
                                            atomic_orbital_radial_derivative,
                                            atomic_orbital_angular_derivative[direction_a],
                                            strained_atomic_orbitals);

                    auto atomic_band_derivative = contract(strained_atomic_orbitals.data(),
                                                           number_of_atomic_orbitals,
                                                           wavefunctions.data(),
                                                           number_of_occupied_bands);
                    auto left_overlap_derivative = contract(strained_atomic_orbitals.data(),
                                                            number_of_atomic_orbitals,
                                                            atomic_orbitals.data(),
                                                            number_of_atomic_orbitals);
                    std::vector<Complex> overlap_derivative(
                        static_cast<size_t>(number_of_atomic_orbitals) * number_of_atomic_orbitals);
                    for (int column = 0; column < number_of_atomic_orbitals; column++)
                        for (int row = 0; row < number_of_atomic_orbitals; row++)
                            overlap_derivative[orbital_matrix_element(row, column)] =
                                left_overlap_derivative[orbital_matrix_element(row, column)] +
                                std::conj(
                                    left_overlap_derivative[orbital_matrix_element(column, row)]);

                    auto inverse_square_root_derivative =
                        lowdin_inverse_square_root_derivative(lowdin_transform, overlap_derivative);
                    std::vector<Complex> projection_derivative(
                        static_cast<size_t>(number_of_atomic_orbitals) * number_of_occupied_bands,
                        Complex(0.0, 0.0));
                    for (int occupied_band_index = 0;
                         occupied_band_index < number_of_occupied_bands;
                         occupied_band_index++)
                        for (int orbital_index = 0; orbital_index < number_of_atomic_orbitals;
                             orbital_index++)
                            for (int inner_orbital_index = 0;
                                 inner_orbital_index < number_of_atomic_orbitals;
                                 inner_orbital_index++)
                                projection_derivative[orbital_matrix_element(
                                    orbital_index, occupied_band_index)] +=
                                    inverse_square_root_derivative[orbital_matrix_element(
                                        orbital_index, inner_orbital_index)] *
                                        atomic_band_overlaps[orbital_matrix_element(
                                            inner_orbital_index, occupied_band_index)] +
                                    lowdin_transform.inverse_square_root[orbital_matrix_element(
                                        orbital_index, inner_orbital_index)] *
                                        atomic_band_derivative[orbital_matrix_element(
                                            inner_orbital_index, occupied_band_index)];

                    for (size_t manifold_index = 0; manifold_index < number_of_manifolds;
                         manifold_index++) {
                        const auto& manifold = manifolds[manifold_index];
                        int manifold_size = static_cast<int>(manifold.orbital_indices.size());
                        int row_offset = manifold_row_offsets[manifold_index];
                        Matrix<Complex> manifold_overlaps(static_cast<size_t>(number_of_bands),
                                                          static_cast<size_t>(manifold_size));
                        Matrix<Complex> derivative_overlaps(static_cast<size_t>(number_of_bands),
                                                            static_cast<size_t>(manifold_size));
                        for (int orbital_within_manifold = 0;
                             orbital_within_manifold < manifold_size;
                             orbital_within_manifold++) {
                            int orbital_index =
                                manifold
                                    .orbital_indices[static_cast<size_t>(orbital_within_manifold)];
                            for (int occupied_band_index = 0;
                                 occupied_band_index < number_of_occupied_bands;
                                 occupied_band_index++) {
                                size_t band_row =
                                    static_cast<size_t>(occupied_band_indices[static_cast<size_t>(
                                        occupied_band_index)]);
                                manifold_overlaps(band_row,
                                                  static_cast<size_t>(orbital_within_manifold)) =
                                    projection[orbital_matrix_element(orbital_index,
                                                                      occupied_band_index)];
                                derivative_overlaps(band_row,
                                                    static_cast<size_t>(orbital_within_manifold)) =
                                    projection_derivative[orbital_matrix_element(
                                        orbital_index, occupied_band_index)];
                            }
                        }

                        const auto& combined_d_matrix =
                            scf_state_.hubbard_d_matrix_per_spin[spin_index];
                        std::vector<Complex> manifold_d_matrix(static_cast<size_t>(manifold_size) *
                                                               manifold_size);
                        for (int row = 0; row < manifold_size; row++)
                            for (int column = 0; column < manifold_size; column++)
                                manifold_d_matrix[static_cast<size_t>(row * manifold_size +
                                                                      column)] =
                                    combined_d_matrix[static_cast<size_t>(row_offset + row) +
                                                      static_cast<size_t>(row_offset + column) *
                                                          total_hubbard_orbitals];

                        result_.hubbard[direction_a][direction_b] -=
                            inverse_cell_volume *
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

        symmetrize_stress_tensor(result_.hubbard);
    }

    /**
     * Every atomic orbital's descriptor, in `enumerate_atomic_orbitals`' order -- the order
     * `build_atomic_orbitals_g_space` fills, and the order `HubbardManifold::orbital_indices`
     * indexes into. Walked here rather than taken from `AtomicOrbitalLabel` because the
     * radial table is addressed by the shell's index within the dataset, which the label
     * does not carry.
     */
    [[nodiscard]] std::vector<AtomicOrbitalDescriptor>
    enumerate_atomic_orbital_descriptors() const {
        const auto& pseudopotential_set =
            scf_state_.quantum_system().pseudopotential_basis().pseudopotential_set();
        std::vector<AtomicOrbitalDescriptor> descriptors;
        for (int atom_index = 0; atom_index < crystal_structure_.number_of_atoms(); atom_index++) {
            int species_type_index = crystal_structure_.species_type_index(atom_index);
            const auto& pseudopotential = pseudopotential_set[species_type_index];
            for (int wavefunction_index = 0;
                 wavefunction_index < static_cast<int>(pseudopotential.wavefunctions_atomic.size());
                 wavefunction_index++) {
                const auto& atomic_wavefunction =
                    pseudopotential.wavefunctions_atomic[static_cast<size_t>(wavefunction_index)];
                if (atomic_wavefunction.occupations < 0.0)
                    continue;
                int angular_momentum = atomic_wavefunction.angular_momentum;
                for (int magnetic_index = 0; magnetic_index < 2 * angular_momentum + 1;
                     magnetic_index++)
                    descriptors.push_back({static_cast<int>(descriptors.size()),
                                           atom_index,
                                           species_type_index,
                                           wavefunction_index,
                                           angular_momentum,
                                           angular_momentum * angular_momentum + magnetic_index});
            }
        }
        return descriptors;
    }

    /**
     * k+G geometry, Y_lm, dY_lm/dq and the per-atom structure factors at one k-point.
     *
     * Y_lm and its derivative are evaluated on k+G in units of 2pi/a, which is what
     * `build_atomic_orbitals_g_space` and `compute_kleinman_bylander_projectors` both do;
     * the radial tables are addressed in bohr^-1. `lmax` covers both the atomic shells and
     * the beta channels so one batch serves both.
     */
    [[nodiscard]] PlaneWaveGeometry
    build_plane_wave_geometry(const Vector3D& kpoint_cartesian,
                              const std::vector<int>& plane_wave_indices,
                              int number_of_plane_waves) const {
        const auto& basis = scf_state_.quantum_system().pseudopotential_basis();
        const auto& pseudopotential_set = basis.pseudopotential_set();
        Real two_pi_over_lattice_parameter = crystal_structure_.two_pi_over_lattice_parameter();

        PlaneWaveGeometry geometry;
        geometry.reciprocal_units.resize(number_of_plane_waves);
        geometry.magnitude_squared_reciprocal_units.resize(number_of_plane_waves);
        geometry.cartesian.resize(number_of_plane_waves);
        geometry.magnitude_bohr.resize(number_of_plane_waves);
        geometry.inverse_magnitude_bohr.resize(number_of_plane_waves);

        for (int g_index = 0; g_index < number_of_plane_waves; g_index++) {
            const auto& g_vector =
                planewave_basis_.g_vectors()[plane_wave_indices[static_cast<size_t>(g_index)]];
            std::array<Real, 3> k_plus_g = {kpoint_cartesian.x + g_vector.g_cart.x,
                                            kpoint_cartesian.y + g_vector.g_cart.y,
                                            kpoint_cartesian.z + g_vector.g_cart.z};
            geometry.reciprocal_units[g_index] = k_plus_g;
            Real magnitude_squared =
                k_plus_g[0] * k_plus_g[0] + k_plus_g[1] * k_plus_g[1] + k_plus_g[2] * k_plus_g[2];
            geometry.magnitude_squared_reciprocal_units[g_index] = magnitude_squared;
            for (int direction = 0; direction < 3; direction++)
                geometry.cartesian[g_index][direction] =
                    k_plus_g[direction] * two_pi_over_lattice_parameter;
            Real magnitude_bohr = std::sqrt(magnitude_squared) * two_pi_over_lattice_parameter;
            geometry.magnitude_bohr[g_index] = magnitude_bohr;
            geometry.inverse_magnitude_bohr[g_index] =
                (magnitude_bohr > math::tolerances::bessel_argument_zero_threshold)
                    ? 1.0 / magnitude_bohr
                    : 0.0;
        }

        int maximum_angular_momentum = 0;
        for (int species_index = 0; species_index < static_cast<int>(pseudopotential_set.size());
             species_index++) {
            for (const auto& atomic_wavefunction :
                 pseudopotential_set[species_index].wavefunctions_atomic)
                if (atomic_wavefunction.occupations >= 0.0)
                    maximum_angular_momentum =
                        std::max(maximum_angular_momentum, atomic_wavefunction.angular_momentum);
            for (const auto& projector : pseudopotential_set[species_index].projectors)
                maximum_angular_momentum =
                    std::max(maximum_angular_momentum, projector.angular_momentum);
        }
        int number_of_lm_components =
            (maximum_angular_momentum + 1) * (maximum_angular_momentum + 1);

        compute_real_spherical_harmonics_batch(number_of_lm_components,
                                               number_of_plane_waves,
                                               geometry.reciprocal_units,
                                               geometry.magnitude_squared_reciprocal_units,
                                               geometry.spherical_harmonics);
        for (int direction = 0; direction < 3; direction++)
            compute_spherical_harmonic_derivatives_batch(
                number_of_lm_components,
                number_of_plane_waves,
                geometry.reciprocal_units,
                geometry.magnitude_squared_reciprocal_units,
                direction,
                geometry.spherical_harmonic_derivatives[static_cast<size_t>(direction)]);

        geometry.structure_factors.resize(
            static_cast<size_t>(crystal_structure_.number_of_atoms()));
        for (int atom_index = 0; atom_index < crystal_structure_.number_of_atoms(); atom_index++) {
            Vector3D atomic_position =
                crystal_structure_.convert_from_crystal_to_cartesian(atom_index);
            auto& factors = geometry.structure_factors[static_cast<size_t>(atom_index)];
            factors.resize(number_of_plane_waves);
            for (int g_index = 0; g_index < number_of_plane_waves; g_index++) {
                const auto& k_plus_g = geometry.reciprocal_units[g_index];
                Real phase = constants::two_pi *
                             (k_plus_g[0] * atomic_position.x + k_plus_g[1] * atomic_position.y +
                              k_plus_g[2] * atomic_position.z);
                factors[g_index] = compute_phase_factor(phase);
            }
        }
        return geometry;
    }

    /**
     * The Hubbard orbitals and the two halves of their strain derivative, packed
     * column-major (number_of_plane_waves x total_hubbard_orbitals).
     *
     * `orbitals` comes from the shared `build_atomic_orbitals_g_space` -- the same call the
     * SCF and the force make -- so what is differentiated here is exactly the projector the
     * energy used. The radial half is dchi/dq from the same tabulated chi (0.01 spacing,
     * 900 points, smooth), the angular half is chi times dY/dq_a with the 2pi/a unit
     * conversion `NonlocalPseudopotentialStressTensorCalculator` applies to its own.
     */
    void build_atomic_orbital_strain_parts(
        int kpoint_index,
        const std::vector<int>& plane_wave_indices,
        int number_of_plane_waves,
        const PlaneWaveGeometry& geometry,
        const std::vector<AtomicOrbitalDescriptor>& hubbard_descriptors,
        const std::vector<std::vector<std::vector<Real>>>& atomic_wavefunction_table,
        Real q_spacing,
        int number_of_q_points,
        std::vector<Complex>& orbitals,
        std::vector<Complex>& radial_derivative,
        std::array<std::vector<Complex>, 3>& angular_derivative) const {
        const auto& pseudopotential_set =
            scf_state_.quantum_system().pseudopotential_basis().pseudopotential_set();
        int total_hubbard_orbitals = static_cast<int>(hubbard_descriptors.size());
        Real two_pi_over_lattice_parameter = crystal_structure_.two_pi_over_lattice_parameter();

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

        size_t block = static_cast<size_t>(number_of_plane_waves) * total_hubbard_orbitals;
        orbitals.assign(block, Complex(0.0, 0.0));
        radial_derivative.assign(block, Complex(0.0, 0.0));
        for (int direction = 0; direction < 3; direction++)
            angular_derivative[static_cast<size_t>(direction)].assign(block, Complex(0.0, 0.0));

        for (int packed = 0; packed < total_hubbard_orbitals; packed++) {
            const auto& descriptor = hubbard_descriptors[static_cast<size_t>(packed)];
            int global_orbital = descriptor.global_orbital_index;

            /**
             * i^l, NOT `compute_angular_phase_factor`, which is (-i)^l.
             * `build_atomic_orbitals_g_space` carries i^l on the atomic orbitals while
             * `compute_kleinman_bylander_projectors` carries (-i)^l on the betas; the two
             * conventions differ, and taking the beta one here would multiply every l = 1
             * derivative part by -1 relative to the orbital it differentiates -- worth a
             * factor of 16 in the silicon 3p stress, since it flips the radial and angular
             * halves against the 1/sqrt(Omega) half rather than dropping them.
             */
            Complex angular_phase =
                std::conj(compute_angular_phase_factor(descriptor.angular_momentum));
            const auto& structure_factor =
                geometry.structure_factors[static_cast<size_t>(descriptor.atom_index)];
            const auto& radial_table =
                atomic_wavefunction_table[static_cast<size_t>(descriptor.species_type_index)]
                                         [static_cast<size_t>(descriptor.wavefunction_index)];
            size_t column = static_cast<size_t>(packed) * number_of_plane_waves;

            for (int g_index = 0; g_index < number_of_plane_waves; g_index++) {
                orbitals[column + g_index] =
                    all_atomic_orbitals[static_cast<size_t>(global_orbital)]
                                       [static_cast<size_t>(g_index)];

                Real magnitude = geometry.magnitude_bohr[g_index];
                Real radial_value = lagrange_interpolation_4point(
                    radial_table, magnitude, q_spacing, number_of_q_points);
                Real radial_slope = lagrange_interpolation_4point_derivative(
                    radial_table, magnitude, q_spacing, number_of_q_points);
                Real harmonic =
                    geometry.spherical_harmonics[static_cast<size_t>(g_index)]
                                                [static_cast<size_t>(descriptor.combined_lm_index)];
                Complex common = structure_factor[static_cast<size_t>(g_index)] * angular_phase;

                radial_derivative[column + g_index] = radial_slope * harmonic * common;
                for (int direction = 0; direction < 3; direction++) {
                    Real harmonic_slope =
                        geometry.spherical_harmonic_derivatives[static_cast<size_t>(direction)]
                                                               [static_cast<size_t>(g_index)]
                                                               [static_cast<size_t>(
                                                                   descriptor.combined_lm_index)];
                    angular_derivative[static_cast<size_t>(direction)][column + g_index] =
                        (radial_value / two_pi_over_lattice_parameter) * harmonic_slope * common;
                }
            }
        }
    }

    /**
     * The beta projectors and the two halves of their strain derivative, packed
     * column-major (number_of_plane_waves x total_projectors).
     *
     * The same construction `NonlocalPseudopotentialStressTensorCalculator` uses for its
     * own dbeta/deps, re-expressed against the shared geometry so that beta and phi are
     * differentiated identically. The undifferentiated beta comes from the shared
     * `compute_kleinman_bylander_projectors` rather than being rebuilt here, for the same
     * reason phi comes from `build_atomic_orbitals_g_space`: it is what the SCF's own S is
     * made of. `total_projectors == 0` (norm-conserving) leaves every array empty and
     * switches the whole dS/deps assembly off.
     */
    void build_beta_strain_parts(int kpoint_index,
                                 const std::vector<int>& plane_wave_indices,
                                 int number_of_plane_waves,
                                 int total_projectors,
                                 const PlaneWaveGeometry& geometry,
                                 std::vector<Complex>& beta_projectors,
                                 std::vector<Complex>& radial_derivative,
                                 std::array<std::vector<Complex>, 3>& angular_derivative) const {
        size_t block = static_cast<size_t>(number_of_plane_waves) * total_projectors;
        beta_projectors.assign(block, Complex(0.0, 0.0));
        radial_derivative.assign(block, Complex(0.0, 0.0));
        for (int direction = 0; direction < 3; direction++)
            angular_derivative[static_cast<size_t>(direction)].assign(block, Complex(0.0, 0.0));
        if (total_projectors == 0)
            return;

        const auto& basis = scf_state_.quantum_system().pseudopotential_basis();
        Matrix<Complex> canonical_projectors;
        compute_kleinman_bylander_projectors(kpoint_grid_[kpoint_index].coordinates,
                                             crystal_structure_,
                                             planewave_basis_,
                                             basis.pseudopotential_set(),
                                             basis.beta_function_table(),
                                             plane_wave_indices,
                                             number_of_plane_waves,
                                             total_projectors,
                                             canonical_projectors);
        for (int projector = 0; projector < total_projectors; projector++)
            for (int g_index = 0; g_index < number_of_plane_waves; g_index++)
                beta_projectors[static_cast<size_t>(projector) * number_of_plane_waves + g_index] =
                    canonical_projectors[projector][g_index];

        Real inverse_sqrt_cell_volume = 1.0 / std::sqrt(crystal_structure_.cell_volume());
        Real two_pi_over_lattice_parameter = crystal_structure_.two_pi_over_lattice_parameter();

        int projector_index = 0;
        for (int atom_index = 0; atom_index < crystal_structure_.number_of_atoms(); atom_index++) {
            int species_type_index = crystal_structure_.species_type_index(atom_index);
            const auto& structure_factor =
                geometry.structure_factors[static_cast<size_t>(atom_index)];
            for (int beta_index = 0; beta_index < basis.number_of_projectors(species_type_index);
                 beta_index++) {
                int angular_momentum = basis.beta_angular_momentum(species_type_index, beta_index);
                Complex angular_phase = compute_angular_phase_factor(angular_momentum);
                Complex prefactor = inverse_sqrt_cell_volume * angular_phase;

                for (int magnetic_projection : harmonic_order_sequence(angular_momentum)) {
                    int lm_index =
                        harmonic_combined_degree_order_index(angular_momentum, magnetic_projection);
                    size_t column = static_cast<size_t>(projector_index) * number_of_plane_waves;
                    for (int g_index = 0; g_index < number_of_plane_waves; g_index++) {
                        Real magnitude = geometry.magnitude_bohr[g_index];
                        Real beta_value =
                            basis.interpolate_beta(species_type_index, beta_index, magnitude);
                        Real beta_slope =
                            basis.interpolate_dbeta(species_type_index, beta_index, magnitude);
                        Real harmonic = geometry.spherical_harmonics[static_cast<size_t>(g_index)]
                                                                    [static_cast<size_t>(lm_index)];
                        Complex common = structure_factor[static_cast<size_t>(g_index)] * prefactor;

                        radial_derivative[column + g_index] = beta_slope * harmonic * common;
                        for (int direction = 0; direction < 3; direction++) {
                            Real harmonic_slope =
                                geometry
                                    .spherical_harmonic_derivatives[static_cast<size_t>(direction)]
                                                                   [static_cast<size_t>(g_index)]
                                                                   [static_cast<size_t>(lm_index)];
                            angular_derivative[static_cast<size_t>(direction)][column + g_index] =
                                (beta_value / two_pi_over_lattice_parameter) * harmonic_slope *
                                common;
                        }
                    }
                    projector_index++;
                }
            }
        }
    }

    /**
     * dA/deps_ab = -[ radial_a_b + angular_a * (k+G)_b ] - delta_ab A/2, in place.
     *
     * `radial` is f'(q) Y_lm S i^l /sqrt(Omega) and is weighted here by
     * (k+G)_a (k+G)_b/|k+G|; `angular` is f(q) (dY_lm/dq_a) S i^l /sqrt(Omega) and is
     * weighted by (k+G)_b. Quantum ESPRESSO carries those same two weights, with the
     * overall minus already folded into them.
     */
    static void apply_strain_derivative(int number_of_plane_waves,
                                        int number_of_functions,
                                        const PlaneWaveGeometry& geometry,
                                        int direction_a,
                                        int direction_b,
                                        const std::vector<Complex>& base,
                                        const std::vector<Complex>& radial,
                                        const std::vector<Complex>& angular,
                                        std::vector<Complex>& strained) {
        strained.assign(base.size(), Complex(0.0, 0.0));
        Real diagonal = (direction_a == direction_b) ? 0.5 : 0.0;
        for (int function_index = 0; function_index < number_of_functions; function_index++) {
            size_t column = static_cast<size_t>(function_index) * number_of_plane_waves;
            for (int g_index = 0; g_index < number_of_plane_waves; g_index++) {
                Real component_a = geometry.cartesian[g_index][static_cast<size_t>(direction_a)];
                Real component_b = geometry.cartesian[g_index][static_cast<size_t>(direction_b)];
                Real radial_weight =
                    component_a * component_b * geometry.inverse_magnitude_bohr[g_index];
                strained[column + g_index] = -(radial[column + g_index] * radial_weight +
                                               angular[column + g_index] * component_b) -
                                             diagonal * base[column + g_index];
            }
        }
    }

    /**
     * Add sum_ij q_ij left(m,i) right(j,v) into `target`, over every atom's own projector
     * block. S runs over every atom that carries projectors, Hubbard or not, so this sum
     * does too -- the same reason a Hubbard *force* appears on non-Hubbard atoms.
     */
    template <typename AugmentationCharge,
              typename OrbitalProjectorIndex,
              typename ProjectorBandIndex,
              typename OrbitalBandIndex>
    static void accumulate_augmentation(int number_of_atoms,
                                        const std::vector<int>& channels_by_atom,
                                        const std::vector<int>& projector_offset_by_atom,
                                        const AugmentationCharge& augmentation_charge,
                                        int total_hubbard_orbitals,
                                        int number_of_bands,
                                        const std::vector<Complex>& left,
                                        const std::vector<Complex>& right,
                                        const OrbitalProjectorIndex& orbital_projector,
                                        const ProjectorBandIndex& projector_band,
                                        const OrbitalBandIndex& orbital_band,
                                        std::vector<Complex>& target) {
        for (int atom_index = 0; atom_index < number_of_atoms; atom_index++) {
            int offset = projector_offset_by_atom[static_cast<size_t>(atom_index)];
            int number_of_channels = channels_by_atom[static_cast<size_t>(atom_index)];
            for (int channel_i = 0; channel_i < number_of_channels; channel_i++) {
                for (int channel_j = 0; channel_j < number_of_channels; channel_j++) {
                    Real q_value = augmentation_charge(atom_index, channel_i, channel_j);
                    if (std::abs(q_value) < math::tolerances::d_matrix_element_threshold)
                        continue;
                    int projector_i = offset + channel_i;
                    int projector_j = offset + channel_j;
                    for (int orbital = 0; orbital < total_hubbard_orbitals; orbital++) {
                        Complex left_value = left[orbital_projector(orbital, projector_i)];
                        if (left_value == Complex(0.0, 0.0))
                            continue;
                        for (int band = 0; band < number_of_bands; band++)
                            target[orbital_band(orbital, band)] +=
                                q_value * left_value * right[projector_band(projector_j, band)];
                    }
                }
            }
        }
    }

    /**
     * Rewrite a Gamma-only half-sphere ZGEMM result as its full-sphere value.
     *
     * Generalizes `convert_gamma_only_projector_overlaps_to_full_sphere`'s flag: the G = 0
     * term is read from the operands themselves rather than assumed present or absent. It
     * is neither here -- a strain derivative is -[(k+G)-weighted terms] - delta_ab A/2, so
     * it vanishes at G = 0 off the diagonal and does not on it, and hard-coding either rule
     * would be wrong for half the tensor.
     */
    static void complete_gamma_only_overlaps(int number_of_plane_waves,
                                             int number_of_left,
                                             int number_of_right,
                                             const Complex* left,
                                             const Complex* right,
                                             int g_zero_index,
                                             std::vector<Complex>& overlaps) {
        for (int left_index = 0; left_index < number_of_left; left_index++) {
            for (int right_index = 0; right_index < number_of_right; right_index++) {
                Real g_zero_product =
                    g_zero_index >= 0
                        ? (std::conj(left[static_cast<size_t>(left_index) * number_of_plane_waves +
                                          g_zero_index]) *
                           right[static_cast<size_t>(right_index) * number_of_plane_waves +
                                 g_zero_index])
                              .real()
                        : 0.0;
                size_t flat_index = static_cast<size_t>(left_index) +
                                    static_cast<size_t>(right_index) * number_of_left;
                overlaps[flat_index] =
                    gamma_only_full_sphere_overlap(overlaps[flat_index], g_zero_product);
            }
        }
    }

    /**
     * Replace the tensor by its symmetric part.
     *
     * Averaged, not mirrored: the dY_lm/dq_a (k+G)_b term is genuinely asymmetric in
     * (a,b) before symmetrization, and the full 3x3 is accumulated above so that the
     * antisymmetric half is cancelled rather than discarded.
     */
    static void symmetrize_stress_tensor(Real stress[3][3]) {
        for (int direction_a = 0; direction_a < 3; direction_a++) {
            for (int direction_b = direction_a + 1; direction_b < 3; direction_b++) {
                Real average =
                    0.5 * (stress[direction_a][direction_b] + stress[direction_b][direction_a]);
                stress[direction_a][direction_b] = average;
                stress[direction_b][direction_a] = average;
            }
        }
    }

    const Crystal& crystal_structure_;
    const PlanewaveBasis& planewave_basis_;
    const KPointsGrid& kpoint_grid_;
    const NormConservingPlanewaveKohnShamSCFState& scf_state_;
    StressTensorContributions& result_;
};

} // namespace q3
