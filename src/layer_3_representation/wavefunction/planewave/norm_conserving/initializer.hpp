#pragma once

/**
 * NormConservingPlanewaveWavefunctionInitializer — initialize wavefunctions from atomic orbitals.
 *
 * Provides the "atomic+random" starting guess for the Kohn-Sham eigensolver:
 * 1. Tabulates atomic pseudo-wavefunctions from the pseudopotential files
 * 2. Computes atomic orbital projections onto the plane-wave basis at each k-point
 * 3. Applies a small random perturbation to break symmetry
 * 4. Orthogonalizes the wavefunctions via subspace rotation (Löwdin / Cholesky)
 * 5. For ultrasoft PPs, applies the S-operator overlap correction
 *
 * This generates a starting wavefunction set for the Davidson or direct diagonalizer,
 * avoiding the need for expensive random initialization.
 */
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "layer_0_core/logging/init.hpp"
#include "layer_0_core/math/integration/simpson.hpp"
#include "layer_0_core/math/interpolation/lagrange.hpp"
#include "layer_0_core/math/linear_algebra/blas.hpp"
#include "layer_0_core/math/phase_factor.hpp"
#include "layer_0_core/math/spherical/bessel.hpp"
#include "layer_0_core/math/spherical/harmonics.hpp"
#include "layer_0_core/math/tolerances.hpp"
#include "layer_0_core/types/init.hpp"
#include "layer_1_structure/crystal/crystal.hpp"
#include "layer_1_structure/reciprocal_space/kpoints_grid.hpp"
#include "layer_3_representation/basis/planewave/init.hpp"
#include "layer_3_representation/pseudopotential/planewave/internal/atomic_orbital_builder.hpp"
#include "layer_3_representation/pseudopotential/planewave/internal/projector.hpp"
#include "layer_3_representation/pseudopotential/planewave/internal/wavefunction_atomic.hpp"
#include "layer_3_representation/pseudopotential/set.hpp"
#include "layer_3_representation/quantum_system/kohn_sham/planewave/init.hpp"
#include "layer_3_representation/quantum_system/kohn_sham/planewave/norm_conserving/init.hpp"
#include "layer_3_representation/quantum_system/kohn_sham/planewave/ultrasoft/init.hpp"

#include "layer_0_core/math/linear_algebra/lapack.hpp"

namespace q3 {
using namespace q3::blas;
using namespace q3::lapack;

/**
 * Computes starting wavefunctions from atomic orbitals (chi) in the pseudopotential.
 *
 * The procedure is the standard LCAO starting guess for a plane-wave SCF
 * (Payne et al., Rev. Mod. Phys. 64, 1045 (1992), sec. V): superpose the
 * pseudo-atomic orbitals of every atom, pad the set to the requested band
 * count, then take the Rayleigh-Ritz solution of H c = e S c on that
 * subspace. Rayleigh-Ritz is used because it is the variationally optimal
 * selection of `number_of_bands` vectors out of a larger non-orthogonal
 * subspace, and because it delivers them S-orthonormal, which is what the
 * Davidson solver requires on entry.
 *
 * Quantum ESPRESSO's pw.x reaches the same three stages. Its routine names
 * are not reproduced here and its stage boundaries are not the ones below:
 * its driver also performs wavefunction file I/O and Hubbard-projector
 * orthogonalisation, neither of which this class does, and the per-k-point
 * driver loop lives in the SCF iterator rather than here.
 */
class NormConservingPlanewaveWavefunctionInitializer {
  public:
    /**
     * Precompute the radial Fourier transform interpolation table for all species:
     *
     *     table[species][shell][q] = 4π/√Ω ∫ chi(r) j_l(q·r) r dr
     *
     * The 4π and the j_l come from the Rayleigh expansion of a plane wave, the
     * 1/√Ω from the plane-wave normalisation, and the single power of r from
     * UPF storing r·R_nl(r) rather than R_nl(r).
     *
     * The q axis is innermost because the only consumer is a 4-point Lagrange
     * stencil that reads four consecutive q entries, and lagrange_interpolation_4point
     * takes a one-dimensional q table. Species is outermost because the shell
     * count varies per species, so the jagged axis has to be the inner of the
     * two; that leaves the shell axis with no remaining freedom.
     */
    void initialize_interpolation_table(const PseudopotentialSet& pseudopotential_set,
                                        Real cell_volume) {
        number_of_q_points_ = math::tolerances::default_number_of_q_points_for_atomic_wavefunctions;

        atomic_wavefunction_table_ = tabulate_atomic_wavefunctions_in_q_space(
            pseudopotential_set, cell_volume, q_spacing_, number_of_q_points_);

        log::info("     Atomic wavefunctions interpolation table initialized ({} q-points, dq={})",
                  number_of_q_points_,
                  q_spacing_);
    }

    /**
     * Compute the starting vector set in G-space for one k-point.
     * Returns atomic_wavefunctions[starting_vector_count][number_of_plane_waves],
     * where starting_vector_count = max(atomic orbital count, band count).
     *
     * Two steps: the pure atomic orbitals come from the shared builder (see
     * atomic_orbital_builder.hpp for their derivation), and this method then
     * brings the set up to the band count -- padding the tail with random
     * vectors when the atomic orbitals do not reach it.
     */
    void compute_atomic_wavefunctions(int kpoint_index,
                                      int number_of_plane_waves,
                                      int number_of_bands,
                                      const std::vector<int>& g_vector_indices,
                                      const Crystal& crystal_structure,
                                      const KPointsGrid& kpoint_grid,
                                      const PlanewaveBasis& planewave_basis,
                                      const PseudopotentialSet& pseudopotential_set,
                                      std::vector<std::vector<Complex>>& atomic_wavefunctions) {
        /**
         * Build the pure atomic orbital set (no padding) via the shared builder --
         * identical construction used by PDOS/Löwdin projection
         * (layer_3_representation/pseudopotential/planewave/internal/atomic_orbital_builder.hpp).
         */
        std::vector<std::vector<Complex>> pure_atomic_orbitals;
        build_atomic_orbitals_g_space(kpoint_index,
                                      number_of_plane_waves,
                                      g_vector_indices,
                                      crystal_structure,
                                      kpoint_grid,
                                      planewave_basis,
                                      pseudopotential_set,
                                      atomic_wavefunction_table_,
                                      q_spacing_,
                                      number_of_q_points_,
                                      pure_atomic_orbitals);

        int number_of_atomic_wavefunctions = static_cast<int>(pure_atomic_orbitals.size());
        int total_wavefunction_count = std::max(number_of_atomic_wavefunctions, number_of_bands);
        atomic_wavefunctions.assign(total_wavefunction_count,
                                    std::vector<Complex>(number_of_plane_waves, Complex(0.0, 0.0)));
        for (int wavefunction_index = 0; wavefunction_index < number_of_atomic_wavefunctions;
             wavefunction_index++) {
            atomic_wavefunctions[wavefunction_index] =
                std::move(pure_atomic_orbitals[wavefunction_index]);
        }

        /**
         * Symmetry-breaking hook, currently INERT: the coefficient below is 0.0,
         * so the loop that follows multiplies every coefficient by exactly 1 and
         * no perturbation is applied. It is kept as the place a perturbation
         * would go, not as one that happens. See the audit note (R-D5-1) --
         * either the branch goes, or a magnitude criterion gets written down.
         *
         * The branch exists at all because a perturbation is only useful when
         * the atomic orbitals already cover the band count: when they do not,
         * the random padding below breaks the same degeneracy for free.
         */
        static constexpr int default_random_seed = 42;
        std::mt19937 random_number_generator(default_random_seed +
                                             kpoint_index); // deterministic seed per k-point
        std::uniform_real_distribution<Real> distribution(0.0, 1.0);

        if (number_of_atomic_wavefunctions >= number_of_bands) {
            for (int wavefunction_loop_index = 0;
                 wavefunction_loop_index < total_wavefunction_count;
                 wavefunction_loop_index++) {
                for (int g_index = 0; g_index < number_of_plane_waves; g_index++) {
                    Real random_amplitude = distribution(random_number_generator);
                    Real random_phase = distribution(random_number_generator);
                    Real perturbation_arg = constants::two_pi * random_phase;
                    atomic_wavefunctions[wavefunction_loop_index][g_index] *=
                        (1.0 + 0.0 * Complex(random_amplitude * std::cos(perturbation_arg),
                                             random_amplitude * std::sin(perturbation_arg)));
                }
            }
        } else {
            /**
             * Fill the missing vectors with random coefficients. Each is drawn from
             * a hash of the k-point, the vector and the plane wave's Miller indices,
             * not from a generator stepped along the plane-wave list: ranks sharing a
             * pool each hold a different part of that list, and k-point pools number
             * their k-points locally, so a stepped generator gave every topology its
             * own starting guess. The ultrasoft silicon dataset carries no atomic
             * orbitals and starts from these vectors alone, and took a different SCF
             * path on every rank count to the same fixed point.
             */
            const Vector3D& kpoint_coordinates = kpoint_grid[kpoint_index].coordinates;
            for (int wavefunction_loop_index = number_of_atomic_wavefunctions;
                 wavefunction_loop_index < total_wavefunction_count;
                 wavefunction_loop_index++) {
                for (int g_index = 0; g_index < number_of_plane_waves; g_index++) {
                    const IntegerVector3D& miller =
                        planewave_basis.g_vectors()[g_vector_indices[g_index]].miller;
                    Real random_amplitude = hashed_unit_interval(
                        kpoint_coordinates, wavefunction_loop_index, miller, 0);
                    Real random_phase = hashed_unit_interval(
                        kpoint_coordinates, wavefunction_loop_index, miller, 1);
                    Real phase_argument = constants::two_pi * random_phase;
                    atomic_wavefunctions[wavefunction_loop_index][g_index] =
                        Complex(random_amplitude * std::cos(phase_argument),
                                random_amplitude * std::sin(phase_argument));
                }
            }
        }
    }

    /**
     * Rayleigh-Ritz on the starting subspace: form H_ij and S_ij in the basis of
     * the starting vectors, solve the generalised eigenproblem H c = e S c, and
     * rotate. Taking the lowest `number_of_bands` Ritz pairs is the variationally
     * optimal selection out of that subspace, and the Ritz vectors come out
     * S-orthonormal, which is the eigensolver's entry condition.
     *
     * Gamma-only takes the real path throughout: under psi(-G) = conj(psi(G)) the
     * subspace matrices are real symmetric, so a real solver is both cheaper and
     * the one that keeps the rotated basis gamma-symmetric.
     *
     * This overload accepts pre-computed KB projectors to avoid redundant recomputation.
     *
     * `smooth_potential` is the local potential on this rank's part of the
     * wavefunction grid, already multiplied by 1/N of that grid -- what the
     * Hamiltonian application holds, not the density-grid potential the SCF
     * stores.
     */
    void rotate_wavefunctions(
        int kpoint_index,
        int kpoint_index_state,
        int number_of_plane_waves,
        int number_of_bands,
        const std::vector<int>& g_vector_indices,
        const std::vector<Real>& kinetic_energy_squared,
        const std::vector<Real>& smooth_potential,
        const Crystal& crystal_structure,
        const KPointsGrid& kpoint_grid,
        const PlanewaveBasis& planewave_basis,
        const execution::PlanewaveExecutionPolicy& execution_policy,
        const NormConservingPlanewaveKohnShamQuantumSystem& pseudopotential_quantum_system,
        const std::vector<std::vector<Complex>>& atomic_wavefunctions,
        PlanewaveKohnShamQuantumSystem& quantum_system,
        const Matrix<Complex>& precomputed_kleinman_bylander_potential) {
        int number_of_starting_wavefunctions = static_cast<int>(atomic_wavefunctions.size());
        int leading_dimension = std::max(number_of_plane_waves, 1);

        /**
         * Pack atomic wavefunctions into contiguous column-major matrix:
         * contiguous_wavefunctions[g + wfc * npw] = atomic_wavefunctions[wfc][g]
         */
        std::vector<Complex> contiguous_wavefunctions(static_cast<size_t>(number_of_plane_waves) *
                                                          number_of_starting_wavefunctions,
                                                      Complex(0.0, 0.0));
        for (int wavefunction_index = 0; wavefunction_index < number_of_starting_wavefunctions;
             wavefunction_index++) {
            std::memcpy(&contiguous_wavefunctions[wavefunction_index * number_of_plane_waves],
                        atomic_wavefunctions[wavefunction_index].data(),
                        number_of_plane_waves * sizeof(Complex));
        }

        // Apply H to all atomic wfcs using smooth grid + ZGEMM for KB projectors
        std::vector<Complex> contiguous_hamiltonian_wavefunctions(
            static_cast<size_t>(number_of_plane_waves) * number_of_starting_wavefunctions,
            Complex(0.0, 0.0));
        apply_hamiltonian_batch_contiguous(number_of_plane_waves,
                                           number_of_starting_wavefunctions,
                                           g_vector_indices,
                                           kinetic_energy_squared,
                                           smooth_potential,
                                           crystal_structure,
                                           kpoint_grid,
                                           kpoint_index,
                                           planewave_basis,
                                           execution_policy,
                                           pseudopotential_quantum_system,
                                           precomputed_kleinman_bylander_potential,
                                           contiguous_wavefunctions.data(),
                                           contiguous_hamiltonian_wavefunctions.data());

        /**
         * Apply S to every starting vector. For norm-conserving pseudopotentials
         * S = I, so the overlap-applied set IS the input set and nothing is
         * allocated: everything downstream reads it, so pointing at the input
         * is enough. Ultrasoft needs the q_ij augmentation and so needs its own
         * buffer. This block is the tallest point of a run's setup, and memory
         * released here is not returned to the system, so the array not taken
         * is worth more than its lifetime suggests.
         */
        std::vector<Complex> contiguous_overlap_storage;
        Complex* contiguous_overlap_wavefunctions = contiguous_wavefunctions.data();
        if (pseudopotential_quantum_system.has_ultrasoft()) {
            // US overlap: S|ψ⟩ = |ψ⟩ + Σ_ab q_ab |β_a⟩⟨β_b|ψ⟩
            contiguous_overlap_storage.assign(contiguous_wavefunctions.begin(),
                                              contiguous_wavefunctions.end());
            contiguous_overlap_wavefunctions = contiguous_overlap_storage.data();
            apply_overlap_contiguous(number_of_plane_waves,
                                     number_of_starting_wavefunctions,
                                     crystal_structure,
                                     execution_policy,
                                     pseudopotential_quantum_system,
                                     precomputed_kleinman_bylander_potential,
                                     contiguous_wavefunctions.data(),
                                     contiguous_overlap_wavefunctions);
        }

        // Build H_ij = χ†·Hχ and S_ij = χ†·Sχ via ZGEMM
        std::vector<Complex> hamiltonian_matrix(
            static_cast<size_t>(number_of_starting_wavefunctions) *
                number_of_starting_wavefunctions,
            Complex(0.0, 0.0));
        std::vector<Complex> overlap_matrix(static_cast<size_t>(number_of_starting_wavefunctions) *
                                                number_of_starting_wavefunctions,
                                            Complex(0.0, 0.0));

        std::complex<double> alpha_one(1.0, 0.0);
        std::complex<double> beta_zero(0.0, 0.0);

        auto* cast_wavefunctions =
            reinterpret_cast<std::complex<double>*>(contiguous_wavefunctions.data());
        auto* cast_hamiltonian_wavefunctions =
            reinterpret_cast<std::complex<double>*>(contiguous_hamiltonian_wavefunctions.data());
        auto* cast_overlap_wavefunctions =
            reinterpret_cast<std::complex<double>*>(contiguous_overlap_wavefunctions);

        if (planewave_basis.is_gamma_only()) {
            /**
             * Gamma-only: use DGEMM on interleaved real/imaginary parts.
             * Reinterpret complex arrays as double arrays of length 2*npw.
             * DGEMM("T","N", n, n, 2*npw, alpha=2) computes 2*Re(Σ conj(a)*b)
             * which is the gamma inner product before G=0 correction.
             */
            int leading_dimension = std::max(number_of_plane_waves, 1);
            int double_number_of_plane_waves = 2 * number_of_plane_waves;
            int double_leading_dimension = 2 * leading_dimension;
            double alpha_two = 2.0;
            double d_beta_zero = 0.0;

            auto* real_wavefunctions = reinterpret_cast<double*>(contiguous_wavefunctions.data());
            auto* real_hamiltonian_wavefunctions =
                reinterpret_cast<double*>(contiguous_hamiltonian_wavefunctions.data());
            auto* real_overlap_wavefunctions =
                reinterpret_cast<double*>(contiguous_overlap_wavefunctions);

            // H_ij = 2*Re(χ†·Hχ) via DGEMM
            std::vector<double> hamiltonian_matrix_real(
                static_cast<size_t>(number_of_starting_wavefunctions) *
                    number_of_starting_wavefunctions,
                0.0);
            dgemm_("T",
                   "N",
                   &number_of_starting_wavefunctions,
                   &number_of_starting_wavefunctions,
                   &double_number_of_plane_waves,
                   &alpha_two,
                   real_wavefunctions,
                   &double_leading_dimension,
                   real_hamiltonian_wavefunctions,
                   &double_leading_dimension,
                   &d_beta_zero,
                   hamiltonian_matrix_real.data(),
                   &number_of_starting_wavefunctions);

            // S_ij = 2*Re(χ†·Sχ) via DGEMM
            std::vector<double> overlap_matrix_real(
                static_cast<size_t>(number_of_starting_wavefunctions) *
                    number_of_starting_wavefunctions,
                0.0);
            dgemm_("T",
                   "N",
                   &number_of_starting_wavefunctions,
                   &number_of_starting_wavefunctions,
                   &double_number_of_plane_waves,
                   &alpha_two,
                   real_wavefunctions,
                   &double_leading_dimension,
                   real_overlap_wavefunctions,
                   &double_leading_dimension,
                   &d_beta_zero,
                   overlap_matrix_real.data(),
                   &number_of_starting_wavefunctions);

            // G=0 correction: subtract Re(conj(left_g0)*right_g0)
            if (planewave_basis.gamma_origin_is_local()) {
                for (int column = 0; column < number_of_starting_wavefunctions; ++column) {
                    Complex h_right_g0 =
                        contiguous_hamiltonian_wavefunctions[column * number_of_plane_waves];
                    Complex s_right_g0 =
                        contiguous_overlap_wavefunctions[column * number_of_plane_waves];
                    for (int row = 0; row < number_of_starting_wavefunctions; ++row) {
                        Complex left_g0 = contiguous_wavefunctions[row * number_of_plane_waves];
                        int index = column * number_of_starting_wavefunctions + row;
                        hamiltonian_matrix_real[index] -= (std::conj(left_g0) * h_right_g0).real();
                        overlap_matrix_real[index] -= (std::conj(left_g0) * s_right_g0).real();
                    }
                }
            }

            // Copy to Complex matrices for zhegv
            for (int index = 0;
                 index < number_of_starting_wavefunctions * number_of_starting_wavefunctions;
                 ++index) {
                hamiltonian_matrix[index] = Complex(hamiltonian_matrix_real[index], 0.0);
                overlap_matrix[index] = Complex(overlap_matrix_real[index], 0.0);
            }
        } else {
            int leading_dimension = std::max(number_of_plane_waves, 1);
            /**
             * Non-gamma: use ZGEMM
             * H_ij = χ†·Hχ: ZGEMM('C', 'N', n_wfc, n_wfc, npw)
             */
            zgemm_("C",
                   "N",
                   &number_of_starting_wavefunctions,
                   &number_of_starting_wavefunctions,
                   &number_of_plane_waves,
                   &alpha_one,
                   cast_wavefunctions,
                   &leading_dimension,
                   cast_hamiltonian_wavefunctions,
                   &leading_dimension,
                   &beta_zero,
                   reinterpret_cast<std::complex<double>*>(hamiltonian_matrix.data()),
                   &number_of_starting_wavefunctions);

            // S_ij = χ†·Sχ: ZGEMM('C', 'N', n_wfc, n_wfc, npw)
            zgemm_("C",
                   "N",
                   &number_of_starting_wavefunctions,
                   &number_of_starting_wavefunctions,
                   &number_of_plane_waves,
                   &alpha_one,
                   cast_wavefunctions,
                   &leading_dimension,
                   cast_overlap_wavefunctions,
                   &leading_dimension,
                   &beta_zero,
                   reinterpret_cast<std::complex<double>*>(overlap_matrix.data()),
                   &number_of_starting_wavefunctions);
        }

        /**
         * Reduce H_ij and S_ij across band ranks within this pool.
         * Each band rank holds the same G-vectors (replicated), but in the future
         * when G-space is distributed, each rank holds a slab and we need
         * reduce_within_pool_vector to sum the partial dot products.
         */
        execution_policy.reduce_within_pool_vector(
            reinterpret_cast<std::vector<Real>&>(hamiltonian_matrix));
        execution_policy.reduce_within_pool_vector(
            reinterpret_cast<std::vector<Real>&>(overlap_matrix));

        /**
         * Diagonalize: H*c = e*S*c
         * For gamma-only, use dsygv (real symmetric) to guarantee real
         * eigenvectors that preserve gamma symmetry after ZGEMM rotation.
         */
        std::vector<Real> subspace_eigenvalues(number_of_starting_wavefunctions, 0.0);
        int info;
        int eigenproblem_type = 1;

        if (planewave_basis.is_gamma_only()) {
            // Extract real parts from the Complex matrices
            int matrix_size = number_of_starting_wavefunctions * number_of_starting_wavefunctions;
            std::vector<Real> hamiltonian_real(matrix_size);
            std::vector<Real> overlap_real(matrix_size);
            for (int matrix_element_index = 0; matrix_element_index < matrix_size;
                 ++matrix_element_index) {
                hamiltonian_real[matrix_element_index] =
                    hamiltonian_matrix[matrix_element_index].real();
                overlap_real[matrix_element_index] = overlap_matrix[matrix_element_index].real();
            }

            int work_size = std::max(1, 3 * number_of_starting_wavefunctions - 1);
            std::vector<Real> work(work_size);
            Real optimal_work_size;
            int query_size = -1;
            dsygv_(&eigenproblem_type,
                   "V",
                   "U",
                   &number_of_starting_wavefunctions,
                   hamiltonian_real.data(),
                   &number_of_starting_wavefunctions,
                   overlap_real.data(),
                   &number_of_starting_wavefunctions,
                   subspace_eigenvalues.data(),
                   &optimal_work_size,
                   &query_size,
                   &info);
            work_size = static_cast<int>(optimal_work_size);
            work.resize(work_size);

            dsygv_(&eigenproblem_type,
                   "V",
                   "U",
                   &number_of_starting_wavefunctions,
                   hamiltonian_real.data(),
                   &number_of_starting_wavefunctions,
                   overlap_real.data(),
                   &number_of_starting_wavefunctions,
                   subspace_eigenvalues.data(),
                   work.data(),
                   &work_size,
                   &info);

            // Store real eigenvectors back into hamiltonian_matrix for ZGEMM rotation
            for (int matrix_element_index = 0; matrix_element_index < matrix_size;
                 ++matrix_element_index) {
                hamiltonian_matrix[matrix_element_index] =
                    Complex(hamiltonian_real[matrix_element_index], 0.0);
            }
        } else {
            int work_size = 2 * number_of_starting_wavefunctions +
                            number_of_starting_wavefunctions * number_of_starting_wavefunctions;
            std::vector<Complex> complex_work_space(work_size);
            std::vector<Real> real_work_space(
                std::max(1, 3 * number_of_starting_wavefunctions - 2));

            zhegv_(&eigenproblem_type,
                   "V",
                   "U",
                   &number_of_starting_wavefunctions,
                   reinterpret_cast<std::complex<double>*>(hamiltonian_matrix.data()),
                   &number_of_starting_wavefunctions,
                   reinterpret_cast<std::complex<double>*>(overlap_matrix.data()),
                   &number_of_starting_wavefunctions,
                   subspace_eigenvalues.data(),
                   reinterpret_cast<std::complex<double>*>(complex_work_space.data()),
                   &work_size,
                   real_work_space.data(),
                   &info);
        }

        if (info != 0) {
            log::error("LAPACK eigensolver failed in rotate_wavefunctions with info = {}", info);
        }

        // Store eigenvalues
        for (int band_index = 0; band_index < number_of_bands; band_index++) {
            quantum_system.set_eigenvalue(
                kpoint_index_state, band_index, subspace_eigenvalues[band_index]);
        }

        /**
         * Rotate wavefunctions: evc = χ · eigenvectors via ZGEMM
         * evc(npw × n_bands) = χ(npw × n_wfc) · c(n_wfc × n_bands)
         * The eigenvectors are stored column-major in hamiltonian_matrix after zhegv.
         */
        std::vector<Complex> rotated_wavefunctions(
            static_cast<size_t>(number_of_plane_waves) * number_of_bands, Complex(0.0, 0.0));

        zgemm_("N",
               "N",
               &number_of_plane_waves,
               &number_of_bands,
               &number_of_starting_wavefunctions,
               &alpha_one,
               cast_wavefunctions,
               &leading_dimension,
               reinterpret_cast<std::complex<double>*>(hamiltonian_matrix.data()),
               &number_of_starting_wavefunctions,
               &beta_zero,
               reinterpret_cast<std::complex<double>*>(rotated_wavefunctions.data()),
               &leading_dimension);

        /**
         * Gamma-only: enforce real G=0 for the rotated wavefunctions, which the
         * two-band FFT trick requires.
         *
         * The atomic orbitals are already real at G=0 here -- at gamma the k-phase
         * and the structure factor are both unity, and chi_l(0) = 0 for l >= 1
         * because j_l(0) = 0, so only the l = 0 shells contribute and they carry
         * i^0 = 1. The complex G=0 component this removes comes from the random
         * padding vectors, which are generated with an arbitrary phase. Because
         * the subspace matrices above were formed from the uncleaned vectors, the
         * G=0 correction there uses the general Re[conj(a0)*b0] form rather than
         * the real-parts-only form that would need this cleanup to run first.
         */
        if (planewave_basis.is_gamma_only() && planewave_basis.gamma_origin_is_local()) {
            for (int band_index = 0; band_index < number_of_bands; ++band_index) {
                Complex& g_zero_coefficient =
                    rotated_wavefunctions[band_index * number_of_plane_waves];
                g_zero_coefficient = Complex(g_zero_coefficient.real(), 0.0);
            }
        }

        // Copy rotated wavefunctions into quantum system
        for (int band_index = 0; band_index < number_of_bands; band_index++) {
            Complex* band_data =
                quantum_system.wavefunction_band_data(kpoint_index_state, band_index);
            std::memcpy(band_data,
                        &rotated_wavefunctions[band_index * number_of_plane_waves],
                        number_of_plane_waves * sizeof(Complex));
        }
    }

  private:
    /**
     * A reproducible number in [0, 1) for one coefficient of one starting vector,
     * keyed on what identifies the coefficient physically -- the k-point's
     * coordinates, the vector, the plane wave's Miller indices -- and on a stream
     * index, so the amplitude and the phase are independent. SplitMix64 finalizers
     * chain the keys; the top 53 bits become the fraction.
     */
    [[nodiscard]] static Real hashed_unit_interval(const Vector3D& kpoint_coordinates,
                                                   int vector_index,
                                                   const IntegerVector3D& miller,
                                                   int stream) {
        auto mix = [](std::uint64_t value) {
            value += 0x9E3779B97F4A7C15ULL;
            value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ULL;
            value = (value ^ (value >> 27)) * 0x94D049BB133111EBULL;
            return value ^ (value >> 31);
        };
        /**
         * k-point coordinates are exact rationals of the grid in crystal units, so
         * rounding them to 1e-9 names the same k-point identically everywhere.
         */
        constexpr Real kpoint_resolution = 1.0e9;
        std::uint64_t state = mix(0x51A3C0FFEEULL);
        for (Real component : {kpoint_coordinates.x, kpoint_coordinates.y, kpoint_coordinates.z}) {
            state = mix(state ^
                        static_cast<std::uint64_t>(std::llround(component * kpoint_resolution)));
        }
        state = mix(state ^ static_cast<std::uint64_t>(vector_index));
        for (int component : {miller.x, miller.y, miller.z}) {
            state = mix(state ^ static_cast<std::uint64_t>(static_cast<std::int64_t>(component)));
        }
        state = mix(state ^ static_cast<std::uint64_t>(stream));
        return static_cast<Real>(state >> 11) * 0x1.0p-53;
    }

    /// Interpolation table: atomic_wavefunction_table_[species_type][wfc_index][q_index]
    std::vector<std::vector<std::vector<Real>>> atomic_wavefunction_table_;
    Real q_spacing_ =
        math::tolerances::default_q_grid_spacing; // grid spacing in atomic units (matching QE)
    int number_of_q_points_ = 0;                  // table size

    /**
     * Apply H|ψ⟩ for a batch of vectors using contiguous layout.
     *
     * Uses the smooth (wavefunction) FFT grid instead of the dense (charge density) grid
     * for ~3.4× faster FFTs, and batched ZGEMM for KB projectors.
     *
     * Input/output layout: column-major, psi[g + vector * npw].
     */
    static void apply_hamiltonian_batch_contiguous(
        int number_of_plane_waves,
        int number_of_vectors,
        const std::vector<int>& g_vector_indices,
        const std::vector<Real>& kinetic_energy_squared,
        const std::vector<Real>& smooth_potential, // Accepts precomputed smooth potential
        const Crystal& crystal_structure,
        const KPointsGrid& kpoint_grid,
        int kpoint_index,
        const PlanewaveBasis& planewave_basis,
        const execution::PlanewaveExecutionPolicy& execution_policy,
        const NormConservingPlanewaveKohnShamQuantumSystem& pseudopotential_quantum_system,
        const Matrix<Complex>& kleinman_bylander_potential,
        const Complex* psi_wavefunctions,
        Complex* hamiltonian_psi_wavefunctions) {
        // Use smooth (wavefunction) FFT grid for V·ψ product
        int local_smooth_grid_points = planewave_basis.fft_wave().local_grid_points();
        /**
         * Sized for whichever of the two layouts is larger: the scatter fills
         * reciprocal space, the transform leaves real space, and under the
         * stick layout those counts differ.
         */
        int transform_buffer_points =
            std::max(local_smooth_grid_points, planewave_basis.fft_wave().local_g_space_points());

        // Precompute G→smooth FFT mapping (constant across all vectors)
        std::vector<int> g_to_fft_mapping(number_of_plane_waves);
        for (int g_vector_index = 0; g_vector_index < number_of_plane_waves; g_vector_index++) {
            const auto& g_vector = planewave_basis.g_vectors()[g_vector_indices[g_vector_index]];
            g_to_fft_mapping[g_vector_index] =
                planewave_basis.fft_wave().g_vector_to_fft_grid_index(
                    g_vector.miller.x, g_vector.miller.y, g_vector.miller.z);
        }

        // Pre-compute -G FFT mapping for gamma-only
        std::vector<int> minus_g_to_fft_mapping;
        bool is_gamma_only = planewave_basis.is_gamma_only();
        /**
         * G = 0 is its own mirror. On the rank that holds it, it is index 0 and
         * is written once; on every other rank index 0 is an ordinary G with a
         * mirror of its own, so the mirrored loops start there.
         */
        int first_mirrored_index = planewave_basis.gamma_origin_is_local() ? 1 : 0;
        if (is_gamma_only) {
            minus_g_to_fft_mapping.resize(number_of_plane_waves);
            for (int g_vector_index = 0; g_vector_index < number_of_plane_waves; g_vector_index++) {
                const auto& g_vector =
                    planewave_basis.g_vectors()[g_vector_indices[g_vector_index]];
                minus_g_to_fft_mapping[g_vector_index] =
                    planewave_basis.fft_wave().g_vector_to_fft_grid_index(
                        -g_vector.miller.x, -g_vector.miller.y, -g_vector.miller.z);
            }
        }

        // Process vectors: kinetic + local potential via smooth FFT
        std::vector<Complex> psi_real_space(transform_buffer_points, Complex(0.0, 0.0));

        if (is_gamma_only) {
            // Gamma-only: two-bands-at-a-time
            for (int pair_index = 0; pair_index < number_of_vectors; pair_index += 2) {
                int band_1 = pair_index;
                int band_2 = pair_index + 1;
                bool has_second_band = (band_2 < number_of_vectors);

                const Complex* psi_1 = &psi_wavefunctions[band_1 * number_of_plane_waves];
                Complex* hpsi_1 = &hamiltonian_psi_wavefunctions[band_1 * number_of_plane_waves];
                const Complex* psi_2 =
                    has_second_band ? &psi_wavefunctions[band_2 * number_of_plane_waves] : nullptr;
                Complex* hpsi_2 =
                    has_second_band ? &hamiltonian_psi_wavefunctions[band_2 * number_of_plane_waves]
                                    : nullptr;

                // 1. Kinetic energy (done per-band, not part of the FFT trick)
                for (int g_index = 0; g_index < number_of_plane_waves; ++g_index) {
                    hpsi_1[g_index] = kinetic_energy_squared[g_index] * psi_1[g_index];
                }
                if (has_second_band) {
                    for (int g_index = 0; g_index < number_of_plane_waves; ++g_index) {
                        hpsi_2[g_index] = kinetic_energy_squared[g_index] * psi_2[g_index];
                    }
                }

                /**
                 * 2. Pack two bands: combined(G) = ψ₁(G) + i·ψ₂(G)
                 *    Scatter to FFT grid with full-sphere reconstruction:
                 *    grid[fft(+G)] = ψ₁(G) + i·ψ₂(G)
                 *    grid[fft(-G)] = conj(ψ₁(G)) + i·conj(ψ₂(G))
                 */
                std::memset(psi_real_space.data(), 0, transform_buffer_points * sizeof(Complex));

                if (has_second_band) {
                    if (planewave_basis.gamma_origin_is_local()) {
                        // G=0 component: (Re psi_1, Re psi_2), once
                        int fft_index_0 = g_to_fft_mapping[0];
                        if (fft_index_0 >= 0) {
                            psi_real_space[fft_index_0] = Complex(psi_1[0].real(), psi_2[0].real());
                        }
                    }
                    // Every other G: scatter +G and -G
                    for (int g_index = first_mirrored_index; g_index < number_of_plane_waves;
                         ++g_index) {
                        int fft_plus = g_to_fft_mapping[g_index];
                        int fft_minus = minus_g_to_fft_mapping[g_index];
                        if (fft_plus >= 0) {
                            // +G: ψ₁(G) + i·ψ₂(G)
                            Complex p1 = psi_1[g_index];
                            Complex p2 = psi_2[g_index];
                            psi_real_space[fft_plus] =
                                Complex(p1.real() - p2.imag(), p1.imag() + p2.real());
                            // -G: conj(ψ₁(G)) + i·conj(ψ₂(G))
                            psi_real_space[fft_minus] =
                                Complex(p1.real() + p2.imag(), -p1.imag() + p2.real());
                        }
                    }
                } else {
                    // Single band: standard gamma scatter
                    for (int g_index = 0; g_index < number_of_plane_waves; ++g_index) {
                        int fft_index = g_to_fft_mapping[g_index];
                        if (fft_index >= 0) {
                            psi_real_space[fft_index] = psi_1[g_index];
                        }
                    }
                    for (int g_index = first_mirrored_index; g_index < number_of_plane_waves;
                         ++g_index) {
                        int fft_minus = minus_g_to_fft_mapping[g_index];
                        psi_real_space[fft_minus] = std::conj(psi_1[g_index]);
                    }
                }

                // Inverse FFT → real space
                planewave_basis.fft_wave().inverse_raw(psi_real_space.data());

                /**
                 * Multiply by V(r): both bands simultaneously
                 * After inverse FFT: Re(grid) = ψ₁(r), Im(grid) = ψ₂(r)
                 * V·ψ₁ = V*Re, V·ψ₂ = V*Im → recombine as complex V*Re + i*V*Im
                 */
                for (int index_grid = 0; index_grid < local_smooth_grid_points; ++index_grid) {
                    psi_real_space[index_grid] *= smooth_potential[index_grid];
                }

                // Forward FFT → G-space
                planewave_basis.fft_wave().forward(psi_real_space.data());

                /**
                 * Separate the two bands using ±G symmetry:
                 * (V·ψ₁)(G) = [result(+G) + conj(result(-G))] / 2
                 * (V·ψ₂)(G) = [result(+G) - conj(result(-G))] / (2i)
                 */
                if (has_second_band) {
                    if (planewave_basis.gamma_origin_is_local()) {
                        // G=0 component
                        int fft_index_0 = g_to_fft_mapping[0];
                        if (fft_index_0 >= 0) {
                            Complex result_0 = psi_real_space[fft_index_0];
                            hpsi_1[0] += Complex(result_0.real(), 0.0);
                            hpsi_2[0] += Complex(result_0.imag(), 0.0);
                        }
                    }
                    // Every other G: use ±G symmetry
                    for (int g_index = first_mirrored_index; g_index < number_of_plane_waves;
                         ++g_index) {
                        int fft_plus = g_to_fft_mapping[g_index];
                        int fft_minus = minus_g_to_fft_mapping[g_index];
                        if (fft_plus >= 0) {
                            Complex result_plus = psi_real_space[fft_plus];
                            Complex result_minus_conj = std::conj(psi_real_space[fft_minus]);
                            // (V·ψ₁)(G) = (result+ + conj(result-)) / 2
                            hpsi_1[g_index] += (result_plus + result_minus_conj) * 0.5;
                            /**
                             * (V·ψ₂)(G) = (result+ - conj(result-)) / (2i)
                             *            = -i * (result+ - conj(result-)) / 2
                             */
                            Complex diff = (result_plus - result_minus_conj) * 0.5;
                            hpsi_2[g_index] += Complex(diff.imag(), -diff.real());
                        }
                    }
                } else {
                    // Single band: standard gather
                    for (int g_index = 0; g_index < number_of_plane_waves; ++g_index) {
                        int fft_index = g_to_fft_mapping[g_index];
                        if (fft_index >= 0) {
                            hpsi_1[g_index] += psi_real_space[fft_index];
                        }
                    }
                }
            }
        } else {
            // Non-gamma: process one band at a time (original path)
            for (int vector_index = 0; vector_index < number_of_vectors; ++vector_index) {
                const Complex* psi_vector =
                    &psi_wavefunctions[vector_index * number_of_plane_waves];
                Complex* hamiltonian_psi_vector =
                    &hamiltonian_psi_wavefunctions[vector_index * number_of_plane_waves];

                // 1. Kinetic energy
                for (int g_index = 0; g_index < number_of_plane_waves; ++g_index) {
                    hamiltonian_psi_vector[g_index] =
                        kinetic_energy_squared[g_index] * psi_vector[g_index];
                }

                // 2. Local potential via smooth grid FFT
                std::memset(psi_real_space.data(), 0, transform_buffer_points * sizeof(Complex));
                for (int g_index = 0; g_index < number_of_plane_waves; ++g_index) {
                    int fft_index = g_to_fft_mapping[g_index];
                    if (fft_index >= 0) {
                        psi_real_space[fft_index] = psi_vector[g_index];
                    }
                }
                planewave_basis.fft_wave().inverse_raw(psi_real_space.data());
                for (int index_grid = 0; index_grid < local_smooth_grid_points; ++index_grid) {
                    psi_real_space[index_grid] *= smooth_potential[index_grid];
                }
                planewave_basis.fft_wave().forward(psi_real_space.data());
                for (int g_index = 0; g_index < number_of_plane_waves; ++g_index) {
                    int fft_index = g_to_fft_mapping[g_index];
                    if (fft_index >= 0) {
                        hamiltonian_psi_vector[g_index] += psi_real_space[fft_index];
                    }
                }
            }
        }

        // 3. Nonlocal PP (KB projectors) via batched ZGEMM for all atoms at once
        if (kleinman_bylander_potential.empty()) {
            return;
        }

        int total_number_of_projectors = static_cast<int>(kleinman_bylander_potential.rows());

        // Step 1: ⟨β_all|ψ⟩
        std::vector<Complex> all_overlap_coefficients(
            static_cast<size_t>(total_number_of_projectors) * number_of_vectors, Complex(0.0, 0.0));

        std::complex<double> alpha_one(1.0, 0.0);
        std::complex<double> beta_zero(0.0, 0.0);
        std::complex<double> beta_one(1.0, 0.0);

        if (is_gamma_only) {
            // Gamma-only: DGEMM on interleaved real/imag parts
            int double_number_of_plane_waves = 2 * number_of_plane_waves;
            int double_leading_dimension = std::max(double_number_of_plane_waves, 1);
            double alpha_two = 2.0;
            double d_beta_zero = 0.0;

            auto* real_projectors = reinterpret_cast<const double*>(kleinman_bylander_potential[0]);
            auto* real_wavefunctions = reinterpret_cast<const double*>(psi_wavefunctions);

            std::vector<double> overlap_real(
                static_cast<size_t>(total_number_of_projectors) * number_of_vectors, 0.0);
            dgemm_("T",
                   "N",
                   &total_number_of_projectors,
                   &number_of_vectors,
                   &double_number_of_plane_waves,
                   &alpha_two,
                   const_cast<double*>(real_projectors),
                   &double_leading_dimension,
                   const_cast<double*>(real_wavefunctions),
                   &double_leading_dimension,
                   &d_beta_zero,
                   overlap_real.data(),
                   &total_number_of_projectors);

            // Remove the origin from the double count where this rank holds it; copy to Complex
            for (int vector_index = 0; vector_index < number_of_vectors; ++vector_index) {
                const Complex* psi_g0 = &psi_wavefunctions[vector_index * number_of_plane_waves];
                for (int projector_index = 0; projector_index < total_number_of_projectors;
                     ++projector_index) {
                    int index = projector_index + vector_index * total_number_of_projectors;
                    if (planewave_basis.gamma_origin_is_local()) {
                        Complex beta_g0 = kleinman_bylander_potential(projector_index, 0);
                        overlap_real[index] -= (std::conj(beta_g0) * psi_g0[0]).real();
                    }
                    all_overlap_coefficients[index] = Complex(overlap_real[index], 0.0);
                }
            }
        } else {
            int leading_dimension = std::max(number_of_plane_waves, 1);
            // Non-gamma: ZGEMM
            zgemm_("C",
                   "N",
                   &total_number_of_projectors,
                   &number_of_vectors,
                   &number_of_plane_waves,
                   &alpha_one,
                   reinterpret_cast<const std::complex<double>*>(kleinman_bylander_potential[0]),
                   &leading_dimension,
                   reinterpret_cast<const std::complex<double>*>(psi_wavefunctions),
                   &leading_dimension,
                   &beta_zero,
                   reinterpret_cast<std::complex<double>*>(all_overlap_coefficients.data()),
                   &total_number_of_projectors);
        }

        /**
         * Each rank summed over its own plane waves; the overlap is the sum
         * over the pool. A no-op while the coefficients are replicated, which
         * is why its absence went unnoticed: it only changes the starting
         * guess, and the SCF converges to the same answer from a wrong one.
         */
        execution_policy.reduce_within_pool_vector(
            reinterpret_cast<std::vector<Real>&>(all_overlap_coefficients));

        // Step 2: Apply block-diagonal D-matrix per atom
        std::vector<Complex> scaled_overlap_coefficients(
            static_cast<size_t>(total_number_of_projectors) * number_of_vectors, Complex(0.0, 0.0));

        int starting_projector_index = 0;
        for (int atom_index = 0; atom_index < crystal_structure.number_of_atoms(); atom_index++) {
            int species_index = crystal_structure.species_type_index(atom_index);
            int number_of_projectors =
                pseudopotential_quantum_system.pseudopotential_basis().number_of_projectors(
                    species_index);
            std::vector<int> angular_momentum_values;
            for (int index_projector = 0; index_projector < number_of_projectors;
                 index_projector++) {
                int angular_momentum =
                    pseudopotential_quantum_system.pseudopotential_basis().beta_angular_momentum(
                        species_index, index_projector);
                for (int magnetic_projection : harmonic_order_sequence(angular_momentum)) {
                    (void)magnetic_projection;
                    angular_momentum_values.push_back(angular_momentum);
                }
            }
            int number_of_projectors_atom = static_cast<int>(angular_momentum_values.size());

            if (number_of_projectors_atom > 0) {
                for (int vector_index = 0; vector_index < number_of_vectors; vector_index++) {
                    int offset_in =
                        vector_index * total_number_of_projectors + starting_projector_index;
                    int offset_out =
                        vector_index * total_number_of_projectors + starting_projector_index;
                    for (int projector_index_a = 0; projector_index_a < number_of_projectors_atom;
                         projector_index_a++) {
                        for (int projector_index_b = 0;
                             projector_index_b < number_of_projectors_atom;
                             projector_index_b++) {
                            Real d_matrix_element = pseudopotential_quantum_system.d_matrix_element(
                                atom_index, species_index, projector_index_a, projector_index_b);
                            if (std::abs(d_matrix_element) >
                                math::tolerances::d_matrix_element_threshold) {
                                scaled_overlap_coefficients[offset_out + projector_index_a] +=
                                    d_matrix_element *
                                    all_overlap_coefficients[offset_in + projector_index_b];
                            }
                        }
                    }
                }
                starting_projector_index += number_of_projectors_atom;
            }
        }

        int leading_dimension_apply = std::max(number_of_plane_waves, 1);
        // Step 3: H|ψ⟩ += β_all · w via single ZGEMM('N','N', npw, n_vectors, total_proj)
        zgemm_("N",
               "N",
               &number_of_plane_waves,
               &number_of_vectors,
               &total_number_of_projectors,
               &alpha_one,
               reinterpret_cast<const std::complex<double>*>(kleinman_bylander_potential[0]),
               &leading_dimension_apply,
               reinterpret_cast<const std::complex<double>*>(scaled_overlap_coefficients.data()),
               &total_number_of_projectors,
               &beta_one,
               reinterpret_cast<std::complex<double>*>(hamiltonian_psi_wavefunctions),
               &leading_dimension_apply);
    }

    /**
     * Apply S|ψ⟩ for a batch of vectors using contiguous layout.
     *
     * For NC pseudopotentials, S = I so this is a no-op (caller already memcpy'd).
     * For US pseudopotentials, adds the augmentation overlap contribution.
     *
     * Input/output layout: column-major, psi[g + vector * npw].
     * The output overlap_psi is modified in-place (should be pre-initialized to psi).
     */
    static void apply_overlap_contiguous(
        int number_of_plane_waves,
        int number_of_vectors,
        const Crystal& crystal_structure,
        const execution::PlanewaveExecutionPolicy& execution_policy,
        const NormConservingPlanewaveKohnShamQuantumSystem& pseudopotential_quantum_system,
        const Matrix<Complex>& kleinman_bylander_potential,
        const Complex* psi_wavefunctions,
        Complex* overlap_psi_wavefunctions) {
        if (!pseudopotential_quantum_system.has_ultrasoft()) {
            return;
        }

        int total_number_of_projectors = static_cast<int>(kleinman_bylander_potential.rows());

        // Step 1: ⟨β_all|ψ⟩
        std::vector<Complex> all_overlap_coefficients(
            static_cast<size_t>(total_number_of_projectors) * number_of_vectors, Complex(0.0, 0.0));

        std::complex<double> alpha_one(1.0, 0.0);
        std::complex<double> beta_zero(0.0, 0.0);
        std::complex<double> beta_one(1.0, 0.0);

        bool is_gamma_only = pseudopotential_quantum_system.basis().is_gamma_only();

        if (is_gamma_only) {
            int double_number_of_plane_waves = 2 * number_of_plane_waves;
            int double_leading_dimension = std::max(double_number_of_plane_waves, 1);
            double alpha_two = 2.0;
            double double_beta_zero = 0.0;

            auto* real_projectors = reinterpret_cast<const double*>(kleinman_bylander_potential[0]);
            auto* real_wavefunctions = reinterpret_cast<const double*>(psi_wavefunctions);

            std::vector<double> overlap_real(
                static_cast<size_t>(total_number_of_projectors) * number_of_vectors, 0.0);
            dgemm_("T",
                   "N",
                   &total_number_of_projectors,
                   &number_of_vectors,
                   &double_number_of_plane_waves,
                   &alpha_two,
                   const_cast<double*>(real_projectors),
                   &double_leading_dimension,
                   const_cast<double*>(real_wavefunctions),
                   &double_leading_dimension,
                   &double_beta_zero,
                   overlap_real.data(),
                   &total_number_of_projectors);

            // G=0 correction, on the rank whose list begins with the origin
            bool has_zero_g_vector = pseudopotential_quantum_system.basis().gamma_origin_is_local();
            if (has_zero_g_vector) {
                for (int vector_index = 0; vector_index < number_of_vectors; ++vector_index) {
                    const Complex* psi_g0 =
                        &psi_wavefunctions[vector_index * number_of_plane_waves];
                    for (int projector_index = 0; projector_index < total_number_of_projectors;
                         ++projector_index) {
                        Complex beta_g0 = kleinman_bylander_potential(projector_index, 0);
                        int index = projector_index + vector_index * total_number_of_projectors;
                        overlap_real[index] -= (std::conj(beta_g0) * psi_g0[0]).real();
                        all_overlap_coefficients[index] = Complex(overlap_real[index], 0.0);
                    }
                }
            } else {
                for (size_t index = 0; index < overlap_real.size(); ++index) {
                    all_overlap_coefficients[index] = Complex(overlap_real[index], 0.0);
                }
            }
        } else {
            int leading_dimension = std::max(number_of_plane_waves, 1);
            zgemm_("C",
                   "N",
                   &total_number_of_projectors,
                   &number_of_vectors,
                   &number_of_plane_waves,
                   &alpha_one,
                   reinterpret_cast<const std::complex<double>*>(kleinman_bylander_potential[0]),
                   &leading_dimension,
                   reinterpret_cast<const std::complex<double>*>(psi_wavefunctions),
                   &leading_dimension,
                   &beta_zero,
                   reinterpret_cast<std::complex<double>*>(all_overlap_coefficients.data()),
                   &total_number_of_projectors);
        }

        // Reduce overlap coefficients across band ranks within this pool
        execution_policy.reduce_within_pool_vector(
            reinterpret_cast<std::vector<Real>&>(all_overlap_coefficients));

        // Step 2: Apply q_ij augmentation overlap per atom
        std::vector<Complex> scaled_overlap_coefficients(
            static_cast<size_t>(total_number_of_projectors) * number_of_vectors, Complex(0.0, 0.0));

        int starting_projector_index = 0;
        for (int atom_index = 0; atom_index < crystal_structure.number_of_atoms(); atom_index++) {
            int element_type = crystal_structure.species_type_index(atom_index);
            int number_of_beta_functions =
                pseudopotential_quantum_system.pseudopotential_basis().number_of_projectors(
                    element_type);
            std::vector<int> angular_momentums, magnetic_projections, beta_indices;
            for (int beta_index = 0; beta_index < number_of_beta_functions; beta_index++) {
                int angular_momentum =
                    pseudopotential_quantum_system.pseudopotential_basis().beta_angular_momentum(
                        element_type, beta_index);
                for (int magnetic_projection : harmonic_order_sequence(angular_momentum)) {
                    angular_momentums.push_back(angular_momentum);
                    magnetic_projections.push_back(magnetic_projection);
                    beta_indices.push_back(beta_index);
                }
            }
            int number_of_projectors_on_atom = static_cast<int>(angular_momentums.size());
            if (number_of_projectors_on_atom > 0) {
                for (int vector_index = 0; vector_index < number_of_vectors; vector_index++) {
                    int offset =
                        vector_index * total_number_of_projectors + starting_projector_index;
                    for (int projector_a_index = 0;
                         projector_a_index < number_of_projectors_on_atom;
                         projector_a_index++) {
                        for (int projector_b_index = 0;
                             projector_b_index < number_of_projectors_on_atom;
                             projector_b_index++) {
                            int beta_a = beta_indices[projector_a_index];
                            int beta_b = beta_indices[projector_b_index];
                            Real q_value = 0.0;
                            if (angular_momentums[projector_a_index] ==
                                    angular_momentums[projector_b_index] &&
                                magnetic_projections[projector_a_index] ==
                                    magnetic_projections[projector_b_index]) {
                                q_value =
                                    static_cast<const UltrasoftPlanewaveKohnShamQuantumSystem&>(
                                        pseudopotential_quantum_system)
                                        .ultrasoft_basis()
                                        .augmentation_overlap_integral(
                                            element_type, beta_a, beta_b);
                            }
                            if (std::abs(q_value) > math::tolerances::d_matrix_element_threshold) {
                                scaled_overlap_coefficients[offset + projector_a_index] +=
                                    q_value * all_overlap_coefficients[offset + projector_b_index];
                            }
                        }
                    }
                }
                starting_projector_index += number_of_projectors_on_atom;
            }
        }

        int leading_dimension_step3 = std::max(number_of_plane_waves, 1);
        // Step 3: S|ψ⟩ += β_all · w via ZGEMM
        zgemm_("N",
               "N",
               &number_of_plane_waves,
               &number_of_vectors,
               &total_number_of_projectors,
               &alpha_one,
               reinterpret_cast<const std::complex<double>*>(kleinman_bylander_potential[0]),
               &leading_dimension_step3,
               reinterpret_cast<const std::complex<double>*>(scaled_overlap_coefficients.data()),
               &total_number_of_projectors,
               &beta_one,
               reinterpret_cast<std::complex<double>*>(overlap_psi_wavefunctions),
               &leading_dimension_step3);
    }
};

} // namespace q3
