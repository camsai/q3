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
#include <optional>
#include <string>
#include <vector>

#include "layer_0_core/execution/threading.hpp"
#include "layer_0_core/logging/init.hpp"
#include "layer_0_core/math/integration/simpson.hpp"
#include "layer_0_core/math/interpolation/lagrange.hpp"
#include "layer_0_core/math/linear_algebra/blas.hpp"
#include "layer_0_core/math/phase_factor.hpp"
#include "layer_0_core/math/spherical/bessel.hpp"
#include "layer_0_core/math/spherical/harmonics.hpp"
#include "layer_0_core/math/tolerances.hpp"
#include "layer_0_core/profiling/stage_timer.hpp"
#include "layer_0_core/types/init.hpp"
#include "layer_1_structure/crystal/crystal.hpp"
#include "layer_1_structure/reciprocal_space/kpoints_grid.hpp"
#include "layer_3_representation/basis/planewave/init.hpp"
#include "layer_3_representation/pseudopotential/planewave/internal/atomic_orbital_builder.hpp"
#include "layer_3_representation/pseudopotential/planewave/internal/wavefunction_atomic.hpp"
#include "layer_3_representation/pseudopotential/planewave/kleinman_bylander_projector_set.hpp"
#include "layer_3_representation/pseudopotential/set.hpp"
#include "layer_3_representation/quantum_system/kohn_sham/planewave/init.hpp"
#include "layer_3_representation/quantum_system/kohn_sham/planewave/norm_conserving/init.hpp"
#include "layer_3_representation/quantum_system/kohn_sham/planewave/ultrasoft/init.hpp"
#include "layer_3_representation/wavefunction/planewave/starting_subspace_solver.hpp"

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
        /**
         * The orbitals are moved in, and only the vectors beyond them are allocated here.
         * Allocating and zeroing every vector first, to replace most of them a line later,
         * was 200 MB per k-point on a 64-atom cell for nothing.
         */
        atomic_wavefunctions.clear();
        atomic_wavefunctions.resize(static_cast<std::size_t>(total_wavefunction_count));
        for (int wavefunction_index = 0; wavefunction_index < number_of_atomic_wavefunctions;
             wavefunction_index++) {
            atomic_wavefunctions[wavefunction_index] =
                std::move(pure_atomic_orbitals[wavefunction_index]);
        }
        for (int wavefunction_index = number_of_atomic_wavefunctions;
             wavefunction_index < total_wavefunction_count;
             wavefunction_index++) {
            atomic_wavefunctions[wavefunction_index].assign(
                static_cast<std::size_t>(number_of_plane_waves), Complex(0.0, 0.0));
        }

        /**
         * No perturbation is applied when the atomic orbitals already cover
         * the band count. A hook for one stood here with its coefficient at
         * 0.0: it multiplied every coefficient by exactly 1, and to do that
         * drew two random numbers and took a sine and a cosine per plane wave
         * per vector -- most of this method's time on a large cell. When the
         * orbitals fall short, the random padding below breaks the same
         * degeneracy for free.
         */
        if (number_of_atomic_wavefunctions < number_of_bands) {
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
     * `subspace_solver` does the three operations that scale with the plane-wave count --
     * H on the starting vectors, the two subspace projections and the rotation -- and the
     * caller has prepared it for this k-point and potential. What is left here is the same
     * whoever does them: packing the vectors, the small eigenproblem, and the eigenvalues.
     * SerialStartingSubspaceSolver is the host implementation; a diagonalizer with a device
     * offers its own.
     */
    void rotate_wavefunctions(int kpoint_index_state,
                              int number_of_plane_waves,
                              int number_of_bands,
                              const PlanewaveBasis& planewave_basis,
                              const execution::PlanewaveExecutionPolicy& execution_policy,
                              const std::vector<std::vector<Complex>>& atomic_wavefunctions,
                              PlanewaveKohnShamQuantumSystem& quantum_system,
                              StartingSubspaceSolver& subspace_solver) {
        StageTimer whole_rotation_timer("setup.rotation.initializer");
        std::optional<StageTimer> pack_timer(std::in_place, "setup.rotation.pack");
        int number_of_starting_wavefunctions = static_cast<int>(atomic_wavefunctions.size());

        /**
         * Pack atomic wavefunctions into contiguous column-major matrix:
         * contiguous_wavefunctions[g + wfc * npw] = atomic_wavefunctions[wfc][g]
         */
        std::vector<Complex> contiguous_wavefunctions(static_cast<size_t>(number_of_plane_waves) *
                                                          number_of_starting_wavefunctions,
                                                      Complex(0.0, 0.0));
        // One vector per thread at a time: the copy is as large as the starting set itself.
        int pack_threads =
            execution::threading::thread_count_for(number_of_starting_wavefunctions, 1);
#pragma omp parallel for schedule(static) num_threads(pack_threads) if (pack_threads > 1)
        for (int wavefunction_index = 0; wavefunction_index < number_of_starting_wavefunctions;
             wavefunction_index++) {
            std::memcpy(&contiguous_wavefunctions[static_cast<std::size_t>(wavefunction_index) *
                                                  number_of_plane_waves],
                        atomic_wavefunctions[wavefunction_index].data(),
                        number_of_plane_waves * sizeof(Complex));
        }

        pack_timer.reset();
        // The solver keeps χ and Hχ where it computes them; see the interface.
        std::optional<StageTimer> hamiltonian_timer(std::in_place, "setup.rotation.hamiltonian");
        subspace_solver.apply_hamiltonian(contiguous_wavefunctions.data(),
                                          number_of_starting_wavefunctions);

        // H_ij = χ†·Hχ and S_ij = χ†·Sχ
        hamiltonian_timer.reset();
        std::optional<StageTimer> subspace_timer(std::in_place, "setup.rotation.subspace_matrices");
        std::vector<Complex> hamiltonian_matrix(
            static_cast<size_t>(number_of_starting_wavefunctions) *
                number_of_starting_wavefunctions,
            Complex(0.0, 0.0));
        std::vector<Complex> overlap_matrix(static_cast<size_t>(number_of_starting_wavefunctions) *
                                                number_of_starting_wavefunctions,
                                            Complex(0.0, 0.0));
        subspace_solver.project(hamiltonian_matrix.data(), overlap_matrix.data());

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
        subspace_timer.reset();
        std::optional<StageTimer> eigensolver_timer(std::in_place, "setup.rotation.eigensolver");
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

        eigensolver_timer.reset();
        if (info != 0) {
            log::error("LAPACK eigensolver failed in rotate_wavefunctions with info = {}", info);
        }
        StageTimer rotation_timer("setup.rotation.rotate");

        // Store eigenvalues
        for (int band_index = 0; band_index < number_of_bands; band_index++) {
            quantum_system.set_eigenvalue(
                kpoint_index_state, band_index, subspace_eigenvalues[band_index]);
        }

        /**
         * Rotate wavefunctions: evc = χ · eigenvectors
         * evc(npw × n_bands) = χ(npw × n_wfc) · c(n_wfc × n_bands)
         * The eigenvectors are stored column-major in hamiltonian_matrix after zhegv.
         * The solver does the product where it holds χ and stores the bands in the quantum
         * system itself.
         */
        subspace_solver.rotate_into_bands(
            hamiltonian_matrix.data(), number_of_bands, quantum_system, kpoint_index_state);

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
                Complex& g_zero_coefficient = quantum_system.mutable_wavefunction_band_data(
                    kpoint_index_state, band_index)[0];
                g_zero_coefficient = Complex(g_zero_coefficient.real(), 0.0);
            }
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
};

} // namespace q3
