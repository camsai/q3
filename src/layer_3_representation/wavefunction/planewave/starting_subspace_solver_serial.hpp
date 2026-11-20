#pragma once

/**
 * SerialStartingSubspaceSolver — the starting rotation's large operations on the host.
 *
 * The implementation of StartingSubspaceSolver for a run with nothing faster: H on the
 * starting vectors through the wavefunction FFT grid and BLAS, the subspace projections
 * as BLAS products, and the rotation as one more. It is constructed for one k-point and
 * one local potential and reads the starting vectors where the caller holds them, so it
 * allocates Hχ, S·χ where S is not the identity, and the rotated bands, and nothing else.
 *
 * `smooth_potential` is the local potential on this rank's part of the wavefunction
 * grid, already multiplied by 1/N of that grid -- what the Hamiltonian application
 * holds, not the density-grid potential the SCF stores.
 */

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstring>
#include <vector>

#include "layer_0_core/execution/threading.hpp"
#include "layer_0_core/logging/init.hpp"
#include "layer_0_core/math/linear_algebra/blas.hpp"
#include "layer_0_core/math/tolerances.hpp"
#include "layer_0_core/types/init.hpp"
#include "layer_1_structure/crystal/crystal.hpp"
#include "layer_1_structure/reciprocal_space/kpoints_grid.hpp"
#include "layer_3_representation/basis/planewave/init.hpp"
#include "layer_3_representation/pseudopotential/planewave/kleinman_bylander_projector_set.hpp"
#include "layer_3_representation/quantum_system/kohn_sham/planewave/init.hpp"
#include "layer_3_representation/quantum_system/kohn_sham/planewave/norm_conserving/init.hpp"
#include "layer_3_representation/quantum_system/kohn_sham/planewave/ultrasoft/init.hpp"
#include "layer_3_representation/wavefunction/planewave/starting_subspace_solver.hpp"

namespace q3 {
using namespace q3::blas;

/// H, the subspace projections and the rotation of one k-point's starting vectors, on the host.
class SerialStartingSubspaceSolver : public StartingSubspaceSolver {
  public:
    /// Everything given by reference must outlive this solver.
    SerialStartingSubspaceSolver(
        int kpoint_index,
        int number_of_plane_waves,
        const std::vector<int>& g_vector_indices,
        const std::vector<Real>& kinetic_energy_squared,
        const std::vector<Real>& smooth_potential,
        const Crystal& crystal_structure,
        const KPointsGrid& kpoint_grid,
        const PlanewaveBasis& planewave_basis,
        const execution::PlanewaveExecutionPolicy& execution_policy,
        const NormConservingPlanewaveKohnShamQuantumSystem& pseudopotential_quantum_system,
        const KleinmanBylanderProjectorSet& kleinman_bylander_projector_set)
        : kpoint_index_{kpoint_index}
        , number_of_plane_waves_{number_of_plane_waves}
        , g_vector_indices_{g_vector_indices}
        , kinetic_energy_squared_{kinetic_energy_squared}
        , smooth_potential_{smooth_potential}
        , crystal_structure_{crystal_structure}
        , kpoint_grid_{kpoint_grid}
        , planewave_basis_{planewave_basis}
        , execution_policy_{execution_policy}
        , pseudopotential_quantum_system_{pseudopotential_quantum_system}
        , kleinman_bylander_projector_set_{kleinman_bylander_projector_set} {}

    /**
     * Apply H to the vectors, on the smooth grid with ZGEMM for the projectors. The
     * vectors are read where they are, here and in the two calls that follow, so they
     * must stay unchanged until rotate_into_bands() has returned.
     */
    void apply_hamiltonian(const Complex* vectors, int number_of_vectors) override {
        vectors_ = vectors;
        number_of_vectors_ = number_of_vectors;
        hamiltonian_vectors_.assign(
            static_cast<size_t>(number_of_plane_waves_) * number_of_vectors_, Complex(0.0, 0.0));
        apply_hamiltonian_batch_contiguous(number_of_plane_waves_,
                                           number_of_vectors_,
                                           g_vector_indices_,
                                           kinetic_energy_squared_,
                                           smooth_potential_,
                                           crystal_structure_,
                                           kpoint_grid_,
                                           kpoint_index_,
                                           planewave_basis_,
                                           execution_policy_,
                                           pseudopotential_quantum_system_,
                                           kleinman_bylander_projector_set_,
                                           vectors_,
                                           hamiltonian_vectors_.data());
    }

    /// H_ij = χ†·Hχ and S_ij = χ†·Sχ as BLAS products, real ones in the gamma-point representation.
    void project(Complex* hamiltonian_matrix, Complex* overlap_matrix) override {
        /**
         * Apply S to every starting vector. For norm-conserving pseudopotentials
         * S = I, so the overlap-applied set IS the input set and nothing is
         * allocated: everything downstream reads it, so pointing at the input
         * is enough. Ultrasoft needs the q_ij augmentation and so needs its own
         * buffer. This block is the tallest point of a run's setup, and memory
         * released here is not returned to the system, so the array not taken
         * is worth more than its lifetime suggests.
         */
        std::vector<Complex> overlap_storage;
        const Complex* overlap_vectors = vectors_;
        if (pseudopotential_quantum_system_.has_ultrasoft()) {
            // US overlap: S|ψ⟩ = |ψ⟩ + Σ_ab q_ab |β_a⟩⟨β_b|ψ⟩
            overlap_storage.assign(vectors_,
                                   vectors_ + static_cast<size_t>(number_of_plane_waves_) *
                                                  number_of_vectors_);
            apply_overlap_contiguous(number_of_plane_waves_,
                                     number_of_vectors_,
                                     crystal_structure_,
                                     execution_policy_,
                                     pseudopotential_quantum_system_,
                                     kleinman_bylander_projector_set_,
                                     vectors_,
                                     overlap_storage.data());
            overlap_vectors = overlap_storage.data();
        }

        std::complex<double> alpha_one(1.0, 0.0);
        std::complex<double> beta_zero(0.0, 0.0);
        auto* cast_wavefunctions = reinterpret_cast<const std::complex<double>*>(vectors_);
        auto* cast_hamiltonian_wavefunctions =
            reinterpret_cast<const std::complex<double>*>(hamiltonian_vectors_.data());
        auto* cast_overlap_wavefunctions =
            reinterpret_cast<const std::complex<double>*>(overlap_vectors);

        if (planewave_basis_.is_gamma_only()) {
            /**
             * Gamma-only: use DGEMM on interleaved real/imaginary parts.
             * Reinterpret complex arrays as double arrays of length 2*npw.
             * DGEMM("T","N", n, n, 2*npw, alpha=2) computes 2*Re(Σ conj(a)*b)
             * which is the gamma inner product before G=0 correction.
             */
            int leading_dimension = std::max(number_of_plane_waves_, 1);
            int double_number_of_plane_waves = 2 * number_of_plane_waves_;
            int double_leading_dimension = 2 * leading_dimension;
            double alpha_two = 2.0;
            double d_beta_zero = 0.0;

            auto* real_wavefunctions = reinterpret_cast<const double*>(vectors_);
            auto* real_hamiltonian_wavefunctions =
                reinterpret_cast<const double*>(hamiltonian_vectors_.data());
            auto* real_overlap_wavefunctions = reinterpret_cast<const double*>(overlap_vectors);

            // H_ij = 2*Re(χ†·Hχ) via DGEMM
            std::vector<double> hamiltonian_matrix_real(
                static_cast<size_t>(number_of_vectors_) * number_of_vectors_, 0.0);
            dgemm_("T",
                   "N",
                   &number_of_vectors_,
                   &number_of_vectors_,
                   &double_number_of_plane_waves,
                   &alpha_two,
                   real_wavefunctions,
                   &double_leading_dimension,
                   real_hamiltonian_wavefunctions,
                   &double_leading_dimension,
                   &d_beta_zero,
                   hamiltonian_matrix_real.data(),
                   &number_of_vectors_);

            // S_ij = 2*Re(χ†·Sχ) via DGEMM
            std::vector<double> overlap_matrix_real(
                static_cast<size_t>(number_of_vectors_) * number_of_vectors_, 0.0);
            dgemm_("T",
                   "N",
                   &number_of_vectors_,
                   &number_of_vectors_,
                   &double_number_of_plane_waves,
                   &alpha_two,
                   real_wavefunctions,
                   &double_leading_dimension,
                   real_overlap_wavefunctions,
                   &double_leading_dimension,
                   &d_beta_zero,
                   overlap_matrix_real.data(),
                   &number_of_vectors_);

            // G=0 correction: subtract Re(conj(left_g0)*right_g0)
            if (planewave_basis_.gamma_origin_is_local()) {
                for (int column = 0; column < number_of_vectors_; ++column) {
                    Complex h_right_g0 = hamiltonian_vectors_[column * number_of_plane_waves_];
                    Complex s_right_g0 = overlap_vectors[column * number_of_plane_waves_];
                    for (int row = 0; row < number_of_vectors_; ++row) {
                        Complex left_g0 = vectors_[row * number_of_plane_waves_];
                        int index = column * number_of_vectors_ + row;
                        hamiltonian_matrix_real[index] -= (std::conj(left_g0) * h_right_g0).real();
                        overlap_matrix_real[index] -= (std::conj(left_g0) * s_right_g0).real();
                    }
                }
            }

            // Copy to Complex matrices for zhegv
            for (int index = 0; index < number_of_vectors_ * number_of_vectors_; ++index) {
                hamiltonian_matrix[index] = Complex(hamiltonian_matrix_real[index], 0.0);
                overlap_matrix[index] = Complex(overlap_matrix_real[index], 0.0);
            }
        } else {
            int leading_dimension = std::max(number_of_plane_waves_, 1);
            /**
             * Non-gamma: use ZGEMM
             * H_ij = χ†·Hχ: ZGEMM('C', 'N', n_wfc, n_wfc, npw)
             */
            zgemm_("C",
                   "N",
                   &number_of_vectors_,
                   &number_of_vectors_,
                   &number_of_plane_waves_,
                   &alpha_one,
                   cast_wavefunctions,
                   &leading_dimension,
                   cast_hamiltonian_wavefunctions,
                   &leading_dimension,
                   &beta_zero,
                   reinterpret_cast<std::complex<double>*>(hamiltonian_matrix),
                   &number_of_vectors_);

            // S_ij = χ†·Sχ: ZGEMM('C', 'N', n_wfc, n_wfc, npw)
            zgemm_("C",
                   "N",
                   &number_of_vectors_,
                   &number_of_vectors_,
                   &number_of_plane_waves_,
                   &alpha_one,
                   cast_wavefunctions,
                   &leading_dimension,
                   cast_overlap_wavefunctions,
                   &leading_dimension,
                   &beta_zero,
                   reinterpret_cast<std::complex<double>*>(overlap_matrix),
                   &number_of_vectors_);
        }
    }

    /**
     * evc = χ · c as one host ZGEMM, copied into the k-point state's bands.
     */
    void rotate_into_bands(const Complex* coefficients,
                           int number_of_bands,
                           PlanewaveKohnShamQuantumSystem& quantum_system,
                           int kpoint_index_state) override {
        int leading_dimension = std::max(number_of_plane_waves_, 1);
        std::complex<double> alpha_one(1.0, 0.0);
        std::complex<double> beta_zero(0.0, 0.0);
        std::vector<Complex> rotated_wavefunctions(
            static_cast<size_t>(number_of_plane_waves_) * number_of_bands, Complex(0.0, 0.0));

        zgemm_("N",
               "N",
               &number_of_plane_waves_,
               &number_of_bands,
               &number_of_vectors_,
               &alpha_one,
               reinterpret_cast<const std::complex<double>*>(vectors_),
               &leading_dimension,
               reinterpret_cast<const std::complex<double>*>(coefficients),
               &number_of_vectors_,
               &beta_zero,
               reinterpret_cast<std::complex<double>*>(rotated_wavefunctions.data()),
               &leading_dimension);

        for (int band_index = 0; band_index < number_of_bands; band_index++) {
            Complex* band_data =
                quantum_system.mutable_wavefunction_band_data(kpoint_index_state, band_index);
            std::memcpy(band_data,
                        &rotated_wavefunctions[band_index * number_of_plane_waves_],
                        number_of_plane_waves_ * sizeof(Complex));
        }
    }

  private:
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
        const KleinmanBylanderProjectorSet& kleinman_bylander_projector_set,
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
        std::optional<StageTimer> local_timer(std::in_place, "setup.rotation.hamiltonian.local");
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
                planewave_basis.fft_wave().wavefunction_inverse_raw(psi_real_space.data());

                /**
                 * Multiply by V(r): both bands simultaneously
                 * After inverse FFT: Re(grid) = ψ₁(r), Im(grid) = ψ₂(r)
                 * V·ψ₁ = V*Re, V·ψ₂ = V*Im → recombine as complex V*Re + i*V*Im
                 */
                for (int index_grid = 0; index_grid < local_smooth_grid_points; ++index_grid) {
                    psi_real_space[index_grid] *= smooth_potential[index_grid];
                }

                // Forward FFT → G-space
                planewave_basis.fft_wave().wavefunction_forward(psi_real_space.data());

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
                planewave_basis.fft_wave().wavefunction_inverse_raw(psi_real_space.data());
                for (int index_grid = 0; index_grid < local_smooth_grid_points; ++index_grid) {
                    psi_real_space[index_grid] *= smooth_potential[index_grid];
                }
                planewave_basis.fft_wave().wavefunction_forward(psi_real_space.data());
                for (int g_index = 0; g_index < number_of_plane_waves; ++g_index) {
                    int fft_index = g_to_fft_mapping[g_index];
                    if (fft_index >= 0) {
                        hamiltonian_psi_vector[g_index] += psi_real_space[fft_index];
                    }
                }
            }
        }

        local_timer.reset();
        StageTimer nonlocal_timer("setup.rotation.hamiltonian.nonlocal");
        // 3. Nonlocal PP (KB projectors) via batched ZGEMM for all atoms at once
        if (kleinman_bylander_projector_set.empty()) {
            return;
        }

        int total_number_of_projectors = kleinman_bylander_projector_set.number_of_projectors();

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

            auto* real_projectors =
                reinterpret_cast<const double*>(kleinman_bylander_projector_set.host());
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
                        Complex beta_g0 =
                            kleinman_bylander_projector_set.projector(projector_index)[0];
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
                   reinterpret_cast<const std::complex<double>*>(
                       kleinman_bylander_projector_set.host()),
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
        zgemm_(
            "N",
            "N",
            &number_of_plane_waves,
            &number_of_vectors,
            &total_number_of_projectors,
            &alpha_one,
            reinterpret_cast<const std::complex<double>*>(kleinman_bylander_projector_set.host()),
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
        const KleinmanBylanderProjectorSet& kleinman_bylander_projector_set,
        const Complex* psi_wavefunctions,
        Complex* overlap_psi_wavefunctions) {
        if (!pseudopotential_quantum_system.has_ultrasoft()) {
            return;
        }

        int total_number_of_projectors = kleinman_bylander_projector_set.number_of_projectors();

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

            auto* real_projectors =
                reinterpret_cast<const double*>(kleinman_bylander_projector_set.host());
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
                        Complex beta_g0 =
                            kleinman_bylander_projector_set.projector(projector_index)[0];
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
                   reinterpret_cast<const std::complex<double>*>(
                       kleinman_bylander_projector_set.host()),
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
        zgemm_(
            "N",
            "N",
            &number_of_plane_waves,
            &number_of_vectors,
            &total_number_of_projectors,
            &alpha_one,
            reinterpret_cast<const std::complex<double>*>(kleinman_bylander_projector_set.host()),
            &leading_dimension_step3,
            reinterpret_cast<const std::complex<double>*>(scaled_overlap_coefficients.data()),
            &total_number_of_projectors,
            &beta_one,
            reinterpret_cast<std::complex<double>*>(overlap_psi_wavefunctions),
            &leading_dimension_step3);
    }

    int kpoint_index_;
    int number_of_plane_waves_;
    const std::vector<int>& g_vector_indices_;
    const std::vector<Real>& kinetic_energy_squared_;
    const std::vector<Real>& smooth_potential_;
    const Crystal& crystal_structure_;
    const KPointsGrid& kpoint_grid_;
    const PlanewaveBasis& planewave_basis_;
    const execution::PlanewaveExecutionPolicy& execution_policy_;
    const NormConservingPlanewaveKohnShamQuantumSystem& pseudopotential_quantum_system_;
    const KleinmanBylanderProjectorSet& kleinman_bylander_projector_set_;

    /// The vectors last given to apply_hamiltonian(), where the caller holds them.
    const Complex* vectors_ = nullptr;
    int number_of_vectors_ = 0;
    /// H applied to them.
    std::vector<Complex> hamiltonian_vectors_;
};

} // namespace q3
