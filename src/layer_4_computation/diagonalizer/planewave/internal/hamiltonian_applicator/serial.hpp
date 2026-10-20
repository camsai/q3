#pragma once

/**
 * SerialPlanewaveHamiltonianApplicator — CPU serial implementation of H|ψ⟩.
 *
 * Extracted from PlanewaveDavidsonDiagonalizer. Contains all Hamiltonian
 * application logic: smooth potential computation, FFT scatter/gather,
 * kinetic energy, and KB projector application.
 *
 * Constructor takes run-constant planewave context by reference.
 * The caller must ensure all references outlive this object.
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstring>
#include <vector>

#include "io/logger.hpp"
#include "layer_0_core/execution/init.hpp"
#include "layer_0_core/execution/threading.hpp"
#include "layer_0_core/math/linear_algebra/blas.hpp"
#include "layer_0_core/math/tolerances.hpp"
#include "layer_0_core/profiling/counters.hpp"
#include "layer_0_core/types/matrix.hpp"
#include "layer_1_structure/crystal/crystal.hpp"
#include "layer_1_structure/reciprocal_space/kpoints_grid.hpp"
#include "layer_3_representation/basis/planewave/init.hpp"
#include "layer_3_representation/pseudopotential/planewave/internal/projector.hpp"
#include "layer_3_representation/quantum_system/kohn_sham/planewave/norm_conserving/init.hpp"
#include "layer_4_computation/diagonalizer/planewave/internal/hamiltonian_applicator/init.hpp"
#include "layer_4_computation/diagonalizer/planewave/internal/hamiltonian_applicator/planewave_potential_utils.hpp"
#include "layer_4_computation/effective_potential/internal/hubbard/hubbard.hpp"

namespace q3 {

class SerialPlanewaveHamiltonianApplicator : public PlanewaveHamiltonianApplicator {
  public:
    /**
     * Construct with run-constant planewave context.
     * All references must outlive this object.
     */
    SerialPlanewaveHamiltonianApplicator(
        const Crystal& crystal_structure,
        const PlanewaveBasis& planewave_basis,
        const KPointsGrid& kpoint_grid,
        const NormConservingPlanewaveKohnShamQuantumSystem& pseudopotential_quantum_system,
        const execution::ExecutionPolicy& execution_policy)
        : crystal_structure_{crystal_structure}
        , planewave_basis_{planewave_basis}
        , kpoint_grid_{kpoint_grid}
        , pseudopotential_quantum_system_{pseudopotential_quantum_system}
        , execution_policy_{execution_policy} {}

    /**
     * Update the effective potential for a new SCF iteration.
     *
     * Restricts the potential from the dense grid to the smooth grid via G-space
     * truncation, then pre-scales by 1/N_smooth for the local potential application.
     */
    void update_potential(const std::vector<Real>& real_space_potential) override {
        cached_smooth_potential_ =
            compute_smooth_grid_potential(real_space_potential, planewave_basis_);
        int local_smooth_grid_points = planewave_basis_.fft_wave().local_grid_points();
        cached_scaled_potential_.resize(local_smooth_grid_points);
        Real inverse_total_points = 1.0 / planewave_basis_.fft_wave().total_grid_points();
        for (int grid_index = 0; grid_index < local_smooth_grid_points; grid_index++) {
            cached_scaled_potential_[grid_index] =
                cached_smooth_potential_[grid_index] * inverse_total_points;
        }
        /**
         * For task-group band distribution: Allgather the slab-local potential
         * into a full-grid potential so each rank can do serial FFTs.
         */
        if (execution_policy_.group_size() > 1) {
            int total_smooth_points = planewave_basis_.fft_wave().total_grid_points();
            cached_full_grid_potential_.resize(total_smooth_points);
            execution_policy_.allgather_within_group(cached_scaled_potential_.data(),
                                                     local_smooth_grid_points,
                                                     cached_full_grid_potential_.data(),
                                                     total_smooth_points);
        }

        smooth_potential_cache_valid_ = true;
        cached_block_d_matrix_.clear();
        cached_block_d_offsets_.clear();
    }

    /**
     * Configure for a specific k-point.
     * Pre-computes G-vector → FFT grid index mappings and caches kinetic energy.
     */
    void prepare_kpoint(int kpoint_index,
                        int number_of_plane_waves,
                        const std::vector<int>& g_vector_indices,
                        const std::vector<Real>& kinetic_energy_squared) override {
        current_kpoint_index_ = kpoint_index;
        current_number_of_plane_waves_ = number_of_plane_waves;
        current_g_vector_indices_ = &g_vector_indices;
        current_kinetic_energy_squared_ = &kinetic_energy_squared;
        is_gamma_only_current_ = planewave_basis_.is_gamma_only();
        gamma_origin_is_local_current_ = planewave_basis_.gamma_origin_is_local();

        precompute_g_to_fft_mapping(number_of_plane_waves, g_vector_indices, planewave_basis_);
    }

    /**
     * Apply H|ψ⟩ to a batch of vectors using the cached potential and k-point data.
     */
    void apply_batch(const Complex* input_wavefunctions,
                     Complex* output_hamiltonian_wavefunctions,
                     int number_of_vectors) override {
        /**
         * With bands distributed, each rank applies H to its own contiguous
         * slice of the batch and one allgather completes the output. Every rank
         * holds whole coefficient vectors here, so each transform is local and
         * serial -- unlike the reciprocal-space split, where the transform is
         * collective and no rank gets to skip one.
         *
         * Falls back to applying the whole batch when there are fewer vectors
         * than ranks: below that the split leaves ranks idle while still paying
         * for the allgather.
         */
        int band_group_size = execution_policy_.band_distribution_size();
        if (band_group_size > 1 && number_of_vectors >= band_group_size) {
            auto partition =
                execution::partition_bands(number_of_vectors,
                                           band_group_size,
                                           execution_policy_.band_distribution_rank_index(),
                                           current_number_of_plane_waves_,
                                           is_gamma_only_current_ ? 2 : 1);
            if (partition.local_number_of_bands > 0) {
                std::size_t offset = static_cast<std::size_t>(partition.local_first_band) *
                                     current_number_of_plane_waves_;
                apply_hamiltonian_batch(current_number_of_plane_waves_,
                                        partition.local_number_of_bands,
                                        *current_g_vector_indices_,
                                        *current_kinetic_energy_squared_,
                                        cached_scaled_potential_,
                                        input_wavefunctions + offset,
                                        output_hamiltonian_wavefunctions + offset);
            }
            execution_policy_.allgather_across_bands(output_hamiltonian_wavefunctions,
                                                     partition.counts.data(),
                                                     partition.displacements.data());
            return;
        }
        apply_hamiltonian_batch(current_number_of_plane_waves_,
                                number_of_vectors,
                                *current_g_vector_indices_,
                                *current_kinetic_energy_squared_,
                                cached_scaled_potential_,
                                input_wavefunctions,
                                output_hamiltonian_wavefunctions);
    }

    void invalidate_smooth_potential_cache() override {
        smooth_potential_cache_valid_ = false;
        cached_block_d_matrix_.clear();
        cached_block_d_offsets_.clear();
    }

    // ─── Accessors for Davidson to use during transition ──────────────

    /**
     * Whether the smooth potential cache is currently valid.
     */
    [[nodiscard]] bool is_smooth_potential_cache_valid() const override {
        return smooth_potential_cache_valid_;
    }

    /**
     * Access the cached scaled potential (pre-divided by N_smooth).
     */
    [[nodiscard]] const std::vector<Real>& cached_scaled_potential() const {
        return cached_scaled_potential_;
    }

    /**
     * Access the G-to-FFT mapping for the current k-point.
     */
    [[nodiscard]] const std::vector<int>& g_to_fft_mapping() const {
        return g_to_fft_mapping_;
    }

    /**
     * Access the minus-G-to-FFT mapping for gamma-only calculations.
     */
    [[nodiscard]] const std::vector<int>& minus_g_to_fft_mapping() const {
        return minus_g_to_fft_mapping_;
    }

    /**
     * Whether the current k-point uses gamma-only half-sphere G-vectors.
     */
    [[nodiscard]] bool is_gamma_only_current() const override {
        return is_gamma_only_current_;
    }

    [[nodiscard]] bool gamma_origin_is_local_current() const override {
        return gamma_origin_is_local_current_;
    }

    /**
     * Set pre-computed KB projectors for a specific k-point.
     * If set, apply_batch() will use these instead of recomputing them.
     */
    void set_cached_projectors(int kpoint_index, const q3::Matrix<Complex>* projectors) override {
        if (kpoint_index >= static_cast<int>(cached_projector_map_.size())) {
            cached_projector_map_.resize(kpoint_index + 1, nullptr);
        }
        cached_projector_map_[kpoint_index] = projectors;
    }

    /**
     * Get pre-computed KB projectors for the current k-point, or nullptr if none.
     */
    [[nodiscard]] const q3::Matrix<Complex>*
    get_cached_projectors(int kpoint_index) const override {
        if (kpoint_index < static_cast<int>(cached_projector_map_.size())) {
            return cached_projector_map_[kpoint_index];
        }
        return nullptr;
    }

    /**
     * Set pre-computed DFT+U (Hubbard) atomic-orbital projectors for a specific k-point.
     * Geometry-only, set once per k-point at SCF setup (mirrors set_cached_projectors).
     */
    void set_cached_hubbard_projectors(int kpoint_index,
                                       const q3::Matrix<Complex>* projectors) override {
        if (kpoint_index >= static_cast<int>(cached_hubbard_projector_map_.size())) {
            cached_hubbard_projector_map_.resize(kpoint_index + 1, nullptr);
        }
        cached_hubbard_projector_map_[kpoint_index] = projectors;
    }

    /**
     * Set the dense block-diagonal Hubbard D-matrix. Refreshed once per SCF iteration by
     * the caller -- unlike the KB D-matrix, this is never lazily rebuilt here (it depends
     * on the occupation matrix, which this class has no way to compute).
     */
    void set_hubbard_d_matrix(const std::vector<Complex>* dense_d_matrix_column_major) override {
        cached_hubbard_d_matrix_ptr_ = dense_d_matrix_column_major;
    }

  private:
    /**
     * How many bands to transform at once.
     *
     * Capped by the FFT grid's own limit: a grid that hands out one set of
     * scratch arrays can only have one transform in flight, so its band loop
     * stays serial no matter how many threads the machine has. Distributed
     * and GPU-backed grids are in that position today, which is why threading
     * here is a property of the grid rather than of the build flag.
     *
     * Returns 1 -- do not fork a team -- for a single thread, a single band,
     * or a call already inside a parallel region.
     */
    [[nodiscard]] static int band_thread_count(const FFTGrid& fft_grid, int number_of_iterations) {
        int transform_slots = fft_grid.concurrent_transform_slots();
        if (transform_slots < 2) {
            return 1;
        }
        return std::min(transform_slots,
                        execution::threading::thread_count_for(number_of_iterations, 1));
    }

    // ─── Core Hamiltonian application ────────────────────────────────

    /**
     * Apply the full Kohn-Sham Hamiltonian H|ψ⟩ to a batch of vectors in parallel.
     *
     * 1. Kinetic energy: diagonal in G-space.
     * 2. Local potential: FFT to real space, multiply by V(r), FFT back in parallel.
     * 3. Nonlocal KB projectors: Level-3 BLAS zgemm matrix multiplication.
     */
    void apply_hamiltonian_batch(int number_of_plane_waves,
                                 int number_of_vectors,
                                 const std::vector<int>& g_vector_indices,
                                 const std::vector<Real>& kinetic_energy_squared,
                                 const std::vector<Real>& real_space_potential,
                                 const Complex* psi_wavefunctions,
                                 Complex* hamiltonian_psi_wavefunctions) const {

        int leading_dimension = std::max(number_of_plane_waves, 1);
        ProfilingCounters::instance().hamiltonian_application_count += number_of_vectors;
        int local_smooth_grid_points = planewave_basis_.fft_wave().local_grid_points();
        /**
         * The scatter fills reciprocal space, and under the stick layout that
         * count is not the real-space count; zeroing by the wrong one left a
         * stale tail from the previous forward transform in the next scatter.
         */
        int local_smooth_g_points = planewave_basis_.fft_wave().local_g_space_points();
        auto hpsi_timer_start = std::chrono::high_resolution_clock::now();

        if (is_gamma_only_current_) {
            /**
             * Gamma-only: two-bands-at-a-time — pack two real-valued wavefunctions
             * into Re/Im of one complex FFT, halving the FFT count.
             * Uses zero-copy buffer access: scatter/gather directly to/from FFT buffers.
             */
            const FFTGrid& fft_wave_gamma = planewave_basis_.fft_wave();
            /**
             * G = 0 is its own mirror. On the rank that holds it, it is index 0
             * and is written once; on every other rank index 0 is an ordinary
             * G with a mirror of its own, so the mirrored loops start there.
             */
            int first_mirrored_index = gamma_origin_is_local_current_ ? 1 : 0;
            [[maybe_unused]] int gamma_band_threads =
                band_thread_count(fft_wave_gamma, (number_of_vectors + 1) / 2);
            /**
             * One iteration is one packed pair of bands, and pairs are as
             * independent of each other as single bands are on the other
             * branch: disjoint reads, disjoint writes, nothing summed across
             * iterations. Bit-identical at any thread count for the same
             * reason.
             */
#pragma omp parallel for schedule(static)                                                          \
    num_threads(gamma_band_threads) if (gamma_band_threads > 1)
            for (int pair_index = 0; pair_index < number_of_vectors; pair_index += 2) {
                int transform_slot = execution::threading::thread_index();
                Complex* fft_input = fft_wave_gamma.slot_input_buffer(transform_slot);
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

                // 1. Kinetic energy
                for (int g_index = 0; g_index < number_of_plane_waves; ++g_index) {
                    hpsi_1[g_index] = kinetic_energy_squared[g_index] * psi_1[g_index];
                }
                if (has_second_band) {
                    for (int g_index = 0; g_index < number_of_plane_waves; ++g_index) {
                        hpsi_2[g_index] = kinetic_energy_squared[g_index] * psi_2[g_index];
                    }
                }

                // 2. Pack and scatter two bands directly to FFT input buffer
                std::memset(fft_input, 0, local_smooth_g_points * sizeof(Complex));

                if (has_second_band) {
                    if (gamma_origin_is_local_current_) {
                        // G=0: (Re psi_1, Re psi_2), once
                        int fft_index_0 = g_to_fft_mapping_[0];
                        if (fft_index_0 >= 0) {
                            fft_input[fft_index_0] = Complex(psi_1[0].real(), psi_2[0].real());
                        }
                    }
                    // Every other G: +G and -G
                    for (int g_index = first_mirrored_index; g_index < number_of_plane_waves;
                         ++g_index) {
                        int fft_plus = g_to_fft_mapping_[g_index];
                        int fft_minus = minus_g_to_fft_mapping_[g_index];
                        if (fft_plus >= 0) {
                            Complex p1 = psi_1[g_index];
                            Complex p2 = psi_2[g_index];
                            // +G: ψ₁(G) + i·ψ₂(G)
                            fft_input[fft_plus] =
                                Complex(p1.real() - p2.imag(), p1.imag() + p2.real());
                            // -G: conj(ψ₁(G)) + i·conj(ψ₂(G))
                            fft_input[fft_minus] =
                                Complex(p1.real() + p2.imag(), -p1.imag() + p2.real());
                        }
                    }
                } else {
                    // Single band: standard gamma scatter
                    for (int g_index = 0; g_index < number_of_plane_waves; ++g_index) {
                        int fft_index = g_to_fft_mapping_[g_index];
                        if (fft_index >= 0) {
                            fft_input[fft_index] = psi_1[g_index];
                        }
                    }
                    for (int g_index = first_mirrored_index; g_index < number_of_plane_waves;
                         ++g_index) {
                        int fft_minus = minus_g_to_fft_mapping_[g_index];
                        fft_input[fft_minus] = std::conj(psi_1[g_index]);
                    }
                }

                fft_wave_gamma.execute_inverse_raw_on_slot(transform_slot);
                Complex* fft_output = fft_wave_gamma.slot_output_buffer(transform_slot);

                for (int grid_index = 0; grid_index < local_smooth_grid_points; ++grid_index) {
                    fft_output[grid_index] *= real_space_potential[grid_index];
                }

                fft_wave_gamma.prepare_forward_from_output_on_slot(transform_slot);
                fft_wave_gamma.execute_forward_on_slot(transform_slot);
                fft_output = fft_wave_gamma.slot_output_buffer(transform_slot);

                // Separate the two bands using ±G symmetry
                if (has_second_band) {
                    if (gamma_origin_is_local_current_) {
                        // G=0
                        int fft_index_0 = g_to_fft_mapping_[0];
                        if (fft_index_0 >= 0) {
                            Complex result_0 = fft_output[fft_index_0];
                            hpsi_1[0] += Complex(result_0.real(), 0.0);
                            hpsi_2[0] += Complex(result_0.imag(), 0.0);
                        }
                    }
                    // Every other G
                    for (int g_index = first_mirrored_index; g_index < number_of_plane_waves;
                         ++g_index) {
                        int fft_plus = g_to_fft_mapping_[g_index];
                        int fft_minus = minus_g_to_fft_mapping_[g_index];
                        if (fft_plus >= 0) {
                            Complex result_plus = fft_output[fft_plus];
                            Complex result_minus_conj = std::conj(fft_output[fft_minus]);
                            hpsi_1[g_index] += (result_plus + result_minus_conj) * 0.5;
                            Complex diff = (result_plus - result_minus_conj) * 0.5;
                            hpsi_2[g_index] += Complex(diff.imag(), -diff.real());
                        }
                    }
                } else {
                    // Single band: standard gather
                    for (int g_index = 0; g_index < number_of_plane_waves; ++g_index) {
                        int fft_index = g_to_fft_mapping_[g_index];
                        if (fft_index >= 0) {
                            hpsi_1[g_index] += fft_output[fft_index];
                        }
                    }
                }
            }
        } else {
            /**
             * Non-gamma path: per-band FFT loop using zero-copy buffer access.
             *
             * Scatter/gather directly to/from the FFT grid's internal buffers,
             * eliminating 4 memcpy per band. For in-place plans (MPI FFT),
             * input_buffer == output_buffer, so prepare_forward is a no-op.
             */
            const FFTGrid& fft_wave = planewave_basis_.fft_wave();
            [[maybe_unused]] int band_threads = band_thread_count(fft_wave, number_of_vectors);
            /**
             * Bands are independent here: each one reads its own slice of
             * psi, writes its own slice of H|psi>, and touches nothing the
             * others do. Nothing is reduced, so no summation order changes
             * and the results are the same to the last bit however the
             * iterations are handed out.
             *
             * The scatter and gather below -- the sphere-to-grid mapping and
             * back -- ride along inside this region rather than forking teams
             * of their own; at a few thousand G-vectors a nested team would
             * cost more than it saves.
             */
#pragma omp parallel for schedule(static) num_threads(band_threads) if (band_threads > 1)
            for (int vector_index = 0; vector_index < number_of_vectors; ++vector_index) {
                int transform_slot = execution::threading::thread_index();
                Complex* fft_input = fft_wave.slot_input_buffer(transform_slot);
                const Complex* psi_vector =
                    &psi_wavefunctions[vector_index * number_of_plane_waves];
                Complex* hamiltonian_psi_vector =
                    &hamiltonian_psi_wavefunctions[vector_index * number_of_plane_waves];

                // 1. Kinetic energy
                for (int g_index = 0; g_index < number_of_plane_waves; ++g_index) {
                    hamiltonian_psi_vector[g_index] =
                        kinetic_energy_squared[g_index] * psi_vector[g_index];
                }

                // 2. Local potential via zero-copy FFT
                std::memset(fft_input, 0, local_smooth_g_points * sizeof(Complex));
                for (int g_index = 0; g_index < number_of_plane_waves; ++g_index) {
                    int fft_index = g_to_fft_mapping_[g_index];
                    if (fft_index >= 0) {
                        fft_input[fft_index] = psi_vector[g_index];
                    }
                }
                fft_wave.execute_inverse_raw_on_slot(transform_slot);
                Complex* fft_output = fft_wave.slot_output_buffer(transform_slot);
                for (int grid_index = 0; grid_index < local_smooth_grid_points; ++grid_index) {
                    fft_output[grid_index] *= real_space_potential[grid_index];
                }
                fft_wave.prepare_forward_from_output_on_slot(transform_slot);
                fft_wave.execute_forward_on_slot(transform_slot);
                fft_output = fft_wave.slot_output_buffer(transform_slot);
                for (int g_index = 0; g_index < number_of_plane_waves; ++g_index) {
                    int fft_index = g_to_fft_mapping_[g_index];
                    if (fft_index >= 0) {
                        hamiltonian_psi_vector[g_index] += fft_output[fft_index];
                    }
                }
            }
        }
        auto hpsi_timer_after_fft = std::chrono::high_resolution_clock::now();

        /**
         * 3. Nonlocal PP (KB projectors) via batched Level-3 BLAS zgemm
         *
         * Get the KB projectors for this k-point.
         */
        Matrix<Complex> computed_projectors;
        const Matrix<Complex>* kleinman_bylander_potential_ptr = nullptr;
        if (current_kpoint_index_ < static_cast<int>(cached_projector_map_.size()) &&
            cached_projector_map_[current_kpoint_index_] != nullptr) {
            kleinman_bylander_potential_ptr = cached_projector_map_[current_kpoint_index_];
        } else {
            Vector3D kpoint_coordinates = kpoint_grid_[current_kpoint_index_].coordinates;
            compute_kleinman_bylander_projectors(
                kpoint_coordinates,
                crystal_structure_,
                planewave_basis_,
                pseudopotential_quantum_system_.pseudopotential_basis().pseudopotential_set(),
                pseudopotential_quantum_system_.pseudopotential_basis().beta_function_table(),
                g_vector_indices,
                number_of_plane_waves,
                pseudopotential_quantum_system_.pseudopotential_basis()
                    .total_number_of_projectors(),
                computed_projectors);
            kleinman_bylander_potential_ptr = &computed_projectors;
        }
        const auto& kleinman_bylander_potential = *kleinman_bylander_potential_ptr;

        if (kleinman_bylander_potential.empty()) {
            return;
        }

        int total_number_of_projectors = static_cast<int>(kleinman_bylander_potential.rows());

        auto* cast_wavefunctions = reinterpret_cast<const std::complex<double>*>(psi_wavefunctions);
        auto* cast_hamiltonian_wavefunctions =
            reinterpret_cast<std::complex<double>*>(hamiltonian_psi_wavefunctions);
        auto* cast_projector_matrix =
            reinterpret_cast<const std::complex<double>*>(kleinman_bylander_potential.data());

        Complex alpha_one(1.0, 0.0);
        Complex beta_zero(0.0, 0.0);
        Complex beta_one(1.0, 0.0);
        auto* cast_alpha_one = reinterpret_cast<const std::complex<double>*>(&alpha_one);
        auto* cast_beta_zero = reinterpret_cast<const std::complex<double>*>(&beta_zero);
        auto* cast_beta_one = reinterpret_cast<const std::complex<double>*>(&beta_one);

        // Step 1: Compute all overlaps ⟨β_all|ψ⟩
        std::vector<Complex> all_overlap_coefficients(
            total_number_of_projectors * number_of_vectors, Complex(0.0, 0.0));

        if (is_gamma_only_current_) {
            // Gamma-only: use DGEMM on interleaved real/imag parts.
            int double_number_of_plane_waves = 2 * number_of_plane_waves;
            int double_leading_dimension = 2 * leading_dimension;
            double alpha_two = 2.0;
            double d_beta_zero = 0.0;

            auto* real_projectors =
                reinterpret_cast<const double*>(kleinman_bylander_potential.data());
            auto* real_wavefunctions = reinterpret_cast<const double*>(psi_wavefunctions);

            std::vector<double> overlap_real(
                static_cast<size_t>(total_number_of_projectors) * number_of_vectors, 0.0);
            q3::blas::dgemm_("T",
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

            /**
             * Remove the origin from the double count, on the rank that holds
             * it, and copy to Complex. The reduction below completes the sum.
             */
            for (int vector_index = 0; vector_index < number_of_vectors; ++vector_index) {
                const Complex* psi_g0 = &psi_wavefunctions[vector_index * number_of_plane_waves];
                for (int projector_index = 0; projector_index < total_number_of_projectors;
                     ++projector_index) {
                    int index = projector_index + vector_index * total_number_of_projectors;
                    if (gamma_origin_is_local_current_) {
                        Complex beta_g0 = kleinman_bylander_potential(projector_index, 0);
                        overlap_real[index] -= (std::conj(beta_g0) * psi_g0[0]).real();
                    }
                    all_overlap_coefficients[index] = Complex(overlap_real[index], 0.0);
                }
            }

            // Reduce KB projector overlaps across band group.
            execution_policy_.reduce_within_group(all_overlap_coefficients.data(),
                                                  total_number_of_projectors * number_of_vectors);
        } else {
            // Non-gamma: use ZGEMM
            auto* cast_all_overlap_coefficients =
                reinterpret_cast<std::complex<double>*>(all_overlap_coefficients.data());

            q3::blas::zgemm_("C",
                             "N",
                             &total_number_of_projectors,
                             &number_of_vectors,
                             &number_of_plane_waves,
                             cast_alpha_one,
                             cast_projector_matrix,
                             &leading_dimension,
                             cast_wavefunctions,
                             &leading_dimension,
                             cast_beta_zero,
                             cast_all_overlap_coefficients,
                             &total_number_of_projectors);

            // Reduce KB projector overlaps across band group.
            execution_policy_.reduce_within_group(all_overlap_coefficients.data(),
                                                  total_number_of_projectors * number_of_vectors);
        }
        auto hpsi_timer_after_overlap = std::chrono::high_resolution_clock::now();

        /**
         * Step 2: Apply the D matrix, which is block-diagonal by atom.
         *
         * It used to be packed into a dense P x P matrix and applied with one
         * ZGEMM. For 432 silicon atoms P is 7776, so that matrix held 60M
         * complex numbers of which 140k were nonzero -- 967 MB to carry, and
         * 432 times the arithmetic, to multiply by a great many zeros. The
         * blocks are stored contiguously instead and applied one atom at a
         * time.
         */
        if (cached_block_d_matrix_.empty() || cached_block_d_offsets_.empty()) {
            cached_block_d_offsets_.clear();
            cached_block_d_sizes_.clear();
            cached_block_d_projector_starts_.clear();
            std::size_t packed_size = 0;
            int starting_projector_index = 0;
            for (int atom_index = 0; atom_index < crystal_structure_.number_of_atoms();
                 atom_index++) {
                int species_index = crystal_structure_.species_type_index(atom_index);
                int number_of_projectors_for_atom =
                    pseudopotential_quantum_system_.pseudopotential_basis()
                        .number_of_projector_channels(species_index);
                cached_block_d_offsets_.push_back(packed_size);
                cached_block_d_sizes_.push_back(number_of_projectors_for_atom);
                cached_block_d_projector_starts_.push_back(starting_projector_index);
                packed_size += static_cast<std::size_t>(number_of_projectors_for_atom) *
                               static_cast<std::size_t>(number_of_projectors_for_atom);
                starting_projector_index += number_of_projectors_for_atom;
            }
            cached_block_d_matrix_.assign(packed_size, Complex(0.0, 0.0));

            for (std::size_t atom = 0; atom < cached_block_d_sizes_.size(); ++atom) {
                int atom_index = static_cast<int>(atom);
                int species_index = crystal_structure_.species_type_index(atom_index);
                int block_size = cached_block_d_sizes_[atom];
                Complex* block = cached_block_d_matrix_.data() + cached_block_d_offsets_[atom];
                for (int projector_index_a = 0; projector_index_a < block_size;
                     projector_index_a++) {
                    for (int projector_index_b = 0; projector_index_b < block_size;
                         projector_index_b++) {
                        Real d_matrix_value =
                            pseudopotential_quantum_system_.d_matrix_element(atom_index,
                                                                             species_index,
                                                                             projector_index_a,
                                                                             projector_index_b,
                                                                             spin_channel_);
                        if (std::abs(d_matrix_value) >
                            math::tolerances::d_matrix_element_threshold) {
                            block[static_cast<std::size_t>(projector_index_a) +
                                  static_cast<std::size_t>(projector_index_b) * block_size] =
                                Complex(d_matrix_value, 0.0);
                        }
                    }
                }
            }
        }

        std::vector<Complex> all_scaled_overlap_coefficients(
            static_cast<std::size_t>(total_number_of_projectors) * number_of_vectors,
            Complex(0.0, 0.0));
        auto* cast_all_scaled_overlap_coefficients =
            reinterpret_cast<std::complex<double>*>(all_scaled_overlap_coefficients.data());
        auto* cast_overlap_input =
            reinterpret_cast<const std::complex<double>*>(all_overlap_coefficients.data());

        for (std::size_t atom = 0; atom < cached_block_d_sizes_.size(); ++atom) {
            int block_size = cached_block_d_sizes_[atom];
            if (block_size == 0) {
                continue;
            }
            int projector_start = cached_block_d_projector_starts_[atom];
            auto* cast_block = reinterpret_cast<const std::complex<double>*>(
                cached_block_d_matrix_.data() + cached_block_d_offsets_[atom]);
            q3::blas::zgemm_(
                "N",
                "N",
                &block_size,
                &number_of_vectors,
                &block_size,
                cast_alpha_one,
                const_cast<std::complex<double>*>(cast_block),
                &block_size,
                const_cast<std::complex<double>*>(cast_overlap_input + projector_start),
                &total_number_of_projectors,
                cast_beta_zero,
                cast_all_scaled_overlap_coefficients + projector_start,
                &total_number_of_projectors);
        }

        /**
         * Step 3: Accumulate into H|ψ⟩ in a single ZGEMM:
         *   H|ψ_i⟩ += β_all × w_all
         */
        auto* cast_all_scaled_overlap_coefficients_const =
            reinterpret_cast<const std::complex<double>*>(all_scaled_overlap_coefficients.data());

        q3::blas::zgemm_("N",
                         "N",
                         &number_of_plane_waves,
                         &number_of_vectors,
                         &total_number_of_projectors,
                         cast_alpha_one,
                         cast_projector_matrix,
                         &leading_dimension,
                         cast_all_scaled_overlap_coefficients_const,
                         &total_number_of_projectors,
                         cast_beta_one,
                         cast_hamiltonian_wavefunctions,
                         &leading_dimension);

        /**
         * 4. DFT+U (Hubbard) nonlocal-like projector term -- structurally identical to the
         * KB nonlocal block above (steps 1-3), but with atomic-orbital projectors and a
         * D-matrix that depends on the occupation matrix (refreshed once per SCF iteration
         * by the caller, not lazily rebuilt here).
         */
        bool has_hubbard =
            current_kpoint_index_ < static_cast<int>(cached_hubbard_projector_map_.size()) &&
            cached_hubbard_projector_map_[current_kpoint_index_] != nullptr &&
            cached_hubbard_d_matrix_ptr_ != nullptr;
        if (has_hubbard) {
            const auto& hubbard_projectors = *cached_hubbard_projector_map_[current_kpoint_index_];
            int total_hubbard_orbitals = static_cast<int>(hubbard_projectors.rows());
            if (total_hubbard_orbitals > 0) {
                auto* cast_hubbard_projector_matrix =
                    reinterpret_cast<const std::complex<double>*>(hubbard_projectors.data());

                // Step 1: overlaps <phi_m|psi_v>.
                std::vector<Complex> hubbard_overlaps(static_cast<size_t>(total_hubbard_orbitals) *
                                                          number_of_vectors,
                                                      Complex(0.0, 0.0));
                auto* cast_hubbard_overlaps =
                    reinterpret_cast<std::complex<double>*>(hubbard_overlaps.data());
                q3::blas::zgemm_("C",
                                 "N",
                                 &total_hubbard_orbitals,
                                 &number_of_vectors,
                                 &number_of_plane_waves,
                                 cast_alpha_one,
                                 cast_hubbard_projector_matrix,
                                 &leading_dimension,
                                 cast_wavefunctions,
                                 &leading_dimension,
                                 cast_beta_zero,
                                 cast_hubbard_overlaps,
                                 &total_hubbard_orbitals);

                /**
                 * Gamma-only: the ZGEMM summed half the G sphere, so complete
                 * <phi_m|psi> the same way the KB block above completes
                 * <beta|psi> -- 2*Re[...] with the G = 0 term, which this half
                 * sphere contains, subtracted once.
                 *
                 * The KB block reaches for a real DGEMM because its projector
                 * count and its share of H|psi> are large; here the orbital
                 * count is 2l+1 per Hubbard atom and the correction is a scalar
                 * pass over a matrix the ZGEMM already wrote, so the complex
                 * GEMM is kept and only its result is repaired. G = 0 is the
                 * first plane wave (the list is sorted by |k+G|^2) on the rank
                 * that holds it, as the KB block also assumes; a rank that
                 * does not hold it subtracts nothing.
                 *
                 * Only step 1 needs this. Step 3 writes H|psi> back onto the
                 * same half sphere, and the coefficients it projects with are
                 * real -- D is real because n is real symmetric at Gamma -- so
                 * the result still satisfies the half sphere's reality
                 * condition.
                 */
                if (is_gamma_only_current_) {
                    for (int vector_index = 0; vector_index < number_of_vectors; ++vector_index) {
                        const Complex* wavefunction_g_zero =
                            &psi_wavefunctions[static_cast<size_t>(vector_index) *
                                               number_of_plane_waves];
                        for (int orbital_index = 0; orbital_index < total_hubbard_orbitals;
                             ++orbital_index) {
                            Real g_zero_product =
                                gamma_origin_is_local_current_
                                    ? (std::conj(hubbard_projectors(
                                           static_cast<size_t>(orbital_index), 0)) *
                                       wavefunction_g_zero[0])
                                          .real()
                                    : 0.0;
                            size_t flat_index =
                                static_cast<size_t>(vector_index) * total_hubbard_orbitals +
                                orbital_index;
                            hubbard_overlaps[flat_index] = gamma_only_full_sphere_overlap(
                                hubbard_overlaps[flat_index], g_zero_product);
                        }
                    }
                }
                execution_policy_.reduce_within_group(hubbard_overlaps.data(),
                                                      total_hubbard_orbitals * number_of_vectors);

                // Step 2: apply the dense block-diagonal D-matrix.
                std::vector<Complex> hubbard_scaled_overlaps(
                    static_cast<size_t>(total_hubbard_orbitals) * number_of_vectors,
                    Complex(0.0, 0.0));
                auto* cast_hubbard_d_matrix = reinterpret_cast<const std::complex<double>*>(
                    cached_hubbard_d_matrix_ptr_->data());
                auto* cast_hubbard_overlaps_const =
                    reinterpret_cast<const std::complex<double>*>(hubbard_overlaps.data());
                auto* cast_hubbard_scaled_overlaps =
                    reinterpret_cast<std::complex<double>*>(hubbard_scaled_overlaps.data());
                q3::blas::zgemm_("N",
                                 "N",
                                 &total_hubbard_orbitals,
                                 &number_of_vectors,
                                 &total_hubbard_orbitals,
                                 cast_alpha_one,
                                 cast_hubbard_d_matrix,
                                 &total_hubbard_orbitals,
                                 cast_hubbard_overlaps_const,
                                 &total_hubbard_orbitals,
                                 cast_beta_zero,
                                 cast_hubbard_scaled_overlaps,
                                 &total_hubbard_orbitals);

                // Step 3: project back and accumulate into H|psi>.
                auto* cast_hubbard_scaled_overlaps_const =
                    reinterpret_cast<const std::complex<double>*>(hubbard_scaled_overlaps.data());
                q3::blas::zgemm_("N",
                                 "N",
                                 &number_of_plane_waves,
                                 &number_of_vectors,
                                 &total_hubbard_orbitals,
                                 cast_alpha_one,
                                 cast_hubbard_projector_matrix,
                                 &leading_dimension,
                                 cast_hubbard_scaled_overlaps_const,
                                 &total_hubbard_orbitals,
                                 cast_beta_one,
                                 cast_hamiltonian_wavefunctions,
                                 &leading_dimension);
            }
        }

        // Accumulate H|ψ⟩ sub-component timing
        auto hpsi_timer_end = std::chrono::high_resolution_clock::now();
        auto& timing = ProfilingCounters::instance().davidson_timing;
        timing.hamiltonian_fft_time_ms +=
            std::chrono::duration<double, std::milli>(hpsi_timer_after_fft - hpsi_timer_start)
                .count();
        timing.hamiltonian_nonlocal_time_ms +=
            std::chrono::duration<double, std::milli>(hpsi_timer_end - hpsi_timer_after_fft)
                .count();
        timing.hamiltonian_allreduce_time_ms += std::chrono::duration<double, std::milli>(
                                                    hpsi_timer_after_overlap - hpsi_timer_after_fft)
                                                    .count();
    }

    // ─── Smooth potential computation ───────────────────────────────

    /**
     * Restrict the effective potential from the dense (charge density) FFT grid
     * to the smooth (wavefunction) FFT grid via G-space truncation.
     *
     * Uses shared utility for the core FFT + G-space truncation, then adds
     * MPI reduce for band-distributed grids.
     */
    std::vector<Real>
    compute_smooth_grid_potential(const std::vector<Real>& dense_real_space_potential,
                                  const PlanewaveBasis& planewave_basis) const {

        int total_smooth_points = planewave_basis.fft_wave().total_grid_points();

        // Core: dense FFT -> G-space truncation -> smooth V(G)
        auto full_smooth_potential_g =
            compute_smooth_grid_potential_from_dense(dense_real_space_potential, planewave_basis);

        // MPI: Allreduce to ensure all ranks have the full V_smooth(G)
        execution_policy_.reduce_group_distributed_result(full_smooth_potential_g.data(),
                                                          total_smooth_points);

        // Inverse FFT to real space
        return inverse_fft_smooth_potential(full_smooth_potential_g, planewave_basis);
    }

    // ─── G-vector to FFT mapping ────────────────────────────────────

    /**
     * Pre-compute G-vector to smooth FFT grid index mapping.
     *
     * Uses the shared core mapping, then adds gamma-only (-G) and
     * serial (task-group) mappings specific to the serial applicator.
     */
    void precompute_g_to_fft_mapping(int number_of_plane_waves,
                                     const std::vector<int>& g_vector_indices,
                                     const PlanewaveBasis& planewave_basis) {
        // Core G -> smooth FFT index mapping (shared with GPU applicator)
        compute_g_to_fft_index_mapping(
            number_of_plane_waves, g_vector_indices, planewave_basis, g_to_fft_mapping_);

        // For gamma-only: pre-compute the -G FFT grid index for each G-vector.
        if (planewave_basis.is_gamma_only()) {
            minus_g_to_fft_mapping_.resize(number_of_plane_waves);
            for (int g_vector_index = 0; g_vector_index < number_of_plane_waves; g_vector_index++) {
                const auto& g_vector =
                    planewave_basis.g_vectors()[g_vector_indices[g_vector_index]];
                int miller_index_1 = static_cast<int>(std::round(g_vector.miller.x));
                int miller_index_2 = static_cast<int>(std::round(g_vector.miller.y));
                int miller_index_3 = static_cast<int>(std::round(g_vector.miller.z));
                minus_g_to_fft_mapping_[g_vector_index] =
                    planewave_basis.fft_wave().g_vector_to_fft_grid_index(
                        -miller_index_1, -miller_index_2, -miller_index_3);
            }
        } else {
            minus_g_to_fft_mapping_.clear();
        }

        // Serial (full-grid) mapping for task-group band distribution.
        {
            int fft_grid_dimension_1 = planewave_basis.fft_wave().grid_dimension_1();
            int fft_grid_dimension_2 = planewave_basis.fft_wave().grid_dimension_2();
            int fft_grid_dimension_3 = planewave_basis.fft_wave().grid_dimension_3();
            g_to_fft_serial_mapping_.resize(number_of_plane_waves);
            const auto& serial_fft = planewave_basis.fft_wave_serial();
            for (int g_vector_index = 0; g_vector_index < number_of_plane_waves; g_vector_index++) {
                const auto& g_vector =
                    planewave_basis.g_vectors()[g_vector_indices[g_vector_index]];
                int miller_index_1 = static_cast<int>(std::round(g_vector.miller.x));
                int miller_index_2 = static_cast<int>(std::round(g_vector.miller.y));
                int miller_index_3 = static_cast<int>(std::round(g_vector.miller.z));
                if (std::abs(miller_index_1) <= fft_grid_dimension_1 / 2 &&
                    std::abs(miller_index_2) <= fft_grid_dimension_2 / 2 &&
                    std::abs(miller_index_3) <= fft_grid_dimension_3 / 2) {
                    g_to_fft_serial_mapping_[g_vector_index] =
                        serial_fft.g_vector_to_fft_grid_index(
                            miller_index_1, miller_index_2, miller_index_3);
                } else {
                    g_to_fft_serial_mapping_[g_vector_index] = -1;
                }
            }
        }
    }

    // ─── Run-constant context (references) ──────────────────────────

    const Crystal& crystal_structure_;
    const PlanewaveBasis& planewave_basis_;
    const KPointsGrid& kpoint_grid_;
    const NormConservingPlanewaveKohnShamQuantumSystem& pseudopotential_quantum_system_;
    const execution::ExecutionPolicy& execution_policy_;

    // ─── Per-k-point state ──────────────────────────────────────────

    int current_kpoint_index_ = 0;
    int current_number_of_plane_waves_ = 0;
    const std::vector<int>* current_g_vector_indices_ = nullptr;
    const std::vector<Real>* current_kinetic_energy_squared_ = nullptr;
    bool is_gamma_only_current_ = false;
    bool gamma_origin_is_local_current_ = true;

    // ─── FFT mappings ───────────────────────────────────────────────

    std::vector<int> g_to_fft_mapping_;
    std::vector<int> minus_g_to_fft_mapping_;
    std::vector<int> g_to_fft_serial_mapping_;

    // ─── Cached data ────────────────────────────────────────────────

    // Pre-computed KB projectors per k-point (set externally, nullptr means compute internally).
    std::vector<const q3::Matrix<Complex>*> cached_projector_map_;
    /**
     * Pre-computed DFT+U (Hubbard) atomic-orbital projectors per k-point (set externally;
     * nullptr means no Hubbard correction is active).
     */
    std::vector<const q3::Matrix<Complex>*> cached_hubbard_projector_map_;
    /**
     * Dense block-diagonal Hubbard D-matrix (column-major), refreshed once per SCF iteration
     * by the caller. Not owned; not lazily rebuilt (see set_hubbard_d_matrix()).
     */
    const std::vector<Complex>* cached_hubbard_d_matrix_ptr_ = nullptr;
    // Cached smooth grid potential to avoid redundant dense-to-smooth FFTs.
    mutable std::vector<Real> cached_smooth_potential_;
    mutable std::vector<Real> cached_scaled_potential_;
    bool smooth_potential_cache_valid_ = false;
    // Cached dense block-diagonal D-matrix for the nonlocal PP application.
    /**
     * The D matrix as contiguous per-atom blocks, with an offset, a size and a
     * projector start for each atom. Replaces a dense P x P matrix that was
     * almost entirely zero.
     */
    mutable std::vector<Complex> cached_block_d_matrix_;
    mutable std::vector<std::size_t> cached_block_d_offsets_;
    mutable std::vector<int> cached_block_d_sizes_;
    mutable std::vector<int> cached_block_d_projector_starts_;
    // Full-grid potential for task-group band distribution (Allgathered from slabs).
    mutable std::vector<Real> cached_full_grid_potential_;
};

} // namespace q3
