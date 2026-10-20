#pragma once

/**
 * PlanewaveChargeDensityAccumulator — construct the smooth charge density
 * from the occupied Kohn-Sham wavefunctions.
 *
 * This class holds const references to the three data sources it needs
 * (basis, k-point grid, quantum system) and exposes a single public method
 * `accumulate()` that writes the real-space charge density into the output
 * array provided by the caller.
 *
 * Keeping this logic here removes ~75 lines of FFT bookkeeping from
 * PlanewaveKohnShamSCFIterator, which can now simply construct an accumulator
 * and call it.
 */

#include <algorithm>
#include <cmath>
#include <complex>
#include <vector>

#include "layer_0_core/execution/threading.hpp"
#include "layer_0_core/math/tolerances.hpp"
#include "layer_0_core/types/init.hpp"
#include "layer_1_structure/reciprocal_space/kpoints_grid.hpp"
#include "layer_3_representation/basis/planewave/init.hpp"
#include "layer_3_representation/quantum_system/kohn_sham/planewave/init.hpp"

namespace q3 {

/**
 * Accumulates the smooth plane-wave charge density from the occupied Kohn-Sham bands.
 *
 * Holds non-owning const references — all referenced objects must remain alive
 * for the lifetime of this accumulator.
 */
class PlanewaveChargeDensityAccumulator {
  public:
    /**
     * Construct an accumulator bound to the given basis, k-point grid, and quantum system.
     */
    PlanewaveChargeDensityAccumulator(const PlanewaveBasis& basis,
                                      const KPointsGrid& kpoint_grid,
                                      const PlanewaveKohnShamQuantumSystem& quantum_system)
        : basis_(basis), kpoint_grid_(kpoint_grid), quantum_system_(quantum_system) {}

    /**
     * Compute the real-space charge density and write it into `charge_density_out`.
     *
     * Uses the smooth (wave) FFT grid for band transforms when available,
     * then upscales to the rho grid. This is 3-4x faster than transforming
     * each band on the full rho grid.
     *
     * For gamma-only calculations, uses the two-band packing trick to halve the
     * number of FFT calls: two real bands f1, f2 are packed as f1(G) + i*f2(G),
     * transformed together, then |f1|^2 and |f2|^2 recovered from Re/Im parts.
     *
     * @param number_of_bands           Total number of bands per k-point state.
     * @param number_of_spin_components 1 (non-polarized) or 2 (LSDA).
     * @param inverse_cell_volume       1.0 / crystal cell volume in a.u.
     * @param charge_density_out        Output: [spin_index][grid_index].
     *                                  Resized and zeroed by this call.
     */
    void accumulate(int number_of_bands,
                    int number_of_spin_components,
                    Real inverse_cell_volume,
                    std::vector<std::vector<Real>>& charge_density_out) const {
        /**
         * The rho-grid path scatters each plane wave straight into the density
         * grid, which is right only if this rank holds, on the density grid,
         * every stick it holds on the wavefunction grid. Serial, pool and band
         * layouts hold every stick. Ranks sharing a pool split the two grids
         * by stick independently, so a plane wave whose density-grid stick
         * belonged to another rank was dropped: silicon with the density cutoff
         * at twice the wavefunction cutoff, where the two grids are the same
         * size, was 0.79 Ry off on two ranks. On the exact wavefunction grid
         * (QE-format input) they are the same size whenever the density cutoff
         * is at most four times the wavefunction cutoff. The smooth path goes through
         * reciprocal space and is right for any ownership. Every rank of a
         * split grid holds less than the whole of it, so all of them choose the
         * same path, as the collective transforms on either path require.
         */
        const bool density_grid_is_split =
            basis_.fft_rho().local_grid_points() < basis_.fft_rho().total_grid_points();
        bool use_smooth_grid = density_grid_is_split || (basis_.fft_wave().total_grid_points() <
                                                         basis_.fft_rho().total_grid_points());

        if (use_smooth_grid) {
            accumulate_on_smooth_grid(number_of_bands,
                                      number_of_spin_components,
                                      inverse_cell_volume,
                                      charge_density_out);
        } else {
            accumulate_on_rho_grid(number_of_bands,
                                   number_of_spin_components,
                                   inverse_cell_volume,
                                   charge_density_out);
        }
    }

  private:
    const PlanewaveBasis& basis_;
    const KPointsGrid& kpoint_grid_;
    const PlanewaveKohnShamQuantumSystem& quantum_system_;

    /**
     * Index of the first plane wave whose mirror needs filling. G = 0 is its
     * own mirror and, on the rank that holds it, sits at index 0; on every
     * other rank index 0 is an ordinary G with a mirror of its own.
     */
    [[nodiscard]] int first_mirrored_index() const {
        return basis_.gamma_origin_is_local() ? 1 : 0;
    }

    /**
     * Scatter wavefunction coefficients to an FFT grid using pre-computed mapping.
     *
     * For gamma-only scatter, also fills -G positions via conjugate symmetry.
     * Zeros the workspace before scattering.
     */
    void scatter_wavefunction_to_grid(const Complex* band_data,
                                      int plane_waves,
                                      const std::vector<int>& g_to_mapping,
                                      const std::vector<int>& minus_g_to_mapping,
                                      bool scatter_minus_g,
                                      std::vector<Complex>& workspace) const {
        std::fill(workspace.begin(), workspace.end(), Complex(0.0, 0.0));
        for (int g_index = 0; g_index < plane_waves; ++g_index) {
            int fft_index = g_to_mapping[g_index];
            if (fft_index >= 0) {
                workspace[static_cast<size_t>(fft_index)] = band_data[g_index];
            }
        }
        if (scatter_minus_g) {
            for (int g_index = first_mirrored_index(); g_index < plane_waves; ++g_index) {
                int minus_fft_index = minus_g_to_mapping[g_index];
                if (minus_fft_index >= 0) {
                    workspace[static_cast<size_t>(minus_fft_index)] = std::conj(band_data[g_index]);
                }
            }
        }
    }

    /**
     * Pack two real-space bands into one complex FFT workspace (gamma-only two-band trick).
     *
     * Packs band_data_1 and band_data_2 as: f1(G) + i*f2(G).
     * Fills -G positions with conj(f1(G)) + i*conj(f2(G)) (conjugate symmetry).
     * If band_data_2 is nullptr, only band_data_1 is packed (zero imaginary part).
     * Zeros the workspace before packing.
     */
    void scatter_two_bands_gamma(const Complex* band_data_1,
                                 const Complex* band_data_2,
                                 int plane_waves,
                                 const std::vector<int>& g_to_mapping,
                                 const std::vector<int>& minus_g_to_mapping,
                                 std::vector<Complex>& workspace) const {
        std::fill(workspace.begin(), workspace.end(), Complex(0.0, 0.0));
        if (band_data_2 != nullptr) {
            for (int g_index = 0; g_index < plane_waves; ++g_index) {
                int fft_index = g_to_mapping[g_index];
                if (fft_index >= 0) {
                    workspace[static_cast<size_t>(fft_index)] =
                        band_data_1[g_index] + Complex(0.0, 1.0) * band_data_2[g_index];
                }
            }
            for (int g_index = first_mirrored_index(); g_index < plane_waves; ++g_index) {
                int minus_fft_index = minus_g_to_mapping[g_index];
                if (minus_fft_index >= 0) {
                    workspace[static_cast<size_t>(minus_fft_index)] =
                        std::conj(band_data_1[g_index]) +
                        Complex(0.0, 1.0) * std::conj(band_data_2[g_index]);
                }
            }
        } else {
            // Only one band: pack as f1(G) with zero imaginary part
            for (int g_index = 0; g_index < plane_waves; ++g_index) {
                int fft_index = g_to_mapping[g_index];
                if (fft_index >= 0) {
                    workspace[static_cast<size_t>(fft_index)] = band_data_1[g_index];
                }
            }
            for (int g_index = first_mirrored_index(); g_index < plane_waves; ++g_index) {
                int minus_fft_index = minus_g_to_mapping[g_index];
                if (minus_fft_index >= 0) {
                    workspace[static_cast<size_t>(minus_fft_index)] =
                        std::conj(band_data_1[g_index]);
                }
            }
        }
    }

    /**
     * Run `accumulate_band_iteration` over `number_of_iterations` band
     * transforms, spread across threads, and fold what they produce into
     * `density_out`.
     *
     * The body is handed its iteration index, a workspace of its own, the
     * density array it should add into, and the transform slot it may drive.
     *
     * **This is a reduction, and at more than one thread it does not preserve
     * the summation order.** Serially, a grid point accumulates band 0, then
     * band 1, then band 2, and so on. Here each thread sums its own contiguous
     * run of bands into a partial array and the partials are added in
     * thread-index order, which regroups the same terms:
     * `(b0+b1) + (b2+b3)` rather than `((b0+b1)+b2)+b3`. Floating-point
     * addition is not associative, so the last bits of the density can move.
     * The movement is regrouping and nothing else -- every term is present
     * exactly once, in increasing band order within each partial, and the
     * partials are combined in a fixed order, so two runs at the same thread
     * count agree exactly.
     *
     * The alternative, a fixed number of partial arrays independent of the
     * thread count, would hold the last bits still at every thread count but
     * would allocate and combine them at one thread too -- which is the case
     * that has to stay free. So: at one thread, `partial_densities` is empty,
     * the body writes straight into `density_out`, and the arithmetic is
     * exactly what the serial code did, bit for bit.
     */
    template <typename BandIterationBody>
    void accumulate_bands_into(int number_of_iterations,
                               const FFTGrid& fft_grid,
                               int workspace_points,
                               int local_grid_points,
                               Real* density_out,
                               BandIterationBody accumulate_band_iteration) const {
        int transform_slots = fft_grid.concurrent_transform_slots();
        int band_threads =
            (transform_slots < 2)
                ? 1
                : std::min(transform_slots,
                           execution::threading::thread_count_for(number_of_iterations, 1));

        std::vector<std::vector<Real>> partial_densities;
        if (band_threads > 1) {
            partial_densities.assign(
                static_cast<std::size_t>(band_threads - 1),
                std::vector<Real>(static_cast<std::size_t>(local_grid_points), 0.0));
        }

#pragma omp parallel num_threads(band_threads) if (band_threads > 1)
        {
            int transform_slot = execution::threading::thread_index();
            Real* density_target =
                (transform_slot == 0)
                    ? density_out
                    : partial_densities[static_cast<std::size_t>(transform_slot - 1)].data();
            std::vector<Complex> workspace(static_cast<std::size_t>(workspace_points));
            long long first_iteration = 0;
            long long last_iteration = number_of_iterations;
            execution::threading::slice_bounds(number_of_iterations,
                                               execution::threading::team_size(),
                                               transform_slot,
                                               first_iteration,
                                               last_iteration);
            for (long long iteration = first_iteration; iteration < last_iteration; ++iteration) {
                accumulate_band_iteration(
                    static_cast<int>(iteration), workspace, density_target, transform_slot);
            }
        }

        /**
         * Combined by thread index, not by whichever partial finished first:
         * the order has to be a property of the code rather than of the run,
         * or two runs at the same thread count would disagree.
         */
        for (const std::vector<Real>& partial_density : partial_densities) {
            for (int grid_index = 0; grid_index < local_grid_points; ++grid_index) {
                density_out[grid_index] += partial_density[static_cast<std::size_t>(grid_index)];
            }
        }
    }

    /**
     * Original implementation: FFT each band on the dense rho grid.
     * Used for norm-conserving pseudopotentials where wave grid = rho grid.
     *
     * For gamma-only calculations uses the two-band packing trick to halve FFT count.
     */
    void accumulate_on_rho_grid(int number_of_bands,
                                int number_of_spin_components,
                                Real inverse_cell_volume,
                                std::vector<std::vector<Real>>& charge_density_out) const {
        int local_grid_points = basis_.fft_rho().local_grid_points();
        int number_of_kpoints = kpoint_grid_.number_of_kpoints();
        int number_of_kpoint_states = number_of_kpoints * number_of_spin_components;

        charge_density_out.resize(static_cast<std::size_t>(number_of_spin_components));
        for (int spin_index = 0; spin_index < number_of_spin_components; ++spin_index) {
            charge_density_out[static_cast<std::size_t>(spin_index)].assign(
                static_cast<std::size_t>(local_grid_points), 0.0);
        }

        // Pre-compute FFT grid dimensions (constant for all bands)
        int fft_dimension_1 = basis_.fft_rho().grid_dimension_1();
        int fft_dimension_2 = basis_.fft_rho().grid_dimension_2();
        int fft_dimension_3 = basis_.fft_rho().grid_dimension_3();

        /**
         * How large a transform workspace each band iteration needs. The array
         * itself is allocated per thread inside accumulate_bands_into(): one
         * shared across bands would be overwritten by whichever thread reached
         * it first.
         */
        int fft_workspace_points =
            std::max(local_grid_points, basis_.fft_rho().local_g_space_points());

        for (int kpoint_index_state = 0; kpoint_index_state < number_of_kpoint_states;
             ++kpoint_index_state) {
            int spin_index = kpoint_index_state / number_of_kpoints;
            int kpoint_index = kpoint_index_state % number_of_kpoints;
            Real kpoint_weight = kpoint_grid_[kpoint_index].weight;
            int plane_waves_at_kpoint = basis_.number_of_plane_waves_by_kpoint_index(kpoint_index);
            const auto& kpoint_g_indices = basis_.g_vector_indices_by_kpoint_index(kpoint_index);

            // Pre-compute G->FFT grid mapping for this k-point (constant across bands)
            std::vector<int> g_to_fft_mapping(plane_waves_at_kpoint, -1);
            for (int g_index = 0; g_index < plane_waves_at_kpoint; ++g_index) {
                const auto& g_vector =
                    basis_.g_vectors()[kpoint_g_indices[static_cast<std::size_t>(g_index)]];
                int miller_x = static_cast<int>(std::round(g_vector.miller.x));
                int miller_y = static_cast<int>(std::round(g_vector.miller.y));
                int miller_z = static_cast<int>(std::round(g_vector.miller.z));
                if (std::abs(miller_x) <= fft_dimension_1 / 2 &&
                    std::abs(miller_y) <= fft_dimension_2 / 2 &&
                    std::abs(miller_z) <= fft_dimension_3 / 2) {
                    g_to_fft_mapping[g_index] =
                        basis_.fft_rho().g_vector_to_fft_grid_index(miller_x, miller_y, miller_z);
                }
            }

            // Pre-compute -G FFT mapping for gamma-only scatter
            std::vector<int> minus_g_to_fft_mapping;
            if (basis_.is_gamma_only()) {
                minus_g_to_fft_mapping.resize(plane_waves_at_kpoint, -1);
                for (int g_index = 0; g_index < plane_waves_at_kpoint; ++g_index) {
                    const auto& g_vector =
                        basis_.g_vectors()[kpoint_g_indices[static_cast<std::size_t>(g_index)]];
                    int miller_x = static_cast<int>(std::round(g_vector.miller.x));
                    int miller_y = static_cast<int>(std::round(g_vector.miller.y));
                    int miller_z = static_cast<int>(std::round(g_vector.miller.z));
                    minus_g_to_fft_mapping[g_index] = basis_.fft_rho().g_vector_to_fft_grid_index(
                        -miller_x, -miller_y, -miller_z);
                }
            }

            auto* density_out_ptr = charge_density_out[static_cast<std::size_t>(spin_index)].data();

            if (basis_.is_gamma_only()) {
                /**
                 * Two-band gamma trick: pair up occupied bands and process two per FFT.
                 * Collect occupied band indices first.
                 */
                std::vector<int> occupied_band_indices;
                for (int band_index = 0; band_index < number_of_bands; ++band_index) {
                    Real occupation =
                        quantum_system_.occupation_by_kpoint_state_index_and_band_index(
                            kpoint_index_state, band_index);
                    if (occupation >= math::tolerances::occupation_threshold) {
                        occupied_band_indices.push_back(band_index);
                    }
                }
                // Process pairs of bands with one FFT each
                int number_of_band_pairs = static_cast<int>((occupied_band_indices.size() + 1) / 2);
                accumulate_bands_into(
                    number_of_band_pairs,
                    basis_.fft_rho(),
                    fft_workspace_points,
                    local_grid_points,
                    density_out_ptr,
                    [&](int pair_index,
                        std::vector<Complex>& fft_workspace,
                        Real* density_target,
                        int transform_slot) {
                        std::size_t pair_start = static_cast<std::size_t>(pair_index) * 2;
                        int band_index_1 = occupied_band_indices[pair_start];
                        bool has_second_band = (pair_start + 1 < occupied_band_indices.size());
                        int band_index_2 =
                            has_second_band ? occupied_band_indices[pair_start + 1] : -1;

                        const Complex* band_data_1 = quantum_system_.wavefunction_band_data(
                            kpoint_index_state, band_index_1);
                        const Complex* band_data_2 = has_second_band
                                                         ? quantum_system_.wavefunction_band_data(
                                                               kpoint_index_state, band_index_2)
                                                         : nullptr;

                        Real occupation_1 =
                            quantum_system_.occupation_by_kpoint_state_index_and_band_index(
                                kpoint_index_state, band_index_1);
                        Real occupation_2 =
                            has_second_band
                                ? quantum_system_.occupation_by_kpoint_state_index_and_band_index(
                                      kpoint_index_state, band_index_2)
                                : 0.0;

                        scatter_two_bands_gamma(band_data_1,
                                                band_data_2,
                                                plane_waves_at_kpoint,
                                                g_to_fft_mapping,
                                                minus_g_to_fft_mapping,
                                                fft_workspace);
                        basis_.fft_rho().inverse_raw_on_slot(fft_workspace.data(), transform_slot);

                        // psi1(r) = Re(result), psi2(r) = Im(result) - both real in gamma-only
                        Real weight_1 = kpoint_weight * occupation_1 * inverse_cell_volume;
                        Real weight_2 = kpoint_weight * occupation_2 * inverse_cell_volume;
                        for (int grid_index = 0; grid_index < local_grid_points; ++grid_index) {
                            Real psi1 = fft_workspace[static_cast<size_t>(grid_index)].real();
                            Real psi2 = fft_workspace[static_cast<size_t>(grid_index)].imag();
                            density_target[grid_index] +=
                                weight_1 * psi1 * psi1 + weight_2 * psi2 * psi2;
                        }
                    });
            } else {
                // Non-gamma: process each occupied band individually
                accumulate_bands_into(
                    number_of_bands,
                    basis_.fft_rho(),
                    fft_workspace_points,
                    local_grid_points,
                    density_out_ptr,
                    [&](int band_index,
                        std::vector<Complex>& fft_workspace,
                        Real* density_target,
                        int transform_slot) {
                        Real occupation =
                            quantum_system_.occupation_by_kpoint_state_index_and_band_index(
                                kpoint_index_state, band_index);
                        if (occupation < math::tolerances::occupation_threshold) {
                            return;
                        }
                        const Complex* band_data =
                            quantum_system_.wavefunction_band_data(kpoint_index_state, band_index);

                        scatter_wavefunction_to_grid(band_data,
                                                     plane_waves_at_kpoint,
                                                     g_to_fft_mapping,
                                                     minus_g_to_fft_mapping,
                                                     false,
                                                     fft_workspace);
                        basis_.fft_rho().inverse_raw_on_slot(fft_workspace.data(), transform_slot);

                        Real weight_factor = kpoint_weight * occupation * inverse_cell_volume;
                        for (int grid_index = 0; grid_index < local_grid_points; ++grid_index) {
                            density_target[grid_index] +=
                                weight_factor *
                                std::norm(fft_workspace[static_cast<std::size_t>(grid_index)]);
                        }
                    });
            }
        }
    }

    /**
     * Optimized implementation: FFT each band on the smaller smooth (wave) grid,
     * accumulate |psi|^2 on the smooth grid, then upscale to the rho grid at the end.
     *
     * The smooth density rho_smooth(r) on the wave grid can contain aliasing because
     * |psi|^2 has G-components up to 2*Gcut_wfc. However, for ultrasoft pseudopotentials
     * the augmentation charge Q(G) is added separately on the full G-space, and the
     * smooth density only needs G-vectors up to Gcut_wfc for the Hartree/XC calculation
     * (since V_eff is smooth). The upscaling step correctly transfers only the smooth
     * G-components to the rho grid.
     *
     * For gamma-only calculations uses the two-band packing trick to halve FFT count.
     */
    void accumulate_on_smooth_grid(int number_of_bands,
                                   int number_of_spin_components,
                                   Real inverse_cell_volume,
                                   std::vector<std::vector<Real>>& charge_density_out) const {
        int smooth_total_points = basis_.fft_wave().total_grid_points();
        int smooth_local_points = basis_.fft_wave().local_grid_points();
        int rho_total_points = basis_.fft_rho().total_grid_points();
        int rho_local_points = basis_.fft_rho().local_grid_points();
        int number_of_kpoints = kpoint_grid_.number_of_kpoints();
        int number_of_kpoint_states = number_of_kpoints * number_of_spin_components;

        // Precompute G->smooth FFT mapping (constant across all bands and k-points)
        int wave_dim_1 = basis_.fft_wave().grid_dimension_1();
        int wave_dim_2 = basis_.fft_wave().grid_dimension_2();
        int wave_dim_3 = basis_.fft_wave().grid_dimension_3();

        // Accumulate |psi|^2 on the smooth grid per spin
        std::vector<std::vector<Real>> smooth_density(
            static_cast<std::size_t>(number_of_spin_components));
        for (int spin_index = 0; spin_index < number_of_spin_components; ++spin_index) {
            smooth_density[static_cast<std::size_t>(spin_index)].assign(
                static_cast<std::size_t>(smooth_local_points), 0.0);
        }

        /// Per-thread workspace size; see the rho-grid path for why it is not shared.
        int fft_workspace_points =
            std::max(smooth_local_points, basis_.fft_wave().local_g_space_points());

        for (int kpoint_index_state = 0; kpoint_index_state < number_of_kpoint_states;
             ++kpoint_index_state) {
            int spin_index = kpoint_index_state / number_of_kpoints;
            int kpoint_index = kpoint_index_state % number_of_kpoints;
            Real kpoint_weight = kpoint_grid_[kpoint_index].weight;
            int plane_waves_at_kpoint = basis_.number_of_plane_waves_by_kpoint_index(kpoint_index);

            // Pre-build G->smooth-grid mapping for this k-point
            const auto& kpoint_g_indices = basis_.g_vector_indices_by_kpoint_index(kpoint_index);
            std::vector<int> g_to_smooth_mapping(plane_waves_at_kpoint, -1);
            for (int g_index = 0; g_index < plane_waves_at_kpoint; ++g_index) {
                const auto& g_vector =
                    basis_.g_vectors()[kpoint_g_indices[static_cast<std::size_t>(g_index)]];
                int miller_x = static_cast<int>(std::round(g_vector.miller.x));
                int miller_y = static_cast<int>(std::round(g_vector.miller.y));
                int miller_z = static_cast<int>(std::round(g_vector.miller.z));
                if (std::abs(miller_x) <= wave_dim_1 / 2 && std::abs(miller_y) <= wave_dim_2 / 2 &&
                    std::abs(miller_z) <= wave_dim_3 / 2) {
                    g_to_smooth_mapping[g_index] =
                        basis_.fft_wave().g_vector_to_fft_grid_index(miller_x, miller_y, miller_z);
                }
            }

            // Pre-compute -G FFT mapping for gamma-only scatter on smooth grid
            std::vector<int> minus_g_to_smooth_mapping;
            if (basis_.is_gamma_only()) {
                minus_g_to_smooth_mapping.resize(plane_waves_at_kpoint, -1);
                for (int g_index = 0; g_index < plane_waves_at_kpoint; ++g_index) {
                    const auto& g_vector =
                        basis_.g_vectors()[kpoint_g_indices[static_cast<std::size_t>(g_index)]];
                    int miller_x = static_cast<int>(std::round(g_vector.miller.x));
                    int miller_y = static_cast<int>(std::round(g_vector.miller.y));
                    int miller_z = static_cast<int>(std::round(g_vector.miller.z));
                    if (std::abs(miller_x) <= wave_dim_1 / 2 &&
                        std::abs(miller_y) <= wave_dim_2 / 2 &&
                        std::abs(miller_z) <= wave_dim_3 / 2) {
                        minus_g_to_smooth_mapping[g_index] =
                            basis_.fft_wave().g_vector_to_fft_grid_index(
                                -miller_x, -miller_y, -miller_z);
                    }
                }
            }

            auto* smooth_density_ptr = smooth_density[static_cast<std::size_t>(spin_index)].data();

            if (basis_.is_gamma_only()) {
                // Two-band gamma trick: process pairs of occupied bands with one FFT each.
                std::vector<int> occupied_band_indices;
                for (int band_index = 0; band_index < number_of_bands; ++band_index) {
                    Real occupation =
                        quantum_system_.occupation_by_kpoint_state_index_and_band_index(
                            kpoint_index_state, band_index);
                    if (occupation >= math::tolerances::occupation_threshold) {
                        occupied_band_indices.push_back(band_index);
                    }
                }
                int number_of_band_pairs = static_cast<int>((occupied_band_indices.size() + 1) / 2);
                accumulate_bands_into(
                    number_of_band_pairs,
                    basis_.fft_wave(),
                    fft_workspace_points,
                    smooth_local_points,
                    smooth_density_ptr,
                    [&](int pair_index,
                        std::vector<Complex>& fft_workspace,
                        Real* density_target,
                        int transform_slot) {
                        std::size_t pair_start = static_cast<std::size_t>(pair_index) * 2;
                        int band_index_1 = occupied_band_indices[pair_start];
                        bool has_second_band = (pair_start + 1 < occupied_band_indices.size());
                        int band_index_2 =
                            has_second_band ? occupied_band_indices[pair_start + 1] : -1;

                        const Complex* band_data_1 = quantum_system_.wavefunction_band_data(
                            kpoint_index_state, band_index_1);
                        const Complex* band_data_2 = has_second_band
                                                         ? quantum_system_.wavefunction_band_data(
                                                               kpoint_index_state, band_index_2)
                                                         : nullptr;

                        Real occupation_1 =
                            quantum_system_.occupation_by_kpoint_state_index_and_band_index(
                                kpoint_index_state, band_index_1);
                        Real occupation_2 =
                            has_second_band
                                ? quantum_system_.occupation_by_kpoint_state_index_and_band_index(
                                      kpoint_index_state, band_index_2)
                                : 0.0;

                        scatter_two_bands_gamma(band_data_1,
                                                band_data_2,
                                                plane_waves_at_kpoint,
                                                g_to_smooth_mapping,
                                                minus_g_to_smooth_mapping,
                                                fft_workspace);
                        basis_.fft_wave().inverse_raw_on_slot(fft_workspace.data(), transform_slot);

                        Real weight_1 = kpoint_weight * occupation_1 * inverse_cell_volume;
                        Real weight_2 = kpoint_weight * occupation_2 * inverse_cell_volume;
                        for (int grid_index = 0; grid_index < smooth_local_points; ++grid_index) {
                            Real psi1 = fft_workspace[static_cast<size_t>(grid_index)].real();
                            Real psi2 = fft_workspace[static_cast<size_t>(grid_index)].imag();
                            density_target[grid_index] +=
                                weight_1 * psi1 * psi1 + weight_2 * psi2 * psi2;
                        }
                    });
            } else {
                // Non-gamma: process each occupied band individually
                accumulate_bands_into(
                    number_of_bands,
                    basis_.fft_wave(),
                    fft_workspace_points,
                    smooth_local_points,
                    smooth_density_ptr,
                    [&](int band_index,
                        std::vector<Complex>& fft_workspace,
                        Real* density_target,
                        int transform_slot) {
                        Real occupation =
                            quantum_system_.occupation_by_kpoint_state_index_and_band_index(
                                kpoint_index_state, band_index);
                        if (occupation < math::tolerances::occupation_threshold) {
                            return;
                        }
                        const Complex* band_data =
                            quantum_system_.wavefunction_band_data(kpoint_index_state, band_index);

                        scatter_wavefunction_to_grid(band_data,
                                                     plane_waves_at_kpoint,
                                                     g_to_smooth_mapping,
                                                     minus_g_to_smooth_mapping,
                                                     false,
                                                     fft_workspace);
                        basis_.fft_wave().inverse_raw_on_slot(fft_workspace.data(), transform_slot);

                        Real weight_factor = kpoint_weight * occupation * inverse_cell_volume;
                        for (int grid_index = 0; grid_index < smooth_local_points; ++grid_index) {
                            density_target[grid_index] +=
                                weight_factor *
                                std::norm(fft_workspace[static_cast<std::size_t>(grid_index)]);
                        }
                    });
            }
        }

        // Upscale smooth density to rho grid via G-space
        charge_density_out.resize(static_cast<std::size_t>(number_of_spin_components));
        for (int spin_index = 0; spin_index < number_of_spin_components; ++spin_index) {
            // Forward FFT on smooth grid: rho(r) -> rho(G)
            std::vector<Complex> density_g_smooth(
                static_cast<std::size_t>(
                    std::max(smooth_local_points, basis_.fft_wave().local_g_space_points())),
                Complex(0.0, 0.0));
            for (int grid_index = 0; grid_index < smooth_local_points; ++grid_index) {
                density_g_smooth[static_cast<std::size_t>(grid_index)] =
                    Complex(smooth_density[static_cast<std::size_t>(spin_index)]
                                          [static_cast<std::size_t>(grid_index)],
                            0.0);
            }
            basis_.fft_wave().forward(density_g_smooth.data());

            // Normalize forward FFT and make G-space redundant
            Real inverse_smooth = 1.0 / smooth_total_points;
            for (auto& value : density_g_smooth) {
                value *= inverse_smooth;
            }
            /**
             * A redundant rho(G) is needed to scatter it onto the rho grid. The
             * grid places its own reciprocal-space points into the full array --
             * a slab or a set of sticks, it knows which -- and the pool-wide
             * reduction fills in the rest.
             */
            std::vector<Complex> density_g_smooth_redundant(smooth_total_points, Complex(0.0, 0.0));
            basis_.fft_wave().scatter_local_g_space(density_g_smooth.data(),
                                                    density_g_smooth_redundant.data());
            basis_.fft_wave().reduce_g_space(density_g_smooth_redundant.data(),
                                             smooth_total_points);

            /**
             * Scatter smooth G-components to rho grid (redundant rho(G) -> local rho(G) on rho
             * grid)
             */
            std::vector<Complex> density_g_rho(
                static_cast<std::size_t>(
                    std::max(rho_local_points, basis_.fft_rho().local_g_space_points())),
                Complex(0.0, 0.0));
            for (int m1 = -(wave_dim_1 / 2); m1 <= wave_dim_1 / 2; ++m1) {
                for (int m2 = -(wave_dim_2 / 2); m2 <= wave_dim_2 / 2; ++m2) {
                    for (int m3 = -(wave_dim_3 / 2); m3 <= wave_dim_3 / 2; ++m3) {
                        int smooth_index_global =
                            (m1 < 0 ? m1 + wave_dim_1 : m1) * wave_dim_2 * wave_dim_3 +
                            (m2 < 0 ? m2 + wave_dim_2 : m2) * wave_dim_3 +
                            (m3 < 0 ? m3 + wave_dim_3 : m3);
                        int rho_index_local =
                            basis_.fft_rho().g_vector_to_fft_grid_index(m1, m2, m3);

                        if (rho_index_local >= 0) {
                            density_g_rho[static_cast<std::size_t>(rho_index_local)] =
                                density_g_smooth_redundant[static_cast<std::size_t>(
                                    smooth_index_global)];
                        }
                    }
                }
            }

            // Backward FFT on rho grid: rho(G) -> rho(r)
            basis_.fft_rho().inverse_raw(density_g_rho.data());

            // Extract real part
            charge_density_out[static_cast<std::size_t>(spin_index)].resize(
                static_cast<std::size_t>(rho_local_points));
            for (int grid_index = 0; grid_index < rho_local_points; ++grid_index) {
                charge_density_out[static_cast<std::size_t>(spin_index)]
                                  [static_cast<std::size_t>(grid_index)] =
                                      density_g_rho[static_cast<std::size_t>(grid_index)].real();
            }
        }
    }
};

} // namespace q3
