#pragma once

/**
 * GPUPlanewaveChargeDensityAccumulator — the smooth charge density from the
 * occupied Kohn-Sham bands, with the band transforms on the device.
 *
 * The density is one inverse transform per occupied band and a sum of squares
 * over the grid. On the host that is the largest single cost of a GPU run's SCF
 * iteration: the build holds one rank per device, so the transforms that the
 * CPU build spreads over a node's ranks all fall to one. The device already
 * holds a batched transform of the wavefunction grid for the Hamiltonian, and
 * this class drives the same one.
 */

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "layer_0_core/execution/gpu/device/init.hpp"
#include "layer_0_core/execution/gpu/device/internal/buffer.hpp"
#include "layer_0_core/math/tolerances.hpp"
#include "layer_0_core/profiling/stage_timer.hpp"
#include "layer_0_core/types/init.hpp"
#include "layer_1_structure/reciprocal_space/kpoints_grid.hpp"
#include "layer_3_representation/basis/planewave/init.hpp"
#include "layer_3_representation/basis/planewave/internal/fft/gpu/init.hpp"
#include "layer_3_representation/quantum_system/kohn_sham/planewave/init.hpp"
#include "layer_4_computation/charge_density/planewave/internal/accumulator.hpp"
#include "layer_4_computation/diagonalizer/planewave/internal/hamiltonian_applicator/gpu_fft_batch.hpp"
#include "layer_4_computation/diagonalizer/planewave/internal/hamiltonian_applicator/gpu_kernels.hpp"

namespace q3 {

/**
 * Accumulates the smooth plane-wave charge density with device transforms.
 *
 * Holds non-owning references, like its base: the basis, k-point grid, quantum
 * system and device transform must outlive it.
 */
class GPUPlanewaveChargeDensityAccumulator : public PlanewaveChargeDensityAccumulator {
  public:
    /**
     * Construct an accumulator that transforms bands on `gpu_fft_grid`.
     *
     * @param gpu_fft_grid    Device transform of the wavefunction grid; the
     *                        Hamiltonian application initializes it.
     * @param device_backend  Device the transform and the buffers live on.
     */
    GPUPlanewaveChargeDensityAccumulator(
        const PlanewaveBasis& basis,
        const KPointsGrid& kpoint_grid,
        const PlanewaveKohnShamQuantumSystem& quantum_system,
        const gpu::GPUFFTGrid& gpu_fft_grid,
        std::shared_ptr<execution::gpu::device::GPUDeviceBackend> device_backend)
        : PlanewaveChargeDensityAccumulator(basis, kpoint_grid, quantum_system)
        , gpu_fft_grid_(gpu_fft_grid)
        , device_backend_(std::move(device_backend))
        , device_mapping_(0, device_backend_)
        , device_bands_(0, device_backend_)
        , device_weights_(0, device_backend_)
        , device_workspace_(0, device_backend_)
        , device_density_(0, device_backend_) {}

    /**
     * Compute the real-space charge density and write it into `charge_density_out`.
     *
     * Every band is transformed on the wavefunction grid, on the device, and
     * the result is carried to the density grid as the host path does. Where
     * the device transform does not describe this rank's grid, the host path
     * runs instead; see device_transform_applies().
     */
    void accumulate(int number_of_bands,
                    int number_of_spin_components,
                    Real inverse_cell_volume,
                    std::vector<std::vector<Real>>& charge_density_out) const override {
        if (!device_transform_applies()) {
            PlanewaveChargeDensityAccumulator::accumulate(number_of_bands,
                                                          number_of_spin_components,
                                                          inverse_cell_volume,
                                                          charge_density_out);
            return;
        }

        int grid_points = basis_.fft_wave().total_grid_points();
        int number_of_kpoints = kpoint_grid_.number_of_kpoints();
        std::size_t density_bytes = static_cast<std::size_t>(grid_points) * sizeof(Real);

        std::vector<std::vector<Real>> smooth_density(
            static_cast<std::size_t>(number_of_spin_components),
            std::vector<Real>(static_cast<std::size_t>(grid_points), 0.0));
        execution::gpu::device::GPUDeviceBuffer<Real>& device_density = device_density_;
        ensure_capacity(device_density, static_cast<std::size_t>(grid_points));

        for (int spin_index = 0; spin_index < number_of_spin_components; ++spin_index) {
            std::vector<Real>& spin_density = smooth_density[static_cast<std::size_t>(spin_index)];
            // The host array is still all zeros here, which is what the device sum starts from.
            device_backend_->copy(device_density.data(), spin_density.data(), density_bytes);
            for (int kpoint_index = 0; kpoint_index < number_of_kpoints; ++kpoint_index) {
                accumulate_kpoint_state(spin_index * number_of_kpoints + kpoint_index,
                                        kpoint_index,
                                        number_of_bands,
                                        inverse_cell_volume,
                                        device_density.data());
            }
            device_backend_->synchronize();
            device_backend_->copy(spin_density.data(), device_density.data(), density_bytes);
        }

        if (basis_.wavefunction_grid_is_density_grid()) {
            charge_density_out = std::move(smooth_density);
        } else {
            upscale_smooth_density(smooth_density, charge_density_out);
        }
    }

  private:
    /**
     * Most bands transformed in one pass. The eigensolver's batched transform
     * keeps plans for runs of up to this many grids, so a pass of this size is
     * one transform call, and its workspace stays a small multiple of the grid
     * whatever the number of bands.
     */
    static constexpr int maximum_bands_per_pass = 32;

    const gpu::GPUFFTGrid& gpu_fft_grid_;
    std::shared_ptr<execution::gpu::device::GPUDeviceBackend> device_backend_;

    /// Device buffers kept between calls; see ensure_capacity().
    mutable execution::gpu::device::GPUDeviceBuffer<int> device_mapping_;
    mutable execution::gpu::device::GPUDeviceBuffer<Complex> device_bands_;
    mutable execution::gpu::device::GPUDeviceBuffer<Real> device_weights_;
    mutable execution::gpu::device::GPUDeviceBuffer<Complex> device_workspace_;
    mutable execution::gpu::device::GPUDeviceBuffer<Real> device_density_;

    /**
     * Whether the device transform can stand in for this rank's wavefunction grid.
     *
     * It transforms a whole grid, so the rank must hold the whole grid: a rank
     * sharing its pool's grid holds a part. It must have been initialized to
     * this grid's size, which the first Hamiltonian application does. And the
     * gamma-point path packs two real bands into one transform, which the
     * device kernels here do not.
     */
    [[nodiscard]] bool device_transform_applies() const {
        const FFTGrid& wave_grid = basis_.fft_wave();
        return !basis_.is_gamma_only() &&
               wave_grid.local_grid_points() == wave_grid.total_grid_points() &&
               gpu_fft_grid_.total_grid_points() == wave_grid.total_grid_points();
    }

    /**
     * Add one k-point state's occupied bands to the device density.
     */
    void accumulate_kpoint_state(int kpoint_index_state,
                                 int kpoint_index,
                                 int number_of_bands,
                                 Real inverse_cell_volume,
                                 Real* device_density) const {
        const FFTGrid& wave_grid = basis_.fft_wave();
        int grid_points = wave_grid.total_grid_points();
        int plane_waves = basis_.number_of_plane_waves_by_kpoint_index(kpoint_index);
        Real kpoint_weight = kpoint_grid_[kpoint_index].weight;

        std::vector<int> occupied_band_indices;
        std::vector<Real> band_weights;
        for (int band_index = 0; band_index < number_of_bands; ++band_index) {
            Real occupation = quantum_system_.occupation_by_kpoint_state_index_and_band_index(
                kpoint_index_state, band_index);
            if (occupation >= math::tolerances::occupation_threshold) {
                occupied_band_indices.push_back(band_index);
                band_weights.push_back(kpoint_weight * occupation * inverse_cell_volume);
            }
        }
        int number_of_occupied_bands = static_cast<int>(occupied_band_indices.size());
        if (number_of_occupied_bands == 0 || plane_waves == 0) {
            return;
        }

        std::optional<StageTimer> prepare_timer(std::in_place, "density.device.prepare");
        // Where each plane wave of this k-point sits on the wavefunction grid.
        const auto& kpoint_g_indices = basis_.g_vector_indices_by_kpoint_index(kpoint_index);
        int dimension_1 = wave_grid.grid_dimension_1();
        int dimension_2 = wave_grid.grid_dimension_2();
        int dimension_3 = wave_grid.grid_dimension_3();
        std::vector<int> g_to_grid_mapping(static_cast<std::size_t>(plane_waves), -1);
        for (int g_index = 0; g_index < plane_waves; ++g_index) {
            const auto& g_vector =
                basis_.g_vectors()[kpoint_g_indices[static_cast<std::size_t>(g_index)]];
            int miller_x = static_cast<int>(std::round(g_vector.miller.x));
            int miller_y = static_cast<int>(std::round(g_vector.miller.y));
            int miller_z = static_cast<int>(std::round(g_vector.miller.z));
            if (std::abs(miller_x) <= dimension_1 / 2 && std::abs(miller_y) <= dimension_2 / 2 &&
                std::abs(miller_z) <= dimension_3 / 2) {
                g_to_grid_mapping[static_cast<std::size_t>(g_index)] =
                    wave_grid.g_vector_to_fft_grid_index(miller_x, miller_y, miller_z);
            }
        }
        execution::gpu::device::GPUDeviceBuffer<int>& device_mapping = device_mapping_;
        ensure_capacity(device_mapping, static_cast<std::size_t>(plane_waves));
        device_backend_->copy(device_mapping.data(),
                              g_to_grid_mapping.data(),
                              static_cast<std::size_t>(plane_waves) * sizeof(int));

        int bands_per_pass = diagonalizer::planewave::fft_batch_band_count(
            grid_points, std::min(number_of_occupied_bands, maximum_bands_per_pass));
        std::size_t band_bytes = static_cast<std::size_t>(plane_waves) * sizeof(Complex);
        execution::gpu::device::GPUDeviceBuffer<Complex>& device_bands = device_bands_;
        execution::gpu::device::GPUDeviceBuffer<Real>& device_weights = device_weights_;
        execution::gpu::device::GPUDeviceBuffer<Complex>& device_workspace = device_workspace_;
        ensure_capacity(device_weights, static_cast<std::size_t>(bands_per_pass));
        ensure_capacity(device_workspace, static_cast<std::size_t>(bands_per_pass) * grid_points);
        /**
         * The bands are read from the k-point state's block on the device, where the
         * eigensolver left them. The block is uploaded here only if the host wrote it last.
         */
        const Complex* device_band_block =
            quantum_system_.device_band_block(kpoint_index_state, device_backend_);

        prepare_timer.reset();
        for (int pass_start = 0; pass_start < number_of_occupied_bands;
             pass_start += bands_per_pass) {
            int bands_this_pass = std::min(bands_per_pass, number_of_occupied_bands - pass_start);
            std::optional<StageTimer> upload_timer(std::in_place, "density.device.upload_bands");
            int first_band = occupied_band_indices[static_cast<std::size_t>(pass_start)];
            int last_band =
                occupied_band_indices[static_cast<std::size_t>(pass_start + bands_this_pass - 1)];
            const Complex* pass_bands =
                device_band_block + static_cast<std::size_t>(first_band) * plane_waves;
            if (last_band - first_band + 1 != bands_this_pass) {
                /**
                 * Occupied bands with an empty one between them: gathered into one run, on
                 * the device. Occupied bands are otherwise a run of the block already, and
                 * the transform reads them in place.
                 */
                ensure_capacity(device_bands,
                                static_cast<std::size_t>(bands_per_pass) * plane_waves);
                for (int pass_band = 0; pass_band < bands_this_pass; ++pass_band) {
                    int band_index =
                        occupied_band_indices[static_cast<std::size_t>(pass_start + pass_band)];
                    device_backend_->copy(
                        device_bands.data() + static_cast<std::size_t>(pass_band) * plane_waves,
                        device_band_block + static_cast<std::size_t>(band_index) * plane_waves,
                        band_bytes);
                }
                pass_bands = device_bands.data();
            }
            device_backend_->copy(device_weights.data(),
                                  band_weights.data() + pass_start,
                                  static_cast<std::size_t>(bands_this_pass) * sizeof(Real));

            upload_timer.reset();
            StageTimer transform_timer("density.device.transform_and_sum");
            diagonalizer::planewave::gpu::launch_scatter_g_to_fft_grid(bands_this_pass,
                                                                       plane_waves,
                                                                       grid_points,
                                                                       device_mapping.data(),
                                                                       nullptr,
                                                                       false,
                                                                       pass_bands,
                                                                       device_workspace.data());
            gpu_fft_grid_.inverse_transform_batch(device_workspace.data(), bands_this_pass);
            diagonalizer::planewave::gpu::launch_accumulate_density(bands_this_pass,
                                                                    grid_points,
                                                                    device_weights.data(),
                                                                    device_workspace.data(),
                                                                    device_density);
            // Waited for here so that the stage time is the work and not its queueing.
            device_backend_->synchronize();
        }
    }

    /**
     * Grow `buffer` to hold `elements`; never shrink it.
     *
     * The buffers are kept between calls. A device buffer taken fresh is
     * unified memory the device has not touched, and its first use faults every
     * page in: allocated per k-point, the transform workspace alone cost more
     * than the transforms run in it.
     */
    template <typename Element>
    void ensure_capacity(execution::gpu::device::GPUDeviceBuffer<Element>& buffer,
                         std::size_t elements) const {
        if (buffer.size() < elements) {
            buffer = execution::gpu::device::GPUDeviceBuffer<Element>(elements, device_backend_);
        }
    }
};

} // namespace q3
