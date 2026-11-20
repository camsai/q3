#pragma once

/**
 * Bands of a k-point state as one column-major matrix on a device.
 *
 * Shared by the device consumers that multiply something into a selection of
 * bands. Bands that are one run of the k-point state's block are read in
 * place, from the device copy of the block; bands with a gap between them are
 * gathered, device to device, into a workspace the caller keeps.
 */

#include <cstddef>
#include <memory>
#include <vector>

#include "layer_0_core/execution/gpu/device/init.hpp"
#include "layer_0_core/execution/gpu/device/internal/buffer.hpp"
#include "layer_0_core/types/init.hpp"
#include "layer_3_representation/quantum_system/kohn_sham/planewave/init.hpp"

namespace q3 {

/**
 * The bands `band_indices` (ascending) of a k-point state, as number_of_plane_waves rows by
 * band_indices.size() columns on the device. The pointer is into the state's device block
 * when the bands are consecutive, and into `gather_workspace` otherwise, which is grown to
 * fit and never shrunk.
 */
[[nodiscard]] inline const Complex* bands_as_one_matrix_gpu(
    const PlanewaveKohnShamQuantumSystem& quantum_system,
    int kpoint_state_index,
    int number_of_plane_waves,
    const std::vector<int>& band_indices,
    const std::shared_ptr<execution::gpu::device::GPUDeviceBackend>& device_backend,
    execution::gpu::device::GPUDeviceBuffer<Complex>& gather_workspace) {
    const std::size_t plane_waves = static_cast<std::size_t>(number_of_plane_waves);
    const Complex* device_block =
        quantum_system.device_band_block(kpoint_state_index, device_backend);
    const bool is_one_run =
        band_indices.empty() ||
        band_indices.back() - band_indices.front() + 1 == static_cast<int>(band_indices.size());
    if (is_one_run) {
        return band_indices.empty()
                   ? device_block
                   : device_block + static_cast<std::size_t>(band_indices.front()) * plane_waves;
    }
    if (gather_workspace.size() < plane_waves * band_indices.size()) {
        gather_workspace = execution::gpu::device::GPUDeviceBuffer<Complex>(
            plane_waves * band_indices.size(), device_backend);
    }
    for (std::size_t position = 0; position < band_indices.size(); ++position) {
        device_backend->copy(gather_workspace.data() + position * plane_waves,
                             device_block +
                                 static_cast<std::size_t>(band_indices[position]) * plane_waves,
                             plane_waves * sizeof(Complex));
    }
    return gather_workspace.data();
}

} // namespace q3
