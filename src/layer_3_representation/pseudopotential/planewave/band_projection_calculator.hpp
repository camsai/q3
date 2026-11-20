#pragma once

/**
 * BandProjectionCalculator — the overlaps <column|band> of a set of plane-wave
 * columns with bands of one k-point state.
 *
 * What the projector occupations, the nonlocal force and the nonlocal stress
 * all need: a matrix of columns over a k-point's plane waves (the
 * Kleinman-Bylander projectors, or their derivatives) contracted with the
 * bands. The columns are a DeviceMirroredArray and the bands are the quantum
 * system's block, so an implementation reads each where it is current: the
 * host implementation calls the host BLAS on the host copies, and a device
 * implementation multiplies on the device and brings down only the overlaps,
 * which are small. The callers are the same code either way.
 *
 * Shapes. The columns are column-major with number_of_plane_waves rows; the
 * first number_of_columns of them are used. The overlaps are column-major,
 * number_of_columns rows by one column per entry of band_indices:
 *
 *   overlaps[column + position × number_of_columns]
 *       = Σ_G conj(columns[G, column]) · bands[G, band_indices[position]]
 *
 * band_indices are band indices of the k-point state, in ascending order.
 */

#include <vector>

#include "layer_0_core/execution/gpu/device/device_mirrored_array.hpp"
#include "layer_0_core/types/init.hpp"
#include "layer_3_representation/quantum_system/kohn_sham/planewave/init.hpp"

namespace q3 {

/// Contracts plane-wave columns with the bands of a k-point state; see the file comment.
class BandProjectionCalculator {
  public:
    virtual ~BandProjectionCalculator() = default;

    /// The overlaps of the columns with the bands, as in the file comment.
    virtual void project(const execution::gpu::device::DeviceMirroredArray<Complex>& columns,
                         int number_of_columns,
                         int number_of_plane_waves,
                         const PlanewaveKohnShamQuantumSystem& quantum_system,
                         int kpoint_state_index,
                         const std::vector<int>& band_indices,
                         Complex* overlaps) const = 0;

    /**
     * The overlaps of the columns with the bands after each band coefficient has been
     * multiplied by prefactor × plane_wave_weights[G]:
     *
     *   overlaps = Σ_G conj(columns[G, column]) · prefactor · weights[G] · bands[G, band]
     *
     * A derivative with respect to an atomic position is this with the weights a Cartesian
     * component of k+G and the prefactor -i.
     */
    virtual void
    project_scaled_bands(const execution::gpu::device::DeviceMirroredArray<Complex>& columns,
                         int number_of_columns,
                         int number_of_plane_waves,
                         const PlanewaveKohnShamQuantumSystem& quantum_system,
                         int kpoint_state_index,
                         const std::vector<int>& band_indices,
                         const Real* plane_wave_weights,
                         Complex prefactor,
                         Complex* overlaps) const = 0;

    /**
     * For a half sphere of plane waves (the gamma-only representation): twice the real part
     * of the overlaps, taken as one real product over the 2 × number_of_plane_waves real and
     * imaginary parts. The G = 0 term is counted twice in the result, as it is in any sum
     * over a half sphere that holds the origin; removing one copy is the caller's, which
     * knows whether the origin is among its plane waves.
     */
    virtual void
    project_half_sphere(const execution::gpu::device::DeviceMirroredArray<Complex>& columns,
                        int number_of_columns,
                        int number_of_plane_waves,
                        const PlanewaveKohnShamQuantumSystem& quantum_system,
                        int kpoint_state_index,
                        const std::vector<int>& band_indices,
                        Real* overlaps) const = 0;

  protected:
    /// Whether the indices are one run of consecutive bands, so the block can be read in place.
    [[nodiscard]] static bool is_one_run(const std::vector<int>& band_indices) {
        return band_indices.empty() || band_indices.back() - band_indices.front() + 1 ==
                                           static_cast<int>(band_indices.size());
    }
};

} // namespace q3
