#pragma once

/**
 * GPUBandProjectionCalculator — band projections on a device.
 *
 * The product is taken where the operands already are. The columns are read
 * through their device view, which uploads them once and again only after they
 * have been rewritten on the host; the bands are read from the k-point state's
 * device block, where the eigensolver left them. What comes down to the host is
 * the overlap matrix: columns × bands numbers, against the plane waves × bands
 * that a host product would have needed brought down first.
 *
 * Bands that are one run of the block are read in place; bands with a gap
 * between them are gathered on the device. Scaling the bands by a weight per
 * plane wave uses the kernel that applies the kinetic energy, which is exactly
 * that operation: a real diagonal applied to a batch of vectors.
 *
 * The workspaces are kept between calls. A device buffer taken fresh is unified
 * memory the device has not touched, and its first use faults every page in.
 * They are shared, so calls are serialized.
 */

#include <cstddef>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <utility>
#include <vector>

#include "layer_0_core/execution/gpu/device/init.hpp"
#include "layer_0_core/execution/gpu/device/internal/buffer.hpp"
#include "layer_0_core/math/linear_algebra/gpu/blas/init.hpp"
#include "layer_0_core/types/init.hpp"
#include "layer_3_representation/pseudopotential/planewave/band_projection_calculator.hpp"
#include "layer_4_computation/diagonalizer/planewave/internal/hamiltonian_applicator/band_matrix_gpu.hpp"
#include "layer_4_computation/diagonalizer/planewave/internal/hamiltonian_applicator/gpu_kernels.hpp"

namespace q3 {

/// Contracts plane-wave columns with bands on a device; see the file comment.
class GPUBandProjectionCalculator : public BandProjectionCalculator {
  public:
    /**
     * @param device_backend  The device the products run on. Must not be null.
     * @param device_blas     The device BLAS of that backend. Must outlive this object.
     */
    GPUBandProjectionCalculator(
        std::shared_ptr<execution::gpu::device::GPUDeviceBackend> device_backend,
        const execution::gpu::blas::GPUBLAS& device_blas)
        : device_backend_(require(std::move(device_backend)))
        , device_blas_(device_blas)
        , device_bands_(0, device_backend_)
        , device_weights_(0, device_backend_)
        , device_overlaps_(0, device_backend_)
        , device_real_overlaps_(0, device_backend_) {}

    /// See BandProjectionCalculator::project(); taken on the device.
    void project(const execution::gpu::device::DeviceMirroredArray<Complex>& columns,
                 int number_of_columns,
                 int number_of_plane_waves,
                 const PlanewaveKohnShamQuantumSystem& quantum_system,
                 int kpoint_state_index,
                 const std::vector<int>& band_indices,
                 Complex* overlaps) const override {
        multiply(columns,
                 number_of_columns,
                 number_of_plane_waves,
                 quantum_system,
                 kpoint_state_index,
                 band_indices,
                 nullptr,
                 Complex(1.0, 0.0),
                 overlaps);
    }

    /// See BandProjectionCalculator::project_scaled_bands(); taken on the device.
    void project_scaled_bands(const execution::gpu::device::DeviceMirroredArray<Complex>& columns,
                              int number_of_columns,
                              int number_of_plane_waves,
                              const PlanewaveKohnShamQuantumSystem& quantum_system,
                              int kpoint_state_index,
                              const std::vector<int>& band_indices,
                              const Real* plane_wave_weights,
                              Complex prefactor,
                              Complex* overlaps) const override {
        multiply(columns,
                 number_of_columns,
                 number_of_plane_waves,
                 quantum_system,
                 kpoint_state_index,
                 band_indices,
                 plane_wave_weights,
                 prefactor,
                 overlaps);
    }

    /// See BandProjectionCalculator::project_half_sphere(); taken on the device.
    void project_half_sphere(const execution::gpu::device::DeviceMirroredArray<Complex>& columns,
                             int number_of_columns,
                             int number_of_plane_waves,
                             const PlanewaveKohnShamQuantumSystem& quantum_system,
                             int kpoint_state_index,
                             const std::vector<int>& band_indices,
                             Real* overlaps) const override {
        if (number_of_columns == 0 || band_indices.empty()) {
            return;
        }
        std::lock_guard<std::mutex> lock(workspace_mutex_);
        const int number_of_bands = static_cast<int>(band_indices.size());
        const Complex* device_columns = columns.device(device_backend_);
        const Complex* device_bands = bands_as_one_matrix(
            quantum_system, kpoint_state_index, number_of_plane_waves, band_indices);
        const std::size_t number_of_overlaps =
            static_cast<std::size_t>(number_of_columns) * static_cast<std::size_t>(number_of_bands);
        ensure_capacity(device_real_overlaps_, number_of_overlaps);
        device_blas_.dgemm('T',
                           'N',
                           number_of_columns,
                           number_of_bands,
                           2 * number_of_plane_waves,
                           2.0,
                           reinterpret_cast<const Real*>(device_columns),
                           2 * number_of_plane_waves,
                           reinterpret_cast<const Real*>(device_bands),
                           2 * number_of_plane_waves,
                           0.0,
                           device_real_overlaps_.data(),
                           number_of_columns);
        device_backend_->synchronize();
        device_backend_->copy(
            overlaps, device_real_overlaps_.data(), number_of_overlaps * sizeof(Real));
    }

  private:
    [[nodiscard]] static std::shared_ptr<execution::gpu::device::GPUDeviceBackend>
    require(std::shared_ptr<execution::gpu::device::GPUDeviceBackend> device_backend) {
        if (!device_backend) {
            throw std::invalid_argument(
                "GPUBandProjectionCalculator: the device backend must not be null");
        }
        return device_backend;
    }

    /// overlaps = columns† · (prefactor × weights ∘ bands); no weights means none are applied.
    void multiply(const execution::gpu::device::DeviceMirroredArray<Complex>& columns,
                  int number_of_columns,
                  int number_of_plane_waves,
                  const PlanewaveKohnShamQuantumSystem& quantum_system,
                  int kpoint_state_index,
                  const std::vector<int>& band_indices,
                  const Real* plane_wave_weights,
                  Complex prefactor,
                  Complex* overlaps) const {
        if (number_of_columns == 0 || band_indices.empty()) {
            return;
        }
        std::lock_guard<std::mutex> lock(workspace_mutex_);
        const int number_of_bands = static_cast<int>(band_indices.size());
        const std::size_t plane_waves = static_cast<std::size_t>(number_of_plane_waves);
        const Complex* device_columns = columns.device(device_backend_);
        const Complex* device_bands = bands_as_one_matrix(
            quantum_system, kpoint_state_index, number_of_plane_waves, band_indices);

        if (plane_wave_weights != nullptr) {
            ensure_capacity(device_weights_, plane_waves);
            device_backend_->copy(
                device_weights_.data(), plane_wave_weights, plane_waves * sizeof(Real));
            /**
             * Scaled into the gather workspace. When the bands were gathered there, the kernel
             * reads and writes the same element of the same vector, which it may.
             */
            ensure_capacity(device_bands_, plane_waves * static_cast<std::size_t>(number_of_bands));
            diagonalizer::planewave::gpu::launch_apply_kinetic_energy(number_of_bands,
                                                                      number_of_plane_waves,
                                                                      device_weights_.data(),
                                                                      device_bands,
                                                                      device_bands_.data());
            device_bands = device_bands_.data();
        }

        const std::size_t number_of_overlaps =
            static_cast<std::size_t>(number_of_columns) * static_cast<std::size_t>(number_of_bands);
        ensure_capacity(device_overlaps_, number_of_overlaps);
        device_blas_.gemm('C',
                          'N',
                          number_of_columns,
                          number_of_bands,
                          number_of_plane_waves,
                          prefactor,
                          device_columns,
                          number_of_plane_waves,
                          device_bands,
                          number_of_plane_waves,
                          Complex(0.0, 0.0),
                          device_overlaps_.data(),
                          number_of_columns);
        device_backend_->synchronize();
        device_backend_->copy(
            overlaps, device_overlaps_.data(), number_of_overlaps * sizeof(Complex));
    }

    /// The chosen bands on the device, in place when they are one run; see band_matrix_gpu.hpp.
    [[nodiscard]] const Complex*
    bands_as_one_matrix(const PlanewaveKohnShamQuantumSystem& quantum_system,
                        int kpoint_state_index,
                        int number_of_plane_waves,
                        const std::vector<int>& band_indices) const {
        return bands_as_one_matrix_gpu(quantum_system,
                                       kpoint_state_index,
                                       number_of_plane_waves,
                                       band_indices,
                                       device_backend_,
                                       device_bands_);
    }

    /// Grow `buffer` to hold `elements`; never shrink it.
    template <typename Element>
    void ensure_capacity(execution::gpu::device::GPUDeviceBuffer<Element>& buffer,
                         std::size_t elements) const {
        if (buffer.size() < elements) {
            buffer = execution::gpu::device::GPUDeviceBuffer<Element>(elements, device_backend_);
        }
    }

    std::shared_ptr<execution::gpu::device::GPUDeviceBackend> device_backend_;
    const execution::gpu::blas::GPUBLAS& device_blas_;
    mutable std::mutex workspace_mutex_;
    mutable execution::gpu::device::GPUDeviceBuffer<Complex> device_bands_;
    mutable execution::gpu::device::GPUDeviceBuffer<Real> device_weights_;
    mutable execution::gpu::device::GPUDeviceBuffer<Complex> device_overlaps_;
    mutable execution::gpu::device::GPUDeviceBuffer<Real> device_real_overlaps_;
};

} // namespace q3
