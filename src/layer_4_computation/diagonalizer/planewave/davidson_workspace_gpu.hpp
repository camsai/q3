#pragma once

/**
 * GPUDavidsonWorkspace — the Davidson solver's working arrays in device memory.
 *
 * The basis, its images under H and S, the subspace matrices and the band scratch are
 * device buffers, so the products on them run where they are held. Eigenvalues stay on
 * the host, where the loop control reads them, with a device copy published on request.
 */

#include <cstring>
#include <memory>
#include <utility>

#include "layer_0_core/execution/gpu/device/init.hpp"
#include "layer_0_core/execution/gpu/device/internal/buffer.hpp"
#include "layer_0_core/types/init.hpp"
#include "layer_4_computation/diagonalizer/planewave/davidson.hpp"

namespace q3 {

/// The Davidson working arrays as device buffers.
class GPUDavidsonWorkspace : public PlanewaveDavidsonWorkspace {
  public:
    /// Empty buffers on the given backend's device; resize() allocates them.
    explicit GPUDavidsonWorkspace(
        std::shared_ptr<execution::gpu::device::GPUDeviceBackend> device_backend)
        : device_backend_{std::move(device_backend)}
        , basis_vectors_(0, device_backend_)
        , hamiltonian_basis_vectors_(0, device_backend_)
        , overlap_basis_vectors_(0, device_backend_)
        , subspace_hamiltonian_(0, device_backend_)
        , subspace_overlap_(0, device_backend_)
        , band_scratch_(0, device_backend_)
        , eigenvalues_device_(0, device_backend_) {}

    /// Allocate and clear every array for this basis size, unless it already fits.
    void resize(int maximum_basis_size,
                int number_of_plane_waves,
                int number_of_bands,
                bool overlap_is_identity) override {
        /**
         * The device path still allocates its S buffers. Aliasing them would
         * need the same write audit on every GPU kernel, and that wants its own
         * change with a device to test it on.
         */
        (void)overlap_is_identity;
        int current_size = basis_vectors_.size() / number_of_plane_waves;
        if (current_size >= maximum_basis_size) {
            return;
        }

        basis_vectors_ = execution::gpu::device::GPUDeviceBuffer<Complex>(
            maximum_basis_size * number_of_plane_waves, device_backend_);
        hamiltonian_basis_vectors_ = execution::gpu::device::GPUDeviceBuffer<Complex>(
            maximum_basis_size * number_of_plane_waves, device_backend_);
        overlap_basis_vectors_ = execution::gpu::device::GPUDeviceBuffer<Complex>(
            maximum_basis_size * number_of_plane_waves, device_backend_);
        subspace_hamiltonian_ = execution::gpu::device::GPUDeviceBuffer<Complex>(
            maximum_basis_size * maximum_basis_size, device_backend_);
        subspace_overlap_ = execution::gpu::device::GPUDeviceBuffer<Complex>(
            maximum_basis_size * maximum_basis_size, device_backend_);
        band_scratch_ = execution::gpu::device::GPUDeviceBuffer<Complex>(
            number_of_bands * number_of_plane_waves, device_backend_);
        eigenvalues_device_ =
            execution::gpu::device::GPUDeviceBuffer<Real>(maximum_basis_size, device_backend_);

        local_eigenvalues_host_.assign(maximum_basis_size, 0.0);

        /**
         * Zero out initial memory, on the device. The k-points of a run differ in their
         * plane-wave counts, so the first SCF iteration resizes this workspace more than once,
         * and clearing three basis-sized buffers from the host each time pulled their pages
         * across and left the device to fault them back in.
         */
        auto clear = [this](auto& buffer) {
            device_backend_->fill_zero(buffer.data(), buffer.size() * sizeof(*buffer.data()));
        };
        clear(basis_vectors_);
        clear(hamiltonian_basis_vectors_);
        clear(overlap_basis_vectors_);
        clear(subspace_hamiltonian_);
        clear(subspace_overlap_);
        clear(band_scratch_);
        clear(eigenvalues_device_);
    }

    /// The basis vectors V, on the device.
    Complex* basis_vectors() override {
        return basis_vectors_.data();
    }
    /// The basis vectors V, on the device.
    const Complex* basis_vectors() const override {
        return basis_vectors_.data();
    }

    /// H·V, on the device.
    Complex* hamiltonian_basis_vectors() override {
        return hamiltonian_basis_vectors_.data();
    }
    /// H·V, on the device.
    const Complex* hamiltonian_basis_vectors() const override {
        return hamiltonian_basis_vectors_.data();
    }

    /// S·V, on the device, in a buffer of its own.
    Complex* overlap_basis_vectors() override {
        return overlap_basis_vectors_.data();
    }
    /// S·V, on the device, in a buffer of its own.
    const Complex* overlap_basis_vectors() const override {
        return overlap_basis_vectors_.data();
    }

    /// V†·H·V, on the device.
    Complex* subspace_hamiltonian() override {
        return subspace_hamiltonian_.data();
    }
    /// V†·H·V, on the device.
    const Complex* subspace_hamiltonian() const override {
        return subspace_hamiltonian_.data();
    }

    /// V†·S·V, on the device.
    Complex* subspace_overlap() override {
        return subspace_overlap_.data();
    }
    /// V†·S·V, on the device.
    const Complex* subspace_overlap() const override {
        return subspace_overlap_.data();
    }

    /// Scratch for one block of bands, on the device.
    Complex* band_scratch() override {
        return band_scratch_.data();
    }
    /// Scratch for one block of bands, on the device.
    const Complex* band_scratch() const override {
        return band_scratch_.data();
    }

    /// The device copy of the eigenvalues.
    Real* eigenvalues_device() {
        return eigenvalues_device_.data();
    }
    /// The device copy of the eigenvalues.
    const Real* eigenvalues_device() const {
        return eigenvalues_device_.data();
    }

    /// Copy the host eigenvalues into their device copy.
    void sync_to_device() override {
        std::memcpy(eigenvalues_device_.data(),
                    local_eigenvalues_host_.data(),
                    local_eigenvalues_host_.size() * sizeof(Real));
    }

    /// Wait for the device to finish what was queued on these arrays.
    void sync_to_host() override {
        device_backend_->synchronize();
    }

  private:
    std::shared_ptr<execution::gpu::device::GPUDeviceBackend> device_backend_;
    execution::gpu::device::GPUDeviceBuffer<Complex> basis_vectors_;
    execution::gpu::device::GPUDeviceBuffer<Complex> hamiltonian_basis_vectors_;
    execution::gpu::device::GPUDeviceBuffer<Complex> overlap_basis_vectors_;
    execution::gpu::device::GPUDeviceBuffer<Complex> subspace_hamiltonian_;
    execution::gpu::device::GPUDeviceBuffer<Complex> subspace_overlap_;
    execution::gpu::device::GPUDeviceBuffer<Complex> band_scratch_;
    execution::gpu::device::GPUDeviceBuffer<Real> eigenvalues_device_;
};

} // namespace q3
