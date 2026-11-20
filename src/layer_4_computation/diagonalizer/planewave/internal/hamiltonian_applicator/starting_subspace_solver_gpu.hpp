#pragma once

/**
 * GPUStartingSubspaceSolver — the starting rotation's large operations on a
 * device, for a caller that holds its vectors in host memory.
 *
 * The starting vectors go to the device once. H is applied there, the two
 * subspace projections and the rotation are device products, and only the two
 * small matrices come back: the rotated vectors are the k-point state's bands,
 * and are written into its band block on the device, where the eigensolver's
 * first seed reads them.
 */

#include <algorithm>
#include <cstddef>
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>

#include "layer_0_core/execution/gpu/device/init.hpp"
#include "layer_0_core/execution/gpu/device/internal/buffer.hpp"
#include "layer_0_core/math/linear_algebra/gpu/blas/init.hpp"
#include "layer_0_core/profiling/stage_timer.hpp"
#include "layer_0_core/types/init.hpp"
#include "layer_3_representation/quantum_system/kohn_sham/planewave/init.hpp"
#include "layer_3_representation/wavefunction/planewave/starting_subspace_solver.hpp"
#include "layer_4_computation/diagonalizer/planewave/internal/hamiltonian_applicator/init.hpp"

namespace q3 {

/// Applies H, projects and rotates on a device, for vectors held in host memory.
class GPUStartingSubspaceSolver : public StartingSubspaceSolver {
  public:
    /**
     * Wrap `applicator`, which must already be prepared for the k-point and
     * potential, and must outlive this object, as must `gpu_blas`.
     */
    GPUStartingSubspaceSolver(
        PlanewaveHamiltonianApplicator& applicator,
        const execution::gpu::blas::GPUBLAS& gpu_blas,
        std::shared_ptr<execution::gpu::device::GPUDeviceBackend> device_backend,
        int number_of_plane_waves)
        : applicator_(applicator)
        , gpu_blas_(gpu_blas)
        , device_backend_(std::move(device_backend))
        , number_of_plane_waves_(number_of_plane_waves)
        , device_vectors_(0, device_backend_)
        , device_hamiltonian_vectors_(0, device_backend_)
        , device_matrix_(0, device_backend_) {}

    /// Re-target at another k-point's plane-wave count; the applicator is prepared by the caller.
    void set_number_of_plane_waves(int number_of_plane_waves) {
        number_of_plane_waves_ = number_of_plane_waves;
    }

    /// Copy the vectors to the device and apply H there; both stay on the device.
    void apply_hamiltonian(const Complex* vectors, int number_of_vectors) override {
        number_of_vectors_ = number_of_vectors;
        std::size_t elements = static_cast<std::size_t>(number_of_plane_waves_) * number_of_vectors;
        if (elements == 0) {
            return;
        }
        /**
         * Kept between calls and only ever grown: a fresh device buffer is unified memory the
         * device faults in page by page on first use, which for two batch-sized buffers per
         * k-point was a visible part of the rotation.
         */
        if (device_vectors_.size() < elements) {
            device_vectors_ =
                execution::gpu::device::GPUDeviceBuffer<Complex>(elements, device_backend_);
            device_hamiltonian_vectors_ =
                execution::gpu::device::GPUDeviceBuffer<Complex>(elements, device_backend_);
        }
        std::optional<StageTimer> stage_timer(std::in_place, "setup.rotation.hamiltonian.upload");
        device_backend_->copy(device_vectors_.data(), vectors, elements * sizeof(Complex));
        stage_timer.emplace("setup.rotation.hamiltonian.device");
        applicator_.apply_batch(
            device_vectors_.data(), device_hamiltonian_vectors_.data(), number_of_vectors);
        device_backend_->synchronize();
    }

    /// χ†Hχ and χ†χ as two device products; the small results are copied to the host.
    void project(Complex* hamiltonian_matrix, Complex* overlap_matrix) override {
        std::size_t matrix_elements =
            static_cast<std::size_t>(number_of_vectors_) * number_of_vectors_;
        if (matrix_elements == 0) {
            return;
        }
        if (device_matrix_.size() < matrix_elements) {
            device_matrix_ =
                execution::gpu::device::GPUDeviceBuffer<Complex>(matrix_elements, device_backend_);
        }
        project_onto_vectors(device_hamiltonian_vectors_.data(), hamiltonian_matrix);
        project_onto_vectors(device_vectors_.data(), overlap_matrix);
    }

    /**
     * χ·c as one device product, written straight into the k-point state's band block on
     * the device. The bands stay there for the eigensolver's seed; nothing is copied to the
     * host.
     */
    void rotate_into_bands(const Complex* coefficients,
                           int number_of_bands,
                           PlanewaveKohnShamQuantumSystem& quantum_system,
                           int kpoint_index_state) override {
        std::size_t coefficient_elements =
            static_cast<std::size_t>(number_of_vectors_) * number_of_bands;
        if (coefficient_elements == 0 || number_of_plane_waves_ == 0) {
            return;
        }
        if (quantum_system.number_of_plane_waves_for_kpoint_state(kpoint_index_state) !=
                number_of_plane_waves_ ||
            quantum_system.number_of_bands() != number_of_bands) {
            throw std::invalid_argument(
                "GPUStartingSubspaceSolver::rotate_into_bands: the k-point state's band "
                "block is not number_of_plane_waves x number_of_bands as this solver was "
                "prepared and asked for");
        }
        if (device_matrix_.size() < coefficient_elements) {
            device_matrix_ = execution::gpu::device::GPUDeviceBuffer<Complex>(coefficient_elements,
                                                                              device_backend_);
        }
        device_backend_->copy(
            device_matrix_.data(), coefficients, coefficient_elements * sizeof(Complex));
        int leading_dimension = std::max(number_of_plane_waves_, 1);
        gpu_blas_.gemm(
            'N',
            'N',
            number_of_plane_waves_,
            number_of_bands,
            number_of_vectors_,
            Complex(1.0, 0.0),
            device_vectors_.data(),
            leading_dimension,
            device_matrix_.data(),
            number_of_vectors_,
            Complex(0.0, 0.0),
            quantum_system.overwrite_device_band_block(kpoint_index_state, device_backend_),
            leading_dimension);
        device_backend_->synchronize();
    }

  private:
    /// result = χ† · right, number_of_vectors squared, computed on the device.
    void project_onto_vectors(const Complex* device_right, Complex* host_result) {
        int leading_dimension = std::max(number_of_plane_waves_, 1);
        gpu_blas_.gemm('C',
                       'N',
                       number_of_vectors_,
                       number_of_vectors_,
                       number_of_plane_waves_,
                       Complex(1.0, 0.0),
                       device_vectors_.data(),
                       leading_dimension,
                       device_right,
                       leading_dimension,
                       Complex(0.0, 0.0),
                       device_matrix_.data(),
                       number_of_vectors_);
        device_backend_->synchronize();
        device_backend_->copy(host_result,
                              device_matrix_.data(),
                              static_cast<std::size_t>(number_of_vectors_) * number_of_vectors_ *
                                  sizeof(Complex));
    }

    PlanewaveHamiltonianApplicator& applicator_;
    const execution::gpu::blas::GPUBLAS& gpu_blas_;
    std::shared_ptr<execution::gpu::device::GPUDeviceBackend> device_backend_;
    int number_of_plane_waves_;
    int number_of_vectors_ = 0;
    execution::gpu::device::GPUDeviceBuffer<Complex> device_vectors_;
    execution::gpu::device::GPUDeviceBuffer<Complex> device_hamiltonian_vectors_;
    execution::gpu::device::GPUDeviceBuffer<Complex> device_matrix_;
};

} // namespace q3
