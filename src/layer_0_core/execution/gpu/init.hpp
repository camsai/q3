#pragma once

/**
 * GPUExecutionPolicy — GPU implementation of ExecutionPolicy.
 *
 * Delegates group-collective vector operations (dot product, norm, axpy) to
 * GPUBLAS for device-resident data. All pointer arguments must come from
 * GPUDeviceBuffer (unified memory).
 *
 * GPUBLAS is implemented in Phase 1. This file defines the interface and
 * constructor signature; method bodies will delegate to GPUBLAS once it exists.
 *
 * No CUDA headers. No Metal headers. Vendor-neutral.
 */

#include "layer_0_core/execution/gpu/device/init.hpp"
#include "layer_0_core/execution/init.hpp"

#include <cstring>
#include <memory>
#include <stdexcept>

namespace q3 {
namespace execution {
namespace gpu {

/**
 * Forward declaration — GPUBLAS is defined at:
 * layer_0_core/math/linear_algebra/gpu/blas/init.hpp
 */
namespace blas {
class GPUBLAS;
} // namespace blas

class GPUExecutionPolicy : public virtual ExecutionPolicy {
  public:
    /**
     * Construct a GPU execution policy.
     *
     * @param device_backend  Shared pointer to the GPU device backend.
     *                        Must not be null.
     */
    explicit GPUExecutionPolicy(std::shared_ptr<device::GPUDeviceBackend> device_backend)
        : device_backend_(std::move(device_backend)) {
        if (!device_backend_) {
            throw std::runtime_error("GPUExecutionPolicy: device_backend must not be null");
        }
    }

    /**
     * Compute the inner product x† · y on GPU-resident data.
     *
     * Intentionally unwired: the GPU compute paths (hamiltonian applicator,
     * subspace manager) call GPUBLAS directly on device buffers instead of
     * going through this generic policy interface. Throws if reached.
     */
    ComplexValue dot_product(int number_of_elements,
                             const ComplexValue* vector_x,
                             const ComplexValue* vector_y) const override {
        (void)number_of_elements;
        (void)vector_x;
        (void)vector_y;
        throw std::runtime_error("GPUExecutionPolicy::dot_product is not wired — GPU compute "
                                 "paths use GPUBLAS directly on device buffers");
    }

    /**
     * Scaled vector addition on GPU-resident data: vector_y += scaling_factor * vector_x.
     *
     * Intentionally unwired: the GPU compute paths call GPUBLAS directly on
     * device buffers instead of going through this generic policy interface.
     * Throws if reached.
     */
    void add_scaled_vector(int number_of_elements,
                           ComplexValue scaling_factor,
                           const ComplexValue* vector_x,
                           ComplexValue* vector_y) const override {
        (void)number_of_elements;
        (void)scaling_factor;
        (void)vector_x;
        (void)vector_y;
        throw std::runtime_error("GPUExecutionPolicy::add_scaled_vector is not wired — GPU "
                                 "compute paths use GPUBLAS directly on device buffers");
    }

    /**
     * Euclidean norm on GPU-resident data.
     *
     * Intentionally unwired: the GPU compute paths call GPUBLAS directly on
     * device buffers instead of going through this generic policy interface.
     * Throws if reached.
     */
    Real norm(int number_of_elements, const ComplexValue* vector_x) const override {
        (void)number_of_elements;
        (void)vector_x;
        throw std::runtime_error("GPUExecutionPolicy::norm is not wired — GPU compute "
                                 "paths use GPUBLAS directly on device buffers");
    }

    /**
     * GPU execution is always single-process (no MPI in this policy).
     */
    bool is_root() const override {
        return true;
    }

    int group_rank_index() const override {
        return 0;
    }

    int group_size() const override {
        return 1;
    }

    int number_of_processes() const override {
        return 1;
    }

    int global_number_of_elements(int local_number_of_elements) const override {
        return local_number_of_elements;
    }

    void reduce_within_group(ComplexValue* /*data*/, int /*number_of_elements*/) const override {}

    void reduce_within_group(Real* /*data*/, int /*number_of_elements*/) const override {}

    void allgather_within_group(const Real* local_data,
                                int local_count,
                                Real* full_data,
                                int /*total_count*/) const override {
        std::memcpy(full_data, local_data, local_count * sizeof(Real));
    }

    /**
     * Access the underlying GPU device backend.
     */
    const device::GPUDeviceBackend& device_backend() const {
        return *device_backend_;
    }

  protected:
    std::shared_ptr<device::GPUDeviceBackend> device_backend_;
};

} // namespace gpu
} // namespace execution
} // namespace q3
