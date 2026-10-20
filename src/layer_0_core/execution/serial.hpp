#pragma once

/**
 * SerialExecutionPolicy — serial CPU implementation of ExecutionPolicy.
 *
 * Delegates group-collective vector operations to BLAS routines on host memory.
 * No inter-process communication. Used for single-process builds and as
 * the group-collective baseline inside PlanewaveSerialExecutionPolicy.
 *
 * No MPI headers. No CUDA headers. No global state.
 */

#include "layer_0_core/execution/init.hpp"
#include "layer_0_core/math/linear_algebra/blas.hpp"

#include <cmath>
#include <cstring>

namespace q3 {
namespace execution {

class SerialExecutionPolicy : public virtual ExecutionPolicy {
  public:
    /**
     * Compute the inner product x† · y using the existing q3_complex_conjugate_dot_product helper.
     */
    ComplexValue dot_product(int number_of_elements,
                             const ComplexValue* vector_x,
                             const ComplexValue* vector_y) const override {
        return blas::q3_complex_conjugate_dot_product(number_of_elements, vector_x, vector_y);
    }

    /**
     * Scaled vector addition via BLAS zaxpy_: vector_y += scaling_factor * vector_x.
     */
    void add_scaled_vector(int number_of_elements,
                           ComplexValue scaling_factor,
                           const ComplexValue* vector_x,
                           ComplexValue* vector_y) const override {
        const int stride = 1;
        blas::zaxpy_(&number_of_elements, &scaling_factor, vector_x, &stride, vector_y, &stride);
    }

    /**
     * Euclidean norm via BLAS dznrm2_.
     */
    Real norm(int number_of_elements, const ComplexValue* vector_x) const override {
        const int stride = 1;
        return blas::dznrm2_(&number_of_elements, vector_x, &stride);
    }

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
};

} // namespace execution
} // namespace q3
