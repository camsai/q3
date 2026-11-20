#pragma once

#include <cmath>
#include <complex>
#include <stdexcept>

#include "layer_0_core/math/linear_algebra/blas.hpp"
#include "layer_0_core/math/linear_algebra/gpu/blas/init.hpp"
#include "layer_0_core/math/linear_algebra/gpu/blas/triangular_solve_serial.hpp"

namespace q3::test_support {

/**
 * Device BLAS that runs on the host: the matrix products and the triangular
 * solves go to the host BLAS and the vector operations are plain loops. Single
 * precision is refused.
 */
class CPUBackedGPUBLAS : public execution::gpu::blas::GPUBLAS {
  public:
    using GPUBLAS::GPUBLAS;

    void gemm(char transpose_a,
              char transpose_b,
              int m,
              int n,
              int k,
              Complex alpha,
              const Complex* matrix_a,
              int leading_dimension_a,
              const Complex* matrix_b,
              int leading_dimension_b,
              Complex beta,
              Complex* matrix_c,
              int leading_dimension_c) const override {
        q3::blas::zgemm_(&transpose_a,
                         &transpose_b,
                         &m,
                         &n,
                         &k,
                         reinterpret_cast<const std::complex<double>*>(&alpha),
                         reinterpret_cast<const std::complex<double>*>(matrix_a),
                         &leading_dimension_a,
                         reinterpret_cast<const std::complex<double>*>(matrix_b),
                         &leading_dimension_b,
                         reinterpret_cast<const std::complex<double>*>(&beta),
                         reinterpret_cast<std::complex<double>*>(matrix_c),
                         &leading_dimension_c);
    }
    Complex dot_product(int number_of_elements,
                        const Complex* vector_x,
                        const Complex* vector_y) const override {
        Complex result(0.0, 0.0);
        for (int element = 0; element < number_of_elements; ++element) {
            result += std::conj(vector_x[element]) * vector_y[element];
        }
        return result;
    }
    double norm(int number_of_elements, const Complex* vector_x) const override {
        double sum_of_squares = 0.0;
        for (int element = 0; element < number_of_elements; ++element) {
            sum_of_squares += std::norm(vector_x[element]);
        }
        return std::sqrt(sum_of_squares);
    }
    void scale(int number_of_elements, Complex scaling_factor, Complex* vector_x) const override {
        for (int element = 0; element < number_of_elements; ++element) {
            vector_x[element] *= scaling_factor;
        }
    }
    void axpy(int number_of_elements,
              Complex scaling_factor,
              const Complex* vector_x,
              Complex* vector_y) const override {
        for (int element = 0; element < number_of_elements; ++element) {
            vector_y[element] += scaling_factor * vector_x[element];
        }
    }
    void gemm_single(char,
                     char,
                     int,
                     int,
                     int,
                     ComplexSingle,
                     const ComplexSingle*,
                     int,
                     const ComplexSingle*,
                     int,
                     ComplexSingle,
                     ComplexSingle*,
                     int) const override {
        throw std::logic_error("CPUBackedGPUBLAS: single precision is not implemented");
    }
    ComplexSingle
    dot_product_single(int, const ComplexSingle*, const ComplexSingle*) const override {
        throw std::logic_error("CPUBackedGPUBLAS: single precision is not implemented");
    }
    float norm_single(int, const ComplexSingle*) const override {
        throw std::logic_error("CPUBackedGPUBLAS: single precision is not implemented");
    }
    void dgemm(char transpose_a,
               char transpose_b,
               int m,
               int n,
               int k,
               Real alpha,
               const Real* matrix_a,
               int leading_dimension_a,
               const Real* matrix_b,
               int leading_dimension_b,
               Real beta,
               Real* matrix_c,
               int leading_dimension_c) const override {
        q3::blas::dgemm_(&transpose_a,
                         &transpose_b,
                         &m,
                         &n,
                         &k,
                         &alpha,
                         matrix_a,
                         &leading_dimension_a,
                         matrix_b,
                         &leading_dimension_b,
                         &beta,
                         matrix_c,
                         &leading_dimension_c);
    }
    void trsm(char side,
              char upper_or_lower,
              char transpose_a,
              char unit_diagonal,
              int m,
              int n,
              Complex alpha,
              const Complex* matrix_a,
              int leading_dimension_a,
              Complex* matrix_b,
              int leading_dimension_b) const override {
        execution::gpu::blas::triangular_solve_serial(side,
                                                      upper_or_lower,
                                                      transpose_a,
                                                      unit_diagonal,
                                                      m,
                                                      n,
                                                      alpha,
                                                      matrix_a,
                                                      leading_dimension_a,
                                                      matrix_b,
                                                      leading_dimension_b);
    }
    void dtrsm(char side,
               char upper_or_lower,
               char transpose_a,
               char unit_diagonal,
               int m,
               int n,
               Real alpha,
               const Real* matrix_a,
               int leading_dimension_a,
               Real* matrix_b,
               int leading_dimension_b) const override {
        execution::gpu::blas::triangular_solve_serial(side,
                                                      upper_or_lower,
                                                      transpose_a,
                                                      unit_diagonal,
                                                      m,
                                                      n,
                                                      alpha,
                                                      matrix_a,
                                                      leading_dimension_a,
                                                      matrix_b,
                                                      leading_dimension_b);
    }
};

} // namespace q3::test_support
