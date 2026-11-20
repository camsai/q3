#pragma once

#include <complex>
#include <stdexcept>
#include <string>
#include <vector>

#include "layer_0_core/math/linear_algebra/gpu/lapack/init.hpp"
#include "layer_0_core/math/linear_algebra/lapack.hpp"

namespace q3::test_support {

/// Device eigensolver that runs on the host, through LAPACK's ZHEEV and ZHEGV.
class CPUBackedGPUEigensolver : public execution::gpu::lapack::GPUEigensolver {
  public:
    using GPUEigensolver::GPUEigensolver;

    void hermitian_eigenvalue_solve(int matrix_dimension,
                                    Complex* matrix_data,
                                    int leading_dimension,
                                    Real* eigenvalues) override {
        int workspace_size = 4 * matrix_dimension + 1;
        std::vector<std::complex<double>> workspace(workspace_size);
        std::vector<double> real_workspace(3 * matrix_dimension + 1);
        int info = 0;
        q3::lapack::zheev_("V",
                           "U",
                           &matrix_dimension,
                           reinterpret_cast<std::complex<double>*>(matrix_data),
                           &leading_dimension,
                           eigenvalues,
                           workspace.data(),
                           &workspace_size,
                           real_workspace.data(),
                           &info);
        if (info != 0) {
            throw std::runtime_error("CPUBackedGPUEigensolver: ZHEEV failed, info = " +
                                     std::to_string(info));
        }
    }

    void generalized_hermitian_eigenvalue_solve(int matrix_dimension,
                                                Complex* matrix_a_data,
                                                int leading_dimension_a,
                                                Complex* matrix_b_data,
                                                int leading_dimension_b,
                                                Real* eigenvalues) override {
        int problem_type = 1;
        int workspace_size = 4 * matrix_dimension + 1;
        std::vector<std::complex<double>> workspace(workspace_size);
        std::vector<double> real_workspace(3 * matrix_dimension + 1);
        int info = 0;
        q3::lapack::zhegv_(&problem_type,
                           "V",
                           "U",
                           &matrix_dimension,
                           reinterpret_cast<std::complex<double>*>(matrix_a_data),
                           &leading_dimension_a,
                           reinterpret_cast<std::complex<double>*>(matrix_b_data),
                           &leading_dimension_b,
                           eigenvalues,
                           workspace.data(),
                           &workspace_size,
                           real_workspace.data(),
                           &info);
        if (info != 0) {
            throw std::runtime_error("CPUBackedGPUEigensolver: ZHEGV failed, info = " +
                                     std::to_string(info));
        }
    }
};

} // namespace q3::test_support
