#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <vector>

#include "layer_0_core/execution/gpu/device/device_transfer_totals.hpp"
#include "layer_0_core/execution/gpu/device/internal/buffer.hpp"
#include "layer_0_core/execution/serial.hpp"
#include "layer_4_computation/diagonalizer/planewave/internal/overlap/init.hpp"
#include "layer_4_computation/diagonalizer/planewave/internal/overlap/overlap_block_gpu_copied.hpp"
#include "layer_4_computation/diagonalizer/planewave/internal/subspace_manager/gpu.hpp"
#include "utils/gpu_blas_cpu_backed.hpp"
#include "utils/gpu_device_backend_cpu_backed.hpp"
#include "utils/gpu_eigensolver_cpu_backed.hpp"

using namespace q3;
using execution::gpu::device::DeviceTransferTotals;
using execution::gpu::device::GPUDeviceBuffer;
using test_support::CPUBackedGPUBLAS;
using test_support::CPUBackedGPUDeviceBackend;
using test_support::CPUBackedGPUEigensolver;

namespace {

constexpr int basis_size = 5;
constexpr int maximum_basis_size = 8;

/// A device eigensolver that always reports failure.
class FailingEigensolver : public CPUBackedGPUEigensolver {
  public:
    using CPUBackedGPUEigensolver::CPUBackedGPUEigensolver;
    void generalized_hermitian_eigenvalue_solve(int, Complex*, int, Complex*, int, Real*) override {
        throw std::runtime_error("the stand-in eigensolver was told to fail");
    }
};

/**
 * A Hermitian H and a positive-definite Hermitian S of order basis_size, held the way the
 * Davidson holds them: in device buffers with the leading dimension of its largest subspace.
 */
struct SubspaceProblem {
    std::shared_ptr<CPUBackedGPUDeviceBackend> device_backend =
        std::make_shared<CPUBackedGPUDeviceBackend>();
    CPUBackedGPUBLAS device_blas{device_backend};
    execution::SerialExecutionPolicy execution_policy;
    GPUPlanewaveSubspaceManager manager{device_blas, execution_policy, device_backend};
    GPUDeviceBuffer<Complex> hamiltonian{
        static_cast<std::size_t>(maximum_basis_size) * maximum_basis_size, device_backend};
    GPUDeviceBuffer<Complex> overlap{
        static_cast<std::size_t>(maximum_basis_size) * maximum_basis_size, device_backend};

    SubspaceProblem() {
        // The stand-in's device memory is host memory, so the matrices are written directly.
        for (int column = 0; column < maximum_basis_size; ++column) {
            for (int row = 0; row < maximum_basis_size; ++row) {
                element(hamiltonian, row, column) = Complex(99.0, 99.0);
                element(overlap, row, column) = Complex(99.0, 99.0);
            }
        }
        for (int column = 0; column < basis_size; ++column) {
            for (int row = 0; row <= column; ++row) {
                Complex off_diagonal(0.1 * std::sin(1.0 + row + 2.0 * column),
                                     row == column ? 0.0 : 0.07 * std::cos(row - 3.0 * column));
                Complex hamiltonian_value =
                    row == column ? Complex(1.0 + 0.6 * row, 0.0) : 3.0 * off_diagonal;
                Complex overlap_value = row == column ? Complex(1.0, 0.0) : off_diagonal;
                element(hamiltonian, row, column) = hamiltonian_value;
                element(hamiltonian, column, row) = std::conj(hamiltonian_value);
                element(overlap, row, column) = overlap_value;
                element(overlap, column, row) = std::conj(overlap_value);
            }
        }
    }

    static Complex& element(GPUDeviceBuffer<Complex>& matrix, int row, int column) {
        return matrix.data()[static_cast<std::size_t>(column) * maximum_basis_size + row];
    }

    /// Largest |H v - λ S v| over the eigenpairs, for eigenvectors of leading dimension basis_size.
    Real largest_residual(const std::vector<Real>& eigenvalues,
                          const std::vector<Complex>& eigenvectors) {
        Real largest = 0.0;
        for (int pair = 0; pair < basis_size; ++pair) {
            for (int row = 0; row < basis_size; ++row) {
                Complex residual(0.0, 0.0);
                for (int column = 0; column < basis_size; ++column) {
                    Complex coefficient =
                        eigenvectors[static_cast<std::size_t>(pair) * basis_size + column];
                    residual += (element(hamiltonian, row, column) -
                                 eigenvalues[pair] * element(overlap, row, column)) *
                                coefficient;
                }
                largest = std::max(largest, std::abs(residual));
            }
        }
        return largest;
    }

    int
    solve(std::vector<Real>& eigenvalues, std::vector<Complex>& eigenvectors, bool is_gamma_only) {
        eigenvalues.assign(basis_size, 0.0);
        eigenvectors.assign(static_cast<std::size_t>(maximum_basis_size) * maximum_basis_size,
                            Complex(0.0, 0.0));
        return manager.diagonalize_subspace(basis_size,
                                            maximum_basis_size,
                                            is_gamma_only,
                                            hamiltonian.data(),
                                            overlap.data(),
                                            eigenvalues.data(),
                                            eigenvectors.data());
    }
};

} // namespace

TEST(GPUPlanewaveSubspaceManager, TheDeviceSolveGivesTheEigenpairsOfTheHostSolve) {
    SubspaceProblem problem;
    std::vector<Real> host_eigenvalues;
    std::vector<Complex> host_eigenvectors;
    ASSERT_EQ(problem.solve(host_eigenvalues, host_eigenvectors, /*is_gamma_only=*/false), 0);

    CPUBackedGPUEigensolver device_eigensolver(problem.device_backend);
    problem.manager.set_device_eigensolver(&device_eigensolver);
    std::vector<Real> device_eigenvalues;
    std::vector<Complex> device_eigenvectors;
    ASSERT_EQ(problem.solve(device_eigenvalues, device_eigenvectors, /*is_gamma_only=*/false), 0);

    for (int pair = 0; pair < basis_size; ++pair) {
        EXPECT_NEAR(device_eigenvalues[pair], host_eigenvalues[pair], 1.0e-12);
    }
    EXPECT_LT(problem.largest_residual(host_eigenvalues, host_eigenvectors), 1.0e-12);
    EXPECT_LT(problem.largest_residual(device_eigenvalues, device_eigenvectors), 1.0e-12);
}

TEST(GPUPlanewaveSubspaceManager, TheDeviceSolveLeavesTheCallersMatricesAsTheyWere) {
    SubspaceProblem problem;
    const std::vector<Complex> hamiltonian_before(
        problem.hamiltonian.data(), problem.hamiltonian.data() + problem.hamiltonian.size());
    const std::vector<Complex> overlap_before(problem.overlap.data(),
                                              problem.overlap.data() + problem.overlap.size());
    CPUBackedGPUEigensolver device_eigensolver(problem.device_backend);
    problem.manager.set_device_eigensolver(&device_eigensolver);
    std::vector<Real> eigenvalues;
    std::vector<Complex> eigenvectors;

    ASSERT_EQ(problem.solve(eigenvalues, eigenvectors, /*is_gamma_only=*/false), 0);

    for (std::size_t element = 0; element < hamiltonian_before.size(); ++element) {
        EXPECT_EQ(problem.hamiltonian.data()[element], hamiltonian_before[element]) << element;
        EXPECT_EQ(problem.overlap.data()[element], overlap_before[element]) << element;
    }
}

TEST(GPUPlanewaveSubspaceManager, TheDeviceSolveBringsDownTheEigenvectorsAndEigenvaluesOnly) {
    SubspaceProblem problem;
    const std::size_t staged_bytes =
        static_cast<std::size_t>(basis_size) * maximum_basis_size * sizeof(Complex);
    std::vector<Real> eigenvalues;
    std::vector<Complex> eigenvectors;

    DeviceTransferTotals before = problem.device_backend->transfer_totals();
    ASSERT_EQ(problem.solve(eigenvalues, eigenvectors, /*is_gamma_only=*/false), 0);
    const DeviceTransferTotals host_solve = problem.device_backend->transfer_totals().since(before);

    CPUBackedGPUEigensolver device_eigensolver(problem.device_backend);
    problem.manager.set_device_eigensolver(&device_eigensolver);
    before = problem.device_backend->transfer_totals();
    ASSERT_EQ(problem.solve(eigenvalues, eigenvectors, /*is_gamma_only=*/false), 0);
    const DeviceTransferTotals device_solve =
        problem.device_backend->transfer_totals().since(before);

    // Host solve: both subspace matrices come down. Device solve: one matrix and the eigenvalues.
    EXPECT_EQ(host_solve.device_to_host_bytes, 2 * staged_bytes);
    EXPECT_EQ(device_solve.device_to_host_bytes, staged_bytes + basis_size * sizeof(Real));
    EXPECT_EQ(device_solve.host_to_device_bytes, 0u);
}

TEST(GPUPlanewaveSubspaceManager, AFailedDeviceSolveIsReportedAsAFailedSolve) {
    SubspaceProblem problem;
    FailingEigensolver device_eigensolver(problem.device_backend);
    problem.manager.set_device_eigensolver(&device_eigensolver);
    std::vector<Real> eigenvalues;
    std::vector<Complex> eigenvectors;

    EXPECT_NE(problem.solve(eigenvalues, eigenvectors, /*is_gamma_only=*/false), 0);
}

TEST(GPUPlanewaveSubspaceManager, WithoutADeviceEigensolverTheHostSolves) {
    SubspaceProblem problem;
    FailingEigensolver device_eigensolver(problem.device_backend);
    problem.manager.set_device_eigensolver(&device_eigensolver);
    problem.manager.set_device_eigensolver(nullptr);
    std::vector<Real> eigenvalues;
    std::vector<Complex> eigenvectors;

    ASSERT_EQ(problem.solve(eigenvalues, eigenvectors, /*is_gamma_only=*/false), 0);
    EXPECT_LT(problem.largest_residual(eigenvalues, eigenvectors), 1.0e-12);
}

namespace {

constexpr int orthogonalized_plane_waves = 12;
constexpr int orthogonalized_existing = 2;
constexpr int orthogonalized_candidates = 3;
constexpr int orthogonalized_vectors = orthogonalized_existing + orthogonalized_candidates;

/// S = diag(1 + g/10): a general overlap, formed and updated the default way.
class DiagonalOverlapOperator : public OverlapOperator {
  public:
    void apply(int number_of_plane_waves,
               const Complex* wavefunction,
               Complex* overlap_wavefunction) const override {
        for (int plane_wave = 0; plane_wave < number_of_plane_waves; ++plane_wave) {
            overlap_wavefunction[plane_wave] = (1.0 + 0.1 * plane_wave) * wavefunction[plane_wave];
        }
    }
    void build_dense_matrix(int, std::vector<Complex>&) const override {}
};

/// S = 1 with the overlap block the norm-conserving operator hands out.
class UnitOverlapOperator : public OverlapOperator {
  public:
    void apply(int number_of_plane_waves,
               const Complex* wavefunction,
               Complex* overlap_wavefunction) const override {
        ++host_applications;
        for (int plane_wave = 0; plane_wave < number_of_plane_waves; ++plane_wave) {
            overlap_wavefunction[plane_wave] = wavefunction[plane_wave];
        }
    }
    void build_dense_matrix(int, std::vector<Complex>&) const override {}
    std::unique_ptr<GPUOverlapBlock>
    create_gpu_overlap_block(execution::gpu::device::GPUDeviceBackend& device_backend,
                             int number_of_plane_waves,
                             int number_of_vectors,
                             const Complex* vectors,
                             Complex* overlap_vectors) const override {
        return std::make_unique<CopiedGPUOverlapBlock>(
            device_backend,
            static_cast<std::size_t>(number_of_plane_waves) * number_of_vectors,
            vectors,
            overlap_vectors);
    }
    mutable int host_applications = 0;
};

/**
 * Two vectors already S-orthonormal and three candidates, with S·V beside them in a
 * buffer of its own, as the device workspace holds them.
 */
struct OrthogonalizationProblem {
    std::shared_ptr<CPUBackedGPUDeviceBackend> device_backend =
        std::make_shared<CPUBackedGPUDeviceBackend>();
    CPUBackedGPUBLAS device_blas{device_backend};
    execution::SerialExecutionPolicy execution_policy;
    GPUPlanewaveSubspaceManager manager{device_blas, execution_policy, device_backend};
    std::vector<Complex> basis =
        std::vector<Complex>(orthogonalized_plane_waves * orthogonalized_vectors);
    std::vector<Complex> overlap_basis = std::vector<Complex>(
        orthogonalized_plane_waves * orthogonalized_vectors, Complex(99.0, 99.0));

    explicit OrthogonalizationProblem(const OverlapOperator& overlap_operator) {
        for (int vector = 0; vector < orthogonalized_vectors; ++vector) {
            for (int plane_wave = 0; plane_wave < orthogonalized_plane_waves; ++plane_wave) {
                column(basis, vector)[plane_wave] =
                    Complex(std::sin(0.7 * (plane_wave + 1) * (vector + 1)),
                            std::cos(0.3 * (plane_wave + 2) + 1.1 * vector));
            }
        }
        // Gram-Schmidt in the S metric for the vectors that are already in the basis.
        for (int vector = 0; vector < orthogonalized_existing; ++vector) {
            for (int earlier = 0; earlier < vector; ++earlier) {
                Complex projection = product(column(basis, earlier),
                                             column(overlap_basis, vector),
                                             overlap_operator,
                                             column(basis, vector));
                for (int plane_wave = 0; plane_wave < orthogonalized_plane_waves; ++plane_wave) {
                    column(basis, vector)[plane_wave] -=
                        projection * column(basis, earlier)[plane_wave];
                }
            }
            Real norm = std::sqrt(product(column(basis, vector),
                                          column(overlap_basis, vector),
                                          overlap_operator,
                                          column(basis, vector))
                                      .real());
            for (int plane_wave = 0; plane_wave < orthogonalized_plane_waves; ++plane_wave) {
                column(basis, vector)[plane_wave] /= norm;
            }
            overlap_operator.apply(
                orthogonalized_plane_waves, column(basis, vector), column(overlap_basis, vector));
        }
    }

    static Complex* column(std::vector<Complex>& block, int vector) {
        return &block[static_cast<std::size_t>(vector) * orthogonalized_plane_waves];
    }

    /// left† · S · right, with S·right written to `scratch`.
    static Complex product(const Complex* left,
                           Complex* scratch,
                           const OverlapOperator& overlap_operator,
                           const Complex* right) {
        overlap_operator.apply(orthogonalized_plane_waves, right, scratch);
        Complex sum(0.0, 0.0);
        for (int plane_wave = 0; plane_wave < orthogonalized_plane_waves; ++plane_wave) {
            sum += std::conj(left[plane_wave]) * scratch[plane_wave];
        }
        return sum;
    }

    int orthogonalize(const OverlapOperator& overlap_operator) {
        return manager.orthogonalize_batch(orthogonalized_existing,
                                           orthogonalized_candidates,
                                           orthogonalized_plane_waves,
                                           /*is_gamma_only=*/false,
                                           &overlap_operator,
                                           basis.data(),
                                           overlap_basis.data());
    }

    /**
     * Largest deviation of V†(S·V) from the identity and of the S block from S applied to V,
     * over the first `orthonormal_vectors` columns.
     */
    void expect_orthonormal_with_a_consistent_overlap_block(
        const OverlapOperator& overlap_operator, int orthonormal_vectors = orthogonalized_vectors) {
        std::vector<Complex> applied(orthogonalized_plane_waves);
        for (int right = 0; right < orthonormal_vectors; ++right) {
            overlap_operator.apply(
                orthogonalized_plane_waves, column(basis, right), applied.data());
            for (int plane_wave = 0; plane_wave < orthogonalized_plane_waves; ++plane_wave) {
                EXPECT_LT(std::abs(applied[plane_wave] - column(overlap_basis, right)[plane_wave]),
                          1.0e-12);
            }
            for (int left = 0; left < orthonormal_vectors; ++left) {
                Complex sum(0.0, 0.0);
                for (int plane_wave = 0; plane_wave < orthogonalized_plane_waves; ++plane_wave) {
                    sum += std::conj(column(basis, left)[plane_wave]) *
                           column(overlap_basis, right)[plane_wave];
                }
                EXPECT_LT(std::abs(sum - (left == right ? Complex(1.0, 0.0) : Complex(0.0, 0.0))),
                          1.0e-12);
            }
        }
    }
};

} // namespace

TEST(GPUPlanewaveSubspaceManager, OrthogonalizingUnderAGeneralOverlapKeepsItsBlockInStep) {
    DiagonalOverlapOperator overlap_operator;
    OrthogonalizationProblem problem(overlap_operator);

    EXPECT_EQ(problem.orthogonalize(overlap_operator), orthogonalized_candidates);

    problem.expect_orthonormal_with_a_consistent_overlap_block(overlap_operator);
}

TEST(GPUPlanewaveSubspaceManager, OrthogonalizingUnderAUnitOverlapCopiesItsBlockOnTheDevice) {
    UnitOverlapOperator overlap_operator;
    OrthogonalizationProblem problem(overlap_operator);
    overlap_operator.host_applications = 0;

    EXPECT_EQ(problem.orthogonalize(overlap_operator), orthogonalized_candidates);

    // The operator's host loop is never run: the block is copied where it is.
    EXPECT_EQ(overlap_operator.host_applications, 0);
    problem.expect_orthonormal_with_a_consistent_overlap_block(overlap_operator);
    for (std::size_t index = 0; index < problem.basis.size(); ++index) {
        EXPECT_EQ(problem.overlap_basis[index], problem.basis[index]);
    }
}

TEST(GPUPlanewaveSubspaceManager, AZeroCandidateIsDroppedByTheGramSchmidtFallback) {
    DiagonalOverlapOperator overlap_operator;
    OrthogonalizationProblem problem(overlap_operator);
    // The middle candidate is zero, so the Gram matrix is singular and the Cholesky step fails.
    const int zero_candidate = orthogonalized_existing + 1;
    std::fill(problem.column(problem.basis, zero_candidate),
              problem.column(problem.basis, zero_candidate) + orthogonalized_plane_waves,
              Complex(0.0, 0.0));

    EXPECT_EQ(problem.orthogonalize(overlap_operator), orthogonalized_candidates - 1);

    problem.expect_orthonormal_with_a_consistent_overlap_block(overlap_operator,
                                                               orthogonalized_vectors - 1);
}
