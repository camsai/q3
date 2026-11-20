#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <memory>
#include <vector>

#include "layer_0_core/execution/serial.hpp"
#include "layer_4_computation/diagonalizer/planewave/internal/overlap/init.hpp"
#include "layer_4_computation/diagonalizer/planewave/internal/subspace_manager/factory_gpu.hpp"
#include "layer_4_computation/diagonalizer/planewave/internal/subspace_manager/gpu.hpp"
#include "layer_4_computation/diagonalizer/planewave/internal/subspace_manager/gpu_gamma_point.hpp"
#include "utils/gpu_blas_cpu_backed.hpp"
#include "utils/gpu_device_backend_cpu_backed.hpp"

using namespace q3;
using test_support::CPUBackedGPUBLAS;
using test_support::CPUBackedGPUDeviceBackend;

namespace {

constexpr int number_of_plane_waves = 12;
constexpr int number_of_existing = 2;
constexpr int number_of_candidates = 3;
constexpr int number_of_vectors = number_of_existing + number_of_candidates;

/// S = 1, formed and updated the default way.
class UnitOverlapOperator : public OverlapOperator {
  public:
    void apply(int plane_waves,
               const Complex* wavefunction,
               Complex* overlap_wavefunction) const override {
        std::copy(wavefunction, wavefunction + plane_waves, overlap_wavefunction);
    }
    void build_dense_matrix(int, std::vector<Complex>&) const override {}
};

/**
 * Two vectors already orthonormal under the gamma-point inner product and three
 * candidates, all with a real G = 0 coefficient, with S·V beside them in a buffer of its
 * own, as the device workspace holds them.
 */
struct GammaPointOrthogonalizationProblem {
    std::shared_ptr<CPUBackedGPUDeviceBackend> device_backend =
        std::make_shared<CPUBackedGPUDeviceBackend>();
    CPUBackedGPUBLAS device_blas{device_backend};
    execution::SerialExecutionPolicy execution_policy;
    GammaPointGPUPlanewaveSubspaceManager manager{device_blas, execution_policy, device_backend};
    UnitOverlapOperator overlap_operator;
    bool origin_is_local;
    std::vector<Complex> basis = std::vector<Complex>(number_of_plane_waves * number_of_vectors);
    std::vector<Complex> overlap_basis =
        std::vector<Complex>(number_of_plane_waves * number_of_vectors, Complex(99.0, 99.0));

    explicit GammaPointOrthogonalizationProblem(bool origin_is_local)
        : origin_is_local{origin_is_local} {
        manager.set_gamma_origin_is_local(origin_is_local);
        for (int vector = 0; vector < number_of_vectors; ++vector) {
            for (int plane_wave = 0; plane_wave < number_of_plane_waves; ++plane_wave) {
                column(basis, vector)[plane_wave] = Complex(
                    std::sin(0.7 * (plane_wave + 1) * (vector + 1)),
                    plane_wave == 0 ? 0.0 : std::cos(0.3 * (plane_wave + 2) + 1.1 * vector));
            }
        }
        // Gram-Schmidt under the gamma-point inner product for the vectors already in the basis.
        for (int vector = 0; vector < number_of_existing; ++vector) {
            for (int earlier = 0; earlier < vector; ++earlier) {
                Real projection = inner_product(column(basis, earlier), column(basis, vector));
                for (int plane_wave = 0; plane_wave < number_of_plane_waves; ++plane_wave) {
                    column(basis, vector)[plane_wave] -=
                        projection * column(basis, earlier)[plane_wave];
                }
            }
            Real norm = std::sqrt(inner_product(column(basis, vector), column(basis, vector)));
            for (int plane_wave = 0; plane_wave < number_of_plane_waves; ++plane_wave) {
                column(basis, vector)[plane_wave] /= norm;
            }
            std::copy(column(basis, vector),
                      column(basis, vector) + number_of_plane_waves,
                      column(overlap_basis, vector));
        }
    }

    static Complex* column(std::vector<Complex>& block, int vector) {
        return &block[static_cast<std::size_t>(vector) * number_of_plane_waves];
    }

    /// 2·Re(left†·right) over the stored half sphere, less the G = 0 term where it is held.
    Real inner_product(const Complex* left, const Complex* right) const {
        Real sum = 0.0;
        for (int plane_wave = 0; plane_wave < number_of_plane_waves; ++plane_wave) {
            sum += 2.0 * (std::conj(left[plane_wave]) * right[plane_wave]).real();
        }
        if (origin_is_local) {
            sum -= (std::conj(left[0]) * right[0]).real();
        }
        return sum;
    }

    int orthogonalize() {
        return manager.orthogonalize_batch(number_of_existing,
                                           number_of_candidates,
                                           number_of_plane_waves,
                                           /*is_gamma_only=*/true,
                                           &overlap_operator,
                                           basis.data(),
                                           overlap_basis.data());
    }

    /**
     * The first `orthonormal_vectors` columns are orthonormal under the gamma-point inner
     * product, their S block is the vectors themselves, and their G = 0 coefficients are real.
     */
    void expect_orthonormal_with_a_consistent_overlap_block(int orthonormal_vectors) {
        for (int right = 0; right < orthonormal_vectors; ++right) {
            EXPECT_EQ(column(basis, right)[0].imag(), 0.0);
            for (int plane_wave = 0; plane_wave < number_of_plane_waves; ++plane_wave) {
                EXPECT_LT(std::abs(column(basis, right)[plane_wave] -
                                   column(overlap_basis, right)[plane_wave]),
                          1.0e-12);
            }
            for (int left = 0; left < orthonormal_vectors; ++left) {
                EXPECT_NEAR(inner_product(column(basis, left), column(overlap_basis, right)),
                            left == right ? 1.0 : 0.0,
                            1.0e-12)
                    << left << " " << right;
            }
        }
    }
};

} // namespace

TEST(GammaPointGPUPlanewaveSubspaceManager, OrthogonalizesWhereTheOriginIsLocal) {
    GammaPointOrthogonalizationProblem problem(/*origin_is_local=*/true);

    EXPECT_EQ(problem.orthogonalize(), number_of_candidates);

    problem.expect_orthonormal_with_a_consistent_overlap_block(number_of_vectors);
}

TEST(GammaPointGPUPlanewaveSubspaceManager, OrthogonalizesWhereAnotherRankHoldsTheOrigin) {
    GammaPointOrthogonalizationProblem problem(/*origin_is_local=*/false);

    EXPECT_EQ(problem.orthogonalize(), number_of_candidates);

    problem.expect_orthonormal_with_a_consistent_overlap_block(number_of_vectors);
}

TEST(GammaPointGPUPlanewaveSubspaceManager, AZeroCandidateIsDroppedByTheGramSchmidtFallback) {
    GammaPointOrthogonalizationProblem problem(/*origin_is_local=*/true);
    // The middle candidate is zero, so the Gram matrix is singular and the Cholesky step fails.
    const int zero_candidate = number_of_existing + 1;
    std::fill(problem.column(problem.basis, zero_candidate),
              problem.column(problem.basis, zero_candidate) + number_of_plane_waves,
              Complex(0.0, 0.0));

    EXPECT_EQ(problem.orthogonalize(), number_of_candidates - 1);

    problem.expect_orthonormal_with_a_consistent_overlap_block(number_of_vectors - 1);
}

TEST(GammaPointGPUPlanewaveSubspaceManager, TheFactoryChoosesTheManagerOfTheRepresentation) {
    std::shared_ptr<CPUBackedGPUDeviceBackend> device_backend =
        std::make_shared<CPUBackedGPUDeviceBackend>();
    CPUBackedGPUBLAS device_blas{device_backend};
    execution::SerialExecutionPolicy execution_policy;

    std::unique_ptr<GPUPlanewaveSubspaceManager> gamma_point_manager =
        create_gpu_planewave_subspace_manager(
            /*is_gamma_only=*/true, device_blas, execution_policy, device_backend);
    std::unique_ptr<GPUPlanewaveSubspaceManager> complex_manager =
        create_gpu_planewave_subspace_manager(
            /*is_gamma_only=*/false, device_blas, execution_policy, device_backend);

    ASSERT_NE(gamma_point_manager, nullptr);
    ASSERT_NE(complex_manager, nullptr);
    EXPECT_NE(dynamic_cast<GammaPointGPUPlanewaveSubspaceManager*>(gamma_point_manager.get()),
              nullptr);
    EXPECT_EQ(dynamic_cast<GammaPointGPUPlanewaveSubspaceManager*>(complex_manager.get()), nullptr);
}
