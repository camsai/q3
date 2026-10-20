#include <gtest/gtest.h>

#include <complex>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include "layer_0_core/math/linear_algebra/dense_matrix.hpp"
#include "layer_0_core/math/linear_algebra/dense_matrix_product.hpp"
#include "layer_0_core/math/linear_algebra/matrix_operation.hpp"
#include "layer_0_core/types/init.hpp"
#include "layer_4_computation/interpolator/method/wannier/construction_inputs.hpp"
#include "layer_4_computation/interpolator/method/wannier/inputs.hpp"
#include "layer_4_computation/interpolator/method/wannier/internal/projection_gauge.hpp"
#include "layer_4_computation/interpolator/method/wannier/subspace/isolated_manifold_selection.hpp"

using namespace q3;

namespace {

/// `number_of_bands` bands for two Wannier functions on a two-point line, the same A at both k.
WannierInputs two_orbital_record(int number_of_bands, const std::vector<Complex>& projection) {
    WannierInputs record;
    record.kpoint_grid_dimensions = {2, 1, 1};
    record.number_of_bands = number_of_bands;
    record.number_of_trial_orbitals = 2;
    for (int kpoint = 0; kpoint < 2; kpoint++) {
        record.kpoint_coordinates.emplace_back(0.5 * kpoint, 0.0, 0.0);
        std::vector<Real> eigenvalues;
        for (int band = 0; band < number_of_bands; band++) {
            eigenvalues.push_back(0.1 * band + kpoint);
        }
        record.eigenvalues_rydberg.push_back(eigenvalues);
        record.projection_matrices.push_back(projection);
    }
    return record;
}

const std::vector<Complex> skewed_projection = {
    Complex(1.0, 0.0), Complex(0.2, -0.1), Complex(0.0, 0.3), Complex(0.8, 0.0)};

} // namespace

TEST(IsolatedManifoldSelection, TheFrameIsTheProjectionGaugeAndEveryBandIsKept) {
    const WannierInputs record = two_orbital_record(2, skewed_projection);
    const WannierConstructionInputs inputs(record);
    const WannierSubspace subspace = IsolatedManifoldSelection().choose(inputs);

    ASSERT_EQ(subspace.frames.size(), 2u);
    ASSERT_EQ(subspace.classification.kpoints.size(), 2u);
    const DenseMatrix<Complex> expected =
        ProjectionGauge().compute(DenseMatrix<Complex>(2, 2, skewed_projection));
    for (std::size_t kpoint = 0; kpoint < 2; kpoint++) {
        EXPECT_EQ(subspace.frames[kpoint], expected);
        EXPECT_EQ(subspace.classification.kpoints[kpoint].kept_bands, (std::vector<int>{0, 1}));
        EXPECT_EQ(subspace.classification.kpoints[kpoint].number_frozen(), 0);
    }
    const DenseMatrix<Complex> gram = multiply(MatrixOperation::ConjugateTranspose,
                                               subspace.frames[0],
                                               MatrixOperation::None,
                                               subspace.frames[0]);
    EXPECT_NEAR(std::abs(gram(0, 0) - Complex(1.0, 0.0)), 0.0, 1.0e-12);
    EXPECT_NEAR(std::abs(gram(1, 1) - Complex(1.0, 0.0)), 0.0, 1.0e-12);
    EXPECT_NEAR(std::abs(gram(0, 1)), 0.0, 1.0e-12);
    EXPECT_FALSE(subspace.report.performed);
}

/// Three bands for two functions is not an isolated manifold, and both ways out are named.
TEST(IsolatedManifoldSelection, RefusesAWindowWiderThanTheWannierFunctions) {
    const std::vector<Complex> three_by_two = {Complex(1.0, 0.0),
                                               Complex(0.0, 0.0),
                                               Complex(0.0, 0.0),
                                               Complex(0.0, 0.0),
                                               Complex(1.0, 0.0),
                                               Complex(0.0, 0.0)};
    const WannierInputs record = two_orbital_record(3, three_by_two);
    const WannierConstructionInputs inputs(record);
    try {
        const WannierSubspace subspace = IsolatedManifoldSelection().choose(inputs);
        FAIL() << "a window wider than the Wannier functions has no projection gauge of its own";
    } catch (const std::invalid_argument& error) {
        const std::string message = error.what();
        EXPECT_NE(message.find("3 bands and 2 Wannier functions"), std::string::npos) << message;
        EXPECT_NE(message.find("disentanglement: projectability"), std::string::npos) << message;
    }
}
