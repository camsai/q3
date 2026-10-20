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
#include "layer_4_computation/interpolator/method/wannier/diagnostic.hpp"
#include "layer_4_computation/interpolator/method/wannier/diagnostic_error.hpp"
#include "layer_4_computation/interpolator/method/wannier/inputs.hpp"
#include "layer_4_computation/interpolator/method/wannier/internal/band_classification.hpp"
#include "layer_4_computation/interpolator/method/wannier/settings/init.hpp"
#include "layer_4_computation/interpolator/method/wannier/subspace/projectability_disentanglement_selection.hpp"
#include "utils/wannier_chain_model.hpp"

using namespace q3;
using q3::test_utils::inputs_of;
using q3::test_utils::make_wannier_chain;
using q3::test_utils::WannierChainModel;

namespace {

/**
 * The chain as a record whose every band is a candidate: projectability
 * halfway between the thresholds, and a Fermi level below every eigenvalue
 * so the energy rule freezes nothing. The iteration then chooses N states
 * out of N -- the whole window -- and what it must reproduce is the
 * window's own Omega_I.
 */
WannierInputs candidate_record(const WannierChainModel& model) {
    WannierInputs record = inputs_of(model);
    record.fermi_energy_rydberg = -1.0;
    for (std::size_t kpoint = 0; kpoint < record.kpoint_coordinates.size(); kpoint++) {
        record.projectabilities.emplace_back(
            static_cast<std::size_t>(model.number_of_wannier_functions), 0.5);
    }
    return record;
}

/// Omega_I of the chain in closed form: sum_b w_b N (1 - rho^2) over both directions.
Real chain_gauge_invariant_spread(const WannierChainModel& model, Real diagonal_modulus) {
    Real total = 0.0;
    for (const Real weight : model.weights) {
        total += weight * static_cast<Real>(model.number_of_wannier_functions) *
                 (1.0 - diagonal_modulus * diagonal_modulus);
    }
    return total;
}

} // namespace

TEST(ProjectabilityDisentanglementSelection,
     ChoosingTheWholeWindowReproducesItsGaugeInvariantSpread) {
    const WannierChainModel model = make_wannier_chain(6, {0.0, 2.0}, 5.5, 0.9);
    const WannierInputs record = candidate_record(model);
    const WannierConstructionInputs inputs(record);
    const WannierSubspace subspace =
        ProjectabilityDisentanglementSelection(ProjectabilityDisentanglementSettings{})
            .choose(inputs);

    EXPECT_TRUE(subspace.report.performed);
    EXPECT_FALSE(subspace.report.classification.metallic);
    EXPECT_EQ(subspace.report.classification.number_of_bands_crossing_fermi, 0);
    EXPECT_NEAR(subspace.report.gauge_invariant_spread_bohr_squared,
                chain_gauge_invariant_spread(model, 0.9),
                1.0e-10);
    EXPECT_GT(subspace.report.iterations, 0);

    ASSERT_EQ(subspace.frames.size(), 6u);
    for (const DenseMatrix<Complex>& frame : subspace.frames) {
        ASSERT_EQ(frame.rows(), 2);
        ASSERT_EQ(frame.columns(), 2);
        const DenseMatrix<Complex> gram =
            multiply(MatrixOperation::ConjugateTranspose, frame, MatrixOperation::None, frame);
        for (int row = 0; row < 2; row++) {
            for (int column = 0; column < 2; column++) {
                const Complex expected = row == column ? Complex(1.0, 0.0) : Complex(0.0, 0.0);
                EXPECT_NEAR(std::abs(gram(row, column) - expected), 0.0, 1.0e-10);
            }
        }
    }
    for (const KPointBandClassification& entry : subspace.classification.kpoints) {
        EXPECT_EQ(entry.number_kept(), 2);
        EXPECT_EQ(entry.number_frozen(), 0);
    }
}

/**
 * A band frozen by energy that the trial orbitals cannot reach makes the
 * gauge inside the chosen subspace singular, and the refusal has to name
 * where and which band: by the time the SVD speaks, the k-point and the band
 * are gone. Three bands, two trial orbitals; the third band lies far below
 * the Fermi level, so the energy rule freezes it, and no orbital projects on
 * it, so the frozen direction is one A has no column for.
 */
TEST(ProjectabilityDisentanglementSelection, ASingularGaugeNamesTheKPointAndTheFrozenBand) {
    const WannierChainModel model = make_wannier_chain(6, {0.0, 2.0, 1.0}, 5.5, 0.9);
    WannierInputs record = inputs_of(model);
    record.number_of_trial_orbitals = 2;
    record.fermi_energy_rydberg = -1.0;
    for (std::size_t kpoint = 0; kpoint < record.kpoint_coordinates.size(); kpoint++) {
        record.eigenvalues_rydberg[kpoint] = {1.0, 2.0, -5.0};
        record.projectabilities.push_back({0.5, 0.5, 0.02});
        DenseMatrix<Complex> projection(3, 2, Complex(0.0, 0.0));
        projection(0, 0) = Complex(1.0, 0.0);
        projection(1, 1) = Complex(1.0, 0.0);
        record.projection_matrices[kpoint] = projection.raw();
    }
    const WannierConstructionInputs inputs(record);

    try {
        (void)ProjectabilityDisentanglementSelection(ProjectabilityDisentanglementSettings{})
            .choose(inputs);
        FAIL() << "a frozen band with no projection should have made the gauge singular";
    } catch (const WannierDiagnosticError& error) {
        EXPECT_EQ(error.diagnostic(), WannierDiagnostic::gauge_singular);
        const std::string message = error.what();
        EXPECT_NE(message.find("This is k-point 0 of the disentangled window"), std::string::npos)
            << message;
        EXPECT_NE(message.find("1 of 3 kept bands are frozen"), std::string::npos) << message;
        EXPECT_NE(message.find("band 3 at p = 0.020000"), std::string::npos) << message;
    }
}

TEST(ProjectabilityDisentanglementSelection, RefusesARecordWithoutOverlaps) {
    WannierInputs record = candidate_record(make_wannier_chain(4, {0.0, 1.0}, 5.0, 0.9));
    record.overlap_matrices.clear();
    const WannierConstructionInputs inputs(record);
    try {
        const WannierSubspace subspace =
            ProjectabilityDisentanglementSelection(ProjectabilityDisentanglementSettings{})
                .choose(inputs);
        FAIL() << "the subspace iteration is defined through the overlaps";
    } catch (const WannierDiagnosticError& error) {
        EXPECT_EQ(error.diagnostic(), WannierDiagnostic::disentanglement_unsupported);
        const std::string message = error.what();
        EXPECT_NE(message.find("carries none"), std::string::npos) << message;
        EXPECT_NE(message.find("cannot be disentangled"), std::string::npos) << message;
    }
}

TEST(ProjectabilityDisentanglementSelection, RefusesARecordWithoutProjectabilities) {
    WannierInputs record = candidate_record(make_wannier_chain(4, {0.0, 1.0}, 5.0, 0.9));
    record.projectabilities.clear();
    const WannierConstructionInputs inputs(record);
    try {
        const WannierSubspace subspace =
            ProjectabilityDisentanglementSelection(ProjectabilityDisentanglementSettings{})
                .choose(inputs);
        FAIL() << "the classifier reads projectabilities the record does not carry";
    } catch (const WannierDiagnosticError& error) {
        EXPECT_EQ(error.diagnostic(), WannierDiagnostic::disentanglement_unsupported);
        const std::string message = error.what();
        EXPECT_NE(message.find("projectability p_nk"), std::string::npos) << message;
        EXPECT_NE(message.find("carries none"), std::string::npos) << message;
    }
}

TEST(ProjectabilityDisentanglementSelection, InvertedThresholdsAreRefusedBeforeAnyRecordIsRead) {
    ProjectabilityDisentanglementSettings settings;
    settings.thresholds.projectability_minimum = 0.9;
    settings.thresholds.projectability_maximum = 0.1;
    EXPECT_THROW(ProjectabilityDisentanglementSelection{settings}, std::invalid_argument);
}
