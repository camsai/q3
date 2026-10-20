#include "layer_0_core/math/tolerances.hpp"
#include <gtest/gtest.h>

using namespace q3::math::tolerances;

/**
 * ─── Pinned Values ──────────────────────────────────────────────────────────
 * These tests pin every tolerance to its exact value so that accidental
 * changes to tolerances.hpp are immediately caught by the test suite.
 */

TEST(Tolerances, PinnedMathematicalThresholds) {
    EXPECT_DOUBLE_EQ(zero_vector_magnitude_threshold, 1e-10);
    EXPECT_DOUBLE_EQ(numerical_zero_threshold, 1e-12);
    EXPECT_DOUBLE_EQ(minimum_finite_difference_step, 1e-10);
    EXPECT_DOUBLE_EQ(iterative_solver_norm_threshold, 1e-14);
    EXPECT_DOUBLE_EQ(preconditioner_denominator_floor, 1e-4);
    EXPECT_DOUBLE_EQ(gaunt_coefficient_threshold, 1e-6);
    EXPECT_DOUBLE_EQ(relative_change_scale_floor, 1e-30);
}

TEST(Tolerances, PinnedPhysicsThresholds) {
    EXPECT_DOUBLE_EQ(d_matrix_element_threshold, 1e-15);
    EXPECT_DOUBLE_EQ(cell_volume_threshold, 1e-10);
    EXPECT_DOUBLE_EQ(occupation_threshold, 1e-10);
    EXPECT_DOUBLE_EQ(smearing_width_threshold, 1e-10);
    EXPECT_DOUBLE_EQ(energy_cutoff_rounding_tolerance, 1e-8);
    EXPECT_DOUBLE_EQ(default_symmetry_precision, 1e-5);
    EXPECT_DOUBLE_EQ(reciprocal_space_finite_difference_step, 1e-5);
    EXPECT_DOUBLE_EQ(bessel_argument_zero_threshold, 1e-8);
    EXPECT_DOUBLE_EQ(fermi_energy_bisection_convergence, 1e-12);
    EXPECT_DOUBLE_EQ(minimum_electron_density, 1e-12);
    EXPECT_DOUBLE_EQ(ewald_convergence_threshold, 1e-7);
    EXPECT_DOUBLE_EQ(energy_consistency_threshold, 1e-8);
    EXPECT_DOUBLE_EQ(stress_finite_difference_scale, 1e-6);
}

TEST(Tolerances, PinnedWannierConstructionThresholds) {
    EXPECT_DOUBLE_EQ(wannier_projection_singular_value_threshold, 1e-8);
    EXPECT_DOUBLE_EQ(wannier_diagonal_overlap_threshold, 1e-10);
    EXPECT_DOUBLE_EQ(wannier_negative_spread_tolerance, 1e-8);
    EXPECT_DOUBLE_EQ(lattice_shell_relative_width, 1e-6);
}

TEST(Tolerances, PinnedEigenvalueTolerances) {
    EXPECT_DOUBLE_EQ(initial_eigenvalue_tolerance, 1e-2);
    EXPECT_DOUBLE_EQ(minimum_eigenvalue_tolerance, 1e-13);
}

TEST(Tolerances, PinnedRadialIntegrationParameters) {
    EXPECT_EQ(use_radial_integration_truncation, true);
    EXPECT_DOUBLE_EQ(radial_integration_truncation_radius, 10.0);
}

TEST(Tolerances, PinnedQGridParameters) {
    EXPECT_DOUBLE_EQ(default_q_grid_spacing, 0.01);
    EXPECT_EQ(default_number_of_q_points_for_atomic_wavefunctions, 2000);
}

// ─── Positivity ─────────────────────────────────────────────────────────────

TEST(Tolerances, AllNumericalThresholdsArePositive) {
    EXPECT_GT(zero_vector_magnitude_threshold, 0.0);
    EXPECT_GT(numerical_zero_threshold, 0.0);
    EXPECT_GT(minimum_finite_difference_step, 0.0);
    EXPECT_GT(iterative_solver_norm_threshold, 0.0);
    EXPECT_GT(preconditioner_denominator_floor, 0.0);
    EXPECT_GT(gaunt_coefficient_threshold, 0.0);
    EXPECT_GT(d_matrix_element_threshold, 0.0);
    EXPECT_GT(cell_volume_threshold, 0.0);
    EXPECT_GT(occupation_threshold, 0.0);
    EXPECT_GT(smearing_width_threshold, 0.0);
    EXPECT_GT(energy_cutoff_rounding_tolerance, 0.0);
    EXPECT_GT(default_symmetry_precision, 0.0);
    EXPECT_GT(reciprocal_space_finite_difference_step, 0.0);
    EXPECT_GT(bessel_argument_zero_threshold, 0.0);
    EXPECT_GT(fermi_energy_bisection_convergence, 0.0);
    EXPECT_GT(minimum_electron_density, 0.0);
    EXPECT_GT(ewald_convergence_threshold, 0.0);
    EXPECT_GT(energy_consistency_threshold, 0.0);
    EXPECT_GT(stress_finite_difference_scale, 0.0);
}

TEST(Tolerances, AllThresholdsAreLessThanOne) {
    EXPECT_LT(zero_vector_magnitude_threshold, 1.0);
    EXPECT_LT(numerical_zero_threshold, 1.0);
    EXPECT_LT(minimum_finite_difference_step, 1.0);
    EXPECT_LT(iterative_solver_norm_threshold, 1.0);
    EXPECT_LT(preconditioner_denominator_floor, 1.0);
    EXPECT_LT(gaunt_coefficient_threshold, 1.0);
    EXPECT_LT(d_matrix_element_threshold, 1.0);
    EXPECT_LT(cell_volume_threshold, 1.0);
    EXPECT_LT(occupation_threshold, 1.0);
    EXPECT_LT(smearing_width_threshold, 1.0);
    EXPECT_LT(energy_cutoff_rounding_tolerance, 1.0);
    EXPECT_LT(default_symmetry_precision, 1.0);
    EXPECT_LT(reciprocal_space_finite_difference_step, 1.0);
    EXPECT_LT(bessel_argument_zero_threshold, 1.0);
    EXPECT_LT(fermi_energy_bisection_convergence, 1.0);
    EXPECT_LT(minimum_electron_density, 1.0);
    EXPECT_LT(ewald_convergence_threshold, 1.0);
    EXPECT_LT(energy_consistency_threshold, 1.0);
    EXPECT_LT(stress_finite_difference_scale, 1.0);
    EXPECT_LT(initial_eigenvalue_tolerance, 1.0);
    EXPECT_LT(minimum_eigenvalue_tolerance, 1.0);
}

// ─── Ordering Constraints ───────────────────────────────────────────────────

TEST(Tolerances, NumericalZeroThresholdIsStricterThanVectorMagnitudeThreshold) {
    EXPECT_LT(numerical_zero_threshold, zero_vector_magnitude_threshold);
}

TEST(Tolerances, InitialEigenvalueToleranceExceedsMinimumEigenvalueTolerance) {
    EXPECT_GT(initial_eigenvalue_tolerance, minimum_eigenvalue_tolerance);
}

// ─── Physics Constants ──────────────────────────────────────────────────────

TEST(Tolerances, RadialIntegrationTruncationRadiusIsPositive) {
    EXPECT_GT(radial_integration_truncation_radius, 0.0);
}

TEST(Tolerances, DefaultQGridSpacingIsPositive) {
    EXPECT_GT(default_q_grid_spacing, 0.0);
}

TEST(Tolerances, DefaultNumberOfQPointsForAtomicWavefunctionsExceedsOneHundred) {
    EXPECT_GT(default_number_of_q_points_for_atomic_wavefunctions, 100);
}
