#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <vector>

#include "layer_3_representation/pseudopotential/planewave/internal/lowdin_orthogonalization.hpp"
#include "layer_4_computation/diagonalizer/planewave/internal/overlap/init.hpp"
#include "layer_4_computation/diagonalizer/planewave/internal/overlap/orbital_metric.hpp"

using namespace q3;

namespace {

constexpr int number_of_plane_waves = 6;
constexpr int number_of_orbitals = 3;

/**
 * S = I + strength |beta><beta|, the shape an ultrasoft/PAW augmentation
 * overlap takes with a single projector (Vanderbilt, PRB 41, 7892 (1990)),
 * with an explicit beta so the expected metric can be written out by hand.
 * A positive `strength` keeps S positive definite, which Löwdin's S^{-1/2}
 * requires.
 */
class SingleProjectorOverlapOperator : public OverlapOperator {
  public:
    SingleProjectorOverlapOperator(std::vector<Complex> projector, Real strength)
        : projector_(std::move(projector)), strength_(strength) {}

    void apply(int plane_wave_count,
               const Complex* wavefunction,
               Complex* overlap_wavefunction) const override {
        Complex projector_overlap(0.0, 0.0);
        for (int g_index = 0; g_index < plane_wave_count; g_index++) {
            projector_overlap += std::conj(projector_[g_index]) * wavefunction[g_index];
        }
        for (int g_index = 0; g_index < plane_wave_count; g_index++) {
            overlap_wavefunction[g_index] =
                wavefunction[g_index] + strength_ * projector_[g_index] * projector_overlap;
        }
    }

    void build_dense_matrix(int plane_wave_count,
                            std::vector<Complex>& overlap_matrix) const override {
        overlap_matrix.assign(static_cast<std::size_t>(plane_wave_count) * plane_wave_count,
                              Complex(0.0, 0.0));
        for (int column = 0; column < plane_wave_count; column++) {
            for (int row = 0; row < plane_wave_count; row++) {
                Complex value = (row == column) ? Complex(1.0, 0.0) : Complex(0.0, 0.0);
                value += strength_ * projector_[row] * std::conj(projector_[column]);
                overlap_matrix[static_cast<std::size_t>(row) + column * plane_wave_count] = value;
            }
        }
    }

    [[nodiscard]] const std::vector<Complex>& projector() const {
        return projector_;
    }

    [[nodiscard]] Real strength() const {
        return strength_;
    }

  private:
    std::vector<Complex> projector_;
    Real strength_;
};

/// S = I written out as an operator, to separate "no operator" from "S is the identity".
class IdentityOverlapOperator : public OverlapOperator {
  public:
    void apply(int plane_wave_count,
               const Complex* wavefunction,
               Complex* overlap_wavefunction) const override {
        for (int g_index = 0; g_index < plane_wave_count; g_index++) {
            overlap_wavefunction[g_index] = wavefunction[g_index];
        }
    }

    [[nodiscard]] bool is_identity() const override {
        return true;
    }

    void build_dense_matrix(int plane_wave_count,
                            std::vector<Complex>& overlap_matrix) const override {
        overlap_matrix.assign(static_cast<std::size_t>(plane_wave_count) * plane_wave_count,
                              Complex(0.0, 0.0));
        for (int index = 0; index < plane_wave_count; index++) {
            overlap_matrix[static_cast<std::size_t>(index) + index * plane_wave_count] =
                Complex(1.0, 0.0);
        }
    }
};

/// Deterministic, linearly independent, deliberately non-orthogonal orbitals.
std::vector<std::vector<Complex>> make_orbitals() {
    std::vector<std::vector<Complex>> orbitals(number_of_orbitals,
                                               std::vector<Complex>(number_of_plane_waves));
    for (int orbital_index = 0; orbital_index < number_of_orbitals; orbital_index++) {
        for (int g_index = 0; g_index < number_of_plane_waves; g_index++) {
            Real real_part = 1.0 / (1.0 + orbital_index + g_index);
            Real imaginary_part = 0.1 * (orbital_index + 1) * (g_index % 3) - 0.05 * g_index;
            orbitals[orbital_index][g_index] = Complex(real_part, imaginary_part);
        }
    }
    return orbitals;
}

std::vector<Complex> make_projector() {
    std::vector<Complex> projector(number_of_plane_waves);
    for (int g_index = 0; g_index < number_of_plane_waves; g_index++) {
        projector[g_index] = Complex(0.3 - 0.07 * g_index, 0.11 * g_index - 0.2);
    }
    return projector;
}

/// <left|S|right>, evaluated through the operator itself.
Complex overlap_in_metric(const std::vector<Complex>& left,
                          const std::vector<Complex>& right,
                          const OverlapOperator& overlap_operator) {
    std::vector<Complex> metric_applied(number_of_plane_waves);
    overlap_operator.apply(number_of_plane_waves, right.data(), metric_applied.data());
    Complex value(0.0, 0.0);
    for (int g_index = 0; g_index < number_of_plane_waves; g_index++) {
        value += std::conj(left[g_index]) * metric_applied[g_index];
    }
    return value;
}

} // namespace

// ── S = I: the metric-aware path must reproduce the plain one exactly ──

TEST(OrbitalMetric, NullOperatorLeavesOrbitalsUntouched) {
    auto orbitals = make_orbitals();
    auto metric_applied = apply_overlap_operator_to_orbitals(orbitals, nullptr);
    ASSERT_EQ(metric_applied.size(), orbitals.size());
    for (int orbital_index = 0; orbital_index < number_of_orbitals; orbital_index++) {
        for (int g_index = 0; g_index < number_of_plane_waves; g_index++) {
            EXPECT_EQ(metric_applied[orbital_index][g_index], orbitals[orbital_index][g_index]);
        }
    }
}

/**
 * Bit for bit, not approximately: this identity is what keeps the
 * norm-conserving projections unchanged by the move to the S metric, and an
 * agreement that only held to a tolerance would not be evidence of that.
 */
TEST(OrbitalMetric, NullOperatorReproducesThePlainOverlapMatrixExactly) {
    auto orbitals = make_orbitals();
    auto plain = compute_orbital_overlap_matrix(orbitals);
    auto in_s_metric = compute_orbital_overlap_matrix_in_s_metric(orbitals, nullptr);
    ASSERT_EQ(in_s_metric.size(), plain.size());
    for (std::size_t index = 0; index < plain.size(); index++) {
        EXPECT_EQ(in_s_metric[index], plain[index]) << "element " << index;
    }
}

TEST(OrbitalMetric, ExplicitIdentityOperatorReproducesThePlainOverlapMatrixExactly) {
    auto orbitals = make_orbitals();
    IdentityOverlapOperator identity;
    auto plain = compute_orbital_overlap_matrix(orbitals);
    auto in_s_metric = compute_orbital_overlap_matrix_in_s_metric(orbitals, &identity);
    ASSERT_EQ(in_s_metric.size(), plain.size());
    for (std::size_t index = 0; index < plain.size(); index++) {
        EXPECT_EQ(in_s_metric[index], plain[index]) << "element " << index;
    }
}

// ── S != I: the augmentation term actually reaches the Gram matrix ─────

TEST(OrbitalMetric, AugmentedOperatorAddsTheProjectorTermToTheOverlapMatrix) {
    auto orbitals = make_orbitals();
    SingleProjectorOverlapOperator overlap_operator(make_projector(), 0.75);

    auto in_s_metric = compute_orbital_overlap_matrix_in_s_metric(orbitals, &overlap_operator);
    auto plain = compute_orbital_overlap_matrix(orbitals);

    for (int column = 0; column < number_of_orbitals; column++) {
        for (int row = 0; row < number_of_orbitals; row++) {
            Complex projector_on_row(0.0, 0.0);
            Complex projector_on_column(0.0, 0.0);
            for (int g_index = 0; g_index < number_of_plane_waves; g_index++) {
                projector_on_row +=
                    std::conj(overlap_operator.projector()[g_index]) * orbitals[row][g_index];
                projector_on_column +=
                    std::conj(overlap_operator.projector()[g_index]) * orbitals[column][g_index];
            }
            std::size_t index = static_cast<std::size_t>(row) + column * number_of_orbitals;
            Complex expected = plain[index] + overlap_operator.strength() *
                                                  std::conj(projector_on_row) * projector_on_column;
            EXPECT_NEAR(in_s_metric[index].real(), expected.real(), 1e-12);
            EXPECT_NEAR(in_s_metric[index].imag(), expected.imag(), 1e-12);
        }
    }
}

/**
 * The property the whole change exists for: Löwdin orthogonalization run in
 * the S metric produces orbitals that are orthonormal *in that metric*,
 * <phi'_i|S|phi'_j> = delta_ij. The second half of the test is the failure it
 * rules out -- the same orbitals orthogonalized in the plain metric are not.
 */
TEST(OrbitalMetric, LowdinInTheSMetricGivesSOrthonormalOrbitals) {
    SingleProjectorOverlapOperator overlap_operator(make_projector(), 0.75);

    auto orbitals = make_orbitals();
    lowdin_orthogonalize_orbitals(
        orbitals, compute_orbital_overlap_matrix_in_s_metric(orbitals, &overlap_operator));
    for (int row = 0; row < number_of_orbitals; row++) {
        for (int column = 0; column < number_of_orbitals; column++) {
            Complex value = overlap_in_metric(orbitals[row], orbitals[column], overlap_operator);
            Real expected = (row == column) ? 1.0 : 0.0;
            EXPECT_NEAR(value.real(), expected, 1e-12) << "row " << row << " column " << column;
            EXPECT_NEAR(value.imag(), 0.0, 1e-12) << "row " << row << " column " << column;
        }
    }

    auto wrong_metric_orbitals = make_orbitals();
    lowdin_orthogonalize_orbitals(wrong_metric_orbitals);
    Real largest_deviation = 0.0;
    for (int row = 0; row < number_of_orbitals; row++) {
        for (int column = 0; column < number_of_orbitals; column++) {
            Complex value = overlap_in_metric(
                wrong_metric_orbitals[row], wrong_metric_orbitals[column], overlap_operator);
            Real expected = (row == column) ? 1.0 : 0.0;
            largest_deviation =
                std::max(largest_deviation, std::abs(value - Complex(expected, 0.0)));
        }
    }
    EXPECT_GT(largest_deviation, 1e-3)
        << "the plain-metric orthogonalization happened to be S-orthonormal too, so this "
           "fixture no longer distinguishes the two metrics";
}
