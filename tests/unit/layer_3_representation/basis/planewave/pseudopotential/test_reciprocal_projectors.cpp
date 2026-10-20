/**
 * Q-space projector ingestion tests.
 *
 * Q3 applies projectors as β(q). Datasets that state β(q) directly are
 * resampled onto Q3's q-grid rather than inverted to r and
 * transformed back, and these tests pin both halves of that: the resampler
 * itself, and that BetaFunctionTable actually takes the q-space path when the
 * dataset offers one.
 */

#include "io/pseudopotential/factory.hpp"
#include "io/pseudopotential/upf_reader.hpp"
#include "layer_3_representation/basis/planewave/pseudopotential/internal/beta_function_table/factory.hpp"
#include "layer_3_representation/basis/planewave/pseudopotential/internal/beta_function_table/init.hpp"
#include "layer_3_representation/basis/planewave/pseudopotential/internal/beta_function_table/paw.hpp"
#include "layer_3_representation/basis/planewave/pseudopotential/internal/reciprocal_projector_resampling.hpp"
#include "layer_3_representation/pseudopotential/paw/init.hpp"
#include "layer_3_representation/pseudopotential/set.hpp"
#include <gtest/gtest.h>

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

using namespace q3;

/**
 * Expose the protected per-projector tabulation so the tests can probe it at
 * chosen q values without going through a full basis initialization.
 */
struct ExposedBaseTable : BetaFunctionTable {
    using BetaFunctionTable::tabulate_projector;
};

TEST(ReciprocalResamplingTest, ReproducesSamplesExactlyAtGridPoints) {
    /**
     * An interpolant that does not pass through its own samples would shift
     * every projector slightly — a small, systematic error in the nonlocal
     * energy that no test of the SCF alone would localize.
     */
    std::vector<Real> samples;
    const Real spacing = 0.25;
    for (int index = 0; index < 40; index++) {
        Real q_value = index * spacing;
        samples.push_back(std::exp(-0.3 * q_value) * std::cos(q_value));
    }

    std::vector<Real> q_values;
    for (int index = 0; index < 40; index++)
        q_values.push_back(index * spacing);

    std::vector<Real> result;
    resample_uniform_reciprocal_table(samples, spacing, 0, q_values, result);

    ASSERT_EQ(result.size(), samples.size());
    for (size_t index = 0; index < samples.size(); index++)
        EXPECT_NEAR(result[index], samples[index], 1e-12) << "at grid point " << index;
}

TEST(ReciprocalResamplingTest, ConvergesAtFourthOrderBetweenGridPoints) {
    /**
     * Between samples the interpolant is approximate, so the meaningful claim
     * is its order rather than any single tolerance: a cubic through four
     * points has O(h⁴) error, and halving the spacing should cut the error by
     * roughly sixteen. Asserting the order catches a broken interpolant that a
     * loose fixed threshold would let through.
     */
    auto reference = [](Real q_value) { return std::exp(-0.5 * q_value * q_value); };
    const std::vector<Real> q_values = {0.37, 1.23, 2.06};

    auto worst_error_at_spacing = [&](Real spacing) {
        std::vector<Real> samples;
        for (int index = 0; index * spacing < 10.0; index++)
            samples.push_back(reference(index * spacing));
        std::vector<Real> result;
        resample_uniform_reciprocal_table(samples, spacing, 0, q_values, result);
        Real worst = 0.0;
        for (size_t index = 0; index < q_values.size(); index++)
            worst = std::max(worst, std::abs(result[index] - reference(q_values[index])));
        return worst;
    };

    Real coarse = worst_error_at_spacing(0.1);
    Real fine = worst_error_at_spacing(0.05);
    EXPECT_LT(coarse, 1e-4);
    EXPECT_GT(coarse / fine, 8.0) << "coarse " << coarse << " fine " << fine;
}

TEST(ReciprocalResamplingTest, ParityMirrorSharpensTheFirstInterval) {
    /**
     * The first interval needs a sample below q = 0, supplied by parity rather
     * than by clamping.
     * For an even-l projector, an even reference function must then be
     * interpolated there as accurately as anywhere else -- clamping instead
     * costs roughly an order of magnitude exactly where projectors carry the
     * most weight.
     */
    auto even_reference = [](Real q_value) { return std::exp(-0.5 * q_value * q_value); };
    const Real spacing = 0.1;
    std::vector<Real> samples;
    for (int index = 0; index < 100; index++)
        samples.push_back(even_reference(index * spacing));

    std::vector<Real> result;
    resample_uniform_reciprocal_table(samples, spacing, 0, {0.05}, result);
    Real first_interval_error = std::abs(result[0] - even_reference(0.05));

    resample_uniform_reciprocal_table(samples, spacing, 0, {1.25}, result);
    Real interior_error = std::abs(result[0] - even_reference(1.25));

    EXPECT_LT(first_interval_error, 10.0 * std::max(interior_error, 1e-9))
        << "first interval " << first_interval_error << " vs interior " << interior_error;
}

TEST(ReciprocalResamplingTest, IsZeroBeyondTheTabulatedRange) {
    /**
     * Past the dataset's own q_max the projector is unknown, not small.
     * Extrapolating would invent structure the dataset never stated, so the
     * table reads zero and a basis that reaches there is a configuration error
     * rather than a silent approximation.
     */
    std::vector<Real> samples(20, 1.0);
    const Real spacing = 0.5;
    std::vector<Real> q_values = {0.0, 9.5, 9.75, 12.0, 100.0};
    std::vector<Real> result;
    resample_uniform_reciprocal_table(samples, spacing, 0, q_values, result);

    EXPECT_NEAR(result[0], 1.0, 1e-12);
    EXPECT_NEAR(result[1], 1.0, 1e-12); // last sample
    EXPECT_EQ(result[3], 0.0);
    EXPECT_EQ(result[4], 0.0);
}

TEST(ReciprocalResamplingTest, EmptyOrDegenerateTableYieldsZeros) {
    std::vector<Real> result;
    resample_uniform_reciprocal_table({}, 0.5, 0, {0.0, 1.0}, result);
    ASSERT_EQ(result.size(), 2u);
    EXPECT_EQ(result[0], 0.0);

    resample_uniform_reciprocal_table({1.0, 2.0}, 0.0, 0, {0.0, 1.0}, result);
    EXPECT_EQ(result[1], 0.0);
}

TEST(BetaFunctionTableTest, UpfDatasetStillTakesTheRadialPath) {
    /**
     * The branch is on what the dataset carries, not on its format, so the
     * existing UPF path has to be untouched — including for UPF *PAW*
     * datasets, which are PawPseudopotential but supply projectors radially.
     */
    auto pseudopotential =
        UPFReader::read(std::string(EXAMPLES_DIR) + "/data/pseudopotentials/paw/Si.upf");
    const auto& paw = dynamic_cast<const PawPseudopotential&>(*pseudopotential);
    ASSERT_FALSE(paw.has_reciprocal_projectors());
    ASSERT_FALSE(paw.projectors[0].radial_values.empty());

    std::vector<Real> q_values = {0.0, 0.5, 1.0, 2.0};
    std::vector<Real> tabulated;
    ExposedBaseTable table;
    table.tabulate_projector(paw, 0, q_values, tabulated);

    ASSERT_EQ(tabulated.size(), q_values.size());
    Real largest = 0.0;
    for (Real value : tabulated)
        largest = std::max(largest, std::abs(value));
    EXPECT_GT(largest, 1e-6);
}
