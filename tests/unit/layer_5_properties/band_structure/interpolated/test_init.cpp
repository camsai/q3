#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "layer_4_computation/interpolator/target/quantum_state/local_orbital.hpp"
#include "layer_5_properties/band_structure/interpolated/init.hpp"
#include "utils/cubic_tight_binding.hpp"

using namespace q3;

using namespace q3::test_support;

TEST(InterpolatedBandStructureCalculator, FollowsTheAnalyticDispersionAlongAPath) {
    const Real onsite = -0.25;
    const Real hopping = 0.1;
    LocalOrbitalQuantumStateInterpolator interpolator =
        build_cubic_interpolator(3, onsite, hopping);
    InterpolatedBandStructureCalculator calculator(interpolator);

    std::vector<Vector3D> path;
    for (int step = 0; step <= 20; step++) {
        path.emplace_back(0.5 * step / 20.0, 0.0, 0.0);
    }
    InterpolatedBandStructure bands = calculator.compute(path);

    ASSERT_EQ(bands.eigenvalues_rydberg.size(), path.size());
    ASSERT_EQ(bands.kpoint_coordinates.size(), path.size());
    for (std::size_t index = 0; index < path.size(); index++) {
        ASSERT_EQ(bands.eigenvalues_rydberg[index].size(), 1u);
        EXPECT_NEAR(bands.eigenvalues_rydberg[index][0],
                    analytic_cubic_band(path[index], onsite, hopping),
                    1e-10)
            << "path point " << index;
    }
}

TEST(InterpolatedBandStructureCalculator, HandlesAnEmptyPath) {
    LocalOrbitalQuantumStateInterpolator interpolator = build_cubic_interpolator(2, 0.0, 0.1);
    InterpolatedBandStructureCalculator calculator(interpolator);
    InterpolatedBandStructure bands = calculator.compute({});
    EXPECT_TRUE(bands.eigenvalues_rydberg.empty());
    EXPECT_TRUE(bands.kpoint_coordinates.empty());
}
