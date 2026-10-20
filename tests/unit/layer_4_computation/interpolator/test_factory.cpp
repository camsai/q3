#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <optional>
#include <stdexcept>
#include <vector>

#include "layer_0_core/types/constants.hpp"
#include "layer_4_computation/interpolator/construction.hpp"
#include "layer_4_computation/interpolator/factory.hpp"
#include "layer_4_computation/interpolator/input_data/record.hpp"

using namespace q3;

namespace {

Real analytic_cubic_band(const Vector3D& kpoint, Real onsite, Real hopping) {
    return onsite +
           2.0 * hopping *
               (std::cos(constants::two_pi * kpoint.x) + std::cos(constants::two_pi * kpoint.y) +
                std::cos(constants::two_pi * kpoint.z));
}

/**
 * A two-band Wannier record on a coarse cubic mesh with the identity as the
 * projection at every k-point, so the projection gauge is the identity and
 * the record's eigenvalues must come back unchanged at the mesh points.
 */
InterpolationRecord two_band_wannier_record(int mesh) {
    InterpolationRecord record;
    record.method = "wannier";
    record.real_lattice_bohr[0] = Vector3D(10.2, 0.0, 0.0);
    record.real_lattice_bohr[1] = Vector3D(0.0, 10.2, 0.0);
    record.real_lattice_bohr[2] = Vector3D(0.0, 0.0, 10.2);
    const Real inverse = constants::two_pi / 10.2;
    record.reciprocal_lattice_inverse_bohr[0] = Vector3D(inverse, 0.0, 0.0);
    record.reciprocal_lattice_inverse_bohr[1] = Vector3D(0.0, inverse, 0.0);
    record.reciprocal_lattice_inverse_bohr[2] = Vector3D(0.0, 0.0, inverse);

    WannierInputs& wannier = record.wannier;
    wannier.kpoint_grid_dimensions = {mesh, mesh, mesh};
    wannier.number_of_bands = 2;
    wannier.number_of_trial_orbitals = 2;
    for (int x = 0; x < mesh; x++) {
        for (int y = 0; y < mesh; y++) {
            for (int z = 0; z < mesh; z++) {
                Vector3D kpoint(static_cast<Real>(x) / mesh,
                                static_cast<Real>(y) / mesh,
                                static_cast<Real>(z) / mesh);
                wannier.kpoint_coordinates.push_back(kpoint);
                wannier.eigenvalues_rydberg.push_back({analytic_cubic_band(kpoint, -1.0, 0.3),
                                                       analytic_cubic_band(kpoint, 2.0, -0.15)});
                wannier.projection_matrices.push_back(
                    {Complex(1.0, 0.0), Complex(0.0, 0.0), Complex(0.0, 0.0), Complex(1.0, 0.0)});
            }
        }
    }
    return record;
}

} // namespace

TEST(QuantumStateInterpolatorFactory, RefusesAnUnknownMethod) {
    InterpolationRecord record;
    record.method = "fourier";
    EXPECT_THROW((void)create_quantum_state_interpolator(record), std::runtime_error);
}

TEST(QuantumStateInterpolatorFactory, BuildsTheWannierInterpolatorFromTheRecordAlone) {
    /**
     * The record names its method and carries its lattice, so nothing but
     * the record is needed -- the shape the evaluation step relies on, since
     * it has a file and no crystal.
     */
    const int mesh = 3;
    InterpolationRecord record = two_band_wannier_record(mesh);
    QuantumStateInterpolatorConstruction construction = create_quantum_state_interpolator(record);
    const std::unique_ptr<QuantumStateInterpolator>& interpolator = construction.interpolator;
    ASSERT_NE(interpolator, nullptr);
    EXPECT_EQ(interpolator->number_of_bands(), 2);
    EXPECT_EQ(interpolator->description().find("wannier (projection gauge)"), 0u)
        << interpolator->description();

    const WannierInputs& wannier = record.wannier;
    for (std::size_t index = 0; index < wannier.kpoint_coordinates.size(); index++) {
        std::vector<Real> interpolated =
            interpolator->eigenvalues_at(wannier.kpoint_coordinates[index]);
        std::vector<Real> expected = wannier.eigenvalues_rydberg[index];
        std::sort(expected.begin(), expected.end());
        ASSERT_EQ(interpolated.size(), 2u);
        EXPECT_NEAR(interpolated[0], expected[0], 1e-10) << "k-point " << index;
        EXPECT_NEAR(interpolated[1], expected[1], 1e-10) << "k-point " << index;
    }
    EXPECT_NEAR(interpolator->diagnostics().at("total_bloch_sum_weight"),
                static_cast<Real>(mesh * mesh * mesh),
                1e-10);
}
