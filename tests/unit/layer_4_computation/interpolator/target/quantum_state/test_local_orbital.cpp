#include <gtest/gtest.h>

#include <cmath>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "layer_4_computation/interpolator/target/quantum_state/local_orbital.hpp"
#include "utils/cubic_tight_binding.hpp"

using namespace q3;

using namespace q3::test_support;

TEST(LocalOrbitalQuantumStateInterpolator, ReproducesTheAnalyticBandOnAndOffTheMesh) {
    const Real onsite = -0.4;
    const Real hopping = 0.12;
    LocalOrbitalQuantumStateInterpolator interpolator =
        build_cubic_interpolator(3, onsite, hopping);
    EXPECT_EQ(interpolator.number_of_bands(), 1);

    const std::vector<Vector3D> kpoints = {
        Vector3D(0.0, 0.0, 0.0),
        Vector3D(1.0 / 3.0, 0.0, 2.0 / 3.0),
        Vector3D(0.1234, 0.0, 0.0),
        Vector3D(0.5, 0.25, 0.125),
        Vector3D(-0.11, 0.62, 0.05),
    };
    for (const Vector3D& kpoint : kpoints) {
        std::vector<Real> eigenvalues = interpolator.eigenvalues_at(kpoint);
        ASSERT_EQ(eigenvalues.size(), 1u);
        EXPECT_NEAR(eigenvalues[0], analytic_cubic_band(kpoint, onsite, hopping), 1e-10)
            << "k = (" << kpoint.x << ", " << kpoint.y << ", " << kpoint.z << ")";
    }
}

TEST(LocalOrbitalQuantumStateInterpolator, ReportsTheModelSizeAndTheWeightIdentity) {
    /**
     * The total Bloch-sum weight of a k -> R model is the number of k-points
     * it was built from -- the Wigner-Seitz degeneracy identity, restated
     * where a consumer can read it without knowing how the model was made.
     */
    const int mesh = 3;
    LocalOrbitalQuantumStateInterpolator interpolator = build_cubic_interpolator(mesh, 0.0, 0.1);
    std::map<std::string, Real> diagnostics = interpolator.diagnostics();

    ASSERT_EQ(diagnostics.count("total_bloch_sum_weight"), 1u);
    EXPECT_EQ(diagnostics.count("number_of_lattice_vectors"), 0u)
        << "the lattice-vector count is an integer and belongs in description(), not a Real map";
    EXPECT_NEAR(
        diagnostics.at("total_bloch_sum_weight"), static_cast<Real>(mesh * mesh * mesh), 1e-10);
    EXPECT_NE(interpolator.description().find("test"), std::string::npos);
}

TEST(LocalOrbitalQuantumStateInterpolator, RefusesANonPositiveDefiniteOverlap) {
    /**
     * A generalized eigenproblem needs S(k) positive definite. A model whose
     * orbitals are linearly dependent at some k has none, and the solver must
     * say so rather than return whatever LAPACK left in the buffer.
     */
    LocalOrbitalHamiltonian hamiltonian;
    hamiltonian.set_domain(2, {{IntegerVector3D(0, 0, 0), Vector3D(0.0, 0.0, 0.0), 1.0}});
    hamiltonian.set_hamiltonian(
        {{Complex(1.0, 0.0), Complex(0.0, 0.0), Complex(0.0, 0.0), Complex(1.0, 0.0)}});
    /// S = [[1, 1], [1, 1]]: singular, so the two orbitals are the same state.
    hamiltonian.set_overlap(
        {{Complex(1.0, 0.0), Complex(1.0, 0.0), Complex(1.0, 0.0), Complex(1.0, 0.0)}});

    const LocalOrbitalQuantumStateInterpolator interpolator(std::move(hamiltonian), "singular");
    EXPECT_THROW(interpolator.eigenvalues_at(Vector3D(0.0, 0.0, 0.0)), std::runtime_error);
}
