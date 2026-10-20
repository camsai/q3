#include <gtest/gtest.h>

#include <cmath>
#include <complex>
#include <vector>

#include "layer_0_core/types/constants.hpp"
#include "layer_3_representation/local_orbital/init.hpp"

using namespace q3;

namespace {

/**
 * A one-orbital chain along the first lattice direction with on-site
 * energy `onsite` and nearest-neighbour hopping `hopping`: the domain is
 * {-1, 0, +1} with unit weights, as a Hamiltonian from real-space integrals
 * would have.
 */
LocalOrbitalHamiltonian chain(Real onsite, Real hopping) {
    LocalOrbitalHamiltonian hamiltonian;
    hamiltonian.set_domain(1,
                           {{IntegerVector3D(-1, 0, 0), Vector3D(-1.0, 0.0, 0.0), 1.0},
                            {IntegerVector3D(0, 0, 0), Vector3D(0.0, 0.0, 0.0), 1.0},
                            {IntegerVector3D(1, 0, 0), Vector3D(1.0, 0.0, 0.0), 1.0}});
    hamiltonian.set_hamiltonian(
        {{Complex(hopping, 0.0)}, {Complex(onsite, 0.0)}, {Complex(hopping, 0.0)}});
    return hamiltonian;
}

} // namespace

TEST(LocalOrbitalHamiltonian, BlochSumReproducesTheAnalyticChainDispersion) {
    LocalOrbitalHamiltonian hamiltonian = chain(-0.3, 0.1);
    for (Real k : {0.0, 0.1234, 0.25, 0.5, -0.37}) {
        std::vector<Complex> block = hamiltonian.hamiltonian_at_kpoint(Vector3D(k, 0.0, 0.0));
        ASSERT_EQ(block.size(), 1u);
        EXPECT_NEAR(block[0].real(), -0.3 + 2.0 * 0.1 * std::cos(constants::two_pi * k), 1e-12);
        EXPECT_NEAR(block[0].imag(), 0.0, 1e-12);
    }
}

TEST(LocalOrbitalHamiltonian, OverlapIsTheIdentityWhenNoneWasSet) {
    LocalOrbitalHamiltonian hamiltonian = chain(-0.3, 0.1);
    EXPECT_FALSE(hamiltonian.has_nontrivial_overlap());
    std::vector<Complex> overlap = hamiltonian.overlap_at_kpoint(Vector3D(0.3, 0.1, 0.0));
    ASSERT_EQ(overlap.size(), 1u);
    EXPECT_NEAR(overlap[0].real(), 1.0, 1e-15);
    EXPECT_NEAR(overlap[0].imag(), 0.0, 1e-15);
}

TEST(LocalOrbitalHamiltonian, OverlapBlocksBlochSumLikeTheHamiltonian) {
    /**
     * A non-orthogonal chain: S(0) = 1, S(+-1) = s. The Bloch sum of the
     * overlap must follow exactly the same rule as the Hamiltonian's, since
     * a generalized eigenproblem downstream depends on the two being
     * transformed identically.
     */
    LocalOrbitalHamiltonian hamiltonian = chain(-0.3, 0.1);
    const Real s = 0.2;
    hamiltonian.set_overlap({{Complex(s, 0.0)}, {Complex(1.0, 0.0)}, {Complex(s, 0.0)}});
    EXPECT_TRUE(hamiltonian.has_nontrivial_overlap());
    for (Real k : {0.0, 0.2, 0.5}) {
        std::vector<Complex> overlap = hamiltonian.overlap_at_kpoint(Vector3D(k, 0.0, 0.0));
        EXPECT_NEAR(overlap[0].real(), 1.0 + 2.0 * s * std::cos(constants::two_pi * k), 1e-12);
    }
}

TEST(LocalOrbitalHamiltonian, BlochSumWeightsScaleTheContribution) {
    /**
     * A boundary vector shared between two images carries weight 1/2; the
     * Bloch sum must apply it, otherwise a Wannier-derived Hamiltonian
     * double-counts every Wigner-Seitz boundary block.
     */
    LocalOrbitalHamiltonian weighted;
    weighted.set_domain(1,
                        {{IntegerVector3D(0, 0, 0), Vector3D(0.0, 0.0, 0.0), 1.0},
                         {IntegerVector3D(2, 0, 0), Vector3D(2.0, 0.0, 0.0), 0.5},
                         {IntegerVector3D(-2, 0, 0), Vector3D(-2.0, 0.0, 0.0), 0.5}});
    weighted.set_hamiltonian({{Complex(0.0, 0.0)}, {Complex(1.0, 0.0)}, {Complex(1.0, 0.0)}});
    // At k = 0 the two boundary blocks contribute 0.5 each: total 1, not 2.
    EXPECT_NEAR(weighted.hamiltonian_at_kpoint(Vector3D(0.0, 0.0, 0.0))[0].real(), 1.0, 1e-12);
}

TEST(LocalOrbitalHamiltonian, BlochSumOfAHermitianSetIsHermitian) {
    /**
     * Two orbitals, complex hoppings, with H(-R) = H(R)^dagger by
     * construction. H(k) must then be Hermitian at every k.
     */
    LocalOrbitalHamiltonian hamiltonian;
    hamiltonian.set_domain(2,
                           {{IntegerVector3D(0, 0, 0), Vector3D(0.0, 0.0, 0.0), 1.0},
                            {IntegerVector3D(1, 0, 0), Vector3D(1.0, 0.0, 0.0), 1.0},
                            {IntegerVector3D(-1, 0, 0), Vector3D(-1.0, 0.0, 0.0), 1.0}});
    // Column-major 2x2 blocks: [a c; b d] stored as {a, b, c, d}.
    std::vector<Complex> onsite = {
        Complex(0.5, 0.0), Complex(0.1, 0.2), Complex(0.1, -0.2), Complex(-0.5, 0.0)};
    std::vector<Complex> forward = {
        Complex(0.3, 0.1), Complex(0.0, 0.4), Complex(0.2, -0.1), Complex(0.05, 0.0)};
    // H(-R) = H(R)^dagger: conjugate transpose of `forward`.
    std::vector<Complex> backward = {
        std::conj(forward[0]), std::conj(forward[2]), std::conj(forward[1]), std::conj(forward[3])};
    hamiltonian.set_hamiltonian({onsite, forward, backward});

    for (Real k : {0.0, 0.17, 0.5, -0.31}) {
        std::vector<Complex> block = hamiltonian.hamiltonian_at_kpoint(Vector3D(k, 0.0, 0.0));
        for (int row = 0; row < 2; row++) {
            for (int column = 0; column < 2; column++) {
                Complex element = block[static_cast<std::size_t>(column) * 2 + row];
                Complex mirrored = block[static_cast<std::size_t>(row) * 2 + column];
                EXPECT_NEAR(std::abs(element - std::conj(mirrored)), 0.0, 1e-12)
                    << "k " << k << " (" << row << ", " << column << ")";
            }
        }
    }
}

TEST(LocalOrbitalHamiltonian, RefusesMisshapenBlocksAndMissingDomain) {
    LocalOrbitalHamiltonian hamiltonian;
    EXPECT_THROW(hamiltonian.set_hamiltonian({{Complex(1.0, 0.0)}}), std::runtime_error);
    hamiltonian.set_domain(2, {{IntegerVector3D(0, 0, 0), Vector3D(0.0, 0.0, 0.0), 1.0}});
    // Two lattice vectors' worth of blocks for a one-vector domain.
    EXPECT_THROW(hamiltonian.set_hamiltonian({std::vector<Complex>(4), std::vector<Complex>(4)}),
                 std::invalid_argument);
    // A 1x1 block for a 2-orbital basis.
    EXPECT_THROW(hamiltonian.set_hamiltonian({std::vector<Complex>(1)}), std::invalid_argument);
    EXPECT_THROW(hamiltonian.hamiltonian_at_kpoint(Vector3D(0.0, 0.0, 0.0)), std::runtime_error);
    EXPECT_THROW(hamiltonian.set_domain(0, {}), std::invalid_argument);
}
