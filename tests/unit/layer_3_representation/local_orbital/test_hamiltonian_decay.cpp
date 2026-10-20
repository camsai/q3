#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

#include "layer_0_core/types/init.hpp"
#include "layer_3_representation/local_orbital/hamiltonian_decay.hpp"
#include "layer_3_representation/local_orbital/init.hpp"

using namespace q3;

namespace {

/**
 * A one-dimensional Hamiltonian on five lattice vectors, the outer pair on
 * the Wigner-Seitz boundary with weight one half, and hoppings that fall
 * off with distance: 3 on site, 0.5 at the neighbours, 0.2 at the boundary.
 */
LocalOrbitalHamiltonian decaying_chain(int orbitals) {
    std::vector<LocalOrbitalLatticeVector> domain;
    std::vector<std::vector<Complex>> blocks;
    for (int translation = -2; translation <= 2; translation++) {
        const bool boundary = std::abs(translation) == 2;
        domain.push_back({IntegerVector3D{translation, 0, 0},
                          Vector3D{5.0 * translation, 0.0, 0.0},
                          boundary ? 0.5 : 1.0});
        std::vector<Complex> block(static_cast<std::size_t>(orbitals) * orbitals,
                                   Complex(0.0, 0.0));
        const Real magnitude = translation == 0 ? 3.0 : (boundary ? 0.2 : 0.5);
        for (int orbital = 0; orbital < orbitals; orbital++) {
            block[static_cast<std::size_t>(orbital) * orbitals + orbital] = Complex(magnitude, 0.0);
        }
        blocks.push_back(std::move(block));
    }
    LocalOrbitalHamiltonian hamiltonian;
    hamiltonian.set_domain(orbitals, std::move(domain));
    hamiltonian.set_hamiltonian(std::move(blocks));
    return hamiltonian;
}

} // namespace

TEST(HamiltonianDecay, ItIsTheOutermostBlochSumTermAgainstTheLargestHopping) {
    const HamiltonianDecay decay = HamiltonianDecay::measured_on(decaying_chain(1));

    EXPECT_NEAR(decay.outermost_shell_distance, 10.0, 1.0e-12);
    EXPECT_EQ(decay.outermost_shell_size, 2);
    // The boundary block is 0.2 but enters the Bloch sum with weight one half.
    EXPECT_NEAR(decay.outermost_shell_maximum, 0.1, 1.0e-12);
    // The on-site 3.0 is not a hopping: it carries the zero of energy.
    EXPECT_NEAR(decay.largest_hopping, 0.5, 1.0e-12);
    EXPECT_NEAR(decay.ratio, 0.2, 1.0e-12);
}

TEST(HamiltonianDecay, AnOnSiteOffDiagonalElementCountsAsAHopping) {
    LocalOrbitalHamiltonian hamiltonian = decaying_chain(2);
    std::vector<std::vector<Complex>> blocks = hamiltonian.hamiltonian_blocks();
    blocks[2][1] = Complex(0.0, 0.7); // R = 0, row 1 column 0
    blocks[2][2] = Complex(0.0, -0.7);
    hamiltonian.set_hamiltonian(std::move(blocks));

    const HamiltonianDecay decay = HamiltonianDecay::measured_on(hamiltonian);
    EXPECT_NEAR(decay.largest_hopping, 0.7, 1.0e-12);
    EXPECT_NEAR(decay.ratio, 0.1 / 0.7, 1.0e-12);
}

TEST(HamiltonianDecay, AHamiltonianWithoutBlocksIsRefused) {
    const LocalOrbitalHamiltonian hamiltonian;
    EXPECT_THROW((void)HamiltonianDecay::measured_on(hamiltonian), std::runtime_error);
}
