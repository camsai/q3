#include <gtest/gtest.h>

#include <cmath>
#include <complex>
#include <vector>

#include "layer_0_core/types/init.hpp"
#include "layer_0_core/types/matrix.hpp"
#include "layer_3_representation/pseudopotential/planewave/internal/lowdin_orthogonalization.hpp"
#include "layer_4_computation/effective_potential/internal/hubbard/hubbard.hpp"

using namespace q3;

namespace {

Real real_inner_product(const std::vector<Complex>& a, const std::vector<Complex>& b) {
    Complex sum(0.0, 0.0);
    for (size_t g = 0; g < a.size(); g++)
        sum += std::conj(a[g]) * b[g];
    return sum.real();
}

Complex complex_inner_product(const std::vector<Complex>& a, const std::vector<Complex>& b) {
    Complex sum(0.0, 0.0);
    for (size_t g = 0; g < a.size(); g++)
        sum += std::conj(a[g]) * b[g];
    return sum;
}

} // namespace

/**
 * ── ortho-atomic must Lowdin-orthogonalize the FULL system-wide orbital set ──
 *
 * setup_hubbard() (iterator/kohn_sham/planewave/pseudopotential/init.hpp) applies
 * lowdin_orthogonalize_orbitals() to ALL atomic orbitals (every atom, every shell),
 * THEN selects out the Hubbard manifold's columns -- matching Quantum ESPRESSO's
 * convention, which orthogonalizes the system-wide atomic-orbital set first.
 * A shortcut that orthogonalized only the Hubbard manifold in isolation would be a
 * materially different (non-standard) definition whenever some other ("spectator")
 * atomic orbital has nonzero overlap with a manifold member, since Lowdin's S^-1/2
 * couples every orbital connected through the overlap matrix, not just directly
 * overlapping pairs. These tests prove the two are NOT interchangeable, so a future
 * refactor that narrows setup_hubbard()'s orthogonalization to the manifold alone
 * would be silently wrong, not just non-standard.
 */

TEST(HubbardOrthoAtomicProjectors, JointOrthogonalizationDiffersFromManifoldOnlyShortcut) {
    /**
     * manifold_orbital_0 and manifold_orbital_1 are the (non-orthogonal) Hubbard
     * manifold; spectator_orbital represents every OTHER atomic orbital in the
     * system. spectator_orbital overlaps manifold_orbital_1 (dot = 1) but not
     * manifold_orbital_0 (dot = 0) -- an asymmetric coupling, so this isn't a
     * degenerate/trivial configuration.
     */
    std::vector<Complex> manifold_orbital_0 = {{1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}};
    std::vector<Complex> manifold_orbital_1 = {{1.0, 0.0}, {1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}};
    std::vector<Complex> spectator_orbital = {{0.0, 0.0}, {1.0, 0.0}, {1.0, 0.0}, {0.0, 0.0}};

    // The shortcut this test rules out: orthogonalize the manifold alone.
    std::vector<std::vector<Complex>> manifold_only = {manifold_orbital_0, manifold_orbital_1};
    lowdin_orthogonalize_orbitals(manifold_only);

    // setup_hubbard()'s actual logic: orthogonalize the full system, then select.
    std::vector<std::vector<Complex>> full_system = {
        manifold_orbital_0, manifold_orbital_1, spectator_orbital};
    lowdin_orthogonalize_orbitals(full_system);

    // Both give an orthonormal MANIFOLD subset in isolation ...
    EXPECT_NEAR(real_inner_product(manifold_only[0], manifold_only[0]), 1.0, 1e-10);
    EXPECT_NEAR(real_inner_product(manifold_only[0], manifold_only[1]), 0.0, 1e-10);
    EXPECT_NEAR(real_inner_product(full_system[0], full_system[0]), 1.0, 1e-10);
    EXPECT_NEAR(real_inner_product(full_system[0], full_system[1]), 0.0, 1e-10);

    /**
     * ... but the joint transform is NOT the manifold-only shortcut, because the
     * spectator orbital has nonzero overlap with manifold_orbital_1.
     */
    Real difference_norm_squared = 0.0;
    for (size_t g = 0; g < manifold_orbital_0.size(); g++) {
        Complex difference = full_system[0][g] - manifold_only[0][g];
        difference_norm_squared += std::norm(difference);
    }
    EXPECT_GT(difference_norm_squared, 1e-3)
        << "Joint (full-system) Lowdin orthogonalization must differ from a "
           "manifold-only shortcut when a spectator orbital overlaps a manifold "
           "member -- a near-zero difference here means setup_hubbard() may have "
           "regressed to orthogonalizing only the Hubbard manifold.";
}

/**
 * ── Physical consequence: the Hubbard occupation matrix and energy genuinely ──
 * ── change between "atomic" and "ortho-atomic" when a spectator orbital     ──
 * ── couples to the manifold.                                                ──
 */
TEST(HubbardOrthoAtomicProjectors, EnergyDiffersFromAtomicWhenManifoldOverlapsSpectator) {
    std::vector<Complex> manifold_orbital_0 = {{1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}};
    std::vector<Complex> manifold_orbital_1 = {{1.0, 0.0}, {1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}};
    std::vector<Complex> spectator_orbital = {{0.0, 0.0}, {1.0, 0.0}, {1.0, 0.0}, {0.0, 0.0}};

    /**
     * A single synthetic "converged wavefunction" band, chosen with nonzero
     * overlap onto both the manifold and the spectator so the two conventions
     * have something to disagree about.
     */
    std::vector<Complex> psi = {{0.2, 0.0}, {1.0, 0.0}, {0.5, 0.0}, {0.3, 0.0}};

    constexpr int manifold_size = 2;
    constexpr Real u_rydberg = 1.0;
    std::vector<Real> band_occupations = {1.0};
    constexpr Real kpoint_weight = 1.0;

    // "atomic": raw, non-orthogonalized manifold orbitals as projectors.
    Matrix<Complex> overlaps_atomic(1, manifold_size);
    overlaps_atomic(0, 0) = complex_inner_product(manifold_orbital_0, psi);
    overlaps_atomic(0, 1) = complex_inner_product(manifold_orbital_1, psi);
    std::vector<Complex> occupation_atomic(manifold_size * manifold_size, Complex(0.0, 0.0));
    accumulate_hubbard_occupation_matrix(
        1, manifold_size, overlaps_atomic, band_occupations, kpoint_weight, occupation_atomic);
    auto result_atomic =
        compute_hubbard_energy_and_potential(manifold_size, u_rydberg, occupation_atomic);

    // "ortho-atomic": manifold orbitals after joint (full-system) orthogonalization.
    std::vector<std::vector<Complex>> full_system = {
        manifold_orbital_0, manifold_orbital_1, spectator_orbital};
    lowdin_orthogonalize_orbitals(full_system);
    Matrix<Complex> overlaps_ortho_atomic(1, manifold_size);
    overlaps_ortho_atomic(0, 0) = complex_inner_product(full_system[0], psi);
    overlaps_ortho_atomic(0, 1) = complex_inner_product(full_system[1], psi);
    std::vector<Complex> occupation_ortho_atomic(manifold_size * manifold_size, Complex(0.0, 0.0));
    accumulate_hubbard_occupation_matrix(1,
                                         manifold_size,
                                         overlaps_ortho_atomic,
                                         band_occupations,
                                         kpoint_weight,
                                         occupation_ortho_atomic);
    auto result_ortho_atomic =
        compute_hubbard_energy_and_potential(manifold_size, u_rydberg, occupation_ortho_atomic);

    EXPECT_GT(std::abs(result_atomic.energy - result_ortho_atomic.energy), 1e-3)
        << "atomic E_U=" << result_atomic.energy
        << " ortho-atomic E_U=" << result_ortho_atomic.energy
        << " -- expected the two projector conventions to give materially "
           "different Hubbard energies for this coupled manifold/spectator setup.";
}

/**
 * ── Degenerate case matching this codebase's real Ni-FCC test fixture: when ──
 * ── the manifold is already orthogonal to every other orbital, "atomic" and ──
 * ── "ortho-atomic" must coincide (documented in TODOs.md's DFT+U write-up). ──
 */
TEST(HubbardOrthoAtomicProjectors, CoincidesWithAtomicWhenManifoldAlreadyOrthogonalToSpectators) {
    std::vector<Complex> manifold_orbital_0 = {{1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}};
    std::vector<Complex> manifold_orbital_1 = {{0.0, 0.0}, {1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}};
    std::vector<Complex> spectator_orbital = {{0.0, 0.0}, {0.0, 0.0}, {1.0, 0.0}, {0.0, 0.0}};

    std::vector<std::vector<Complex>> full_system = {
        manifold_orbital_0, manifold_orbital_1, spectator_orbital};
    lowdin_orthogonalize_orbitals(full_system);

    /**
     * Already-orthonormal orbitals are (up to the eigensolver's numerically exact
     * reproduction of the identity transform) unchanged by Lowdin orthogonalization.
     */
    EXPECT_NEAR(real_inner_product(full_system[0], manifold_orbital_0), 1.0, 1e-10);
    EXPECT_NEAR(real_inner_product(full_system[1], manifold_orbital_1), 1.0, 1e-10);
}
