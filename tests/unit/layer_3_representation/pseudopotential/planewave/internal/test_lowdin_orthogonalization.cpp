#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <stdexcept>

#include "io/pseudopotential/upf_reader.hpp"
#include "layer_1_structure/crystal/crystal.hpp"
#include "layer_1_structure/reciprocal_space/kpoints_grid.hpp"
#include "layer_3_representation/basis/planewave/init.hpp"
#include "layer_3_representation/pseudopotential/planewave/internal/atomic_orbital_builder.hpp"
#include "layer_3_representation/pseudopotential/planewave/internal/lowdin_orthogonalization.hpp"
#include "layer_3_representation/pseudopotential/planewave/internal/wavefunction_atomic.hpp"
#include "layer_4_computation/iterator/kohn_sham/planewave/execution/serial.hpp"

using namespace q3;

namespace {

execution::PlanewaveSerialExecutionPolicy serial_policy;

PseudopotentialSet load_nickel_pseudopotential() {
    PseudopotentialSet pseudopotential_set;
    pseudopotential_set.push_back(
        UPFReader::read(std::string(EXAMPLES_DIR) + "/data/pseudopotentials/nc/Ni.upf"));
    return pseudopotential_set;
}

Crystal make_nickel_fcc(const PseudopotentialSet& pseudopotential_set) {
    Crystal crystal_structure;
    crystal_structure.set_from_lattice_type(2, 6.65);
    AtomSpecies nickel{"Ni", 58.69, pseudopotential_set[0].valence_charge, "Ni.upf"};
    crystal_structure.add_species(nickel);
    crystal_structure.add_atom({0, {0.0, 0.0, 0.0}});
    crystal_structure.compute_derived_quantities();
    return crystal_structure;
}

/// <a|b> for full (non-gamma-only) G-space storage: plain conjugate dot product.
Complex inner_product(const std::vector<Complex>& a, const std::vector<Complex>& b) {
    Complex value(0.0, 0.0);
    for (size_t g_index = 0; g_index < a.size(); g_index++) {
        value += std::conj(a[g_index]) * b[g_index];
    }
    return value;
}

/**
 * A deterministic Hermitian matrix, column-major (element (row, column) at
 * row + column * size). `shift` is added to the diagonal, which is what makes
 * the result positive definite and therefore a legal overlap matrix.
 */
std::vector<Complex> hermitian_matrix(int size, unsigned seed, Real shift = 0.0) {
    auto pseudo_random = [seed](int a, int b) {
        unsigned hashed = seed * 2654435761u + static_cast<unsigned>(a) * 40503u +
                          static_cast<unsigned>(b) * 7919u;
        hashed ^= hashed >> 13;
        hashed *= 1274126177u;
        hashed ^= hashed >> 16;
        return static_cast<Real>(hashed % 2000u) / 1000.0 - 1.0;
    };
    std::vector<Complex> matrix(static_cast<size_t>(size) * size, Complex(0.0, 0.0));
    for (int column = 0; column < size; column++) {
        for (int row = 0; row <= column; row++) {
            Real real_part = pseudo_random(row, column);
            Real imaginary_part = (row == column) ? 0.0 : pseudo_random(column, row);
            matrix[static_cast<size_t>(row + column * size)] = Complex(real_part, imaginary_part);
            matrix[static_cast<size_t>(column + row * size)] = Complex(real_part, -imaginary_part);
        }
        matrix[static_cast<size_t>(column + column * size)] += Complex(shift, 0.0);
    }
    return matrix;
}

std::vector<Complex>
combine(const std::vector<Complex>& base, Real coefficient, const std::vector<Complex>& direction) {
    std::vector<Complex> result(base.size());
    for (size_t index = 0; index < base.size(); index++)
        result[index] = base[index] + coefficient * direction[index];
    return result;
}

std::vector<std::vector<Complex>> build_nickel_orbitals_at_gamma() {
    auto pseudopotential_set = load_nickel_pseudopotential();
    Crystal crystal_structure = make_nickel_fcc(pseudopotential_set);

    PlanewaveBasis basis;
    basis.generate(serial_policy, crystal_structure, 30.0, 120.0);

    KPointsGrid kpoint_grid;
    kpoint_grid.generate(2, 2, 2, 1, 1, 1); // shifted, non-gamma-only k-point

    std::vector<int> g_vector_indices;
    std::vector<Real> kinetic_energy_squared;
    basis.compute_gk(
        kpoint_grid[0].coordinates, crystal_structure, g_vector_indices, kinetic_energy_squared);
    int number_of_plane_waves = static_cast<int>(g_vector_indices.size());

    auto atomic_wavefunction_table = tabulate_atomic_wavefunctions_in_q_space(
        pseudopotential_set, crystal_structure.cell_volume(), 0.01, 900);

    std::vector<std::vector<Complex>> orbitals;
    build_atomic_orbitals_g_space(0,
                                  number_of_plane_waves,
                                  g_vector_indices,
                                  crystal_structure,
                                  kpoint_grid,
                                  basis,
                                  pseudopotential_set,
                                  atomic_wavefunction_table,
                                  0.01,
                                  900,
                                  orbitals);
    return orbitals;
}

} // namespace

/**
 * ── The raw (pre-orthogonalization) atomic orbitals are NOT orthogonal ──
 * (a meaningful precondition: if they already were, Löwdin would be a no-op
 * and the orthonormality check below wouldn't actually be testing anything).
 */

TEST(LowdinOrthogonalization, RawNickelOrbitalsAreNotAlreadyOrthogonal) {
    auto orbitals = build_nickel_orbitals_at_gamma();
    ASSERT_EQ(orbitals.size(), 10u);

    /**
     * Both 3S (index 0) and 4S (index 9) are l=0 shells on the same atom --
     * generic pseudo-atomic radial functions from different reference
     * configurations are not guaranteed orthogonal.
     */
    Complex overlap_3s_4s = inner_product(orbitals[0], orbitals[9]);
    EXPECT_GT(std::abs(overlap_3s_4s), 1e-3)
        << "Expected non-trivial 3S/4S overlap before orthogonalization";
}

// ── After Löwdin orthogonalization, <phi'_i|phi'_j> = delta_ij ──────────

TEST(LowdinOrthogonalization, OrbitalsAreOrthonormalAfterOrthogonalization) {
    auto orbitals = build_nickel_orbitals_at_gamma();
    lowdin_orthogonalize_orbitals(orbitals);
    ASSERT_EQ(orbitals.size(), 10u);

    Real tolerance = 1e-8;
    for (size_t i = 0; i < orbitals.size(); i++) {
        for (size_t j = 0; j < orbitals.size(); j++) {
            Complex overlap = inner_product(orbitals[i], orbitals[j]);
            Real expected_real = (i == j) ? 1.0 : 0.0;
            EXPECT_NEAR(overlap.real(), expected_real, tolerance)
                << "Re<phi'_" << i << "|phi'_" << j << "> should be " << expected_real;
            EXPECT_NEAR(overlap.imag(), 0.0, tolerance)
                << "Im<phi'_" << i << "|phi'_" << j << "> should be 0";
        }
    }
}

// ── Löwdin orthogonalization preserves the orbital count and plane-wave count ──

TEST(LowdinOrthogonalization, PreservesShape) {
    auto orbitals = build_nickel_orbitals_at_gamma();
    size_t number_of_orbitals = orbitals.size();
    size_t number_of_plane_waves = orbitals[0].size();

    lowdin_orthogonalize_orbitals(orbitals);

    EXPECT_EQ(orbitals.size(), number_of_orbitals);
    for (const auto& orbital : orbitals) {
        EXPECT_EQ(orbital.size(), number_of_plane_waves);
    }
}

TEST(LowdinOrthogonalization, DistributedSlabsUseTheSummedOverlap) {
    std::vector<std::vector<Complex>> first_slab = {
        {Complex(1.0, 0.0), Complex(0.0, 0.0)},
        {Complex(0.5, 0.2), Complex(1.0, 0.0)},
    };
    std::vector<std::vector<Complex>> second_slab = {
        {Complex(1.0, 0.0), Complex(0.2, -0.1)},
        {Complex(0.0, 0.0), Complex(1.0, 0.0)},
    };

    auto complete_overlap = compute_orbital_overlap_matrix(first_slab);
    auto second_slab_overlap = compute_orbital_overlap_matrix(second_slab);
    ASSERT_EQ(complete_overlap.size(), second_slab_overlap.size());
    for (size_t index = 0; index < complete_overlap.size(); index++)
        complete_overlap[index] += second_slab_overlap[index];

    lowdin_orthogonalize_orbitals(first_slab, complete_overlap);
    lowdin_orthogonalize_orbitals(second_slab, complete_overlap);

    for (size_t row = 0; row < first_slab.size(); row++) {
        for (size_t column = 0; column < first_slab.size(); column++) {
            Complex overlap = inner_product(first_slab[row], first_slab[column]) +
                              inner_product(second_slab[row], second_slab[column]);
            Real expected = row == column ? 1.0 : 0.0;
            EXPECT_NEAR(overlap.real(), expected, 1e-12);
            EXPECT_NEAR(overlap.imag(), 0.0, 1e-12);
        }
    }
}

// ── A single orbital is trivially "orthogonalized" to itself, renormalized ──

TEST(LowdinOrthogonalization, SingleOrbitalIsRenormalized) {
    std::vector<std::vector<Complex>> orbitals = {{Complex(2.0, 0.0), Complex(0.0, 0.0)}};
    lowdin_orthogonalize_orbitals(orbitals);

    ASSERT_EQ(orbitals.size(), 1u);
    Complex norm_squared = inner_product(orbitals[0], orbitals[0]);
    EXPECT_NEAR(norm_squared.real(), 1.0, 1e-10);
    EXPECT_NEAR(norm_squared.imag(), 0.0, 1e-10);
}

// ── Linearly dependent orbitals (singular overlap) must fail loudly ─────

TEST(LowdinOrthogonalization, LinearlyDependentOrbitalsThrow) {
    std::vector<std::vector<Complex>> orbitals = {
        {Complex(1.0, 0.0), Complex(0.0, 0.0)},
        {Complex(2.0, 0.0), Complex(0.0, 0.0)}, // parallel to the first -- singular overlap
    };
    EXPECT_THROW(lowdin_orthogonalize_orbitals(orbitals), std::runtime_error);
}

/**
 * ── d(S^{-1/2}) against a finite difference of S^{-1/2} itself ──────────
 *
 * The closed form is the whole content of the DFT+U ortho-atomic force: it
 * supplies the term from every atomic orbital's own position-dependence
 * through the joint transform, which the atomic convention has no analogue
 * of. It is also the one piece that cannot be checked by inspection --
 * S^{-1/2} and dS do not commute, so the naive "-1/2 S^{-3/2} dS" is wrong
 * and wrong in a way that still looks dimensionally right.
 *
 * Differentiating along a straight line S(t) = S + t*dS makes the check exact
 * and independent of any physics: the analytic result must equal
 * [S^{-1/2}(h) - S^{-1/2}(-h)] / 2h to the finite difference's own accuracy.
 */
TEST(LowdinInverseSquareRootDerivative, MatchesAFiniteDifferenceOfTheTransform) {
    constexpr int size = 5;
    auto overlap = hermitian_matrix(size, 11u, /*shift=*/4.0);
    auto direction = hermitian_matrix(size, 29u);

    auto transform = compute_lowdin_transform(overlap, size);
    auto analytic = lowdin_inverse_square_root_derivative(transform, direction);

    constexpr Real step = 1e-5;
    auto plus = compute_lowdin_transform(combine(overlap, step, direction), size);
    auto minus = compute_lowdin_transform(combine(overlap, -step, direction), size);

    Real largest_deviation = 0.0;
    for (size_t index = 0; index < analytic.size(); index++) {
        Complex numerical =
            (plus.inverse_square_root[index] - minus.inverse_square_root[index]) / (2.0 * step);
        largest_deviation = std::max(largest_deviation, std::abs(analytic[index] - numerical));
    }
    EXPECT_LT(largest_deviation, 1e-8)
        << "d(S^-1/2) disagrees with a central difference of S^-1/2 by " << largest_deviation;

    /**
     * The premise: the derivative is not small compared with the tolerance,
     * so the agreement above is a real measurement rather than two zeros.
     */
    Real largest_element = 0.0;
    for (const auto& value : analytic)
        largest_element = std::max(largest_element, std::abs(value));
    EXPECT_GT(largest_element, 1e-2) << "the test perturbation barely moves S^-1/2";
}

/**
 * The naive derivative that commutes S^{-1/2} past dS. HubbardForceCalculator
 * reads d(S^{-1/2}) with the index order transposed relative to how it is
 * written, which is only legitimate because the result is Hermitian; and the
 * eigenbasis denominator is what makes it so. A one-line slip that dropped it
 * would still produce a plausible matrix.
 */
TEST(LowdinInverseSquareRootDerivative, IsHermitianWhenTheOverlapDerivativeIs) {
    constexpr int size = 6;
    auto transform = compute_lowdin_transform(hermitian_matrix(size, 3u, /*shift=*/5.0), size);
    auto derivative = lowdin_inverse_square_root_derivative(transform, hermitian_matrix(size, 17u));

    for (int column = 0; column < size; column++)
        for (int row = 0; row < size; row++) {
            Complex upper = derivative[static_cast<size_t>(row + column * size)];
            Complex lower = derivative[static_cast<size_t>(column + row * size)];
            EXPECT_NEAR(upper.real(), lower.real(), 1e-12);
            EXPECT_NEAR(upper.imag(), -lower.imag(), 1e-12);
        }
}

TEST(LowdinInverseSquareRootDerivative, RejectsAnOverlapDerivativeOfTheWrongSize) {
    auto transform = compute_lowdin_transform(hermitian_matrix(4, 5u, /*shift=*/4.0), 4);
    EXPECT_THROW(lowdin_inverse_square_root_derivative(transform, hermitian_matrix(3, 7u)),
                 std::runtime_error);
}
