#include <gtest/gtest.h>

#include <cmath>
#include <complex>
#include <vector>

#include "layer_4_computation/iterator/internal/mixing/preconditioner/kerker.hpp"
#include "layer_4_computation/iterator/internal/mixing/preconditioner/local_thomas_fermi.hpp"

using namespace q3;

namespace {

constexpr int kFirstNonzero = 1;

/**
 * A discrete Fourier pair standing in for the density transforms.
 *
 * The real engine uses FFTs over a plane-wave G-sphere; all this preconditioner
 * needs of them is that they are mutual inverses, so a dense DFT of the same
 * size reproduces the property under test at a size that is easy to reason
 * about. Vectors are kept Hermitian (g[n-k] = conj(g[k])) so the real-space
 * image is real and the round trip is exact.
 */
struct FourierPair {
    int size;

    void to_real(const std::vector<Complex>& reciprocal, std::vector<Real>& real_space) const {
        real_space.assign(size, 0.0);
        for (int point = 0; point < size; ++point) {
            Complex accumulator(0.0, 0.0);
            for (int mode = 0; mode < size; ++mode) {
                Real phase = 2.0 * M_PI * point * mode / size;
                accumulator += reciprocal[mode] * Complex(std::cos(phase), std::sin(phase));
            }
            real_space[point] = accumulator.real();
        }
    }

    void to_reciprocal(const std::vector<Real>& real_space,
                       std::vector<Complex>& reciprocal) const {
        reciprocal.assign(size, Complex(0.0, 0.0));
        for (int mode = 0; mode < size; ++mode) {
            Complex accumulator(0.0, 0.0);
            for (int point = 0; point < size; ++point) {
                Real phase = -2.0 * M_PI * point * mode / size;
                accumulator += real_space[point] * Complex(std::cos(phase), std::sin(phase));
            }
            reciprocal[mode] = accumulator / static_cast<Real>(size);
        }
    }
};

/// |G|² symmetric under k → n−k, as a real G-sphere is.
std::vector<Real> symmetric_g_squared(int size) {
    std::vector<Real> g_squared(size, 0.0);
    for (int mode = 1; mode < size; ++mode) {
        int wavenumber = std::min(mode, size - mode);
        g_squared[mode] = static_cast<Real>(wavenumber * wavenumber);
    }
    return g_squared;
}

/// A Hermitian residual, so its real-space image is real.
std::vector<Complex> hermitian_residual(int size) {
    std::vector<Complex> residual(size, Complex(0.0, 0.0));
    residual[0] = Complex(0.7, 0.0); // G = 0, must survive untouched
    for (int mode = 1; mode <= size / 2; ++mode) {
        Complex value(0.3 * mode, 0.15 * (mode % 3) - 0.1);
        if (mode == size - mode)
            value = Complex(value.real(), 0.0); // self-conjugate mode
        residual[mode] = value;
        residual[size - mode] = std::conj(value);
    }
    return residual;
}

Real thomas_fermi_screening(Real density) {
    return 4.0 * std::cbrt(3.0 * M_PI * M_PI * std::max(density, 0.0)) / M_PI;
}

} // namespace

TEST(LocalThomasFermiPreconditioner, NameIsLocalTf) {
    FourierPair transforms{8};
    LocalThomasFermiPreconditioner preconditioner(
        symmetric_g_squared(8),
        1.0,
        kFirstNonzero,
        [&](const std::vector<Real>& r, std::vector<Complex>& g) {
            transforms.to_reciprocal(r, g);
        },
        [&](const std::vector<Complex>& g, std::vector<Real>& r) { transforms.to_real(g, r); });
    EXPECT_EQ(preconditioner.name(), "local-tf");
}

/**
 * The identity the whole implementation rests on.
 *
 * A uniform density makes k_TF²(r) constant, and [G² + c] is then diagonal, so
 * the solve must return exactly the Kerker filter built from that constant. If
 * this drifts, the operator, the solve, or the screening formula is wrong.
 */
TEST(LocalThomasFermiPreconditioner, UniformDensityReproducesKerker) {
    const int size = 8;
    const Real density = 0.05;   // bohr⁻³, well above the vacuum floor below
    const Real q0_squared = 0.1; // low enough not to clamp

    FourierPair transforms{size};
    std::vector<Real> g_squared = symmetric_g_squared(size);

    LocalThomasFermiPreconditioner preconditioner(
        g_squared,
        q0_squared,
        kFirstNonzero,
        [&](const std::vector<Real>& r, std::vector<Complex>& g) {
            transforms.to_reciprocal(r, g);
        },
        [&](const std::vector<Complex>& g, std::vector<Real>& r) { transforms.to_real(g, r); });

    preconditioner.update(std::vector<Real>(size, density));

    Real expected_screening = thomas_fermi_screening(density);
    ASSERT_GT(expected_screening, q0_squared) << "test would not exercise the local branch";
    EXPECT_NEAR(preconditioner.mean_screening(), expected_screening, 1.0e-12);

    std::vector<Complex> residual = hermitian_residual(size);
    std::vector<Complex> expected = residual;
    KerkerPreconditioner kerker(g_squared, expected_screening, kFirstNonzero);
    kerker.apply(expected);

    preconditioner.apply(residual);

    for (int mode = kFirstNonzero; mode < size; ++mode) {
        EXPECT_NEAR(residual[mode].real(), expected[mode].real(), 1.0e-9) << "mode " << mode;
        EXPECT_NEAR(residual[mode].imag(), expected[mode].imag(), 1.0e-9) << "mode " << mode;
    }
}

/**
 * On an inhomogeneous density the answer has no closed form, so the test checks
 * the defining equation instead: the returned x must satisfy
 * [G² + k_TF²(r)] x = G² F, with the operator rebuilt independently here.
 */
TEST(LocalThomasFermiPreconditioner, SolvesItsDefiningEquationOnAnInhomogeneousDensity) {
    const int size = 16;
    const Real q0_squared = 0.05;

    FourierPair transforms{size};
    std::vector<Real> g_squared = symmetric_g_squared(size);

    // A slab-like profile: dense in the middle, near-vacuum at the edges.
    std::vector<Real> density(size, 0.0);
    for (int point = 0; point < size; ++point) {
        Real centred = static_cast<Real>(point - size / 2) / (size / 4.0);
        density[point] = 0.2 * std::exp(-centred * centred);
    }

    LocalThomasFermiPreconditioner preconditioner(
        g_squared,
        q0_squared,
        kFirstNonzero,
        [&](const std::vector<Real>& r, std::vector<Complex>& g) {
            transforms.to_reciprocal(r, g);
        },
        [&](const std::vector<Complex>& g, std::vector<Real>& r) { transforms.to_real(g, r); });
    preconditioner.update(density);

    std::vector<Complex> input = hermitian_residual(size);
    std::vector<Complex> solution = input;
    preconditioner.apply(solution);

    // Rebuild k_TF²(r) with the vacuum floor, exactly as the class documents it.
    std::vector<Real> screening(size, 0.0);
    for (int point = 0; point < size; ++point) {
        screening[point] = std::max(thomas_fermi_screening(density[point]), q0_squared);
    }

    /**
     * Apply [G² + k_TF²(r)] to the solution, on the subspace the solve runs in.
     *
     * apply() returns the caller's G = 0 component untouched, by contract, but
     * the solve itself represents that component as zero -- so it has to be
     * projected out here too. Leaving it in feeds a spurious DC term through
     * the real-space multiply, which spreads over every G and shows up only in
     * the real parts, a DC component being purely real.
     */
    std::vector<Complex> projected = solution;
    projected[0] = Complex(0.0, 0.0);

    std::vector<Real> solution_real;
    transforms.to_real(projected, solution_real);
    for (int point = 0; point < size; ++point) {
        solution_real[point] *= screening[point];
    }
    std::vector<Complex> operator_applied;
    transforms.to_reciprocal(solution_real, operator_applied);
    for (int mode = kFirstNonzero; mode < size; ++mode) {
        operator_applied[mode] += g_squared[mode] * projected[mode];
    }

    /**
     * The solve exits on a *relative* residual, so the accuracy to demand here
     * scales with the size of the right-hand side rather than being an absolute
     * constant -- an absolute bound would silently become a different test as
     * soon as the residual used here was rescaled.
     */
    Real right_hand_side_norm = 0.0;
    for (int mode = kFirstNonzero; mode < size; ++mode) {
        right_hand_side_norm += std::norm(g_squared[mode] * input[mode]);
    }
    right_hand_side_norm = std::sqrt(right_hand_side_norm);
    Real tolerance = 1.0e-7 * right_hand_side_norm;

    for (int mode = kFirstNonzero; mode < size; ++mode) {
        Complex right_hand_side = g_squared[mode] * input[mode];
        EXPECT_NEAR(operator_applied[mode].real(), right_hand_side.real(), tolerance)
            << "mode " << mode;
        EXPECT_NEAR(operator_applied[mode].imag(), right_hand_side.imag(), tolerance)
            << "mode " << mode;
    }
}

/// Charge conservation: the G = 0 component is never touched.
TEST(LocalThomasFermiPreconditioner, LeavesGZeroUntouched) {
    const int size = 8;
    FourierPair transforms{size};
    LocalThomasFermiPreconditioner preconditioner(
        symmetric_g_squared(size),
        0.5,
        kFirstNonzero,
        [&](const std::vector<Real>& r, std::vector<Complex>& g) {
            transforms.to_reciprocal(r, g);
        },
        [&](const std::vector<Complex>& g, std::vector<Real>& r) { transforms.to_real(g, r); });
    preconditioner.update(std::vector<Real>(size, 0.03));

    std::vector<Complex> residual = hermitian_residual(size);
    Complex charge_component = residual[0];
    preconditioner.apply(residual);
    EXPECT_DOUBLE_EQ(residual[0].real(), charge_component.real());
    EXPECT_DOUBLE_EQ(residual[0].imag(), charge_component.imag());
}

/**
 * The vacuum floor. Screening from ρ → 0 vanishes, which is correct physics but
 * leaves the operator singular exactly where a slab calculation needs it, so
 * k_TF² is floored at q0² instead.
 */
TEST(LocalThomasFermiPreconditioner, VacuumFloorKeepsScreeningAtQ0) {
    const int size = 8;
    const Real q0_squared = 0.8;
    FourierPair transforms{size};
    LocalThomasFermiPreconditioner preconditioner(
        symmetric_g_squared(size),
        q0_squared,
        kFirstNonzero,
        [&](const std::vector<Real>& r, std::vector<Complex>& g) {
            transforms.to_reciprocal(r, g);
        },
        [&](const std::vector<Complex>& g, std::vector<Real>& r) { transforms.to_real(g, r); });

    preconditioner.update(std::vector<Real>(size, 0.0));
    EXPECT_DOUBLE_EQ(preconditioner.mean_screening(), q0_squared);

    // In pure vacuum the floor makes the operator uniform, i.e. Kerker at q0.
    std::vector<Complex> residual = hermitian_residual(size);
    std::vector<Complex> expected = residual;
    KerkerPreconditioner kerker(symmetric_g_squared(size), q0_squared, kFirstNonzero);
    kerker.apply(expected);
    preconditioner.apply(residual);
    for (int mode = kFirstNonzero; mode < size; ++mode) {
        EXPECT_NEAR(residual[mode].real(), expected[mode].real(), 1.0e-9) << "mode " << mode;
    }
}

/// A negative density from FFT round-off must not produce an indefinite operator.
TEST(LocalThomasFermiPreconditioner, ClampsNegativeDensityToTheFloor) {
    const int size = 8;
    const Real q0_squared = 0.25;
    FourierPair transforms{size};
    LocalThomasFermiPreconditioner preconditioner(
        symmetric_g_squared(size),
        q0_squared,
        kFirstNonzero,
        [&](const std::vector<Real>& r, std::vector<Complex>& g) {
            transforms.to_reciprocal(r, g);
        },
        [&](const std::vector<Complex>& g, std::vector<Real>& r) { transforms.to_real(g, r); });

    std::vector<Real> density(size, -1.0e-8);
    preconditioner.update(density);
    EXPECT_DOUBLE_EQ(preconditioner.mean_screening(), q0_squared);

    std::vector<Complex> residual = hermitian_residual(size);
    preconditioner.apply(residual);
    for (int mode = kFirstNonzero; mode < size; ++mode) {
        EXPECT_TRUE(std::isfinite(residual[mode].real()));
        EXPECT_TRUE(std::isfinite(residual[mode].imag()));
    }
}

/// Denser matter screens harder, so the cell-averaged k_TF² rises with density.
TEST(LocalThomasFermiPreconditioner, MeanScreeningGrowsWithDensity) {
    const int size = 8;
    FourierPair transforms{size};
    auto make = [&](Real density) {
        LocalThomasFermiPreconditioner preconditioner(
            symmetric_g_squared(size),
            0.01,
            kFirstNonzero,
            [&](const std::vector<Real>& r, std::vector<Complex>& g) {
                transforms.to_reciprocal(r, g);
            },
            [&](const std::vector<Complex>& g, std::vector<Real>& r) { transforms.to_real(g, r); });
        preconditioner.update(std::vector<Real>(size, density));
        return preconditioner.mean_screening();
    };
    EXPECT_LT(make(0.01), make(0.1));
    EXPECT_LT(make(0.1), make(1.0));
}

/**
 * Where ranks share a pool, each holds only some real-space planes, and the
 * average screening -- the diagonal of the conjugate-gradient solve -- must be
 * the whole grid's. Averaged over each rank's own planes it differed from rank
 * to rank, the solves took different steps around the same collective
 * transforms, and the run hung. Two simulated ranks each hold half of an
 * inhomogeneous density, and a third holds none of it, as a rank can when there
 * are more ranks than planes; each one's reduction adds what the others hold.
 * All three must report the average over the whole density.
 */
TEST(LocalThomasFermiPreconditioner, MeanScreeningIsTheAverageOverTheWholeSharedGrid) {
    const Real q0_squared = 0.1;
    const std::vector<Real> density = {0.01, 0.02, 0.04, 0.08, 0.16, 0.32};
    const std::vector<Real> first_half(density.begin(), density.begin() + 3);
    const std::vector<Real> second_half(density.begin() + 3, density.end());

    auto screening_sum = [&](const std::vector<Real>& part) {
        Real sum = 0.0;
        for (Real value : part) {
            sum += std::max(thomas_fermi_screening(value), q0_squared);
        }
        return sum;
    };
    Real whole_grid_mean = screening_sum(density) / static_cast<Real>(density.size());
    ASSERT_GT(std::abs(screening_sum(first_half) / 3.0 - whole_grid_mean), 0.1)
        << "the halves must disagree for the test to mean anything";

    auto mean_on_rank =
        [&](const std::vector<Real>& local_part, Real others_sum, Real others_count) {
            LocalThomasFermiPreconditioner preconditioner(
                symmetric_g_squared(8),
                q0_squared,
                kFirstNonzero,
                [](const std::vector<Real>&, std::vector<Complex>&) {},
                [](const std::vector<Complex>&, std::vector<Real>&) {},
                [others_sum, others_count](std::vector<Real>& sum_and_count) {
                    sum_and_count[0] += others_sum;
                    sum_and_count[1] += others_count;
                });
            preconditioner.update(local_part);
            return preconditioner.mean_screening();
        };

    EXPECT_NEAR(mean_on_rank(first_half, screening_sum(second_half), 3.0), whole_grid_mean, 1e-14);
    EXPECT_NEAR(mean_on_rank(second_half, screening_sum(first_half), 3.0), whole_grid_mean, 1e-14);
    EXPECT_NEAR(mean_on_rank({}, screening_sum(density), 6.0), whole_grid_mean, 1e-14);
}
