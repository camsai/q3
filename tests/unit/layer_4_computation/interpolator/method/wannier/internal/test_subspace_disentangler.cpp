#include <gtest/gtest.h>

#include <cmath>
#include <complex>
#include <stdexcept>
#include <utility>
#include <vector>

#include <string>

#include "layer_0_core/math/linear_algebra/dense_matrix.hpp"
#include "layer_0_core/types/init.hpp"
#include "layer_1_structure/reciprocal_space/kpoint_stencil.hpp"
#include "layer_4_computation/interpolator/method/wannier/diagnostic.hpp"
#include "layer_4_computation/interpolator/method/wannier/diagnostic_error.hpp"
#include "layer_4_computation/interpolator/method/wannier/internal/band_classification.hpp"
#include "layer_4_computation/interpolator/method/wannier/internal/bloch_overlaps.hpp"
#include "layer_4_computation/interpolator/method/wannier/internal/subspace_disentangler.hpp"

using namespace q3;

namespace {

constexpr int kpoint_count = 4;

/**
 * A ring of four k-points with two neighbour directions, forward and back,
 * each of weight one half. Every k-point carries `bands` states, and the
 * overlap between neighbours is diagonal with the entries in `smoothness`:
 * an entry of one is a band that matches its neighbour perfectly, zero is one
 * that does not match at all. Real physics is never this clean, but it makes
 * the answer knowable, which is the point of a unit test.
 *
 * The blocks and the neighbour table are kept as the nested vectors a record
 * stores, because a test that wants a malformed input mutates them; the
 * stencil and the matrices are built from them where they are used, so that
 * the refusal happens inside the expression the test wraps.
 */
struct Ring {
    int bands = 0;
    std::vector<Real> weights = {0.5, 0.5};
    std::vector<std::vector<int>> neighbors;
    std::vector<std::vector<std::vector<Complex>>> overlaps;

    Ring(int band_count, const std::vector<Real>& smoothness) : bands(band_count) {
        for (int kpoint = 0; kpoint < kpoint_count; kpoint++) {
            neighbors.push_back(
                {(kpoint + 1) % kpoint_count, (kpoint + kpoint_count - 1) % kpoint_count});
            std::vector<Complex> diagonal(static_cast<std::size_t>(bands) * bands,
                                          Complex(0.0, 0.0));
            for (int band = 0; band < bands; band++) {
                diagonal[static_cast<std::size_t>(band) * bands + band] =
                    Complex(smoothness[static_cast<std::size_t>(band)], 0.0);
            }
            overlaps.push_back({diagonal, diagonal});
        }
    }

    /// No b-vectors: the gauge-invariant spread needs weights and neighbours only.
    [[nodiscard]] KPointStencil stencil() const {
        return KPointStencil::from_table(neighbors, weights);
    }
    [[nodiscard]] BlochOverlaps matrices() const {
        return BlochOverlaps::square(overlaps, bands);
    }
};

/// Every band a candidate, nothing frozen or discarded.
std::vector<KPointBandClassification> all_candidates(int bands) {
    std::vector<KPointBandClassification> classification;
    for (int kpoint = 0; kpoint < kpoint_count; kpoint++) {
        KPointBandClassification entry;
        entry.roles.assign(static_cast<std::size_t>(bands), BandRole::Candidate);
        for (int band = 0; band < bands; band++) {
            entry.kept_bands.push_back(band);
        }
        classification.push_back(entry);
    }
    return classification;
}

/// A subspace holding the single band `band`, in a window of `bands` states.
std::vector<DenseMatrix<Complex>> guess_of_band(int bands, int band) {
    std::vector<DenseMatrix<Complex>> subspaces;
    for (int kpoint = 0; kpoint < kpoint_count; kpoint++) {
        DenseMatrix<Complex> subspace(bands, 1, Complex(0.0, 0.0));
        subspace(band, 0) = Complex(1.0, 0.0);
        subspaces.push_back(subspace);
    }
    return subspaces;
}

/// A one-dimensional guess spread equally over every band, normalised.
std::vector<DenseMatrix<Complex>> guess_of_every_band(int bands) {
    const Real amplitude = 1.0 / std::sqrt(static_cast<Real>(bands));
    return std::vector<DenseMatrix<Complex>>(
        kpoint_count, DenseMatrix<Complex>(bands, 1, Complex(amplitude, 0.0)));
}

/// The iteration on a ring, everything built inside so a broken ring is refused here.
DisentanglementResult disentangle(const Ring& ring,
                                  const std::vector<KPointBandClassification>& classification,
                                  int number_of_wannier_functions,
                                  std::vector<DenseMatrix<Complex>> guess,
                                  const DisentanglementSettings& settings = {}) {
    const KPointStencil stencil = ring.stencil();
    return SubspaceDisentangler(stencil, classification, number_of_wannier_functions, settings)
        .run(ring.matrices(), std::move(guess));
}

/**
 * Omega_I of a given set of subspaces on a Ring, computed here from the
 * definition rather than by the class under test. The diagonal overlap makes
 * it short enough to be obviously right, which is what an independent check
 * has to be to be worth having.
 */
Real gauge_invariant_spread(const std::vector<DenseMatrix<Complex>>& subspaces,
                            const Ring& ring,
                            const std::vector<Real>& smoothness,
                            int number_of_wannier_functions) {
    Real total = 0.0;
    for (int kpoint = 0; kpoint < kpoint_count; kpoint++) {
        for (std::size_t direction = 0; direction < ring.weights.size(); direction++) {
            const DenseMatrix<Complex>& here = subspaces[static_cast<std::size_t>(kpoint)];
            const DenseMatrix<Complex>& there =
                subspaces[static_cast<std::size_t>(ring.neighbors[kpoint][direction])];
            Real retained = 0.0;
            for (int left = 0; left < number_of_wannier_functions; left++) {
                for (int right = 0; right < number_of_wannier_functions; right++) {
                    Complex element(0.0, 0.0);
                    for (int band = 0; band < ring.bands; band++) {
                        element += std::conj(here(band, left)) *
                                   smoothness[static_cast<std::size_t>(band)] * there(band, right);
                    }
                    retained += std::norm(element);
                }
            }
            total += ring.weights[direction] *
                     (static_cast<Real>(number_of_wannier_functions) - retained);
        }
    }
    return total / static_cast<Real>(kpoint_count);
}

/// The band each k-point's one-dimensional subspace actually selected.
int selected_band(const DenseMatrix<Complex>& subspace) {
    int best = 0;
    Real largest = 0.0;
    for (int row = 0; row < subspace.rows(); row++) {
        if (std::norm(subspace(row, 0)) > largest) {
            largest = std::norm(subspace(row, 0));
            best = row;
        }
    }
    return best;
}

} // namespace

TEST(SubspaceDisentangler, APerfectlySmoothBandIsAFixedPointWithZeroSpread) {
    /**
     * Band 0 matches its neighbours exactly and band 1 does not match at all.
     * Started on band 0 the iteration must stay there, and the
     * gauge-invariant spread must be zero: a subspace that is identical at
     * every k-point has nothing left to spread.
     */
    Ring ring(2, {1.0, 0.0});
    const DisentanglementResult result =
        disentangle(ring, all_candidates(2), 1, guess_of_band(2, 0));

    EXPECT_NEAR(result.gauge_invariant_spread_bohr_squared, 0.0, 1e-12);
    for (const DenseMatrix<Complex>& subspace : result.subspaces) {
        EXPECT_EQ(selected_band(subspace), 0);
    }
}

TEST(SubspaceDisentangler, WithADiagonalOverlapEveryBandIsAFixedPointAndTheGuessDecides) {
    /**
     * This pins the iteration's real character rather than a flattering
     * version of it. It is a LOCAL optimiser: with a diagonal overlap there is
     * no coupling to rotate through, so each band reproduces itself and the
     * starting guess decides which one you get. The spread records which basin
     * you landed in -- zero for the band that matches its neighbours exactly,
     * and 1 - 0.6^2 per unit weight for the one that does not.
     *
     * This is exactly why the initial guess is the projection onto the trial
     * orbitals rather than anything arbitrary: the physics of what the model
     * is supposed to describe enters there, and nowhere else.
     */
    Ring ring(2, {1.0, 0.6});

    const DisentanglementResult from_smooth =
        disentangle(ring, all_candidates(2), 1, guess_of_band(2, 0));
    const DisentanglementResult from_rough =
        disentangle(ring, all_candidates(2), 1, guess_of_band(2, 1));

    for (const DenseMatrix<Complex>& subspace : from_smooth.subspaces) {
        EXPECT_EQ(selected_band(subspace), 0);
    }
    for (const DenseMatrix<Complex>& subspace : from_rough.subspaces) {
        EXPECT_EQ(selected_band(subspace), 1);
    }
    EXPECT_NEAR(from_smooth.gauge_invariant_spread_bohr_squared, 0.0, 1e-12);
    EXPECT_NEAR(from_rough.gauge_invariant_spread_bohr_squared, 1.0 - 0.36, 1e-12);
    EXPECT_LT(from_smooth.gauge_invariant_spread_bohr_squared,
              from_rough.gauge_invariant_spread_bohr_squared)
        << "the objective still ranks the two basins correctly";
}

TEST(SubspaceDisentangler, FromAMixedGuessItFindsTheSmoothestBandAndLowersTheSpread) {
    /**
     * The test the rest of this file does not make: that the iteration
     * OPTIMISES. Every other case here is a fixed point, an invariant or a
     * refusal, and all of them would still pass if the update did nothing.
     *
     * Three bands whose overlaps with their neighbours are 0.3, 0.6 and 0.9,
     * and a guess spread equally over all three, which is a fixed point of
     * nothing. Band 2 is the smoothest, so it is the one-dimensional subspace
     * of lowest Omega_I, and the iteration has to walk there from a guess that
     * starts two thirds elsewhere:
     *
     *   guess    Omega_I = 1 - |(0.3 + 0.6 + 0.9)/3|^2 = 0.64
     *   answer   Omega_I = 1 - 0.9^2                   = 0.19
     */
    const std::vector<Real> smoothness = {0.3, 0.6, 0.9};
    Ring ring(3, smoothness);
    const DisentanglementResult result =
        disentangle(ring, all_candidates(3), 1, guess_of_every_band(3));

    for (const DenseMatrix<Complex>& subspace : result.subspaces) {
        EXPECT_EQ(selected_band(subspace), 2) << "the smoothest band is the one to keep";
    }
    EXPECT_NEAR(result.gauge_invariant_spread_bohr_squared, 1.0 - 0.81, 1e-9);
    EXPECT_LT(result.gauge_invariant_spread_bohr_squared, 0.64)
        << "the iteration did not improve on its own starting guess";
}

TEST(SubspaceDisentangler, TheReportedSpreadBelongsToTheSubspacesReturned) {
    /**
     * Omega_I is accumulated while Z is built, so it describes the subspaces
     * the iteration started that step from, not the ones it finished with.
     * Within the convergence tolerance those are the same subspace; they are
     * not the same NUMBER, and the number is what gets printed beside the
     * model and written into the results file. Recomputed here from the
     * definition, against the subspaces actually handed back.
     */
    const std::vector<Real> smoothness = {0.3, 0.6, 0.9};
    Ring ring(3, smoothness);
    const DisentanglementResult result =
        disentangle(ring, all_candidates(3), 1, guess_of_every_band(3));

    EXPECT_NEAR(result.gauge_invariant_spread_bohr_squared,
                gauge_invariant_spread(result.subspaces, ring, smoothness, 1),
                1e-12);
}

TEST(SubspaceDisentangler, AnOverlapBlockOfTheWrongSizeIsRefusedRatherThanReadPast) {
    /**
     * Every overlap block is indexed as number_of_captured_bands squared. A
     * block of any other size is not a wrong answer, it is a read past the end
     * of the vector -- so it has to be caught before the first slice, not
     * inferred from the result.
     */
    Ring ring(2, {1.0, 0.0});
    ring.overlaps[1][0].pop_back();
    EXPECT_THROW(disentangle(ring, all_candidates(2), 1, guess_of_band(2, 0)),
                 std::invalid_argument);
}

TEST(SubspaceDisentangler, ANeighborOutsideTheKPointSetIsRefused) {
    /// The neighbour table indexes the k-point arrays directly; the same rule applies.
    Ring ring(2, {1.0, 0.0});
    ring.neighbors[2][1] = kpoint_count;
    EXPECT_THROW(disentangle(ring, all_candidates(2), 1, guess_of_band(2, 0)),
                 std::invalid_argument);
}

TEST(SubspaceDisentangler, OverlapsThatAreNotBetweenOrthonormalStatesAreRefused) {
    /**
     * Found by writing a bad test: a hand-built overlap [[0.9, 0.3], [0.3, 0.9]]
     * has singular values 1.2 and 0.6, so it is not the overlap of any two
     * orthonormal sets of Bloch states. Fed to the iteration it produced a
     * gauge-invariant spread of -0.44 bohr^2 -- an impossible number, since
     * the positive term of Omega_I is bounded above by the subspace dimension.
     *
     * The optimisation cannot detect this from any one matrix cheaply, but the
     * sign of the result gives it away for free, so that is where the check
     * lives. A corrupt record fails loudly rather than returning a model whose
     * spread is negative.
     */
    Ring ring(2, {1.0, 0.0});
    const std::vector<Complex> too_large = {
        Complex(0.9, 0.0), Complex(0.3, 0.0), Complex(0.3, 0.0), Complex(0.9, 0.0)};
    for (int kpoint = 0; kpoint < kpoint_count; kpoint++) {
        ring.overlaps[static_cast<std::size_t>(kpoint)] = {too_large, too_large};
    }

    try {
        const DisentanglementResult result =
            disentangle(ring, all_candidates(2), 1, guess_of_band(2, 0));
        FAIL() << "a negative Omega_I is impossible for overlaps of orthonormal states";
    } catch (const WannierDiagnosticError& error) {
        EXPECT_EQ(error.diagnostic(), WannierDiagnostic::overlaps_not_orthonormal);
        EXPECT_NE(std::string(error.what()).find("came out negative"), std::string::npos)
            << error.what();
    }
}

TEST(SubspaceDisentangler, ADiscardedBandNeverEntersTheSubspace) {
    /**
     * The band the optimiser would most like is discarded by classification.
     * If discarded states were left in the working basis it would be selected
     * anyway, and the classification would be decorative.
     */
    Ring ring(3, {1.0, 0.2, 0.1});
    std::vector<KPointBandClassification> classification;
    for (int kpoint = 0; kpoint < kpoint_count; kpoint++) {
        KPointBandClassification entry;
        entry.roles = {BandRole::Discarded, BandRole::Candidate, BandRole::Candidate};
        entry.kept_bands = {1, 2};
        classification.push_back(entry);
    }

    std::vector<DenseMatrix<Complex>> guess(
        kpoint_count,
        DenseMatrix<Complex>(2, 1, std::vector<Complex>{Complex(1.0, 0.0), Complex(0.0, 0.0)}));

    const DisentanglementResult result = disentangle(ring, classification, 1, guess);

    for (const DenseMatrix<Complex>& subspace : result.subspaces) {
        ASSERT_EQ(subspace.rows(), 2) << "the working basis is the two kept bands";
        EXPECT_EQ(selected_band(subspace), 0)
            << "position 0 of the kept list is band 1, the smoothest state still available";
    }
}

TEST(SubspaceDisentangler, AFrozenBandSurvivesUntouched) {
    /**
     * Band 1 is frozen although band 0 is smoother. Freezing means kept
     * exactly, so with one Wannier function the answer is band 1 and the
     * optimiser never gets a say.
     */
    Ring ring(2, {1.0, 0.3});
    std::vector<KPointBandClassification> classification;
    for (int kpoint = 0; kpoint < kpoint_count; kpoint++) {
        KPointBandClassification entry;
        entry.roles = {BandRole::Candidate, BandRole::Frozen};
        entry.kept_bands = {0, 1};
        entry.frozen_positions = {1};
        classification.push_back(entry);
    }

    const DisentanglementResult result = disentangle(ring, classification, 1, guess_of_band(2, 1));
    for (const DenseMatrix<Complex>& subspace : result.subspaces) {
        EXPECT_EQ(selected_band(subspace), 1);
        EXPECT_NEAR(std::abs(subspace(1, 0)), 1.0, 1e-12);
        EXPECT_NEAR(std::abs(subspace(0, 0)), 0.0, 1e-12);
    }
}

TEST(SubspaceDisentangler, TheSubspaceStaysOrthonormal) {
    Ring ring(4, {1.0, 0.8, 0.5, 0.2});
    std::vector<DenseMatrix<Complex>> guess;
    for (int kpoint = 0; kpoint < kpoint_count; kpoint++) {
        DenseMatrix<Complex> subspace(4, 2, Complex(0.0, 0.0));
        subspace(0, 0) = Complex(1.0, 0.0); // first column: band 0
        subspace(1, 1) = Complex(1.0, 0.0); // second column: band 1
        guess.push_back(subspace);
    }

    const DisentanglementResult result = disentangle(ring, all_candidates(4), 2, guess);

    for (const DenseMatrix<Complex>& subspace : result.subspaces) {
        for (int left = 0; left < 2; left++) {
            for (int right = 0; right < 2; right++) {
                Complex inner(0.0, 0.0);
                for (int band = 0; band < 4; band++) {
                    inner += std::conj(subspace(band, left)) * subspace(band, right);
                }
                const Real expected = left == right ? 1.0 : 0.0;
                EXPECT_NEAR(inner.real(), expected, 1e-10);
                EXPECT_NEAR(inner.imag(), 0.0, 1e-10);
            }
        }
    }
}

TEST(SubspaceDisentangler, NonConvergenceThrowsRatherThanReturningTheSubspace) {
    Ring ring(3, {1.0, 0.9, 0.8});
    DisentanglementSettings settings;
    settings.maximum_iterations = 2;
    settings.convergence_window = 5; // cannot be satisfied within two iterations
    try {
        const DisentanglementResult result =
            disentangle(ring, all_candidates(3), 1, guess_of_band(3, 2), settings);
        FAIL() << "two iterations cannot satisfy a five-step window";
    } catch (const WannierDiagnosticError& error) {
        /// The code, and the spreads as numbers in the prose's own six-decimal form.
        EXPECT_EQ(error.diagnostic(), WannierDiagnostic::disentanglement_unconverged);
        const std::string message = error.what();
        EXPECT_NE(message.find("did not converge in 2 iterations"), std::string::npos) << message;
        ASSERT_EQ(error.spreads_bohr_squared().size(), 2u);
        EXPECT_NE(message.find(error.spreads_text()), std::string::npos)
            << message << " should name " << error.spreads_text();
    }
}

/// Settings that describe no iteration are refused when the disentangler is built.
TEST(SubspaceDisentangler, SettingsThatDescribeNoIterationAreRefused) {
    Ring ring(2, {1.0, 0.5});

    DisentanglementSettings no_budget;
    no_budget.maximum_iterations = 0;
    EXPECT_THROW(disentangle(ring, all_candidates(2), 1, guess_of_band(2, 0), no_budget),
                 std::invalid_argument);

    DisentanglementSettings no_mixing;
    no_mixing.damping = 0.0;
    EXPECT_THROW(disentangle(ring, all_candidates(2), 1, guess_of_band(2, 0), no_mixing),
                 std::invalid_argument);

    DisentanglementSettings over_mixing;
    over_mixing.damping = 1.5;
    EXPECT_THROW(disentangle(ring, all_candidates(2), 1, guess_of_band(2, 0), over_mixing),
                 std::invalid_argument);
}

TEST(SubspaceDisentangler, AMisshapenInitialSubspaceIsRefused) {
    Ring ring(2, {1.0, 0.5});
    /// One row where two bands are kept, one column where two functions are wanted.
    std::vector<DenseMatrix<Complex>> guess(kpoint_count,
                                            DenseMatrix<Complex>(1, 1, Complex(1.0, 0.0)));
    EXPECT_THROW(disentangle(ring, all_candidates(2), 2, guess), std::invalid_argument);
}
