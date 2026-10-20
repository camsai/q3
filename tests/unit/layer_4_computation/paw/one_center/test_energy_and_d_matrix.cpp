/**
 * A2.4a and A2.5 gates.
 *
 * The centrepiece is the finite-difference identity: the one-center D_ij must
 * be the derivative of the one-center energy with respect to ρ_ij. QE's own
 * implementation guarantees that structurally (it builds D by perturbing a
 * single pair's occupancy), so asserting it here checks that our two
 * independently written routines agree about every sign, factor of two and
 * double-count *inside this module*.
 *
 * What it deliberately does not cover, per the A2.0 memo: the grid-side
 * ∫v_eff Q_ij term, and the split of terms between grid and sphere. Those are
 * fixed by the memo and verified only at the µRy gate.
 *
 * Two details make the gate meaningful rather than decorative:
 *   - perturbations are applied to symmetrized pairs, so the derivative
 *     convention is unambiguous in the very test meant to catch factor-of-two
 *     errors;
 *   - they are taken around physically sized occupancies, because LibXC clamps
 *     densities near 1e-10 and a central difference across that kink cannot
 *     reach the target tolerance.
 */

#include "io/pseudopotential/upf_reader.hpp"
#include "layer_4_computation/effective_potential/internal/exchange_correlation/libxc.hpp"
#include "layer_4_computation/paw/one_center/d_matrix.hpp"
#include "layer_4_computation/paw/one_center/energy.hpp"
#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <string>
#include <vector>

using namespace q3;
using namespace q3::paw_one_center;

namespace {

// LDA (Slater exchange + Perdew-Zunger correlation) — the A2.4a functional.
constexpr int lda_exchange_id = 1;    // XC_LDA_X
constexpr int lda_correlation_id = 9; // XC_LDA_C_PZ

std::unique_ptr<Pseudopotential> load_paw(const std::string& element) {
    return UPFReader::read(std::string(EXAMPLES_DIR) + "/data/pseudopotentials/paw/" + element +
                           ".upf");
}

/// Everything the one-center routines need for one dataset.
struct Fixture {
    std::unique_ptr<Pseudopotential> owner;
    const PawPseudopotential* paw{};
    ChannelExpansion expansion;
    GauntCoefficientTable gaunt;
    std::vector<AngularPoint> angular_grid;
    std::vector<std::vector<Real>> harmonics;
    XC_LibXC functional{lda_exchange_id, lda_correlation_id};

    explicit Fixture(const std::string& element) {
        owner = load_paw(element);
        paw = dynamic_cast<const PawPseudopotential*>(owner.get());
        expansion = expand_channels(*paw);
        int maximum = 0;
        for (const auto& wave : paw->partial_waves_pseudo)
            maximum = std::max(maximum, wave.angular_momentum);
        gaunt = generate_gaunt_coefficients(maximum);
        angular_grid = build_angular_grid(2 * maximum);
        harmonics = tabulate_harmonics(angular_grid, density_lm_count(*paw));
    }

    [[nodiscard]] OneCenterDensities densities_of(const OnSiteOccupancies& occupancies) const {
        return build_one_center_densities(*paw, expansion, occupancies, gaunt, 0);
    }

    [[nodiscard]] OneCenterEnergy energy_of(const OnSiteOccupancies& occupancies) const {
        auto densities = build_one_center_densities(*paw, expansion, occupancies, gaunt, 0);
        return compute_one_center_energy(*paw, densities, angular_grid, harmonics, functional);
    }
};

/**
 * Occupancies of physical size: the dataset's own reference occupations on the
 * diagonal, plus small symmetric off-diagonal mixing so the cross terms an SCF
 * produces are exercised too.
 */
OnSiteOccupancies physical_occupancies_for(const ChannelExpansion& expansion,
                                           const PawPseudopotential& paw) {
    auto occupancies = OnSiteOccupancies::zeros(expansion.size());
    for (int index = 0; index < expansion.size(); index++) {
        int radial = expansion.radial_channel[index];
        int degeneracy = 2 * expansion.angular_momentum[index] + 1;
        Real reference = radial < static_cast<int>(paw.channel_occupations.size())
                             ? paw.channel_occupations[radial]
                             : 0.0;
        occupancies.at(0, index, index) = reference / degeneracy + 0.05;
    }
    for (int first = 0; first < expansion.size(); first++) {
        for (int second = first + 1; second < expansion.size(); second++) {
            Real value = 0.01 * ((first + second) % 3 - 1);
            occupancies.at(0, first, second) = value;
            occupancies.at(0, second, first) = value;
        }
    }
    return occupancies;
}

OnSiteOccupancies physical_occupancies(const Fixture& fixture) {
    return physical_occupancies_for(fixture.expansion, *fixture.paw);
}

} // namespace

TEST(OneCenterEnergyTest, BranchesAreReportedSeparatelyAndAreFinite) {
    /**
     * The four numbers are the failure ladder's second rung, so they have to
     * exist individually rather than only as a total.
     */
    Fixture fixture("Si");
    auto energy = fixture.energy_of(physical_occupancies(fixture));

    EXPECT_TRUE(std::isfinite(energy.all_electron.hartree_energy));
    EXPECT_TRUE(std::isfinite(energy.all_electron.xc_energy));
    EXPECT_TRUE(std::isfinite(energy.pseudo.hartree_energy));
    EXPECT_TRUE(std::isfinite(energy.pseudo.xc_energy));
    EXPECT_TRUE(std::isfinite(energy.total()));

    // Hartree self-energy of a real density is positive in both branches.
    EXPECT_GT(energy.all_electron.hartree_energy, 0.0);
    EXPECT_GT(energy.pseudo.hartree_energy, 0.0);
    // Exchange-correlation is attractive.
    EXPECT_LT(energy.all_electron.xc_energy, 0.0);
    EXPECT_LT(energy.pseudo.xc_energy, 0.0);

    /**
     * The all-electron branch must dominate: it sees the sharp core-region
     * density the pseudo branch is built to smooth away. If these came out
     * comparable, the two branches would be using the same density.
     */
    EXPECT_GT(energy.all_electron.hartree_energy, energy.pseudo.hartree_energy);
}

TEST(OneCenterEnergyTest, ZeroOccupancyGivesZeroValenceEnergy) {
    /**
     * With no valence occupancy the Hartree term must vanish exactly. XC does
     * not, because the frozen cores remain — and that difference is itself the
     * check that the core densities are entering XC and not Hartree.
     */
    Fixture fixture("Si");
    auto occupancies = OnSiteOccupancies::zeros(fixture.expansion.size());
    auto energy = fixture.energy_of(occupancies);

    EXPECT_NEAR(energy.all_electron.hartree_energy, 0.0, 1e-12);
    EXPECT_NEAR(energy.pseudo.hartree_energy, 0.0, 1e-12);
    EXPECT_LT(energy.all_electron.xc_energy, 0.0);
}

TEST(OneCenterEnergyTest, PseudoBranchUsesThePseudizedCore) {
    /**
     * The memo's sharpest instruction: the all-electron branch takes the
     * frozen all-electron core, the pseudo branch the pseudized one. Using a
     * single core for both is invisible in every dataset-level identity and
     * shifts every core-corrected element.
     *
     * The datasets state genuinely different cores, so the two branches must
     * produce different XC energies even from identical valence densities —
     * which is what this asserts by feeding both branches the same channels.
     */
    Fixture fixture("Si");
    ASSERT_FALSE(fixture.paw->charge_core_all_electron.radial_values.empty());
    ASSERT_FALSE(fixture.paw->charge_core.radial_values.empty());

    auto occupancies = physical_occupancies(fixture);
    auto densities =
        build_one_center_densities(*fixture.paw, fixture.expansion, occupancies, fixture.gaunt, 0);
    // Same valence channels into both branches: any difference is the core.
    densities.pseudo = densities.all_electron;
    auto energy = compute_one_center_energy(
        *fixture.paw, densities, fixture.angular_grid, fixture.harmonics, fixture.functional);

    EXPECT_NEAR(energy.all_electron.hartree_energy, energy.pseudo.hartree_energy, 1e-10);
    EXPECT_GT(std::abs(energy.all_electron.xc_energy - energy.pseudo.xc_energy), 1e-6);
}

TEST(OneCenterDMatrixTest, IsTheDerivativeOfTheOneCenterEnergy) {
    /**
     * The strongest gate in the milestone: D_ij ≡ ∂(E¹ − Ẽ¹)/∂ρ_ij.
     *
     * Perturbing a symmetrized pair (δρ_ij = δρ_ji = ε/2 for i ≠ j) means the
     * central difference measures exactly what the matrix element claims to
     * be, with no factor-of-two ambiguity left in the test itself. Diagonal
     * elements take the full ε.
     */
    for (const auto& element : {"Si", "Cu"}) {
        Fixture fixture(element);
        auto base = physical_occupancies(fixture);
        auto energy = fixture.energy_of(base);
        auto d_matrix =
            compute_one_center_d_matrix(*fixture.paw, fixture.expansion, fixture.gaunt, energy);

        const int number_of_channels = fixture.expansion.size();
        /**
         * A step-size scan puts the optimum at 1e-3: larger steps show the
         * central difference's own truncation, smaller ones lose the signal to
         * roundoff in an energy of order 10 Ry. At 1e-3 the two routines agree
         * to ~1e-12 on the diagonal.
         */
        const Real step = 1e-3;

        Real worst_relative = 0.0;
        Real scale = 0.0;
        for (const auto& element_index : d_matrix)
            scale = std::max(scale, std::abs(element_index));
        ASSERT_GT(scale, 1e-6);

        /**
         * A representative spread of pairs rather than all of them: the full
         * matrix would mean thousands of energy evaluations.
         */
        for (int first = 0; first < number_of_channels; first += 3) {
            for (int second = first; second < number_of_channels; second += 4) {
                auto central_difference = [&](Real width) {
                    auto plus = base;
                    auto minus = base;
                    if (first == second) {
                        plus.at(0, first, first) += width;
                        minus.at(0, first, first) -= width;
                    } else {
                        plus.at(0, first, second) += 0.5 * width;
                        plus.at(0, second, first) += 0.5 * width;
                        minus.at(0, first, second) -= 0.5 * width;
                        minus.at(0, second, first) -= 0.5 * width;
                    }
                    return (fixture.energy_of(plus).total() - fixture.energy_of(minus).total()) /
                           (2.0 * width);
                };
                Real derivative = central_difference(step);
                Real predicted = d_matrix[static_cast<size_t>(first) * number_of_channels + second];

                worst_relative = std::max(worst_relative, std::abs(derivative - predicted) / scale);
            }
        }
        /**
         * Measured: 3.5e-6 (Si) and 2.5e-7 (Cu), relative to the largest
         * matrix element. The bound is not tighter for a physical reason
         * rather than a numerical one. A compensated PAW pseudo density
         * genuinely goes negative in places — QE prints "negative rho" on the
         * same datasets — and the XC wrapper evaluates the functional at
         * |n|, so the energy has a kink wherever a perturbation moves the
         * density across zero. Pairs that stay clear of it agree far better:
         * a step-size scan on the s-channel diagonal puts D_ij and the central
         * difference within 2.4e-12 of each other, which is the real
         * measurement of this identity. Richardson extrapolation was tried and
         * makes it worse, exactly as a non-smooth integrand predicts.
         */
        EXPECT_LT(worst_relative, 1e-5) << element << " (scale " << scale << ")";
    }
}

TEST(OneCenterDMatrixTest, IsSymmetric) {
    /**
     * ρ_ij is symmetric, so its conjugate D_ij must be too. An asymmetric
     * result would mean the channel pair was handled inconsistently between
     * the two branches.
     */
    Fixture fixture("Si");
    auto energy = fixture.energy_of(physical_occupancies(fixture));
    auto d_matrix =
        compute_one_center_d_matrix(*fixture.paw, fixture.expansion, fixture.gaunt, energy);

    int number_of_channels = fixture.expansion.size();
    Real scale = 0.0;
    for (Real value : d_matrix)
        scale = std::max(scale, std::abs(value));
    for (int first = 0; first < number_of_channels; first++) {
        for (int second = 0; second < number_of_channels; second++) {
            EXPECT_NEAR(d_matrix[first * number_of_channels + second],
                        d_matrix[second * number_of_channels + first],
                        1e-9 * scale);
        }
    }
}

// ── A2.4b: gradient-corrected (GGA) ─────────────────────────────────

namespace {

constexpr int pbe_exchange_id = 101;    // XC_GGA_X_PBE
constexpr int pbe_correlation_id = 130; // XC_GGA_C_PBE

/// The LDA fixture plus the harmonic gradients a GGA evaluation needs.
struct GradientFixture {
    std::unique_ptr<Pseudopotential> owner;
    const PawPseudopotential* paw{};
    ChannelExpansion expansion;
    GauntCoefficientTable gaunt;
    std::vector<AngularPoint> angular_grid;
    std::vector<std::vector<Real>> harmonics;
    std::vector<std::vector<std::array<Real, 3>>> harmonic_gradients;
    XC_LibXC functional{pbe_exchange_id, pbe_correlation_id};

    explicit GradientFixture(const std::string& element) {
        owner = load_paw(element);
        paw = dynamic_cast<const PawPseudopotential*>(owner.get());
        expansion = expand_channels(*paw);
        int maximum = 0;
        for (const auto& wave : paw->partial_waves_pseudo)
            maximum = std::max(maximum, wave.angular_momentum);
        gaunt = generate_gaunt_coefficients(maximum);
        angular_grid = build_angular_grid(2 * maximum);
        int number_of_lm = density_lm_count(*paw);
        harmonics = tabulate_harmonics(angular_grid, number_of_lm);
        harmonic_gradients = tabulate_harmonic_gradients(angular_grid, number_of_lm);
    }

    [[nodiscard]] OneCenterEnergy energy_of(const OnSiteOccupancies& occupancies) const {
        auto densities = build_one_center_densities(*paw, expansion, occupancies, gaunt, 0);
        return compute_one_center_energy(
            *paw, densities, angular_grid, harmonics, functional, &harmonic_gradients);
    }
};

} // namespace

TEST(OneCenterGradientTest, RadialDerivativeIsFourthOrderOnALogGrid) {
    /**
     * The stencil differentiates in index space, which is what keeps it valid
     * for the r^γ behaviour scalar-relativistic partial waves show near the
     * origin. Asserting the order rather than a tolerance is what distinguishes
     * a correct stencil from a merely plausible one.
     */
    auto worst_error_at = [](int points) {
        Real first_radius = 1e-4;
        Real last_radius = 20.0;
        Real spacing = std::log(last_radius / first_radius) / (points - 1);
        std::vector<Real> radius(points);
        std::vector<Real> measure(points);
        std::vector<Real> values(points);
        for (int index = 0; index < points; index++) {
            radius[index] = first_radius * std::exp(spacing * index);
            measure[index] = radius[index] * spacing;
            values[index] = std::exp(-0.7 * radius[index]) * radius[index];
        }
        auto derivative = radial_derivative(values, measure, points);
        Real worst = 0.0;
        for (int index = 0; index < points; index++) {
            Real expected = std::exp(-0.7 * radius[index]) * (1.0 - 0.7 * radius[index]);
            worst = std::max(worst, std::abs(derivative[index] - expected));
        }
        return worst;
    };
    Real coarse = worst_error_at(401);
    Real fine = worst_error_at(801);
    EXPECT_LT(coarse, 1e-5);
    EXPECT_GT(coarse / fine, 8.0) << "coarse " << coarse << " fine " << fine;
}

TEST(OneCenterGradientTest, DifferentiatesTheDatasetsOwnPartialWaves) {
    /**
     * The plan's specific warning: validate the endpoint scheme on real
     * partial waves, not only on smooth analytic functions, because their
     * non-integer power-law behaviour near the origin is exactly what a
     * polynomial-assuming stencil gets wrong.
     *
     * The check is self-contained: integrating the derivative back must
     * recover the function.
     */
    auto pseudopotential = load_paw("Cu");
    const auto& paw = dynamic_cast<const PawPseudopotential&>(*pseudopotential);
    for (int channel = 0; channel < paw.number_of_projectors; channel++) {
        const auto& wave = paw.partial_waves_all_electron[channel].radial_values;
        auto derivative = radial_derivative(wave, paw.radial_grid_derivative, paw.mesh_size);
        auto recovered = cumulative_integral(derivative, paw.radial_grid_derivative, paw.mesh_size);

        Real scale = 0.0;
        for (Real value : wave)
            scale = std::max(scale, std::abs(value));
        Real worst = 0.0;
        /**
         * Skip the first points, where the function itself is ~0 and the
         * integration constant dominates.
         */
        for (int point = 5; point < paw.mesh_size; point++)
            worst = std::max(worst, std::abs(recovered[point] - (wave[point] - wave[0])));
        EXPECT_LT(worst, 1e-4 * scale) << "channel " << channel;
    }
}

TEST(OneCenterGradientTest, GradientIsConsistentWithFiniteDifferences) {
    /**
     * The assembled ∇n must agree with differencing the density itself. This
     * catches a wrong 1/r on the tangential part or a mixed-up harmonic
     * gradient, neither of which changes the density and both of which would
     * quietly change every GGA energy.
     */
    GradientFixture fixture("Si");
    auto occupancies = physical_occupancies_for(fixture.expansion, *fixture.paw);
    auto densities =
        build_one_center_densities(*fixture.paw, fixture.expansion, occupancies, fixture.gaunt, 0);

    int number_of_lm = densities.number_of_lm;
    int mesh_size = fixture.paw->mesh_size;

    std::vector<std::vector<Real>> channel_values(number_of_lm, std::vector<Real>(mesh_size, 0.0));
    for (int lm = 0; lm < number_of_lm; lm++) {
        for (int point = 0; point < mesh_size; point++) {
            Real radius = fixture.paw->radial_grid[point];
            channel_values[lm][point] = densities.all_electron[lm][point] / (radius * radius);
        }
    }
    std::vector<std::vector<Real>> channel_derivatives(number_of_lm);
    for (int lm = 0; lm < number_of_lm; lm++)
        channel_derivatives[lm] =
            radial_derivative(channel_values[lm], fixture.paw->radial_grid_derivative, mesh_size);

    // Radial component at a mid-grid point, against a direct difference of n.
    int probe = mesh_size / 2;
    std::vector<Real> values_at(number_of_lm);
    std::vector<Real> derivatives_at(number_of_lm);
    for (int lm = 0; lm < number_of_lm; lm++) {
        values_at[lm] = channel_values[lm][probe];
        derivatives_at[lm] = channel_derivatives[lm][probe];
    }
    auto gradient = build_gradient_at_radius(values_at,
                                             derivatives_at,
                                             fixture.paw->radial_grid[probe],
                                             fixture.harmonics,
                                             fixture.harmonic_gradients);

    Real worst = 0.0;
    Real scale = 0.0;
    for (size_t angle = 0; angle < fixture.angular_grid.size(); angle += 7) {
        Real forward = 0.0;
        Real backward = 0.0;
        for (int lm = 0; lm < number_of_lm; lm++) {
            forward += channel_values[lm][probe + 1] * fixture.harmonics[angle][lm];
            backward += channel_values[lm][probe - 1] * fixture.harmonics[angle][lm];
        }
        Real numerical = (forward - backward) / (fixture.paw->radial_grid[probe + 1] -
                                                 fixture.paw->radial_grid[probe - 1]);
        worst = std::max(worst, std::abs(numerical - gradient.radial_component[angle]));
        scale = std::max(scale, std::abs(gradient.radial_component[angle]));
    }
    ASSERT_GT(scale, 0.0);
    EXPECT_LT(worst, 1e-3 * scale);
}

TEST(OneCenterDMatrixTest, GgaDerivativeIdentityHolds) {
    /**
     * The A2.5 gate repeated under PBE. This is the reason the gradient term
     * is integrated by parts rather than taken as a divergence: with the
     * by-parts form the identity holds by construction, so any failure here is
     * a real error in the assembly and not an artefact of differentiating a
     * numerically computed divergence.
     */
    for (const auto& element : {"Si", "Cu"}) {
        GradientFixture fixture(element);
        auto base = physical_occupancies_for(fixture.expansion, *fixture.paw);
        auto energy = fixture.energy_of(base);
        ASSERT_TRUE(energy.all_electron.has_gradient_correction());

        auto d_matrix =
            compute_one_center_d_matrix(*fixture.paw, fixture.expansion, fixture.gaunt, energy);
        int number_of_channels = fixture.expansion.size();
        const Real step = 1e-3;

        Real scale = 0.0;
        for (Real value : d_matrix)
            scale = std::max(scale, std::abs(value));
        ASSERT_GT(scale, 1e-6);

        Real worst_relative = 0.0;
        for (int first = 0; first < number_of_channels; first += 3) {
            for (int second = first; second < number_of_channels; second += 4) {
                auto plus = base;
                auto minus = base;
                if (first == second) {
                    plus.at(0, first, first) += step;
                    minus.at(0, first, first) -= step;
                } else {
                    plus.at(0, first, second) += 0.5 * step;
                    plus.at(0, second, first) += 0.5 * step;
                    minus.at(0, first, second) -= 0.5 * step;
                    minus.at(0, second, first) -= 0.5 * step;
                }
                Real derivative =
                    (fixture.energy_of(plus).total() - fixture.energy_of(minus).total()) /
                    (2.0 * step);
                Real predicted = d_matrix[static_cast<size_t>(first) * number_of_channels + second];
                worst_relative = std::max(worst_relative, std::abs(derivative - predicted) / scale);
            }
        }
        /**
         * Measured 3.3e-8 (Si) and 6.4e-8 (Cu) — comparable to the LDA gate,
         * which is the point: integrating the gradient term by parts keeps the
         * identity exact by construction, so PBE is no harder to satisfy than
         * LDA. Taking a numerical divergence instead would have left an error
         * here that no step size could remove.
         *
         * This gate earned its keep: it caught the tangential weight carrying
         * 1/r where it needed r — an error of r⁴ that showed up as a
         * plausible-looking 0.3% and would have been invisible in the energy
         * alone.
         */
        EXPECT_LT(worst_relative, 1e-6) << element << " (scale " << scale << ")";
    }
}

/**
 * The collapse gate for the spin-polarized one-center path.
 *
 * Splitting an unpolarized occupancy evenly between the two channels describes
 * exactly the same physical state, so the spin-polarized routines must return
 * the unpolarized numbers — not approximately, but to the precision the two
 * code paths can share. This is the cheapest check that catches a wrong
 * weighting, a dropped cross term in the gradient correction, or a core
 * density counted twice; every one of those leaves the moment plausible and
 * the energy wrong by tens of Rydberg.
 */
TEST(OneCenterEnergySpinTest, EvenlySplitOccupanciesReproduceTheUnpolarizedEnergy) {
    for (const char* element : {"Si", "Cu"}) {
        Fixture fixture(element);
        auto unpolarized = physical_occupancies(fixture);
        auto unpolarized_energy = fixture.energy_of(unpolarized);

        /**
         * A LibXC handle is initialized for a fixed spin mode, so the polarized
         * side needs its own. Reusing the unpolarized handle silently
         * misreads the interleaved (ρ↑, ρ↓) array LibXC expects.
         */
        XC_LibXC polarized_functional{lda_exchange_id, lda_correlation_id, 2};

        auto split = OnSiteOccupancies::zeros(fixture.expansion.size(), 2);
        for (int spin = 0; spin < 2; spin++)
            for (int first = 0; first < fixture.expansion.size(); first++)
                for (int second = 0; second < fixture.expansion.size(); second++)
                    split.at(spin, first, second) = 0.5 * unpolarized.at(0, first, second);

        std::vector<OneCenterDensities> densities_by_spin;
        for (int spin = 0; spin < 2; spin++)
            densities_by_spin.push_back(build_one_center_densities(
                *fixture.paw, fixture.expansion, split, fixture.gaunt, spin));

        /**
         * Both option settings, because they take different branches. The
         * atomic-referenced convention subtracts the core term, and that subtraction is
         * where the spin path first went wrong: it was evaluated through the
         * unpolarized routines while holding a polarized handle, which for
         * nickel is a 106 Ry error and invisible with the option off.
         */
        for (bool subtract_core : {false, true}) {
            OneCenterOptions options;
            options.subtract_core_exchange_correlation = subtract_core;
            auto reference = compute_one_center_energy(*fixture.paw,
                                                       fixture.densities_of(unpolarized),
                                                       fixture.angular_grid,
                                                       fixture.harmonics,
                                                       fixture.functional,
                                                       nullptr,
                                                       options);
            auto spin_energies = compute_one_center_energy_spin(*fixture.paw,
                                                                densities_by_spin,
                                                                fixture.angular_grid,
                                                                fixture.harmonics,
                                                                polarized_functional,
                                                                nullptr,
                                                                options);

            Real spin_total = spin_energies[0].total() + spin_energies[1].total();
            EXPECT_NEAR(spin_total, reference.total(), 1e-6)
                << element << " (subtract_core=" << subtract_core
                << "): spin-split one-center total does not collapse to the unpolarized one";
            EXPECT_NEAR(spin_energies[0].all_electron.hartree_energy,
                        reference.all_electron.hartree_energy,
                        1e-8)
                << element << ": Hartree must come from the total density alone";
            EXPECT_NEAR(
                spin_energies[0].all_electron.xc_energy, reference.all_electron.xc_energy, 1e-6)
                << element << " (subtract_core=" << subtract_core
                << "): unpolarized XC must be reproduced by the spin-polarized functional";
        }
        (void)unpolarized_energy;
    }
}

/**
 * The same collapse gate for the gradient-corrected path.
 *
 * A GGA has a failure mode the LDA cannot reach: its dependence on ∇n_σ runs
 * through the cross term σ_↑↓ as well as σ_σσ, so the field conjugate to
 * ∇δn_σ is 2 v_σσ ∇n_σ + v_↑↓ ∇n_σ'. Dropping the cross term leaves the
 * energy plausible and the by-parts weights wrong, which shows up as a
 * one-center total that is off by tens of Rydberg while the magnetic moment
 * still looks reasonable.
 */
TEST(OneCenterEnergySpinTest, EvenlySplitOccupanciesReproduceTheGradientCorrectedEnergy) {
    constexpr int pbe_exchange_id = 101;    // XC_GGA_X_PBE
    constexpr int pbe_correlation_id = 130; // XC_GGA_C_PBE

    for (const char* element : {"Si", "Cu"}) {
        Fixture fixture(element);
        XC_LibXC unpolarized_functional{pbe_exchange_id, pbe_correlation_id, 1};
        XC_LibXC polarized_functional{pbe_exchange_id, pbe_correlation_id, 2};

        int number_of_lm = density_lm_count(*fixture.paw);
        auto harmonic_gradients = tabulate_harmonic_gradients(fixture.angular_grid, number_of_lm);

        auto unpolarized = physical_occupancies(fixture);
        auto unpolarized_densities = build_one_center_densities(
            *fixture.paw, fixture.expansion, unpolarized, fixture.gaunt, 0);
        auto unpolarized_energy = compute_one_center_energy(*fixture.paw,
                                                            unpolarized_densities,
                                                            fixture.angular_grid,
                                                            fixture.harmonics,
                                                            unpolarized_functional,
                                                            &harmonic_gradients);

        auto split = OnSiteOccupancies::zeros(fixture.expansion.size(), 2);
        for (int spin = 0; spin < 2; spin++)
            for (int first = 0; first < fixture.expansion.size(); first++)
                for (int second = 0; second < fixture.expansion.size(); second++)
                    split.at(spin, first, second) = 0.5 * unpolarized.at(0, first, second);

        std::vector<OneCenterDensities> densities_by_spin;
        for (int spin = 0; spin < 2; spin++)
            densities_by_spin.push_back(build_one_center_densities(
                *fixture.paw, fixture.expansion, split, fixture.gaunt, spin));
        for (bool subtract_core : {false, true}) {
            OneCenterOptions options;
            options.subtract_core_exchange_correlation = subtract_core;
            auto reference = compute_one_center_energy(*fixture.paw,
                                                       unpolarized_densities,
                                                       fixture.angular_grid,
                                                       fixture.harmonics,
                                                       unpolarized_functional,
                                                       &harmonic_gradients,
                                                       options);
            auto spin_energies = compute_one_center_energy_spin(*fixture.paw,
                                                                densities_by_spin,
                                                                fixture.angular_grid,
                                                                fixture.harmonics,
                                                                polarized_functional,
                                                                &harmonic_gradients,
                                                                options);

            Real spin_total = spin_energies[0].total() + spin_energies[1].total();
            EXPECT_NEAR(spin_total, reference.total(), 1e-6)
                << element << " (subtract_core=" << subtract_core
                << "): spin-split GGA one-center total does not collapse";
            EXPECT_NEAR(
                spin_energies[0].all_electron.xc_energy, reference.all_electron.xc_energy, 1e-6)
                << element << " (subtract_core=" << subtract_core
                << "): spin-polarized GGA exchange-correlation does not collapse";
        }
        (void)unpolarized_energy;
    }
}
