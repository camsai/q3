/**
 * A2.3 gate — the compensation charge's defining property.
 *
 * ñ¹ + n̂ must carry the same multipole moments as n¹, for *arbitrary*
 * occupancies. That is not a convenience: it is what makes the two-grid split
 * of a PAW calculation valid at all, because it is what makes the
 * electrostatics outside the augmentation sphere identical between the
 * all-electron and pseudo descriptions.
 *
 * The check needs no reference code and no SCF — only a dataset and a random
 * ρ_ij — which is why it can gate this step before anything downstream exists.
 * Randomized occupancies matter: a moment identity can hold accidentally for
 * the diagonal, physically-plausible case and fail for the cross terms that a
 * real SCF actually produces.
 */

#include "io/pseudopotential/factory.hpp"
#include "io/pseudopotential/upf_reader.hpp"
#include "layer_4_computation/paw/one_center/densities.hpp"
#include "utils/one_center_occupancies.hpp"
#include <gtest/gtest.h>

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

using namespace q3;
using namespace q3::paw_one_center;
using q3::test_support::random_occupancies;

namespace {

std::unique_ptr<Pseudopotential> load_paw(const std::string& element) {
    return UPFReader::read(std::string(EXAMPLES_DIR) + "/data/pseudopotentials/paw/" + element +
                           ".upf");
}

} // namespace

TEST(OneCenterDensitiesTest, ChannelExpansionMatchesTheProjectorConvention) {
    auto pseudopotential = load_paw("Si");
    const auto& paw = dynamic_cast<const PawPseudopotential&>(*pseudopotential);
    auto expansion = expand_channels(paw);

    // Si: channels l = 0,0,1,1 → 1+1+3+3 = 8 expanded channels.
    EXPECT_EQ(expansion.size(), 8);
    int expected = 0;
    for (int channel = 0; channel < paw.number_of_projectors; channel++)
        expected += 2 * paw.partial_waves_pseudo[channel].angular_momentum + 1;
    EXPECT_EQ(expansion.size(), expected);

    for (int index = 0; index < expansion.size(); index++) {
        int radial = expansion.radial_channel[index];
        EXPECT_EQ(expansion.angular_momentum[index],
                  paw.partial_waves_pseudo[radial].angular_momentum);
        // lm index must fall inside its own l block.
        int angular_momentum = expansion.angular_momentum[index];
        EXPECT_GE(expansion.lm_index[index], angular_momentum * angular_momentum);
        EXPECT_LT(expansion.lm_index[index], (angular_momentum + 1) * (angular_momentum + 1));
    }
}

TEST(OneCenterDensitiesTest, CompensatedPseudoDensityMatchesAllElectronMoments) {
    /**
     * The gate. Every lm channel's moment must agree between n¹ and ñ¹ + n̂.
     *
     * Tolerance is relative to the largest all-electron moment rather than
     * absolute: the moments span orders of magnitude across channels, and an
     * absolute bound would be vacuous for the small ones and impossible for
     * the large.
     */
    for (const auto& element : {"Si", "Cu"}) {
        auto pseudopotential = load_paw(element);
        const auto& paw = dynamic_cast<const PawPseudopotential&>(*pseudopotential);
        auto expansion = expand_channels(paw);

        int maximum_angular_momentum = 0;
        for (const auto& wave : paw.partial_waves_pseudo)
            maximum_angular_momentum = std::max(maximum_angular_momentum, wave.angular_momentum);
        auto gaunt = generate_gaunt_coefficients(maximum_angular_momentum);

        for (unsigned seed : {1u, 7u, 42u}) {
            auto occupancies = random_occupancies(expansion.size(), seed);
            auto densities = build_one_center_densities(paw, expansion, occupancies, gaunt, 0);

            Real largest = 0.0;
            Real worst_difference = 0.0;
            for (int lm = 0; lm < densities.number_of_lm; lm++) {
                int angular_momentum = angular_momentum_of_lm(lm);
                Real all_electron = density_channel_moment(densities.all_electron[lm],
                                                           angular_momentum,
                                                           paw.radial_grid,
                                                           paw.radial_grid_derivative,
                                                           paw.mesh_size);
                Real pseudo = density_channel_moment(densities.pseudo[lm],
                                                     angular_momentum,
                                                     paw.radial_grid,
                                                     paw.radial_grid_derivative,
                                                     paw.mesh_size);
                largest = std::max(largest, std::abs(all_electron));
                worst_difference = std::max(worst_difference, std::abs(all_electron - pseudo));
            }
            ASSERT_GT(largest, 1e-6) << "test built a trivially zero density";
            EXPECT_LT(worst_difference, 1e-6 * largest)
                << element << " seed " << seed << " largest moment " << largest;
        }
    }
}

TEST(OneCenterDensitiesTest, DensitiesAreLinearInTheOccupancies) {
    /**
     * n¹ and ñ¹ are linear in ρ_ij by construction, and A2.5's
     * finite-difference gate leans on that: a nonlinearity introduced by, say,
     * accumulating into the wrong channel would make the derivative test
     * meaningless while still producing plausible densities.
     */
    auto pseudopotential = load_paw("Si");
    const auto& paw = dynamic_cast<const PawPseudopotential&>(*pseudopotential);
    auto expansion = expand_channels(paw);
    auto gaunt = generate_gaunt_coefficients(1);

    auto first = random_occupancies(expansion.size(), 3u);
    auto second = random_occupancies(expansion.size(), 11u);
    auto combined = OnSiteOccupancies::zeros(expansion.size());
    for (size_t index = 0; index < combined.values.size(); index++)
        combined.values[index] = 2.0 * first.values[index] - 3.0 * second.values[index];

    auto density_first = build_one_center_densities(paw, expansion, first, gaunt, 0);
    auto density_second = build_one_center_densities(paw, expansion, second, gaunt, 0);
    auto density_combined = build_one_center_densities(paw, expansion, combined, gaunt, 0);

    Real worst = 0.0;
    Real scale = 0.0;
    for (int lm = 0; lm < density_combined.number_of_lm; lm++) {
        for (int point = 0; point < paw.mesh_size; point += 37) {
            Real expected = 2.0 * density_first.all_electron[lm][point] -
                            3.0 * density_second.all_electron[lm][point];
            worst = std::max(worst, std::abs(density_combined.all_electron[lm][point] - expected));
            scale = std::max(scale, std::abs(expected));
        }
    }
    ASSERT_GT(scale, 0.0);
    EXPECT_LT(worst, 1e-12 * scale);
}

TEST(OneCenterDensitiesTest, ZeroOccupancyGivesZeroDensity) {
    auto pseudopotential = load_paw("Si");
    const auto& paw = dynamic_cast<const PawPseudopotential&>(*pseudopotential);
    auto expansion = expand_channels(paw);
    auto gaunt = generate_gaunt_coefficients(1);

    auto occupancies = OnSiteOccupancies::zeros(expansion.size());
    auto densities = build_one_center_densities(paw, expansion, occupancies, gaunt, 0);
    for (int lm = 0; lm < densities.number_of_lm; lm++) {
        for (int point = 0; point < paw.mesh_size; point += 101) {
            EXPECT_EQ(densities.all_electron[lm][point], 0.0);
            EXPECT_EQ(densities.pseudo[lm][point], 0.0);
        }
    }
}
