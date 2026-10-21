/**
 * A3 gates — the seams the PAW SCF iterator rests on.
 *
 * The iterator itself is exercised end to end by running an SCF; what cannot
 * be caught that way is a *silent* error, and there are two candidates, both
 * of which produce a converged calculation with a wrong answer:
 *
 *   1. **Dispatch by type where capability was meant.** A PAW dataset answers
 *      no to `== ultrasoft` and yes to `has_augmentation_charges()`. Every
 *      factory that asked the first question was routing PAW down the
 *      norm-conserving path and dropping its augmentation charges.
 *   2. **Channel-order disagreement.** ρ_ij is consumed directly — no conversion
 *      — so the plane-wave side's expanded projector order and the one-center
 *      module's have to be the same order. If they are not, every occupancy
 *      lands in the wrong element while the matrix stays symmetric and
 *      plausible.
 *
 * Both are stated here as predicates rather than left to the SCF to reveal.
 */

#include "io/pseudopotential/factory.hpp"
#include "io/pseudopotential/upf_reader.hpp"
#include "layer_3_representation/basis/planewave/pseudopotential/internal/projector_mapping_table.hpp"
#include "layer_4_computation/effective_potential/internal/exchange_correlation/libxc.hpp"
#include "layer_4_computation/paw/one_center/scf_terms.hpp"
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

using namespace q3;
using namespace q3::paw_one_center;

namespace {

std::unique_ptr<Pseudopotential> load_paw(const std::string& element) {
    return UPFReader::read(std::string(EXAMPLES_DIR) + "/data/pseudopotentials/paw/" + element +
                           ".upf");
}

std::unique_ptr<Pseudopotential> load_ultrasoft() {
    return UPFReader::read(std::string(EXAMPLES_DIR) + "/data/pseudopotentials/us/Si_PBE_USPP.UPF");
}

} // namespace

TEST(PawScfTermsTest, PawAnswersTheAugmentationCapabilityButNotTheUltrasoftType) {
    /**
     * The distinction the factories now turn on. Both halves matter: yes to
     * the capability is what routes PAW to the Q(G) tables and the
     * S-operator; no to the type is what keeps the iterator selection —
     * which checks PAW first — from handing it to the ultrasoft iterator and
     * dropping every one-center term.
     */
    PseudopotentialSet set;
    set.push_back(load_paw("Si"));

    EXPECT_TRUE(set.has_augmentation_charges());
    EXPECT_TRUE(set.has_pseudopotential_type(PseudopotentialType::projector_augmented_wave));
    EXPECT_FALSE(set.has_pseudopotential_type(PseudopotentialType::ultrasoft));

    PseudopotentialSet ultrasoft_set;
    ultrasoft_set.push_back(load_ultrasoft());
    EXPECT_TRUE(ultrasoft_set.has_augmentation_charges());
    EXPECT_FALSE(
        ultrasoft_set.has_pseudopotential_type(PseudopotentialType::projector_augmented_wave));
}

TEST(PawScfTermsTest, ChannelExpansionMatchesTheProjectorMappingTableOrder) {
    /**
     * The load-bearing one. `expand_channels` walks the partial waves;
     * `ProjectorMappingTable` walks the β projectors. They are built from
     * different arrays by different code, and ρ_ij is indexed by the second
     * while the one-center engine indexes by the first.
     *
     * Compared element by element rather than by count: two expansions can
     * agree on length and still pair (l, m) differently, which is exactly the
     * failure that stays invisible.
     */
    for (const auto& element : {"Si", "Cu"}) {
        auto pseudopotential = load_paw(element);
        const auto& paw = dynamic_cast<const PawPseudopotential&>(*pseudopotential);
        ASSERT_TRUE(channel_expansion_matches_projectors(paw)) << element;

        auto expansion = expand_channels(paw);

        // Rebuild the plane-wave side's ordering the way the mapping table does.
        std::vector<int> mapping_radial_channel;
        std::vector<int> mapping_lm_index;
        for (int projector = 0; projector < static_cast<int>(paw.projectors.size()); projector++) {
            int angular_momentum = paw.projectors[projector].angular_momentum;
            for (int magnetic_projection : harmonic_order_sequence(angular_momentum)) {
                mapping_radial_channel.push_back(projector);
                mapping_lm_index.push_back(
                    harmonic_combined_degree_order_index(angular_momentum, magnetic_projection));
            }
        }

        ASSERT_EQ(expansion.size(), static_cast<int>(mapping_lm_index.size())) << element;
        for (int channel = 0; channel < expansion.size(); channel++) {
            EXPECT_EQ(expansion.radial_channel[channel], mapping_radial_channel[channel])
                << element << " channel " << channel;
            EXPECT_EQ(expansion.lm_index[channel], mapping_lm_index[channel])
                << element << " channel " << channel;
        }
    }
}

TEST(PawScfTermsTest, DIsTheDerivativeOfTheOneCenterEnergyThroughTheDriver) {
    /**
     * The A2.5 identity re-asserted at the level the SCF actually calls, not
     * only at the level the one-center module's own gate covers. The driver
     * adds channel bookkeeping, a ρ_ij stride, and a per-species context
     * between the SCF and the engine, and an indexing slip in any of those
     * would break D ≡ ∂E/∂ρ_ij while leaving both quantities individually
     * plausible.
     *
     * Checked by finite differences on a single symmetric occupancy pair,
     * through `update()` — the same call the iterator makes each step.
     */
    Crystal crystal;
    crystal.set_from_lattice_type(2, 10.263048); // FCC silicon
    crystal.add_species(AtomSpecies{"Si", 28.08, 4.0, "Si.upf"});
    crystal.add_atom({0, {0.0, 0.0, 0.0}});
    crystal.compute_derived_quantities();

    PseudopotentialSet set;
    set.push_back(load_paw("Si"));
    XC_LibXC functional{1, 9}; // LDA: Slater exchange + Perdew-Zunger correlation

    PawOneCenterTerms terms;
    terms.build(crystal, set, functional);
    ASSERT_TRUE(terms.is_active());

    int number_of_channels = terms.number_of_channels(0);
    ASSERT_GT(number_of_channels, 0);

    Matrix<Real> occupancies;
    occupancies.assign(1, number_of_channels * number_of_channels, 0.0);
    for (int channel = 0; channel < number_of_channels; channel++)
        occupancies[0][channel * number_of_channels + channel] = 0.35;

    const int first = 0;
    const int second = number_of_channels - 1;
    const Real step = 1e-5;

    terms.update(crystal, occupancies, number_of_channels);
    Real analytic = terms.d_correction(0)[first * number_of_channels + second];

    auto energy_at = [&](Real offset) {
        Matrix<Real> perturbed = occupancies;
        perturbed[0][first * number_of_channels + second] += offset;
        perturbed[0][second * number_of_channels + first] += offset;
        terms.update(crystal, perturbed, number_of_channels);
        return terms.energy();
    };

    /**
     * The perturbation moves two symmetric elements, so the difference
     * quotient is 2·∂E/∂ρ_ij — the same factor the module's own gate accounts
     * for, restated here so the two cannot silently disagree.
     */
    Real numerical = (energy_at(step) - energy_at(-step)) / (2.0 * step) / 2.0;

    EXPECT_NEAR(analytic, numerical, 1e-6 * std::max(1.0, std::abs(numerical)))
        << "D_ij is not ∂E/∂ρ_ij through the SCF driver";
}
