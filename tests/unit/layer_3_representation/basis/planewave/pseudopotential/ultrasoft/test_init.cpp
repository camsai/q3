#include <gtest/gtest.h>

#include <algorithm>
#include <complex>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "io/pseudopotential/upf_reader.hpp"
#include "layer_0_core/types/matrix.hpp"
#include "layer_1_structure/crystal/crystal.hpp"
#include "layer_3_representation/basis/planewave/init.hpp"
#include "layer_3_representation/basis/planewave/pseudopotential/ultrasoft/init.hpp"
#include "layer_3_representation/pseudopotential/set.hpp"
#include "layer_4_computation/iterator/kohn_sham/planewave/execution/serial.hpp"

using namespace q3;

namespace {

execution::PlanewaveSerialExecutionPolicy serial_policy;

/**
 * Zincblende SiC with an ultrasoft silicon and a norm-conserving carbon.
 *
 * A cell that mixes the two kinds is what these tests are about, and it is not
 * an exotic configuration: combining pseudopotentials of different kinds is
 * ordinary practice when one element needs a particular dataset (a core-hole
 * capable ultrasoft, say) and the others do not.
 *
 * `UltrasoftPlanewaveBasis` is instantiated whenever *any* species needs
 * augmentation, so in a cell like this one it is asked about the
 * norm-conserving species too. Every method on it therefore has to answer for a
 * species that has no augmentation data at all.
 */
struct MixedSpeciesCell {
    static constexpr int ultrasoft_species = 0;
    static constexpr int norm_conserving_species = 1;

    Crystal crystal_structure;
    PlanewaveBasis planewave_basis;
    UltrasoftPlanewaveBasis ultrasoft_basis;

    MixedSpeciesCell() {
        auto silicon = UPFReader::read(std::string(EXAMPLES_DIR) +
                                       "/data/pseudopotentials/us/Si_PBE_USPP.UPF");
        auto carbon =
            UPFReader::read(std::string(EXAMPLES_DIR) + "/data/pseudopotentials/nc/C.upf");

        AtomSpecies silicon_species{"Si", 28.085, silicon->valence_charge, ""};
        AtomSpecies carbon_species{"C", 12.011, carbon->valence_charge, ""};

        crystal_structure.set_from_lattice_type(2, 8.2368);
        crystal_structure.add_species(silicon_species);
        crystal_structure.add_species(carbon_species);
        crystal_structure.add_atom({ultrasoft_species, {0.0, 0.0, 0.0}});
        crystal_structure.add_atom({norm_conserving_species, {0.25, 0.25, 0.25}});
        crystal_structure.compute_derived_quantities();

        PseudopotentialSet pseudopotential_set;
        pseudopotential_set.push_back(std::move(silicon));
        pseudopotential_set.push_back(std::move(carbon));

        planewave_basis.generate(serial_policy, crystal_structure, 20.0, 160.0);
        ultrasoft_basis.init(crystal_structure, planewave_basis, pseudopotential_set);
    }
};

} // namespace

TEST(UltrasoftPlanewaveBasisMixedSpecies, SetHoldsOneAugmentedAndOnePlainSpecies) {
    /**
     * The premise the rest of the file rests on. If the fixture ever stops
     * being a genuinely mixed cell — say the carbon file is replaced with an
     * ultrasoft one — the tests below would still pass while testing nothing.
     */
    MixedSpeciesCell cell;
    const auto& pseudopotential_set = cell.ultrasoft_basis.pseudopotential_set();

    ASSERT_EQ(pseudopotential_set.size(), 2);
    EXPECT_TRUE(
        pseudopotential_set[MixedSpeciesCell::ultrasoft_species].has_augmentation_charges());
    EXPECT_FALSE(
        pseudopotential_set[MixedSpeciesCell::norm_conserving_species].has_augmentation_charges());
}

TEST(UltrasoftPlanewaveBasisMixedSpecies, AugmentationOverlapIsZeroForNormConservingSpecies) {
    /**
     * q_ij = ∫Q_ij(r)dr is zero for a species with no augmentation charges, and
     * asking for it must simply say so.
     *
     * Before the per-species gate this call reached `PseudopotentialSet::as_ultrasoft`,
     * a `dynamic_cast` guarded only by an `assert`: a Debug build aborted on the
     * assertion and a Release build — the one users run — dereferenced null. It is
     * reached from the wavefunction initializer's overlap on the very first SCF
     * step, so a mixed cell could not start a calculation at all.
     */
    MixedSpeciesCell cell;
    int number_of_channels = cell.ultrasoft_basis.number_of_projector_channels(
        MixedSpeciesCell::norm_conserving_species);
    ASSERT_GT(number_of_channels, 0);

    int number_of_betas =
        cell.ultrasoft_basis.number_of_projectors(MixedSpeciesCell::norm_conserving_species);
    for (int beta_a = 0; beta_a < number_of_betas; ++beta_a) {
        for (int beta_b = 0; beta_b < number_of_betas; ++beta_b) {
            EXPECT_DOUBLE_EQ(cell.ultrasoft_basis.augmentation_overlap_integral(
                                 MixedSpeciesCell::norm_conserving_species, beta_a, beta_b),
                             0.0)
                << "q_ij for norm-conserving beta pair (" << beta_a << ", " << beta_b << ")";
        }
    }
}

TEST(UltrasoftPlanewaveBasisMixedSpecies, AugmentationOverlapSurvivesForUltrasoftSpecies) {
    /**
     * The gate must not swallow the ultrasoft species along with the
     * norm-conserving one: returning zero everywhere would also stop the crash,
     * and would silently turn every ultrasoft calculation into a wrong one.
     */
    MixedSpeciesCell cell;
    int number_of_betas =
        cell.ultrasoft_basis.number_of_projectors(MixedSpeciesCell::ultrasoft_species);
    ASSERT_GT(number_of_betas, 0);

    bool any_nonzero = false;
    for (int beta_a = 0; beta_a < number_of_betas; ++beta_a) {
        for (int beta_b = 0; beta_b < number_of_betas; ++beta_b) {
            Real overlap = cell.ultrasoft_basis.augmentation_overlap_integral(
                MixedSpeciesCell::ultrasoft_species, beta_a, beta_b);
            // Symmetric in (i, j), as the underlying q_ij matrix is.
            EXPECT_DOUBLE_EQ(overlap,
                             cell.ultrasoft_basis.augmentation_overlap_integral(
                                 MixedSpeciesCell::ultrasoft_species, beta_b, beta_a));
            if (overlap != 0.0)
                any_nonzero = true;
        }
    }
    EXPECT_TRUE(any_nonzero) << "an ultrasoft species must keep a nonzero augmentation overlap";
}

TEST(UltrasoftPlanewaveBasisMixedSpecies, QFunctionTableIsBuiltOnlyForTheAugmentedSpecies) {
    /**
     * `precompute_q_functions_g` skips a species with no augmentation charges,
     * which is why `add_augmentation_charges` cannot index that species' table.
     * Stated here so the coupling between the two is visible rather than
     * inferred from a crash.
     */
    MixedSpeciesCell cell;
    EXPECT_TRUE(cell.ultrasoft_basis.has_q_functions_g(MixedSpeciesCell::ultrasoft_species));
    EXPECT_FALSE(cell.ultrasoft_basis.has_q_functions_g(MixedSpeciesCell::norm_conserving_species));
}

TEST(UltrasoftPlanewaveBasisMixedSpecies, NormConservingAtomAddsNoAugmentationCharge) {
    /**
     * ρ_ij is accumulated over *every* atom, so a norm-conserving atom in a
     * mixed cell arrives here with nonzero ⟨ψ|β_i⟩⟨β_j|ψ⟩. It has no Q(G) table,
     * and the atom loop used to read one anyway: `q_function_g_data` took the
     * address of an element of an empty vector and the accumulation then read
     * `number_of_g_vectors` complex values through it. That is out of bounds on
     * every SCF iteration, and it corrupts the density quietly rather than
     * failing outright.
     *
     * The check is a comparison rather than a crash test for that reason. The
     * density built with ρ_ij on both atoms must equal the density built with
     * ρ_ij on the ultrasoft atom alone: whatever the norm-conserving atom's
     * coefficients are, they contribute nothing. Reading past the end of an
     * empty vector gives whatever the allocator left there, so the two runs
     * agreeing exactly is what says the entry was skipped and not merely
     * small.
     */
    MixedSpeciesCell cell;

    int number_of_g_vectors = cell.planewave_basis.number_of_g_vectors();
    int maximum_channels = std::max(
        cell.ultrasoft_basis.number_of_projector_channels(MixedSpeciesCell::ultrasoft_species),
        cell.ultrasoft_basis.number_of_projector_channels(
            MixedSpeciesCell::norm_conserving_species));

    Matrix<Real> both_atoms(2, static_cast<size_t>(maximum_channels) * maximum_channels, 0.25);
    Matrix<Real> ultrasoft_atom_only(
        2, static_cast<size_t>(maximum_channels) * maximum_channels, 0.25);
    for (size_t channel_pair = 0; channel_pair < ultrasoft_atom_only.cols(); ++channel_pair) {
        ultrasoft_atom_only(MixedSpeciesCell::norm_conserving_species, channel_pair) = 0.0;
    }

    std::vector<Complex> density_from_both(number_of_g_vectors, Complex(0.0, 0.0));
    std::vector<Complex> density_from_ultrasoft_only(number_of_g_vectors, Complex(0.0, 0.0));

    cell.ultrasoft_basis.add_augmentation_charges(
        cell.crystal_structure, both_atoms, density_from_both);
    cell.ultrasoft_basis.add_augmentation_charges(
        cell.crystal_structure, ultrasoft_atom_only, density_from_ultrasoft_only);

    ASSERT_EQ(density_from_both.size(), density_from_ultrasoft_only.size());
    for (int g_index = 0; g_index < number_of_g_vectors; ++g_index) {
        EXPECT_DOUBLE_EQ(density_from_both[g_index].real(),
                         density_from_ultrasoft_only[g_index].real())
            << "G-vector " << g_index;
        EXPECT_DOUBLE_EQ(density_from_both[g_index].imag(),
                         density_from_ultrasoft_only[g_index].imag())
            << "G-vector " << g_index;
    }

    /**
     * And the ultrasoft atom really did contribute, so the equality above is
     * not two empty densities matching.
     */
    bool any_nonzero = false;
    for (const auto& value : density_from_both) {
        if (value != Complex(0.0, 0.0)) {
            any_nonzero = true;
            break;
        }
    }
    EXPECT_TRUE(any_nonzero);
}
