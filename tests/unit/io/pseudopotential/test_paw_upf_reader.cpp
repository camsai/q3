/**
 * PAW UPF v2 reader and dataset self-consistency tests.
 *
 * The self-consistency checks are the substance here: a PAW dataset
 * overdetermines itself, so parsing it correctly can be verified from the file
 * alone. The tolerances below were established by evaluating each identity on
 * these two datasets, not assumed — Cu is the looser of the pair because it
 * carries six channels up to l = 2.
 */

#include "io/pseudopotential/loader.hpp"
#include "io/pseudopotential/upf_reader.hpp"
#include "layer_3_representation/pseudopotential/paw/init.hpp"
#include "layer_3_representation/pseudopotential/paw/validation.hpp"
#include "layer_3_representation/pseudopotential/set.hpp"
#include <gtest/gtest.h>

#include "utils/temporary_path.hpp"

#include <cstdio>
#include <fstream>
#include <iterator>
#include <string>

using namespace q3;

namespace {

std::string paw_fixture(const std::string& element) {
    return std::string(EXAMPLES_DIR) + "/data/pseudopotentials/paw/" + element + ".upf";
}

const PawPseudopotential& as_paw(const Pseudopotential& pseudopotential) {
    return dynamic_cast<const PawPseudopotential&>(pseudopotential);
}

} // namespace

// ── Reader: structure and metadata ──────────────────────────────────

TEST(PawUPFReaderTest, SiliconReadsAsPawWithOneCenterData) {
    auto pseudopotential = UPFReader::read(paw_fixture("Si"));

    ASSERT_EQ(pseudopotential->pseudopotential_type_enum(),
              PseudopotentialType::projector_augmented_wave);
    EXPECT_TRUE(pseudopotential->has_augmentation_charges());
    EXPECT_NEAR(pseudopotential->valence_charge, 4.0, 1e-9);
    EXPECT_EQ(pseudopotential->number_of_projectors, 4);
    EXPECT_EQ(pseudopotential->mesh_size, 2342);

    const auto& paw = as_paw(*pseudopotential);
    ASSERT_EQ(static_cast<int>(paw.partial_waves_all_electron.size()), 4);
    ASSERT_EQ(static_cast<int>(paw.partial_waves_pseudo.size()), 4);

    // Channel angular momenta: two s channels then two p channels.
    EXPECT_EQ(paw.partial_waves_all_electron[0].angular_momentum, 0);
    EXPECT_EQ(paw.partial_waves_all_electron[1].angular_momentum, 0);
    EXPECT_EQ(paw.partial_waves_all_electron[2].angular_momentum, 1);
    EXPECT_EQ(paw.partial_waves_all_electron[3].angular_momentum, 1);

    // Every partial wave is tabulated on the full radial mesh.
    for (int channel_index = 0; channel_index < 4; channel_index++) {
        EXPECT_EQ(
            static_cast<int>(paw.partial_waves_all_electron[channel_index].radial_values.size()),
            paw.mesh_size);
        EXPECT_EQ(static_cast<int>(paw.partial_waves_pseudo[channel_index].radial_values.size()),
                  paw.mesh_size);
    }

    EXPECT_EQ(paw.augmentation_shape, "BESSEL");
    EXPECT_NEAR(paw.augmentation_cutoff_radius, 1.9094499, 1e-6);
    EXPECT_EQ(paw.augmentation_cutoff_radius_index, 1932);
    EXPECT_EQ(paw.maximum_angular_momentum_augmentation, 2);
    EXPECT_EQ(paw.number_of_augmentation_angular_momenta, 3);

    // Reference occupations of the generating atom: 3s² 3p².
    ASSERT_EQ(static_cast<int>(paw.channel_occupations.size()), 4);
    EXPECT_NEAR(paw.channel_occupations[0], 2.0, 1e-12);
    EXPECT_NEAR(paw.channel_occupations[1], 0.0, 1e-12);
    EXPECT_NEAR(paw.channel_occupations[2], 2.0, 1e-12);
    EXPECT_NEAR(paw.channel_occupations[3], 0.0, 1e-12);

    EXPECT_EQ(static_cast<int>(paw.charge_core_all_electron.radial_values.size()), paw.mesh_size);
    EXPECT_EQ(static_cast<int>(paw.potential_local_all_electron.radial_values.size()),
              paw.mesh_size);
}

TEST(PawUPFReaderTest, CopperReadsSixChannelsUpToDAngularMomentum) {
    auto pseudopotential = UPFReader::read(paw_fixture("Cu"));

    ASSERT_EQ(pseudopotential->pseudopotential_type_enum(),
              PseudopotentialType::projector_augmented_wave);
    EXPECT_NEAR(pseudopotential->valence_charge, 19.0, 1e-9);
    EXPECT_EQ(pseudopotential->number_of_projectors, 6);

    const auto& paw = as_paw(*pseudopotential);
    EXPECT_EQ(paw.augmentation_shape, "SINC**2");
    EXPECT_EQ(paw.maximum_angular_momentum_augmentation, 4);
    EXPECT_EQ(paw.number_of_augmentation_angular_momenta, 5);
    EXPECT_EQ(paw.partial_waves_all_electron[4].angular_momentum, 2);
    EXPECT_EQ(paw.partial_waves_all_electron[5].angular_momentum, 2);

    /**
     * These datasets disagree with themselves about the mesh: the PP_HEADER
     * mesh_size attribute reads 922 for Cu (2000 for Si) while PP_MESH — the
     * grid the data is actually tabulated on — reads 2342. PP_MESH wins, and it
     * has to: the augmentation cutoff sits at index 1943, so trusting the header
     * would truncate the mesh inside the augmentation region and silently
     * corrupt every one-center integral.
     */
    EXPECT_EQ(paw.mesh_size, 2342);
    EXPECT_EQ(static_cast<int>(paw.radial_grid.size()), 2342);
    EXPECT_EQ(static_cast<int>(paw.radial_grid_derivative.size()), 2342);
    EXPECT_EQ(static_cast<int>(paw.partial_waves_all_electron[0].radial_values.size()), 2342);
    EXPECT_LT(paw.augmentation_cutoff_radius_index, paw.mesh_size);
}

TEST(PawUPFReaderTest, OlderAugmentationAngularMomentumSpellingIsAccepted) {
    /**
     * Older PSLibrary PAW files use l_max_aug where newer writers use
     * lmax_aug. Missing that alias silently left the PAW angular expansion at
     * zero before the loader could evaluate the dataset's support contract.
     */
    std::ifstream input_stream(paw_fixture("Cu"));
    ASSERT_TRUE(input_stream.is_open());
    std::string content((std::istreambuf_iterator<char>(input_stream)),
                        std::istreambuf_iterator<char>());
    auto attribute_position = content.find("lmax_aug");
    ASSERT_NE(attribute_position, std::string::npos);
    content.replace(attribute_position, std::string("lmax_aug").size(), "l_max_aug");

    std::string temporary_path = test_support::make_temporary_path(".UPF");
    {
        std::ofstream output_stream(temporary_path);
        output_stream << content;
    }
    auto pseudopotential = UPFReader::read(temporary_path);
    EXPECT_EQ(as_paw(*pseudopotential).maximum_angular_momentum_augmentation, 4);
    std::remove(temporary_path.c_str());
}

TEST(PawUPFReaderTest, MultipoleMomentsAreAngularMomentumMajor) {
    /**
     * The multipole array is stored L-major. Reading it in the wrong order
     * would still produce a correctly sized table, so pin the layout by the one
     * block whose values are independently known: the L = 0 block is the
     * augmentation charge matrix, which the dataset also stores in PP_Q.
     */
    auto pseudopotential = UPFReader::read(paw_fixture("Si"));
    const auto& paw = as_paw(*pseudopotential);

    ASSERT_EQ(static_cast<int>(paw.multipole_moments.size()), 3);
    ASSERT_EQ(static_cast<int>(paw.multipole_moments[0].size()), 4);
    ASSERT_EQ(static_cast<int>(paw.multipole_moments[0][0].size()), 4);

    EXPECT_NEAR(paw.multipole_moments[0][0][0], -0.049021272184, 1e-9);
    EXPECT_NEAR(paw.multipole_moments[0][0][1], 0.726604974, 1e-9);
    EXPECT_NEAR(paw.multipole_moments[0][1][1], -9.18672608, 1e-8);

    // Channels forbidden by the triangle condition carry no compensation charge.
    EXPECT_NEAR(paw.multipole_moments[0][0][2], 0.0, 1e-14);
    EXPECT_NEAR(paw.multipole_moments[0][0][3], 0.0, 1e-14);
}

// ── Dataset self-consistency ────────────────────────────────────────

TEST(PawDatasetConsistencyTest, MultipoleMomentsMatchPartialWaveIntegrals) {
    /**
     * The tightest check available from the file alone: every triangle-allowed
     * moment must reproduce ∫ (φ_i φ_j − φ̃_i φ̃_j) r^L dr computed from the
     * partial waves. Measured worst deviations: Si 2.2e-09, Cu 1.7e-07.
     */
    for (const auto& element : {"Si", "Cu"}) {
        auto pseudopotential = UPFReader::read(paw_fixture(element));
        Real deviation = paw_validation::worst_multipole_moment_deviation(as_paw(*pseudopotential));
        EXPECT_LT(deviation, 1e-6) << "element " << element;
    }
}

TEST(PawDatasetConsistencyTest, MonopoleBlockEqualsAugmentationCharges) {
    /**
     * The L = 0 multipole block and the augmentation charge matrix are the same
     * quantity reached two ways; the datasets store them bit-identically.
     */
    for (const auto& element : {"Si", "Cu"}) {
        auto pseudopotential = UPFReader::read(paw_fixture(element));
        Real deviation =
            paw_validation::worst_augmentation_charge_deviation(as_paw(*pseudopotential));
        EXPECT_LT(deviation, 1e-12) << "element " << element;
    }
}

TEST(PawDatasetConsistencyTest, ProjectorsAreDualToPseudoPartialWaves) {
    /**
     * ⟨p̃_i|φ̃_j⟩ = δ_ij is the defining property of the PAW projectors, and the
     * check that would catch projectors paired to the wrong partial waves.
     * Measured worst deviations: Si 1.7e-06, Cu 7.2e-04 (six channels, tabulated
     * on a finite mesh).
     */
    auto silicon = UPFReader::read(paw_fixture("Si"));
    EXPECT_LT(paw_validation::worst_projector_duality_deviation(as_paw(*silicon)), 1e-5);

    auto copper = UPFReader::read(paw_fixture("Cu"));
    EXPECT_LT(paw_validation::worst_projector_duality_deviation(as_paw(*copper)), 2e-3);
}

TEST(PawDatasetConsistencyTest, AllElectronCoreChargeCountsFrozenElectrons) {
    /**
     * 4π ∫ ρ_core r² dr is the number of electrons the dataset froze into the
     * core. Both fixtures freeze ten: Si keeps 4 of 14 as valence, Cu keeps 19
     * of 29.
     */
    auto silicon = UPFReader::read(paw_fixture("Si"));
    EXPECT_NEAR(paw_validation::integrated_all_electron_core_charge(as_paw(*silicon)), 10.0, 1e-5);

    auto copper = UPFReader::read(paw_fixture("Cu"));
    EXPECT_NEAR(paw_validation::integrated_all_electron_core_charge(as_paw(*copper)), 10.0, 1e-5);
}

TEST(PawDatasetConsistencyTest, AllElectronPotentialRecoversNuclearCharge) {
    /**
     * V_AE(r) → −2Z/r toward the origin in Rydberg units, so the innermost grid
     * point recovers Z. The innermost point is not the origin, so the recovered
     * value falls a few hundredths short — enough to identify the element,
     * which is what this check is for.
     */
    auto silicon = UPFReader::read(paw_fixture("Si"));
    EXPECT_NEAR(
        paw_validation::nuclear_charge_from_all_electron_potential(as_paw(*silicon)), 14.0, 0.05);

    auto copper = UPFReader::read(paw_fixture("Cu"));
    EXPECT_NEAR(
        paw_validation::nuclear_charge_from_all_electron_potential(as_paw(*copper)), 29.0, 0.05);
}

TEST(PawDatasetConsistencyTest, AugmentationChannelSelectionRuleMatchesStoredZeros) {
    /**
     * Cross-check the triangle condition the validation module applies against
     * the zeros the dataset actually stores: every channel the rule forbids must
     * hold an exact zero, and Si must have exactly the 20 allowed channels its
     * s,s,p,p configuration implies.
     */
    auto pseudopotential = UPFReader::read(paw_fixture("Si"));
    const auto& paw = as_paw(*pseudopotential);

    int allowed_channel_count = 0;
    for (int angular_momentum = 0; angular_momentum < 3; angular_momentum++) {
        for (int channel_1 = 0; channel_1 < 4; channel_1++) {
            for (int channel_2 = 0; channel_2 < 4; channel_2++) {
                bool allowed = paw_validation::is_allowed_augmentation_channel(
                    paw.partial_waves_all_electron[channel_1].angular_momentum,
                    paw.partial_waves_all_electron[channel_2].angular_momentum,
                    angular_momentum);
                if (allowed) {
                    allowed_channel_count++;
                } else {
                    EXPECT_EQ(paw.multipole_moments[angular_momentum][channel_1][channel_2], 0.0)
                        << "L=" << angular_momentum << " i=" << channel_1 << " j=" << channel_2;
                }
            }
        }
    }
    EXPECT_EQ(allowed_channel_count, 20);
}

// ── PAW must reach the PAW pathway, not the ultrasoft one ───────────

TEST(PawCalculationGuardTest, LoaderAcceptsPawDatasets) {
    /**
     * The loader used to refuse PAW outright, because the downstream factories
     * dispatched on `== ultrasoft` and would have computed a PAW dataset as
     * norm-conserving. Both halves of that changed together, so this asserts
     * both: the dataset loads, and it loads *as PAW* — the type survives the
     * round trip into the set, which is what the factories now dispatch on.
     */
    std::vector<AtomSpecies> species(1);
    species[0].pseudopotential_file = paw_fixture("Si");

    PseudopotentialSet set = load_pseudopotentials(species, "");
    ASSERT_EQ(set.size(), 1);
    EXPECT_EQ(set[0].pseudopotential_type_enum(), PseudopotentialType::projector_augmented_wave);
    EXPECT_TRUE(set.has_pseudopotential_type(PseudopotentialType::projector_augmented_wave));

    /**
     * And it answers yes to the augmentation capability, which is what routes
     * it to the Q(G) tables and the S-operator. A PAW dataset that answered no
     * would converge to a wrong answer rather than fail.
     */
    EXPECT_TRUE(set.has_augmentation_charges());
    EXPECT_FALSE(set.has_pseudopotential_type(PseudopotentialType::ultrasoft));
}

TEST(PawCalculationGuardTest, LoaderRefusesUnappliedCoreEnergyConvention) {
    std::ifstream input_stream(paw_fixture("Cu"));
    ASSERT_TRUE(input_stream.is_open());
    std::string content((std::istreambuf_iterator<char>(input_stream)),
                        std::istreambuf_iterator<char>());
    std::string paw_tag = "<PP_PAW paw_data_format=\"2\"";
    auto paw_position = content.find(paw_tag);
    ASSERT_NE(paw_position, std::string::npos);
    content.insert(paw_position + paw_tag.size(), " core_energy=\"-3096.879768\"");

    std::string temporary_path = test_support::make_temporary_path(".UPF");
    {
        std::ofstream output_stream(temporary_path);
        output_stream << content;
    }

    std::vector<AtomSpecies> species(1);
    species[0].pseudopotential_file = temporary_path;
    try {
        load_pseudopotentials(species, "");
        FAIL() << "a PAW core_energy convention without implementation was accepted";
    } catch (const std::runtime_error& error) {
        std::string message = error.what();
        EXPECT_NE(message.find("core_energy"), std::string::npos) << message;
        EXPECT_NE(message.find("silently wrong absolute energy"), std::string::npos) << message;
    }
    std::remove(temporary_path.c_str());
}

TEST(PawCalculationGuardTest, LoaderStillAcceptsNormConservingDatasets) {
    std::vector<AtomSpecies> species(1);
    species[0].pseudopotential_file =
        std::string(EXAMPLES_DIR) + "/data/pseudopotentials/nc/Si.upf";

    auto pseudopotential_set = load_pseudopotentials(species, "");
    EXPECT_EQ(pseudopotential_set.size(), 1);
    EXPECT_EQ(pseudopotential_set[0].pseudopotential_type_enum(),
              PseudopotentialType::norm_conserving);
}

TEST(PawUPFReaderTest, UPFv1PawStillRejectedAtReadTime) {
    /**
     * Only v2 PAW ingestion exists; a v1 PAW header must still fail as a
     * recognized-but-unsupported type rather than be parsed down the NC path.
     */
    std::string temporary_path = test_support::make_temporary_path(".UPF");
    {
        std::ofstream output_stream(temporary_path);
        output_stream << "<PP_INFO>\n</PP_INFO>\n"
                      << "<PP_HEADER>\n"
                      << "   0                   Version Number\n"
                      << "  Si                   Element\n"
                      << "   PAW                 Ultrasoft pseudopotential\n";
    }
    EXPECT_THROW(UPFReader::read(temporary_path), UnsupportedPseudopotentialTypeError);
    std::remove(temporary_path.c_str());
}
