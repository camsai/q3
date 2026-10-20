#include <gtest/gtest.h>

#include <cstdio>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "io/pseudopotential/loader.hpp"
#include "utils/temporary_path.hpp"

using namespace q3;

// EXAMPLES_DIR is defined as a compile-time macro by CMakeLists.txt

namespace {

const std::string pseudopotential_directory = std::string(EXAMPLES_DIR) + "/data/pseudopotentials";

AtomSpecies species_with(const std::string& symbol, const std::string& relative_path) {
    AtomSpecies species;
    species.symbol = symbol;
    species.mass = 0.0;
    species.valence_charge = 0.0;
    species.pseudopotential_file = pseudopotential_directory + "/" + relative_path;
    return species;
}

/// The message functional_from_pseudopotentials throws with, or empty if it returns.
std::string refusal_message(const std::vector<AtomSpecies>& species) {
    try {
        functional_from_pseudopotentials(species, "");
    } catch (const std::runtime_error& error) {
        return error.what();
    }
    return {};
}

/**
 * A copy of the bundled silicon dataset whose header states `label` instead
 * of "PBE", written to a temporary file the caller removes.
 */
std::string write_silicon_dataset_stating(const std::string& label) {
    std::ifstream source(pseudopotential_directory + "/nc/Si.upf");
    std::stringstream contents;
    contents << source.rdbuf();
    std::string text = contents.str();
    const std::string attribute = "functional=\"PBE\"";
    const auto position = text.find(attribute);
    if (position == std::string::npos)
        throw std::runtime_error("nc/Si.upf no longer states functional=\"PBE\"");
    text.replace(position, attribute.size(), "functional=\"" + label + "\"");

    const std::string path = test_support::make_temporary_path(".upf");
    std::ofstream output(path);
    output << text;
    return path;
}

AtomSpecies species_from_file(const std::string& symbol, const std::string& path) {
    AtomSpecies species = species_with(symbol, "");
    species.pseudopotential_file = path;
    return species;
}

} // namespace

// ─── The functional the datasets state ──────────────────────────────────────

/**
 * Every bundled dataset, across both UPF versions and all three kinds, and
 * both the short-name and the component spellings of its label.
 */
TEST(FunctionalFromPseudopotentials, ReadsEveryBundledDataset) {
    const std::vector<std::pair<std::string, std::string>> expected = {
        {"nc/Si.upf", "pbe"},                 // v2, functional="PBE"
        {"nc/Si.pz-vbc.UPF", "pz"},           // v1, "SLA PZ NOGX NOGC PZ"
        {"us/Si_PBE_USPP.UPF", "pbe"},        // v1, "SLA PW PBX PBC PBE"
        {"us/cu_pbe_v1.2.uspp.F.UPF", "pbe"}, // v1, "SLA PW PBE PBE PBE"
        {"paw/Si.upf", "pbe"},                // v2, "SLA PW PBX PBC"
        {"nc/Cu.upf", "pbe"},
        {"nc/C.upf", "pbe"},
    };
    for (const auto& [relative_path, functional] : expected) {
        EXPECT_EQ(functional_from_pseudopotentials({species_with("Si", relative_path)}, ""),
                  functional)
            << relative_path;
    }
}

TEST(FunctionalFromPseudopotentials, AgreeingSpeciesResolve) {
    EXPECT_EQ(functional_from_pseudopotentials(
                  {species_with("Si", "nc/Si.upf"), species_with("C", "paw/C.upf")}, ""),
              "pbe");
}

/**
 * Datasets of one functional agree however their headers spell it: in any case,
 * as a short name or as components, and either way for QE's LibXC form.
 */
TEST(FunctionalFromPseudopotentials, SpellingsDifferingInCaseOrShapeAgree) {
    const std::string lower_case = write_silicon_dataset_stating("pbe");
    const std::string components = write_silicon_dataset_stating("sla-pw-pbx-pbc");
    const std::string libxc_upper = write_silicon_dataset_stating("XC-000I-000I-101L-130L");
    const std::string libxc_lower = write_silicon_dataset_stating("xc-000i-000i-101l-130l");

    const std::string short_and_components =
        functional_from_pseudopotentials({species_with("C", "nc/C.upf"),
                                          species_from_file("Si", lower_case),
                                          species_from_file("Ge", components)},
                                         "");
    const std::string libxc_both_cases = functional_from_pseudopotentials(
        {species_from_file("Si", libxc_upper), species_from_file("Ge", libxc_lower)}, "");
    for (const auto& path : {lower_case, components, libxc_upper, libxc_lower})
        std::remove(path.c_str());

    EXPECT_EQ(short_and_components, "pbe");
    EXPECT_EQ(libxc_both_cases, "xc-000i-000i-101l-130l");
}

TEST(FunctionalFromPseudopotentials, ResolvesRelativeToTheDirectory) {
    AtomSpecies silicon = species_with("Si", "nc/Si.upf");
    silicon.pseudopotential_file = "Si.upf";
    EXPECT_EQ(functional_from_pseudopotentials({silicon}, pseudopotential_directory + "/nc"),
              "pbe");
}

// ─── Refusals ────────────────────────────────────────────────────────────────

/**
 * Datasets of different functionals: there is no functional the result
 * belongs to, so neither is picked.
 */
TEST(FunctionalFromPseudopotentials, DisagreeingSpeciesAreRefused) {
    const std::string message =
        refusal_message({species_with("Si", "nc/Si.upf"), species_with("Ge", "nc/Si.pz-vbc.UPF")});
    ASSERT_FALSE(message.empty());
    EXPECT_NE(message.find("Si.upf (Si) states pbe"), std::string::npos) << message;
    EXPECT_NE(message.find("Si.pz-vbc.UPF (Ge) states pz"), std::string::npos) << message;
    EXPECT_NE(message.find("model.functional"), std::string::npos) << message;
}

/// Perdew-Wang LDA is an LDA, but not the PZ one, so it is refused rather than approximated.
TEST(FunctionalFromPseudopotentials, UnmappedLabelIsRefusedByName) {
    const std::string path = write_silicon_dataset_stating("SLA PW NOGX NOGC");
    const std::string message = refusal_message({species_from_file("Si", path)});
    std::remove(path.c_str());

    ASSERT_FALSE(message.empty());
    EXPECT_NE(message.find("states 'SLA PW NOGX NOGC'"), std::string::npos) << message;
}

TEST(FunctionalFromPseudopotentials, UnreadableDatasetIsRefusedWithItsPath) {
    const std::string message = refusal_message({species_with("Si", "nc/no_such_dataset.upf")});
    ASSERT_FALSE(message.empty());
    EXPECT_NE(message.find("no_such_dataset.upf (Si) cannot be read"), std::string::npos)
        << message;
}

/**
 * A species that names no dataset, as in a deck missing its dataset file, leaves
 * nothing to read: the functional stays unresolved, and the missing dataset is
 * reported where pseudopotentials load.
 */
TEST(FunctionalFromPseudopotentials, NoDatasetNamedLeavesItUnresolved) {
    EXPECT_EQ(functional_from_pseudopotentials({}, ""), "");

    AtomSpecies unnamed = species_with("Si", "nc/Si.upf");
    unnamed.pseudopotential_file.clear();
    EXPECT_EQ(functional_from_pseudopotentials({species_with("C", "nc/C.upf"), unnamed}, ""), "");
}

// ─── Dataset position inside a shared file ──────────────────────────────────

TEST(DatasetIndexWithinFile, CountsEarlierSpeciesNamingTheSameFile) {
    const std::vector<AtomSpecies> species = {species_with("Ga", "concatenated_datasets"),
                                              species_with("Si", "nc/Si.upf"),
                                              species_with("As", "concatenated_datasets")};
    EXPECT_EQ(dataset_index_within_file(species, 0, ""), 0);
    EXPECT_EQ(dataset_index_within_file(species, 1, ""), 0);
    EXPECT_EQ(dataset_index_within_file(species, 2, ""), 1);
}
