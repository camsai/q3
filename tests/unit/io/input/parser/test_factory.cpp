#include <gtest/gtest.h>

#include <cstdio>
#include <fstream>
#include <iterator>
#include <string>

#include "io/input/parser/factory.hpp"
#include "io/input/parser/qe.hpp"
#include "io/input/parser/yaml.hpp"
#include "utils/temporary_path.hpp"

using namespace q3;

// ─── InputParserFactory Tests ───────────────────────────────────────────────

TEST(InputParserFactory, DotInExtensionReturnsQEParser) {
    auto parser = create_input_parser("test.in");
    ASSERT_NE(parser, nullptr);
    EXPECT_NE(dynamic_cast<QEInputParser*>(parser.get()), nullptr);
}

TEST(InputParserFactory, DotQEExtensionReturnsQEParser) {
    auto parser = create_input_parser("test.qe");
    ASSERT_NE(parser, nullptr);
    EXPECT_NE(dynamic_cast<QEInputParser*>(parser.get()), nullptr);
}

TEST(InputParserFactory, DotYmlExtensionReturnsYAMLParser) {
    auto parser = create_input_parser("test.yml");
    ASSERT_NE(parser, nullptr);
    EXPECT_NE(dynamic_cast<YAMLInputParser*>(parser.get()), nullptr);
}

TEST(InputParserFactory, DotYamlExtensionReturnsYAMLParser) {
    auto parser = create_input_parser("test.yaml");
    ASSERT_NE(parser, nullptr);
    EXPECT_NE(dynamic_cast<YAMLInputParser*>(parser.get()), nullptr);
}

TEST(InputParserFactory, DotJsonExtensionDefaultsToYAMLParser) {
    auto parser = create_input_parser("test.json");
    ASSERT_NE(parser, nullptr);
    EXPECT_NE(dynamic_cast<YAMLInputParser*>(parser.get()), nullptr);
}

// ─── parse_input: the functional an input does not name ─────────────────────

namespace {

// EXAMPLES_DIR is defined as a compile-time macro by CMakeLists.txt
const std::string pseudopotential_directory = std::string(EXAMPLES_DIR) + "/data/pseudopotentials";

/// A silicon YAML deck with the given `model` block (empty: none at all).
std::string write_silicon_yaml(const std::string& model_block, const std::string& dataset) {
    const std::string path = test_support::make_temporary_path(".yml");
    std::ofstream output(path);
    output << "material: " << EXAMPLES_DIR << "/data/materials/silicon.json\n"
           << model_block << "method:\n"
           << "  type: pseudopotential\n"
           << "  pseudopotentials:\n"
           << "    Si: " << pseudopotential_directory << "/" << dataset << "\n"
           << "control:\n"
           << "  calculation: scf\n";
    return path;
}

/// A silicon QE-format deck with the given extra &SYSTEM line (empty: none).
std::string write_silicon_qe_deck(const std::string& system_line, const std::string& dataset) {
    const std::string path = test_support::make_temporary_path(".in");
    std::ofstream output(path);
    output << "&CONTROL\n  calculation = 'scf'\n  pseudo_dir = '" << pseudopotential_directory
           << "/nc'\n/\n"
           << "&SYSTEM\n  ibrav = 2\n  celldm(1) = 10.263048\n  nat = 2\n  ntyp = 1\n"
           << "  ecutwfc = 20.0\n"
           << system_line << "/\n&ELECTRONS\n/\n"
           << "ATOMIC_SPECIES\n  Si 28.086 " << dataset << "\n"
           << "ATOMIC_POSITIONS crystal\n  Si 0.0 0.0 0.0\n  Si 0.25 0.25 0.25\n"
           << "K_POINTS automatic\n  2 2 2 0 0 0\n";
    return path;
}

std::string functional_parsed_from(const std::string& path) {
    const std::string functional = parse_input(path).exchange_correlation_functional;
    std::remove(path.c_str());
    return functional;
}

} // namespace

/**
 * The case a user hit: PBE datasets and no functional in the deck. Q3 ran
 * Perdew-Zunger LDA, its own former default, and nothing said so.
 */
TEST(ParseInput, YamlWithoutFunctionalTakesThePseudopotentials) {
    EXPECT_EQ(functional_parsed_from(write_silicon_yaml("", "nc/Si.upf")), "pbe");
    EXPECT_EQ(functional_parsed_from(write_silicon_yaml("", "nc/Si.pz-vbc.UPF")), "pz");
    EXPECT_EQ(
        functional_parsed_from(write_silicon_yaml("model:\n  type: dft\n", "us/Si_PBE_USPP.UPF")),
        "pbe");
}

TEST(ParseInput, QeDeckWithoutInputDftTakesThePseudopotentials) {
    EXPECT_EQ(functional_parsed_from(write_silicon_qe_deck("", "Si.upf")), "pbe");
    EXPECT_EQ(functional_parsed_from(write_silicon_qe_deck("", "Si.pz-vbc.UPF")), "pz");
}

/// What the input names wins, as `input_dft` does in pw.x.
TEST(ParseInput, NamedFunctionalOverridesThePseudopotentials) {
    EXPECT_EQ(functional_parsed_from(write_silicon_yaml("model:\n  functional: pz\n", "nc/Si.upf")),
              "pz");
    EXPECT_EQ(
        functional_parsed_from(write_silicon_qe_deck("  input_dft = 'PBE'\n", "Si.pz-vbc.UPF")),
        "pbe");
}

/**
 * One spelling, whatever the reader and however the deck wrote the name: what
 * `--check` reports and the provenance record carries is lower case, as the
 * example decks and ESSE write it. The YAML reader used to upper-case the name
 * while the QE reader kept it as written.
 */
TEST(ParseInput, TheFunctionalIsRecordedLowerCase) {
    EXPECT_EQ(
        functional_parsed_from(write_silicon_yaml("model:\n  functional: PBEsol\n", "nc/Si.upf")),
        "pbesol");
    EXPECT_EQ(functional_parsed_from(
                  write_silicon_yaml("model:\n  functional: LIBXC:117:130\n", "nc/Si.upf")),
              "libxc:117:130");
    EXPECT_EQ(functional_parsed_from(write_silicon_qe_deck("  input_dft = 'PBEsol'\n", "Si.upf")),
              "pbesol");
    EXPECT_EQ(functional_parsed_from(
                  write_silicon_qe_deck("  input_dft = 'XC-000I-000I-101L-130L'\n", "Si.upf")),
              "xc-000i-000i-101l-130l");
}

// ─── --material and relaxation steps ────────────────────────────────────────

namespace {

/// An ESSE material with explicit vectors in angstrom and the given crystal positions.
std::string write_step_material(const std::string& coordinates) {
    const std::string path = test_support::make_temporary_path(".json");
    std::ofstream output(path);
    output << R"({"lattice": {"vectors": {"a": [-2.8, 0.0, 2.8], "b": [0.0, 2.8, 2.8],)"
           << R"( "c": [-2.8, 2.8, 0.0], "units": "angstrom"}},)"
           << R"( "basis": {"elements": [{"id": 1, "value": "Si"}, {"id": 2, "value": "Si"}],)"
           << R"( "coordinates": [)" << coordinates << "]}}";
    return path;
}

} // namespace

/**
 * The deck keeps its settings and species; the material gives the cell and the
 * positions. The deck's alat stays the unit the cell is stated in, as across
 * pw.x's steps.
 */
TEST(SubstituteMaterial, ReplacesOnlyTheStructureOfAQeDeck) {
    const std::string deck = write_silicon_qe_deck("", "Si.upf");
    const std::string material = write_step_material(
        R"({"id": 1, "value": [0.0, 0.0, 0.0]}, {"id": 2, "value": [0.26, 0.25, 0.25]})");

    InputData input = parse_input(deck);
    substitute_material(input, material);

    EXPECT_EQ(input.calculation, "self_consistent_field");
    EXPECT_FALSE(input.start_from_saved_charge_density);
    EXPECT_EQ(input.bravais_lattice_index, 0);
    EXPECT_DOUBLE_EQ(input.cell_dimension_1, 10.263048);
    EXPECT_NEAR(input.lattice_vectors[0][2] * input.cell_dimension_1,
                2.8 / constants::BOHR_RADIUS_ANGSTROM,
                1e-12);
    ASSERT_EQ(input.atoms.size(), 2u);
    EXPECT_DOUBLE_EQ(input.atoms[1].position[0], 0.26);
    EXPECT_EQ(input.species.size(), 1u);
    EXPECT_DOUBLE_EQ(input.species[0].mass, 28.086);
    std::remove(deck.c_str());
    std::remove(material.c_str());
}

TEST(SubstituteMaterial, ADifferentAtomCountIsRefused) {
    const std::string deck = write_silicon_qe_deck("", "Si.upf");
    const std::string material = write_step_material(R"({"id": 1, "value": [0.0, 0.0, 0.0]})");

    InputData input = parse_input(deck);
    EXPECT_THROW(substitute_material(input, material), std::runtime_error);
    std::remove(deck.c_str());
    std::remove(material.c_str());
}

/**
 * A relax deck runs as one SCF at its own geometry, from a saved density when
 * one fits, and records that it was asked for a relaxation -- a step of the
 * relaxation q3.workflows drives, on whichever binary it is given.
 */
TEST(ParseInput, ARelaxDeckRunsAsOneSelfConsistentFieldStep) {
    const std::string deck = write_silicon_qe_deck("", "Si.upf");
    {
        std::string text;
        {
            std::ifstream input_stream(deck);
            text.assign(std::istreambuf_iterator<char>(input_stream), {});
        }
        text.replace(text.find("'scf'"), 5, "'relax'");
        std::ofstream output(deck);
        output << text << "&IONS\n/\n";
    }
    const InputData input = parse_input(deck);
    EXPECT_EQ(input.calculation, "self_consistent_field");
    EXPECT_TRUE(input.start_from_saved_charge_density);
    EXPECT_TRUE(input.is_relaxation_step);
    std::remove(deck.c_str());
}

/// pw.x's BFGS tuning moves its path, never its minimum, so a relax deck carrying it still reads.
TEST(ParseInput, QeDeckAcceptsPwBfgsTuningKeys) {
    const std::string deck = write_silicon_qe_deck("", "Si.upf");
    {
        std::ofstream output(deck, std::ios::app);
        output << "&IONS\n  ion_dynamics = 'bfgs'\n  trust_radius_ini = 0.3\n  bfgs_ndim = 3\n"
               << "  upscale = 50\n  pot_extrapolation = 'second_order'\n/\n";
    }
    EXPECT_NO_THROW(parse_input(deck));
    std::remove(deck.c_str());
}
