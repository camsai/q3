#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>

#include "io/input/parser/yaml.hpp"
#include "layer_1_structure/crystal/lattice.hpp"
#include "utils/temporary_path.hpp"

using namespace q3;

// EXAMPLES_DIR is defined as a compile-time macro by CMakeLists.txt

static const std::string silicon_nc_input =
    std::string(EXAMPLES_DIR) + "/total_energy--semiconductor--silicon/nc.yml";

static const std::string copper_nc_input =
    std::string(EXAMPLES_DIR) + "/total_energy--metal--copper/nc.yml";

// ── Silicon NC parsing tests ────────────────────────────────────────

TEST(YAMLParser, SiliconNCHasTwoAtoms) {
    YAMLInputParser parser;
    InputData data = parser.parse(silicon_nc_input);
    EXPECT_EQ(data.number_of_atoms, 2);
}

TEST(YAMLParser, SiliconNCSpeciesIncludesSi) {
    YAMLInputParser parser;
    InputData data = parser.parse(silicon_nc_input);
    bool found_silicon = false;
    for (const auto& species : data.species) {
        if (species.symbol == "Si") {
            found_silicon = true;
            break;
        }
    }
    EXPECT_TRUE(found_silicon);
}

TEST(YAMLParser, SiliconNCLatticeParameterIsPositive) {
    YAMLInputParser parser;
    InputData data = parser.parse(silicon_nc_input);
    EXPECT_GT(data.cell_dimension_1, 0.0);
}

TEST(YAMLParser, CopperNCAtomsIncludeCu) {
    YAMLInputParser parser;
    InputData data = parser.parse(copper_nc_input);
    bool found_copper = false;
    for (const auto& species : data.species) {
        if (species.symbol == "Cu") {
            found_copper = true;
            break;
        }
    }
    EXPECT_TRUE(found_copper);
}

TEST(YAMLParser, SiliconNCWavefunctionEnergyCutoffIsReasonable) {
    YAMLInputParser parser;
    InputData data = parser.parse(silicon_nc_input);
    // The silicon NC example specifies wavefunction cutoff of 48.0 Ry
    EXPECT_GT(data.wavefunction_energy_cutoff, 0.0);
    EXPECT_NEAR(data.wavefunction_energy_cutoff, 48.0, 1e-10);
}

TEST(YAMLParser, SiliconNCExchangeCorrelationFunctionalIsSet) {
    YAMLInputParser parser;
    InputData data = parser.parse(silicon_nc_input);
    // Kept as the deck writes it; parse_input() is what lower-cases every reader's name
    EXPECT_FALSE(data.exchange_correlation_functional.empty());
    EXPECT_EQ(data.exchange_correlation_functional, "pbe");
}

// ── Error-path tests ────────────────────────────────────────────────

TEST(YAMLParser, MissingMaterialKeyThrows) {
    // Write a YAML file without the required 'material' key
    std::string temporary_path = test_support::make_temporary_path(".yml");
    {
        std::ofstream output_stream(temporary_path);
        output_stream << "model:\n  functional: pbe\n";
    }
    YAMLInputParser parser;
    EXPECT_THROW(parser.parse(temporary_path), std::runtime_error);
    std::remove(temporary_path.c_str());
}

TEST(YAMLParser, NonexistentFileThrows) {
    YAMLInputParser parser;
    EXPECT_THROW(parser.parse("/nonexistent/path/to/input.yml"), YAML::BadFile);
}

// ── Additional error-path tests ─────────────────────────────────────

TEST(YAMLParser, MalformedYAMLSyntaxThrows) {
    std::string temporary_path = test_support::make_temporary_path(".yml");
    {
        std::ofstream output_stream(temporary_path);
        output_stream << "material:\n  lattice:\n    - broken: [unmatched\n";
    }
    YAMLInputParser parser;
    EXPECT_THROW(parser.parse(temporary_path), std::exception);
    std::remove(temporary_path.c_str());
}

TEST(YAMLParser, MaterialWithoutLatticeThrows) {
    std::string temporary_path = test_support::make_temporary_path(".yml");
    {
        std::ofstream output_stream(temporary_path);
        output_stream << "material:\n"
                      << "  basis:\n"
                      << "    elements:\n"
                      << "      - id: 1\n"
                      << "        value: Si\n"
                      << "    coordinates:\n"
                      << "      - id: 1\n"
                      << "        value: [0.0, 0.0, 0.0]\n";
    }
    YAMLInputParser parser;
    EXPECT_THROW(parser.parse(temporary_path), std::runtime_error);
    std::remove(temporary_path.c_str());
}

TEST(YAMLParser, MaterialAsListThrows) {
    /**
     * The material key must be a scalar (file path) or a map (inline definition),
     * but not a sequence. Verify that a sequence throws.
     */
    std::string temporary_path = test_support::make_temporary_path(".yml");
    {
        std::ofstream output_stream(temporary_path);
        output_stream << "material:\n  - item1\n  - item2\n";
    }
    YAMLInputParser parser;
    EXPECT_THROW(parser.parse(temporary_path), std::runtime_error);
    std::remove(temporary_path.c_str());
}

TEST(YAMLParser, EmptyYAMLFileThrows) {
    std::string temporary_path = test_support::make_temporary_path(".yml");
    {
        std::ofstream output_stream(temporary_path);
        output_stream << "";
    }
    YAMLInputParser parser;
    EXPECT_THROW(parser.parse(temporary_path), std::runtime_error);
    std::remove(temporary_path.c_str());
}

TEST(YAMLParser, MaterialWithoutBasisThrows) {
    std::string temporary_path = test_support::make_temporary_path(".yml");
    {
        std::ofstream output_stream(temporary_path);
        output_stream << "material:\n"
                      << "  lattice:\n"
                      << "    type: FCC\n"
                      << "    a: 5.43\n";
    }
    YAMLInputParser parser;
    EXPECT_THROW(parser.parse(temporary_path), std::runtime_error);
    std::remove(temporary_path.c_str());
}

TEST(YAMLParser, NonCubicLatticeTypeParsesAsExplicitVectors) {
    /**
     * Formerly threw "Unsupported lattice type"; general-lattice support now
     * builds explicit vectors from a/b/c/alpha/beta/gamma for any type.
     */
    std::string temporary_path = test_support::make_temporary_path(".yml");
    {
        std::ofstream output_stream(temporary_path);
        output_stream << "material:\n"
                      << "  lattice:\n"
                      << "    type: RHL\n"
                      << "    a: 5.0\n"
                      << "    alpha: 60\n"
                      << "    beta: 60\n"
                      << "    gamma: 60\n"
                      << "  basis:\n"
                      << "    elements:\n"
                      << "      - id: 1\n"
                      << "        value: Si\n"
                      << "    coordinates:\n"
                      << "      - id: 1\n"
                      << "        value: [0.0, 0.0, 0.0]\n";
    }
    YAMLInputParser parser;
    InputData data = parser.parse(temporary_path);
    std::remove(temporary_path.c_str());

    EXPECT_TRUE(data.has_explicit_lattice_vectors);
    // All three vectors have length a and 60° pairwise angles.
    for (int row = 0; row < 3; ++row) {
        Real length = std::sqrt(data.lattice_vectors[row][0] * data.lattice_vectors[row][0] +
                                data.lattice_vectors[row][1] * data.lattice_vectors[row][1] +
                                data.lattice_vectors[row][2] * data.lattice_vectors[row][2]);
        EXPECT_NEAR(length, 1.0, 1e-10) << "row " << row; // in units of alat = a
    }
    Real dot_ab = 0.0;
    for (int column = 0; column < 3; ++column) {
        dot_ab += data.lattice_vectors[0][column] * data.lattice_vectors[1][column];
    }
    EXPECT_NEAR(dot_ab, 0.5, 1e-10); // cos 60°
}

TEST(YAMLParser, MethodSubtypePAWIsAccepted) {
    /**
     * `method.subtype: paw` used to be rejected at parse time, because there
     * was no PAW implementation and such an input would otherwise have run
     * down the NC path. Now that the PAW SCF iterator exists, parsing must
     * accept it — the guard and the implementation moved together, and a
     * parser that still refused would make the iterator unreachable from a
     * real input file.
     */
    std::string temporary_path = test_support::make_temporary_path(".yml");
    {
        std::ofstream output_stream(temporary_path);
        output_stream << "material:\n"
                      << "  lattice:\n"
                      << "    type: FCC\n"
                      << "    a: 5.43\n"
                      << "  basis:\n"
                      << "    elements:\n"
                      << "      - id: 1\n"
                      << "        value: Si\n"
                      << "    coordinates:\n"
                      << "      - id: 1\n"
                      << "        value: [0.0, 0.0, 0.0]\n"
                      << "method:\n"
                      << "  type: pseudopotential\n"
                      << "  subtype: paw\n";
    }
    YAMLInputParser parser;
    EXPECT_NO_THROW({ InputData data = parser.parse(temporary_path); });
    std::remove(temporary_path.c_str());
}

TEST(YAMLParser, MismatchedElementsAndCoordinatesThrows) {
    /**
     * The elements and coordinates arrays must have the same length.
     * Providing different counts must throw.
     */
    std::string temporary_path = test_support::make_temporary_path(".yml");
    {
        std::ofstream output_stream(temporary_path);
        output_stream << "material:\n"
                      << "  lattice:\n"
                      << "    type: FCC\n"
                      << "    a: 5.43\n"
                      << "  basis:\n"
                      << "    elements:\n"
                      << "      - id: 1\n"
                      << "        value: Si\n"
                      << "      - id: 2\n"
                      << "        value: Si\n"
                      << "    coordinates:\n"
                      << "      - id: 1\n"
                      << "        value: [0.0, 0.0, 0.0]\n";
    }
    YAMLInputParser parser;
    EXPECT_THROW(parser.parse(temporary_path), std::runtime_error);
    std::remove(temporary_path.c_str());
}

// ── Spin / per-species starting_magnetization ───────────────────────

static std::string write_temporary_yaml(const std::string& content) {
    std::string temporary_path = test_support::make_temporary_path(".yml");
    std::ofstream output_stream(temporary_path);
    output_stream << content;
    return temporary_path;
}

static const std::string chromium_afm_yaml = R"(
material:
  lattice:
    type: SC
    a: 5.45
    units: bohr
  basis:
    elements:
      - id: 0
        value: Cr1
      - id: 1
        value: Cr2
    coordinates:
      - id: 0
        value: [0.0, 0.0, 0.0]
      - id: 1
        value: [0.5, 0.5, 0.5]
method:
  type: pseudopotential
  subtype: norm-conserving
  scf:
    number_of_spin_components: 2
    starting_magnetization:
      Cr1: 0.5
      Cr2: -0.5
control:
  calculation: scf
)";

TEST(YAMLParserSpinTest, ParsesPerSpeciesStartingMagnetization) {
    std::string path = write_temporary_yaml(chromium_afm_yaml);
    YAMLInputParser parser;
    InputData data = parser.parse(path);
    std::remove(path.c_str());

    EXPECT_EQ(data.number_of_spin_components, 2);
    ASSERT_EQ(data.species.size(), 2u);
    EXPECT_EQ(data.species[0].symbol, "Cr1");
    EXPECT_EQ(data.species[1].symbol, "Cr2");
    ASSERT_EQ(data.starting_magnetization.size(), 2u);
    EXPECT_NEAR(data.starting_magnetization[0], 0.5, 1e-12);
    EXPECT_NEAR(data.starting_magnetization[1], -0.5, 1e-12);
}

TEST(YAMLParserSpinTest, NspinKeyRejectedWithGuidance) {
    /**
     * The QE abbreviation belongs to the QE-namelist parser only; the manifest
     * must reject it rather than silently run unpolarized.
     */
    std::string yaml_content = chromium_afm_yaml;
    size_t position = yaml_content.find("number_of_spin_components");
    yaml_content.replace(position, std::string("number_of_spin_components").size(), "nspin");
    std::string path = write_temporary_yaml(yaml_content);
    YAMLInputParser parser;
    EXPECT_THROW(parser.parse(path), std::runtime_error);
    std::remove(path.c_str());
}

TEST(YAMLParserSpinTest, UnsupportedSpinComponentCountThrows) {
    /**
     * Only 1 (unpolarized) and 2 (collinear LSDA) are valid; noncollinear is
     * not implemented and must fail at parse time, not deep in the SCF.
     */
    std::string yaml_content = chromium_afm_yaml;
    size_t position = yaml_content.find("number_of_spin_components: 2");
    yaml_content.replace(position,
                         std::string("number_of_spin_components: 2").size(),
                         "number_of_spin_components: 4");
    std::string path = write_temporary_yaml(yaml_content);
    YAMLInputParser parser;
    EXPECT_THROW(parser.parse(path), std::runtime_error);
    std::remove(path.c_str());
}

TEST(YAMLParserSpinTest, UnknownSpeciesInMagnetizationMapThrows) {
    std::string yaml_content = chromium_afm_yaml;
    size_t position = yaml_content.find("Cr2: -0.5");
    yaml_content.replace(position, std::string("Cr2: -0.5").size(), "Ni: -0.5");
    std::string path = write_temporary_yaml(yaml_content);
    YAMLInputParser parser;
    EXPECT_THROW(parser.parse(path), std::runtime_error);
    std::remove(path.c_str());
}

TEST(YAMLParserSpinTest, DefaultIsUnpolarized) {
    YAMLInputParser parser;
    InputData data = parser.parse(silicon_nc_input);
    EXPECT_EQ(data.number_of_spin_components, 1);
    EXPECT_TRUE(data.starting_magnetization.empty());
}

// ── General-lattice (non-cubic) cells ───────────────────────────────

static const std::string graphite_hex_yaml = R"(
material:
  lattice:
    type: HEX
    a: 2.464
    c: 6.711
    gamma: 120
    units: angstrom
  basis:
    elements:
      - id: 0
        value: C
      - id: 1
        value: C
      - id: 2
        value: C
      - id: 3
        value: C
    coordinates:
      - id: 0
        value: [0.0, 0.0, 0.25]
      - id: 1
        value: [0.0, 0.0, 0.75]
      - id: 2
        value: [0.333333333333, 0.666666666667, 0.25]
      - id: 3
        value: [0.666666666667, 0.333333333333, 0.75]
control:
  calculation: scf
)";

// Use the production constant so a constants regression fails these tests loudly.
static constexpr double BOHR_PER_ANGSTROM = 1.0 / constants::BOHR_RADIUS_ANGSTROM;

TEST(YAMLParserGeneralLattice, HexagonalCellProducesExplicitVectors) {
    std::string path = write_temporary_yaml(graphite_hex_yaml);
    YAMLInputParser parser;
    InputData data = parser.parse(path);
    std::remove(path.c_str());

    EXPECT_TRUE(data.has_explicit_lattice_vectors);
    EXPECT_EQ(data.bravais_lattice_index, 0);

    // alat = a (in bohr); rows in units of alat.
    double expected_alat = 2.464 * BOHR_PER_ANGSTROM;
    EXPECT_NEAR(data.cell_dimension_1, expected_alat, 1e-8);

    /**
     * QE ibrav=4 convention: a1 = (1,0,0), a2 = (-1/2, √3/2, 0), a3 = (0,0,c/a).
     * Our gamma=120° construction gives a2 = (cos 120°, sin 120°, 0) = (-1/2, √3/2, 0).
     */
    double c_over_a = 6.711 / 2.464;
    EXPECT_NEAR(data.lattice_vectors[0][0], 1.0, 1e-10);
    EXPECT_NEAR(data.lattice_vectors[0][1], 0.0, 1e-10);
    EXPECT_NEAR(data.lattice_vectors[1][0], -0.5, 1e-10);
    EXPECT_NEAR(data.lattice_vectors[1][1], std::sqrt(3.0) / 2.0, 1e-10);
    EXPECT_NEAR(data.lattice_vectors[2][2], c_over_a, 1e-10);
    EXPECT_NEAR(data.lattice_vectors[2][0], 0.0, 1e-10);
}

TEST(YAMLParserGeneralLattice, HexagonalMatchesBravaisIndex4Construction) {
    std::string path = write_temporary_yaml(graphite_hex_yaml);
    YAMLInputParser parser;
    InputData data = parser.parse(path);
    std::remove(path.c_str());

    // Round-trip: the explicit vectors must reproduce the QE ibrav=4 cell.
    Matrix3x3 reference =
        get_lattice_vectors_from_bravais_index(4, data.cell_dimension_1, 0, 6.711 / 2.464, 0, 0, 0);
    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
            EXPECT_NEAR(data.lattice_vectors[row][column], reference[row][column], 1e-10)
                << "row " << row << " column " << column;
        }
    }
}

TEST(YAMLParserGeneralLattice, ExplicitVectorsParsed) {
    std::string yaml_content = R"(
material:
  lattice:
    vectors:
      a: [10.0, 0.0, 0.0]
      b: [0.0, 12.0, 0.0]
      c: [0.0, 0.0, 30.0]
      alat: 1.0
      units: bohr
  basis:
    elements:
      - id: 0
        value: Al
    coordinates:
      - id: 0
        value: [0.0, 0.0, 0.0]
control:
  calculation: scf
)";
    std::string path = write_temporary_yaml(yaml_content);
    YAMLInputParser parser;
    InputData data = parser.parse(path);
    std::remove(path.c_str());

    EXPECT_TRUE(data.has_explicit_lattice_vectors);
    EXPECT_NEAR(data.cell_dimension_1, 10.0, 1e-12);
    EXPECT_NEAR(data.lattice_vectors[0][0], 1.0, 1e-12);
    EXPECT_NEAR(data.lattice_vectors[1][1], 1.2, 1e-12);
    EXPECT_NEAR(data.lattice_vectors[2][2], 3.0, 1e-12);
}

TEST(YAMLParserGeneralLattice, CubicLatticesKeepBravaisPath) {
    // FCC/BCC/SC must keep the (validated) Bravais-index construction.
    YAMLInputParser parser;
    InputData data = parser.parse(silicon_nc_input);
    EXPECT_FALSE(data.has_explicit_lattice_vectors);
    EXPECT_EQ(data.bravais_lattice_index, 2);
}

TEST(YAMLParserGeneralLattice, InvalidAnglesThrow) {
    std::string yaml_content = graphite_hex_yaml;
    size_t position = yaml_content.find("gamma: 120");
    yaml_content.replace(position, std::string("gamma: 120").size(), "gamma: 180");
    std::string path = write_temporary_yaml(yaml_content);
    YAMLInputParser parser;
    EXPECT_THROW(parser.parse(path), std::runtime_error);
    std::remove(path.c_str());
}

TEST(YAMLParserGeneralLattice, VectorUnitsTakePrecedenceOverLatticeUnits) {
    /**
     * When both lattice.units and lattice.vectors.units are present, the more
     * specific annotation (on the vectors themselves) wins.
     */
    std::string yaml_content = R"(
material:
  lattice:
    units: angstrom
    vectors:
      a: [10.0, 0.0, 0.0]
      b: [0.0, 12.0, 0.0]
      c: [0.0, 0.0, 30.0]
      units: bohr
  basis:
    elements:
      - id: 0
        value: Al
    coordinates:
      - id: 0
        value: [0.0, 0.0, 0.0]
control:
  calculation: scf
)";
    std::string path = write_temporary_yaml(yaml_content);
    YAMLInputParser parser;
    InputData data = parser.parse(path);
    std::remove(path.c_str());

    // Interpreted in bohr (inner units), NOT converted from angstrom.
    EXPECT_NEAR(data.cell_dimension_1, 10.0, 1e-12);
}

/**
 * ── Unknown manifest key rejection ───────────────────────────────────
 *
 * A misspelled or unsupported key under model/method/control previously fell
 * straight to the silent-default path: the setting the user thought they were
 * changing simply never applied, with no error at all. reject_unknown_keys()
 * (yaml.hpp) now catches this at parse time. material/lattice/basis are
 * deliberately NOT covered (see TODOs.md): every example in this repo
 * references an external material JSON file rather than embedding one
 * inline, so that path carries far less real-world typo risk.
 */

static const std::string full_manifest_yaml = R"(
material:
  lattice:
    type: SC
    a: 5.45
    units: bohr
  basis:
    elements:
      - id: 0
        value: Si
    coordinates:
      - id: 0
        value: [0.0, 0.0, 0.0]
model:
  type: dft
  subtype: gga
  functional: pbe
method:
  type: pseudopotential
  subtype: norm-conserving
  cutoffs:
    wavefunction: 30.0
    density: 120.0
  scf:
    convergence_threshold: 1.0e-08
    mixing:
      mode: broyden
      beta: 0.7
    smearing:
      type: gaussian
      degauss: 0.02
    kgrid:
      dimensions: [2, 2, 2]
      shifts: [0, 0, 0]
    hubbard:
      - species: Si
        orbital_label: 3D
        u_ev: 3.0
control:
  calculation: scf
  band_path:
    - label: G
      coordinates: [0.0, 0.0, 0.0]
)";

/// Replace the first occurrence of `target` with `replacement` in `content`.
static std::string replace_first(const std::string& content,
                                 const std::string& target,
                                 const std::string& replacement) {
    std::string result = content;
    size_t position = result.find(target);
    if (position == std::string::npos)
        throw std::runtime_error("replace_first: target not found in fixture: " + target);
    result.replace(position, target.size(), replacement);
    return result;
}

TEST(YAMLParserUnknownKeys, UnknownRootKeyThrows) {
    std::string yaml_content =
        replace_first(full_manifest_yaml, "model:", "modle:\n  x: 1\nmodel:");
    std::string path = write_temporary_yaml(yaml_content);
    YAMLInputParser parser;
    EXPECT_THROW(parser.parse(path), std::runtime_error);
    std::remove(path.c_str());
}

TEST(YAMLParserUnknownKeys, UnknownModelKeyThrows) {
    std::string yaml_content =
        replace_first(full_manifest_yaml, "functional: pbe", "functionl: pbe");
    std::string path = write_temporary_yaml(yaml_content);
    YAMLInputParser parser;
    EXPECT_THROW(parser.parse(path), std::runtime_error);
    std::remove(path.c_str());
}

TEST(YAMLParserUnknownKeys, ModelAndMethodTypeSubtypeAreAcceptedButUnused) {
    /**
     * Regression guard: model.type/model.subtype/method.type are written by
     * every example (and by q3.io.input_writer.write_scf_input) but not yet
     * consumed by Q3 -- DFT/Kohn-Sham + planewave-pseudopotential are its only
     * supported model/method, so nothing branches on these fields today. They
     * must stay in the allow-lists regardless, or every existing example
     * input would start throwing "unknown key" on its very first field.
     */
    std::string path = write_temporary_yaml(full_manifest_yaml);
    YAMLInputParser parser;
    EXPECT_NO_THROW(parser.parse(path));
    std::remove(path.c_str());
}

TEST(YAMLParserUnknownKeys, UnknownMethodKeyThrows) {
    std::string yaml_content =
        replace_first(full_manifest_yaml, "subtype: norm-conserving", "subtyp: norm-conserving");
    std::string path = write_temporary_yaml(yaml_content);
    YAMLInputParser parser;
    EXPECT_THROW(parser.parse(path), std::runtime_error);
    std::remove(path.c_str());
}

TEST(YAMLParserUnknownKeys, UnknownCutoffsKeyThrows) {
    std::string yaml_content =
        replace_first(full_manifest_yaml, "wavefunction: 30.0", "wave_function: 30.0");
    std::string path = write_temporary_yaml(yaml_content);
    YAMLInputParser parser;
    EXPECT_THROW(parser.parse(path), std::runtime_error);
    std::remove(path.c_str());
}

TEST(YAMLParserUnknownKeys, UnknownScfKeyThrows) {
    std::string yaml_content = replace_first(
        full_manifest_yaml, "convergence_threshold: 1.0e-08", "convergance_threshold: 1.0e-08");
    std::string path = write_temporary_yaml(yaml_content);
    YAMLInputParser parser;
    EXPECT_THROW(parser.parse(path), std::runtime_error);
    std::remove(path.c_str());
}

TEST(YAMLParserUnknownKeys, WriteOnlyCheckpointIntervalIsRejected) {
    std::string yaml_content =
        replace_first(full_manifest_yaml,
                      "control:\n  calculation: scf",
                      "control:\n  calculation: scf\n  checkpoint_interval: 5");
    std::string path = write_temporary_yaml(yaml_content);
    YAMLInputParser parser;
    EXPECT_THROW(parser.parse(path), std::runtime_error);
    std::remove(path.c_str());
}

TEST(YAMLParserUnknownKeys, UnknownMixingKeyThrows) {
    std::string yaml_content = replace_first(full_manifest_yaml, "mode: broyden", "mdoe: broyden");
    std::string path = write_temporary_yaml(yaml_content);
    YAMLInputParser parser;
    EXPECT_THROW(parser.parse(path), std::runtime_error);
    std::remove(path.c_str());
}

TEST(YAMLParserUnknownKeys, UnknownSmearingKeyThrows) {
    /**
     * Regression guard for the exact bug found in q3.io.input_writer.write_scf_input:
     * it wrote smearing.scheme/.width, which this parser silently ignored
     * (smearing never applied). Now caught at parse time.
     */
    std::string yaml_content =
        replace_first(full_manifest_yaml, "type: gaussian", "scheme: gaussian");
    std::string path = write_temporary_yaml(yaml_content);
    YAMLInputParser parser;
    EXPECT_THROW(parser.parse(path), std::runtime_error);
    std::remove(path.c_str());
}

TEST(YAMLParserSmearing, MethfesselPaxtonOrderIsRead) {
    std::string yaml_content =
        replace_first(full_manifest_yaml,
                      "type: gaussian\n      degauss: 0.02",
                      "type: methfessel-paxton\n      degauss: 0.02\n      order: 2");
    std::string path = write_temporary_yaml(yaml_content);
    YAMLInputParser parser;
    InputData data = parser.parse(path);
    EXPECT_EQ(data.smearing, "methfessel-paxton");
    EXPECT_EQ(data.methfessel_paxton_order, 2);
    std::remove(path.c_str());
}

TEST(YAMLParserSmearing, OrderDefaultsToOne) {
    std::string path = write_temporary_yaml(full_manifest_yaml);
    YAMLInputParser parser;
    EXPECT_EQ(parser.parse(path).methfessel_paxton_order, 1);
    std::remove(path.c_str());
}

TEST(YAMLParserSmearing, OrderWithNonMethfesselPaxtonTypeThrows) {
    // An order on Gaussian smearing would be silently ignored, so it is refused.
    std::string yaml_content =
        replace_first(full_manifest_yaml, "degauss: 0.02", "degauss: 0.02\n      order: 2");
    std::string path = write_temporary_yaml(yaml_content);
    YAMLInputParser parser;
    EXPECT_THROW(parser.parse(path), std::runtime_error);
    std::remove(path.c_str());
}

TEST(YAMLParserSmearing, OrderBelowOneThrows) {
    std::string yaml_content = replace_first(full_manifest_yaml,
                                             "type: gaussian\n      degauss: 0.02",
                                             "type: mp\n      degauss: 0.02\n      order: 0");
    std::string path = write_temporary_yaml(yaml_content);
    YAMLInputParser parser;
    EXPECT_THROW(parser.parse(path), std::runtime_error);
    std::remove(path.c_str());
}

TEST(YAMLParserUnknownKeys, UnknownKgridKeyThrows) {
    std::string yaml_content =
        replace_first(full_manifest_yaml, "dimensions: [2, 2, 2]", "dimension: [2, 2, 2]");
    std::string path = write_temporary_yaml(yaml_content);
    YAMLInputParser parser;
    EXPECT_THROW(parser.parse(path), std::runtime_error);
    std::remove(path.c_str());
}

TEST(YAMLParserUnknownKeys, UnknownHubbardEntryKeyThrows) {
    std::string yaml_content = replace_first(full_manifest_yaml, "u_ev: 3.0", "u_eV: 3.0");
    std::string path = write_temporary_yaml(yaml_content);
    YAMLInputParser parser;
    EXPECT_THROW(parser.parse(path), std::runtime_error);
    std::remove(path.c_str());
}

TEST(YAMLParserUnknownKeys, UnknownControlKeyThrows) {
    std::string yaml_content =
        replace_first(full_manifest_yaml, "calculation: scf", "clculation: scf");
    std::string path = write_temporary_yaml(yaml_content);
    YAMLInputParser parser;
    EXPECT_THROW(parser.parse(path), std::runtime_error);
    std::remove(path.c_str());
}

TEST(YAMLParserUnknownKeys, UnknownBandPathEntryKeyThrows) {
    std::string yaml_content = replace_first(full_manifest_yaml, "label: G", "lable: G");
    std::string path = write_temporary_yaml(yaml_content);
    YAMLInputParser parser;
    EXPECT_THROW(parser.parse(path), std::runtime_error);
    std::remove(path.c_str());
}

// ── Deterministic FFT planning ────────────────────────────────────────

TEST(YAMLParser, DeterministicFftPlanningDefaultsToFalse) {
    YAMLInputParser parser;
    InputData data = parser.parse(silicon_nc_input);
    EXPECT_FALSE(data.use_deterministic_fft_planning);
}

TEST(YAMLParser, DeterministicFftPlanningParsesTrue) {
    std::string yaml_content =
        replace_first(full_manifest_yaml,
                      "control:\n  calculation: scf",
                      "control:\n  calculation: scf\n  use_deterministic_fft_planning: true");
    std::string path = write_temporary_yaml(yaml_content);
    YAMLInputParser parser;
    InputData data = parser.parse(path);
    std::remove(path.c_str());
    EXPECT_TRUE(data.use_deterministic_fft_planning);
}

// ── Exact wavefunction grid ───────────────────────────────────────────

TEST(YAMLParser, ExactWavefunctionGridDefaultsToFalse) {
    // A YAML deck keeps the 2 x ecutwfc grid unless it asks for the exact one.
    YAMLInputParser parser;
    InputData data = parser.parse(silicon_nc_input);
    EXPECT_FALSE(data.use_exact_wavefunction_grid);
}

TEST(YAMLParser, ExactWavefunctionGridParsesTrue) {
    std::string yaml_content =
        replace_first(full_manifest_yaml,
                      "control:\n  calculation: scf",
                      "control:\n  calculation: scf\n  use_exact_wavefunction_grid: true");
    std::string path = write_temporary_yaml(yaml_content);
    YAMLInputParser parser;
    InputData data = parser.parse(path);
    std::remove(path.c_str());
    EXPECT_TRUE(data.use_exact_wavefunction_grid);
}

// ── Symmetry disable flag ─────────────────────────────────────────────

TEST(YAMLParser, DisableSymmetryDefaultsToFalseAndParsesTrue) {
    YAMLInputParser parser;
    EXPECT_FALSE(parser.parse(silicon_nc_input).disable_symmetry);

    std::string yaml_content =
        replace_first(full_manifest_yaml,
                      "control:\n  calculation: scf",
                      "control:\n  calculation: scf\n  disable_symmetry: true");
    std::string path = write_temporary_yaml(yaml_content);
    InputData data = parser.parse(path);
    std::remove(path.c_str());
    EXPECT_TRUE(data.disable_symmetry);
}

TEST(YAMLParser, RetiredNosymSpellingIsRejected) {
    /**
     * The pre-rename key must fail loudly at parse time (reject_unknown_keys),
     * not be silently ignored -- an ignored nosym would re-enable symmetry
     * reduction behind the user's back.
     */
    std::string yaml_content = replace_first(full_manifest_yaml,
                                             "control:\n  calculation: scf",
                                             "control:\n  calculation: scf\n  nosym: true");
    std::string path = write_temporary_yaml(yaml_content);
    YAMLInputParser parser;
    EXPECT_THROW(parser.parse(path), std::runtime_error);
    std::remove(path.c_str());
}

// ── DFT+U ortho-atomic projectors ───────────────────────────────────────

TEST(YAMLParser, HubbardProjectorsDefaultsToAtomic) {
    YAMLInputParser parser;
    InputData data = parser.parse(silicon_nc_input);
    EXPECT_EQ(data.hubbard_projector_type, "atomic");
}

TEST(YAMLParser, HubbardProjectorsParsesOrthoAtomic) {
    std::string yaml_content = replace_first(
        full_manifest_yaml, "    hubbard:", "    hubbard_projectors: ortho-atomic\n    hubbard:");
    std::string path = write_temporary_yaml(yaml_content);
    YAMLInputParser parser;
    InputData data = parser.parse(path);
    std::remove(path.c_str());
    EXPECT_EQ(data.hubbard_projector_type, "ortho-atomic");
}

TEST(YAMLParser, HubbardProjectorsRejectsInvalidValue) {
    std::string yaml_content = replace_first(
        full_manifest_yaml, "    hubbard:", "    hubbard_projectors: wannier\n    hubbard:");
    std::string path = write_temporary_yaml(yaml_content);
    YAMLInputParser parser;
    EXPECT_THROW(parser.parse(path), std::runtime_error);
    std::remove(path.c_str());
}

// ── Dispersion correction ───────────────────────────────────────────────

TEST(YAMLParser, ModelDispersionDefaultsToNone) {
    YAMLInputParser parser;
    InputData data = parser.parse(silicon_nc_input);
    EXPECT_EQ(data.dispersion_correction, "none");
}

TEST(YAMLParser, ModelDispersionParsesD3bjCaseInsensitively) {
    std::string yaml_content =
        replace_first(full_manifest_yaml, "functional: pbe", "functional: pbe\n  dispersion: D3BJ");
    std::string path = write_temporary_yaml(yaml_content);
    YAMLInputParser parser;
    InputData data = parser.parse(path);
    std::remove(path.c_str());
    EXPECT_EQ(data.dispersion_correction, "d3bj");
}

TEST(YAMLParser, ModelDispersionRejectsUnknownValue) {
    /**
     * Previously an unrecognized scheme was stored verbatim and silently
     * inert -- the run proceeded without the correction the user asked for.
     */
    std::string yaml_content =
        replace_first(full_manifest_yaml, "functional: pbe", "functional: pbe\n  dispersion: d3");
    std::string path = write_temporary_yaml(yaml_content);
    YAMLInputParser parser;
    EXPECT_THROW(parser.parse(path), std::runtime_error);
    std::remove(path.c_str());
}

// ── Isolated-system (Makov-Payne) correction ────────────────────────────

TEST(YAMLParser, IsolatedSystemCorrectionDefaultsToNone) {
    YAMLInputParser parser;
    InputData data = parser.parse(silicon_nc_input);
    EXPECT_EQ(data.isolated_system_correction, "none");
}

TEST(YAMLParser, IsolatedSystemCorrectionParsesMakovPayne) {
    std::string yaml_content =
        replace_first(full_manifest_yaml,
                      "control:\n  calculation: scf",
                      "control:\n  calculation: scf\n  isolated_system_correction: makov-payne");
    std::string path = write_temporary_yaml(yaml_content);
    YAMLInputParser parser;
    InputData data = parser.parse(path);
    std::remove(path.c_str());
    EXPECT_EQ(data.isolated_system_correction, "makov-payne");
}

TEST(YAMLParser, IsolatedSystemCorrectionParsesMartynaTuckerman) {
    std::string yaml_content = replace_first(
        full_manifest_yaml,
        "control:\n  calculation: scf",
        "control:\n  calculation: scf\n  isolated_system_correction: martyna-tuckerman");
    std::string path = write_temporary_yaml(yaml_content);
    YAMLInputParser parser;
    InputData data = parser.parse(path);
    std::remove(path.c_str());
    EXPECT_EQ(data.isolated_system_correction, "martyna-tuckerman");
}

TEST(YAMLParser, IsolatedSystemCorrectionRejectsInvalidValue) {
    /**
     * "point-countercharge" is a real QE-family isolated-system scheme name
     * (not yet implemented in Q3 -- see IsolatedSystemCorrectionCalculator's
     * docstring) picked specifically so this test can't accidentally pass
     * again the next time a new scheme is added, the way it would have if it
     * still asserted on "martyna-tuckerman" after that scheme shipped.
     */
    std::string yaml_content = replace_first(
        full_manifest_yaml,
        "control:\n  calculation: scf",
        "control:\n  calculation: scf\n  isolated_system_correction: point-countercharge");
    std::string path = write_temporary_yaml(yaml_content);
    YAMLInputParser parser;
    EXPECT_THROW(parser.parse(path), std::runtime_error);
    std::remove(path.c_str());
}

// ── Verbosity ───────────────────────────────────────────────────────────

TEST(YAMLParser, VerbosityDefaultsToDefault) {
    YAMLInputParser parser;
    InputData data = parser.parse(silicon_nc_input);
    EXPECT_EQ(data.verbosity, "default");
}

TEST(YAMLParser, VerbosityParsesHigh) {
    std::string yaml_content = replace_first(full_manifest_yaml,
                                             "control:\n  calculation: scf",
                                             "control:\n  calculation: scf\n  verbosity: high");
    std::string path = write_temporary_yaml(yaml_content);
    YAMLInputParser parser;
    InputData data = parser.parse(path);
    std::remove(path.c_str());
    EXPECT_EQ(data.verbosity, "high");
}

TEST(YAMLParser, VerbosityRejectsInvalidValue) {
    /**
     * "loud" rather than one of spdlog's own level names, because the failure
     * this guards against is the parser handing the value straight to
     * spdlog::level::from_str -- which maps anything it does not recognize to
     * "off", turning a typo into a silent run.
     */
    std::string yaml_content = replace_first(full_manifest_yaml,
                                             "control:\n  calculation: scf",
                                             "control:\n  calculation: scf\n  verbosity: loud");
    std::string path = write_temporary_yaml(yaml_content);
    YAMLInputParser parser;
    EXPECT_THROW(parser.parse(path), std::runtime_error);
    std::remove(path.c_str());
}

// ── BerkeleyGW mean-field export ────────────────────────────────────────

TEST(YAMLParser, MeanFieldExportDefaultsToNone) {
    YAMLInputParser parser;
    InputData data = parser.parse(silicon_nc_input);
    EXPECT_EQ(data.mean_field_export, "none");
}

TEST(YAMLParser, MeanFieldExportParsesBerkeleyGW) {
    std::string yaml_content =
        replace_first(full_manifest_yaml,
                      "control:\n  calculation: scf",
                      "control:\n  calculation: scf\n  mean_field_export: berkeleygw");
    std::string path = write_temporary_yaml(yaml_content);
    YAMLInputParser parser;
    InputData data = parser.parse(path);
    std::remove(path.c_str());
    EXPECT_EQ(data.mean_field_export, "berkeleygw");
}

TEST(YAMLParser, MeanFieldExportRejectsInvalidValue) {
    std::string yaml_content =
        replace_first(full_manifest_yaml,
                      "control:\n  calculation: scf",
                      "control:\n  calculation: scf\n  mean_field_export: yambo");
    std::string path = write_temporary_yaml(yaml_content);
    YAMLInputParser parser;
    EXPECT_THROW(parser.parse(path), std::runtime_error);
    std::remove(path.c_str());
}

// ── DFT+U occupation seeding ────────────────────────────────────────────

TEST(YAMLParser, HubbardStartingEigenvaluesParseToZeroBasedForm) {
    std::string yaml_content =
        replace_first(full_manifest_yaml,
                      "    hubbard:",
                      "    hubbard_starting_eigenvalues:\n"
                      "      - {species: Si, eigenvalue: 3, spin: 1, value: 0.9}\n"
                      "    hubbard:");
    std::string path = write_temporary_yaml(yaml_content);
    YAMLInputParser parser;
    InputData data = parser.parse(path);
    std::remove(path.c_str());

    ASSERT_EQ(data.hubbard_starting_eigenvalues.size(), 1u);
    EXPECT_EQ(data.hubbard_starting_eigenvalues[0].species_symbol, "Si");
    EXPECT_EQ(data.hubbard_starting_eigenvalues[0].eigenvalue_index, 2); // 1-based in, 0-based out
    EXPECT_EQ(data.hubbard_starting_eigenvalues[0].spin_index, 0);
    EXPECT_DOUBLE_EQ(data.hubbard_starting_eigenvalues[0].target_occupation, 0.9);
}

TEST(YAMLParser, HubbardStartingEigenvaluesRejectBadSpinAndValue) {
    for (const std::string& bad_entry :
         {std::string("      - {species: Si, eigenvalue: 3, spin: 3, value: 0.9}\n"),
          std::string("      - {species: Si, eigenvalue: 3, spin: 1, value: 1.5}\n"),
          std::string("      - {species: Si, eigenvalue: 0, spin: 1, value: 0.9}\n")}) {
        std::string yaml_content =
            replace_first(full_manifest_yaml,
                          "    hubbard:",
                          "    hubbard_starting_eigenvalues:\n" + bad_entry + "    hubbard:");
        std::string path = write_temporary_yaml(yaml_content);
        YAMLInputParser parser;
        EXPECT_THROW(parser.parse(path), std::runtime_error);
        std::remove(path.c_str());
    }
}

/**
 * ── Schema-driven key allow-lists ────────────────────────────────────
 *
 * reject_unknown_keys() no longer carries its own copy of the accepted keys.
 * It looks each level up in q3::generated::input_schema, generated from
 * schemas/q3/input.json, so adding a manifest key means editing the schema
 * and regenerating -- the two cannot drift apart the way a hand-maintained
 * list here and a schema over there could. These tests pin that coupling: if
 * an object stops being closed in the schema, or a level's name changes, the
 * lookup starts throwing for *every* manifest carrying that block, and these
 * fail first with the path that broke.
 */

/// Every level parse() validates, in the spelling yaml.hpp passes.
static const std::vector<std::string> validated_schema_paths = {
    "", // manifest root
    "model",
    "method",
    "method.cutoffs",
    "method.scf",
    "method.scf.hubbard[]",
    "method.scf.hubbard_starting_eigenvalues[]",
    "method.scf.mixing",
    "method.scf.smearing",
    "method.scf.kgrid",
    "method.band_structure",
    "control",
    "control.band_path[]"};

TEST(InputSchemaKeys, EveryLevelTheParserValidatesIsClosedInTheSchema) {
    for (const std::string& schema_path : validated_schema_paths) {
        EXPECT_NO_THROW(generated::input_schema::allowed_keys(schema_path))
            << "schemas/q3/input.json must close the object at '" << schema_path
            << "' with \"additionalProperties\": false";
        EXPECT_FALSE(generated::input_schema::allowed_keys(schema_path).empty())
            << "schemas/q3/input.json lists no keys at '" << schema_path << "'";
    }
}

TEST(InputSchemaKeys, BothScfSpellingsShareOneAllowList) {
    /**
     * `scf` and `self_consistent_field` are one $ref'd definition in the
     * schema, so a key added under either spelling is accepted under both.
     */
    EXPECT_EQ(generated::input_schema::allowed_keys("method.scf"),
              generated::input_schema::allowed_keys("method.self_consistent_field"));
}

TEST(InputSchemaKeys, MaterialIsDeliberatelyLeftOpen) {
    /**
     * An ESSE material legitimately carries fields Q3 doesn't consume, so the
     * schema closes no object under `material` and the parser never asks it
     * to -- asking would be a bug in yaml.hpp, and throws rather than
     * silently rejecting the user's structure.
     */
    EXPECT_THROW(generated::input_schema::allowed_keys("material"), std::runtime_error);
}

TEST(InputSchemaKeys, ScfAllowListCoversTheKeysTheParserReads) {
    /**
     * Spot-check the direction the schema alone can't guarantee: keys yaml.hpp
     * actually reads out of method.scf must be ones the schema lists, or a
     * manifest using them would be rejected before the read ever happens.
     */
    const auto& allowed = generated::input_schema::allowed_keys("method.scf");
    for (const std::string& key : {"convergence_threshold",
                                   "max_iterations",
                                   "diagonalization",
                                   "number_of_bands",
                                   "total_charge",
                                   "hubbard",
                                   "hubbard_projectors",
                                   "hubbard_starting_eigenvalues",
                                   "hubbard_fixed_occupation_iterations",
                                   "mixing",
                                   "number_of_spin_components",
                                   "starting_magnetization",
                                   "smearing",
                                   "kgrid"}) {
        EXPECT_NE(std::find(allowed.begin(), allowed.end(), key), allowed.end())
            << "method.scf." << key << " is read by yaml.hpp but missing from the schema";
    }
    // The QE spelling stays out: the parser answers it with its own message.
    EXPECT_EQ(std::find(allowed.begin(), allowed.end(), "nspin"), allowed.end());
}

TEST(YAMLParserUnknownKeys, RejectionNamesTheManifestLevel) {
    /**
     * The message a user sees is built from the schema path, so a nested
     * level still reads as its dotted manifest key and an array's items read
     * as an "... entry" -- not as the schema's internal "[]" spelling.
     */
    std::string yaml_content =
        replace_first(full_manifest_yaml, "      beta: 0.7", "      bta: 0.7");
    std::string path = write_temporary_yaml(yaml_content);
    YAMLInputParser parser;
    try {
        parser.parse(path);
        ADD_FAILURE() << "expected an unknown-key throw for method.scf.mixing.bta";
    } catch (const std::runtime_error& error) {
        EXPECT_NE(std::string(error.what()).find("in method.scf.mixing"), std::string::npos)
            << error.what();
    }
    std::remove(path.c_str());

    yaml_content = replace_first(full_manifest_yaml, "        u_ev: 3.0", "        u_eV: 3.0");
    path = write_temporary_yaml(yaml_content);
    try {
        parser.parse(path);
        ADD_FAILURE() << "expected an unknown-key throw for method.scf.hubbard[].u_eV";
    } catch (const std::runtime_error& error) {
        EXPECT_NE(std::string(error.what()).find("in method.scf.hubbard entry"), std::string::npos)
            << error.what();
    }
    std::remove(path.c_str());
}

TEST(YAMLParserUnknownKeys, RetiredNspinSpellingNamesItsReplacement) {
    /**
     * `nspin` is not in the schema, but it must not fall through to the
     * generic "unknown key" message: a QE deck's spelling gets pointed at the
     * manifest one instead.
     */
    std::string yaml_content = replace_first(full_manifest_yaml, "  scf:", "  scf:\n    nspin: 2");
    std::string path = write_temporary_yaml(yaml_content);
    YAMLInputParser parser;
    try {
        parser.parse(path);
        ADD_FAILURE() << "expected a throw for method.scf.nspin";
    } catch (const std::runtime_error& error) {
        EXPECT_NE(std::string(error.what()).find("number_of_spin_components"), std::string::npos)
            << error.what();
    }
    std::remove(path.c_str());
}

// ── control.interpolation ──────────────────────────────────────────────────

namespace {

/// The fixture with `calculation: scf` replaced by `calculation` and an interpolation block.
std::string manifest_with_interpolation(const std::string& calculation, const std::string& block) {
    return replace_first(full_manifest_yaml,
                         "calculation: scf",
                         "calculation: " + calculation + "\n  interpolation:\n" + block);
}

std::string parse_error_message(const std::string& yaml_content) {
    std::string path = write_temporary_yaml(yaml_content);
    YAMLInputParser parser;
    std::string message;
    try {
        parser.parse(path);
    } catch (const std::runtime_error& error) {
        message = error.what();
    }
    std::remove(path.c_str());
    return message;
}

} // namespace

TEST(YAMLParserInterpolation, BlockIsParsedOnAnNscfBuildStep) {
    std::string path =
        write_temporary_yaml(manifest_with_interpolation("nscf",
                                                         "    method: wannier\n"
                                                         "    grid: [4, 4, 2]\n"
                                                         "    projection_centers:\n"
                                                         "      - [0.0, 0.0, 0.0]\n"
                                                         "      - [0.5, 0.0, 0.0]\n"
                                                         "    projection_decay_bohr: 1.5\n"
                                                         "    export_formats: [wannier90]\n"));
    YAMLInputParser parser;
    InputData data = parser.parse(path);
    std::remove(path.c_str());

    EXPECT_EQ(data.calculation, "non_self_consistent_field");
    ASSERT_TRUE(data.interpolation.requested());
    EXPECT_EQ(data.interpolation.method, "wannier");
    EXPECT_EQ(data.interpolation.grid, (std::array<int, 3>{4, 4, 2}));
    ASSERT_EQ(data.interpolation.projection_centers.size(), 2u);
    EXPECT_DOUBLE_EQ(data.interpolation.projection_centers[1].x, 0.5);
    EXPECT_DOUBLE_EQ(data.interpolation.projection_decay_bohr, 1.5);
    ASSERT_EQ(data.interpolation.export_formats.size(), 1u);
    EXPECT_EQ(data.interpolation.export_formats[0], "wannier90");
    EXPECT_FALSE(data.interpolation.wants_density_of_states());
}

TEST(YAMLParserInterpolation, EvaluationStepNeedsNoGridAndCarriesTheDosMesh) {
    std::string path = write_temporary_yaml(manifest_with_interpolation(
        "interpolation", "    method: wannier\n    dos_mesh: [24, 24, 24]\n"));
    YAMLInputParser parser;
    InputData data = parser.parse(path);
    std::remove(path.c_str());

    EXPECT_EQ(data.calculation, "interpolation");
    EXPECT_FALSE(data.interpolation.has_grid());
    EXPECT_TRUE(data.interpolation.wants_density_of_states());
    EXPECT_EQ(data.interpolation.dos_mesh, (std::array<int, 3>{24, 24, 24}));
}

TEST(YAMLParserInterpolation, TheFrozenWindowIsAutomaticUnlessANumberIsGiven) {
    /**
     * `auto` is the default and the word: both leave the window to the
     * classifier. A number is applied as given, and anything else is refused
     * by name rather than read as zero.
     */
    std::string path = write_temporary_yaml(manifest_with_interpolation(
        "interpolation", "    method: wannier\n    disentanglement: projectability\n"));
    YAMLInputParser parser;
    InputData by_default = parser.parse(path);
    std::remove(path.c_str());
    EXPECT_TRUE(by_default.interpolation.frozen_window_is_automatic);
    EXPECT_EQ(by_default.interpolation.localization_maximum_iterations, 3000);

    path = write_temporary_yaml(
        manifest_with_interpolation("interpolation",
                                    "    method: wannier\n    frozen_window_above_fermi_ev: auto\n"
                                    "    localization_maximum_iterations: 12\n"));
    InputData spelled_out = parser.parse(path);
    std::remove(path.c_str());
    EXPECT_TRUE(spelled_out.interpolation.frozen_window_is_automatic);
    EXPECT_EQ(spelled_out.interpolation.localization_maximum_iterations, 12);

    path = write_temporary_yaml(manifest_with_interpolation(
        "interpolation", "    method: wannier\n    frozen_window_above_fermi_ev: -5.0\n"));
    InputData explicit_window = parser.parse(path);
    std::remove(path.c_str());
    EXPECT_FALSE(explicit_window.interpolation.frozen_window_is_automatic);
    EXPECT_DOUBLE_EQ(explicit_window.interpolation.frozen_window_above_fermi_ev, -5.0);

    EXPECT_NE(
        parse_error_message(
            manifest_with_interpolation(
                "interpolation", "    method: wannier\n    frozen_window_above_fermi_ev: wide\n"))
            .find("frozen_window_above_fermi_ev"),
        std::string::npos);
    EXPECT_NE(
        parse_error_message(
            manifest_with_interpolation(
                "interpolation", "    method: wannier\n    localization_maximum_iterations: 0\n"))
            .find("localization_maximum_iterations"),
        std::string::npos);
    EXPECT_NE(
        parse_error_message(manifest_with_interpolation("nscf",
                                                        "    method: wannier\n    grid: [4, 4, 4]\n"
                                                        "    localization_maximum_iterations: 5\n"))
            .find("has no effect on calculation"),
        std::string::npos)
        << "an evaluation key on the build step is refused like the others";
}

TEST(YAMLParserInterpolation, AbsentBlockLeavesTheSpecificationEmpty) {
    std::string path = write_temporary_yaml(full_manifest_yaml);
    YAMLInputParser parser;
    InputData data = parser.parse(path);
    std::remove(path.c_str());
    EXPECT_FALSE(data.interpolation.requested());
}

TEST(YAMLParserInterpolation, MethodIsRequired) {
    std::string message =
        parse_error_message(manifest_with_interpolation("nscf", "    grid: [4, 4, 4]\n"));
    EXPECT_NE(message.find("control.interpolation.method"), std::string::npos) << message;
}

TEST(YAMLParserInterpolation, BuildStepWithoutGridIsRefused) {
    std::string message =
        parse_error_message(manifest_with_interpolation("nscf", "    method: wannier\n"));
    EXPECT_NE(message.find("control.interpolation.grid"), std::string::npos) << message;
}

TEST(YAMLParserInterpolation, MalformedGridIsRefused) {
    std::string message = parse_error_message(
        manifest_with_interpolation("nscf", "    method: wannier\n    grid: [4, 4]\n"));
    EXPECT_NE(message.find("three positive integers"), std::string::npos) << message;
    message = parse_error_message(
        manifest_with_interpolation("nscf", "    method: wannier\n    grid: [4, 0, 4]\n"));
    EXPECT_NE(message.find("three positive integers"), std::string::npos) << message;
}

TEST(YAMLParserInterpolation, BlockOnACalculationThatWouldIgnoreItIsRefused) {
    /**
     * An SCF never reads the block, so accepting it would let the run finish
     * having built nothing. The message must name both calculation types that
     * do consume it.
     */
    std::string message = parse_error_message(
        manifest_with_interpolation("scf", "    method: wannier\n    grid: [4, 4, 4]\n"));
    EXPECT_NE(message.find("would ignore it"), std::string::npos) << message;
    EXPECT_NE(message.find("q3_utility interpolate"), std::string::npos) << message;
}

TEST(YAMLParserInterpolation, UnknownKeyInsideTheBlockIsRefused) {
    std::string message = parse_error_message(manifest_with_interpolation(
        "nscf", "    method: wannier\n    grid: [4, 4, 4]\n    projection_centres: []\n"));
    EXPECT_NE(message.find("projection_centres"), std::string::npos) << message;
}

TEST(YAMLParserInterpolation, MalformedProjectionCentreIsRefused) {
    std::string message = parse_error_message(manifest_with_interpolation(
        "nscf",
        "    method: wannier\n    grid: [4, 4, 4]\n    projection_centers:\n      - [0.0, 0.0]\n"));
    EXPECT_NE(message.find("projection_centers"), std::string::npos) << message;
}

TEST(YAMLParserInterpolation, UnknownMethodIsRefusedBeforeAnyCalculation) {
    std::string message = parse_error_message(
        manifest_with_interpolation("nscf", "    method: wannier90\n    grid: [4, 4, 4]\n"));
    EXPECT_NE(message.find("control.interpolation.method must be 'wannier'"), std::string::npos)
        << message;
}

TEST(YAMLParserInterpolation, UnknownExportFormatIsRefusedBeforeAnyCalculation) {
    /// A typo used to surface only after the full-grid NSCF had run, and lost its results file.
    std::string message = parse_error_message(manifest_with_interpolation(
        "nscf", "    method: wannier\n    grid: [4, 4, 4]\n    export_formats: [wannier-90]\n"));
    EXPECT_NE(message.find("control.interpolation.export_formats"), std::string::npos) << message;
    EXPECT_NE(message.find("wannier-90"), std::string::npos) << message;
}

TEST(YAMLParserInterpolation, NonPositiveProjectionDecayIsRefused) {
    std::string message = parse_error_message(manifest_with_interpolation(
        "nscf", "    method: wannier\n    grid: [4, 4, 4]\n    projection_decay_bohr: 0.0\n"));
    EXPECT_NE(message.find("projection_decay_bohr must be positive"), std::string::npos) << message;
}

TEST(YAMLParserInterpolation, EvaluationStepRefusesABandPathVertexWithoutCoordinates) {
    /**
     * q3_utility interpolate walks the vertices itself and has no high-symmetry
     * table; a vertex defaulted to the origin would be evaluated as Gamma
     * under another label, so the parser refuses it on this calculation type.
     */
    const std::string manifest = replace_first(
        manifest_with_interpolation("interpolation", "    method: wannier\n"),
        "  band_path:\n    - label: G\n      coordinates: [0.0, 0.0, 0.0]\n",
        "  band_path:\n    - label: X\n      coordinates: [0.0, 0.5, 0.5]\n    - L\n");
    std::string message = parse_error_message(manifest);
    EXPECT_NE(message.find("vertex 'L' has no coordinates"), std::string::npos) << message;
}

TEST(YAMLParserInterpolation, BuildStepAcceptsALabelOnlyBandPathVertex) {
    /// Only the evaluation step walks the vertices; the flavor needs the labels alone.
    const std::string manifest = replace_first(
        manifest_with_interpolation("nscf", "    method: wannier\n    grid: [4, 4, 4]\n"),
        "  band_path:\n    - label: G\n      coordinates: [0.0, 0.0, 0.0]\n",
        "  band_path:\n    - G\n    - L\n");
    std::string path = write_temporary_yaml(manifest);
    YAMLInputParser parser;
    InputData data = parser.parse(path);
    std::remove(path.c_str());
    ASSERT_EQ(data.band_path.size(), 2u);
    EXPECT_EQ(data.band_path[1].first, "L");
}

TEST(YAMLParserInterpolation, ABandRangeIsParsed) {
    std::string path = write_temporary_yaml(manifest_with_interpolation(
        "nscf", "    method: wannier\n    grid: [4, 4, 4]\n    bands: [6, 9]\n"));
    YAMLInputParser parser;
    InputData data = parser.parse(path);
    std::remove(path.c_str());

    EXPECT_TRUE(data.interpolation.has_band_selection());
    EXPECT_EQ(data.interpolation.first_band_index(), 5) << "the key is 1-based, the index is not";
    EXPECT_EQ(data.interpolation.number_of_selected_bands(), 4);
}

TEST(YAMLParserInterpolation, NoBandRangeMeansTheLowestBands) {
    std::string path = write_temporary_yaml(
        manifest_with_interpolation("nscf", "    method: wannier\n    grid: [4, 4, 4]\n"));
    YAMLInputParser parser;
    InputData data = parser.parse(path);
    std::remove(path.c_str());

    EXPECT_FALSE(data.interpolation.has_band_selection());
    EXPECT_EQ(data.interpolation.first_band_index(), 0);
}

TEST(YAMLParserInterpolation, AKeyWrittenOnTheWrongStepIsRefusedRatherThanIgnored) {
    /**
     * One block serves the build step and the evaluation step, and until this
     * refusal a key written on the wrong one was silently dropped: a
     * `disentanglement` on the NSCF manifest would build an entangled model
     * that nothing had disentangled, and the run would exit cleanly.
     */
    const std::string evaluation_key_on_build = parse_error_message(manifest_with_interpolation(
        "nscf", "    method: wannier\n    grid: [4, 4, 4]\n    disentanglement: projectability\n"));
    EXPECT_NE(evaluation_key_on_build.find("control.interpolation.disentanglement"),
              std::string::npos)
        << evaluation_key_on_build;
    EXPECT_NE(evaluation_key_on_build.find("q3_utility interpolate"), std::string::npos)
        << "the message must say where the key does belong: " << evaluation_key_on_build;

    const std::string build_key_on_evaluation = parse_error_message(
        manifest_with_interpolation("interpolation", "    method: wannier\n    bands: [6, 9]\n"));
    EXPECT_NE(build_key_on_evaluation.find("control.interpolation.bands"), std::string::npos)
        << build_key_on_evaluation;
}

TEST(YAMLParserInterpolation, AnInvertedOrZeroBandRangeIsRefused) {
    std::string inverted = parse_error_message(manifest_with_interpolation(
        "nscf", "    method: wannier\n    grid: [4, 4, 4]\n    bands: [9, 6]\n"));
    EXPECT_NE(inverted.find("control.interpolation.bands"), std::string::npos) << inverted;

    std::string zero = parse_error_message(manifest_with_interpolation(
        "nscf", "    method: wannier\n    grid: [4, 4, 4]\n    bands: [0, 4]\n"));
    EXPECT_NE(zero.find("control.interpolation.bands"), std::string::npos) << zero;

    std::string single = parse_error_message(manifest_with_interpolation(
        "nscf", "    method: wannier\n    grid: [4, 4, 4]\n    bands: [4]\n"));
    EXPECT_NE(single.find("[first, last]"), std::string::npos) << single;
}

TEST(YAMLParserInterpolation, TrialShellsAreReadOnTheBuildStep) {
    std::string path = write_temporary_yaml(manifest_with_interpolation(
        "nscf", "    method: wannier\n    grid: [4, 4, 4]\n    trial_shells:\n      Si: [0, 2]\n"));
    YAMLInputParser parser;
    InputData data = parser.parse(path);
    std::remove(path.c_str());
    ASSERT_EQ(data.interpolation.trial_shells.size(), 1u);
    EXPECT_EQ(data.interpolation.trial_shells.at("Si"), (std::vector<int>{0, 2}));
}

TEST(YAMLParserInterpolation, TrialShellsAreRefusedWhereTheyCannotMeanWhatTheySay) {
    const std::string on_evaluation = parse_error_message(manifest_with_interpolation(
        "interpolation", "    method: wannier\n    trial_shells:\n      Si: [0]\n"));
    EXPECT_NE(on_evaluation.find("control.interpolation.trial_shells"), std::string::npos)
        << on_evaluation;

    const std::string with_centres = parse_error_message(manifest_with_interpolation(
        "nscf",
        "    method: wannier\n    grid: [4, 4, 4]\n    trial_shells:\n      Si: [0]\n"
        "    projection_centers:\n      - [0.125, 0.125, 0.125]\n"));
    EXPECT_NE(with_centres.find("one or the other"), std::string::npos) << with_centres;

    const std::string repeated = parse_error_message(manifest_with_interpolation(
        "nscf", "    method: wannier\n    grid: [4, 4, 4]\n    trial_shells:\n      Si: [0, 0]\n"));
    EXPECT_NE(repeated.find("distinct"), std::string::npos) << repeated;

    const std::string negative = parse_error_message(manifest_with_interpolation(
        "nscf", "    method: wannier\n    grid: [4, 4, 4]\n    trial_shells:\n      Si: [-1]\n"));
    EXPECT_NE(negative.find("non-negative"), std::string::npos) << negative;
}

TEST(YAMLParserInterpolation, TheDisentanglementBudgetAndToleranceAreEvaluationKeys) {
    std::string path = write_temporary_yaml(
        manifest_with_interpolation("interpolation",
                                    "    method: wannier\n    disentanglement: projectability\n"
                                    "    disentanglement_maximum_iterations: 16000\n"
                                    "    disentanglement_convergence_tolerance: 1.0e-7\n"));
    YAMLInputParser parser;
    InputData data = parser.parse(path);
    std::remove(path.c_str());
    EXPECT_EQ(data.interpolation.disentanglement_maximum_iterations, 16000);
    EXPECT_DOUBLE_EQ(data.interpolation.disentanglement_convergence_tolerance, 1.0e-7);

    const std::string zero_budget = parse_error_message(manifest_with_interpolation(
        "interpolation", "    method: wannier\n    disentanglement_maximum_iterations: 0\n"));
    EXPECT_NE(zero_budget.find("must be positive"), std::string::npos) << zero_budget;

    const std::string on_build = parse_error_message(
        manifest_with_interpolation("nscf",
                                    "    method: wannier\n    grid: [4, 4, 4]\n    "
                                    "disentanglement_convergence_tolerance: 1.0e-7\n"));
    EXPECT_NE(on_build.find("control.interpolation.disentanglement_convergence_tolerance"),
              std::string::npos)
        << on_build;
}

/**
 * ── Davidson basis expansion factor ──────────────────────────────────
 *
 * The working basis is two arrays of factor x nbnd vectors, which is the
 * largest single term in the solver's memory, so a deck needs to be able to
 * choose it. The default is 2; 4 halves the number of restarts at twice the
 * footprint, and is still accepted.
 */

static const std::string davidson_factor_yaml_template = R"(
material:
  lattice:
    type: SC
    a: 5.45
    units: bohr
  basis:
    elements:
      - id: 0
        value: Si
    coordinates:
      - id: 0
        value: [0.0, 0.0, 0.0]
method:
  type: pseudopotential
  scf:
    davidson:
      basis_expansion_factor: FACTOR
control:
  calculation: scf
)";

static std::string davidson_factor_yaml(const std::string& factor) {
    std::string content = davidson_factor_yaml_template;
    content.replace(content.find("FACTOR"), 6, factor);
    return content;
}

TEST(YAMLInputParser, DavidsonBasisExpansionFactorDefaultsToTwo) {
    /**
     * Strip the davidson block from the template: the default is what the
     * parser supplies when the deck says nothing.
     */
    std::string content = davidson_factor_yaml_template;
    std::size_t block_start = content.find("davidson:");
    ASSERT_NE(block_start, std::string::npos);
    block_start = content.rfind('\n', block_start) + 1;
    std::size_t block_end = content.find('\n', content.find("basis_expansion_factor: FACTOR")) + 1;
    content.erase(block_start, block_end - block_start);
    std::string path = write_temporary_yaml(content);
    YAMLInputParser parser;
    InputData data = parser.parse(path);
    std::remove(path.c_str());
    EXPECT_EQ(data.davidson_basis_expansion_factor, 2);
    EXPECT_FALSE(data.davidson_basis_expansion_factor_set_by_deck)
        << "a default must not read as the deck's choice, or a non-self-consistent pass keeps it";
}

TEST(YAMLInputParser, DavidsonBasisExpansionFactorFourIsStillAccepted) {
    std::string path = write_temporary_yaml(davidson_factor_yaml("4"));
    YAMLInputParser parser;
    InputData data = parser.parse(path);
    std::remove(path.c_str());
    EXPECT_EQ(data.davidson_basis_expansion_factor, 4);
}

TEST(YAMLInputParser, DavidsonBasisExpansionFactorIsReadFromTheDeck) {
    std::string path = write_temporary_yaml(davidson_factor_yaml("2"));
    YAMLInputParser parser;
    InputData data = parser.parse(path);
    std::remove(path.c_str());
    EXPECT_EQ(data.davidson_basis_expansion_factor, 2);
    EXPECT_TRUE(data.davidson_basis_expansion_factor_set_by_deck);
}

/**
 * A factor below two cannot work: the basis would have no room to grow past
 * the bands it is solving for. Refuse it at parse time rather than deadlock
 * or silently clamp.
 */
TEST(YAMLInputParser, DavidsonBasisExpansionFactorBelowTwoIsRefused) {
    std::string path = write_temporary_yaml(davidson_factor_yaml("1"));
    YAMLInputParser parser;
    EXPECT_THROW(parser.parse(path), std::runtime_error);
    std::remove(path.c_str());
}

TEST(YAMLInputParser, UnknownDavidsonKeyIsRefused) {
    std::string content = davidson_factor_yaml("4");
    content.replace(content.find("basis_expansion_factor"), 22, "basis_expansion_facter");
    std::string path = write_temporary_yaml(content);
    YAMLInputParser parser;
    EXPECT_THROW(parser.parse(path), std::runtime_error);
    std::remove(path.c_str());
}

/**
 * A same-species antiferromagnet: both sites are plain `Cr`, and the opposed
 * starting moments come from the per-atom list instead of sublattice labels.
 */
static const std::string chromium_afm_per_atom_yaml = R"(
material:
  lattice:
    type: SC
    a: 5.45
    units: bohr
  basis:
    elements:
      - id: 0
        value: Cr
      - id: 1
        value: Cr
    coordinates:
      - id: 0
        value: [0.0, 0.0, 0.0]
      - id: 1
        value: [0.5, 0.5, 0.5]
method:
  type: pseudopotential
  subtype: norm-conserving
  scf:
    number_of_spin_components: 2
    atomic_starting_magnetization: [0.5, -0.5]
control:
  calculation: scf
)";

TEST(YAMLParserSpinTest, ParsesPerAtomStartingMagnetization) {
    std::string path = write_temporary_yaml(chromium_afm_per_atom_yaml);
    YAMLInputParser parser;
    InputData data = parser.parse(path);
    std::remove(path.c_str());

    ASSERT_EQ(data.species.size(), 1u);
    EXPECT_TRUE(data.starting_magnetization.empty());
    ASSERT_EQ(data.atomic_starting_magnetization.size(), 2u);
    EXPECT_DOUBLE_EQ(data.atomic_starting_magnetization[0], 0.5);
    EXPECT_DOUBLE_EQ(data.atomic_starting_magnetization[1], -0.5);
}

TEST(YAMLParserSpinTest, PerAtomMagnetizationMustListEveryAtom) {
    std::string yaml_content = chromium_afm_per_atom_yaml;
    const std::string list = "[0.5, -0.5]";
    yaml_content.replace(yaml_content.find(list), list.size(), "[0.5]");
    std::string path = write_temporary_yaml(yaml_content);
    YAMLInputParser parser;
    EXPECT_THROW(parser.parse(path), std::runtime_error);
    std::remove(path.c_str());
}

TEST(YAMLParserSpinTest, PerAtomMagnetizationOutOfRangeThrows) {
    std::string yaml_content = chromium_afm_per_atom_yaml;
    const std::string list = "[0.5, -0.5]";
    yaml_content.replace(yaml_content.find(list), list.size(), "[0.5, -1.5]");
    std::string path = write_temporary_yaml(yaml_content);
    YAMLInputParser parser;
    EXPECT_THROW(parser.parse(path), std::runtime_error);
    std::remove(path.c_str());
}

TEST(YAMLParserSpinTest, PerSpeciesAndPerAtomMagnetizationTogetherThrow) {
    std::string yaml_content = chromium_afm_per_atom_yaml;
    const std::string list = "    atomic_starting_magnetization: [0.5, -0.5]\n";
    yaml_content.replace(yaml_content.find(list),
                         list.size(),
                         list + "    starting_magnetization:\n      Cr: 0.5\n");
    std::string path = write_temporary_yaml(yaml_content);
    YAMLInputParser parser;
    EXPECT_THROW(parser.parse(path), std::runtime_error);
    std::remove(path.c_str());
}
