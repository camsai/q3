#include <filesystem>
#include <gtest/gtest.h>

#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "io/input/parser/qe.hpp"
#include "io/input/parser/yaml.hpp"
#include "io/input/system_builder.hpp"
#include "utils/temporary_path.hpp"

using namespace q3;

/**
 * Helper: write a string to a temporary file and return the path.
 * The caller is responsible for removing the file after use.
 */
static std::string write_temporary_input_file(const std::string& content) {
    std::string temporary_path = test_support::make_temporary_path(".in");
    std::ofstream output_stream(temporary_path);
    output_stream << content;
    output_stream.close();
    return temporary_path;
}

static const std::string silicon_fcc_input = R"(
&CONTROL
  calculation = 'scf'
  pseudo_dir = './'
/
&SYSTEM
  ibrav = 2
  celldm(1) = 10.263048
  nat = 2
  ntyp = 1
  ecutwfc = 30.0
/
&ELECTRONS
  conv_thr = 1e-6
/
ATOMIC_SPECIES
  Si 28.08 Si.pz-vbc.UPF
ATOMIC_POSITIONS crystal
  Si 0.0 0.0 0.0
  Si 0.25 0.25 0.25
K_POINTS automatic
  4 4 4 1 1 1
)";

class QEParserTest : public ::testing::Test {
  protected:
    void SetUp() override {
        temporary_file_path = write_temporary_input_file(silicon_fcc_input);
        QEInputParser parser;
        parsed_data = parser.parse(temporary_file_path);
    }

    void TearDown() override {
        std::remove(temporary_file_path.c_str());
    }

    std::string temporary_file_path;
    InputData parsed_data;
};

TEST_F(QEParserTest, ParsesNumberOfAtoms) {
    EXPECT_EQ(parsed_data.number_of_atoms, 2);
}

TEST_F(QEParserTest, ParsesBravaisLatticeIndex) {
    EXPECT_EQ(parsed_data.bravais_lattice_index, 2);
}

TEST_F(QEParserTest, ParsesCellDimension1) {
    EXPECT_NEAR(parsed_data.cell_dimension_1, 10.263048, 1e-6);
}

TEST_F(QEParserTest, ParsesWavefunctionEnergyCutoff) {
    EXPECT_DOUBLE_EQ(parsed_data.wavefunction_energy_cutoff, 30.0);
}

TEST_F(QEParserTest, AlwaysAsksForTheExactWavefunctionGrid) {
    // A pw.x deck runs on pw.x's 4 x ecutwfc grid; the input has no switch for it.
    EXPECT_TRUE(parsed_data.use_exact_wavefunction_grid);
}

TEST_F(QEParserTest, ParsesConvergenceThreshold) {
    EXPECT_NEAR(parsed_data.convergence_threshold, 1e-6, 1e-12);
}

TEST_F(QEParserTest, ParsesKpointGridDimensions) {
    EXPECT_EQ(parsed_data.kpoint_grid_dimension_1, 4);
    EXPECT_EQ(parsed_data.kpoint_grid_dimension_2, 4);
    EXPECT_EQ(parsed_data.kpoint_grid_dimension_3, 4);
}

TEST_F(QEParserTest, ParsesKpointGridShifts) {
    EXPECT_EQ(parsed_data.kpoint_grid_shift_1, 1);
    EXPECT_EQ(parsed_data.kpoint_grid_shift_2, 1);
    EXPECT_EQ(parsed_data.kpoint_grid_shift_3, 1);
}

TEST_F(QEParserTest, ParsesAtomicSpeciesSymbol) {
    ASSERT_EQ(parsed_data.species.size(), 1u);
    EXPECT_EQ(parsed_data.species[0].symbol, "Si");
}

TEST_F(QEParserTest, ParsesAtomicSpeciesMass) {
    ASSERT_EQ(parsed_data.species.size(), 1u);
    EXPECT_NEAR(parsed_data.species[0].mass, 28.08, 1e-6);
}

TEST_F(QEParserTest, ParsesAtomicPositions) {
    ASSERT_EQ(parsed_data.atoms.size(), 2u);
    EXPECT_NEAR(parsed_data.atoms[0].position.x, 0.0, 1e-12);
    EXPECT_NEAR(parsed_data.atoms[0].position.y, 0.0, 1e-12);
    EXPECT_NEAR(parsed_data.atoms[0].position.z, 0.0, 1e-12);
    EXPECT_NEAR(parsed_data.atoms[1].position.x, 0.25, 1e-12);
    EXPECT_NEAR(parsed_data.atoms[1].position.y, 0.25, 1e-12);
    EXPECT_NEAR(parsed_data.atoms[1].position.z, 0.25, 1e-12);
}

TEST_F(QEParserTest, ParsesCalculationType) {
    /**
     * QE's vocabulary is translated, not stored verbatim.
     *
     * This test previously asserted `"scf"` — the parser's own passthrough —
     * which is exactly why a real QE input file could not run: it parsed fine
     * and then died in `create_flavor` with "unsupported calculation type
     * 'scf'". The parser's job is to end at Q3's vocabulary.
     */
    EXPECT_EQ(parsed_data.calculation, "self_consistent_field");
}

TEST(QECalculationTypeTest, MapsEveryQeCalculationVerbToAQ3Flavor) {
    struct Case {
        const char* qe;
        const char* q3;
    };
    for (const Case& expected : {Case{"scf", "self_consistent_field"},
                                 Case{"nscf", "non_self_consistent_field"},
                                 Case{"bands", "non_self_consistent_field"},
                                 Case{"relax", "relaxation"},
                                 Case{"vc-relax", "variable_cell_relaxation"}}) {
        std::string input = std::string("&CONTROL\n  calculation = '") + expected.qe +
                            "'\n  pseudo_dir = './'\n/\n&SYSTEM\n  ibrav = 1\n"
                            "  celldm(1) = 5.45\n  nat = 1\n  ntyp = 1\n  ecutwfc = 20.0\n/\n"
                            "&ELECTRONS\n/\nATOMIC_SPECIES\n  Si 28.0855 Si.upf\n"
                            "ATOMIC_POSITIONS crystal\n  Si 0.0 0.0 0.0\n"
                            "K_POINTS automatic\n  1 1 1 0 0 0\n";
        auto path = std::filesystem::temp_directory_path() / "q3_qe_calculation_case.in";
        std::ofstream(path) << input;
        QEInputParser parser;
        EXPECT_EQ(parser.parse(path.string()).calculation, expected.q3)
            << "QE calculation='" << expected.qe << "'";
        std::filesystem::remove(path);
    }
}

TEST_F(QEParserTest, NonexistentFileThrows) {
    QEInputParser parser;
    EXPECT_THROW(parser.parse("/nonexistent/path/file.in"), std::runtime_error);
}

// ── Spin / per-species starting_magnetization ───────────────────────

static const std::string chromium_afm_input = R"(
&CONTROL
  calculation = 'scf'
  pseudo_dir = './'
/
&SYSTEM
  ibrav = 1
  celldm(1) = 5.45
  nat = 2
  ntyp = 2
  ecutwfc = 60.0
  nspin = 2
  starting_magnetization(1) = 0.5
  starting_magnetization(2) = -0.5
/
&ELECTRONS
  conv_thr = 1e-8
/
ATOMIC_SPECIES
  Cr1 51.996 Cr.upf
  Cr2 51.996 Cr.upf
ATOMIC_POSITIONS crystal
  Cr1 0.0 0.0 0.0
  Cr2 0.5 0.5 0.5
K_POINTS automatic
  4 4 4 0 0 0
)";

TEST(QEParserSpinTest, ParsesIndexedStartingMagnetization) {
    std::string path = write_temporary_input_file(chromium_afm_input);
    QEInputParser parser;
    InputData data = parser.parse(path);
    std::remove(path.c_str());

    EXPECT_EQ(data.number_of_spin_components, 2);
    ASSERT_EQ(data.starting_magnetization.size(), 2u);
    EXPECT_NEAR(data.starting_magnetization[0], 0.5, 1e-12);
    EXPECT_NEAR(data.starting_magnetization[1], -0.5, 1e-12);
}

TEST(QEParserSpinTest, ParsesSpeciesLabelsWithSuffixes) {
    std::string path = write_temporary_input_file(chromium_afm_input);
    QEInputParser parser;
    InputData data = parser.parse(path);
    std::remove(path.c_str());

    ASSERT_EQ(data.species.size(), 2u);
    EXPECT_EQ(data.species[0].symbol, "Cr1");
    EXPECT_EQ(data.species[1].symbol, "Cr2");
    ASSERT_EQ(data.atoms.size(), 2u);
    EXPECT_EQ(data.atoms[0].species_index, 0);
    EXPECT_EQ(data.atoms[1].species_index, 1);
}

TEST(QEParserSpinTest, UnindexedStartingMagnetizationMeansFirstSpecies) {
    std::string input = R"(
&SYSTEM
  ibrav = 3
  celldm(1) = 5.42
  nat = 1
  ntyp = 1
  ecutwfc = 90.0
  nspin = 2
  starting_magnetization = 0.4
/
ATOMIC_SPECIES
  Fe 55.845 Fe.upf
ATOMIC_POSITIONS crystal
  Fe 0.0 0.0 0.0
K_POINTS automatic
  12 12 12 0 0 0
)";
    std::string path = write_temporary_input_file(input);
    QEInputParser parser;
    InputData data = parser.parse(path);
    std::remove(path.c_str());

    EXPECT_EQ(data.number_of_spin_components, 2);
    ASSERT_EQ(data.starting_magnetization.size(), 1u);
    EXPECT_NEAR(data.starting_magnetization[0], 0.4, 1e-12);
}

TEST(QEParserSpinTest, DefaultIsUnpolarizedWithNoMagnetization) {
    std::string path = write_temporary_input_file(silicon_fcc_input);
    QEInputParser parser;
    InputData data = parser.parse(path);
    std::remove(path.c_str());

    EXPECT_EQ(data.number_of_spin_components, 1);
    EXPECT_TRUE(data.starting_magnetization.empty());
}

// ── CELL_PARAMETERS (ibrav = 0) ─────────────────────────────────────

static const std::string aluminum_slab_input = R"(
&CONTROL
  calculation = 'self_consistent_field'
  pseudo_dir = './'
/
&SYSTEM
  ibrav = 0
  nat = 1
  ntyp = 1
  ecutwfc = 30.0
/
&ELECTRONS
  conv_thr = 1e-8
/
ATOMIC_SPECIES
  Al 26.98 Al.upf
ATOMIC_POSITIONS crystal
  Al 0.0 0.0 0.0
CELL_PARAMETERS bohr
  10.0  0.0  0.0
   0.0 12.0  0.0
   0.0  0.0 30.0
K_POINTS automatic
  4 4 1 0 0 0
)";

TEST(QEParserCellParameters, BohrUnitsProduceExplicitVectors) {
    std::string path = write_temporary_input_file(aluminum_slab_input);
    QEInputParser parser;
    InputData data = parser.parse(path);
    std::remove(path.c_str());

    EXPECT_TRUE(data.has_explicit_lattice_vectors);
    EXPECT_EQ(data.bravais_lattice_index, 0);
    // celldm(1) absent: alat = |a1| = 10 bohr, rows in units of alat.
    EXPECT_NEAR(data.cell_dimension_1, 10.0, 1e-12);
    EXPECT_NEAR(data.lattice_vectors[0][0], 1.0, 1e-12);
    EXPECT_NEAR(data.lattice_vectors[1][1], 1.2, 1e-12);
    EXPECT_NEAR(data.lattice_vectors[2][2], 3.0, 1e-12);
}

TEST(QEParserCellParameters, AlatUnitsRequireCelldm) {
    std::string input = aluminum_slab_input;
    size_t position = input.find("CELL_PARAMETERS bohr");
    input.replace(position, std::string("CELL_PARAMETERS bohr").size(), "CELL_PARAMETERS alat");
    std::string path = write_temporary_input_file(input);
    QEInputParser parser;
    EXPECT_THROW(parser.parse(path), std::runtime_error);
    std::remove(path.c_str());
}

TEST(QEParserCellParameters, NonzeroIbravWithCellParametersThrows) {
    std::string input = aluminum_slab_input;
    size_t position = input.find("ibrav = 0");
    input.replace(position, std::string("ibrav = 0").size(), "ibrav = 2");
    std::string path = write_temporary_input_file(input);
    QEInputParser parser;
    EXPECT_THROW(parser.parse(path), std::runtime_error);
    std::remove(path.c_str());
}

TEST(QEParserCellParameters, MatchesYAMLVectorParser) {
    /**
     * Round-trip parity: the same physical cell through the QE CELL_PARAMETERS
     * card and through YAML lattice.vectors must produce identical lattice
     * data (alat + rows).
     */
    std::string qe_path = write_temporary_input_file(aluminum_slab_input);
    QEInputParser qe_parser;
    InputData qe_data = qe_parser.parse(qe_path);
    std::remove(qe_path.c_str());

    std::string yaml_content = R"(
material:
  lattice:
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
    std::string yaml_path = test_support::make_temporary_path(".yml");
    {
        std::ofstream output_stream(yaml_path);
        output_stream << yaml_content;
    }
    YAMLInputParser yaml_parser;
    InputData yaml_data = yaml_parser.parse(yaml_path);
    std::remove(yaml_path.c_str());

    EXPECT_NEAR(qe_data.cell_dimension_1, yaml_data.cell_dimension_1, 1e-12);
    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
            EXPECT_NEAR(
                qe_data.lattice_vectors[row][column], yaml_data.lattice_vectors[row][column], 1e-12)
                << "row " << row << " column " << column;
        }
    }
}

TEST(QEParserSpinTest, NoncollinearNspinRejected) {
    std::string input = chromium_afm_input;
    size_t position = input.find("nspin = 2");
    input.replace(position, std::string("nspin = 2").size(), "nspin = 4");
    std::string path = write_temporary_input_file(input);
    QEInputParser parser;
    EXPECT_THROW(parser.parse(path), std::runtime_error);
    std::remove(path.c_str());
}

TEST(QEParserSpinTest, MagnetizationIndexBeyondSpeciesCountThrows) {
    std::string input = chromium_afm_input;
    size_t position = input.find("starting_magnetization(2)");
    input.replace(
        position, std::string("starting_magnetization(2)").size(), "starting_magnetization(3)");
    std::string path = write_temporary_input_file(input);
    QEInputParser parser;
    EXPECT_THROW(parser.parse(path), std::runtime_error);
    std::remove(path.c_str());
}

TEST(QEParserSpinTest, ManifestSpinKeyRejectedWithGuidance) {
    /**
     * Mirrors YAMLParserSpinTest.NspinKeyRejectedWithGuidance (test_yaml.cpp):
     * the YAML manifest key belongs to the YAML parser only; a namelist input
     * must reject it rather than silently ignoring it and running unpolarized.
     */
    std::string input = chromium_afm_input;
    size_t position = input.find("nspin = 2");
    input.replace(position, std::string("nspin = 2").size(), "number_of_spin_components = 2");
    std::string path = write_temporary_input_file(input);
    QEInputParser parser;
    EXPECT_THROW(parser.parse(path), std::runtime_error);
    std::remove(path.c_str());
}

// ── Smearing / degauss ───────────────────────────────────────────────

static const std::string aluminum_smeared_input = R"(
&CONTROL
  calculation = 'scf'
  pseudo_dir = './'
/
&SYSTEM
  ibrav = 2
  celldm(1) = 7.50
  nat = 1
  ntyp = 1
  ecutwfc = 30.0
  occupations = 'smearing'
  smearing = 'marzari-vanderbilt'
  degauss = 0.02
/
&ELECTRONS
  conv_thr = 1e-8
/
ATOMIC_SPECIES
  Al 26.98 Al.upf
ATOMIC_POSITIONS crystal
  Al 0.0 0.0 0.0
K_POINTS automatic
  4 4 4 0 0 0
)";

TEST(QEParserSmearingTest, ParsesSmearingTypeAndDegauss) {
    std::string path = write_temporary_input_file(aluminum_smeared_input);
    QEInputParser parser;
    InputData data = parser.parse(path);
    std::remove(path.c_str());

    EXPECT_EQ(data.smearing, "marzari-vanderbilt");
    EXPECT_NEAR(data.smearing_width, 0.02, 1e-12);
}

TEST(QEParserSmearingTest, OccupationsSmearingWithoutExplicitTypeDefaultsToGaussian) {
    /**
     * QE's own compatibility rule: occupations='smearing' with no explicit
     * smearing key defaults to Gaussian (qe.hpp's occupations handling).
     */
    std::string input = aluminum_smeared_input;
    size_t position = input.find("smearing = 'marzari-vanderbilt'\n");
    input.erase(position, std::string("smearing = 'marzari-vanderbilt'\n").size());
    std::string path = write_temporary_input_file(input);
    QEInputParser parser;
    InputData data = parser.parse(path);
    std::remove(path.c_str());

    EXPECT_EQ(data.smearing, "gaussian");
    EXPECT_NEAR(data.smearing_width, 0.02, 1e-12);
}

TEST(QEParserSmearingTest, DefaultIsNoSmearing) {
    QEInputParser parser;
    InputData data = parser.parse(write_temporary_input_file(silicon_fcc_input));
    EXPECT_EQ(data.smearing, "none");
    EXPECT_NEAR(data.smearing_width, 0.0, 1e-12);
}

TEST(QEParserSmearingTest, ExplicitSmearingTypeSurvivesLaterOccupationsLine) {
    /**
     * Regression guard for the order-independence of the occupations/smearing
     * interaction: an explicit smearing type set before occupations='smearing'
     * must not be overwritten by the Gaussian default.
     */
    std::string input = R"(
&CONTROL
  calculation = 'scf'
  pseudo_dir = './'
/
&SYSTEM
  ibrav = 2
  celldm(1) = 7.50
  nat = 1
  ntyp = 1
  ecutwfc = 30.0
  smearing = 'fermi-dirac'
  degauss = 0.02
  occupations = 'smearing'
/
&ELECTRONS
  conv_thr = 1e-8
/
ATOMIC_SPECIES
  Al 26.98 Al.upf
ATOMIC_POSITIONS crystal
  Al 0.0 0.0 0.0
K_POINTS automatic
  4 4 4 0 0 0
)";
    std::string path = write_temporary_input_file(input);
    QEInputParser parser;
    InputData data = parser.parse(path);
    std::remove(path.c_str());

    EXPECT_EQ(data.smearing, "fermi-dirac");
}

/**
 * Namelist-key strictness.
 *
 * Until 2026-08-08 this parser ignored every key it did not recognize, the
 * same way it ignored the HUBBARD card — and an ignored key that changes
 * physics produces a converged, plausible, wrong number. These fix the four
 * outcomes a key can have: honoured, accepted-as-neutral, accepted-as-default,
 * and refused.
 */
namespace {

/// A minimal but complete silicon input with `extra_system_keys` spliced in.
std::string silicon_input_with(const std::string& extra_control_keys,
                               const std::string& extra_system_keys,
                               const std::string& extra_electrons_keys = "") {
    return R"(
&CONTROL
  calculation = 'scf'
  pseudo_dir = './'
)" + extra_control_keys +
           R"(
/
&SYSTEM
  ibrav = 2
  celldm(1) = 10.2
  nat = 2
  ntyp = 1
  ecutwfc = 30.0
)" + extra_system_keys +
           R"(
/
&ELECTRONS
  conv_thr = 1e-6
)" + extra_electrons_keys +
           R"(
/
ATOMIC_SPECIES
  Si 28.08 Si.upf
ATOMIC_POSITIONS crystal
  Si 0.0 0.0 0.0
  Si 0.25 0.25 0.25
K_POINTS automatic
  4 4 4 1 1 1
)";
}

InputData parse_text(const std::string& text) {
    std::string path = write_temporary_input_file(text);
    QEInputParser parser;
    try {
        InputData data = parser.parse(path);
        std::remove(path.c_str());
        return data;
    } catch (...) {
        std::remove(path.c_str());
        throw;
    }
}

} // namespace

TEST(QEParserStrictness, UnknownNamelistKeyIsRefused) {
    EXPECT_THROW(parse_text(silicon_input_with("", "  wibble = 3\n")), std::runtime_error);
}

TEST(QEParserStrictness, UnimplementedKeyIsRefusedWithItsReason) {
    try {
        parse_text(silicon_input_with("", "  noncolin = .true.\n"));
        FAIL() << "noncollinear magnetism must not be silently ignored";
    } catch (const std::runtime_error& error) {
        EXPECT_NE(std::string(error.what()).find("noncollinear"), std::string::npos);
    }
}

TEST(QEParserStrictness, FeatureSwitchedOffExplicitlyIsAccepted) {
    /**
     * QE inputs state defaults all the time. `noncolin = .false.` asks for
     * exactly what Q3 does, so refusing it would be a false refusal.
     */
    EXPECT_NO_THROW(parse_text(silicon_input_with("", "  noncolin = .false.\n  noinv = .F.\n")));
}

TEST(QEParserStrictness, UnconstrainedTotalMagnetizationSentinelIsAccepted) {
    // QE's "unspecified" sentinel is -10000; a real constraint is refused.
    EXPECT_NO_THROW(parse_text(silicon_input_with("", "  tot_magnetization = -10000\n")));
    EXPECT_THROW(parse_text(silicon_input_with("", "  tot_magnetization = 2\n")),
                 std::runtime_error);
}

TEST(QEParserStrictness, PhysicsNeutralKeysAreAccepted) {
    EXPECT_NO_THROW(parse_text(silicon_input_with("  verbosity = 'high'\n  tprnfor = .true.\n"
                                                  "  disk_io = 'none'\n",
                                                  "")));
}

TEST(QEParser, VerbosityMapsQuantumEspressoSpellingsOntoControlVerbosity) {
    /**
     * QE's six spellings collapse onto Q3's two along the line QE itself
     * draws at iverbosity > 0. Read rather than ignored, so a deck that asked
     * pw.x for the detailed output gets the detailed output here too.
     */
    EXPECT_EQ(parse_text(silicon_input_with("", "")).verbosity, "default");
    for (const char* loud : {"'high'", "'debug'"}) {
        EXPECT_EQ(parse_text(silicon_input_with(std::string("  verbosity = ") + loud + "\n", ""))
                      .verbosity,
                  "high");
    }
    for (const char* quiet : {"'low'", "'default'", "'medium'", "'minimal'"}) {
        EXPECT_EQ(parse_text(silicon_input_with(std::string("  verbosity = ") + quiet + "\n", ""))
                      .verbosity,
                  "default");
    }
}

TEST(QEParser, VerbosityRejectsAValueQuantumEspressoDoesNotHave) {
    EXPECT_THROW(parse_text(silicon_input_with("  verbosity = 'loud'\n", "")), std::runtime_error);
}

TEST(QEParserStrictness, MisplacedKeyNamesTheNamelistItBelongsIn) {
    try {
        parse_text(silicon_input_with("", "  conv_thr = 1e-8\n"));
        FAIL() << "conv_thr belongs to &electrons";
    } catch (const std::runtime_error& error) {
        EXPECT_NE(std::string(error.what()).find("&electrons"), std::string::npos);
    }
}

TEST(QEParserStrictness, UnknownNamelistIsRefused) {
    std::string input = silicon_input_with("", "");
    input += "&FCP\n  fcp_mu = 0.0\n/\n";
    EXPECT_THROW(parse_text(input), std::runtime_error);
}

TEST(QEParserStrictness, SeveralAssignmentsOnOneLineAreAllClassified) {
    /**
     * QE's own examples pack a namelist onto one line
     * (`ibrav= 2, celldm(1)=10.2, nat= 2`), so the key scan splits rather than
     * assuming one key per line — otherwise strictness would only ever see the
     * first key.
     */
    EXPECT_THROW(parse_text(silicon_input_with("", "  nbnd = 8, wibble = 3\n")),
                 std::runtime_error);
    InputData data = parse_text(silicon_input_with("", "  nbnd = 8, nspin = 1\n"));
    EXPECT_EQ(data.number_of_bands, 8);
}

TEST(QEParserStrictness, CommentedPoseIsNotScannedForKeys) {
    // `! a = 5` in prose must not read as an assignment to the key `A`.
    InputData data = parse_text(silicon_input_with("", "  nbnd = 8   ! a = 5 in the old cell\n"));
    EXPECT_EQ(data.number_of_bands, 8);
    EXPECT_NEAR(data.cell_dimension_1, 10.2, 1e-12);
}

TEST(QEParserStrictness, ArrayContinuationSyntaxIsRefusedRatherThanTruncated) {
    /**
     * `starting_magnetization = 0.5, 0.3` sets two species in Fortran. Reading
     * only the first would silently run a different magnetic configuration.
     */
    EXPECT_THROW(parse_text(silicon_input_with("",
                                               "  nspin = 2\n"
                                               "  starting_magnetization = 0.5, 0.3\n")),
                 std::runtime_error);
}

/**
 * Keys that used to be dropped and now land somewhere in InputData.
 */
TEST(QEParserHonouredKeys, RelaxationThresholdsAreRead) {
    InputData data = parse_text(silicon_input_with("  nstep = 42\n"
                                                   "  forc_conv_thr = 1.0d-4\n"
                                                   "  etot_conv_thr = 1.0d-5\n",
                                                   ""));
    EXPECT_EQ(data.relax_maximum_steps, 42);
    EXPECT_NEAR(data.relax_force_threshold, 1.0e-4, 1e-16);
    EXPECT_NEAR(data.relax_energy_threshold, 1.0e-5, 1e-16);
}

TEST(QEParserHonouredKeys, NosymDisablesSymmetry) {
    EXPECT_TRUE(parse_text(silicon_input_with("", "  nosym = .true.\n")).disable_symmetry);
    EXPECT_FALSE(parse_text(silicon_input_with("", "  nosym = .false.\n")).disable_symmetry);
}

TEST(QEParserHonouredKeys, AssumeIsolatedMapsOntoQ3Names) {
    EXPECT_EQ(parse_text(silicon_input_with("", "  assume_isolated = 'makov-payne'\n"))
                  .isolated_system_correction,
              "makov-payne");
    EXPECT_EQ(
        parse_text(silicon_input_with("", "  assume_isolated = 'mt'\n")).isolated_system_correction,
        "martyna-tuckerman");
    // '2D' and 'esm' are different physics, not aliases.
    EXPECT_THROW(parse_text(silicon_input_with("", "  assume_isolated = '2D'\n")),
                 std::runtime_error);
}

namespace {

/// The refusal message for `system_keys` added to &SYSTEM, or a failure if the deck is accepted.
std::string refusal_for_system_keys(const std::string& system_keys) {
    try {
        parse_text(silicon_input_with("", system_keys));
    } catch (const std::runtime_error& error) {
        return error.what();
    }
    ADD_FAILURE() << "&SYSTEM keys\n" << system_keys << "must be refused";
    return "";
}

} // namespace

TEST(QEParserHonouredKeys, DispersionSubOptionsAreCheckedRatherThanAssumed) {
    /**
     * Q3 implements D3 version 4, two-body only. It used to apply that whatever
     * the input asked for, which made `vdw_corr='dft-d3'` mean something
     * different in Q3 than in QE without saying so.
     */
    EXPECT_EQ(parse_text(silicon_input_with("",
                                            "  vdw_corr = 'dft-d3'\n"
                                            "  dftd3_version = 4\n"
                                            "  dftd3_threebody = .false.\n"))
                  .dispersion_correction,
              "d3bj");
    /**
     * Each deck states the other sub-option as Q3 implements it, so the
     * refusal has to come from the value given, not from a key left out.
     */
    std::string three_body = refusal_for_system_keys("  vdw_corr = 'dft-d3'\n"
                                                     "  dftd3_version = 4\n"
                                                     "  dftd3_threebody = .true.\n");
    EXPECT_NE(three_body.find("dftd3_threebody = .true. is not supported"), std::string::npos)
        << three_body;
    std::string zero_damping = refusal_for_system_keys("  vdw_corr = 'dft-d3'\n"
                                                       "  dftd3_version = 3\n"
                                                       "  dftd3_threebody = .false.\n");
    EXPECT_NE(zero_damping.find("dftd3_version = 3 is not supported"), std::string::npos)
        << zero_damping;
}

TEST(QEParserHonouredKeys, DispersionLeftAtPwDefaultsIsRefusedNamingWhatToWrite) {
    /**
     * pw.x defaults to dftd3_version = 3 (zero damping) and dftd3_threebody =
     * .true., so `vdw_corr = 'dft-d3'` on its own asks pw.x for zero-damped D3
     * with the three-body term. Q3 ran that deck as two-body D3(BJ) and
     * reported success. The refusal names each key left out, the default pw.x
     * takes for it, and the line that asks for what Q3 computes.
     */
    for (const char* spelling : {"'dft-d3'", "'grimme-d3'", "'DFT-D3'"}) {
        std::string message =
            refusal_for_system_keys(std::string("  vdw_corr = ") + spelling + "\n");
        for (const char* expected : {"dftd3_version = 3",
                                     "dftd3_threebody = .true.",
                                     "dftd3_version = 4",
                                     "dftd3_threebody = .false."})
            EXPECT_NE(message.find(expected), std::string::npos) << spelling << ": " << message;
    }
}

TEST(QEParserHonouredKeys, DispersionRefusalNamesOnlyTheSubOptionLeftOut) {
    std::string without_three_body = refusal_for_system_keys("  vdw_corr = 'dft-d3'\n"
                                                             "  dftd3_version = 4\n");
    EXPECT_NE(without_three_body.find("dftd3_threebody = .true."), std::string::npos)
        << without_three_body;
    EXPECT_NE(without_three_body.find("dftd3_threebody = .false."), std::string::npos)
        << without_three_body;
    EXPECT_EQ(without_three_body.find("dftd3_version"), std::string::npos) << without_three_body;

    std::string without_version = refusal_for_system_keys("  vdw_corr = 'dft-d3'\n"
                                                          "  dftd3_threebody = .false.\n");
    EXPECT_NE(without_version.find("dftd3_version = 3"), std::string::npos) << without_version;
    EXPECT_NE(without_version.find("dftd3_version = 4"), std::string::npos) << without_version;
    EXPECT_EQ(without_version.find("dftd3_threebody"), std::string::npos) << without_version;
}

TEST(QEParserHonouredKeys, DispersionSubOptionsMayComeInAnyOrder) {
    /**
     * QE reads a namelist's keys in any order, so the check waits for the
     * whole file rather than judging the deck at the vdw_corr line.
     */
    EXPECT_EQ(parse_text(silicon_input_with("",
                                            "  dftd3_threebody = .false.\n"
                                            "  dftd3_version = 4\n"
                                            "  vdw_corr = 'grimme-d3'\n"))
                  .dispersion_correction,
              "d3bj");
    EXPECT_EQ(
        parse_text(silicon_input_with(
                       "", "  vdw_corr = 'dft-d3', dftd3_version = 4, dftd3_threebody = .F.\n"))
            .dispersion_correction,
        "d3bj");
}

TEST(QEParserHonouredKeys, VdwCorrOtherThanD3IsRefusedRatherThanDropped) {
    /**
     * pw.x's other corrections fell through without a word, and the deck ran
     * with no dispersion correction at all.
     */
    for (const char* other : {"grimme-d2", "dft-d", "ts", "mbd", "xdm"}) {
        std::string message =
            refusal_for_system_keys(std::string("  vdw_corr = '") + other + "'\n");
        EXPECT_NE(message.find(std::string("vdw_corr = '") + other + "'"), std::string::npos)
            << message;
    }
    EXPECT_EQ(parse_text(silicon_input_with("", "  vdw_corr = 'none'\n")).dispersion_correction,
              "none");
}

TEST(QEParserHonouredKeys, RestartModeFromScratchIsAcceptedAndRestartIsNot) {
    EXPECT_NO_THROW(parse_text(silicon_input_with("  restart_mode = 'from_scratch'\n", "")));
    EXPECT_THROW(parse_text(silicon_input_with("  restart_mode = 'restart'\n", "")),
                 std::runtime_error);
}

TEST(QEParserHonouredKeys, IonAndCellNamelistsAreReadRatherThanFallingThrough) {
    /**
     * `&IONS` was not a recognized header, so its keys were scanned as the
     * previous namelist's and found nothing. `ion_dynamics='damp'` is a
     * different trajectory that stops somewhere else than BFGS.
     */
    std::string relax_input = silicon_input_with("", "") + "&IONS\n  ion_dynamics = 'bfgs'\n/\n";
    EXPECT_NO_THROW(parse_text(relax_input));

    std::string damped_input = silicon_input_with("", "") + "&IONS\n  ion_dynamics = 'damp'\n/\n";
    EXPECT_THROW(parse_text(damped_input), std::runtime_error);
}

/**
 * celldm(2..6) were dropped, so every Bravais index that needs an axial ratio
 * or an angle — hexagonal, tetragonal, orthorhombic, monoclinic, triclinic —
 * was built with c/a = 0, a collapsed cell.
 */
TEST(QEParserCellDimensions, AxialRatiosReachTheCrystal) {
    std::string input = R"(
&CONTROL
  calculation = 'scf'
/
&SYSTEM
  ibrav = 4
  celldm(1) = 4.65
  celldm(3) = 2.725
  nat = 1
  ntyp = 1
  ecutwfc = 30.0
/
&ELECTRONS
/
ATOMIC_SPECIES
  C 12.01 C.upf
ATOMIC_POSITIONS crystal
  C 0.0 0.0 0.0
K_POINTS automatic
  4 4 4 0 0 0
)";
    InputData data = parse_text(input);
    EXPECT_NEAR(data.cell_dimension_3, 2.725, 1e-12);

    Crystal crystal = build_crystal_from_input(data, data.species);
    // a3 = a(0, 0, c/a): the hexagonal axis must carry the ratio, not zero.
    EXPECT_NEAR(crystal.lattice_vectors()[2][2] * crystal.lattice_parameter(), 4.65 * 2.725, 1e-9);
}

TEST(QEParserCellDimensions, CrystallographicConstantsFollowQeAbc2Celldm) {
    /**
     * A/B/C in angstrom, converted with Quantum ESPRESSO's documented
     * crystallographic-constants-to-celldm mapping — including cosAB landing
     * in celldm(4) for monoclinic.
     */
    std::string input = R"(
&CONTROL
  calculation = 'scf'
/
&SYSTEM
  ibrav = 12
  A = 5.0
  B = 7.5
  C = 10.0
  cosAB = 0.5
  nat = 1
  ntyp = 1
  ecutwfc = 30.0
/
&ELECTRONS
/
ATOMIC_SPECIES
  Si 28.08 Si.upf
ATOMIC_POSITIONS crystal
  Si 0.0 0.0 0.0
K_POINTS automatic
  2 2 2 0 0 0
)";
    InputData data = parse_text(input);
    EXPECT_NEAR(data.cell_dimension_1, 5.0 * constants::ANGSTROM_TO_BOHR, 1e-9);
    EXPECT_NEAR(data.cell_dimension_2, 1.5, 1e-12);
    EXPECT_NEAR(data.cell_dimension_3, 2.0, 1e-12);
    EXPECT_NEAR(data.cell_dimension_4, 0.5, 1e-12);
}

TEST(QEParserCellDimensions, CelldmAndAAreRefusedTogether) {
    EXPECT_THROW(parse_text(silicon_input_with("", "  A = 5.43\n")), std::runtime_error);
}

// ── starting_ns_eigenvalue: occupation-matrix seeding directives ──

static const std::string feo_seeded_input = R"(
&CONTROL
  calculation = 'scf'
  pseudo_dir = './'
/
&SYSTEM
  ibrav = 0, celldm(1) = 8.19
  nat = 4, ntyp = 3
  ecutwfc = 30.0, ecutrho = 240.0
  nspin = 2
  starting_magnetization(1) = 0.5, starting_magnetization(2) = -0.5
  starting_ns_eigenvalue(5,2,1) = 1.d0, starting_ns_eigenvalue(5,1,2) = 0.85
/
&ELECTRONS
  conv_thr = 1e-8
/
ATOMIC_SPECIES
  Fe1 55.845 Fe.upf
  Fe2 55.845 Fe.upf
  O   16.0   O.upf
CELL_PARAMETERS alat
  0.50 0.50 1.00
  0.50 1.00 0.50
  1.00 0.50 0.50
ATOMIC_POSITIONS crystal
  Fe1 0.00 0.00 0.00
  Fe2 0.50 0.50 0.50
  O   0.25 0.25 0.25
  O   0.75 0.75 0.75
K_POINTS automatic
  2 2 2 0 0 0
HUBBARD (ortho-atomic)
  U Fe1-3d 4.6
  U Fe2-3d 4.6
)";

TEST(QEParserStartingNsEigenvalue, ParsesIndexedDirectivesToSymbolKeyedZeroBasedForm) {
    std::string path = write_temporary_input_file(feo_seeded_input);
    QEInputParser parser;
    InputData data = parser.parse(path);
    std::remove(path.c_str());

    ASSERT_EQ(data.hubbard_starting_eigenvalues.size(), 2u);

    // (5,2,1): eigenvalue 5 -> index 4, spin 2 -> index 1, ityp 1 -> "Fe1".
    EXPECT_EQ(data.hubbard_starting_eigenvalues[0].species_symbol, "Fe1");
    EXPECT_EQ(data.hubbard_starting_eigenvalues[0].eigenvalue_index, 4);
    EXPECT_EQ(data.hubbard_starting_eigenvalues[0].spin_index, 1);
    EXPECT_DOUBLE_EQ(data.hubbard_starting_eigenvalues[0].target_occupation,
                     1.0); // 1.d0 read as Fortran

    EXPECT_EQ(data.hubbard_starting_eigenvalues[1].species_symbol, "Fe2");
    EXPECT_EQ(data.hubbard_starting_eigenvalues[1].eigenvalue_index, 4);
    EXPECT_EQ(data.hubbard_starting_eigenvalues[1].spin_index, 0);
    EXPECT_DOUBLE_EQ(data.hubbard_starting_eigenvalues[1].target_occupation, 0.85);
}

TEST(QEParserStartingNsEigenvalue, CommaInsideIndexParenthesesDoesNotSplitTheEntry) {
    /**
     * The namelist splitter treats top-level commas as entry separators;
     * the (m,ispin,ityp) commas must survive. This deck puts the directive
     * mid-line between two other assignments to force the issue.
     */
    std::string line_packed = feo_seeded_input;
    size_t position = line_packed.find("  starting_ns_eigenvalue(5,2,1) = 1.d0, "
                                       "starting_ns_eigenvalue(5,1,2) = 0.85");
    ASSERT_NE(position, std::string::npos);
    line_packed.replace(position,
                        std::string("  starting_ns_eigenvalue(5,2,1) = 1.d0, "
                                    "starting_ns_eigenvalue(5,1,2) = 0.85")
                            .size(),
                        "  nbnd = 20, starting_ns_eigenvalue(3,1,1) = 0.5, ecutfock = 0");
    // ecutfock would be refused; strip it to keep the deck to the point.
    size_t fock = line_packed.find(", ecutfock = 0");
    line_packed.erase(fock, std::string(", ecutfock = 0").size());

    std::string path = write_temporary_input_file(line_packed);
    QEInputParser parser;
    InputData data = parser.parse(path);
    std::remove(path.c_str());

    ASSERT_EQ(data.hubbard_starting_eigenvalues.size(), 1u);
    EXPECT_EQ(data.hubbard_starting_eigenvalues[0].species_symbol, "Fe1");
    EXPECT_EQ(data.hubbard_starting_eigenvalues[0].eigenvalue_index, 2);
    EXPECT_EQ(data.hubbard_starting_eigenvalues[0].spin_index, 0);
}

TEST(QEParserStartingNsEigenvalue, SpeciesIndexBeyondNtypThrows) {
    std::string bad = feo_seeded_input;
    size_t position = bad.find("starting_ns_eigenvalue(5,2,1)");
    ASSERT_NE(position, std::string::npos);
    bad.replace(position,
                std::string("starting_ns_eigenvalue(5,2,1)").size(),
                "starting_ns_eigenvalue(5,2,7)");
    std::string path = write_temporary_input_file(bad);
    QEInputParser parser;
    EXPECT_THROW(parser.parse(path), std::runtime_error);
    std::remove(path.c_str());
}

TEST(QEParserStartingNsEigenvalue, ValueOutsideZeroOneThrows) {
    std::string bad = feo_seeded_input;
    size_t position = bad.find("= 0.85");
    ASSERT_NE(position, std::string::npos);
    bad.replace(position, 6, "= 1.75");
    std::string path = write_temporary_input_file(bad);
    QEInputParser parser;
    EXPECT_THROW(parser.parse(path), std::runtime_error);
    std::remove(path.c_str());
}

TEST(QEParserStartingNsEigenvalue, MissingIndicesThrow) {
    std::string bad = feo_seeded_input;
    size_t position = bad.find("starting_ns_eigenvalue(5,2,1)");
    bad.replace(position,
                std::string("starting_ns_eigenvalue(5,2,1)").size(),
                "starting_ns_eigenvalue(5,2)");
    std::string path = write_temporary_input_file(bad);
    QEInputParser parser;
    EXPECT_THROW(parser.parse(path), std::runtime_error);
    std::remove(path.c_str());
}

// ── Fortran double-precision literals in the cards ──────────────────

/**
 * The same aluminium slab, written the way Fortran writes reals when the
 * digits matter: `d` exponents throughout the cards, and a `D` among them
 * because the marker is case-insensitive in the language it comes from.
 */
static const std::string aluminum_slab_input_fortran_literals = R"(
&CONTROL
  calculation = 'self_consistent_field'
  pseudo_dir = './'
/
&SYSTEM
  ibrav = 0
  nat = 1
  ntyp = 1
  ecutwfc = 3.0d1
/
&ELECTRONS
  conv_thr = 1.0d-8
/
ATOMIC_SPECIES
  Al 26.98d0 Al.upf
ATOMIC_POSITIONS crystal
  Al 0.0d0 0.0d0 0.0d0
CELL_PARAMETERS bohr
  1.0D1  0.0d0  0.0d0
  0.0d0  1.2d1  0.0d0
  0.0d0  0.0d0  3.0d1
K_POINTS automatic
  4 4 1 0 0 0
)";

TEST(QEParserFortranLiterals, CardsReadDExponentsExactlyAsPlainDecimals) {
    /**
     * The namelist keys have gone through fortran_to_double() all along; the
     * cards read straight into a `Real` with `>>`, which stops at the `d`. The
     * first field on a line silently became its mantissa and every field after
     * it failed to convert, which C++11 then sets to zero -- so this cell
     * parsed as (10, 0, 0), (0, 0, 0), (0, 0, 0), a degenerate cell that the
     * run then carried to completion reporting `cell volume = 0.0000` rather
     * than stopping.
     *
     * Asserted against the plain-decimal input rather than against literals, so
     * the two spellings cannot drift apart: they describe the same slab.
     */
    std::string plain_path = write_temporary_input_file(aluminum_slab_input);
    std::string fortran_path = write_temporary_input_file(aluminum_slab_input_fortran_literals);
    QEInputParser parser;
    InputData plain = parser.parse(plain_path);
    InputData fortran = parser.parse(fortran_path);
    std::remove(plain_path.c_str());
    std::remove(fortran_path.c_str());

    ASSERT_TRUE(fortran.has_explicit_lattice_vectors);
    for (int row = 0; row < 3; row++) {
        for (int column = 0; column < 3; column++) {
            EXPECT_NEAR(
                fortran.lattice_vectors[row][column], plain.lattice_vectors[row][column], 1e-12)
                << "lattice_vectors[" << row << "][" << column << "]";
        }
    }
    EXPECT_NEAR(fortran.cell_dimension_1, plain.cell_dimension_1, 1e-12);
    EXPECT_NEAR(fortran.wavefunction_energy_cutoff, plain.wavefunction_energy_cutoff, 1e-12);
    EXPECT_NEAR(fortran.convergence_threshold, plain.convergence_threshold, 1e-20);

    /**
     * ATOMIC_SPECIES failed more quietly than the cell did: `Al 26.98d0 Al.upf`
     * left `d0` as the next token, so the mass lost its exponent and `d0` was
     * taken as the pseudopotential filename.
     */
    ASSERT_EQ(fortran.species.size(), 1u);
    EXPECT_NEAR(fortran.species[0].mass, 26.98, 1e-12);
    EXPECT_EQ(fortran.species[0].pseudopotential_file, "Al.upf");
}

TEST(QEParserFortranLiterals, ANonNumericCardFieldIsRefusedRatherThanZeroed) {
    /**
     * The zeroing that made the degenerate cell above is what this refuses. A
     * cell that cannot be read is not a cell of volume zero, and a run that
     * reports one has already lost the thing the user asked about.
     */
    std::string input = aluminum_slab_input;
    size_t position = input.find("  10.0  0.0  0.0");
    ASSERT_NE(position, std::string::npos);
    input.replace(position, std::string("  10.0  0.0  0.0").size(), "  10.0  nonsense  0.0");

    std::string path = write_temporary_input_file(input);
    QEInputParser parser;
    try {
        parser.parse(path);
        std::remove(path.c_str());
        FAIL() << "expected an unreadable CELL_PARAMETERS field to be refused";
    } catch (const std::runtime_error& error) {
        std::remove(path.c_str());
        std::string message = error.what();
        EXPECT_NE(message.find("CELL_PARAMETERS"), std::string::npos) << message;
        EXPECT_NE(message.find("nonsense"), std::string::npos) << message;
    }
}

// ── ATOMIC_POSITIONS units ──────────────────────────────────────────

/**
 * The card's units option was discarded until 2026-09-14 and every card read as
 * crystal coordinates, which pw.x does only for `crystal`. The fixture is the deck
 * from camsai/q3#18: BPO4 (Materials Project mp-3589) in a body-centred tetragonal
 * cell given as three explicit vectors. That cell is neither orthogonal nor
 * symmetric, so a transposed or mis-scaled conversion moves atoms visibly; read as
 * crystal, its angstrom rows put two O-O pairs 1.08 Å apart.
 */
namespace {

struct PositionRow {
    std::string symbol;
    Vector3D position;
};

const Real boron_phosphate_cell_angstrom[3][3] = {
    {4.4055882500, -0.0000000040, 0.0000000010},
    {0.0000002830, 4.4055881420, 0.0000000030},
    {-2.2027943440, -2.2027942410, 3.3519305350},
};

const std::vector<PositionRow> boron_phosphate_rows_angstrom = {
    {"B", {0.0000001025, 2.2027939847, 1.6759652700}},
    {"P", {0.0000000000, 0.0000000000, 0.0000000000}},
    {"O", {-1.0624485731, 1.6057891884, 2.5013468678}},
    {"O", {0.5970048232, 1.1403365814, 0.8505836715}},
    {"O", {1.0624573122, -1.6057849961, 2.5013468646}},
    {"O", {3.8085792206, 3.2652426136, 0.8505836745}},
};

/**
 * The crystal coordinates pw.x 7.6 prints (`verbosity = 'high'`) for this structure.
 * It prints the same seven-decimal values whichever of angstrom, bohr, alat, crystal
 * or no option the deck is written in.
 */
const std::vector<Vector3D> boron_phosphate_crystal_coordinates_from_pwx = {
    {0.2500000, 0.7500000, 0.5000000},
    {0.0000000, 0.0000000, 0.0000000},
    {0.1319610, 0.7376095, 0.7462407},
    {0.2623905, 0.3857183, 0.2537593},
    {0.6142817, 0.0086321, 0.7462407},
    {0.9913679, 0.8680390, 0.2537593},
};

/// alat for a CELL_PARAMETERS card in angstrom is |a1|, here in angstrom.
Real boron_phosphate_first_vector_length_angstrom() {
    const auto& first = boron_phosphate_cell_angstrom[0];
    return std::sqrt(first[0] * first[0] + first[1] * first[1] + first[2] * first[2]);
}

/**
 * The deck with its rows scaled from angstrom by `scale_from_angstrom` and headed by
 * `positions_header`. Numbers are written with 17 significant digits, so the text
 * carries each double exactly.
 */
std::string boron_phosphate_input(const std::string& positions_header,
                                  Real scale_from_angstrom,
                                  bool positions_before_cell = false) {
    std::ostringstream cell;
    cell << std::setprecision(17) << "CELL_PARAMETERS angstrom\n";
    for (const auto& row : boron_phosphate_cell_angstrom)
        cell << "  " << row[0] << " " << row[1] << " " << row[2] << "\n";

    std::ostringstream positions;
    positions << std::setprecision(17) << positions_header << "\n";
    for (const auto& row : boron_phosphate_rows_angstrom) {
        Vector3D scaled = row.position * scale_from_angstrom;
        positions << "  " << row.symbol << " " << scaled.x << " " << scaled.y << " " << scaled.z
                  << "\n";
    }

    std::string cards =
        positions_before_cell ? positions.str() + cell.str() : cell.str() + positions.str();
    return R"(
&CONTROL
  calculation = 'scf'
  pseudo_dir = './'
/
&SYSTEM
  ibrav = 0
  nat = 6
  ntyp = 3
  ecutwfc = 60.0
  nbnd = 17
/
&ELECTRONS
  conv_thr = 1.0d-8
  mixing_beta = 0.7
  mixing_mode = 'plain'
/
ATOMIC_SPECIES
  B 10.81000 B.upf
  O 15.99900 O.upf
  P 30.97376 P.upf
)" + cards +
           "K_POINTS automatic\n  9 9 9 0 0 0\n";
}

} // namespace

TEST(QEParserAtomicPositionsUnits, EverySpellingPwxAcceptsReadsTheWayPwxReadsIt) {
    /**
     * Seven decimals is all pw.x prints, so that is the tolerance. A card with no
     * option is read as alat, as pw.x reads it.
     */
    Real one_over_alat_angstrom = 1.0 / boron_phosphate_first_vector_length_angstrom();
    const std::vector<std::pair<std::string, Real>> spellings = {
        {"ATOMIC_POSITIONS angstrom", 1.0},
        {"ATOMIC_POSITIONS bohr", constants::ANGSTROM_TO_BOHR},
        {"ATOMIC_POSITIONS alat", one_over_alat_angstrom},
        {"ATOMIC_POSITIONS", one_over_alat_angstrom},
    };
    for (const auto& [header, scale] : spellings) {
        InputData data = parse_text(boron_phosphate_input(header, scale));
        ASSERT_EQ(data.atoms.size(), boron_phosphate_crystal_coordinates_from_pwx.size()) << header;
        for (size_t atom = 0; atom < data.atoms.size(); atom++) {
            for (int axis = 0; axis < 3; axis++) {
                EXPECT_NEAR(data.atoms[atom].position[axis],
                            boron_phosphate_crystal_coordinates_from_pwx[atom][axis],
                            1e-7)
                    << "'" << header << "', atom " << atom << ", axis " << axis;
            }
        }
    }
}

TEST(QEParserAtomicPositionsUnits, CartesianRowsLandWhereTheCardPutsThem) {
    /**
     * Checked through the Crystal the parsed input builds rather than through the
     * parser's own arithmetic: each atom's crystal coordinates, taken through the
     * Crystal's lattice, must give back the row the card wrote. The positions card
     * comes first here -- QE accepts cards in any order -- so the conversion cannot
     * depend on the cell having been read already.
     */
    Real one_over_alat_angstrom = 1.0 / boron_phosphate_first_vector_length_angstrom();
    const std::vector<std::pair<std::string, Real>> spellings = {
        {"ATOMIC_POSITIONS (angstrom)", 1.0},
        {"ATOMIC_POSITIONS {bohr}", constants::ANGSTROM_TO_BOHR},
        {"atomic_positions Alat", one_over_alat_angstrom},
    };
    for (const auto& [header, scale] : spellings) {
        InputData data = parse_text(boron_phosphate_input(header, scale, true));
        Crystal crystal = build_crystal_from_input(data, data.species);
        ASSERT_EQ(crystal.number_of_atoms(), 6) << header;
        for (int atom = 0; atom < crystal.number_of_atoms(); atom++) {
            Vector3D cartesian_bohr =
                crystal.convert_from_crystal_to_cartesian(atom) * crystal.lattice_parameter();
            for (int axis = 0; axis < 3; axis++) {
                EXPECT_NEAR(cartesian_bohr[axis],
                            boron_phosphate_rows_angstrom[atom].position[axis] *
                                constants::ANGSTROM_TO_BOHR,
                            1e-9)
                    << "'" << header << "', atom " << atom << ", axis " << axis;
            }
        }
    }
}

TEST(QEParserAtomicPositionsUnits, IbravCellsConvertThroughTheIbravVectors) {
    /**
     * No CELL_PARAMETERS card: the cell is ibrav = 4, a1 = a(1,0,0),
     * a2 = a(-1/2,√3/2,0), a3 = a(0,0,c/a), and alat is celldm(1). Graphite-like
     * sites written as Cartesian rows come back as their crystal coordinates, as
     * they do under pw.x 7.6 for both of these spellings.
     */
    const Real alat = 4.65;
    const Real c_over_a = 2.725;
    const std::vector<Vector3D> crystal_coordinates = {
        {0.0, 0.0, 0.25},
        {0.0, 0.0, 0.75},
        {1.0 / 3.0, 2.0 / 3.0, 0.25},
        {2.0 / 3.0, 1.0 / 3.0, 0.75},
    };
    const std::vector<std::pair<std::string, Real>> spellings = {{"bohr", alat}, {"alat", 1.0}};
    for (const auto& [option, scale] : spellings) {
        std::ostringstream input;
        input << std::setprecision(17) << "&CONTROL\n  calculation = 'scf'\n/\n&SYSTEM\n"
              << "  ibrav = 4\n  celldm(1) = " << alat << "\n  celldm(3) = " << c_over_a
              << "\n  nat = 4\n  ntyp = 1\n  ecutwfc = 30.0\n/\n&ELECTRONS\n/\n"
              << "ATOMIC_SPECIES\n  C 12.01 C.upf\nATOMIC_POSITIONS " << option << "\n";
        for (const Vector3D& fractional : crystal_coordinates) {
            Vector3D cartesian_alat{fractional.x - 0.5 * fractional.y,
                                    std::sqrt(3.0) / 2.0 * fractional.y,
                                    c_over_a * fractional.z};
            Vector3D row = cartesian_alat * scale;
            input << "  C " << row.x << " " << row.y << " " << row.z << "\n";
        }
        input << "K_POINTS automatic\n  4 4 2 0 0 0\n";

        InputData data = parse_text(input.str());
        ASSERT_EQ(data.atoms.size(), crystal_coordinates.size()) << option;
        for (size_t atom = 0; atom < data.atoms.size(); atom++) {
            for (int axis = 0; axis < 3; axis++) {
                EXPECT_NEAR(data.atoms[atom].position[axis], crystal_coordinates[atom][axis], 1e-12)
                    << option << ", atom " << atom << ", axis " << axis;
            }
        }
    }
}

TEST(QEParserAtomicPositionsUnits, OptionsQ3CannotHonourAreRefusedByName) {
    /**
     * `crystal_sg` lists only the symmetry-inequivalent atoms, and generating the
     * rest needs a space-group expansion Q3 does not do; reading its rows as the
     * whole cell would drop atoms without a word.
     */
    for (const char* option : {"crystal_sg", "picometre"}) {
        try {
            parse_text(boron_phosphate_input(std::string("ATOMIC_POSITIONS ") + option, 1.0));
            FAIL() << "ATOMIC_POSITIONS " << option << " must be refused";
        } catch (const std::runtime_error& error) {
            EXPECT_NE(std::string(error.what()).find(option), std::string::npos) << error.what();
        }
    }
}

TEST(QEParserAtomicPositionsUnits, CartesianRowsWithoutACellAreRefused) {
    // ibrav = 0 and no CELL_PARAMETERS card: there is no cell to convert the rows in.
    std::string input = boron_phosphate_input("ATOMIC_POSITIONS angstrom", 1.0);
    size_t cell_begin = input.find("CELL_PARAMETERS");
    size_t cell_end = input.find("ATOMIC_POSITIONS");
    ASSERT_LT(cell_begin, cell_end);
    input.erase(cell_begin, cell_end - cell_begin);

    try {
        parse_text(input);
        FAIL() << "Cartesian positions with no cell must be refused";
    } catch (const std::runtime_error& error) {
        EXPECT_NE(std::string(error.what()).find("CELL_PARAMETERS"), std::string::npos)
            << error.what();
    }
}

// ── ATOMIC_POSITIONS if_pos flags ───────────────────────────────────

/**
 * pw.x rows may end in three `if_pos` flags, and a 0 holds that coordinate fixed while the
 * ions move. They do not change an SCF, and q3.workflows' relaxations read them from the
 * deck and keep those coordinates fixed, so the engine accepts them; anything else after the
 * coordinates is refused.
 */
namespace {

/// Silicon with `second_row` as its second ATOMIC_POSITIONS row, under `calculation`.
std::string silicon_input_with_second_row(const std::string& second_row,
                                          const std::string& calculation = "relax") {
    return "&CONTROL\n  calculation = '" + calculation +
           "'\n  pseudo_dir = './'\n/\n&SYSTEM\n  ibrav = 2\n  celldm(1) = 10.2\n  nat = 2\n"
           "  ntyp = 1\n  ecutwfc = 30.0\n/\n&ELECTRONS\n/\n&IONS\n/\nATOMIC_SPECIES\n"
           "  Si 28.08 Si.upf\nATOMIC_POSITIONS crystal\n  Si 0.0 0.0 0.0\n  " +
           second_row + "\nK_POINTS automatic\n  4 4 4 1 1 1\n";
}

/// The refusal message for `second_row`, or a failure if the deck is accepted.
std::string refusal_for_second_row(const std::string& second_row,
                                   const std::string& calculation = "relax") {
    try {
        parse_text(silicon_input_with_second_row(second_row, calculation));
    } catch (const std::runtime_error& error) {
        return error.what();
    }
    ADD_FAILURE() << "'" << second_row << "' under calculation='" << calculation
                  << "' must be refused";
    return "";
}

} // namespace

TEST(QEParserAtomicPositionFlags, AFixedCoordinateIsAcceptedAndChangesNothing) {
    InputData plain = parse_text(silicon_input_with_second_row("Si 0.25 0.25 0.25"));
    for (const char* row :
         {"Si 0.25 0.25 0.25 0 0 1", "Si 0.25 0.25 0.25 1 1 0", "Si 0.25 0.25 0.25 0 0 0"}) {
        InputData flagged = parse_text(silicon_input_with_second_row(row));
        ASSERT_EQ(flagged.atoms.size(), 2u) << row;
        for (int axis = 0; axis < 3; axis++)
            EXPECT_EQ(flagged.atoms[1].position[axis], plain.atoms[1].position[axis]) << row;
    }
}

TEST(QEParserAtomicPositionFlags, ThreeOnesAreAcceptedAndChangeNothing) {
    InputData flagged = parse_text(silicon_input_with_second_row("Si 0.25 0.25 0.25 1 1 1"));
    InputData plain = parse_text(silicon_input_with_second_row("Si 0.25 0.25 0.25"));
    ASSERT_EQ(flagged.atoms.size(), 2u);
    ASSERT_EQ(plain.atoms.size(), 2u);
    for (size_t atom = 0; atom < flagged.atoms.size(); atom++) {
        EXPECT_EQ(flagged.atoms[atom].species_index, plain.atoms[atom].species_index);
        for (int axis = 0; axis < 3; axis++)
            EXPECT_EQ(flagged.atoms[atom].position[axis], plain.atoms[atom].position[axis])
                << "atom " << atom << ", axis " << axis;
    }
}

TEST(QEParserAtomicPositionFlags, TrailingTextThatIsNotThreeFlagsIsRefused) {
    for (const char* row : {"Si 0.25 0.25 0.25 1.0 1 1",
                            "Si 0.25 0.25 0.25 1 1",
                            "Si 0.25 0.25 0.25 1 1 1 1",
                            "Si 0.25 0.25 0.25 2 1 1",
                            "Si 0.25 0.25 0.25 fixed"}) {
        std::string message = refusal_for_second_row(row);
        EXPECT_NE(message.find(row), std::string::npos) << message;
    }
}

TEST(QEParserAtomicPositionFlags, PlainRowsAndTrailingCommentsReadAsBefore) {
    // A `!` comment is stripped before the row is read, so it is not trailing text.
    for (const char* row : {"Si 0.25 0.25 0.25", "Si 0.25 0.25 0.25   ! the second site"}) {
        InputData data = parse_text(silicon_input_with_second_row(row));
        ASSERT_EQ(data.atoms.size(), 2u) << row;
        EXPECT_EQ(data.atoms[1].species_index, 0) << row;
        EXPECT_NEAR(data.atoms[1].position.x, 0.25, 1e-12) << row;
        EXPECT_NEAR(data.atoms[1].position.y, 0.25, 1e-12) << row;
        EXPECT_NEAR(data.atoms[1].position.z, 0.25, 1e-12) << row;
    }
}

// ── CELL_PARAMETERS units option ────────────────────────────────────

/**
 * pw.x reads a CELL_PARAMETERS card that names no units as alat when celldm(1) or A is
 * given and as bohr when neither is -- a deprecated spelling, but the one ase-espresso
 * writes for every deck: celldm(1) set to one angstrom in bohr, the rows in angstrom, no
 * option. The reader used to take bohr regardless, so such a deck ran a cell 1.89 times
 * too small on every axis. The fixture is the 3.5 x 3.5 x 10 Å box of one of those decks;
 * the expected vectors are the ones pw.x 7.6 recorded for it.
 */
namespace {

const Real one_angstrom_in_bohr = 1.8897261245650618;
const std::string celldm_one_angstrom = "  celldm(1) = 1.8897261245650618\n";

/// A one-atom slab deck with `lattice_parameter_line` in &SYSTEM and `header` over its cell.
std::string slab_cell_input(const std::string& header, const std::string& lattice_parameter_line) {
    return "&CONTROL\n  calculation = 'scf'\n  pseudo_dir = './'\n/\n&SYSTEM\n  ibrav = 0\n" +
           lattice_parameter_line +
           "  nat = 1\n  ntyp = 1\n  ecutwfc = 20.0\n/\n&ELECTRONS\n/\n"
           "ATOMIC_SPECIES\n  Ni 58.6934 Ni.upf\nATOMIC_POSITIONS crystal\n  Ni 0.125 0.125 0.3\n" +
           header + "\n  3.5 0 0\n  0 3.5 0\n  0 0 10\nK_POINTS automatic\n  2 2 1 0 0 0\n";
}

/// Row `row` of the parsed cell in bohr: the stored rows are in units of alat.
Vector3D parsed_lattice_vector_bohr(const InputData& data, int row) {
    return data.lattice_vectors[row] * data.cell_dimension_1;
}

} // namespace

TEST(QEParserCellParametersUnits, NoOptionMeansAlatWhenALatticeParameterIsGiven) {
    /**
     * celldm(1) and A are the two spellings of the lattice parameter, and pw.x 7.6
     * builds the same cell from either: a1 = 6.614041435977716 bohr and
     * a3 = 18.89726124565062 bohr, as its data file for the fixture deck records.
     * Read as bohr, a1 would be 3.5 bohr. `A` goes through Q3's own angstrom-to-bohr
     * constant, which differs from the deck's celldm(1) in the eleventh digit.
     */
    const std::vector<std::pair<std::string, Real>> lattice_parameters = {
        {celldm_one_angstrom, one_angstrom_in_bohr},
        {"  A = 1.0\n", constants::ANGSTROM_TO_BOHR},
    };
    for (const auto& [line, alat_bohr] : lattice_parameters) {
        InputData data = parse_text(slab_cell_input("CELL_PARAMETERS", line));
        EXPECT_TRUE(data.has_explicit_lattice_vectors) << line;
        EXPECT_NEAR(data.cell_dimension_1, alat_bohr, 1e-12) << line;
        EXPECT_NEAR(parsed_lattice_vector_bohr(data, 0).x, 6.614041435977716, 1e-9) << line;
        EXPECT_NEAR(parsed_lattice_vector_bohr(data, 1).y, 6.614041435977716, 1e-9) << line;
        EXPECT_NEAR(parsed_lattice_vector_bohr(data, 2).z, 18.89726124565062, 1e-9) << line;

        // Through the Crystal the input builds: the atom sits at (0.4375, 0.4375, 3) Å.
        Crystal crystal = build_crystal_from_input(data, data.species);
        ASSERT_EQ(crystal.number_of_atoms(), 1) << line;
        Vector3D cartesian_bohr =
            crystal.convert_from_crystal_to_cartesian(0) * crystal.lattice_parameter();
        EXPECT_NEAR(cartesian_bohr.x, 0.4375 * one_angstrom_in_bohr, 1e-9) << line;
        EXPECT_NEAR(cartesian_bohr.y, 0.4375 * one_angstrom_in_bohr, 1e-9) << line;
        EXPECT_NEAR(cartesian_bohr.z, 3.0 * one_angstrom_in_bohr, 1e-9) << line;
    }
}

TEST(QEParserCellParametersUnits, NoOptionMeansBohrWhenNoLatticeParameterIsGiven) {
    // pw.x 7.6 builds a1 = 3.5 bohr and a3 = 10 bohr from this deck, with alat = |a1|.
    InputData data = parse_text(slab_cell_input("CELL_PARAMETERS", ""));
    EXPECT_TRUE(data.has_explicit_lattice_vectors);
    EXPECT_NEAR(data.cell_dimension_1, 3.5, 1e-12);
    EXPECT_NEAR(parsed_lattice_vector_bohr(data, 0).x, 3.5, 1e-12);
    EXPECT_NEAR(parsed_lattice_vector_bohr(data, 2).z, 10.0, 1e-12);
}

TEST(QEParserCellParametersUnits, WrittenOptionsReadAsWrittenHoweverTheyAreSpelt) {
    /**
     * An option in braces, in parentheses or bare, in any case, reads as that option
     * whether or not a lattice parameter is given. With celldm(1) set, bohr rows stay
     * absolute and alat stays celldm(1).
     */
    struct Spelling {
        std::string header;
        std::string lattice_parameter_line;
        Real a1_bohr;
        Real alat_bohr;
    };
    const std::vector<Spelling> spellings = {
        {"CELL_PARAMETERS {alat}", celldm_one_angstrom, 6.614041435977716, one_angstrom_in_bohr},
        {"CELL_PARAMETERS (angstrom)", "", 6.614041435977716, 6.614041435977716},
        {"cell_parameters Bohr", "", 3.5, 3.5},
        {"CELL_PARAMETERS bohr", celldm_one_angstrom, 3.5, one_angstrom_in_bohr},
    };
    for (const auto& spelling : spellings) {
        InputData data =
            parse_text(slab_cell_input(spelling.header, spelling.lattice_parameter_line));
        EXPECT_NEAR(data.cell_dimension_1, spelling.alat_bohr, 1e-9) << spelling.header;
        EXPECT_NEAR(parsed_lattice_vector_bohr(data, 0).x, spelling.a1_bohr, 1e-9)
            << spelling.header;
    }
}

TEST(QEParserCellParametersUnits, UnrecognizedOptionsAreRefusedByName) {
    // An option Q3 cannot read must not fall back to bohr without a word.
    try {
        parse_text(slab_cell_input("CELL_PARAMETERS picometre", ""));
        FAIL() << "CELL_PARAMETERS picometre must be refused";
    } catch (const std::runtime_error& error) {
        EXPECT_NE(std::string(error.what()).find("picometre"), std::string::npos) << error.what();
    }
}

/**
 * Fortran namelist input does not care about line breaks: assignments may share the
 * `&name` header's line, and the closing `/` may follow the last assignment. The header's
 * remainder used to be thrown away, so a deck written that way ran on Q3's defaults for
 * every key it set there, and a trailing `/` became part of the value before it.
 */
namespace {

/// Silicon with every namelist written on its header's line, closing `/` included.
const std::string silicon_one_line_namelists_input = R"(
&CONTROL calculation = 'relax', pseudo_dir = './pseudo/' /
&SYSTEM ibrav=2, celldm(1)=10.2, nat=2, ntyp=1, ecutwfc=35.0 /
&ELECTRONS conv_thr = 1.0d-10, mixing_beta = 0.3 /
&IONS /
ATOMIC_SPECIES
  Si 28.08 Si.upf
ATOMIC_POSITIONS crystal
  Si 0.0 0.0 0.0
  Si 0.25 0.25 0.25
K_POINTS automatic
  4 4 4 1 1 1
)";

} // namespace

TEST(QEParserNamelistLayout, AssignmentsOnTheHeaderLineAreRead) {
    InputData data = parse_text(silicon_one_line_namelists_input);
    EXPECT_EQ(data.calculation, "relaxation");
    EXPECT_EQ(data.pseudopotential_directory, "./pseudo/");
    EXPECT_EQ(data.bravais_lattice_index, 2);
    EXPECT_NEAR(data.cell_dimension_1, 10.2, 1e-12);
    EXPECT_EQ(data.number_of_atoms, 2);
    EXPECT_EQ(data.number_of_species, 1);
    EXPECT_DOUBLE_EQ(data.wavefunction_energy_cutoff, 35.0);
    EXPECT_NEAR(data.convergence_threshold, 1e-10, 1e-22);
    EXPECT_DOUBLE_EQ(data.mixing_beta, 0.3);
}

TEST(QEParserNamelistLayout, HeaderLineAssignmentsAreClassified) {
    // The header's remainder goes through the same strictness as any other namelist line.
    EXPECT_THROW(parse_text("&SYSTEM wibble = 3 /\n" + silicon_input_with("", "")),
                 std::runtime_error);
}

TEST(QEParserNamelistLayout, HeaderWithoutASpaceBeforeItsFirstKeyIsStillItsName) {
    // `&SYSTEM,ibrav=2` and `&SYSTEM` alone both name the namelist `system`.
    std::string input = silicon_input_with("", "");
    std::string header = "&SYSTEM\n  ibrav = 2\n";
    input.replace(input.find(header), header.size(), "&SYSTEM,ibrav = 2\n");
    EXPECT_EQ(parse_text(input).bravais_lattice_index, 2);
}

TEST(QEParserNamelistLayout, TrailingSlashClosesTheNamelistAndLeavesTheValueAlone) {
    // `30.0 /` used to reach the numeric reader, and `'relax' /` the calculation verb.
    InputData data = parse_text(R"(
&CONTROL
  calculation = 'relax' /
&SYSTEM
  ibrav = 2, celldm(1) = 10.2, nat = 2, ntyp = 1
  ecutwfc = 30.0 /
&ELECTRONS
  conv_thr = 1e-8 /
ATOMIC_SPECIES
  Si 28.08 Si.upf
ATOMIC_POSITIONS crystal
  Si 0.0 0.0 0.0
  Si 0.25 0.25 0.25
K_POINTS automatic
  4 4 4 1 1 1
)");
    EXPECT_EQ(data.calculation, "relaxation");
    EXPECT_DOUBLE_EQ(data.wavefunction_energy_cutoff, 30.0);
    EXPECT_NEAR(data.convergence_threshold, 1e-8, 1e-20);
    EXPECT_EQ(data.number_of_atoms, 2);
}

TEST(QEParserNamelistLayout, TextAfterTheClosingSlashIsRefused) {
    // Fortran skips the rest of the line after the slash; Q3 refuses rather than drop it.
    std::string input = silicon_input_with("", "  ecutwfc = 30.0 / nbnd = 8\n");
    try {
        parse_text(input);
        FAIL() << "text after the closing '/' must be refused";
    } catch (const std::runtime_error& error) {
        EXPECT_NE(std::string(error.what()).find("nbnd = 8"), std::string::npos) << error.what();
    }
}

TEST(QEParserNamelistLayout, AmpersandEndStillClosesTheNamelist) {
    std::string input = silicon_input_with("", "");
    std::string terminator = "  conv_thr = 1e-6\n\n/\n";
    input.replace(input.find(terminator), terminator.size(), "  conv_thr = 1e-8\n&END\n");
    EXPECT_NEAR(parse_text(input).convergence_threshold, 1e-8, 1e-20);
}

/**
 * pw.x's mixing_mode = 'plain', written or left as pw.x's default, is Broyden with no
 * preconditioning. The reader used to leave the preconditioner at Q3's own "default",
 * which turns Kerker on whenever smearing is on, so a metallic QE deck ran a different
 * mixer from pw.x's: on a Ni(111) slab, 85 SCF iterations against pw.x's 24.
 */
TEST(QEParserMixing, PlainMixingIsUnpreconditionedEvenUnderSmearing) {
    const std::string smearing = "  occupations = 'smearing'\n  degauss = 0.01\n";
    InputData left_to_default = parse_text(silicon_input_with("", smearing));
    EXPECT_EQ(left_to_default.mixing_mode, "broyden");
    EXPECT_EQ(left_to_default.mixing_preconditioning, "none");

    InputData written = parse_text(silicon_input_with("", smearing, "  mixing_mode = 'plain'\n"));
    EXPECT_EQ(written.mixing_mode, "broyden");
    EXPECT_EQ(written.mixing_preconditioning, "none");
}

TEST(QEParserMixing, ThomasFermiModesKeepTheirPreconditioners) {
    InputData average = parse_text(silicon_input_with("", "", "  mixing_mode = 'TF'\n"));
    EXPECT_EQ(average.mixing_mode, "broyden");
    EXPECT_EQ(average.mixing_preconditioning, "kerker");

    InputData local = parse_text(silicon_input_with("", "", "  mixing_mode = 'local-TF'\n"));
    EXPECT_EQ(local.mixing_mode, "broyden");
    EXPECT_EQ(local.mixing_preconditioning, "local-tf");
}
