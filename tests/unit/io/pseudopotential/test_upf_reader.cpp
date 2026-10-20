#include "io/pseudopotential/upf_reader.hpp"
#include "io/pseudopotential/upf_reader/utilities.hpp"
#include "io/pseudopotential/upf_reader/v2/ultrasoft.hpp"
#include "layer_3_representation/pseudopotential/init.hpp"
#include "layer_3_representation/pseudopotential/ultrasoft/init.hpp"
#include "layer_3_representation/quantum_system/kohn_sham/planewave/norm_conserving/init.hpp"
#include "utils/temporary_path.hpp"
#include <gtest/gtest.h>

#include <cstdio>
#include <fstream>
#include <string>

using namespace q3;

TEST(UPFReaderTest, NormConservingRead) {
    auto pp =
        UPFReader::read(std::string(EXAMPLES_DIR) + "/data/pseudopotentials/nc/Si.pz-vbc.UPF");

    EXPECT_EQ(pp->pseudopotential_type_enum(), PseudopotentialType::norm_conserving);
    EXPECT_NEAR(pp->valence_charge, 4.0, 1e-6);
    EXPECT_EQ(pp->number_of_projectors, 2);
    EXPECT_EQ((int)pp->d_ion_matrix.size(), 2);
    EXPECT_NEAR(pp->d_ion_matrix[0][0], 1.52388501179, 1e-6);
    EXPECT_NEAR(pp->d_ion_matrix[1][1], 3.68330413052, 1e-6);
}

TEST(UPFReaderTest, UltrasoftRead) {
    auto pp =
        UPFReader::read(std::string(EXAMPLES_DIR) + "/data/pseudopotentials/us/Si_PBE_USPP.UPF");

    ASSERT_EQ(pp->pseudopotential_type_enum(), PseudopotentialType::ultrasoft);
    EXPECT_NEAR(pp->valence_charge, 4.0, 1e-6);
    EXPECT_EQ(pp->number_of_projectors, 4);

    // Verify Q_ij data is properly read
    const auto& us_pp = static_cast<const UltrasoftPseudopotential&>(*pp);
    EXPECT_EQ((int)us_pp.augmentation_overlap_integrals.size(), 4);
    EXPECT_EQ((int)us_pp.augmentation_function.size(), 16); // nbeta * nbeta

    // Verify Dion matrix is properly read
    EXPECT_NEAR(pp->d_ion_matrix[0][0], 0.480044539008, 1e-8);
    EXPECT_NEAR(pp->d_ion_matrix[0][1], 0.766543755615, 1e-8);
    EXPECT_NEAR(pp->d_ion_matrix[1][1], 1.07394726927, 1e-8);
}

// ── Error-path tests ────────────────────────────────────────────────

TEST(UPFReaderTest, NonexistentFileThrows) {
    EXPECT_THROW(UPFReader::read("/nonexistent/path/to/Si.UPF"), std::runtime_error);
}

namespace {

/// Run a read expected to fail, and return the error message it produced.
std::string message_from_failed_read(const std::string& path) {
    try {
        UPFReader::read(path);
    } catch (const std::runtime_error& error) {
        return error.what();
    }
    return "";
}

} // namespace

TEST(UPFReaderTest, TruncatedUPFv2ContentNamesTheMissingAttribute) {
    /**
     * A file that looks like UPF v2 (has version="2" in first line) but is
     * truncated. The error has to name the file and the field: the standard
     * conversions on their own report "stoi: no conversion", which tells
     * someone holding an unreadable pseudopotential nothing about where to look.
     */
    std::string temporary_path = test_support::make_temporary_path(".UPF");
    {
        std::ofstream output_stream(temporary_path);
        output_stream << "<UPF version=\"2.0.1\">\n"
                      << "<PP_HEADER\n"
                      << "  pseudo_type=\"NC\"\n"
                      << "/>\n";
    }
    EXPECT_THROW(UPFReader::read(temporary_path), std::runtime_error);

    std::string message = message_from_failed_read(temporary_path);
    EXPECT_NE(message.find("z_valence"), std::string::npos) << message;
    EXPECT_NE(message.find("is missing"), std::string::npos) << message;
    EXPECT_NE(message.find(temporary_path), std::string::npos) << message;
    std::remove(temporary_path.c_str());
}

TEST(UPFReaderTest, UnparseableRequiredAttributeIsRejectedNotTruncated) {
    /**
     * std::stoi("20x0") returns 20 and reports success. A mesh size silently
     * truncated from a corrupt attribute would produce a pseudopotential that
     * loads and is wrong, so trailing characters are rejected outright.
     */
    std::string temporary_path = test_support::make_temporary_path(".UPF");
    {
        std::ofstream output_stream(temporary_path);
        output_stream << "<UPF version=\"2.0.1\">\n"
                      << "<PP_HEADER\n"
                      << "  pseudo_type=\"NC\"\n"
                      << "  z_valence=\"4.0\"\n"
                      << "  mesh_size=\"20x0\"\n"
                      << "  number_of_proj=\"2\"\n"
                      << "  l_max=\"1\"\n"
                      << "/>\n";
    }
    std::string message = message_from_failed_read(temporary_path);
    EXPECT_NE(message.find("mesh_size"), std::string::npos) << message;
    EXPECT_NE(message.find("20x0"), std::string::npos) << message;
    std::remove(temporary_path.c_str());
}

TEST(UPFReaderTest, UPFv2PAWDeclaringHeaderWithoutPawSectionsIsMalformed) {
    /**
     * v2 PAW datasets are read, not rejected — see test_paw_upf_reader.cpp for
     * the reader and for the calculation-level guard that keeps a PAW dataset
     * from reaching an SCF. What must still fail is a file that announces
     * pseudo_type="PAW" and then omits the sections a PAW dataset is made of:
     * that is a malformed file, and it is reported as one rather than as a
     * recognized-but-unsupported type. This header omits l_max, so the shared
     * v2 header parser names that field before any PAW section is reached.
     */
    std::string temporary_path = test_support::make_temporary_path(".UPF");
    {
        std::ofstream output_stream(temporary_path);
        output_stream << "<UPF version=\"2.0.1\">\n"
                      << "<PP_HEADER\n"
                      << "  pseudo_type=\"PAW\"\n"
                      << "  element=\"Si\"\n"
                      << "  z_valence=\"4.0\"\n"
                      << "  mesh_size=\"100\"\n"
                      << "  number_of_proj=\"2\"\n"
                      << "  number_of_wfc=\"2\"\n"
                      << "/>\n";
    }
    EXPECT_THROW(UPFReader::read(temporary_path), std::runtime_error);

    std::string message = message_from_failed_read(temporary_path);
    EXPECT_NE(message.find("l_max"), std::string::npos) << message;
    EXPECT_NE(message.find(temporary_path), std::string::npos) << message;
    std::remove(temporary_path.c_str());
}

TEST(UPFReaderTest, UPFv1PAWRejectedWithTypedError) {
    /**
     * A v1 file whose header declares type PAW must be rejected with
     * UnsupportedPseudopotentialTypeError as soon as the header is parsed,
     * before the (missing) NC/US sections are reached.
     */
    std::string temporary_path = test_support::make_temporary_path(".UPF");
    {
        std::ofstream output_stream(temporary_path);
        output_stream << "<PP_INFO>\n"
                      << "Generated for testing\n"
                      << "</PP_INFO>\n"
                      << "<PP_HEADER>\n"
                      << "   0                   Version Number\n"
                      << "  Si                   Element\n"
                      << "   PAW                 PAW pseudopotential\n"
                      << "</PP_HEADER>\n";
    }
    EXPECT_THROW(UPFReader::read(temporary_path), UnsupportedPseudopotentialTypeError);
    std::remove(temporary_path.c_str());
}

// ── q_with_l="false" augmentation: PP_QIJ.i.j tags matched by attribute ──

TEST(UPFReaderTest, QWithLFalseAugmentationTagsAreMatchedByAttributeNotByGuessedName) {
    /**
     * `q_with_l="false"` UPF v2 files (the older un-decomposed augmentation
     * format, still emitted by common generators and widely distributed) name
     * their un-decomposed Q_ij(r) blocks "<PP_QIJ.i.j>" with i, j the
     * 1-based first_index/second_index carried as XML attributes — not
     * "<PP_QIJ.N>" for some linear/composite index N baked into the tag
     * name. A reader that computes N and searches for the literal substring
     * "<PP_QIJ." + N matches the WRONG block for almost every pair: for 3
     * projectors, tags appear in file order 1.1, 1.2, 1.3, 2.2, 2.3, 3.3;
     * pair (i=1, j=2) needs N=2, and content_.find("<PP_QIJ.2") returns the
     * position of "<PP_QIJ.2.2" (the first tag in the file that literally
     * starts with that prefix), not "<PP_QIJ.1.2". This reproduces that
     * exact scenario and checks the (1, 2) pair lands on its own data — the
     * 1.2 block's q_int and radial values — not the 2.2 block's.
     *
     * This one substitution, cross-wired onto every off-diagonal pair,
     * previously made Q3's ultrasoft augmentation charge fail to conserve
     * total electron count from the very first SCF iteration for any
     * q_with_l="false" pseudopotential.
     */
    std::string content = "<UPF version=\"2.0.1\">\n"
                          "<PP_AUGMENTATION q_with_l=\"false\" nqf=\"0\" nqlc=\"1\">\n"
                          "<PP_QIJ.1.1 first_index=\"1\" second_index=\"1\" "
                          "composite_index=\"1\" q_int=\"1.100000000000e0\">\n"
                          "1.100000000000e0 1.100000000000e0 1.100000000000e0\n"
                          "</PP_QIJ.1.1>\n"
                          "<PP_QIJ.1.2 first_index=\"1\" second_index=\"2\" "
                          "composite_index=\"2\" q_int=\"1.200000000000e0\">\n"
                          "1.200000000000e0 1.200000000000e0 1.200000000000e0\n"
                          "</PP_QIJ.1.2>\n"
                          "<PP_QIJ.1.3 first_index=\"1\" second_index=\"3\" "
                          "composite_index=\"4\" q_int=\"1.300000000000e0\">\n"
                          "1.300000000000e0 1.300000000000e0 1.300000000000e0\n"
                          "</PP_QIJ.1.3>\n"
                          "<PP_QIJ.2.2 first_index=\"2\" second_index=\"2\" "
                          "composite_index=\"3\" q_int=\"2.200000000000e0\">\n"
                          "2.200000000000e0 2.200000000000e0 2.200000000000e0\n"
                          "</PP_QIJ.2.2>\n"
                          "<PP_QIJ.2.3 first_index=\"2\" second_index=\"3\" "
                          "composite_index=\"5\" q_int=\"2.300000000000e0\">\n"
                          "2.300000000000e0 2.300000000000e0 2.300000000000e0\n"
                          "</PP_QIJ.2.3>\n"
                          "<PP_QIJ.3.3 first_index=\"3\" second_index=\"3\" "
                          "composite_index=\"6\" q_int=\"3.300000000000e0\">\n"
                          "3.300000000000e0 3.300000000000e0 3.300000000000e0\n"
                          "</PP_QIJ.3.3>\n"
                          "</PP_AUGMENTATION>\n"
                          "</UPF>\n";

    UltrasoftPseudopotential pseudopotential;
    pseudopotential.number_of_projectors = 3;
    pseudopotential.mesh_size = 3;

    UltrasoftUPFv2Reader reader(pseudopotential, content);
    reader.read();

    // q_int attribute: each pair must carry its own tag's value.
    EXPECT_DOUBLE_EQ(pseudopotential.augmentation_overlap_integrals[0][0], 1.1);
    EXPECT_DOUBLE_EQ(pseudopotential.augmentation_overlap_integrals[0][1], 1.2);
    EXPECT_DOUBLE_EQ(pseudopotential.augmentation_overlap_integrals[0][2], 1.3);
    EXPECT_DOUBLE_EQ(pseudopotential.augmentation_overlap_integrals[1][1], 2.2);
    EXPECT_DOUBLE_EQ(pseudopotential.augmentation_overlap_integrals[1][2], 2.3);
    EXPECT_DOUBLE_EQ(pseudopotential.augmentation_overlap_integrals[2][2], 3.3);
    // Symmetric across the diagonal.
    EXPECT_DOUBLE_EQ(pseudopotential.augmentation_overlap_integrals[1][0], 1.2);
    EXPECT_DOUBLE_EQ(pseudopotential.augmentation_overlap_integrals[2][0], 1.3);
    EXPECT_DOUBLE_EQ(pseudopotential.augmentation_overlap_integrals[2][1], 2.3);

    // Radial Q_ij(r) data: same attribution check, at the vector level.
    ASSERT_EQ(pseudopotential.augmentation_function[0 * 3 + 1].size(), 3u);
    EXPECT_DOUBLE_EQ(pseudopotential.augmentation_function[0 * 3 + 1][0], 1.2);
    ASSERT_EQ(pseudopotential.augmentation_function[1 * 3 + 1].size(), 3u);
    EXPECT_DOUBLE_EQ(pseudopotential.augmentation_function[1 * 3 + 1][0], 2.2);
}

// ── pseudo_type spelling normalization ──────────────────────────────

TEST(UPFReaderTest, CanonicalPseudopotentialTypeAcceptsTheSpellingsGeneratorsEmit) {
    /**
     * The UPF header documents "NC", "SL", "US" and "PAW", but the field is not
     * written uniformly: generators also emit "USPP" for an ultrasoft dataset.
     * Everything downstream keys off the canonical trio — the reader factory
     * picks the subclass to construct by comparing against them, and
     * `pseudopotential_kind_name` renders them into the provenance record — so a
     * spelling that misses the comparison decides, silently, that a dataset has
     * no augmentation charges.
     */
    using upf_utilities::canonical_pseudopotential_type;

    EXPECT_EQ(canonical_pseudopotential_type("US"), "US");
    EXPECT_EQ(canonical_pseudopotential_type("USPP"), "US");
    EXPECT_EQ(canonical_pseudopotential_type("us"), "US");
    EXPECT_EQ(canonical_pseudopotential_type("uspp"), "US");
    EXPECT_EQ(canonical_pseudopotential_type("  US  "), "US");

    EXPECT_EQ(canonical_pseudopotential_type("NC"), "NC");
    EXPECT_EQ(canonical_pseudopotential_type("nc"), "NC");
    // "SL" is the semilocal member of the norm-conserving family.
    EXPECT_EQ(canonical_pseudopotential_type("SL"), "NC");

    EXPECT_EQ(canonical_pseudopotential_type("PAW"), "PAW");
    EXPECT_EQ(canonical_pseudopotential_type("paw"), "PAW");

    /**
     * An unrecognized value is passed through trimmed rather than forced into
     * one of the three. It then lands on the norm-conserving construction path
     * exactly as before, and the label it came with survives for diagnostics.
     */
    EXPECT_EQ(canonical_pseudopotential_type("1/r"), "1/r");
    EXPECT_EQ(canonical_pseudopotential_type(""), "");
}

TEST(UPFReaderTest, UPFv2UltrasoftSpelledUSPPIsBuiltAsUltrasoft) {
    /**
     * A UPF v2 carbon USPP written by the `atomic` code reports
     * pseudo_type="USPP". Against an exact match on "US" that read as "not
     * ultrasoft", so the file was constructed as a plain norm-conserving
     * `Pseudopotential`: every Q_ij it carried was dropped on the floor, no Q(G)
     * table was built for it, and the first code to ask the species for its
     * augmentation data failed the `as_ultrasoft` cast.
     *
     * `number_of_proj="0"` keeps the fixture to the header question — the
     * subclass the factory chooses — without needing the projector blocks.
     */
    std::string temporary_path = test_support::make_temporary_path(".UPF");
    {
        std::ofstream output_stream(temporary_path);
        output_stream << "<UPF version=\"2.0.1\">\n"
                      << "<PP_HEADER\n"
                      << "  element=\"C\"\n"
                      << "  pseudo_type=\"USPP\"\n"
                      << "  z_valence=\"4.0\"\n"
                      << "  mesh_size=\"0\"\n"
                      << "  number_of_proj=\"0\"\n"
                      << "  l_max=\"2\"\n"
                      << "/>\n"
                      << "</UPF>\n";
    }

    auto pseudopotential = UPFReader::read(temporary_path);

    EXPECT_EQ(pseudopotential->pseudopotential_type, "US");
    EXPECT_EQ(pseudopotential->pseudopotential_type_enum(), PseudopotentialType::ultrasoft);
    EXPECT_TRUE(pseudopotential->has_augmentation_charges());
    EXPECT_NE(dynamic_cast<const UltrasoftPseudopotential*>(pseudopotential.get()), nullptr);
    std::remove(temporary_path.c_str());
}

TEST(UPFReaderTest, UPFv2NormConservingIsNotDraggedIntoTheUltrasoftPath) {
    /**
     * The other half of the same question: widening the accepted spellings must
     * not start classifying norm-conserving datasets as ultrasoft.
     */
    std::string temporary_path = test_support::make_temporary_path(".UPF");
    {
        std::ofstream output_stream(temporary_path);
        output_stream << "<UPF version=\"2.0.1\">\n"
                      << "<PP_HEADER\n"
                      << "  element=\"C\"\n"
                      << "  pseudo_type=\"NC\"\n"
                      << "  z_valence=\"4.0\"\n"
                      << "  mesh_size=\"0\"\n"
                      << "  number_of_proj=\"0\"\n"
                      << "  l_max=\"1\"\n"
                      << "/>\n"
                      << "</UPF>\n";
    }

    auto pseudopotential = UPFReader::read(temporary_path);

    EXPECT_EQ(pseudopotential->pseudopotential_type, "NC");
    EXPECT_EQ(pseudopotential->pseudopotential_type_enum(), PseudopotentialType::norm_conserving);
    EXPECT_FALSE(pseudopotential->has_augmentation_charges());
    std::remove(temporary_path.c_str());
}

TEST(UPFReaderTest, UPFv1UltrasoftSpelledUSPPIsBuiltAsUltrasoft) {
    /**
     * The v1 header carries the same field as a bare token on a line of its
     * own, and the same spellings turn up there, so the v1 dispatch is
     * normalized alongside the v2 one.
     *
     * The fixture is a complete v1 file rather than a header fragment because
     * the v1 sections are read positionally in sequence; it is kept minimal by
     * declaring no projectors and no wavefunctions, which is enough to settle
     * the question this test asks — which subclass the factory builds.
     */
    std::string temporary_path = test_support::make_temporary_path(".UPF");
    {
        std::ofstream output_stream(temporary_path);
        output_stream << "<PP_INFO>\n"
                      << "Generated for testing\n"
                      << "</PP_INFO>\n"
                      << "<PP_HEADER>\n"
                      << "   0                   Version Number\n"
                      << "   C                   Element\n"
                      << "   USPP                Ultrasoft pseudopotential\n"
                      << "   F                   Nonlinear Core Correction\n"
                      << " SLA  PW   PBE  PBE    Exchange-Correlation functional\n"
                      << "   4.00000000000       Z valence\n"
                      << "   0.00000000000       Total energy\n"
                      << "   0.00000    0.00000  Suggested cutoff for wfc and rho\n"
                      << "   1                   Max angular momentum component\n"
                      << "   3                   Number of points in mesh\n"
                      << "   0    0              Number of Wavefunctions, Number of Projectors\n"
                      << " Wavefunctions         nl  l   occ\n"
                      << "</PP_HEADER>\n"
                      << "<PP_MESH>\n"
                      << "<PP_R>\n"
                      << "  1.0000  2.0000  3.0000\n"
                      << "</PP_R>\n"
                      << "<PP_RAB>\n"
                      << "  1.0000  1.0000  1.0000\n"
                      << "</PP_RAB>\n"
                      << "</PP_MESH>\n"
                      << "<PP_LOCAL>\n"
                      << " -1.0000 -0.5000 -0.2500\n"
                      << "</PP_LOCAL>\n"
                      << "<PP_NONLOCAL>\n"
                      << "<PP_QIJ>\n"
                      << "  0                   nqf\n"
                      << "</PP_QIJ>\n"
                      << "<PP_DIJ>\n"
                      << "  0                   Number of nonzero Dij\n"
                      << "</PP_DIJ>\n"
                      << "</PP_NONLOCAL>\n"
                      << "<PP_PSWFC>\n"
                      << "</PP_PSWFC>\n"
                      << "<PP_RHOATOM>\n"
                      << "  0.1000  0.2000  0.3000\n"
                      << "</PP_RHOATOM>\n";
    }

    auto pseudopotential = UPFReader::read(temporary_path);

    EXPECT_EQ(pseudopotential->pseudopotential_type, "US");
    EXPECT_EQ(pseudopotential->pseudopotential_type_enum(), PseudopotentialType::ultrasoft);
    EXPECT_TRUE(pseudopotential->has_augmentation_charges());
    EXPECT_NE(dynamic_cast<const UltrasoftPseudopotential*>(pseudopotential.get()), nullptr);
    std::remove(temporary_path.c_str());
}
