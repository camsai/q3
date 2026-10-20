#include "io/provenance/planewave/pseudopotential.hpp"

#include <gtest/gtest.h>
#include <string>

#include "utils/sha256.hpp"

using namespace q3;

namespace {

/// A minimal silicon input, enough for a full per-species dataset entry.
InputData make_silicon_input() {
    InputData input;
    input.wavefunction_energy_cutoff = 48.0;
    input.charge_density_energy_cutoff = 192.0;
    input.pseudopotential_directory = std::string(EXAMPLES_DIR) + "/data/pseudopotentials/nc/";

    AtomSpecies silicon;
    silicon.symbol = "Si";
    silicon.mass = 28.0855;
    silicon.pseudopotential_file = "Si.upf";
    input.species.push_back(silicon);
    input.number_of_species = 1;

    return input;
}

/// The datasets this record contributes, as they appear in the finished block.
nlohmann::json pseudopotentials_of(const InputData& input) {
    return PseudopotentialProvenanceRecord(input).to_json()["method"]["pseudopotentials"];
}

} // namespace

/**
 * The pseudopotential's content hash is the identity the registry keys on, so
 * it has to be the hash of the file itself and not of anything Q3 derived from
 * it. Computed here independently and compared.
 */
TEST(PseudopotentialProvenanceRecord, IdentifiesEachPseudopotentialByTheHashOfItsContents) {
    const nlohmann::json pseudopotentials = pseudopotentials_of(make_silicon_input());

    ASSERT_EQ(pseudopotentials.size(), 1u);
    const nlohmann::json& silicon = pseudopotentials[0];

    EXPECT_EQ(silicon["element_symbol"], "Si");
    EXPECT_EQ(silicon["source_filename"], "Si.upf");
    EXPECT_EQ(silicon["kind"], "norm_conserving");
    EXPECT_DOUBLE_EQ(silicon["valence_electron_count"].get<double>(), 4.0);

    const std::string expected_digest = SHA256::hexadecimal_digest_of_file(
        std::string(EXAMPLES_DIR) + "/data/pseudopotentials/nc/Si.upf");
    EXPECT_EQ(silicon["content_sha256"], expected_digest);
}

/**
 * A retained record is copied between machines and registered, so it must not
 * carry the path it happened to be produced from. The filename identifies the
 * dataset to a human; the hash identifies it to everything else.
 */
TEST(PseudopotentialProvenanceRecord, CarriesNoHostSpecificPaths) {
    InputData input = make_silicon_input();
    input.species[0].pseudopotential_file =
        std::string(EXAMPLES_DIR) + "/data/pseudopotentials/nc/Si.upf";

    const std::string serialized = PseudopotentialProvenanceRecord(input).to_json().dump();

    EXPECT_EQ(serialized.find("/Users/"), std::string::npos);
    EXPECT_EQ(serialized.find("/home/"), std::string::npos);
    EXPECT_NE(serialized.find("Si.upf"), std::string::npos);
}

/**
 * An unrecognized pseudopotential type passes through as itself rather than
 * being coerced into one of the three known spellings: a new type showing up
 * under its own name is a question, one silently registering as
 * norm-conserving is a wrong answer.
 */
TEST(PseudopotentialProvenanceRecord, MapsTypeTagsAndPassesUnknownOnesThrough) {
    EXPECT_EQ(pseudopotential_kind_name("NC"), "norm_conserving");
    EXPECT_EQ(pseudopotential_kind_name("US"), "ultrasoft");
    EXPECT_EQ(pseudopotential_kind_name("PAW"), "projector_augmented_wave");
    EXPECT_EQ(pseudopotential_kind_name("SOMETHING_NEW"), "SOMETHING_NEW");
}

/**
 * Reporting on a finished calculation must not be able to end it. A dataset
 * that cannot be re-read degrades that species' entry to what is still known
 * -- an empty digest, which admits it does not know -- rather than throwing
 * away an answer that was already computed.
 */
TEST(PseudopotentialProvenanceRecord, DegradesRatherThanThrowingWhenADatasetCannotBeRead) {
    InputData input = make_silicon_input();
    input.pseudopotential_directory = "/nonexistent/q3/pseudopotentials/";
    input.species[0].pseudopotential_file = "Si.upf";

    nlohmann::json pseudopotentials;
    ASSERT_NO_THROW(pseudopotentials = pseudopotentials_of(input));

    const nlohmann::json& silicon = pseudopotentials[0];
    EXPECT_EQ(silicon["element_symbol"], "Si");
    EXPECT_EQ(silicon["content_sha256"], "");
}
