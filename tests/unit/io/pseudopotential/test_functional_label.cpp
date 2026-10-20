#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "io/pseudopotential/functional_label.hpp"
#include "layer_4_computation/effective_potential/internal/exchange_correlation/factory.hpp"

using namespace q3;

// ─── Short names ─────────────────────────────────────────────────────────────

TEST(FunctionalFromPseudopotentialLabel, ShortNamesMapCaseInsensitively) {
    EXPECT_EQ(functional_from_pseudopotential_label("PBE"), "pbe");
    EXPECT_EQ(functional_from_pseudopotential_label("pbe"), "pbe");
    EXPECT_EQ(functional_from_pseudopotential_label("  PBE  "), "pbe");
    EXPECT_EQ(functional_from_pseudopotential_label("PZ"), "pz");
    EXPECT_EQ(functional_from_pseudopotential_label("LDA"), "pz");
    EXPECT_EQ(functional_from_pseudopotential_label("PBEsol"), "pbesol");
    EXPECT_EQ(functional_from_pseudopotential_label("BLYP"), "blyp");
    EXPECT_EQ(functional_from_pseudopotential_label("revPBE"), "revpbe");
    EXPECT_EQ(functional_from_pseudopotential_label("RPBE"), "rpbe");
}

// ─── Component sets ──────────────────────────────────────────────────────────

TEST(FunctionalFromPseudopotentialLabel, ComponentSetsMapWhateverTheSeparator) {
    EXPECT_EQ(functional_from_pseudopotential_label("SLA PW PBX PBC"), "pbe");
    EXPECT_EQ(functional_from_pseudopotential_label(" SLA  PW   PBX  PBC"), "pbe");
    EXPECT_EQ(functional_from_pseudopotential_label("SLA-PW-PBX-PBC"), "pbe");
    EXPECT_EQ(functional_from_pseudopotential_label("sla+pw+pbx+pbc"), "pbe");
}

TEST(FunctionalFromPseudopotentialLabel, EveryMappedComponentSet) {
    EXPECT_EQ(functional_from_pseudopotential_label("SLA PZ NOGX NOGC"), "pz");
    // Older files spell PBE's two gradient terms "PBE PBE".
    EXPECT_EQ(functional_from_pseudopotential_label("SLA PW PBE PBE"), "pbe");
    EXPECT_EQ(functional_from_pseudopotential_label("SLA PW PSX PSC"), "pbesol");
    EXPECT_EQ(functional_from_pseudopotential_label("SLA LYP B88 BLYP"), "blyp");
    EXPECT_EQ(functional_from_pseudopotential_label("SLA PW REVX PBC"), "revpbe");
    EXPECT_EQ(functional_from_pseudopotential_label("SLA PW RPB PBC"), "revpbe");
}

/**
 * A UPF v1 header repeats the short name after the four components. It has to
 * agree with them, and any other trailing term names a different functional.
 */
TEST(FunctionalFromPseudopotentialLabel, VersionOneShortNameMustAgree) {
    EXPECT_EQ(functional_from_pseudopotential_label("SLA PW PBX PBC PBE"), "pbe");
    EXPECT_EQ(functional_from_pseudopotential_label("SLA PW PBE PBE PBE"), "pbe");
    EXPECT_EQ(functional_from_pseudopotential_label("SLA PZ NOGX NOGC PZ"), "pz");
    EXPECT_EQ(functional_from_pseudopotential_label("SLA PW PBX PBC PZ"), "");
    EXPECT_EQ(functional_from_pseudopotential_label("SLA PW REVX PBC VDW1"), "");
}

// ─── Pass-through and refusal ────────────────────────────────────────────────

TEST(FunctionalFromPseudopotentialLabel, LibxcSpellingPassesThroughTrimmedAndLowerCased) {
    EXPECT_EQ(functional_from_pseudopotential_label("XC-000I-000I-101L-130L"),
              "xc-000i-000i-101l-130l");
    EXPECT_EQ(functional_from_pseudopotential_label("  XC-000I-000I-101L-130L "),
              "xc-000i-000i-101l-130l");
    EXPECT_EQ(functional_from_pseudopotential_label("xc-000i-000i-101l-130l"),
              "xc-000i-000i-101l-130l");
}

/**
 * Case never matters, in any of the three shapes: the datasets of one input are
 * checked for agreement on what this returns, so two spellings of one
 * functional must come back as the same string.
 */
TEST(FunctionalFromPseudopotentialLabel, EveryShapeIgnoresCase) {
    const std::vector<std::vector<std::string>> spellings = {
        {"PBE", "pbe", "Pbe"},
        {"PBESOL", "pbesol", "PBEsol"},
        {"REVPBE", "revpbe", "revPBE"},
        {"SLA PW PBX PBC", "sla pw pbx pbc", "Sla-Pw-Pbx-Pbc"},
        {"SLA PZ NOGX NOGC PZ", "sla pz nogx nogc pz", "SLA PZ NOGX NOGC pz"},
        {"XC-000I-000I-101L-130L", "xc-000i-000i-101l-130l", "Xc-000I-000i-101L-130l"},
    };
    for (const auto& variants : spellings) {
        const std::string expected = functional_from_pseudopotential_label(variants.front());
        ASSERT_FALSE(expected.empty()) << variants.front();
        for (const auto& variant : variants)
            EXPECT_EQ(functional_from_pseudopotential_label(variant), expected) << variant;
    }
}

/**
 * Nothing nearby is substituted. Perdew-Wang LDA is the case that matters most:
 * it is an LDA, but not the PZ one Q3 runs under that name.
 */
TEST(FunctionalFromPseudopotentialLabel, UnmappedLabelsAreEmpty) {
    for (const std::string label : {"",
                                    "   ",
                                    "PW",
                                    "SLA PW NOGX NOGC",
                                    "PW91",
                                    "SLA PW GGX GGC",
                                    "PBE0",
                                    "HSE",
                                    "SCAN",
                                    "GGA",
                                    "SLA PW"}) {
        EXPECT_EQ(functional_from_pseudopotential_label(label), "") << "label '" << label << "'";
    }
}

/**
 * Whatever the mapping returns has to be a name the exchange-correlation
 * factory runs, or the input that relied on it would fail later and further
 * from the cause.
 */
TEST(FunctionalFromPseudopotentialLabel, EveryMappedNameIsOneTheFactoryRuns) {
    const std::vector<std::string> labels = {"PZ",
                                             "PBE",
                                             "PBESOL",
                                             "BLYP",
                                             "REVPBE",
                                             "RPBE",
                                             "SLA PZ NOGX NOGC",
                                             "SLA PW PBX PBC",
                                             "SLA PW PSX PSC",
                                             "SLA LYP B88 BLYP",
                                             "SLA PW REVX PBC",
                                             "XC-000I-000I-101L-130L",
                                             "xc-000i-000i-101l-130l"};
    for (const auto& label : labels) {
        const std::string functional = functional_from_pseudopotential_label(label);
        ASSERT_FALSE(functional.empty()) << "label '" << label << "'";
        EXPECT_NO_THROW(XCFunctionalFactory::create(functional)) << "functional " << functional;
    }
}
