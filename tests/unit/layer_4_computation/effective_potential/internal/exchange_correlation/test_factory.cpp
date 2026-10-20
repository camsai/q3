#include <gtest/gtest.h>

#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>

#include "layer_4_computation/effective_potential/internal/exchange_correlation/factory.hpp"
#include "layer_4_computation/effective_potential/internal/exchange_correlation/init.hpp"

using namespace q3;

// ── Parameterized test: creation and gradient flag ──────────────────

struct XCFactoryTestCase {
    std::string functional_name;
    bool expects_gradient;
    int number_of_spin_components;
};

class XCFactoryParameterized : public ::testing::TestWithParam<XCFactoryTestCase> {};

TEST_P(XCFactoryParameterized, CreateReturnsNonNullWithCorrectGradientFlag) {
    const auto& test_case = GetParam();
    auto functional =
        XCFunctionalFactory::create(test_case.functional_name, test_case.number_of_spin_components);
    ASSERT_NE(functional, nullptr);
    EXPECT_EQ(functional->requires_density_gradient(), test_case.expects_gradient);
}

INSTANTIATE_TEST_SUITE_P(NamedFunctionals,
                         XCFactoryParameterized,
                         ::testing::Values(
                             // Named functionals (canonical case)
                             XCFactoryTestCase{"PZ", false, 1},
                             XCFactoryTestCase{"PBE", true, 1},
                             XCFactoryTestCase{"PBEsol", true, 1},
                             XCFactoryTestCase{"BLYP", true, 1},
                             XCFactoryTestCase{"revPBE", true, 1},
                             XCFactoryTestCase{"LDA", false, 1},
                             // Case-insensitive variants
                             XCFactoryTestCase{"pz", false, 1},
                             XCFactoryTestCase{"pbe", true, 1},
                             XCFactoryTestCase{"pbesol", true, 1},
                             XCFactoryTestCase{"lda", false, 1},
                             // LibXC direct format
                             XCFactoryTestCase{"libxc:1:9", false, 1},
                             // RPBE: Hammer-Hansen-Norskov exchange, PBE correlation
                             XCFactoryTestCase{"RPBE", true, 1},
                             XCFactoryTestCase{"rpbe", true, 1},
                             // LibXC halves given by name rather than id
                             XCFactoryTestCase{"libxc:gga_x_rpbe:gga_c_pbe", true, 1},
                             XCFactoryTestCase{"libxc:GGA_X_RPBE:GGA_C_PBE", true, 1},
                             // ... and mixed, one of each
                             XCFactoryTestCase{"libxc:117:gga_c_pbe", true, 1},
                             // Upper-cased prefix: what the YAML parser hands us
                             XCFactoryTestCase{"LIBXC:117:130", true, 1},
                             // QE LibXC format
                             XCFactoryTestCase{"XC-001I-009I-000L-000L", false, 1},
                             // Spin-polarized creation
                             XCFactoryTestCase{"PZ", false, 2}),
                         [](const ::testing::TestParamInfo<XCFactoryTestCase>& info) {
                             /**
                              * Generate a valid test name from the functional name and spin
                              * components
                              */
                             std::string name = info.param.functional_name;
                             // Replace characters that are invalid in test names
                             for (auto& character : name) {
                                 if (character == ':' || character == '-') {
                                     character = '_';
                                 }
                             }
                             if (info.param.number_of_spin_components == 2) {
                                 name += "_spin2";
                             }
                             return name;
                         });

// ── Error cases (kept as standalone tests) ──────────────────────────

TEST(XCFactory, LibXCFormatMissingCorrelationIdThrows) {
    EXPECT_THROW(XCFunctionalFactory::create("libxc:1"), std::runtime_error);
}

TEST(XCFactory, UnknownFunctionalThrows) {
    EXPECT_THROW(XCFunctionalFactory::create("unknown_functional"), std::runtime_error);
}

// ── Naming a LibXC functional instead of numbering it ───────────────

namespace {

/// Run a create() expected to fail, and return the message it produced.
std::string message_from_failed_create(const std::string& functional_name) {
    try {
        XCFunctionalFactory::create(functional_name);
    } catch (const std::runtime_error& error) {
        return error.what();
    }
    return "";
}

} // namespace

TEST(XCFactory, RpbeAliasIsTheSamePairAsItsLibXCIds) {
    /**
     * `rpbe` is the name the functional is known by, and what a user reaches
     * for first. It used to throw "Unknown XC functional", leaving
     * `libxc:117:130` -- the same two functionals, spelled by number -- as the
     * only way to ask for it. The two must stay the same thing, or the alias is
     * a second definition that can drift from the one it stands for.
     */
    auto by_alias = XCFunctionalFactory::create("rpbe");
    auto by_identifier = XCFunctionalFactory::create("libxc:117:130");
    auto by_component_name = XCFunctionalFactory::create("libxc:gga_x_rpbe:gga_c_pbe");

    ASSERT_NE(by_alias, nullptr);
    ASSERT_NE(by_identifier, nullptr);
    ASSERT_NE(by_component_name, nullptr);
    EXPECT_EQ(by_alias->name(), by_identifier->name());
    EXPECT_EQ(by_alias->name(), by_component_name->name());
}

TEST(XCFactory, UpperCasedLibXCPrefixIsAccepted) {
    /**
     * The YAML parser upper-cases `model.functional` before storing it, so a
     * manifest asking for `libxc:117:130` arrived here as `LIBXC:117:130` and
     * failed the prefix test -- the `libxc:` form was unusable from a manifest
     * entirely, while working from a namelist input, which passes `input_dft`
     * through as written.
     */
    auto upper = XCFunctionalFactory::create("LIBXC:117:130");
    auto lower = XCFunctionalFactory::create("libxc:117:130");
    ASSERT_NE(upper, nullptr);
    EXPECT_EQ(upper->name(), lower->name());
}

TEST(XCFactory, AnUnknownLibXCComponentNameSaysSo) {
    /**
     * A name LibXC does not know must be refused by name. `xc_functional_get_number`
     * answers -1, and passing that on to `xc_func_init` as an id is how a typo
     * turns into a failure that names neither the field nor the file.
     */
    std::string message = message_from_failed_create("libxc:gga_x_nosuchthing:130");
    EXPECT_NE(message.find("gga_x_nosuchthing"), std::string::npos) << message;
    EXPECT_NE(message.find("LibXC does not know"), std::string::npos) << message;
}

TEST(XCFactory, AnEmptyLibXCComponentSaysSo) {
    std::string message = message_from_failed_create("libxc::130");
    EXPECT_NE(message.find("Empty exchange or correlation field"), std::string::npos) << message;
}
