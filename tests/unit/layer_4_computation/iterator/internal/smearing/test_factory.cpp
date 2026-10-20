#include <gtest/gtest.h>

#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>

#include "layer_4_computation/iterator/internal/smearing/factory.hpp"
#include "layer_4_computation/iterator/internal/smearing/fermi_dirac.hpp"
#include "layer_4_computation/iterator/internal/smearing/fixed.hpp"
#include "layer_4_computation/iterator/internal/smearing/gaussian.hpp"
#include "layer_4_computation/iterator/internal/smearing/marzari_vanderbilt.hpp"
#include "layer_4_computation/iterator/internal/smearing/methfessel_paxton.hpp"

using namespace q3;

// ── Parameterized test: valid smearing creation ─────────────────────

enum class ExpectedSmearingType {
    Fixed,
    Gaussian,
    MethfesselPaxton,
    FermiDirac,
    MarzariVanderbilt
};

struct SmearingFactoryTestCase {
    std::string smearing_name;
    Real smearing_width;
    ExpectedSmearingType expected_type;
};

class SmearingFactoryParameterized : public ::testing::TestWithParam<SmearingFactoryTestCase> {};

TEST_P(SmearingFactoryParameterized, CreateReturnsCorrectType) {
    const auto& test_case = GetParam();
    auto smearing = create_smearing(test_case.smearing_name, test_case.smearing_width);
    ASSERT_NE(smearing, nullptr);
    switch (test_case.expected_type) {
        case ExpectedSmearingType::Fixed:
            EXPECT_NE(dynamic_cast<FixedOccupation*>(smearing.get()), nullptr);
            break;
        case ExpectedSmearingType::Gaussian:
            EXPECT_NE(dynamic_cast<GaussianSmearing*>(smearing.get()), nullptr);
            break;
        case ExpectedSmearingType::MethfesselPaxton:
            EXPECT_NE(dynamic_cast<MethfesselPaxtonSmearing*>(smearing.get()), nullptr);
            break;
        case ExpectedSmearingType::FermiDirac:
            EXPECT_NE(dynamic_cast<FermiDiracSmearing*>(smearing.get()), nullptr);
            break;
        case ExpectedSmearingType::MarzariVanderbilt:
            EXPECT_NE(dynamic_cast<MarzariVanderbiltSmearing*>(smearing.get()), nullptr);
            break;
    }
}

INSTANTIATE_TEST_SUITE_P(
    ValidSmearingSchemes,
    SmearingFactoryParameterized,
    ::testing::Values(
        SmearingFactoryTestCase{"fixed", 0.0, ExpectedSmearingType::Fixed},
        SmearingFactoryTestCase{"none", 0.0, ExpectedSmearingType::Fixed},
        SmearingFactoryTestCase{"", 0.0, ExpectedSmearingType::Fixed},
        SmearingFactoryTestCase{"gaussian", 0.01, ExpectedSmearingType::Gaussian},
        SmearingFactoryTestCase{"gauss", 0.02, ExpectedSmearingType::Gaussian},
        SmearingFactoryTestCase{"methfessel-paxton", 0.02, ExpectedSmearingType::MethfesselPaxton},
        SmearingFactoryTestCase{"m-p", 0.02, ExpectedSmearingType::MethfesselPaxton},
        SmearingFactoryTestCase{"mp", 0.02, ExpectedSmearingType::MethfesselPaxton},
        SmearingFactoryTestCase{"fermi-dirac", 0.02, ExpectedSmearingType::FermiDirac},
        SmearingFactoryTestCase{"f-d", 0.02, ExpectedSmearingType::FermiDirac},
        SmearingFactoryTestCase{"fd", 0.02, ExpectedSmearingType::FermiDirac},
        SmearingFactoryTestCase{
            "marzari-vanderbilt", 0.02, ExpectedSmearingType::MarzariVanderbilt},
        SmearingFactoryTestCase{"cold", 0.02, ExpectedSmearingType::MarzariVanderbilt},
        SmearingFactoryTestCase{"m-v", 0.02, ExpectedSmearingType::MarzariVanderbilt},
        SmearingFactoryTestCase{"mv", 0.02, ExpectedSmearingType::MarzariVanderbilt}),
    [](const ::testing::TestParamInfo<SmearingFactoryTestCase>& info) {
        std::string name = info.param.smearing_name;
        if (name.empty()) {
            name = "empty_string";
        }
        for (auto& character : name) {
            if (character == '-') {
                character = '_';
            }
        }
        return name;
    });

// ── Error case (standalone test) ────────────────────────────────────

TEST(SmearingFactory, UnknownNameThrowsRuntimeError) {
    EXPECT_THROW(create_smearing("unknown"), std::runtime_error);
}

TEST(SmearingFactory, MethfesselPaxtonOrderIsForwarded) {
    auto order_three = create_smearing("mp", 0.02, 3);
    const auto* methfessel_paxton =
        dynamic_cast<const MethfesselPaxtonSmearing*>(order_three.get());
    ASSERT_NE(methfessel_paxton, nullptr);
    EXPECT_EQ(methfessel_paxton->order(), 3);

    auto default_order = create_smearing("mp", 0.02);
    const auto* order_one = dynamic_cast<const MethfesselPaxtonSmearing*>(default_order.get());
    ASSERT_NE(order_one, nullptr);
    EXPECT_EQ(order_one->order(), 1);
}
