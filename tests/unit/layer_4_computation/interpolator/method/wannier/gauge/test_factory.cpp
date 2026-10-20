#include <gtest/gtest.h>

#include <memory>
#include <stdexcept>
#include <string>

#include "layer_4_computation/interpolator/method/wannier/construction_inputs.hpp"
#include "layer_4_computation/interpolator/method/wannier/diagnostic.hpp"
#include "layer_4_computation/interpolator/method/wannier/diagnostic_error.hpp"
#include "layer_4_computation/interpolator/method/wannier/gauge/factory.hpp"
#include "layer_4_computation/interpolator/method/wannier/gauge/init.hpp"
#include "layer_4_computation/interpolator/method/wannier/gauge/maximally_localized_gauge_choice.hpp"
#include "layer_4_computation/interpolator/method/wannier/gauge/projection_gauge_choice.hpp"
#include "layer_4_computation/interpolator/method/wannier/inputs.hpp"
#include "layer_4_computation/interpolator/method/wannier/settings/init.hpp"
#include "utils/wannier_chain_model.hpp"

using namespace q3;
using q3::test_utils::inputs_of;
using q3::test_utils::make_wannier_chain;

namespace {

WannierInputs chain_record(int number_of_kpoints) {
    return inputs_of(make_wannier_chain(number_of_kpoints, {0.0, 2.0}, 5.5, 0.9));
}

WannierInputs record_without_overlaps() {
    WannierInputs record = chain_record(6);
    record.overlap_matrices.clear();
    return record;
}

WannierConstructionSettings policy(LocalizationPolicy localization) {
    WannierConstructionSettings settings;
    settings.localization = localization;
    return settings;
}

bool keeps_the_projection_gauge(const std::unique_ptr<GaugeChoice>& choice) {
    return dynamic_cast<const ProjectionGaugeChoice*>(choice.get()) != nullptr;
}

bool localizes(const std::unique_ptr<GaugeChoice>& choice) {
    return dynamic_cast<const MaximallyLocalizedGaugeChoice*>(choice.get()) != nullptr;
}

} // namespace

TEST(GaugeChoiceFactory, NoLocalizationKeepsTheProjectionGaugeWhateverTheRecordCarries) {
    const WannierInputs record = chain_record(6);
    const WannierConstructionInputs inputs(record);
    EXPECT_TRUE(
        keeps_the_projection_gauge(create_gauge_choice(policy(LocalizationPolicy::None), inputs)));
}

TEST(GaugeChoiceFactory, APreferredLocalizationLocalizesWhatTheRecordSupports) {
    const WannierInputs record = chain_record(6);
    const WannierConstructionInputs inputs(record);
    EXPECT_TRUE(localizes(create_gauge_choice(policy(LocalizationPolicy::Preferred), inputs)));
}

TEST(GaugeChoiceFactory, APreferredLocalizationYieldsToARecordWithoutOverlaps) {
    const WannierInputs record = record_without_overlaps();
    const WannierConstructionInputs inputs(record);
    EXPECT_TRUE(keeps_the_projection_gauge(
        create_gauge_choice(policy(LocalizationPolicy::Preferred), inputs)));
}

TEST(GaugeChoiceFactory, ARequiredLocalizationRefusesARecordWithoutOverlaps) {
    const WannierInputs record = record_without_overlaps();
    const WannierConstructionInputs inputs(record);
    try {
        const std::unique_ptr<GaugeChoice> choice =
            create_gauge_choice(policy(LocalizationPolicy::Required), inputs);
        FAIL() << "a demanded localisation must not be downgraded silently";
    } catch (const WannierDiagnosticError& error) {
        EXPECT_EQ(error.diagnostic(), WannierDiagnostic::localization_unsupported);
        const std::string message = error.what();
        EXPECT_NE(message.find("localization: none"), std::string::npos) << message;
    }
}

/// Two k-points along the chain leave k+b and k-b the same point; the refusal costs no iteration.
TEST(GaugeChoiceFactory, ATwoPointAxisIsRefusedBeforeTheSubspaceIsChosen) {
    const WannierInputs record = chain_record(2);
    const WannierConstructionInputs inputs(record);
    try {
        const std::unique_ptr<GaugeChoice> choice =
            create_gauge_choice(policy(LocalizationPolicy::Preferred), inputs);
        FAIL() << "the centres are Berry phases, undefined on a two-point axis";
    } catch (const WannierDiagnosticError& error) {
        EXPECT_EQ(error.diagnostic(), WannierDiagnostic::two_point_axis);
        const std::string message = error.what();
        EXPECT_NE(message.find("reaches the same k-point as its opposite"), std::string::npos)
            << message;
    }
}
