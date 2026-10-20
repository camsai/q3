#include <gtest/gtest.h>

#include <stdexcept>
#include <string>
#include <vector>

#include "layer_0_core/types/init.hpp"
#include "layer_4_computation/interpolator/method/wannier/diagnostic.hpp"
#include "layer_4_computation/interpolator/method/wannier/diagnostic_error.hpp"
#include "layer_4_computation/interpolator/method/wannier/inputs.hpp"
#include "layer_4_computation/interpolator/method/wannier/internal/band_classification.hpp"
#include "layer_4_computation/interpolator/method/wannier/internal/band_classifier.hpp"

using namespace q3;

namespace {

/// Two k-points, four bands: a valence pair, then two states far above.
struct Fixture {
    std::vector<std::vector<Real>> projectabilities = {{0.99, 0.98, 0.004, 0.002},
                                                       {0.99, 0.97, 0.003, 0.001}};
    std::vector<std::vector<Real>> eigenvalues = {{-1.0, -0.8, 0.5, 0.9}, {-1.1, -0.7, 0.6, 1.0}};
    Real fermi = -0.5;
};

/**
 * The record a classifier reads: these projectabilities and eigenvalues, this
 * Fermi level, and as many trial orbitals as Wannier functions are wanted.
 */
WannierInputs record_of(const std::vector<std::vector<Real>>& projectabilities,
                        const std::vector<std::vector<Real>>& eigenvalues_rydberg,
                        Real fermi_energy_rydberg,
                        int number_of_wannier_functions) {
    WannierInputs record;
    record.projectabilities = projectabilities;
    record.eigenvalues_rydberg = eigenvalues_rydberg;
    record.fermi_energy_rydberg = fermi_energy_rydberg;
    record.number_of_trial_orbitals = number_of_wannier_functions;
    return record;
}

} // namespace

TEST(BandClassifier, HighProjectabilityFreezesAndLowProjectabilityDiscards) {
    Fixture fixture;
    const BandClassification classification =
        BandClassifier(BandClassificationThresholds{})
            .classify(record_of(fixture.projectabilities, fixture.eigenvalues, fixture.fermi, 2));

    ASSERT_EQ(classification.kpoints.size(), 2u);
    for (const KPointBandClassification& entry : classification.kpoints) {
        EXPECT_EQ(entry.roles[0], BandRole::Frozen);
        EXPECT_EQ(entry.roles[1], BandRole::Frozen);
        EXPECT_EQ(entry.roles[2], BandRole::Discarded);
        EXPECT_EQ(entry.roles[3], BandRole::Discarded);
        EXPECT_EQ(entry.number_kept(), 2);
        EXPECT_EQ(entry.number_frozen(), 2);
        EXPECT_EQ(entry.kept_bands, (std::vector<int>{0, 1}));
    }
}

TEST(BandClassifier, TheEnergyRuleFreezesAStateProjectabilityWouldOnlyOffer) {
    /**
     * This is the metal case and the reason the energy rule exists. A state at
     * the Fermi level with modest projectability -- a metal's sp band, which
     * the projectability ceiling would leave to the optimiser -- is frozen
     * anyway, or the model ends up with a hole at the Fermi surface that the
     * mesh-point round trip cannot see.
     */
    std::vector<std::vector<Real>> projectabilities = {{0.99, 0.30}};
    std::vector<std::vector<Real>> eigenvalues = {{-1.0, -0.6}};
    BandClassificationThresholds thresholds;
    thresholds.frozen_window_above_fermi_rydberg = 0.0;
    const BandClassifier classifier(thresholds);

    const BandClassification without_energy_rule =
        classifier.classify(record_of(projectabilities, eigenvalues, -2.0, 1));
    EXPECT_EQ(without_energy_rule.kpoints[0].roles[1], BandRole::Candidate)
        << "far above the Fermi level, the state is only offered to the optimisation";

    const BandClassification with_energy_rule =
        classifier.classify(record_of(projectabilities, eigenvalues, -0.5, 2));
    EXPECT_EQ(with_energy_rule.kpoints[0].roles[1], BandRole::Frozen)
        << "at or below the Fermi level it is kept exactly";
}

TEST(BandClassifier, AStateTheTrialOrbitalsCannotReachIsExcludedFromTheFreezeAndReported) {
    /**
     * The singular-gauge rule. A state inside the window with a projectability
     * below the minimum is a direction the projection has no column for, and
     * freezing it made the gauge singular on copper every time. It is left
     * out of the freeze -- discarded, since it is also unreachable -- and the
     * summary says which band and at how many k-points.
     */
    std::vector<std::vector<Real>> projectabilities = {
        {0.99, 0.001, 0.5}, {0.99, 0.002, 0.5}, {0.99, 0.5, 0.5}};
    std::vector<std::vector<Real>> eigenvalues = {
        {-1.0, -0.6, 0.4}, {-1.0, -0.7, 0.4}, {-1.0, -0.65, 0.4}};
    BandClassificationThresholds thresholds;
    thresholds.frozen_window_above_fermi_rydberg = 0.0;

    const BandClassification classification =
        BandClassifier(thresholds).classify(record_of(projectabilities, eigenvalues, -0.5, 2));
    const std::vector<KPointBandClassification>& kpoints = classification.kpoints;
    const BandClassificationSummary& summary = classification.summary;
    EXPECT_EQ(kpoints[0].roles[1], BandRole::Discarded);
    EXPECT_EQ(kpoints[1].roles[1], BandRole::Discarded);
    EXPECT_EQ(kpoints[2].roles[1], BandRole::Frozen) << "reachable there, so frozen there";
    ASSERT_EQ(summary.bands_excluded_from_freeze.size(), 1u);
    EXPECT_EQ(summary.bands_excluded_from_freeze[0].band_index, 1);
    EXPECT_EQ(summary.bands_excluded_from_freeze[0].number_of_kpoints, 2);
    EXPECT_EQ(kpoints[0].number_kept(), 2)
        << "the frozen band and the candidate above; the excluded state is simply absent";
    EXPECT_EQ(kpoints[2].number_kept(), 3);
}

TEST(BandClassifier, TheAutomaticWindowIsThePublishedTwoElectronvoltsAndTheMetalIsReported) {
    /**
     * A band below the Fermi level at one k-point and above it at another is
     * a band crossing the Fermi surface, and one such band makes a metal. The
     * window is the published E_F + 2 eV = 0.147 Ry either way -- it freezes
     * the state at -0.40 Ry here, 0.1 Ry above E_F, and not the one at 0.3 --
     * because it is the window the band distance is measured on; what differs
     * between the two systems is only the verdict the report carries.
     */
    BandClassificationThresholds thresholds;
    thresholds.automatic_frozen_window = true;
    thresholds.projectability_maximum = 1.5; // let the energy rule decide alone

    std::vector<std::vector<Real>> metal_projectabilities = {{0.9, 0.5, 0.5}, {0.9, 0.5, 0.5}};
    std::vector<std::vector<Real>> metal_eigenvalues = {{-1.0, -0.6, -0.40}, {-1.0, -0.4, 0.3}};
    const BandClassification metal =
        BandClassifier(thresholds)
            .classify(record_of(metal_projectabilities, metal_eigenvalues, -0.5, 3));
    EXPECT_TRUE(metal.summary.metallic);
    EXPECT_EQ(metal.summary.number_of_bands_crossing_fermi, 1);
    EXPECT_TRUE(metal.summary.frozen_window_automatic);
    EXPECT_NEAR(metal.summary.frozen_window_above_fermi_rydberg, 2.0 / constants::RY_TO_EV, 1.0e-9);
    EXPECT_EQ(metal.kpoints[0].roles[2], BandRole::Frozen) << "0.1 Ry above E_F, inside";
    EXPECT_EQ(metal.kpoints[1].roles[2], BandRole::Candidate) << "0.8 Ry above, outside";

    std::vector<std::vector<Real>> insulator_eigenvalues = {{-1.0, -0.6, 0.2}, {-1.1, -0.5, 0.3}};
    const BandClassification insulator =
        BandClassifier(thresholds)
            .classify(record_of(metal_projectabilities, insulator_eigenvalues, -0.5, 2));
    EXPECT_FALSE(insulator.summary.metallic)
        << "the highest occupied band reaches E_F exactly, which is not a crossing";
    EXPECT_NEAR(
        insulator.summary.frozen_window_above_fermi_rydberg, 2.0 / constants::RY_TO_EV, 1.0e-9)
        << "the same window; the gap keeps it from reaching the state at 0.2 Ry";
    EXPECT_EQ(insulator.kpoints[0].roles[1], BandRole::Frozen);
    EXPECT_EQ(insulator.kpoints[0].roles[2], BandRole::Candidate);

    thresholds.automatic_frozen_window = false;
    thresholds.frozen_window_above_fermi_rydberg = 1.0;
    const BandClassification overridden =
        BandClassifier(thresholds)
            .classify(record_of(metal_projectabilities, metal_eigenvalues, -0.5, 3));
    EXPECT_TRUE(overridden.summary.metallic) << "the detection is reported either way";
    EXPECT_FALSE(overridden.summary.frozen_window_automatic);
    EXPECT_EQ(overridden.summary.frozen_window_above_fermi_rydberg, 1.0);
    EXPECT_EQ(overridden.kpoints[1].roles[2], BandRole::Frozen)
        << "an explicit value is applied as given";
}

TEST(BandClassifier, AWindowBelowTheManifoldFreezesNothingByEnergy) {
    /**
     * How a test exercises the optimisation on its own: push the energy
     * window under everything and raise the projectability ceiling above one,
     * and every band becomes a candidate.
     */
    Fixture fixture;
    BandClassificationThresholds thresholds;
    thresholds.frozen_window_above_fermi_rydberg = -100.0;
    thresholds.projectability_maximum = 1.5;
    thresholds.projectability_minimum = 0.0;

    const BandClassification classification =
        BandClassifier(thresholds)
            .classify(record_of(fixture.projectabilities, fixture.eigenvalues, fixture.fermi, 2));
    for (const KPointBandClassification& entry : classification.kpoints) {
        EXPECT_EQ(entry.number_frozen(), 0);
        EXPECT_EQ(entry.number_kept(), 4);
        for (const BandRole role : entry.roles) {
            EXPECT_EQ(role, BandRole::Candidate);
        }
    }
}

TEST(BandClassifier, TooManyFrozenBandsIsRefusedWithTheKpointNamed) {
    Fixture fixture;
    try {
        BandClassifier(BandClassificationThresholds{})
            .classify(record_of(fixture.projectabilities, fixture.eigenvalues, fixture.fermi, 1));
        FAIL() << "two frozen bands for one Wannier function must be refused";
    } catch (const WannierDiagnosticError& error) {
        const std::string message = error.what();
        EXPECT_NE(message.find("k-point 0"), std::string::npos) << message;
        EXPECT_NE(message.find("freezes 2"), std::string::npos) << message;
        EXPECT_EQ(error.diagnostic(), WannierDiagnostic::frozen_window_overfull);
    }
}

TEST(BandClassifier, TooFewKeptBandsIsRefusedWithTheKpointNamed) {
    Fixture fixture;
    try {
        BandClassifier(BandClassificationThresholds{})
            .classify(record_of(fixture.projectabilities, fixture.eigenvalues, fixture.fermi, 3));
        FAIL() << "two kept bands cannot fill three Wannier functions";
    } catch (const WannierDiagnosticError& error) {
        const std::string message = error.what();
        EXPECT_NE(message.find("keeps only 2"), std::string::npos) << message;
        EXPECT_EQ(error.diagnostic(), WannierDiagnostic::kept_bands_underfull);
    }
}

TEST(BandClassifier, InvertedProjectabilityThresholdsAreRefused) {
    BandClassificationThresholds thresholds;
    thresholds.projectability_minimum = 0.99;
    thresholds.projectability_maximum = 0.01;
    EXPECT_THROW(BandClassifier{thresholds}, std::invalid_argument);
}
