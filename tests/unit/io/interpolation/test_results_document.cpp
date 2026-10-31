#include <gtest/gtest.h>

#include <map>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

#include "generated/q3/interpolation/results.hpp"
#include "io/interpolation/results_document.hpp"
#include "layer_0_core/types/constants.hpp"
#include "layer_0_core/types/init.hpp"
#include "layer_4_computation/interpolator/method/wannier/report/init.hpp"
#include "layer_4_computation/interpolator/method/wannier/report/quality_report.hpp"
#include "layer_4_computation/interpolator/target/quantum_state/init.hpp"
#include "layer_5_properties/band_structure/interpolated/init.hpp"
#include "layer_5_properties/density_of_states/interpolated/init.hpp"

using namespace q3;

namespace {

/// An interpolator that says what it is told to, so the model block can be read back.
class ScriptedInterpolator : public QuantumStateInterpolator {
  public:
    explicit ScriptedInterpolator(std::map<std::string, Real> diagnostics)
        : diagnostics_(std::move(diagnostics)) {}

    [[nodiscard]] int number_of_bands() const override {
        return 4;
    }
    [[nodiscard]] std::vector<Real> eigenvalues_at(const Vector3D& /*kpoint*/) const override {
        return {};
    }
    [[nodiscard]] std::string description() const override {
        return "wannier (projection gauge): 4 orbitals on 27 lattice vectors";
    }
    [[nodiscard]] std::map<std::string, Real> diagnostics() const override {
        return diagnostics_;
    }

  private:
    std::map<std::string, Real> diagnostics_;
};

InterpolationResultsDocument empty_document() {
    return InterpolationResultsDocument(nlohmann::json{{"generator", "q3"}}, 0.5);
}

/// The document given this construction and this quality report.
nlohmann::json construction_document(const WannierConstructionReport& construction,
                                     const WannierQualityReport& quality = {}) {
    InterpolationResultsDocument document = empty_document();
    document.add_construction(construction, quality);
    return document.json();
}

} // namespace

TEST(InterpolationResultsDocument, ItStartsFromWhatIdentifiesTheRun) {
    const nlohmann::json json = empty_document().json();
    EXPECT_EQ(json.at("calculation"), "interpolation");
    EXPECT_EQ(json.at("provenance"), nlohmann::json({{"generator", "q3"}}));
    EXPECT_EQ(json.at("properties").at("fermi_energy"),
              nlohmann::json({{"name", "fermi_energy"}, {"units", "Ry"}, {"value", 0.5}}));
    EXPECT_FALSE(json.at("properties").contains("eigenvalues_at_kpoints"))
        << "nothing was evaluated yet";
    EXPECT_FALSE(json.at("properties").contains("density_of_states"));
}

/**
 * The Wigner-Seitz weight identity has a key of its own, and whatever else an
 * interpolator reports is kept under `diagnostics` rather than dropped.
 */
TEST(InterpolationResultsDocument, TheModelBlockGivesTheWeightIdentityItsOwnKey) {
    InterpolationResultsDocument document = empty_document();
    document.describe_model(
        "wannier",
        "./output/si.save/interpolation.q3ip",
        ScriptedInterpolator({{"total_bloch_sum_weight", 27.0}, {"something_else", 1.5}}));

    const nlohmann::json model = document.json().at("construction");
    EXPECT_EQ(model.at("method"), "wannier");
    EXPECT_EQ(model.at("record"), "./output/si.save/interpolation.q3ip");
    EXPECT_EQ(model.at("number_of_bands"), 4);
    EXPECT_EQ(model.at("description"),
              "wannier (projection gauge): 4 orbitals on 27 lattice vectors");
    EXPECT_DOUBLE_EQ(model.at("total_bloch_sum_weight").get<double>(), 27.0);
    EXPECT_EQ(model.at("diagnostics"), nlohmann::json({{"something_else", 1.5}}));

    InterpolationResultsDocument plain = empty_document();
    plain.describe_model("wannier", "record", ScriptedInterpolator({}));
    EXPECT_FALSE(plain.json().at("construction").contains("diagnostics"));
}

/// A section a strategy did not run is absent from the file, not written as zeros.
TEST(InterpolationResultsDocument, AnUnevaluatedSpreadWritesOnlyWhatItMeasured) {
    WannierConstructionReport construction;
    construction.spread.gauge_invariant_bohr_squared = 12.4358;
    const nlohmann::json json = construction_document(construction).at("construction").at("spread");
    EXPECT_EQ(json.size(), 2u) << json.dump();
    EXPECT_DOUBLE_EQ(json.at("gauge_invariant_bohr_squared").get<double>(), 12.4358);
    EXPECT_FALSE(json.at("localized").get<bool>());
}

TEST(InterpolationResultsDocument, ALocalizedSpreadCarriesEveryKeyTheReaderExpects) {
    WannierConstructionReport construction;
    SpreadReport& report = construction.spread;
    report.evaluated = true;
    report.gauge_invariant_bohr_squared = 1.0;
    report.total_bohr_squared = 2.0;
    report.centers_bohr = {Vector3D(0.1, 0.2, 0.3)};
    report.spreads_bohr_squared = {2.0};
    report.guided_by_trial_orbitals = true;
    report.trial_orbital_centers_bohr = {Vector3D(0.0, 0.0, 0.0)};
    report.initial_minimizable_bohr_squared = 0.5;
    report.localized = true;
    report.localization_iterations = 61;
    report.localization_converged = true;
    report.localization_gradient_norm = 1.0e-5;

    const nlohmann::json json = construction_document(construction).at("construction").at("spread");
    EXPECT_EQ(json.size(), 11u) << json.dump();
    EXPECT_EQ(json.at("guiding_centers"), "trial orbitals");
    EXPECT_EQ(json.at("centers_bohr"), nlohmann::json({{0.1, 0.2, 0.3}}));
    EXPECT_EQ(json.at("trial_orbital_centers_bohr"), nlohmann::json({{0.0, 0.0, 0.0}}));
    EXPECT_EQ(json.at("spreads_bohr_squared"), nlohmann::json({2.0}));
    EXPECT_DOUBLE_EQ(json.at("total_bohr_squared").get<double>(), 2.0);
    EXPECT_DOUBLE_EQ(json.at("initial_minimizable_bohr_squared").get<double>(), 0.5);
    EXPECT_EQ(json.at("localization_iterations"), 61);
    EXPECT_TRUE(json.at("localization_converged").get<bool>());
    EXPECT_DOUBLE_EQ(json.at("localization_gradient_norm").get<double>(), 1.0e-5);
}

TEST(InterpolationResultsDocument, TheDisentanglementBlockIsOneBasedAndInElectronvolts) {
    WannierConstructionReport construction;
    EXPECT_FALSE(construction_document(construction).at("construction").contains("disentanglement"))
        << "an isolated manifold has no classifier's verdict to report";

    DisentanglementReport& report = construction.disentanglement;
    report.performed = true;
    report.classification.metallic = true;
    report.classification.number_of_bands_crossing_fermi = 1;
    report.classification.frozen_window_above_fermi_rydberg = 2.0 / constants::RY_TO_EV;
    report.classification.frozen_window_automatic = true;
    report.classification.bands_excluded_from_freeze = {{4, 3}};

    const nlohmann::json json =
        construction_document(construction).at("construction").at("disentanglement");
    EXPECT_TRUE(json.at("metallic").get<bool>());
    EXPECT_EQ(json.at("bands_crossing_fermi"), 1);
    EXPECT_NEAR(json.at("frozen_window_above_fermi_ev").get<double>(), 2.0, 1.0e-12);
    EXPECT_TRUE(json.at("frozen_window_automatic").get<bool>());
    EXPECT_EQ(json.at("bands_excluded_from_freeze"),
              nlohmann::json({{{"band", 5}, {"kpoints", 3}}}));
}

TEST(InterpolationResultsDocument, TheQualityBlockWritesOnlyTheSectionsTheReportFilled) {
    WannierQualityReport report;
    report.decay.outermost_shell_distance = 10.0;
    report.decay.outermost_shell_size = 12;
    report.decay.outermost_shell_maximum = 0.001;
    report.decay.largest_hopping = 0.1;
    report.decay.ratio = 0.01;
    WannierConstructionReport construction;

    const nlohmann::json bare = construction_document(construction, report).at("quality");
    EXPECT_EQ(bare.size(), 1u) << bare.dump();
    EXPECT_EQ(bare.at("hamiltonian_decay").at("outermost_shell_size"), 12);
    EXPECT_DOUBLE_EQ(bare.at("hamiltonian_decay").at("ratio").get<double>(), 0.01);

    report.spreads_bohr_squared = {1.0, 2.0};
    report.spread_median_bohr_squared = 1.5;
    report.spread_variance_bohr_squared = 0.25;
    report.spread_maximum_bohr_squared = 2.0;
    report.spread_outliers = {1};
    report.nearest_atom_distance_bohr = {0.1, 0.2};
    report.nearest_bond_midpoint_distance_bohr = {2.0, 2.1};
    report.shortest_bond_bohr = 4.4;
    construction.spread.localized = true;
    construction.spread.localization_converged = false;

    const nlohmann::json full = construction_document(construction, report).at("quality");
    EXPECT_EQ(full.size(), 4u) << full.dump();
    EXPECT_EQ(full.at("spread").at("outliers"), nlohmann::json({1}));
    EXPECT_EQ(full.at("spread").at("above_ten_angstrom_squared"), nlohmann::json::array());
    EXPECT_DOUBLE_EQ(full.at("spread").at("variance_bohr_squared").get<double>(), 0.25);
    EXPECT_DOUBLE_EQ(full.at("centers").at("shortest_bond_bohr").get<double>(), 4.4);
    EXPECT_FALSE(full.at("localization_converged").get<bool>());
}

TEST(InterpolationResultsDocument, TheBandsAreListedAsTheResultsFileListsThem) {
    InterpolatedBandStructure bands;
    bands.eigenvalues_rydberg = {{-0.5, 0.25}, {-0.4, 0.3}};
    InterpolationResultsDocument document = empty_document();
    document.add_bands({Vector3D(0.0, 0.5, 0.5), Vector3D(0.25, 0.0, 0.0)}, bands);

    const nlohmann::json json = document.json();
    const nlohmann::json expected = nlohmann::json::parse(R"({
        "name": "eigenvalues_at_kpoints",
        "units": "Ry",
        "values": [
            {"kpoint": [0.0, 0.5, 0.5], "eigenvalues": [{"energies": [-0.5, 0.25]}]},
            {"kpoint": [0.25, 0.0, 0.0], "eigenvalues": [{"energies": [-0.4, 0.3]}]}
        ]
    })");
    EXPECT_EQ(json.at("properties").at("eigenvalues_at_kpoints"), expected);
}

TEST(InterpolationResultsDocument, TheDensityOfStatesIsTheEsseRecord) {
    InterpolatedDensityOfStates density;
    density.energies_rydberg = {-1.0, 0.0, 1.0};
    density.density_of_states = {0.1, 2.0, 0.3};
    InterpolationResultsDocument document = empty_document();
    document.add_density_of_states(density);

    const nlohmann::json json = document.json().at("properties").at("density_of_states");
    EXPECT_FALSE(json.contains("mesh")) << "the mesh is the manifest's, not the record's";
    EXPECT_EQ(json.at("name"), "density_of_states");
    EXPECT_EQ(json.at("xAxis"), nlohmann::json({{"label", "energy"}, {"units", "Ry"}}));
    EXPECT_EQ(json.at("yAxis"),
              nlohmann::json({{"label", "density of states"}, {"units", "states/unitcell"}}));
    EXPECT_EQ(json.at("xDataArray"), nlohmann::json({-1.0, 0.0, 1.0}));
    EXPECT_EQ(json.at("yDataSeries"), nlohmann::json({{0.1, 2.0, 0.3}}));
    EXPECT_EQ(json.at("legend"), nlohmann::json::array());
}

/**
 * One schema describes the file before and after q3.workflows.wannier
 * annotates it: a file carrying the workflow's verdict, its ladder with the
 * nulls of a raised attempt, and its band distance reads back into the
 * generated struct and serialises to the same document.
 */
TEST(InterpolationResults, AFileTheWorkflowAnnotatedReadsBackAndRoundTrips) {
    const nlohmann::json document = nlohmann::json::parse(R"({
        "calculation": "interpolation",
        "provenance": {"generator": {"engine_name": "q3"}},
        "properties": {"fermi_energy": {"name": "fermi_energy", "units": "Ry", "value": 0.5}},
        "construction": {
            "method": "wannier",
            "description": "wannier (projection gauge): 4 orbitals on 27 lattice vectors",
            "record": "./output/si.save/interpolation.q3ip",
            "number_of_bands": 4,
            "total_bloch_sum_weight": 27.0,
            "spread": {"gauge_invariant_bohr_squared": 0.0, "localized": false}
        },
        "quality": {
            "hamiltonian_decay": {
                "outermost_shell_distance_bohr": 10.0,
                "outermost_shell_size": 12,
                "outermost_shell_maximum_ry": 0.001,
                "largest_hopping_ry": 0.1,
                "ratio": 0.01
            },
            "status": "accept",
            "reason": "eta 0.9 meV below the gate",
            "ladder": [{
                "rung": "default",
                "grid": [4, 4, 4],
                "settings": {"localization": "maximal"},
                "status": "error:unconverged",
                "reason": "Error: disentanglement did not converge in 4000 iterations.",
                "eta_ev": null,
                "eta_maximum_ev": null,
                "localized": null,
                "localization_converged": null
            }],
            "band_distance": {
                "eta_ev": 0.0009,
                "eta_maximum_ev": 0.004,
                "number_of_states": 200,
                "reference": "path",
                "number_of_kpoints": 50
            }
        }
    })");

    const generated::InterpolationResults results = document.get<generated::InterpolationResults>();
    const generated::InterpolationResultsQuality& quality = results.quality;
    ASSERT_TRUE(quality.status.has_value());
    EXPECT_EQ(*quality.status, "accept");
    ASSERT_TRUE(quality.ladder.has_value());
    ASSERT_EQ(quality.ladder->size(), 1u);
    EXPECT_EQ(quality.ladder->front().rung, "default");
    EXPECT_TRUE(quality.ladder->front().eta_ev.is_null());
    EXPECT_EQ(quality.ladder->front().settings.at("localization"), "maximal");
    ASSERT_TRUE(quality.band_distance.has_value());
    EXPECT_EQ(quality.band_distance->number_of_kpoints, 50);
    EXPECT_FALSE(results.construction.disentanglement.has_value());
    EXPECT_FALSE(results.properties.eigenvalues_at_kpoints.has_value());
    EXPECT_EQ(results.provenance.at("generator").at("engine_name"), "q3");

    const nlohmann::json again = results;
    EXPECT_EQ(again, document);
}
