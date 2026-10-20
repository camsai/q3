#include "io/provenance/init.hpp"

#include <gtest/gtest.h>
#include <string>

#include <nlohmann/json.hpp>

using namespace q3;

namespace {

/// A minimal input, enough for every field of the generic record to have a source.
InputData make_input() {
    InputData input;
    input.exchange_correlation_functional = "PBE";
    input.number_of_spin_components = 1;
    input.kpoint_grid_dimension_1 = 2;
    input.kpoint_grid_dimension_2 = 2;
    input.kpoint_grid_dimension_3 = 2;
    input.kpoint_grid_shift_1 = 1;
    input.kpoint_grid_shift_2 = 1;
    input.kpoint_grid_shift_3 = 1;
    input.convergence_threshold = 1.0e-8;
    input.maximum_iterations = 100;
    return input;
}

/**
 * A stand-in for a representation Q3 does not have: it reports itself through
 * the same hook the plane-wave record uses, and needs nothing from this header
 * to do so. That is the property the hook exists for, so it is what is tested
 * -- an assertion about PlanewaveProvenanceRecord here would only show that the
 * one subclass that happens to exist today works.
 */
class RealSpaceGridProvenanceRecord : public QuantumSystemProvenanceRecord {
  public:
    using QuantumSystemProvenanceRecord::QuantumSystemProvenanceRecord;

  protected:
    void add_representation_method_details(nlohmann::json& method) const override {
        method["real_space_grid_spacing_bohr"] = 0.25;
    }
};

} // namespace

/**
 * The generic record describes a quantum system that was solved, and nothing
 * about what solved it: no basis set, no datasets. This is the guard on the
 * separation -- a cutoff or a pseudopotential entry appearing here again is the
 * regression, not a new field.
 */
TEST(QuantumSystemProvenanceRecord, DescribesTheSystemWithoutNamingARepresentation) {
    const InputData input = make_input();
    const nlohmann::json provenance = QuantumSystemProvenanceRecord(input).to_json();

    EXPECT_EQ(provenance["generator"]["engine_name"], "q3");
    EXPECT_EQ(provenance["generator"]["engine_version"], std::string(version));
    EXPECT_EQ(provenance["model"]["exchange_correlation_functional"], "PBE");
    EXPECT_EQ(provenance["model"]["spin_treatment"], "unpolarized");
    EXPECT_EQ(provenance["method"]["kpoint_grid"], nlohmann::json({2, 2, 2}));
    EXPECT_EQ(provenance["method"]["electronic_occupation"], "fixed");
    EXPECT_DOUBLE_EQ(provenance["precision"]["energy_convergence_threshold_rydberg"].get<double>(),
                     1.0e-8);

    const nlohmann::json& method = provenance["method"];
    EXPECT_FALSE(method.contains("wavefunction_cutoff_rydberg"));
    EXPECT_FALSE(method.contains("density_cutoff_rydberg"));
    EXPECT_FALSE(method.contains("pseudopotentials"));
}

/// A representation adds its own identity to "method" without the base knowing it exists.
TEST(QuantumSystemProvenanceRecord, LetsARepresentationContributeItsOwnMethodDetails) {
    const InputData input = make_input();
    const nlohmann::json method = RealSpaceGridProvenanceRecord(input).to_json()["method"];

    EXPECT_DOUBLE_EQ(method["real_space_grid_spacing_bohr"].get<double>(), 0.25);
    EXPECT_EQ(method["kpoint_grid"], nlohmann::json({2, 2, 2}));
    EXPECT_FALSE(method.contains("pseudopotentials"));
}

/// Spin components map onto the consumer's three spin treatments.
TEST(QuantumSystemProvenanceRecord, MapsSpinComponentsToTheSpinTreatmentNames) {
    EXPECT_EQ(spin_treatment_name(1), "unpolarized");
    EXPECT_EQ(spin_treatment_name(2), "collinear");
    EXPECT_EQ(spin_treatment_name(4), "non_collinear");
}
