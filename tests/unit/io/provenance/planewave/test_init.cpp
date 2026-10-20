#include "io/provenance/planewave/init.hpp"

#include <gtest/gtest.h>

#include <nlohmann/json.hpp>

using namespace q3;

/**
 * The basis set is the plane-wave record's contribution and nothing else's: the
 * cutoffs appear, in Rydberg and named so, and the datasets a pseudopotential
 * calculation would also carry do not -- those belong to the subclass below it.
 */
TEST(PlanewaveProvenanceRecord, ReportsTheBasisSetCutoffsAndNoDatasets) {
    InputData input;
    input.wavefunction_energy_cutoff = 48.0;
    input.charge_density_energy_cutoff = 192.0;

    const nlohmann::json method = PlanewaveProvenanceRecord(input).to_json()["method"];

    EXPECT_DOUBLE_EQ(method["wavefunction_cutoff_rydberg"].get<double>(), 48.0);
    EXPECT_DOUBLE_EQ(method["density_cutoff_rydberg"].get<double>(), 192.0);
    EXPECT_FALSE(method.contains("pseudopotentials"));
}
