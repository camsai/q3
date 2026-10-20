#include <gtest/gtest.h>

#include <cstdio>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>

#include "io/mean_field_export/factory.hpp"
#include "io/mean_field_record/from_scf_state.hpp"
#include "layer_1_structure/crystal/crystal.hpp"
#include "layer_1_structure/reciprocal_space/kpoints_grid.hpp"
#include "layer_3_representation/basis/planewave/init.hpp"
#include "layer_3_representation/quantum_system/kohn_sham/planewave/norm_conserving/init.hpp"
#include "layer_4_computation/iterator/kohn_sham/planewave/execution/serial.hpp"
#include "layer_4_computation/state/kohn_sham/planewave/norm_conserving/init.hpp"
#include "utils/temporary_path.hpp"

using namespace q3;

namespace {

execution::PlanewaveSerialExecutionPolicy serial_policy;

/**
 * Smallest possible valid SCF state -- enough for the Null exporter's no-op
 * contract and for confirming BerkeleyGWExporter does not crash on an empty
 * directory argument (its own byte-layout correctness is tested separately
 * under tests/unit/io/mean_field_export/berkeleygw/).
 */
void populate_minimal_state(
    Crystal& crystal_structure,
    PlanewaveBasis& basis,
    KPointsGrid& kpoint_grid,
    std::shared_ptr<NormConservingPlanewaveKohnShamQuantumSystem>& quantum_system,
    NormConservingPlanewaveKohnShamSCFState& scf_state) {
    std::vector<AtomSpecies> species = {AtomSpecies{"Si", 28.0855, 4.0, "Si.upf"}};
    std::vector<Atom> atoms = {Atom{0, {0.0, 0.0, 0.0}}};
    crystal_structure = Crystal::from_input_data(1, 10.0, species, atoms);

    basis.generate(serial_policy, crystal_structure, 8.0, 32.0);
    kpoint_grid.generate(1, 1, 1, 0, 0, 0);
    basis.initialize_kpoint_data(crystal_structure, kpoint_grid);

    quantum_system = std::make_shared<NormConservingPlanewaveKohnShamQuantumSystem>();
    quantum_system->set_kpoint_grid(&kpoint_grid);
    quantum_system->set_basis(&basis);
    quantum_system->set_number_of_bands(2);
    quantum_system->set_number_of_electrons(4.0);
    quantum_system->allocate_electronic_states(basis.kpoint_number_of_plane_waves());

    scf_state.crystal_structure = &crystal_structure;
    scf_state.kpoint_grid = &kpoint_grid;
    scf_state.planewave_basis = &basis;
    scf_state.quantum_system_snapshot_ = quantum_system;
}

} // namespace

// ── Name resolution ──────────────────────────────────────────────────────

TEST(MeanFieldExportFactory, EmptyStringResolvesToNull) {
    auto exporter = create_mean_field_exporter("");
    EXPECT_EQ(exporter->name(), "none");
}

TEST(MeanFieldExportFactory, NoneResolvesToNull) {
    auto exporter = create_mean_field_exporter("none");
    EXPECT_EQ(exporter->name(), "none");
}

TEST(MeanFieldExportFactory, BerkeleygwResolvesToBerkeleyGWExporter) {
    auto exporter = create_mean_field_exporter("berkeleygw");
    EXPECT_EQ(exporter->name(), "berkeleygw");
}

TEST(MeanFieldExportFactory, UnknownNameThrowsAndListsSupportedValues) {
    try {
        auto exporter = create_mean_field_exporter("yambo");
        FAIL() << "expected create_mean_field_exporter to throw for an unknown name";
    } catch (const std::runtime_error& error) {
        std::string message = error.what();
        EXPECT_NE(message.find("yambo"), std::string::npos);
        EXPECT_NE(message.find("none"), std::string::npos);
        EXPECT_NE(message.find("berkeleygw"), std::string::npos);
    }
}

// ── Null exporter contract ───────────────────────────────────────────────

TEST(NullMeanFieldExporter, ExportToDirectoryCreatesNoFiles) {
    Crystal crystal_structure;
    PlanewaveBasis basis;
    KPointsGrid kpoint_grid;
    std::shared_ptr<NormConservingPlanewaveKohnShamQuantumSystem> quantum_system;
    NormConservingPlanewaveKohnShamSCFState scf_state;
    populate_minimal_state(crystal_structure, basis, kpoint_grid, quantum_system, scf_state);

    std::string directory = test_support::make_temporary_path();
    auto exporter = create_mean_field_exporter("none");
    exporter->export_to_directory(build_mean_field_record(scf_state), directory);

    // Not even the directory itself should be created by a no-op export.
    EXPECT_FALSE(std::filesystem::exists(directory));
}

TEST(NullMeanFieldExporter, NameIsNone) {
    NullMeanFieldExporter exporter;
    EXPECT_EQ(exporter.name(), "none");
}
