#include <gtest/gtest.h>

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "io/mean_field_export/berkeleygw/init.hpp"
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

void populate_synthetic_state(
    Crystal& crystal_structure,
    PlanewaveBasis& basis,
    KPointsGrid& kpoint_grid,
    std::shared_ptr<NormConservingPlanewaveKohnShamQuantumSystem>& quantum_system,
    NormConservingPlanewaveKohnShamSCFState& scf_state) {
    std::vector<AtomSpecies> species = {AtomSpecies{"Si", 28.0855, 4.0, "Si.upf"}};
    std::vector<Atom> atoms = {Atom{0, {0.875, 0.875, 0.875}}, Atom{0, {0.125, 0.125, 0.125}}};
    crystal_structure = Crystal::from_input_data(2, 10.0, species, atoms);

    basis.generate(serial_policy, crystal_structure, 8.0, 32.0);
    kpoint_grid.generate(2, 1, 1, 0, 0, 0);
    basis.initialize_kpoint_data(crystal_structure, kpoint_grid);

    quantum_system = std::make_shared<NormConservingPlanewaveKohnShamQuantumSystem>();
    quantum_system->set_kpoint_grid(&kpoint_grid);
    quantum_system->set_basis(&basis);
    quantum_system->set_number_of_bands(4);
    quantum_system->set_number_of_electrons(8.0);
    quantum_system->allocate_electronic_states(basis.kpoint_number_of_plane_waves());

    scf_state.crystal_structure = &crystal_structure;
    scf_state.kpoint_grid = &kpoint_grid;
    scf_state.planewave_basis = &basis;
    scf_state.quantum_system_snapshot_ = quantum_system;

    int total_grid_points = basis.fft_rho().total_grid_points();
    scf_state.charge_density_real.resize(1);
    scf_state.charge_density_real[0].assign(total_grid_points, 1.0);
}

} // namespace

TEST(BerkeleyGWExporter, NameIsBerkeleygw) {
    BerkeleyGWExporter exporter;
    EXPECT_EQ(exporter.name(), "berkeleygw");
}

TEST(BerkeleyGWExporter, ExportToDirectoryWritesWfnAndRhoWhenVxcDataIsIncomplete) {
    Crystal crystal_structure;
    PlanewaveBasis basis;
    KPointsGrid kpoint_grid;
    std::shared_ptr<NormConservingPlanewaveKohnShamQuantumSystem> quantum_system;
    NormConservingPlanewaveKohnShamSCFState scf_state;
    populate_synthetic_state(crystal_structure, basis, kpoint_grid, quantum_system, scf_state);
    /**
     * scf_state.exchange_correlation_potential_g deliberately left empty -- an incomplete SCF
     * state (as opposed to nspin>1, which is out of scope and would also legitimately fail WFN).
     */

    std::string directory = test_support::make_temporary_path();
    ASSERT_FALSE(std::filesystem::exists(directory));

    BerkeleyGWExporter exporter;
    /**
     * The overall export must not throw: vxc.dat's internal throw (missing Vxc(G) data) is
     * caught and logged, so WFN/RHO still get written.
     */
    EXPECT_NO_THROW(exporter.export_to_directory(build_mean_field_record(scf_state), directory));

    EXPECT_TRUE(std::filesystem::exists(directory)) << "export_to_directory must create outdir";
    EXPECT_TRUE(std::filesystem::exists(directory + "/WFN"));
    EXPECT_GT(std::filesystem::file_size(directory + "/WFN"), 0u);
    EXPECT_TRUE(std::filesystem::exists(directory + "/RHO"));
    EXPECT_GT(std::filesystem::file_size(directory + "/RHO"), 0u);
    // No usable G-space Vxc was provided -- vxc.dat must not appear.
    EXPECT_FALSE(std::filesystem::exists(directory + "/vxc.dat"));

    std::filesystem::remove_all(directory);
}

/**
 * A WFN whose header cannot be built is refused after the file is opened. It
 * used to stay behind truncated (104 bytes in camsai/q3#21), where a later
 * BerkeleyGW step could take it for the export's output.
 */
TEST(BerkeleyGWExporter, AFailedWfnLeavesNoFileBehind) {
    Crystal crystal_structure;
    PlanewaveBasis basis;
    KPointsGrid kpoint_grid;
    std::shared_ptr<NormConservingPlanewaveKohnShamQuantumSystem> quantum_system;
    NormConservingPlanewaveKohnShamSCFState scf_state;
    populate_synthetic_state(crystal_structure, basis, kpoint_grid, quantum_system, scf_state);

    MeanFieldRecord record = build_mean_field_record(scf_state);
    record.bravais_lattice_index = 99; // no Bravais lattice has it, so the header refuses

    std::string directory = test_support::make_temporary_path();
    BerkeleyGWExporter exporter;
    EXPECT_THROW(exporter.export_to_directory(record, directory), std::runtime_error);
    EXPECT_FALSE(std::filesystem::exists(directory + "/WFN"));
    EXPECT_FALSE(std::filesystem::exists(directory + "/RHO"));

    std::filesystem::remove_all(directory);
}

TEST(BerkeleyGWExporter, ExportToDirectoryWritesAllThreeFilesWhenVxcDataIsAvailable) {
    Crystal crystal_structure;
    PlanewaveBasis basis;
    KPointsGrid kpoint_grid;
    std::shared_ptr<NormConservingPlanewaveKohnShamQuantumSystem> quantum_system;
    NormConservingPlanewaveKohnShamSCFState scf_state;
    populate_synthetic_state(crystal_structure, basis, kpoint_grid, quantum_system, scf_state);
    /**
     * A complete (if synthetic) SCF state: populate the G-space Vxc the same shape/size as the
     * charge-density G-set, and give every band a normalized wavefunction, matching what a real
     * converged NormConservingPlanewaveKohnShamSCFState carries.
     */
    scf_state.exchange_correlation_potential_g.assign(basis.g_vectors().size(), Complex(0.05, 0.0));
    int number_of_kpoints = kpoint_grid.number_of_kpoints();
    const auto& plane_waves_per_kpoint = basis.kpoint_number_of_plane_waves();
    for (int kpoint_index = 0; kpoint_index < number_of_kpoints; kpoint_index++) {
        int number_of_plane_waves = plane_waves_per_kpoint[kpoint_index];
        for (int band_index = 0; band_index < quantum_system->number_of_bands(); band_index++) {
            Complex* coefficients =
                quantum_system->wavefunction_band_data(kpoint_index, band_index);
            coefficients[0] = Complex(1.0, 0.0);
            for (int plane_wave_index = 1; plane_wave_index < number_of_plane_waves;
                 plane_wave_index++) {
                coefficients[plane_wave_index] = Complex(0.0, 0.0);
            }
        }
    }

    std::string directory = test_support::make_temporary_path();
    ASSERT_FALSE(std::filesystem::exists(directory));

    BerkeleyGWExporter exporter;
    EXPECT_NO_THROW(exporter.export_to_directory(build_mean_field_record(scf_state), directory));

    EXPECT_TRUE(std::filesystem::exists(directory + "/WFN"));
    EXPECT_GT(std::filesystem::file_size(directory + "/WFN"), 0u);
    EXPECT_TRUE(std::filesystem::exists(directory + "/RHO"));
    EXPECT_GT(std::filesystem::file_size(directory + "/RHO"), 0u);
    EXPECT_TRUE(std::filesystem::exists(directory + "/vxc.dat"))
        << "a complete SCF state (spin-unpolarized, non-gamma-only, with a usable G-space Vxc) "
           "must now produce vxc.dat";
    EXPECT_GT(std::filesystem::file_size(directory + "/vxc.dat"), 0u);

    std::filesystem::remove_all(directory);
}
