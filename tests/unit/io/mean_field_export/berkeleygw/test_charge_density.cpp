#include <gtest/gtest.h>

#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "io/checkpoint/qe.hpp"
#include "io/mean_field_export/berkeleygw/charge_density.hpp"
#include "io/mean_field_record/from_scf_state.hpp"
#include "layer_1_structure/crystal/crystal.hpp"
#include "layer_1_structure/reciprocal_space/kpoints_grid.hpp"
#include "layer_1_structure/symmetry/symmetry.hpp"
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
    NormConservingPlanewaveKohnShamSCFState& scf_state,
    int number_of_spin_components = 1) {
    std::vector<AtomSpecies> species = {AtomSpecies{"Si", 28.0855, 4.0, "Si.upf"}};
    std::vector<Atom> atoms = {Atom{0, {0.875, 0.875, 0.875}}, Atom{0, {0.125, 0.125, 0.125}}};
    crystal_structure = Crystal::from_input_data(2, 10.0, species, atoms); // FCC, alat = 10 bohr

    basis.generate(serial_policy, crystal_structure, 8.0, 32.0);
    kpoint_grid.generate(1, 1, 1, 0, 0, 0);
    basis.initialize_kpoint_data(crystal_structure, kpoint_grid);

    quantum_system = std::make_shared<NormConservingPlanewaveKohnShamQuantumSystem>();
    quantum_system->set_kpoint_grid(&kpoint_grid);
    quantum_system->set_basis(&basis);
    quantum_system->set_number_of_bands(4);
    quantum_system->set_number_of_electrons(8.0);
    quantum_system->number_of_spin_components = number_of_spin_components;
    quantum_system->allocate_electronic_states(basis.kpoint_number_of_plane_waves());

    scf_state.crystal_structure = &crystal_structure;
    scf_state.kpoint_grid = &kpoint_grid;
    scf_state.planewave_basis = &basis;
    scf_state.quantum_system_snapshot_ = quantum_system;

    int total_grid_points = basis.fft_rho().total_grid_points();
    scf_state.charge_density_real.resize(number_of_spin_components);
    for (int spin_index = 0; spin_index < number_of_spin_components; spin_index++) {
        scf_state.charge_density_real[spin_index].resize(total_grid_points);
        for (int grid_index = 0; grid_index < total_grid_points; grid_index++) {
            // An arbitrary smooth synthetic density, distinct per spin channel.
            scf_state.charge_density_real[spin_index][grid_index] =
                1.0 + 0.1 * std::sin(0.01 * (grid_index + 1) + spin_index) +
                0.01 * (spin_index + 1);
        }
    }
}

std::string make_temporary_path(const std::string& suffix) {
    return test_support::make_temporary_path(suffix);
}

} // namespace

// ── Body: gvec + rho(G), byte layout and physics invariants ────────────

TEST(BerkeleyGWChargeDensityFile, GvecRecordCountEqualsNumberOfGVectors) {
    Crystal crystal_structure;
    PlanewaveBasis basis;
    KPointsGrid kpoint_grid;
    std::shared_ptr<NormConservingPlanewaveKohnShamQuantumSystem> quantum_system;
    NormConservingPlanewaveKohnShamSCFState scf_state;
    populate_synthetic_state(crystal_structure, basis, kpoint_grid, quantum_system, scf_state);

    std::string path = make_temporary_path(".rho_gvec");
    const MeanFieldRecord record = build_mean_field_record(scf_state);
    BerkeleyGWChargeDensityFile(record).write_to_file(path);

    std::ifstream input_file(path, std::ios::binary);
    QEBinaryRecordFormat record_format;
    std::vector<char> skip(96);
    record_format.read_record(input_file, skip.data(), 96); // title
    std::vector<char> skip2(28);
    record_format.read_record(input_file, skip2.data(), 28); // dimensions (RHO variant)
    std::vector<char> skip3(12);
    record_format.read_record(input_file, skip3.data(), 12); // FFT grid only
    std::vector<char> skip4(20 * sizeof(double));
    record_format.read_record(input_file, skip4.data(), static_cast<int32_t>(20 * sizeof(double)));
    std::vector<char> skip5(20 * sizeof(double));
    record_format.read_record(input_file, skip5.data(), static_cast<int32_t>(20 * sizeof(double)));
    const auto symmetry_operations = SymmetryAnalyzer::get_symmetry_operations(crystal_structure);
    int number_of_operations = static_cast<int>(symmetry_operations.size());
    std::vector<char> skip6(9 * number_of_operations * sizeof(int32_t));
    record_format.read_record(input_file, skip6.data(), static_cast<int32_t>(skip6.size()));
    std::vector<char> skip7(3 * number_of_operations * sizeof(double));
    record_format.read_record(input_file, skip7.data(), static_cast<int32_t>(skip7.size()));
    int number_of_atoms = crystal_structure.number_of_atoms();
    std::vector<char> skip8(number_of_atoms * (3 * sizeof(double) + sizeof(int32_t)));
    record_format.read_record(input_file, skip8.data(), static_cast<int32_t>(skip8.size()));
    ASSERT_FALSE(input_file.fail());

    int32_t nrecord = 0;
    record_format.read_record(input_file, &nrecord, sizeof(nrecord));
    ASSERT_FALSE(input_file.fail());
    EXPECT_EQ(nrecord, 1);

    int32_t number_of_g_vectors = 0;
    record_format.read_record(input_file, &number_of_g_vectors, sizeof(number_of_g_vectors));
    ASSERT_FALSE(input_file.fail());
    EXPECT_EQ(number_of_g_vectors, basis.number_of_g_vectors());

    std::vector<int32_t> gvec(3 * number_of_g_vectors);
    record_format.read_record(
        input_file, gvec.data(), static_cast<int32_t>(gvec.size() * sizeof(int32_t)));
    ASSERT_FALSE(input_file.fail());

    const auto& g_vectors = basis.g_vectors();
    for (int g_index = 0; g_index < number_of_g_vectors; g_index++) {
        EXPECT_EQ(gvec[3 * g_index + 0], g_vectors[g_index].miller.x);
        EXPECT_EQ(gvec[3 * g_index + 1], g_vectors[g_index].miller.y);
        EXPECT_EQ(gvec[3 * g_index + 2], g_vectors[g_index].miller.z);
    }

    std::remove(path.c_str());
}

TEST(BerkeleyGWChargeDensityFile, DensityRecordMatchesVolumeScaledFourierTransform) {
    Crystal crystal_structure;
    PlanewaveBasis basis;
    KPointsGrid kpoint_grid;
    std::shared_ptr<NormConservingPlanewaveKohnShamQuantumSystem> quantum_system;
    NormConservingPlanewaveKohnShamSCFState scf_state;
    populate_synthetic_state(crystal_structure, basis, kpoint_grid, quantum_system, scf_state);

    std::string path = make_temporary_path(".rho_density");
    const MeanFieldRecord record = build_mean_field_record(scf_state);
    BerkeleyGWChargeDensityFile(record).write_to_file(path);

    std::ifstream input_file(path, std::ios::binary);
    QEBinaryRecordFormat record_format;
    std::vector<char> skip(96);
    record_format.read_record(input_file, skip.data(), 96);
    std::vector<char> skip2(28);
    record_format.read_record(input_file, skip2.data(), 28);
    std::vector<char> skip3(12);
    record_format.read_record(input_file, skip3.data(), 12);
    std::vector<char> skip4(20 * sizeof(double));
    record_format.read_record(input_file, skip4.data(), static_cast<int32_t>(20 * sizeof(double)));
    std::vector<char> skip5(20 * sizeof(double));
    record_format.read_record(input_file, skip5.data(), static_cast<int32_t>(20 * sizeof(double)));
    const auto symmetry_operations = SymmetryAnalyzer::get_symmetry_operations(crystal_structure);
    int number_of_operations = static_cast<int>(symmetry_operations.size());
    std::vector<char> skip6(9 * number_of_operations * sizeof(int32_t));
    record_format.read_record(input_file, skip6.data(), static_cast<int32_t>(skip6.size()));
    std::vector<char> skip7(3 * number_of_operations * sizeof(double));
    record_format.read_record(input_file, skip7.data(), static_cast<int32_t>(skip7.size()));
    int number_of_atoms = crystal_structure.number_of_atoms();
    std::vector<char> skip8(number_of_atoms * (3 * sizeof(double) + sizeof(int32_t)));
    record_format.read_record(input_file, skip8.data(), static_cast<int32_t>(skip8.size()));

    int32_t nrecord = 0;
    record_format.read_record(input_file, &nrecord, sizeof(nrecord));
    int32_t number_of_g_vectors = 0;
    record_format.read_record(input_file, &number_of_g_vectors, sizeof(number_of_g_vectors));
    std::vector<char> skip_gvec(3 * number_of_g_vectors * sizeof(int32_t));
    record_format.read_record(input_file, skip_gvec.data(), static_cast<int32_t>(skip_gvec.size()));
    ASSERT_FALSE(input_file.fail());

    int32_t density_nrecord = 0;
    record_format.read_record(input_file, &density_nrecord, sizeof(density_nrecord));
    ASSERT_FALSE(input_file.fail());
    EXPECT_EQ(density_nrecord, 1);
    int32_t density_ng = 0;
    record_format.read_record(input_file, &density_ng, sizeof(density_ng));
    ASSERT_FALSE(input_file.fail());
    EXPECT_EQ(density_ng, number_of_g_vectors);

    std::vector<Complex> rho_g(static_cast<size_t>(number_of_g_vectors));
    record_format.read_record(
        input_file, rho_g.data(), static_cast<int32_t>(rho_g.size() * sizeof(Complex)));
    ASSERT_FALSE(input_file.fail()); // one spin channel: record count == ng exactly

    std::vector<Complex> expected_density_g;
    basis.rho_r_to_g(scf_state.charge_density_real[0], expected_density_g);
    Real cell_volume = crystal_structure.cell_volume();
    for (int g_index = 0; g_index < number_of_g_vectors; g_index++) {
        EXPECT_NEAR(
            rho_g[g_index].real(), (expected_density_g[g_index] * cell_volume).real(), 1e-8);
        EXPECT_NEAR(
            rho_g[g_index].imag(), (expected_density_g[g_index] * cell_volume).imag(), 1e-8);
    }

    std::remove(path.c_str());
}

TEST(BerkeleyGWChargeDensityFile, SpinPolarizedDensityRecordConcatenatesBothChannels) {
    Crystal crystal_structure;
    PlanewaveBasis basis;
    KPointsGrid kpoint_grid;
    std::shared_ptr<NormConservingPlanewaveKohnShamQuantumSystem> quantum_system;
    NormConservingPlanewaveKohnShamSCFState scf_state;
    populate_synthetic_state(crystal_structure,
                             basis,
                             kpoint_grid,
                             quantum_system,
                             scf_state,
                             /*number_of_spin_components=*/2);

    std::string path = make_temporary_path(".rho_spin_polarized");
    // Unlike WFN, RHO fully supports ns=2 -- this must NOT throw.
    const MeanFieldRecord record = build_mean_field_record(scf_state);
    EXPECT_NO_THROW(BerkeleyGWChargeDensityFile(record).write_to_file(path));

    std::ifstream input_file(path, std::ios::binary);
    QEBinaryRecordFormat record_format;
    std::vector<char> skip(96);
    record_format.read_record(input_file, skip.data(), 96);
    std::vector<char> skip2(28);
    record_format.read_record(input_file, skip2.data(), 28);
    std::vector<char> skip3(12);
    record_format.read_record(input_file, skip3.data(), 12);
    std::vector<char> skip4(20 * sizeof(double));
    record_format.read_record(input_file, skip4.data(), static_cast<int32_t>(20 * sizeof(double)));
    std::vector<char> skip5(20 * sizeof(double));
    record_format.read_record(input_file, skip5.data(), static_cast<int32_t>(20 * sizeof(double)));
    const auto symmetry_operations = SymmetryAnalyzer::get_symmetry_operations(crystal_structure);
    int number_of_operations = static_cast<int>(symmetry_operations.size());
    std::vector<char> skip6(9 * number_of_operations * sizeof(int32_t));
    record_format.read_record(input_file, skip6.data(), static_cast<int32_t>(skip6.size()));
    std::vector<char> skip7(3 * number_of_operations * sizeof(double));
    record_format.read_record(input_file, skip7.data(), static_cast<int32_t>(skip7.size()));
    int number_of_atoms = crystal_structure.number_of_atoms();
    std::vector<char> skip8(number_of_atoms * (3 * sizeof(double) + sizeof(int32_t)));
    record_format.read_record(input_file, skip8.data(), static_cast<int32_t>(skip8.size()));

    int32_t nrecord = 0;
    record_format.read_record(input_file, &nrecord, sizeof(nrecord));
    int32_t number_of_g_vectors = 0;
    record_format.read_record(input_file, &number_of_g_vectors, sizeof(number_of_g_vectors));
    std::vector<char> skip_gvec(3 * number_of_g_vectors * sizeof(int32_t));
    record_format.read_record(input_file, skip_gvec.data(), static_cast<int32_t>(skip_gvec.size()));
    ASSERT_FALSE(input_file.fail());

    int32_t density_nrecord = 0;
    record_format.read_record(input_file, &density_nrecord, sizeof(density_nrecord));
    int32_t density_ng = 0;
    record_format.read_record(input_file, &density_ng, sizeof(density_ng));
    ASSERT_FALSE(input_file.fail());
    EXPECT_EQ(density_ng, number_of_g_vectors); // ng itself is unchanged; ns multiplies the payload

    // Payload is 2*ng complex values (spin-major, G-minor): must succeed at exactly this size.
    std::vector<Complex> rho_g(static_cast<size_t>(2) * number_of_g_vectors);
    record_format.read_record(
        input_file, rho_g.data(), static_cast<int32_t>(rho_g.size() * sizeof(Complex)));
    EXPECT_FALSE(input_file.fail());

    Real cell_volume = crystal_structure.cell_volume();
    for (int spin_index = 0; spin_index < 2; spin_index++) {
        std::vector<Complex> expected_density_g;
        basis.rho_r_to_g(scf_state.charge_density_real[spin_index], expected_density_g);
        for (int g_index = 0; g_index < number_of_g_vectors; g_index++) {
            Complex expected = expected_density_g[g_index] * cell_volume;
            Complex actual = rho_g[static_cast<size_t>(spin_index) * number_of_g_vectors + g_index];
            EXPECT_NEAR(actual.real(), expected.real(), 1e-8);
            EXPECT_NEAR(actual.imag(), expected.imag(), 1e-8);
        }
    }

    std::remove(path.c_str());
}
