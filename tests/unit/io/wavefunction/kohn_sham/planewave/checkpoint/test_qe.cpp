#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "io/wavefunction/kohn_sham/planewave/checkpoint/qe.hpp"
#include "layer_1_structure/crystal/crystal.hpp"
#include "layer_1_structure/reciprocal_space/kpoints_grid.hpp"
#include "layer_3_representation/basis/planewave/init.hpp"
#include "layer_3_representation/quantum_system/kohn_sham/planewave/norm_conserving/init.hpp"
#include "layer_4_computation/iterator/kohn_sham/planewave/execution/serial.hpp"
#include "layer_4_computation/state/kohn_sham/planewave/norm_conserving/init.hpp"
#include "utils/temporary_path.hpp"

using namespace q3;

/**
 * Byte-level checks of wfcN.dat against the record layout Quantum ESPRESSO's
 * write_wfc produces (Modules/io_base.f90): a packed 44-byte k-point header,
 * reciprocal vectors and k in 1/Bohr, and Miller indices interleaved per
 * G-vector (Fortran mill(3, igwx)).
 */
namespace {

execution::PlanewaveSerialExecutionPolicy serial_policy;

std::vector<char> read_payload(std::ifstream& input_file) {
    int32_t prefix = 0;
    input_file.read(reinterpret_cast<char*>(&prefix), 4);
    std::vector<char> payload(prefix > 0 ? prefix : 0);
    input_file.read(payload.data(), prefix);
    int32_t suffix = 0;
    input_file.read(reinterpret_cast<char*>(&suffix), 4);
    EXPECT_EQ(prefix, suffix);
    return payload;
}

template <typename Scalar> Scalar field_at(const std::vector<char>& payload, std::size_t offset) {
    Scalar value;
    std::memcpy(&value, payload.data() + offset, sizeof(Scalar));
    return value;
}

} // namespace

TEST(QEPlanewaveKohnShamWavefunctionCheckpoint, WritesPackedHeaderAndInterleavedMiller) {
    std::vector<AtomSpecies> species = {AtomSpecies{"Si", 28.0855, 4.0, "Si.upf"}};
    std::vector<Atom> atoms = {Atom{0, {0.0, 0.0, 0.0}}, Atom{0, {0.25, 0.25, 0.25}}};
    Crystal crystal_structure = Crystal::from_input_data(2, 10.0, species, atoms);
    PlanewaveBasis basis;
    basis.generate(serial_policy, crystal_structure, 8.0, 32.0);
    KPointsGrid kpoint_grid;
    kpoint_grid.generate(2, 2, 2, 1, 1, 1);
    basis.initialize_kpoint_data(crystal_structure, kpoint_grid);

    const int number_of_bands = 2;
    auto quantum_system = std::make_shared<NormConservingPlanewaveKohnShamQuantumSystem>();
    quantum_system->set_kpoint_grid(&kpoint_grid);
    quantum_system->set_basis(&basis);
    quantum_system->set_number_of_bands(number_of_bands);
    quantum_system->set_number_of_electrons(4.0);
    quantum_system->allocate_electronic_states(basis.kpoint_number_of_plane_waves());

    NormConservingPlanewaveKohnShamSCFState scf_state;
    scf_state.crystal_structure = &crystal_structure;
    scf_state.kpoint_grid = &kpoint_grid;
    scf_state.planewave_basis = &basis;
    scf_state.execution_policy = &serial_policy;
    scf_state.quantum_system_snapshot_ = quantum_system;

    const std::filesystem::path directory = test_support::make_temporary_path(".save");
    std::filesystem::create_directories(directory);
    QEPlanewaveKohnShamWavefunctionCheckpoint(scf_state).write_to_file(directory.string());

    const int kpoint_index = 0;
    std::ifstream input_file(directory / "wfc1.dat", std::ios::binary);
    ASSERT_TRUE(input_file.is_open());

    // Record 1: ik, xk(3), ispin, gamma_only, scalef -- 44 packed bytes.
    const auto header = read_payload(input_file);
    ASSERT_EQ(header.size(), 44u);
    EXPECT_EQ(field_at<int32_t>(header, 0), 1);
    const Matrix3x3 b_bohr = crystal_structure.reciprocal_vectors_bohr();
    const auto& kpoint = kpoint_grid[kpoint_index];
    double kpoint_norm_squared = 0.0;
    for (int dimension = 0; dimension < 3; dimension++) {
        const double expected = kpoint.coordinates.x * b_bohr[0][dimension] +
                                kpoint.coordinates.y * b_bohr[1][dimension] +
                                kpoint.coordinates.z * b_bohr[2][dimension];
        kpoint_norm_squared += expected * expected;
        EXPECT_NEAR(field_at<double>(header, 4 + 8 * dimension), expected, 1e-14);
    }
    EXPECT_GT(kpoint_norm_squared, 0.0); // a shifted grid, so offsets are actually exercised
    EXPECT_EQ(field_at<int32_t>(header, 28), 1); // ispin
    EXPECT_EQ(field_at<int32_t>(header, 32), 0); // gamma_only (.FALSE.)
    EXPECT_DOUBLE_EQ(field_at<double>(header, 36), 1.0);

    // Record 2: ngw, igwx, npol, nbnd.
    const auto dimensions = read_payload(input_file);
    ASSERT_EQ(dimensions.size(), 16u);
    const int number_of_plane_waves = basis.kpoint_number_of_plane_waves()[kpoint_index];
    EXPECT_EQ(field_at<int32_t>(dimensions, 0), number_of_plane_waves);
    EXPECT_EQ(field_at<int32_t>(dimensions, 8), 1);
    EXPECT_EQ(field_at<int32_t>(dimensions, 12), number_of_bands);

    // Record 3: b1, b2, b3 in 1/Bohr.
    const auto reciprocal = read_payload(input_file);
    ASSERT_EQ(reciprocal.size(), 9 * sizeof(double));
    for (int vector_index = 0; vector_index < 3; vector_index++) {
        for (int dimension = 0; dimension < 3; dimension++) {
            EXPECT_DOUBLE_EQ(field_at<double>(reciprocal, 8 * (3 * vector_index + dimension)),
                             b_bohr[vector_index][dimension]);
        }
    }

    // Record 4: mill(3, igwx), components contiguous per G-vector.
    const auto miller = read_payload(input_file);
    ASSERT_EQ(miller.size(), 3u * number_of_plane_waves * sizeof(int32_t));
    const auto& g_vectors = basis.g_vectors();
    const auto& g_indices = basis.kpoint_g_vector_indices()[kpoint_index];
    for (int plane_wave = 0; plane_wave < number_of_plane_waves; plane_wave++) {
        const auto& expected = g_vectors[g_indices[plane_wave]].miller;
        EXPECT_EQ(field_at<int32_t>(miller, 4 * (3 * plane_wave + 0)), expected.x);
        EXPECT_EQ(field_at<int32_t>(miller, 4 * (3 * plane_wave + 1)), expected.y);
        EXPECT_EQ(field_at<int32_t>(miller, 4 * (3 * plane_wave + 2)), expected.z);
    }

    // Records 5..: one per band, then end of file.
    for (int band = 0; band < number_of_bands; band++) {
        EXPECT_EQ(read_payload(input_file).size(),
                  static_cast<std::size_t>(number_of_plane_waves) * sizeof(Complex));
    }
    input_file.peek();
    EXPECT_TRUE(input_file.eof());

    input_file.close();
    std::filesystem::remove_all(directory);
}
