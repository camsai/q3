#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "io/charge_density/kohn_sham/planewave/checkpoint/qe.hpp"
#include "layer_1_structure/crystal/crystal.hpp"
#include "layer_1_structure/reciprocal_space/kpoints_grid.hpp"
#include "layer_3_representation/basis/planewave/init.hpp"
#include "layer_4_computation/iterator/kohn_sham/planewave/execution/serial.hpp"
#include "layer_4_computation/state/kohn_sham/planewave/norm_conserving/init.hpp"
#include "utils/temporary_path.hpp"

using namespace q3;

/**
 * Byte-level checks of charge-density.dat against Quantum ESPRESSO's
 * write_rhog record sequence (Modules/io_base.f90): header, b1/b2/b3 in
 * 1/Bohr, Miller indices interleaved per G-vector, then rho(G).
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

TEST(QEPlanewaveKohnShamChargeDensityCheckpoint, WritesQERecordSequence) {
    std::vector<AtomSpecies> species = {AtomSpecies{"Si", 28.0855, 4.0, "Si.upf"}};
    std::vector<Atom> atoms = {Atom{0, {0.0, 0.0, 0.0}}, Atom{0, {0.25, 0.25, 0.25}}};
    Crystal crystal_structure = Crystal::from_input_data(2, 10.0, species, atoms);
    PlanewaveBasis basis;
    basis.generate(serial_policy, crystal_structure, 8.0, 32.0);
    KPointsGrid kpoint_grid;
    kpoint_grid.generate(1, 1, 1, 0, 0, 0);

    NormConservingPlanewaveKohnShamSCFState scf_state;
    scf_state.crystal_structure = &crystal_structure;
    scf_state.kpoint_grid = &kpoint_grid;
    scf_state.planewave_basis = &basis;
    const int number_of_g_vectors = basis.number_of_g_vectors();
    scf_state.charge_density_g.resize(number_of_g_vectors);
    for (int g_index = 0; g_index < number_of_g_vectors; g_index++) {
        scf_state.charge_density_g[g_index] = Complex(0.5 * g_index, -0.25 * g_index);
    }

    const std::string path = test_support::make_temporary_path(".dat");
    QEPlanewaveKohnShamChargeDensityCheckpoint(scf_state).write_to_file(path);
    std::ifstream input_file(path, std::ios::binary);
    ASSERT_TRUE(input_file.is_open());

    // Record 1: gamma_only, ngm_g, nspin.
    const auto header = read_payload(input_file);
    ASSERT_EQ(header.size(), 12u);
    EXPECT_EQ(field_at<int32_t>(header, 0), 0);
    EXPECT_EQ(field_at<int32_t>(header, 4), number_of_g_vectors);
    EXPECT_EQ(field_at<int32_t>(header, 8), 1);

    // Record 2: b1, b2, b3 in 1/Bohr.
    const auto reciprocal = read_payload(input_file);
    ASSERT_EQ(reciprocal.size(), 9 * sizeof(double));
    const Matrix3x3 b_bohr = crystal_structure.reciprocal_vectors_bohr();
    for (int vector_index = 0; vector_index < 3; vector_index++) {
        for (int dimension = 0; dimension < 3; dimension++) {
            EXPECT_DOUBLE_EQ(field_at<double>(reciprocal, 8 * (3 * vector_index + dimension)),
                             b_bohr[vector_index][dimension]);
        }
    }

    // Record 3: mill_g(3, ngm_g), components contiguous per G-vector.
    const auto miller = read_payload(input_file);
    ASSERT_EQ(miller.size(), 3u * number_of_g_vectors * sizeof(int32_t));
    const auto& g_vectors = basis.g_vectors();
    for (int g_index = 0; g_index < number_of_g_vectors; g_index++) {
        EXPECT_EQ(field_at<int32_t>(miller, 4 * (3 * g_index + 0)), g_vectors[g_index].miller.x);
        EXPECT_EQ(field_at<int32_t>(miller, 4 * (3 * g_index + 1)), g_vectors[g_index].miller.y);
        EXPECT_EQ(field_at<int32_t>(miller, 4 * (3 * g_index + 2)), g_vectors[g_index].miller.z);
    }

    // Record 4: rho(G).
    const auto density = read_payload(input_file);
    ASSERT_EQ(density.size(), number_of_g_vectors * sizeof(Complex));
    EXPECT_DOUBLE_EQ(field_at<double>(density, 16 * 3), 1.5);
    EXPECT_DOUBLE_EQ(field_at<double>(density, 16 * 3 + 8), -0.75);

    input_file.close();
    std::filesystem::remove(path);
}
