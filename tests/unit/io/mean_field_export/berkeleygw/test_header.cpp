#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "io/checkpoint/qe.hpp"
#include "io/mean_field_export/berkeleygw/header.hpp"
#include "io/mean_field_record/from_scf_state.hpp"
#include "layer_1_structure/crystal/crystal.hpp"
#include "layer_1_structure/reciprocal_space/kpoints_grid.hpp"
#include "layer_1_structure/symmetry/operation.hpp"
#include "layer_1_structure/symmetry/symmetry.hpp"
#include "layer_3_representation/basis/planewave/init.hpp"
#include "layer_3_representation/quantum_system/kohn_sham/planewave/norm_conserving/init.hpp"
#include "layer_4_computation/iterator/kohn_sham/planewave/execution/serial.hpp"
#include "layer_4_computation/state/kohn_sham/planewave/norm_conserving/init.hpp"
#include "utils/temporary_path.hpp"

using namespace q3;

namespace {

execution::PlanewaveSerialExecutionPolicy serial_policy;

/**
 * Diamond-structure (FCC, 2-atom basis) silicon-like fixture: small enough
 * to keep G-vector/plane-wave counts tractable for byte-exact assertions,
 * but with a non-symmorphic space group (Fd-3m) so mtrx/tnp exercise both
 * symmorphic (zero translation) and non-symmorphic (nonzero translation)
 * operations, and at least one non-symmetric rotation matrix (so a
 * row/column-major mtrx bug would actually change the written bytes).
 */
void populate_synthetic_state(
    Crystal& crystal_structure,
    PlanewaveBasis& basis,
    KPointsGrid& kpoint_grid,
    std::shared_ptr<NormConservingPlanewaveKohnShamQuantumSystem>& quantum_system,
    NormConservingPlanewaveKohnShamSCFState& scf_state,
    int number_of_bands = 6,
    int number_of_spin_components = 1,
    std::array<int, 3> kpoint_grid_dimensions = {2, 1, 1},
    std::array<int, 3> kpoint_grid_shifts = {0, 0, 0},
    bool reduce_kpoint_grid = false) {
    std::vector<AtomSpecies> species = {AtomSpecies{"Si", 28.0855, 4.0, "Si.upf"}};
    std::vector<Atom> atoms = {Atom{0, {0.875, 0.875, 0.875}}, Atom{0, {0.125, 0.125, 0.125}}};
    crystal_structure = Crystal::from_input_data(2, 10.0, species, atoms); // FCC, alat = 10 bohr

    basis.generate(serial_policy, crystal_structure, 8.0, 32.0);

    kpoint_grid.generate(kpoint_grid_dimensions[0],
                         kpoint_grid_dimensions[1],
                         kpoint_grid_dimensions[2],
                         kpoint_grid_shifts[0],
                         kpoint_grid_shifts[1],
                         kpoint_grid_shifts[2]);
    if (reduce_kpoint_grid) {
        kpoint_grid = SymmetryAnalyzer::reduce_kpoints(crystal_structure, kpoint_grid);
    }
    if (number_of_spin_components == 2) {
        kpoint_grid.scale_weights(0.5);
    }
    basis.initialize_kpoint_data(crystal_structure, kpoint_grid);

    quantum_system = std::make_shared<NormConservingPlanewaveKohnShamQuantumSystem>();
    quantum_system->set_kpoint_grid(&kpoint_grid);
    quantum_system->set_basis(&basis);
    quantum_system->set_number_of_bands(number_of_bands);
    quantum_system->set_number_of_electrons(8.0);
    quantum_system->number_of_spin_components = number_of_spin_components;
    quantum_system->allocate_electronic_states(basis.kpoint_number_of_plane_waves());

    scf_state.crystal_structure = &crystal_structure;
    scf_state.kpoint_grid = &kpoint_grid;
    scf_state.planewave_basis = &basis;
    scf_state.quantum_system_snapshot_ = quantum_system;
}

std::string make_temporary_path(const std::string& suffix) {
    return test_support::make_temporary_path(suffix);
}

/// Read `length` bytes from `buffer` as a std::string (no trimming).
std::string bytes_to_string(const std::vector<char>& buffer, size_t offset, size_t length) {
    return {buffer.data() + offset, length};
}

} // namespace

// ── Record 1: title/date/time ───────────────────────────────────────────

TEST(BerkeleyGWCommonHeader, TitleRecordIsNinetySixPaddedBytes) {
    Crystal crystal_structure;
    PlanewaveBasis basis;
    KPointsGrid kpoint_grid;
    std::shared_ptr<NormConservingPlanewaveKohnShamQuantumSystem> quantum_system;
    NormConservingPlanewaveKohnShamSCFState scf_state;
    populate_synthetic_state(crystal_structure, basis, kpoint_grid, quantum_system, scf_state);

    std::string path = make_temporary_path(".wfn_header");
    {
        std::ofstream output_file(path, std::ios::binary);
        const MeanFieldRecord record = build_mean_field_record(scf_state);
        BerkeleyGWCommonHeader header(record, "WFN-Complex", /*include_kpoint_fields=*/true);
        header.write(output_file);
    }

    std::ifstream input_file(path, std::ios::binary);
    QEBinaryRecordFormat record_format;

    std::vector<char> title_record(96);
    record_format.read_record(input_file, title_record.data(), 96);
    ASSERT_FALSE(input_file.fail());

    EXPECT_EQ(bytes_to_string(title_record, 0, 32),
              std::string("WFN-Complex") + std::string(21, ' '));
    // sdate/stime are timestamp-dependent; only their fixed 32-byte framing is asserted here.

    std::remove(path.c_str());
}

TEST(BerkeleyGWCommonHeader, WrongDeclaredLengthFailsTheRead) {
    Crystal crystal_structure;
    PlanewaveBasis basis;
    KPointsGrid kpoint_grid;
    std::shared_ptr<NormConservingPlanewaveKohnShamQuantumSystem> quantum_system;
    NormConservingPlanewaveKohnShamSCFState scf_state;
    populate_synthetic_state(crystal_structure, basis, kpoint_grid, quantum_system, scf_state);

    std::string path = make_temporary_path(".wfn_header_bad_length");
    {
        std::ofstream output_file(path, std::ios::binary);
        const MeanFieldRecord record = build_mean_field_record(scf_state);
        BerkeleyGWCommonHeader header(record, "WFN-Complex", /*include_kpoint_fields=*/true);
        header.write(output_file);
    }

    std::ifstream input_file(path, std::ios::binary);
    QEBinaryRecordFormat record_format;
    std::vector<char> buffer(95); // record 1 is actually 96 bytes
    record_format.read_record(input_file, buffer.data(), 95);
    EXPECT_TRUE(input_file.fail());

    std::remove(path.c_str());
}

// ── Record 2: dimensions ────────────────────────────────────────────────

TEST(BerkeleyGWCommonHeader, DimensionRecordMatchesExpectedFieldsForWfn) {
    Crystal crystal_structure;
    PlanewaveBasis basis;
    KPointsGrid kpoint_grid;
    std::shared_ptr<NormConservingPlanewaveKohnShamQuantumSystem> quantum_system;
    NormConservingPlanewaveKohnShamSCFState scf_state;
    populate_synthetic_state(crystal_structure, basis, kpoint_grid, quantum_system, scf_state);

    const auto symmetry_operations = SymmetryAnalyzer::get_symmetry_operations(crystal_structure);

    std::string path = make_temporary_path(".wfn_header_dim");
    {
        std::ofstream output_file(path, std::ios::binary);
        const MeanFieldRecord record = build_mean_field_record(scf_state);
        BerkeleyGWCommonHeader header(record, "WFN-Complex", /*include_kpoint_fields=*/true);
        header.write(output_file);
    }

    std::ifstream input_file(path, std::ios::binary);
    QEBinaryRecordFormat record_format;
    std::vector<char> title_record(96);
    record_format.read_record(input_file, title_record.data(), 96);
    ASSERT_FALSE(input_file.fail());

    struct {
        int32_t number_of_spin_components;
        int32_t number_of_g_vectors;
        int32_t number_of_symmetry_operations;
        int32_t cell_symmetry;
        int32_t number_of_atoms;
        double charge_density_cutoff_ry;
        int32_t number_of_kpoints;
        int32_t number_of_bands;
        int32_t maximum_plane_waves;
        double wavefunction_cutoff_ry;
    } dimension_record{};
    static_assert(sizeof(dimension_record) >= 48, "test struct must be large enough");

    std::vector<char> raw_bytes(48);
    record_format.read_record(input_file, raw_bytes.data(), 48);
    ASSERT_FALSE(input_file.fail());
    size_t offset = 0;
    auto extract = [&](auto& field) {
        std::memcpy(&field, raw_bytes.data() + offset, sizeof(field));
        offset += sizeof(field);
    };
    extract(dimension_record.number_of_spin_components);
    extract(dimension_record.number_of_g_vectors);
    extract(dimension_record.number_of_symmetry_operations);
    extract(dimension_record.cell_symmetry);
    extract(dimension_record.number_of_atoms);
    extract(dimension_record.charge_density_cutoff_ry);
    extract(dimension_record.number_of_kpoints);
    extract(dimension_record.number_of_bands);
    extract(dimension_record.maximum_plane_waves);
    extract(dimension_record.wavefunction_cutoff_ry);

    const auto& plane_waves_per_kpoint = basis.kpoint_number_of_plane_waves();
    int32_t expected_max_plane_waves =
        *std::max_element(plane_waves_per_kpoint.begin(), plane_waves_per_kpoint.end());

    EXPECT_EQ(dimension_record.number_of_spin_components, 1);
    EXPECT_EQ(dimension_record.number_of_g_vectors, basis.number_of_g_vectors());
    EXPECT_EQ(dimension_record.number_of_symmetry_operations,
              static_cast<int32_t>(symmetry_operations.size()));
    EXPECT_EQ(dimension_record.cell_symmetry, 0); // FCC (bravais_lattice_index 2) => cubic family
    EXPECT_EQ(dimension_record.number_of_atoms, 2);
    EXPECT_DOUBLE_EQ(dimension_record.charge_density_cutoff_ry, 32.0);
    EXPECT_EQ(dimension_record.number_of_kpoints, 2);
    EXPECT_EQ(dimension_record.number_of_bands, 6);
    EXPECT_EQ(dimension_record.maximum_plane_waves, expected_max_plane_waves);
    EXPECT_DOUBLE_EQ(dimension_record.wavefunction_cutoff_ry, 8.0);

    std::remove(path.c_str());
}

TEST(BerkeleyGWCommonHeader, DimensionRecordOmitsKpointFieldsForRho) {
    Crystal crystal_structure;
    PlanewaveBasis basis;
    KPointsGrid kpoint_grid;
    std::shared_ptr<NormConservingPlanewaveKohnShamQuantumSystem> quantum_system;
    NormConservingPlanewaveKohnShamSCFState scf_state;
    populate_synthetic_state(crystal_structure, basis, kpoint_grid, quantum_system, scf_state);

    std::string path = make_temporary_path(".rho_header_dim");
    {
        std::ofstream output_file(path, std::ios::binary);
        const MeanFieldRecord record = build_mean_field_record(scf_state);
        BerkeleyGWCommonHeader header(record, "RHO-Complex", /*include_kpoint_fields=*/false);
        header.write(output_file);
    }

    std::ifstream input_file(path, std::ios::binary);
    QEBinaryRecordFormat record_format;
    std::vector<char> title_record(96);
    record_format.read_record(input_file, title_record.data(), 96);
    ASSERT_FALSE(input_file.fail());

    // RHO's record 2 is exactly 28 bytes (5 int32 + 1 double, no k-point fields).
    std::vector<char> raw_bytes(28);
    record_format.read_record(input_file, raw_bytes.data(), 28);
    EXPECT_FALSE(input_file.fail());

    int32_t cell_symmetry = 0;
    std::memcpy(&cell_symmetry, raw_bytes.data() + 3 * sizeof(int32_t), sizeof(int32_t));
    EXPECT_EQ(cell_symmetry, 0);

    // Record 3 for RHO is FFTgrid(3) only -- 12 bytes, not the WFN 48.
    std::vector<char> grid_record(12);
    record_format.read_record(input_file, grid_record.data(), 12);
    EXPECT_FALSE(input_file.fail());

    std::remove(path.c_str());
}

// ── Record 3: FFT grid / k-point grid / shift ───────────────────────────

TEST(BerkeleyGWCommonHeader, GridRecordMatchesFftRhoAndKpointGridForWfn) {
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
                             6,
                             1,
                             {4, 4, 4},
                             {1, 1, 1},
                             true);
    EXPECT_LT(kpoint_grid.number_of_kpoints(), 64);

    std::string path = make_temporary_path(".wfn_header_grid");
    {
        std::ofstream output_file(path, std::ios::binary);
        const MeanFieldRecord record = build_mean_field_record(scf_state);
        BerkeleyGWCommonHeader header(record, "WFN-Complex", /*include_kpoint_fields=*/true);
        header.write(output_file);
    }

    std::ifstream input_file(path, std::ios::binary);
    QEBinaryRecordFormat record_format;
    std::vector<char> title_record(96);
    record_format.read_record(input_file, title_record.data(), 96);
    std::vector<char> dimension_record(48);
    record_format.read_record(input_file, dimension_record.data(), 48);
    ASSERT_FALSE(input_file.fail());

    std::vector<char> grid_record(48); // 6 int32 + 3 double
    record_format.read_record(input_file, grid_record.data(), 48);
    ASSERT_FALSE(input_file.fail());

    int32_t fft_dims[3];
    int32_t kgrid_dims[3];
    double kshift[3];
    std::memcpy(fft_dims, grid_record.data(), sizeof(fft_dims));
    std::memcpy(kgrid_dims, grid_record.data() + sizeof(fft_dims), sizeof(kgrid_dims));
    std::memcpy(kshift, grid_record.data() + sizeof(fft_dims) + sizeof(kgrid_dims), sizeof(kshift));

    const auto& fft_rho = basis.fft_rho();
    EXPECT_EQ(fft_dims[0], fft_rho.grid_dimension_1());
    EXPECT_EQ(fft_dims[1], fft_rho.grid_dimension_2());
    EXPECT_EQ(fft_dims[2], fft_rho.grid_dimension_3());
    EXPECT_EQ(kgrid_dims[0], 4);
    EXPECT_EQ(kgrid_dims[1], 4);
    EXPECT_EQ(kgrid_dims[2], 4);
    EXPECT_DOUBLE_EQ(kshift[0], 0.5);
    EXPECT_DOUBLE_EQ(kshift[1], 0.5);
    EXPECT_DOUBLE_EQ(kshift[2], 0.5);

    std::remove(path.c_str());
}

// ── Records 4-5: direct/reciprocal lattice + metric tensors ─────────────

TEST(BerkeleyGWCommonHeader, DirectLatticeRecordMatchesCrystalAndMetricTensor) {
    Crystal crystal_structure;
    PlanewaveBasis basis;
    KPointsGrid kpoint_grid;
    std::shared_ptr<NormConservingPlanewaveKohnShamQuantumSystem> quantum_system;
    NormConservingPlanewaveKohnShamSCFState scf_state;
    populate_synthetic_state(crystal_structure, basis, kpoint_grid, quantum_system, scf_state);

    std::string path = make_temporary_path(".wfn_header_lattice");
    {
        std::ofstream output_file(path, std::ios::binary);
        const MeanFieldRecord record = build_mean_field_record(scf_state);
        BerkeleyGWCommonHeader header(record, "WFN-Complex", /*include_kpoint_fields=*/true);
        header.write(output_file);
    }

    std::ifstream input_file(path, std::ios::binary);
    QEBinaryRecordFormat record_format;
    std::vector<char> skip(96);
    record_format.read_record(input_file, skip.data(), 96); // title
    std::vector<char> skip2(48);
    record_format.read_record(input_file, skip2.data(), 48); // dimensions
    std::vector<char> skip3(48);
    record_format.read_record(input_file, skip3.data(), 48); // grid
    ASSERT_FALSE(input_file.fail());

    std::vector<double> lattice_record(20);
    record_format.read_record(
        input_file, lattice_record.data(), static_cast<int32_t>(20 * sizeof(double)));
    ASSERT_FALSE(input_file.fail());

    Real alat = crystal_structure.lattice_parameter();
    const Matrix3x3& direct_lattice = crystal_structure.lattice_vectors();

    EXPECT_DOUBLE_EQ(lattice_record[0], crystal_structure.cell_volume());
    EXPECT_DOUBLE_EQ(lattice_record[1], alat);
    // avec: vector-major (vector 0's x,y,z, then vector 1's, then vector 2's).
    EXPECT_DOUBLE_EQ(lattice_record[2], direct_lattice[0].x);
    EXPECT_DOUBLE_EQ(lattice_record[3], direct_lattice[0].y);
    EXPECT_DOUBLE_EQ(lattice_record[4], direct_lattice[0].z);
    EXPECT_DOUBLE_EQ(lattice_record[5], direct_lattice[1].x);
    EXPECT_DOUBLE_EQ(lattice_record[6], direct_lattice[1].y);
    EXPECT_DOUBLE_EQ(lattice_record[7], direct_lattice[1].z);
    EXPECT_DOUBLE_EQ(lattice_record[8], direct_lattice[2].x);
    EXPECT_DOUBLE_EQ(lattice_record[9], direct_lattice[2].y);
    EXPECT_DOUBLE_EQ(lattice_record[10], direct_lattice[2].z);
    // adot: symmetric Gram matrix a_i . a_j * alat^2.
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            Real expected = direct_lattice[i].dot(direct_lattice[j]) * alat * alat;
            EXPECT_NEAR(lattice_record[11 + 3 * j + i], expected, 1e-10)
                << "adot(" << i << "," << j << ")";
        }
    }

    std::remove(path.c_str());
}

// ── Record 6-7: symmetry rotation matrices + fractional translations ───

TEST(BerkeleyGWCommonHeader, SymmetryRotationRecordIsQETransposeOfRotation) {
    Crystal crystal_structure;
    PlanewaveBasis basis;
    KPointsGrid kpoint_grid;
    std::shared_ptr<NormConservingPlanewaveKohnShamQuantumSystem> quantum_system;
    NormConservingPlanewaveKohnShamSCFState scf_state;
    populate_synthetic_state(crystal_structure, basis, kpoint_grid, quantum_system, scf_state);

    const auto symmetry_operations = SymmetryAnalyzer::get_symmetry_operations(crystal_structure);
    ASSERT_GT(symmetry_operations.size(), 1u);

    /**
     * Sanity check that this fixture actually exercises the transpose:
     * assert at least one operation's rotation matrix is not symmetric (a
     * symmetric matrix is its own transpose, which would make the test pass
     * even with a transposition bug).
     */
    bool has_asymmetric_rotation = false;
    for (const auto& operation : symmetry_operations) {
        for (int row = 0; row < 3 && !has_asymmetric_rotation; row++) {
            for (int column = 0; column < 3; column++) {
                if (operation.rotation[row][column] != operation.rotation[column][row]) {
                    has_asymmetric_rotation = true;
                    break;
                }
            }
        }
    }
    ASSERT_TRUE(has_asymmetric_rotation)
        << "fixture must include a non-symmetric rotation to discriminate row-vs-column-major";

    std::string path = make_temporary_path(".wfn_header_mtrx");
    {
        std::ofstream output_file(path, std::ios::binary);
        const MeanFieldRecord record = build_mean_field_record(scf_state);
        BerkeleyGWCommonHeader header(record, "WFN-Complex", /*include_kpoint_fields=*/true);
        header.write(output_file);
    }

    std::ifstream input_file(path, std::ios::binary);
    QEBinaryRecordFormat record_format;
    std::vector<char> skip(96);
    record_format.read_record(input_file, skip.data(), 96);
    std::vector<char> skip2(48);
    record_format.read_record(input_file, skip2.data(), 48);
    std::vector<char> skip3(48);
    record_format.read_record(input_file, skip3.data(), 48);
    std::vector<char> skip4(20 * sizeof(double));
    record_format.read_record(input_file, skip4.data(), static_cast<int32_t>(20 * sizeof(double)));
    std::vector<char> skip5(20 * sizeof(double));
    record_format.read_record(input_file, skip5.data(), static_cast<int32_t>(20 * sizeof(double)));
    ASSERT_FALSE(input_file.fail());

    int number_of_operations = static_cast<int>(symmetry_operations.size());
    std::vector<int32_t> mtrx(9 * number_of_operations);
    record_format.read_record(
        input_file, mtrx.data(), static_cast<int32_t>(mtrx.size() * sizeof(int32_t)));
    ASSERT_FALSE(input_file.fail());

    /**
     * Expected: mtrx(k,j) = rotation[j][k] (QE's s is rotation^T), flattened
     * Fortran column-major -- k fastest -- which is rotation row by row.
     */
    std::vector<int32_t> expected_mtrx;
    expected_mtrx.reserve(mtrx.size());
    for (const auto& operation : symmetry_operations) {
        for (int j = 0; j < 3; j++) {     // Fortran column (slow)
            for (int k = 0; k < 3; k++) { // Fortran row (fast)
                expected_mtrx.push_back(operation.rotation[j][k]);
            }
        }
    }
    EXPECT_EQ(mtrx, expected_mtrx);

    std::remove(path.c_str());
}

TEST(BerkeleyGWCommonHeader, SymmetryTranslationRecordIsInverseOperationTranslation) {
    Crystal crystal_structure;
    PlanewaveBasis basis;
    KPointsGrid kpoint_grid;
    std::shared_ptr<NormConservingPlanewaveKohnShamQuantumSystem> quantum_system;
    NormConservingPlanewaveKohnShamSCFState scf_state;
    populate_synthetic_state(crystal_structure, basis, kpoint_grid, quantum_system, scf_state);

    const auto symmetry_operations = SymmetryAnalyzer::get_symmetry_operations(crystal_structure);

    /**
     * The diamond-structure (Fd-3m) fixture must have at least one operation
     * that is BOTH non-symmorphic (nonzero fractional_translation) AND has a
     * non-symmetric reciprocal_rotation, so that distinct possible bugs each
     * change the result: a missing final *2*pi (any nonzero translation
     * catches this) and contracting the wrong index of reciprocal_rotation
     * (indistinguishable from the correct answer whenever it is symmetric,
     * e.g. the identity). The sign is pinned against pw2bgw itself by the
     * trigonal-Te test below.
     */
    bool has_discriminating_operation = false;
    for (const auto& operation : symmetry_operations) {
        if (operation.fractional_translation.norm() <= 1e-6)
            continue;
        bool reciprocal_rotation_is_symmetric = true;
        for (int row = 0; row < 3 && reciprocal_rotation_is_symmetric; row++) {
            for (int column = 0; column < 3; column++) {
                if (operation.reciprocal_rotation[row][column] !=
                    operation.reciprocal_rotation[column][row]) {
                    reciprocal_rotation_is_symmetric = false;
                    break;
                }
            }
        }
        if (!reciprocal_rotation_is_symmetric) {
            has_discriminating_operation = true;
            break;
        }
    }
    ASSERT_TRUE(has_discriminating_operation)
        << "fixture must include a non-symmorphic operation with an asymmetric "
           "reciprocal_rotation to discriminate the tnp transform";

    std::string path = make_temporary_path(".wfn_header_tnp");
    {
        std::ofstream output_file(path, std::ios::binary);
        const MeanFieldRecord record = build_mean_field_record(scf_state);
        BerkeleyGWCommonHeader header(record, "WFN-Complex", /*include_kpoint_fields=*/true);
        header.write(output_file);
    }

    std::ifstream input_file(path, std::ios::binary);
    QEBinaryRecordFormat record_format;
    std::vector<char> skip(96);
    record_format.read_record(input_file, skip.data(), 96);
    std::vector<char> skip2(48);
    record_format.read_record(input_file, skip2.data(), 48);
    std::vector<char> skip3(48);
    record_format.read_record(input_file, skip3.data(), 48);
    std::vector<char> skip4(20 * sizeof(double));
    record_format.read_record(input_file, skip4.data(), static_cast<int32_t>(20 * sizeof(double)));
    std::vector<char> skip5(20 * sizeof(double));
    record_format.read_record(input_file, skip5.data(), static_cast<int32_t>(20 * sizeof(double)));
    int number_of_operations = static_cast<int>(symmetry_operations.size());
    std::vector<char> skip6(9 * number_of_operations * sizeof(int32_t));
    record_format.read_record(input_file, skip6.data(), static_cast<int32_t>(skip6.size()));
    ASSERT_FALSE(input_file.fail());

    std::vector<double> tnp(3 * number_of_operations);
    record_format.read_record(
        input_file, tnp.data(), static_cast<int32_t>(tnp.size() * sizeof(double)));
    ASSERT_FALSE(input_file.fail());

    /**
     * tnp / 2*pi = -rotation^-1 @ fractional_translation, wrapped to
     * (-0.5, 0.5], with rotation^-1 inverted here directly rather than
     * taken from reciprocal_rotation.
     */
    constexpr Real eps6 = 1.0e-6;
    for (int operation_index = 0; operation_index < number_of_operations; operation_index++) {
        const auto& operation = symmetry_operations[operation_index];
        const auto& r = operation.rotation;
        const Real determinant = r[0][0] * (r[1][1] * r[2][2] - r[1][2] * r[2][1]) -
                                 r[0][1] * (r[1][0] * r[2][2] - r[1][2] * r[2][0]) +
                                 r[0][2] * (r[1][0] * r[2][1] - r[1][1] * r[2][0]);
        Real inverse[3][3];
        for (int row = 0; row < 3; row++) {
            for (int column = 0; column < 3; column++) {
                // Adjugate: cofactor of r[column][row].
                const int r0 = (column + 1) % 3, r1 = (column + 2) % 3;
                const int c0 = (row + 1) % 3, c1 = (row + 2) % 3;
                inverse[row][column] =
                    (r[r0][c0] * r[r1][c1] - r[r0][c1] * r[r1][c0]) / determinant;
            }
        }
        Real rotated[3];
        for (int component = 0; component < 3; component++) {
            Real sum = 0.0;
            for (int contracted = 0; contracted < 3; contracted++) {
                sum -=
                    inverse[component][contracted] * operation.fractional_translation[contracted];
            }
            rotated[component] = sum;
        }
        for (int component = 0; component < 3; component++) {
            if (rotated[component] >= eps6 + 0.5)
                rotated[component] -= std::trunc(rotated[component] + 0.5);
            if (rotated[component] < eps6 - 0.5)
                rotated[component] -= std::trunc(rotated[component] - 0.5);
        }
        for (int component = 0; component < 3; component++) {
            Real expected = rotated[component] * constants::two_pi;
            EXPECT_NEAR(tnp[3 * operation_index + component], expected, 1e-9)
                << "operation " << operation_index << " component " << component;
        }
    }

    std::remove(path.c_str());
}

// ── Record 8: atom positions + atomic numbers ───────────────────────────

TEST(BerkeleyGWCommonHeader, AtomRecordInterleavesPositionAndAtomicNumber) {
    Crystal crystal_structure;
    PlanewaveBasis basis;
    KPointsGrid kpoint_grid;
    std::shared_ptr<NormConservingPlanewaveKohnShamQuantumSystem> quantum_system;
    NormConservingPlanewaveKohnShamSCFState scf_state;
    populate_synthetic_state(crystal_structure, basis, kpoint_grid, quantum_system, scf_state);

    const auto symmetry_operations = SymmetryAnalyzer::get_symmetry_operations(crystal_structure);
    int number_of_operations = static_cast<int>(symmetry_operations.size());

    std::string path = make_temporary_path(".wfn_header_atoms");
    {
        std::ofstream output_file(path, std::ios::binary);
        const MeanFieldRecord record = build_mean_field_record(scf_state);
        BerkeleyGWCommonHeader header(record, "WFN-Complex", /*include_kpoint_fields=*/true);
        header.write(output_file);
    }

    std::ifstream input_file(path, std::ios::binary);
    QEBinaryRecordFormat record_format;
    std::vector<char> skip(96);
    record_format.read_record(input_file, skip.data(), 96);
    std::vector<char> skip2(48);
    record_format.read_record(input_file, skip2.data(), 48);
    std::vector<char> skip3(48);
    record_format.read_record(input_file, skip3.data(), 48);
    std::vector<char> skip4(20 * sizeof(double));
    record_format.read_record(input_file, skip4.data(), static_cast<int32_t>(20 * sizeof(double)));
    std::vector<char> skip5(20 * sizeof(double));
    record_format.read_record(input_file, skip5.data(), static_cast<int32_t>(20 * sizeof(double)));
    std::vector<char> skip6(9 * number_of_operations * sizeof(int32_t));
    record_format.read_record(input_file, skip6.data(), static_cast<int32_t>(skip6.size()));
    std::vector<char> skip7(3 * number_of_operations * sizeof(double));
    record_format.read_record(input_file, skip7.data(), static_cast<int32_t>(skip7.size()));
    ASSERT_FALSE(input_file.fail());

    int number_of_atoms = crystal_structure.number_of_atoms();
    std::vector<char> atom_record(number_of_atoms * (3 * sizeof(double) + sizeof(int32_t)));
    record_format.read_record(
        input_file, atom_record.data(), static_cast<int32_t>(atom_record.size()));
    ASSERT_FALSE(input_file.fail());

    size_t offset = 0;
    for (int atom_index = 0; atom_index < number_of_atoms; atom_index++) {
        double position[3];
        std::memcpy(position, atom_record.data() + offset, sizeof(position));
        offset += sizeof(position);
        int32_t atomic_number = 0;
        std::memcpy(&atomic_number, atom_record.data() + offset, sizeof(atomic_number));
        offset += sizeof(atomic_number);

        Vector3D expected_position =
            crystal_structure.convert_from_crystal_to_cartesian(atom_index);
        EXPECT_DOUBLE_EQ(position[0], expected_position.x);
        EXPECT_DOUBLE_EQ(position[1], expected_position.y);
        EXPECT_DOUBLE_EQ(position[2], expected_position.z);
        EXPECT_EQ(atomic_number, 14); // Si
    }

    std::remove(path.c_str());
}

// ── cell_symmetry: a free lattice is classified from its vectors ────────

namespace {

/// The cell_symmetry a RHO header records for a crystal given by free lattice vectors.
int32_t written_cell_symmetry_for_free_lattice(const Matrix3x3& lattice_vectors) {
    std::vector<AtomSpecies> species = {AtomSpecies{"Si", 28.0855, 4.0, "Si.upf"}};
    std::vector<Atom> atoms = {Atom{0, {0.0, 0.0, 0.0}}};
    Crystal crystal_structure =
        Crystal::from_lattice_vectors(10.0, lattice_vectors, species, atoms);
    EXPECT_EQ(crystal_structure.bravais_lattice_index(), 0);

    PlanewaveBasis basis;
    basis.generate(serial_policy, crystal_structure, 8.0, 32.0);
    KPointsGrid kpoint_grid;
    kpoint_grid.generate(1, 1, 1, 0, 0, 0);
    basis.initialize_kpoint_data(crystal_structure, kpoint_grid);

    auto quantum_system = std::make_shared<NormConservingPlanewaveKohnShamQuantumSystem>();
    quantum_system->set_kpoint_grid(&kpoint_grid);
    quantum_system->set_basis(&basis);
    quantum_system->set_number_of_bands(2);
    quantum_system->set_number_of_electrons(4.0);
    quantum_system->allocate_electronic_states(basis.kpoint_number_of_plane_waves());

    NormConservingPlanewaveKohnShamSCFState scf_state;
    scf_state.crystal_structure = &crystal_structure;
    scf_state.kpoint_grid = &kpoint_grid;
    scf_state.planewave_basis = &basis;
    scf_state.quantum_system_snapshot_ = quantum_system;

    std::string path = make_temporary_path(".rho_header_free_lattice");
    {
        std::ofstream output_file(path, std::ios::binary);
        const MeanFieldRecord record = build_mean_field_record(scf_state);
        BerkeleyGWCommonHeader header(record, "RHO-Complex", /*include_kpoint_fields=*/false);
        header.write(output_file);
    }
    std::ifstream input_file(path, std::ios::binary);
    QEBinaryRecordFormat record_format;
    std::vector<char> title_record(96);
    record_format.read_record(input_file, title_record.data(), 96);
    std::vector<char> raw_bytes(28);
    record_format.read_record(input_file, raw_bytes.data(), 28);
    EXPECT_FALSE(input_file.fail());
    int32_t cell_symmetry = -1;
    std::memcpy(&cell_symmetry, raw_bytes.data() + 3 * sizeof(int32_t), sizeof(int32_t));
    input_file.close();
    std::remove(path.c_str());
    return cell_symmetry;
}

} // namespace

/**
 * A free lattice used to be refused, which shut out every non-cubic crystal:
 * a YAML lattice outside the cubic family and every QE CELL_PARAMETERS cell
 * arrive as one (camsai/q3#21).
 */
TEST(BerkeleyGWCommonHeader, FreeHexagonalLatticeIsTheHexagonalFamily) {
    const Real half_root_three = std::sqrt(3.0) / 2.0;
    Matrix3x3 hexagonal{{1.0, 0.0, 0.0}, {-0.5, half_root_three, 0.0}, {0.0, 0.0, 1.6}};
    EXPECT_EQ(written_cell_symmetry_for_free_lattice(hexagonal), 1);
}

TEST(BerkeleyGWCommonHeader, FreeCubicLatticeIsTheCubicFamily) {
    Matrix3x3 simple_cubic{{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};
    EXPECT_EQ(written_cell_symmetry_for_free_lattice(simple_cubic), 0);
}

// ── Records 3, 6, 7 against pw2bgw: hand-built records ─────────────────

namespace {

/**
 * The header fields of a record for `crystal_structure` with one plane wave
 * per k-point -- enough for BerkeleyGWCommonHeader, without an SCF state.
 */
MeanFieldRecord header_only_record(const Crystal& crystal_structure,
                                   const std::vector<Vector3D>& kpoint_coordinates) {
    MeanFieldRecord record;
    record.bravais_lattice_index = crystal_structure.bravais_lattice_index();
    record.lattice_parameter = crystal_structure.lattice_parameter();
    record.two_pi_over_lattice_parameter = crystal_structure.two_pi_over_lattice_parameter();
    record.cell_volume = crystal_structure.cell_volume();
    record.lattice_vectors = crystal_structure.lattice_vectors();
    record.reciprocal_lattice_vectors = crystal_structure.reciprocal_lattice_vectors();
    for (int atom_index = 0; atom_index < crystal_structure.number_of_atoms(); atom_index++) {
        record.atom_positions_cartesian.push_back(
            crystal_structure.convert_from_crystal_to_cartesian(atom_index));
        record.atomic_numbers.push_back(crystal_structure.atomic_number(atom_index));
    }
    record.symmetry_operations = SymmetryAnalyzer::get_symmetry_operations(crystal_structure);
    record.fft_grid_dimensions = {24, 24, 36};
    record.charge_density_miller_indices = {IntegerVector3D{0, 0, 0}};
    record.kpoint_coordinates = kpoint_coordinates;
    record.kpoint_weights.assign(kpoint_coordinates.size(), 1.0);
    record.plane_waves_per_kpoint.assign(kpoint_coordinates.size(), 1);
    record.number_of_bands = 1;
    return record;
}

/// Write `record`'s WFN header and return its first eight records' payloads.
std::vector<std::vector<char>> written_header_records(const MeanFieldRecord& record,
                                                      const std::string& suffix) {
    const std::string path = make_temporary_path(suffix);
    {
        std::ofstream output_file(path, std::ios::binary);
        BerkeleyGWCommonHeader(record, "WFN-Complex", /*include_kpoint_fields=*/true)
            .write(output_file);
    }
    std::ifstream input_file(path, std::ios::binary);
    std::vector<std::vector<char>> payloads;
    for (int record_index = 0; record_index < 8; record_index++) {
        int32_t length = 0, trailing_length = 0;
        input_file.read(reinterpret_cast<char*>(&length), sizeof(length));
        std::vector<char> payload(length);
        input_file.read(payload.data(), length);
        input_file.read(reinterpret_cast<char*>(&trailing_length), sizeof(trailing_length));
        EXPECT_EQ(length, trailing_length);
        payloads.push_back(std::move(payload));
    }
    EXPECT_FALSE(input_file.fail());
    std::remove(path.c_str());
    return payloads;
}

/// Distance of `value` from the nearest integer.
Real distance_from_integer(Real value) {
    return std::abs(value - std::round(value));
}

} // namespace

/**
 * Trigonal Te (P3_121; a = 4.457 A, c = 5.929 A, x = 0.2636), whose 3_1 screw
 * makes the sign of tnp observable: c/3 and -c/3 are different translations
 * where diamond's quarter translations and wurtzite's c/2 are not. The table
 * is pw2bgw's own output (QE 7.5, ibrav = 4, the same cell): mtrx(k,j) as
 * {row k} triples and tnp / 2*pi. Q3 must write the same six (mtrx, tnp)
 * pairs -- in any order, since operation order is not part of the format.
 * Before the pw2bgw conventions were adopted, Q3 wrote the transposed
 * matrices and paired four of the six with the wrong translation.
 */
TEST(BerkeleyGWCommonHeader, TrigonalTelluriumSymmetryRecordsMatchPw2bgw) {
    const Real x = 0.2636;
    std::vector<AtomSpecies> species = {AtomSpecies{"Te", 127.6, 16.0, "Te.upf"}};
    std::vector<Atom> atoms = {Atom{0, {x, 0.0, 1.0 / 3.0}},
                               Atom{0, {0.0, x, 2.0 / 3.0}},
                               Atom{0, {1.0 - x, 1.0 - x, 0.0}}};
    Crystal tellurium = Crystal::from_input_data(
        4, 4.457 * constants::ANGSTROM_TO_BOHR, species, atoms, 0.0, 5.929 / 4.457);

    struct Pw2bgwOperation {
        std::array<std::array<int, 3>, 3> mtrx; // [k][j] = mtrx(k,j)
        std::array<Real, 3> tnp_over_two_pi;
    };
    const std::vector<Pw2bgwOperation> pw2bgw = {
        {{{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}}, {0.0, 0.0, 0.0}},
        {{{{1, 0, 0}, {-1, -1, 0}, {0, 0, -1}}}, {0.0, 0.0, -1.0 / 3.0}},
        {{{{0, 1, 0}, {-1, -1, 0}, {0, 0, 1}}}, {0.0, 0.0, -1.0 / 3.0}},
        {{{{-1, -1, 0}, {1, 0, 0}, {0, 0, 1}}}, {0.0, 0.0, 1.0 / 3.0}},
        {{{{-1, -1, 0}, {0, 1, 0}, {0, 0, -1}}}, {0.0, 0.0, 1.0 / 3.0}},
        {{{{0, 1, 0}, {1, 0, 0}, {0, 0, -1}}}, {0.0, 0.0, 0.0}},
    };

    const auto records =
        written_header_records(header_only_record(tellurium, {{0.0, 0.0, 0.0}}), ".wfn_te");
    int32_t number_of_operations = 0;
    std::memcpy(&number_of_operations, records[1].data() + 2 * sizeof(int32_t), sizeof(int32_t));
    ASSERT_EQ(number_of_operations, static_cast<int32_t>(pw2bgw.size()));
    ASSERT_EQ(records[5].size(), 9 * pw2bgw.size() * sizeof(int32_t));
    ASSERT_EQ(records[6].size(), 3 * pw2bgw.size() * sizeof(double));
    const auto* mtrx = reinterpret_cast<const int32_t*>(records[5].data());
    const auto* tnp = reinterpret_cast<const double*>(records[6].data());

    std::vector<bool> matched(pw2bgw.size(), false);
    for (int operation = 0; operation < number_of_operations; operation++) {
        bool found = false;
        for (size_t candidate = 0; candidate < pw2bgw.size() && !found; candidate++) {
            bool same = !matched[candidate];
            for (int j = 0; j < 3 && same; j++)
                for (int k = 0; k < 3 && same; k++)
                    same = mtrx[9 * operation + 3 * j + k] == pw2bgw[candidate].mtrx[k][j];
            for (int component = 0; component < 3 && same; component++)
                same = distance_from_integer(tnp[3 * operation + component] / constants::two_pi -
                                             pw2bgw[candidate].tnp_over_two_pi[component]) < 1e-8;
            if (same)
                matched[candidate] = found = true;
        }
        EXPECT_TRUE(found) << "Q3 operation " << operation
                           << " is not one of pw2bgw's (mtrx, tnp) pairs";
    }
}

TEST(BerkeleyGWCommonHeader, KpointGridFromFullListIsThatGrid) {
    std::vector<Vector3D> grid;
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            for (int l = 0; l < 2; l++)
                grid.push_back({i / 4.0, j / 4.0, l / 2.0});
    const auto [dimensions, shifts] = BerkeleyGWCommonHeader::complete_kpoint_grid(grid);
    EXPECT_EQ(dimensions, (std::array<int32_t, 3>{4, 4, 2}));
    EXPECT_EQ(shifts, (std::array<double, 3>{0.0, 0.0, 0.0}));
}

TEST(BerkeleyGWCommonHeader, KpointGridFromQShiftedListCarriesTheShiftInGridSteps) {
    /**
     * WFNq: the 4x4x2 grid displaced by q0 = (0.001, 0, 0), as printed to ten
     * decimals in an input file. pw2bgw is given wfng_dk1 = 0.004 for it.
     */
    std::vector<Vector3D> grid;
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            for (int l = 0; l < 2; l++)
                grid.push_back({std::round((i / 4.0 + 0.001) * 1e10) / 1e10, j / 4.0, l / 2.0});
    const auto [dimensions, shifts] = BerkeleyGWCommonHeader::complete_kpoint_grid(grid);
    EXPECT_EQ(dimensions, (std::array<int32_t, 3>{4, 4, 2}));
    EXPECT_NEAR(shifts[0], 0.004, 1e-9);
    EXPECT_NEAR(shifts[1], 0.0, 1e-9);
    EXPECT_NEAR(shifts[2], 0.0, 1e-9);
}

TEST(BerkeleyGWCommonHeader, KpointGridFromHalfShiftedListIsHalfShift) {
    std::vector<Vector3D> grid;
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            for (int l = 0; l < 2; l++)
                grid.push_back({(i + 0.5) / 2.0, (j + 0.5) / 2.0, (l + 0.5) / 2.0});
    const auto [dimensions, shifts] = BerkeleyGWCommonHeader::complete_kpoint_grid(grid);
    EXPECT_EQ(dimensions, (std::array<int32_t, 3>{2, 2, 2}));
    EXPECT_EQ(shifts, (std::array<double, 3>{0.5, 0.5, 0.5}));
}

namespace {

/// Diamond silicon (Fd-3m), alat 10 bohr -- the rotations used to unfold a list.
Crystal diamond_silicon() {
    std::vector<AtomSpecies> species = {AtomSpecies{"Si", 28.0855, 4.0, "Si.upf"}};
    std::vector<Atom> atoms = {Atom{0, {0.0, 0.0, 0.0}}, Atom{0, {0.25, 0.25, 0.25}}};
    return Crystal::from_input_data(2, 10.0, species, atoms);
}

/// The coordinates of `grid`'s k-points.
std::vector<Vector3D> coordinates_of(const KPointsGrid& grid) {
    std::vector<Vector3D> coordinates;
    for (int kpoint_index = 0; kpoint_index < grid.number_of_kpoints(); kpoint_index++)
        coordinates.push_back(grid[kpoint_index].coordinates);
    return coordinates;
}

/// The weights of `grid`'s k-points.
std::vector<Real> weights_of(const KPointsGrid& grid) {
    std::vector<Real> weights;
    for (int kpoint_index = 0; kpoint_index < grid.number_of_kpoints(); kpoint_index++)
        weights.push_back(grid[kpoint_index].weight);
    return weights;
}

} // namespace

TEST(BerkeleyGWCommonHeader, KpointGridFromSymmetryReducedListIsTheGridItWasReducedFrom) {
    const Crystal silicon = diamond_silicon();
    const auto operations = SymmetryAnalyzer::get_symmetry_operations(silicon);
    for (const auto& [dimension, shift] : {std::pair{2, 0}, std::pair{4, 0}, std::pair{6, 0}}) {
        KPointsGrid full_grid;
        full_grid.generate(dimension, dimension, dimension, shift, shift, shift);
        const KPointsGrid wedge = SymmetryAnalyzer::reduce_kpoints(silicon, full_grid);
        ASSERT_LT(wedge.number_of_kpoints(), full_grid.number_of_kpoints());
        const auto [dimensions, shifts] = BerkeleyGWCommonHeader::kpoint_grid_from_list(
            coordinates_of(wedge), weights_of(wedge), operations);
        EXPECT_EQ(dimensions, (std::array<int32_t, 3>{dimension, dimension, dimension}))
            << dimension << "^3 grid, shift flag " << shift;
        for (int axis = 0; axis < 3; axis++)
            EXPECT_DOUBLE_EQ(shifts[axis], 0.5 * shift)
                << dimension << "^3 grid, shift flag " << shift << ", axis " << axis;
    }
}

TEST(BerkeleyGWCommonHeader, KpointGridFromHexagonalWedgeIsTheGridItWasReducedFrom) {
    // The wurtzite-ZnO GW case: the 4x4x2 grid of a hexagonal cell.
    std::vector<AtomSpecies> species = {AtomSpecies{"Zn", 65.38, 20.0, "Zn.upf"},
                                        AtomSpecies{"O", 15.999, 6.0, "O.upf"}};
    std::vector<Atom> atoms = {Atom{0, {1.0 / 3.0, 2.0 / 3.0, 0.0}},
                               Atom{0, {2.0 / 3.0, 1.0 / 3.0, 0.5}},
                               Atom{1, {1.0 / 3.0, 2.0 / 3.0, 0.378}},
                               Atom{1, {2.0 / 3.0, 1.0 / 3.0, 0.878}}};
    const Crystal zinc_oxide = Crystal::from_input_data(4, 6.18, species, atoms, 0.0, 1.6166);
    KPointsGrid full_grid;
    full_grid.generate(4, 4, 2, 0, 0, 0);
    const KPointsGrid wedge = SymmetryAnalyzer::reduce_kpoints(zinc_oxide, full_grid);
    ASSERT_LT(wedge.number_of_kpoints(), full_grid.number_of_kpoints());
    const auto [dimensions, shifts] = BerkeleyGWCommonHeader::kpoint_grid_from_list(
        coordinates_of(wedge),
        weights_of(wedge),
        SymmetryAnalyzer::get_symmetry_operations(zinc_oxide));
    EXPECT_EQ(dimensions, (std::array<int32_t, 3>{4, 4, 2}));
    EXPECT_EQ(shifts, (std::array<double, 3>{0.0, 0.0, 0.0}));
}

TEST(BerkeleyGWCommonHeader, KpointGridFromWedgeOfAnAsymmetricGridIsLeftUnset) {
    /**
     * The shifted 2x2x2 grid in FCC's primitive basis is not closed under the
     * point group, so its wedge unfolds off the grid -- and its two points,
     * as given, are a complete 2x1x1 grid, which is not the one they came
     * from. Their unequal weights say so; the result is unset, not that.
     */
    const Crystal silicon = diamond_silicon();
    KPointsGrid full_grid;
    full_grid.generate(2, 2, 2, 1, 1, 1);
    const KPointsGrid wedge = SymmetryAnalyzer::reduce_kpoints(silicon, full_grid);
    const auto [dimensions, shifts] = BerkeleyGWCommonHeader::kpoint_grid_from_list(
        coordinates_of(wedge),
        weights_of(wedge),
        SymmetryAnalyzer::get_symmetry_operations(silicon));
    EXPECT_EQ(dimensions, (std::array<int32_t, 3>{0, 0, 0}));
    EXPECT_EQ(shifts, (std::array<double, 3>{0.0, 0.0, 0.0}));
}

TEST(BerkeleyGWCommonHeader, IncompleteKpointListIsLeftUnset) {
    // Three points of a 4x4x4 grid, with no rotations to complete them.
    const auto [dimensions, shifts] = BerkeleyGWCommonHeader::kpoint_grid_from_list(
        {{0.0, 0.0, 0.0}, {0.25, 0.0, 0.0}, {0.25, 0.5, 0.75}}, {1.0, 1.0, 1.0}, {});
    EXPECT_EQ(dimensions, (std::array<int32_t, 3>{0, 0, 0}));
}

TEST(BerkeleyGWCommonHeader, KpointGridFromQShiftedListIgnoresTheRotationsItBreaks) {
    /**
     * WFNq: a full 4x4x4 list displaced by q0, which the rotations do not
     * preserve -- read as given rather than unfolded into a 1000-division grid.
     */
    const auto operations = SymmetryAnalyzer::get_symmetry_operations(diamond_silicon());
    std::vector<Vector3D> grid;
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            for (int l = 0; l < 4; l++)
                grid.push_back({i / 4.0 + 0.001, j / 4.0, l / 4.0});
    const auto [dimensions, shifts] = BerkeleyGWCommonHeader::kpoint_grid_from_list(
        grid, std::vector<Real>(grid.size(), 1.0), operations);
    EXPECT_EQ(dimensions, (std::array<int32_t, 3>{4, 4, 4}));
    EXPECT_NEAR(shifts[0], 0.004, 1e-9);
    EXPECT_NEAR(shifts[1], 0.0, 1e-9);
    EXPECT_NEAR(shifts[2], 0.0, 1e-9);
}

TEST(BerkeleyGWCommonHeader, KpointGridFromGammaOnlyIsOneByOneByOne) {
    const auto [dimensions, shifts] =
        BerkeleyGWCommonHeader::complete_kpoint_grid({{0.0, 0.0, 0.0}});
    EXPECT_EQ(dimensions, (std::array<int32_t, 3>{1, 1, 1}));
    EXPECT_EQ(shifts, (std::array<double, 3>{0.0, 0.0, 0.0}));
}

TEST(BerkeleyGWCommonHeader, KpointGridFromOffGridListIsLeftUnset) {
    // No grid of up to 1000 divisions holds an irrational coordinate.
    const auto [dimensions, shifts] = BerkeleyGWCommonHeader::complete_kpoint_grid(
        {{0.0, 0.0, 0.0}, {std::sqrt(2.0) / 10.0, 0.0, 0.0}});
    EXPECT_EQ(dimensions, (std::array<int32_t, 3>{0, 0, 0}));
    EXPECT_EQ(shifts, (std::array<double, 3>{0.0, 0.0, 0.0}));
}

TEST(BerkeleyGWCommonHeader, GridRecordRecoversKgridForExplicitKpointList) {
    std::vector<AtomSpecies> species = {AtomSpecies{"Si", 28.0855, 4.0, "Si.upf"}};
    std::vector<Atom> atoms = {Atom{0, {0.0, 0.0, 0.0}}, Atom{0, {0.25, 0.25, 0.25}}};
    Crystal silicon = Crystal::from_input_data(2, 10.0, species, atoms);
    std::vector<Vector3D> shifted_grid;
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            for (int l = 0; l < 3; l++)
                shifted_grid.push_back({i / 2.0 + 0.001, j / 2.0, l / 3.0});
    MeanFieldRecord record = header_only_record(silicon, shifted_grid);
    ASSERT_EQ(record.kpoint_grid_dimensions, (std::array<int, 3>{0, 0, 0}));

    const auto records = written_header_records(record, ".wfn_kgrid_list");
    ASSERT_EQ(records[2].size(), 3 * sizeof(int32_t) + 3 * sizeof(int32_t) + 3 * sizeof(double));
    int32_t kgrid[3];
    double kshift[3];
    std::memcpy(kgrid, records[2].data() + 3 * sizeof(int32_t), sizeof(kgrid));
    std::memcpy(kshift, records[2].data() + 6 * sizeof(int32_t), sizeof(kshift));
    EXPECT_EQ(kgrid[0], 2);
    EXPECT_EQ(kgrid[1], 2);
    EXPECT_EQ(kgrid[2], 3);
    EXPECT_NEAR(kshift[0], 0.002, 1e-12);
    EXPECT_NEAR(kshift[1], 0.0, 1e-12);
    EXPECT_NEAR(kshift[2], 0.0, 1e-12);
}
