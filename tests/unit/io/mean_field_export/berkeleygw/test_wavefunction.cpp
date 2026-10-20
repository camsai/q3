#include <gtest/gtest.h>

#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "io/checkpoint/qe.hpp"
#include "io/mean_field_export/berkeleygw/wavefunction.hpp"
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

/**
 * Diamond-structure (FCC) silicon-like fixture with 6 bands (4 occupied via
 * the default fixed-occupation rule for 8 valence electrons, 2 empty --
 * mirroring the "NSCF with empty bands" way BerkeleyGW is normally used) and
 * fully-populated, individually-normalized synthetic wavefunction
 * coefficients so the coefficient-norm invariant is meaningful.
 */
void populate_synthetic_state(
    Crystal& crystal_structure,
    PlanewaveBasis& basis,
    KPointsGrid& kpoint_grid,
    std::shared_ptr<NormConservingPlanewaveKohnShamQuantumSystem>& quantum_system,
    NormConservingPlanewaveKohnShamSCFState& scf_state,
    int number_of_bands = 6) {
    std::vector<AtomSpecies> species = {AtomSpecies{"Si", 28.0855, 4.0, "Si.upf"}};
    std::vector<Atom> atoms = {Atom{0, {0.875, 0.875, 0.875}}, Atom{0, {0.125, 0.125, 0.125}}};
    crystal_structure = Crystal::from_input_data(2, 10.0, species, atoms); // FCC, alat = 10 bohr

    basis.generate(serial_policy, crystal_structure, 8.0, 32.0);

    kpoint_grid.generate(2, 1, 1, 0, 0, 0);
    basis.initialize_kpoint_data(crystal_structure, kpoint_grid);

    quantum_system = std::make_shared<NormConservingPlanewaveKohnShamQuantumSystem>();
    quantum_system->set_kpoint_grid(&kpoint_grid);
    quantum_system->set_basis(&basis);
    quantum_system->set_number_of_bands(number_of_bands);
    quantum_system->set_number_of_electrons(
        8.0); // 4 occupied bands per number_of_occupied_bands_per_spin()
    quantum_system->number_of_spin_components = 1;
    quantum_system->allocate_electronic_states(basis.kpoint_number_of_plane_waves());

    const auto& plane_waves_per_kpoint = basis.kpoint_number_of_plane_waves();
    int number_of_kpoints = kpoint_grid.number_of_kpoints();
    for (int kpoint_index = 0; kpoint_index < number_of_kpoints; kpoint_index++) {
        int number_of_plane_waves = plane_waves_per_kpoint[kpoint_index];
        for (int band_index = 0; band_index < number_of_bands; band_index++) {
            quantum_system->set_eigenvalue(
                kpoint_index, band_index, 0.1 * band_index + 0.01 * kpoint_index);

            Complex* coefficients =
                quantum_system->wavefunction_band_data(kpoint_index, band_index);
            Real norm_squared = 0.0;
            for (int plane_wave_index = 0; plane_wave_index < number_of_plane_waves;
                 plane_wave_index++) {
                Complex value(1.0 + 0.1 * plane_wave_index + 0.05 * band_index,
                              0.05 * plane_wave_index - 0.02 * band_index);
                coefficients[plane_wave_index] = value;
                norm_squared += std::norm(value);
            }
            Real scale = 1.0 / std::sqrt(norm_squared);
            for (int plane_wave_index = 0; plane_wave_index < number_of_plane_waves;
                 plane_wave_index++) {
                coefficients[plane_wave_index] *= scale;
            }
        }
    }

    scf_state.crystal_structure = &crystal_structure;
    scf_state.kpoint_grid = &kpoint_grid;
    scf_state.planewave_basis = &basis;
    scf_state.quantum_system_snapshot_ = quantum_system;
}

std::string make_temporary_path(const std::string& suffix) {
    return test_support::make_temporary_path(suffix);
}

} // namespace

// ── Physics invariants ───────────────────────────────────────────────────

TEST(BerkeleyGWWavefunctionFile, KpointWeightsSumToOne) {
    Crystal crystal_structure;
    PlanewaveBasis basis;
    KPointsGrid kpoint_grid;
    std::shared_ptr<NormConservingPlanewaveKohnShamQuantumSystem> quantum_system;
    NormConservingPlanewaveKohnShamSCFState scf_state;
    populate_synthetic_state(crystal_structure, basis, kpoint_grid, quantum_system, scf_state);

    std::string path = make_temporary_path(".wfn_weights");
    const MeanFieldRecord record = build_mean_field_record(scf_state);
    BerkeleyGWWavefunctionFile(record).write_to_file(path);

    std::ifstream input_file(path, std::ios::binary);
    QEBinaryRecordFormat record_format;
    std::vector<char> skip(96);
    record_format.read_record(input_file, skip.data(), 96); // title
    std::vector<char> skip2(48);
    record_format.read_record(input_file, skip2.data(), 48); // dimensions
    std::vector<char> skip3(48);
    record_format.read_record(input_file, skip3.data(), 48); // grid
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

    int number_of_kpoints = kpoint_grid.number_of_kpoints();
    std::vector<int32_t> ngk(number_of_kpoints);
    record_format.read_record(
        input_file, ngk.data(), static_cast<int32_t>(ngk.size() * sizeof(int32_t)));
    ASSERT_FALSE(input_file.fail());

    std::vector<double> weights(number_of_kpoints);
    record_format.read_record(
        input_file, weights.data(), static_cast<int32_t>(weights.size() * sizeof(double)));
    ASSERT_FALSE(input_file.fail());

    Real sum = std::accumulate(weights.begin(), weights.end(), 0.0);
    EXPECT_NEAR(sum, 1.0, 1e-10);

    std::remove(path.c_str());
}

TEST(BerkeleyGWWavefunctionFile, BandCoefficientNormsAreApproximatelyOne) {
    Crystal crystal_structure;
    PlanewaveBasis basis;
    KPointsGrid kpoint_grid;
    std::shared_ptr<NormConservingPlanewaveKohnShamQuantumSystem> quantum_system;
    NormConservingPlanewaveKohnShamSCFState scf_state;
    populate_synthetic_state(crystal_structure, basis, kpoint_grid, quantum_system, scf_state);

    std::string path = make_temporary_path(".wfn_norms");
    const MeanFieldRecord record = build_mean_field_record(scf_state);
    BerkeleyGWWavefunctionFile(record).write_to_file(path);

    std::ifstream input_file(path, std::ios::binary);
    QEBinaryRecordFormat record_format;

    // Skip records 1-15 (header + ngk/kw/kpt/ifmin/ifmax/energies/occupations).
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
    const auto symmetry_operations = SymmetryAnalyzer::get_symmetry_operations(crystal_structure);
    int number_of_operations = static_cast<int>(symmetry_operations.size());
    std::vector<char> skip6(9 * number_of_operations * sizeof(int32_t));
    record_format.read_record(input_file, skip6.data(), static_cast<int32_t>(skip6.size()));
    std::vector<char> skip7(3 * number_of_operations * sizeof(double));
    record_format.read_record(input_file, skip7.data(), static_cast<int32_t>(skip7.size()));
    int number_of_atoms = crystal_structure.number_of_atoms();
    std::vector<char> skip8(number_of_atoms * (3 * sizeof(double) + sizeof(int32_t)));
    record_format.read_record(input_file, skip8.data(), static_cast<int32_t>(skip8.size()));

    int number_of_kpoints = kpoint_grid.number_of_kpoints();
    int number_of_bands = quantum_system->number_of_bands();
    std::vector<int32_t> ngk(number_of_kpoints);
    record_format.read_record(
        input_file, ngk.data(), static_cast<int32_t>(ngk.size() * sizeof(int32_t)));
    std::vector<char> skip_weights(number_of_kpoints * sizeof(double));
    record_format.read_record(
        input_file, skip_weights.data(), static_cast<int32_t>(skip_weights.size()));
    std::vector<char> skip_kpts(3 * number_of_kpoints * sizeof(double));
    record_format.read_record(input_file, skip_kpts.data(), static_cast<int32_t>(skip_kpts.size()));
    std::vector<char> skip_ifmin(number_of_kpoints * sizeof(int32_t));
    record_format.read_record(
        input_file, skip_ifmin.data(), static_cast<int32_t>(skip_ifmin.size()));
    std::vector<char> skip_ifmax(number_of_kpoints * sizeof(int32_t));
    record_format.read_record(
        input_file, skip_ifmax.data(), static_cast<int32_t>(skip_ifmax.size()));
    std::vector<char> skip_energies(number_of_bands * number_of_kpoints * sizeof(double));
    record_format.read_record(
        input_file, skip_energies.data(), static_cast<int32_t>(skip_energies.size()));
    std::vector<char> skip_occupations(number_of_bands * number_of_kpoints * sizeof(double));
    record_format.read_record(
        input_file, skip_occupations.data(), static_cast<int32_t>(skip_occupations.size()));
    ASSERT_FALSE(input_file.fail());

    // Records 16-18: nrecord, ng, gvec.
    int32_t nrecord = 0;
    record_format.read_record(input_file, &nrecord, sizeof(nrecord));
    int32_t number_of_g_vectors = 0;
    record_format.read_record(input_file, &number_of_g_vectors, sizeof(number_of_g_vectors));
    std::vector<char> skip_gvec(3 * number_of_g_vectors * sizeof(int32_t));
    record_format.read_record(input_file, skip_gvec.data(), static_cast<int32_t>(skip_gvec.size()));
    ASSERT_FALSE(input_file.fail());

    // Body: per k-point, one gvec_k triple then one triple per band (ns=1).
    for (int kpoint_index = 0; kpoint_index < number_of_kpoints; kpoint_index++) {
        int32_t body_nrecord = 0;
        record_format.read_record(input_file, &body_nrecord, sizeof(body_nrecord));
        ASSERT_FALSE(input_file.fail());
        EXPECT_EQ(body_nrecord, 1);
        int32_t number_of_plane_waves = 0;
        record_format.read_record(
            input_file, &number_of_plane_waves, sizeof(number_of_plane_waves));
        ASSERT_FALSE(input_file.fail());
        EXPECT_EQ(number_of_plane_waves, ngk[kpoint_index]);
        std::vector<int32_t> gvec_k(3 * number_of_plane_waves);
        record_format.read_record(
            input_file, gvec_k.data(), static_cast<int32_t>(gvec_k.size() * sizeof(int32_t)));
        ASSERT_FALSE(input_file.fail());

        for (int band_index = 0; band_index < number_of_bands; band_index++) {
            int32_t band_nrecord = 0;
            record_format.read_record(input_file, &band_nrecord, sizeof(band_nrecord));
            ASSERT_FALSE(input_file.fail());
            EXPECT_EQ(band_nrecord, 1);
            int32_t band_plane_waves = 0;
            record_format.read_record(input_file, &band_plane_waves, sizeof(band_plane_waves));
            ASSERT_FALSE(input_file.fail());
            EXPECT_EQ(band_plane_waves, number_of_plane_waves);

            std::vector<Complex> coefficients(number_of_plane_waves);
            record_format.read_record(input_file,
                                      coefficients.data(),
                                      static_cast<int32_t>(coefficients.size() * sizeof(Complex)));
            ASSERT_FALSE(input_file.fail());

            Real norm_squared = 0.0;
            for (const auto& coefficient : coefficients) {
                norm_squared += std::norm(coefficient);
            }
            EXPECT_NEAR(std::sqrt(norm_squared), 1.0, 1e-9)
                << "kpoint " << kpoint_index << " band " << band_index;
        }
    }

    std::remove(path.c_str());
}

TEST(BerkeleyGWWavefunctionFile, IfmaxConsistentWithFixedOccupations) {
    Crystal crystal_structure;
    PlanewaveBasis basis;
    KPointsGrid kpoint_grid;
    std::shared_ptr<NormConservingPlanewaveKohnShamQuantumSystem> quantum_system;
    NormConservingPlanewaveKohnShamSCFState scf_state;
    populate_synthetic_state(crystal_structure, basis, kpoint_grid, quantum_system, scf_state);

    // 8 valence electrons, unpolarized => number_of_occupied_bands_per_spin() == 4.
    ASSERT_EQ(quantum_system->number_of_occupied_bands_per_spin(), 4);

    std::string path = make_temporary_path(".wfn_ifmax");
    const MeanFieldRecord record = build_mean_field_record(scf_state);
    BerkeleyGWWavefunctionFile(record).write_to_file(path);

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
    const auto symmetry_operations = SymmetryAnalyzer::get_symmetry_operations(crystal_structure);
    int number_of_operations = static_cast<int>(symmetry_operations.size());
    std::vector<char> skip6(9 * number_of_operations * sizeof(int32_t));
    record_format.read_record(input_file, skip6.data(), static_cast<int32_t>(skip6.size()));
    std::vector<char> skip7(3 * number_of_operations * sizeof(double));
    record_format.read_record(input_file, skip7.data(), static_cast<int32_t>(skip7.size()));
    int number_of_atoms = crystal_structure.number_of_atoms();
    std::vector<char> skip8(number_of_atoms * (3 * sizeof(double) + sizeof(int32_t)));
    record_format.read_record(input_file, skip8.data(), static_cast<int32_t>(skip8.size()));

    int number_of_kpoints = kpoint_grid.number_of_kpoints();
    std::vector<char> skip_ngk(number_of_kpoints * sizeof(int32_t));
    record_format.read_record(input_file, skip_ngk.data(), static_cast<int32_t>(skip_ngk.size()));
    std::vector<char> skip_weights(number_of_kpoints * sizeof(double));
    record_format.read_record(
        input_file, skip_weights.data(), static_cast<int32_t>(skip_weights.size()));
    std::vector<char> skip_kpts(3 * number_of_kpoints * sizeof(double));
    record_format.read_record(input_file, skip_kpts.data(), static_cast<int32_t>(skip_kpts.size()));
    ASSERT_FALSE(input_file.fail());

    std::vector<int32_t> ifmin(number_of_kpoints);
    record_format.read_record(
        input_file, ifmin.data(), static_cast<int32_t>(ifmin.size() * sizeof(int32_t)));
    std::vector<int32_t> ifmax(number_of_kpoints);
    record_format.read_record(
        input_file, ifmax.data(), static_cast<int32_t>(ifmax.size() * sizeof(int32_t)));
    ASSERT_FALSE(input_file.fail());

    for (int kpoint_index = 0; kpoint_index < number_of_kpoints; kpoint_index++) {
        EXPECT_EQ(ifmin[kpoint_index], 1);
        EXPECT_EQ(ifmax[kpoint_index], 4);
    }

    std::remove(path.c_str());
}

// ── Scope rejection: spin-polarized WFN is not yet supported ────────────

TEST(BerkeleyGWWavefunctionFile, RejectsSpinPolarizedStateBeforeWritingAnything) {
    Crystal crystal_structure;
    PlanewaveBasis basis;
    KPointsGrid kpoint_grid;
    std::shared_ptr<NormConservingPlanewaveKohnShamQuantumSystem> quantum_system;
    NormConservingPlanewaveKohnShamSCFState scf_state;
    populate_synthetic_state(crystal_structure, basis, kpoint_grid, quantum_system, scf_state);
    quantum_system->number_of_spin_components = 2;
    kpoint_grid.scale_weights(0.5);

    std::string path = make_temporary_path(".wfn_spin_polarized");
    const MeanFieldRecord record = build_mean_field_record(scf_state);
    EXPECT_THROW(BerkeleyGWWavefunctionFile(record).write_to_file(path), std::runtime_error);

    // No partial/corrupt file should be left behind.
    EXPECT_FALSE(std::ifstream(path).good());

    std::remove(path.c_str());
}
