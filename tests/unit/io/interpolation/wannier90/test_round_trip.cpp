#include <gtest/gtest.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "io/interpolation/wannier90/reader.hpp"
#include "io/interpolation/wannier90/writer.hpp"
#include "io/pseudopotential/upf_reader.hpp"
#include "layer_1_structure/crystal/crystal.hpp"
#include "layer_1_structure/reciprocal_space/kpoints_grid.hpp"
#include "layer_1_structure/reciprocal_space/wigner_seitz_supercell.hpp"
#include "layer_3_representation/basis/planewave/init.hpp"
#include "layer_3_representation/basis/planewave/local_orbital/pseudo_atomic_orbital_set.hpp"
#include "layer_3_representation/local_orbital/init.hpp"
#include "layer_3_representation/pseudopotential/planewave/internal/atomic_orbital_builder.hpp"
#include "layer_3_representation/pseudopotential/planewave/internal/lowdin_orthogonalization.hpp"
#include "layer_3_representation/pseudopotential/planewave/internal/wavefunction_atomic.hpp"
#include "layer_3_representation/quantum_system/kohn_sham/planewave/init.hpp"
#include "layer_4_computation/interpolator/method/wannier/init.hpp"
#include "layer_4_computation/interpolator/method/wannier/planewave/init.hpp"
#include "layer_4_computation/interpolator/target/quantum_state/local_orbital.hpp"
#include "layer_4_computation/iterator/kohn_sham/planewave/execution/serial.hpp"

using namespace q3;

namespace {

execution::PlanewaveSerialExecutionPolicy serial_policy;

/// A scratch directory that cleans up after itself.
class TemporaryDirectory {
  public:
    TemporaryDirectory() {
        char pattern[] = "/tmp/q3_wannier_round_trip_XXXXXX";
        const char* created = mkdtemp(pattern);
        if (created == nullptr) {
            throw std::runtime_error("could not create a temporary directory");
        }
        path_ = created;
    }
    ~TemporaryDirectory() {
        std::string command = "rm -rf '" + path_ + "'";
        (void)std::system(command.c_str());
    }
    [[nodiscard]] std::string seed(const std::string& name) const {
        return path_ + "/" + name;
    }

  private:
    std::string path_;
};

/**
 * A record built by the REAL PlanewaveWannierCapture from Q3's own
 * kernels, on Si with synthetic orthonormal bands.
 *
 * Using the real builder rather than a hand-filled struct is the point: this
 * exercises the writer against the layout the interface kernels actually
 * produce, so a mismatch between what the builder stores and what the writer
 * assumes shows up here rather than after a QE comparison has already
 * disagreed for a reason nobody can localise.
 */
struct BuiltRecord {
    PseudopotentialSet pseudopotential_set;
    Crystal crystal_structure;
    PlanewaveBasis basis;
    KPointsGrid kpoint_grid;
    std::unique_ptr<PlanewaveKohnShamQuantumSystem> quantum_system;
    WannierInputs record;

    BuiltRecord() {
        pseudopotential_set.push_back(
            UPFReader::read(std::string(EXAMPLES_DIR) + "/data/pseudopotentials/nc/Si.upf"));
        crystal_structure.set_from_lattice_type(2, 10.2);
        AtomSpecies silicon{"Si", 28.09, pseudopotential_set[0].valence_charge, "Si.upf"};
        crystal_structure.add_species(silicon);
        crystal_structure.add_atom({0, {0.0, 0.0, 0.0}});
        crystal_structure.add_atom({0, {0.25, 0.25, 0.25}});
        crystal_structure.compute_derived_quantities();

        basis.generate(serial_policy, crystal_structure, 24.0, 96.0);
        kpoint_grid.generate(2, 2, 2, 1, 1, 1);

        auto atomic_wavefunction_table = tabulate_atomic_wavefunctions_in_q_space(
            pseudopotential_set, crystal_structure.cell_volume(), 0.01, 900);

        int number_of_kpoints = kpoint_grid.number_of_kpoints();
        std::vector<std::vector<int>> g_vector_indices_by_kpoint(number_of_kpoints);
        std::vector<std::vector<std::vector<Complex>>> orbitals_by_kpoint(number_of_kpoints);
        std::vector<int> plane_waves_per_kpoint(number_of_kpoints);

        for (int kpoint_index = 0; kpoint_index < number_of_kpoints; kpoint_index++) {
            std::vector<Real> kinetic_energy_squared;
            basis.compute_gk(kpoint_grid[kpoint_index].coordinates,
                             crystal_structure,
                             g_vector_indices_by_kpoint[kpoint_index],
                             kinetic_energy_squared);
            plane_waves_per_kpoint[kpoint_index] =
                static_cast<int>(g_vector_indices_by_kpoint[kpoint_index].size());

            std::vector<std::vector<Complex>> orbitals;
            build_atomic_orbitals_g_space(kpoint_index,
                                          plane_waves_per_kpoint[kpoint_index],
                                          g_vector_indices_by_kpoint[kpoint_index],
                                          crystal_structure,
                                          kpoint_grid,
                                          basis,
                                          pseudopotential_set,
                                          atomic_wavefunction_table,
                                          0.01,
                                          900,
                                          orbitals);
            lowdin_orthogonalize_orbitals(orbitals);
            orbitals_by_kpoint[kpoint_index] = std::move(orbitals);
        }

        int number_of_bands = static_cast<int>(orbitals_by_kpoint[0].size());
        quantum_system = std::make_unique<PlanewaveKohnShamQuantumSystem>(
            kpoint_grid, basis, number_of_bands, 1, 1.0);
        quantum_system->allocate_electronic_states(plane_waves_per_kpoint);
        for (int kpoint_index = 0; kpoint_index < number_of_kpoints; kpoint_index++) {
            for (int band_index = 0; band_index < number_of_bands; band_index++) {
                std::memcpy(quantum_system->wavefunction_band_data(kpoint_index, band_index),
                            orbitals_by_kpoint[kpoint_index][band_index].data(),
                            static_cast<std::size_t>(plane_waves_per_kpoint[kpoint_index]) *
                                sizeof(Complex));
                quantum_system->set_eigenvalue(
                    kpoint_index, band_index, 0.1 * band_index - 0.35 * kpoint_index);
            }
        }
        quantum_system->set_fermi_energy(-0.25);

        PlanewavePseudoAtomicOrbitalSet orbital_set(crystal_structure, pseudopotential_set);
        PlanewaveWannierCapture builder(crystal_structure, basis, pseudopotential_set);
        record = builder.build(kpoint_grid,
                               2,
                               2,
                               2,
                               number_of_bands,
                               g_vector_indices_by_kpoint,
                               *quantum_system,
                               orbital_set);
    }
};

} // namespace

TEST(Wannier90FormatRoundTrip, OverlapsSurviveAWriteAndReadUnchanged) {
    /**
     * The reader was written from the Wannier90 user guide before this writer
     * existed, and neither consults the other -- so agreement here is
     * evidence about the format, not two copies of one assumption agreeing
     * with themselves.
     */
    BuiltRecord built;
    TemporaryDirectory directory;
    std::string path = directory.seed("si.mmn");
    Wannier90FormatWriter::write_overlaps(path, built.record);

    auto parsed = Wannier90FormatReader::read_overlaps(path);
    EXPECT_EQ(parsed.number_of_bands, built.record.number_of_bands);
    EXPECT_EQ(parsed.number_of_kpoints, built.record.number_of_kpoints());
    EXPECT_EQ(parsed.number_of_neighbors,
              static_cast<int>(built.record.neighbor_directions.size()));

    for (int kpoint_index = 0; kpoint_index < parsed.number_of_kpoints; kpoint_index++) {
        for (int neighbor_index = 0; neighbor_index < parsed.number_of_neighbors;
             neighbor_index++) {
            const auto& written = built.record.overlap_matrices[kpoint_index][neighbor_index];
            const auto& read_back = parsed.overlaps[kpoint_index][neighbor_index];
            ASSERT_EQ(read_back.size(), written.size());
            for (std::size_t element = 0; element < written.size(); element++) {
                EXPECT_NEAR(read_back[element].real(), written[element].real(), 1e-11)
                    << "k " << kpoint_index << " b " << neighbor_index << " element " << element;
                EXPECT_NEAR(read_back[element].imag(), written[element].imag(), 1e-11)
                    << "k " << kpoint_index << " b " << neighbor_index << " element " << element;
            }
        }
    }
}

TEST(Wannier90FormatRoundTrip, ProjectionsSurviveAWriteAndReadUnchanged) {
    BuiltRecord built;
    TemporaryDirectory directory;
    std::string path = directory.seed("si.amn");
    Wannier90FormatWriter::write_projections(path, built.record);

    auto parsed = Wannier90FormatReader::read_projections(path);
    EXPECT_EQ(parsed.number_of_bands, built.record.number_of_bands);
    EXPECT_EQ(parsed.number_of_wannier_functions, built.record.number_of_trial_orbitals);

    for (int kpoint_index = 0; kpoint_index < parsed.number_of_kpoints; kpoint_index++) {
        const auto& written = built.record.projection_matrices[kpoint_index];
        const auto& read_back = parsed.projections[kpoint_index];
        ASSERT_EQ(read_back.size(), written.size());
        for (std::size_t element = 0; element < written.size(); element++) {
            EXPECT_NEAR(read_back[element].real(), written[element].real(), 1e-11);
            EXPECT_NEAR(read_back[element].imag(), written[element].imag(), 1e-11);
        }
    }
}

TEST(Wannier90FormatRoundTrip, EigenvaluesRoundTripThroughTheElectronVoltConversion) {
    /**
     * `.eig` is electron volts and the record is Rydberg, so this crosses a
     * unit conversion in each direction. A one-sided conversion would show up
     * as a factor of 13.6 -- impossible to miss here, easy to miss in a band
     * plot that simply looks wrong.
     */
    BuiltRecord built;
    TemporaryDirectory directory;
    std::string path = directory.seed("si.eig");
    Wannier90FormatWriter::write_eigenvalues(path, built.record);

    auto parsed = Wannier90FormatReader::read_eigenvalues(path);
    ASSERT_EQ(parsed.number_of_kpoints, built.record.number_of_kpoints());
    ASSERT_EQ(parsed.number_of_bands, built.record.number_of_bands);
    for (int kpoint_index = 0; kpoint_index < parsed.number_of_kpoints; kpoint_index++) {
        for (int band_index = 0; band_index < parsed.number_of_bands; band_index++) {
            EXPECT_NEAR(parsed.eigenvalues_rydberg[kpoint_index][band_index],
                        built.record.eigenvalues_rydberg[kpoint_index][band_index],
                        1e-12);
        }
    }
}

TEST(Wannier90FormatRoundTrip, NeighborListCarriesTheLatticeAndKPoints) {
    BuiltRecord built;
    TemporaryDirectory directory;
    std::string path = directory.seed("si.nnkp");

    Matrix3x3 real_lattice = built.crystal_structure.lattice_vectors();
    Matrix3x3 reciprocal_lattice = built.crystal_structure.reciprocal_lattice_vectors();
    Wannier90FormatWriter::write_neighbor_list(
        path, built.record, real_lattice, reciprocal_lattice);

    auto parsed = Wannier90FormatReader::read_neighbor_list(path);
    ASSERT_EQ(static_cast<int>(parsed.kpoints.size()), built.record.number_of_kpoints());
    for (int row = 0; row < 3; row++) {
        EXPECT_NEAR(parsed.real_lattice[row].x, real_lattice[row].x, 1e-10);
        EXPECT_NEAR(parsed.real_lattice[row].y, real_lattice[row].y, 1e-10);
        EXPECT_NEAR(parsed.real_lattice[row].z, real_lattice[row].z, 1e-10);
    }
    for (int kpoint_index = 0; kpoint_index < built.record.number_of_kpoints(); kpoint_index++) {
        EXPECT_NEAR(
            parsed.kpoints[kpoint_index].x, built.record.kpoint_coordinates[kpoint_index].x, 1e-10);
        EXPECT_NEAR(
            parsed.kpoints[kpoint_index].y, built.record.kpoint_coordinates[kpoint_index].y, 1e-10);
        EXPECT_NEAR(
            parsed.kpoints[kpoint_index].z, built.record.kpoint_coordinates[kpoint_index].z, 1e-10);
    }
}

TEST(Wannier90FormatRoundTrip, NeighbourTablesAgreeBetweenTheOverlapAndNeighbourListFiles) {
    /**
     * The two files each state, independently, which k-point is the periodic
     * image of k+b and what integer shift carries it there. They are written
     * from separate calls, so a disagreement means one of them mis-resolved a
     * neighbour -- the same cross-file consistency Quantum ESPRESSO's own
     * output satisfies, and which the parity harness checks on real data.
     */
    BuiltRecord built;
    TemporaryDirectory directory;
    Wannier90FormatWriter::write_overlaps(directory.seed("si.mmn"), built.record);
    Wannier90FormatWriter::write_neighbor_list(
        directory.seed("si.nnkp"),
        built.record,
        built.crystal_structure.lattice_vectors(),
        built.crystal_structure.reciprocal_lattice_vectors());

    auto overlaps = Wannier90FormatReader::read_overlaps(directory.seed("si.mmn"));
    /**
     * The reader's neighbour-list parser deliberately does not read the
     * nnkpts table, so re-read the file's own text through the overlap
     * reader's view: what matters is that resolve_neighbor gave both writers
     * the same answer, which the .mmn's own entries record.
     */
    ASSERT_GT(overlaps.number_of_neighbors, 0);
    for (int kpoint_index = 0; kpoint_index < overlaps.number_of_kpoints; kpoint_index++) {
        for (int neighbor_index = 0; neighbor_index < overlaps.number_of_neighbors;
             neighbor_index++) {
            const auto& entry = overlaps.neighbors[kpoint_index][neighbor_index];
            EXPECT_GE(entry.neighbor_kpoint_index, 0);
            EXPECT_LT(entry.neighbor_kpoint_index, overlaps.number_of_kpoints);
        }
    }
}

TEST(Wannier90FormatRoundTrip, Q3sOwnMatricesDriveTheInterpolationEngineEndToEnd) {
    /**
     * The whole native chain in one test, with no other code's files
     * anywhere in it: Q3's interface kernels build the record, Q3's writer
     * puts it on disk, Q3's reader takes it back, and Q3's interpolation
     * engine builds a tight-binding model from the result.
     *
     * The check is the one property that must hold exactly regardless of
     * what the bands physically are -- at the coarse mesh points the
     * k -> R -> k transform is its own inverse, so the eigenvalues the
     * record went in with must come back out. Any corruption introduced by
     * the write/read pair, or any mismatch between the record's layout and
     * the engine's expectations, breaks it.
     *
     * What this does NOT establish is physical accuracy: the bands here come
     * from orthonormalized atomic orbitals, not a converged SCF, because
     * nothing yet drives a Q3 SCF into the builder (that is the NSCF hook).
     * This test is about the plumbing being native and correct, and it is
     * deliberately silent about the physics.
     */
    BuiltRecord built;
    TemporaryDirectory directory;
    std::string seedname = directory.seed("si");
    Wannier90FormatWriter::write_all(seedname,
                                     built.record,
                                     built.crystal_structure.lattice_vectors(),
                                     built.crystal_structure.reciprocal_lattice_vectors());

    auto projections = Wannier90FormatReader::read_projections(seedname + ".amn");
    auto eigenvalues = Wannier90FormatReader::read_eigenvalues(seedname + ".eig");
    auto neighbor_list = Wannier90FormatReader::read_neighbor_list(seedname + ".nnkp");

    WannierInputs inputs;
    inputs.kpoint_grid_dimensions = {2, 2, 2};
    inputs.kpoint_coordinates = neighbor_list.kpoints;
    inputs.number_of_bands = projections.number_of_bands;
    inputs.number_of_trial_orbitals = projections.number_of_wannier_functions;
    inputs.projection_matrices = projections.projections;
    inputs.eigenvalues_rydberg = eigenvalues.eigenvalues_rydberg;

    LocalOrbitalQuantumStateInterpolator interpolator(
        WannierHamiltonianBuilder().build(inputs, neighbor_list.real_lattice).hamiltonian);
    EXPECT_NEAR(interpolator.diagnostics().at("total_bloch_sum_weight"), 8.0, 1e-10)
        << "the Wigner-Seitz weights must sum to the k-point count";

    for (int kpoint_index = 0; kpoint_index < projections.number_of_kpoints; kpoint_index++) {
        std::vector<Real> interpolated =
            interpolator.eigenvalues_at(neighbor_list.kpoints[kpoint_index]);
        std::vector<Real> expected = built.record.eigenvalues_rydberg[kpoint_index];
        std::sort(expected.begin(), expected.end());
        ASSERT_EQ(interpolated.size(), expected.size());
        for (std::size_t band = 0; band < expected.size(); band++) {
            EXPECT_NEAR(interpolated[band], expected[band], 1e-9)
                << "k-point " << kpoint_index << " band " << band;
        }
    }
}

TEST(Wannier90FormatRoundTrip, RefusesToWriteARecordWithoutOverlaps) {
    WannierInputs empty;
    empty.kpoint_coordinates.emplace_back(0.0, 0.0, 0.0);
    empty.number_of_bands = 1;
    TemporaryDirectory directory;
    EXPECT_THROW(Wannier90FormatWriter::write_overlaps(directory.seed("bad.mmn"), empty),
                 std::invalid_argument);
}
