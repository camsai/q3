#include <gtest/gtest.h>

#include <cstring>
#include <memory>
#include <vector>

#include "io/pseudopotential/upf_reader.hpp"
#include "layer_0_core/math/linear_algebra/lapack.hpp"
#include "layer_1_structure/crystal/crystal.hpp"
#include "layer_1_structure/reciprocal_space/kpoints_grid.hpp"
#include "layer_3_representation/basis/planewave/init.hpp"
#include "layer_3_representation/basis/planewave/local_orbital/hydrogenic_orbital_set.hpp"
#include "layer_3_representation/basis/planewave/local_orbital/pseudo_atomic_orbital_set.hpp"
#include "layer_3_representation/pseudopotential/planewave/internal/atomic_orbital_builder.hpp"
#include "layer_3_representation/pseudopotential/planewave/internal/lowdin_orthogonalization.hpp"
#include "layer_3_representation/pseudopotential/planewave/internal/wavefunction_atomic.hpp"
#include "layer_3_representation/quantum_system/kohn_sham/planewave/init.hpp"
#include "layer_4_computation/interpolator/method/wannier/planewave/init.hpp"
#include "layer_4_computation/iterator/kohn_sham/planewave/execution/serial.hpp"

using namespace q3;

namespace {

execution::PlanewaveSerialExecutionPolicy serial_policy;

PseudopotentialSet load_silicon_pseudopotential() {
    PseudopotentialSet pseudopotential_set;
    pseudopotential_set.push_back(
        UPFReader::read(std::string(EXAMPLES_DIR) + "/data/pseudopotentials/nc/Si.upf"));
    return pseudopotential_set;
}

Crystal make_silicon_fcc(const PseudopotentialSet& pseudopotential_set) {
    Crystal crystal_structure;
    crystal_structure.set_from_lattice_type(2, 10.2);
    AtomSpecies silicon{"Si", 28.09, pseudopotential_set[0].valence_charge, "Si.upf"};
    crystal_structure.add_species(silicon);
    crystal_structure.add_atom({0, {0.0, 0.0, 0.0}});
    crystal_structure.add_atom({0, {0.25, 0.25, 0.25}});
    crystal_structure.compute_derived_quantities();
    return crystal_structure;
}

/**
 * Every band, at every k-point, is that k-point's own Löwdin-orthonormalized
 * (plain-metric -- Si NC, so S = I) atomic-orbital set: the same trick
 * projections/planewave/test_init.cpp's KnownOrbitalsProjectAsIdentityMatrix
 * uses at one k-point, extended across a full k-point grid so the
 * orchestrator's own per-kpoint loop and record indexing get exercised, not
 * just the kernels it calls (those already have their own unit tests).
 */
struct Fixture {
    PseudopotentialSet pseudopotential_set;
    Crystal crystal_structure;
    PlanewaveBasis basis;
    KPointsGrid kpoint_grid;
    std::vector<std::vector<int>> g_vector_indices_by_kpoint;
    int number_of_bands = 0;
    std::unique_ptr<PlanewaveKohnShamQuantumSystem> quantum_system;

    Fixture()
        : pseudopotential_set(load_silicon_pseudopotential())
        , crystal_structure(make_silicon_fcc(pseudopotential_set)) {
        basis.generate(serial_policy, crystal_structure, 24.0, 96.0);
        kpoint_grid.generate(2, 2, 2, 1, 1, 1);

        auto atomic_wavefunction_table = tabulate_atomic_wavefunctions_in_q_space(
            pseudopotential_set, crystal_structure.cell_volume(), 0.01, 900);

        int number_of_kpoints = kpoint_grid.number_of_kpoints();
        g_vector_indices_by_kpoint.resize(number_of_kpoints);
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

        number_of_bands = static_cast<int>(orbitals_by_kpoint[0].size());

        quantum_system =
            std::make_unique<PlanewaveKohnShamQuantumSystem>(kpoint_grid,
                                                             basis,
                                                             number_of_bands,
                                                             /*number_of_spin_components_value=*/1,
                                                             /*number_of_electrons=*/1.0);
        quantum_system->allocate_electronic_states(plane_waves_per_kpoint);

        for (int kpoint_index = 0; kpoint_index < number_of_kpoints; kpoint_index++) {
            for (int band_index = 0; band_index < number_of_bands; band_index++) {
                std::memcpy(quantum_system->wavefunction_band_data(kpoint_index, band_index),
                            orbitals_by_kpoint[kpoint_index][band_index].data(),
                            static_cast<std::size_t>(plane_waves_per_kpoint[kpoint_index]) *
                                sizeof(Complex));
                quantum_system->set_eigenvalue(
                    kpoint_index, band_index, static_cast<Real>(band_index) - 5.0);
            }
        }
        quantum_system->set_fermi_energy(-1.25);
    }
};

Complex overlap_entry(const std::vector<Complex>& values,
                      int number_of_bands,
                      int band_at_kpoint,
                      int band_at_neighbor) {
    return values[static_cast<std::size_t>(band_at_neighbor) * number_of_bands + band_at_kpoint];
}

int find_opposite_direction(const std::vector<KPointNeighborDirection>& directions,
                            int direction_index) {
    const IntegerVector3D& step = directions[direction_index].grid_step;
    for (int candidate = 0; candidate < static_cast<int>(directions.size()); candidate++) {
        const IntegerVector3D& candidate_step = directions[candidate].grid_step;
        if (candidate_step.x == -step.x && candidate_step.y == -step.y &&
            candidate_step.z == -step.z) {
            return candidate;
        }
    }
    return -1;
}

/// Largest singular value of a dense, column-major complex matrix, via ZGESVD.
Real largest_singular_value(int rows, int columns, const std::vector<Complex>& values) {
    std::vector<Complex> matrix_copy = values;
    std::vector<Real> singular_values(std::min(rows, columns));
    int workspace_size = -1;
    Complex optimal_workspace_size;
    std::vector<Real> real_workspace(5 * static_cast<std::size_t>(std::min(rows, columns)));
    int info = 0;
    int rows_copy = rows;
    int columns_copy = columns;

    lapack::zgesvd_("N",
                    "N",
                    &rows_copy,
                    &columns_copy,
                    matrix_copy.data(),
                    &rows_copy,
                    singular_values.data(),
                    nullptr,
                    &rows_copy,
                    nullptr,
                    &columns_copy,
                    &optimal_workspace_size,
                    &workspace_size,
                    real_workspace.data(),
                    &info);
    workspace_size = static_cast<int>(optimal_workspace_size.real());
    std::vector<Complex> workspace(workspace_size);
    matrix_copy = values;

    lapack::zgesvd_("N",
                    "N",
                    &rows_copy,
                    &columns_copy,
                    matrix_copy.data(),
                    &rows_copy,
                    singular_values.data(),
                    nullptr,
                    &rows_copy,
                    nullptr,
                    &columns_copy,
                    workspace.data(),
                    &workspace_size,
                    real_workspace.data(),
                    &info);
    return singular_values.empty() ? 0.0 : singular_values[0];
}

} // namespace

TEST(PlanewaveWannierCapture, RecordCarriesKpointGridAndNeighborShellsThrough) {
    Fixture fixture;
    PlanewavePseudoAtomicOrbitalSet orbital_set(fixture.crystal_structure,
                                                fixture.pseudopotential_set);
    PlanewaveWannierCapture builder(
        fixture.crystal_structure, fixture.basis, fixture.pseudopotential_set);
    WannierInputs record = builder.build(fixture.kpoint_grid,
                                         2,
                                         2,
                                         2,
                                         fixture.number_of_bands,
                                         fixture.g_vector_indices_by_kpoint,
                                         *fixture.quantum_system,
                                         orbital_set);

    EXPECT_EQ(record.kpoint_grid_dimensions[0], 2);
    EXPECT_EQ(record.kpoint_grid_dimensions[1], 2);
    EXPECT_EQ(record.kpoint_grid_dimensions[2], 2);
    ASSERT_EQ(record.number_of_kpoints(), fixture.kpoint_grid.number_of_kpoints());
    for (int kpoint_index = 0; kpoint_index < record.number_of_kpoints(); kpoint_index++) {
        const Vector3D& expected = fixture.kpoint_grid[kpoint_index].coordinates;
        const Vector3D& actual = record.kpoint_coordinates[kpoint_index];
        EXPECT_NEAR(actual.x, expected.x, 1e-12);
        EXPECT_NEAR(actual.y, expected.y, 1e-12);
        EXPECT_NEAR(actual.z, expected.z, 1e-12);
    }

    ASSERT_FALSE(record.neighbor_directions.empty());
    std::array<Real, 6> tensor_sum = {0, 0, 0, 0, 0, 0};
    for (const KPointNeighborDirection& direction : record.neighbor_directions) {
        const Vector3D& b = direction.reciprocal_lattice_translation;
        tensor_sum[0] += direction.weight * b.x * b.x;
        tensor_sum[1] += direction.weight * b.y * b.y;
        tensor_sum[2] += direction.weight * b.z * b.z;
        tensor_sum[3] += direction.weight * b.x * b.y;
        tensor_sum[4] += direction.weight * b.x * b.z;
        tensor_sum[5] += direction.weight * b.y * b.z;
    }
    EXPECT_NEAR(tensor_sum[0], 1.0, 1e-8);
    EXPECT_NEAR(tensor_sum[1], 1.0, 1e-8);
    EXPECT_NEAR(tensor_sum[2], 1.0, 1e-8);
    EXPECT_NEAR(tensor_sum[3], 0.0, 1e-8);
    EXPECT_NEAR(tensor_sum[4], 0.0, 1e-8);
    EXPECT_NEAR(tensor_sum[5], 0.0, 1e-8);
}

TEST(PlanewaveWannierCapture, ProjectionsAreIdentityAtEveryKpoint) {
    Fixture fixture;
    PlanewavePseudoAtomicOrbitalSet orbital_set(fixture.crystal_structure,
                                                fixture.pseudopotential_set);
    PlanewaveWannierCapture builder(
        fixture.crystal_structure, fixture.basis, fixture.pseudopotential_set);
    WannierInputs record = builder.build(fixture.kpoint_grid,
                                         2,
                                         2,
                                         2,
                                         fixture.number_of_bands,
                                         fixture.g_vector_indices_by_kpoint,
                                         *fixture.quantum_system,
                                         orbital_set);

    ASSERT_EQ(record.number_of_bands, fixture.number_of_bands);
    ASSERT_EQ(record.number_of_trial_orbitals, fixture.number_of_bands);
    EXPECT_EQ(record.number_of_spin_components, 1);

    Real tolerance = 1e-8;
    for (int kpoint_index = 0; kpoint_index < record.number_of_kpoints(); kpoint_index++) {
        const std::vector<Complex>& packed = record.projection_matrices[kpoint_index];
        ASSERT_EQ(static_cast<int>(packed.size()),
                  fixture.number_of_bands * record.number_of_trial_orbitals);
        for (int band_index = 0; band_index < fixture.number_of_bands; band_index++) {
            for (int orbital_index = 0; orbital_index < record.number_of_trial_orbitals;
                 orbital_index++) {
                Complex value =
                    packed[static_cast<std::size_t>(band_index) +
                           static_cast<std::size_t>(orbital_index) * fixture.number_of_bands];
                Real expected = (band_index == orbital_index) ? 1.0 : 0.0;
                EXPECT_NEAR(value.real(), expected, tolerance)
                    << "kpoint " << kpoint_index << " band " << band_index << " orbital "
                    << orbital_index;
                EXPECT_NEAR(value.imag(), 0.0, tolerance)
                    << "kpoint " << kpoint_index << " band " << band_index << " orbital "
                    << orbital_index;
            }
            EXPECT_NEAR(record.projectabilities[kpoint_index][band_index], 1.0, tolerance)
                << "kpoint " << kpoint_index << " band " << band_index;
        }
    }
}

TEST(PlanewaveWannierCapture, EigenvaluesAndFermiEnergyRoundTrip) {
    Fixture fixture;
    PlanewavePseudoAtomicOrbitalSet orbital_set(fixture.crystal_structure,
                                                fixture.pseudopotential_set);
    PlanewaveWannierCapture builder(
        fixture.crystal_structure, fixture.basis, fixture.pseudopotential_set);
    WannierInputs record = builder.build(fixture.kpoint_grid,
                                         2,
                                         2,
                                         2,
                                         fixture.number_of_bands,
                                         fixture.g_vector_indices_by_kpoint,
                                         *fixture.quantum_system,
                                         orbital_set);

    EXPECT_NEAR(record.fermi_energy_rydberg, -1.25, 1e-12);
    for (int kpoint_index = 0; kpoint_index < record.number_of_kpoints(); kpoint_index++) {
        for (int band_index = 0; band_index < fixture.number_of_bands; band_index++) {
            EXPECT_NEAR(record.eigenvalues_rydberg[kpoint_index][band_index],
                        static_cast<Real>(band_index) - 5.0,
                        1e-12);
        }
    }
}

TEST(PlanewaveWannierCapture, OverlapSingularValuesAreBoundedByOneEverywhere) {
    Fixture fixture;
    PlanewavePseudoAtomicOrbitalSet orbital_set(fixture.crystal_structure,
                                                fixture.pseudopotential_set);
    PlanewaveWannierCapture builder(
        fixture.crystal_structure, fixture.basis, fixture.pseudopotential_set);
    WannierInputs record = builder.build(fixture.kpoint_grid,
                                         2,
                                         2,
                                         2,
                                         fixture.number_of_bands,
                                         fixture.g_vector_indices_by_kpoint,
                                         *fixture.quantum_system,
                                         orbital_set);

    for (int kpoint_index = 0; kpoint_index < record.number_of_kpoints(); kpoint_index++) {
        ASSERT_EQ(static_cast<int>(record.overlap_matrices[kpoint_index].size()),
                  static_cast<int>(record.neighbor_directions.size()));
        for (int direction_index = 0;
             direction_index < static_cast<int>(record.neighbor_directions.size());
             direction_index++) {
            Real largest =
                largest_singular_value(fixture.number_of_bands,
                                       fixture.number_of_bands,
                                       record.overlap_matrices[kpoint_index][direction_index]);
            EXPECT_LE(largest, 1.0 + 1e-8)
                << "kpoint " << kpoint_index << " direction " << direction_index;
        }
    }
}

TEST(PlanewaveWannierCapture, OverlapIsHermitianConjugateAcrossNeighborDirections) {
    Fixture fixture;
    PlanewavePseudoAtomicOrbitalSet orbital_set(fixture.crystal_structure,
                                                fixture.pseudopotential_set);
    PlanewaveWannierCapture builder(
        fixture.crystal_structure, fixture.basis, fixture.pseudopotential_set);
    WannierInputs record = builder.build(fixture.kpoint_grid,
                                         2,
                                         2,
                                         2,
                                         fixture.number_of_bands,
                                         fixture.g_vector_indices_by_kpoint,
                                         *fixture.quantum_system,
                                         orbital_set);

    /**
     * This checks the ORCHESTRATOR's own bookkeeping -- that it resolved
     * each (k, direction) pair's neighbor correctly and stored the result
     * at the matching (neighbor, opposite_direction) slot -- not just the
     * kernel underneath, which already has its own isolated version of
     * this identity (test_bloch_overlap_calculator.cpp's
     * MIsHermitianConjugateUnderKPlusBAndMinusB). A wrong neighbor
     * resolution or a mismatched direction index in the record's own
     * loop would break this even if the kernel call itself were correct.
     */
    int number_of_directions = static_cast<int>(record.neighbor_directions.size());
    for (int kpoint_index = 0; kpoint_index < record.number_of_kpoints(); kpoint_index++) {
        for (int direction_index = 0; direction_index < number_of_directions; direction_index++) {
            int opposite_index =
                find_opposite_direction(record.neighbor_directions, direction_index);
            ASSERT_GE(opposite_index, 0);

            ResolvedKPointNeighbor resolved = KPointNeighborShells::resolve_neighbor(
                kpoint_index,
                record.neighbor_directions[direction_index].grid_step,
                record.kpoint_grid_dimensions,
                fixture.kpoint_grid.kpoints());

            const std::vector<Complex>& forward =
                record.overlap_matrices[kpoint_index][direction_index];
            const std::vector<Complex>& backward =
                record.overlap_matrices[resolved.neighbor_kpoint_index][opposite_index];

            for (int band_at_kpoint = 0; band_at_kpoint < fixture.number_of_bands;
                 band_at_kpoint++) {
                for (int band_at_neighbor = 0; band_at_neighbor < fixture.number_of_bands;
                     band_at_neighbor++) {
                    Complex forward_value = overlap_entry(
                        forward, fixture.number_of_bands, band_at_kpoint, band_at_neighbor);
                    Complex backward_value = overlap_entry(
                        backward, fixture.number_of_bands, band_at_neighbor, band_at_kpoint);
                    EXPECT_NEAR(forward_value.real(), backward_value.real(), 1e-8)
                        << "kpoint " << kpoint_index << " direction " << direction_index;
                    EXPECT_NEAR(forward_value.imag(), -backward_value.imag(), 1e-8)
                        << "kpoint " << kpoint_index << " direction " << direction_index;
                }
            }
        }
    }
}

// ── A band range that does not start at band 1 ────────────────────────────────

TEST(PlanewaveWannierCapture, ARangeReadsTheBandsItNames) {
    /**
     * The fixture's eigenvalue at band b is b - 5, so the band a record
     * carries is checkable by its value alone. Capturing two bands from an
     * offset of two must yield the third and fourth bands, not the first two.
     *
     * This is what a semicore-carrying pseudopotential needs: the manifold
     * worth modelling starts above the states the dataset puts underneath it.
     */
    Fixture fixture;
    ASSERT_GE(fixture.number_of_bands, 4) << "the fixture must have room for an offset range";
    PlanewaveHydrogenicOrbitalSet orbital_set(
        {{Vector3D(0.0, 0.0, 0.0), 1.0}, {Vector3D(0.5, 0.0, 0.0), 1.0}});
    PlanewaveWannierCapture builder(
        fixture.crystal_structure, fixture.basis, fixture.pseudopotential_set);

    WannierInputs record = builder.build(fixture.kpoint_grid,
                                         2,
                                         2,
                                         2,
                                         2,
                                         fixture.g_vector_indices_by_kpoint,
                                         *fixture.quantum_system,
                                         orbital_set,
                                         /*first_band_index=*/2);

    ASSERT_EQ(record.number_of_bands, 2);
    for (int kpoint_index = 0; kpoint_index < record.number_of_kpoints(); kpoint_index++) {
        EXPECT_NEAR(record.eigenvalues_rydberg[kpoint_index][0], 2.0 - 5.0, 1e-12)
            << "k-point " << kpoint_index << " must carry band 3, not band 1";
        EXPECT_NEAR(record.eigenvalues_rydberg[kpoint_index][1], 3.0 - 5.0, 1e-12);
    }
}

TEST(PlanewaveWannierCapture, ARangeStartingAtOneMatchesNoRangeAtAll) {
    /// The offset is a generalisation, so its default must change nothing.
    Fixture fixture;
    PlanewaveHydrogenicOrbitalSet orbital_set(
        {{Vector3D(0.0, 0.0, 0.0), 1.0}, {Vector3D(0.5, 0.0, 0.0), 1.0}});
    PlanewaveWannierCapture builder(
        fixture.crystal_structure, fixture.basis, fixture.pseudopotential_set);

    WannierInputs implicit = builder.build(fixture.kpoint_grid,
                                           2,
                                           2,
                                           2,
                                           2,
                                           fixture.g_vector_indices_by_kpoint,
                                           *fixture.quantum_system,
                                           orbital_set);
    WannierInputs explicit_range = builder.build(fixture.kpoint_grid,
                                                 2,
                                                 2,
                                                 2,
                                                 2,
                                                 fixture.g_vector_indices_by_kpoint,
                                                 *fixture.quantum_system,
                                                 orbital_set,
                                                 /*first_band_index=*/0);

    ASSERT_EQ(implicit.projection_matrices.size(), explicit_range.projection_matrices.size());
    for (std::size_t kpoint = 0; kpoint < implicit.projection_matrices.size(); kpoint++) {
        ASSERT_EQ(implicit.projection_matrices[kpoint].size(),
                  explicit_range.projection_matrices[kpoint].size());
        for (std::size_t element = 0; element < implicit.projection_matrices[kpoint].size();
             element++) {
            EXPECT_EQ(implicit.projection_matrices[kpoint][element],
                      explicit_range.projection_matrices[kpoint][element]);
        }
        EXPECT_EQ(implicit.eigenvalues_rydberg[kpoint], explicit_range.eigenvalues_rydberg[kpoint]);
    }
}

TEST(PlanewaveWannierCapture, ARangeBeyondTheComputedBandsIsRefused) {
    /**
     * Asking for bands the run never computed must say so, rather than read
     * past the end of the quantum system's band storage.
     */
    Fixture fixture;
    PlanewaveHydrogenicOrbitalSet orbital_set({{Vector3D(0.0, 0.0, 0.0), 1.0}});
    PlanewaveWannierCapture builder(
        fixture.crystal_structure, fixture.basis, fixture.pseudopotential_set);

    EXPECT_THROW(builder.build(fixture.kpoint_grid,
                               2,
                               2,
                               2,
                               1,
                               fixture.g_vector_indices_by_kpoint,
                               *fixture.quantum_system,
                               orbital_set,
                               /*first_band_index=*/fixture.number_of_bands),
                 std::invalid_argument);
}
