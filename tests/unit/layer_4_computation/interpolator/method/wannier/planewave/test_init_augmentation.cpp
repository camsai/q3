#include <gtest/gtest.h>

#include <array>
#include <cstring>
#include <memory>
#include <vector>

#include "io/pseudopotential/upf_reader.hpp"
#include "layer_0_core/math/linear_algebra/lapack.hpp"
#include "layer_1_structure/crystal/crystal.hpp"
#include "layer_1_structure/reciprocal_space/kpoints_grid.hpp"
#include "layer_3_representation/basis/planewave/init.hpp"
#include "layer_3_representation/basis/planewave/local_orbital/pseudo_atomic_orbital_set.hpp"
#include "layer_3_representation/pseudopotential/planewave/internal/atomic_orbital_builder.hpp"
#include "layer_3_representation/pseudopotential/planewave/internal/lowdin_orthogonalization.hpp"
#include "layer_3_representation/pseudopotential/planewave/internal/projector.hpp"
#include "layer_3_representation/pseudopotential/planewave/internal/wavefunction_atomic.hpp"
#include "layer_3_representation/quantum_system/kohn_sham/planewave/ultrasoft/init.hpp"
#include "layer_4_computation/diagonalizer/planewave/internal/overlap/pseudopotential/factory.hpp"
#include "layer_4_computation/interpolator/method/wannier/planewave/init.hpp"
#include "layer_4_computation/interpolator/method/wannier/planewave/internal/projections.hpp"
#include "layer_4_computation/iterator/kohn_sham/planewave/execution/serial.hpp"

using namespace q3;

namespace {

execution::PlanewaveSerialExecutionPolicy serial_policy;

PseudopotentialSet load_silicon_uspp() {
    PseudopotentialSet pseudopotential_set;
    pseudopotential_set.push_back(
        UPFReader::read(std::string(EXAMPLES_DIR) + "/data/pseudopotentials/us/Si_PBE_USPP.UPF"));
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
 * Same trick as test_init.cpp's Fixture, but S-metric: every band, at every
 * k-point, is that k-point's own S-orthonormalized (<phi_i|S(k)|phi_j> =
 * delta_ij, S(k) built from that k-point's own Kleinman-Bylander
 * projectors) atomic-orbital set -- the ultrasoft/PAW analogue of
 * projections/planewave/test_init.cpp's own identity-projection trick,
 * needed here because build_ultrasoft()'s A_mn(k) applies S, so only
 * S-orthonormal (not plain-orthonormal) synthetic bands project as the
 * identity.
 */
struct UltrasoftFixture {
    PseudopotentialSet pseudopotential_set;
    Crystal crystal_structure;
    PlanewaveBasis basis;
    KPointsGrid kpoint_grid;
    std::vector<std::vector<int>> g_vector_indices_by_kpoint;
    int number_of_bands = 0;
    std::unique_ptr<UltrasoftPlanewaveKohnShamQuantumSystem> quantum_system;

    UltrasoftFixture()
        : pseudopotential_set(load_silicon_uspp())
        , crystal_structure(make_silicon_fcc(pseudopotential_set)) {
        basis.generate(serial_policy, crystal_structure, 24.0, 96.0);
        kpoint_grid.generate(2, 2, 2, 1, 1, 1);

        quantum_system = std::make_unique<UltrasoftPlanewaveKohnShamQuantumSystem>();
        quantum_system->set_kpoint_grid(&kpoint_grid);
        quantum_system->set_basis(&basis);
        quantum_system->init(crystal_structure, basis, pseudopotential_set);
        const UltrasoftPlanewaveBasis& ultrasoft_basis = quantum_system->ultrasoft_basis();

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

            Matrix<Complex> projectors;
            compute_kleinman_bylander_projectors(kpoint_grid[kpoint_index].coordinates,
                                                 crystal_structure,
                                                 basis,
                                                 pseudopotential_set,
                                                 ultrasoft_basis.beta_function_table(),
                                                 g_vector_indices_by_kpoint[kpoint_index],
                                                 plane_waves_per_kpoint[kpoint_index],
                                                 ultrasoft_basis.total_number_of_projectors(),
                                                 projectors);
            std::unique_ptr<OverlapOperator> overlap_operator = create_overlap_operator(
                crystal_structure, *quantum_system, projectors, serial_policy);

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
            std::vector<Complex> overlap_in_s_metric =
                compute_orbital_overlap_matrix_in_s_metric(orbitals, overlap_operator.get());
            lowdin_orthogonalize_orbitals(orbitals, overlap_in_s_metric);
            orbitals_by_kpoint[kpoint_index] = std::move(orbitals);
        }

        number_of_bands = static_cast<int>(orbitals_by_kpoint[0].size());

        quantum_system->set_number_of_bands(number_of_bands);
        quantum_system->set_number_of_electrons(1.0);
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

TEST(PlanewaveWannierCaptureUltrasoft, ProjectionsAreIdentityAtEveryKpoint) {
    UltrasoftFixture fixture;
    PlanewavePseudoAtomicOrbitalSet orbital_set(fixture.crystal_structure,
                                                fixture.pseudopotential_set);
    PlanewaveWannierCapture builder(
        fixture.crystal_structure, fixture.basis, fixture.pseudopotential_set);
    WannierInputs record = builder.build_ultrasoft(fixture.kpoint_grid,
                                                   2,
                                                   2,
                                                   2,
                                                   fixture.number_of_bands,
                                                   fixture.g_vector_indices_by_kpoint,
                                                   *fixture.quantum_system,
                                                   orbital_set,
                                                   serial_policy);

    ASSERT_EQ(record.number_of_bands, fixture.number_of_bands);
    ASSERT_EQ(record.number_of_trial_orbitals, fixture.number_of_bands);

    Real tolerance = 1e-6;
    for (int kpoint_index = 0; kpoint_index < record.number_of_kpoints(); kpoint_index++) {
        const std::vector<Complex>& packed = record.projection_matrices[kpoint_index];
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

TEST(PlanewaveWannierCaptureUltrasoft, EigenvaluesAndFermiEnergyRoundTrip) {
    UltrasoftFixture fixture;
    PlanewavePseudoAtomicOrbitalSet orbital_set(fixture.crystal_structure,
                                                fixture.pseudopotential_set);
    PlanewaveWannierCapture builder(
        fixture.crystal_structure, fixture.basis, fixture.pseudopotential_set);
    WannierInputs record = builder.build_ultrasoft(fixture.kpoint_grid,
                                                   2,
                                                   2,
                                                   2,
                                                   fixture.number_of_bands,
                                                   fixture.g_vector_indices_by_kpoint,
                                                   *fixture.quantum_system,
                                                   orbital_set,
                                                   serial_policy);

    EXPECT_NEAR(record.fermi_energy_rydberg, -1.25, 1e-12);
    for (int kpoint_index = 0; kpoint_index < record.number_of_kpoints(); kpoint_index++) {
        for (int band_index = 0; band_index < fixture.number_of_bands; band_index++) {
            EXPECT_NEAR(record.eigenvalues_rydberg[kpoint_index][band_index],
                        static_cast<Real>(band_index) - 5.0,
                        1e-12);
        }
    }
}

TEST(PlanewaveWannierCaptureUltrasoft, OverlapSingularValuesAreBoundedByOneEverywhere) {
    /**
     * The same general fact build()'s own version of this
     * test checks, now with the augmentation term included: M(k,b) between
     * two S-orthonormal sets (S built independently at each k-point from
     * that k-point's own projectors) is still bounded in operator norm by
     * 1 -- the plan's own stated Phase 1 machine gate on the augmented
     * case specifically ("detects a wrong augmentation normalization").
     *
     * Tolerance is 2e-3, not the near-exact bound the NC version of this
     * test uses, and the reason is physical rather than numerical. The
     * bound sigma <= 1 is a Cauchy-Schwarz statement about one inner
     * product; the augmented overlap M(k,b) = <psi_k|psi_k+b> +
     * sum_ij <psi_k|beta_i> Q_ij(b) e^{-ib.tau} <beta_j|psi_k+b> is that
     * inner product only at b = 0, where Q_ij(0) is the S operator's own
     * q_ij. At finite b the pseudized augmentation function Q_ij(b) is
     * not the Fourier transform of a product of partial waves, so the
     * kernel is no longer a Gram matrix and its singular values may
     * exceed 1 by an amount that grows with |b| (pw2wannier90's matrices
     * carry the same property). Measured on this 2x2x2 fixture: at most
     * 1.2e-3 with the wavevector handed to compute_q_g_single in its
     * 2*pi/alat units, 2.8e-3 when it was handed bohr^-1 by mistake --
     * so the bound stays tight enough to catch that units regression.
     * The kernel's bookkeeping is independently verified exact (1e-10)
     * in test_bloch_overlap_calculator_augmentation.cpp.
     */
    UltrasoftFixture fixture;
    PlanewavePseudoAtomicOrbitalSet orbital_set(fixture.crystal_structure,
                                                fixture.pseudopotential_set);
    PlanewaveWannierCapture builder(
        fixture.crystal_structure, fixture.basis, fixture.pseudopotential_set);
    WannierInputs record = builder.build_ultrasoft(fixture.kpoint_grid,
                                                   2,
                                                   2,
                                                   2,
                                                   fixture.number_of_bands,
                                                   fixture.g_vector_indices_by_kpoint,
                                                   *fixture.quantum_system,
                                                   orbital_set,
                                                   serial_policy);

    for (int kpoint_index = 0; kpoint_index < record.number_of_kpoints(); kpoint_index++) {
        for (int direction_index = 0;
             direction_index < static_cast<int>(record.neighbor_directions.size());
             direction_index++) {
            Real largest =
                largest_singular_value(fixture.number_of_bands,
                                       fixture.number_of_bands,
                                       record.overlap_matrices[kpoint_index][direction_index]);
            EXPECT_LE(largest, 1.0 + 2e-3)
                << "kpoint " << kpoint_index << " direction " << direction_index;
        }
    }
}

TEST(PlanewaveWannierCaptureUltrasoft, OverlapIsHermitianConjugateAcrossNeighborDirections) {
    /**
     * Whether M_mn(k,b) = conj(M_nm(k+b,-b)) survives the augmentation
     * term has not been checked anywhere else in this kernel set --
     * test_bloch_overlap_calculator.cpp's version of this identity is
     * NC-only (S = I, no augmentation active). This is a genuinely new
     * claim: it holds only if compute_q_g_single's augmentation function
     * has the Hermiticity property Q_ij(-b) = conj(Q_ji(b)) that the
     * standard construction assumes.
     */
    UltrasoftFixture fixture;
    PlanewavePseudoAtomicOrbitalSet orbital_set(fixture.crystal_structure,
                                                fixture.pseudopotential_set);
    PlanewaveWannierCapture builder(
        fixture.crystal_structure, fixture.basis, fixture.pseudopotential_set);
    WannierInputs record = builder.build_ultrasoft(fixture.kpoint_grid,
                                                   2,
                                                   2,
                                                   2,
                                                   fixture.number_of_bands,
                                                   fixture.g_vector_indices_by_kpoint,
                                                   *fixture.quantum_system,
                                                   orbital_set,
                                                   serial_policy);

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
                    EXPECT_NEAR(forward_value.real(), backward_value.real(), 1e-6)
                        << "kpoint " << kpoint_index << " direction " << direction_index;
                    EXPECT_NEAR(forward_value.imag(), -backward_value.imag(), 1e-6)
                        << "kpoint " << kpoint_index << " direction " << direction_index;
                }
            }
        }
    }
}
