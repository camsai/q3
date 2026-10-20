#include <gtest/gtest.h>

#include <complex>
#include <vector>

#include "io/pseudopotential/upf_reader.hpp"
#include "layer_0_core/math/linear_algebra/lapack.hpp"
#include "layer_1_structure/crystal/crystal.hpp"
#include "layer_1_structure/reciprocal_space/kpoint_neighbor_shells.hpp"
#include "layer_1_structure/reciprocal_space/kpoints_grid.hpp"
#include "layer_3_representation/basis/planewave/init.hpp"
#include "layer_3_representation/pseudopotential/planewave/internal/atomic_orbital_builder.hpp"
#include "layer_3_representation/pseudopotential/planewave/internal/lowdin_orthogonalization.hpp"
#include "layer_3_representation/pseudopotential/planewave/internal/wavefunction_atomic.hpp"
#include "layer_4_computation/interpolator/method/wannier/planewave/internal/overlaps.hpp"
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
 * Löwdin-orthogonalized (plain metric) atomic orbitals at one k-point --
 * used as synthetic, physically meaningful "bands" so the overlap kernel
 * can be tested without a converged SCF: any two orthonormal sets have an
 * overlap matrix bounded in operator norm by 1, which is exactly the
 * plan's Phase 1 machine gate on M(k,b).
 */
std::vector<std::vector<Complex>> build_orthonormal_orbitals_at_kpoint(
    int kpoint_index,
    const std::vector<int>& g_vector_indices,
    const Crystal& crystal_structure,
    const KPointsGrid& kpoint_grid,
    const PlanewaveBasis& planewave_basis,
    const PseudopotentialSet& pseudopotential_set,
    const std::vector<std::vector<std::vector<Real>>>& atomic_wavefunction_table) {
    std::vector<std::vector<Complex>> orbitals;
    build_atomic_orbitals_g_space(kpoint_index,
                                  static_cast<int>(g_vector_indices.size()),
                                  g_vector_indices,
                                  crystal_structure,
                                  kpoint_grid,
                                  planewave_basis,
                                  pseudopotential_set,
                                  atomic_wavefunction_table,
                                  0.01,
                                  900,
                                  orbitals);
    lowdin_orthogonalize_orbitals(orbitals);
    return orbitals;
}

struct Fixture {
    PseudopotentialSet pseudopotential_set;
    Crystal crystal_structure;
    PlanewaveBasis basis;
    KPointsGrid kpoint_grid;
    MillerIndexLookup global_lookup;
    std::vector<std::vector<std::vector<Real>>> atomic_wavefunction_table;

    Fixture()
        : pseudopotential_set(load_silicon_pseudopotential())
        , crystal_structure(make_silicon_fcc(pseudopotential_set)) {
        basis.generate(serial_policy, crystal_structure, 24.0, 96.0);
        kpoint_grid.generate(4, 4, 4, 0, 0, 0);
        global_lookup.build(basis.g_vectors());
        atomic_wavefunction_table = tabulate_atomic_wavefunctions_in_q_space(
            pseudopotential_set, crystal_structure.cell_volume(), 0.01, 900);
    }

    std::vector<std::vector<Complex>> orbitals_at(int kpoint_index,
                                                  std::vector<int>& out_g_vector_indices) const {
        std::vector<Real> kinetic_energy_squared;
        basis.compute_gk(kpoint_grid[kpoint_index].coordinates,
                         crystal_structure,
                         out_g_vector_indices,
                         kinetic_energy_squared);
        return build_orthonormal_orbitals_at_kpoint(kpoint_index,
                                                    out_g_vector_indices,
                                                    crystal_structure,
                                                    kpoint_grid,
                                                    basis,
                                                    pseudopotential_set,
                                                    atomic_wavefunction_table);
    }
};

/// Largest singular value of a dense complex matrix, via ZGESVD.
Real largest_singular_value(const BlochOverlapResult& result) {
    int rows = result.number_of_bands_at_kpoint;
    int columns = result.number_of_bands_at_neighbor;
    std::vector<Complex> matrix_copy = result.values;
    std::vector<Real> singular_values(std::min(rows, columns));
    int workspace_size = -1;
    Complex optimal_workspace_size;
    std::vector<Real> real_workspace(5 * static_cast<std::size_t>(std::min(rows, columns)));
    int info = 0;

    lapack::zgesvd_("N",
                    "N",
                    &rows,
                    &columns,
                    matrix_copy.data(),
                    &rows,
                    singular_values.data(),
                    nullptr,
                    &rows,
                    nullptr,
                    &columns,
                    &optimal_workspace_size,
                    &workspace_size,
                    real_workspace.data(),
                    &info);
    workspace_size = static_cast<int>(optimal_workspace_size.real());
    std::vector<Complex> workspace(workspace_size);
    matrix_copy = result.values;

    lapack::zgesvd_("N",
                    "N",
                    &rows,
                    &columns,
                    matrix_copy.data(),
                    &rows,
                    singular_values.data(),
                    nullptr,
                    &rows,
                    nullptr,
                    &columns,
                    workspace.data(),
                    &workspace_size,
                    real_workspace.data(),
                    &info);
    return singular_values.empty() ? 0.0 : singular_values[0];
}

} // namespace

/**
 * ── Synthetic plumbing test: a single-G-vector "band" matches only ────
 * ── the shifted Miller triple it should, nothing else ─────────────────
 */

TEST(PlanewaveBlochOverlapCalculator, SingleGVectorBandMatchesOnlyItsShiftedCounterpart) {
    Fixture fixture;
    std::vector<int> g_vector_indices;
    std::vector<Real> kinetic_energy_squared;
    fixture.basis.compute_gk(fixture.kpoint_grid[0].coordinates,
                             fixture.crystal_structure,
                             g_vector_indices,
                             kinetic_energy_squared);
    int number_of_plane_waves = static_cast<int>(g_vector_indices.size());

    /**
     * Band 0: a delta function at the k=Gamma origin G-vector (0,0,0),
     * guaranteed present (it is the lowest-energy plane wave at Gamma).
     */
    int origin_local_position = -1;
    for (int g_index = 0; g_index < number_of_plane_waves; g_index++) {
        const IntegerVector3D& miller = fixture.basis.g_vectors()[g_vector_indices[g_index]].miller;
        if (miller.x == 0 && miller.y == 0 && miller.z == 0) {
            origin_local_position = g_index;
            break;
        }
    }
    ASSERT_GE(origin_local_position, 0);

    std::vector<Complex> band_at_kpoint(number_of_plane_waves, Complex(0.0, 0.0));
    band_at_kpoint[origin_local_position] = Complex(1.0, 0.0);
    /**
     * Same list used as "neighbor" here (shift = 0) -- this test is about
     * the matching logic in isolation, not about a real k-grid neighbor
     * (that combination is exercised by the end-to-end tests below).
     */
    std::vector<Complex> band_at_neighbor = band_at_kpoint;

    PlanewaveBlochOverlapCalculator calculator(fixture.basis, fixture.global_lookup);
    BlochOverlapResult result = calculator.compute(
        g_vector_indices,
        1,
        [&](int) { return band_at_kpoint.data(); },
        g_vector_indices,
        1,
        [&](int) { return band_at_neighbor.data(); },
        IntegerVector3D(0, 0, 0));

    EXPECT_NEAR(result(0, 0).real(), 1.0, 1e-12);
    EXPECT_NEAR(result(0, 0).imag(), 0.0, 1e-12);
    EXPECT_EQ(result.missing_g_vector_count, 0);
}

TEST(PlanewaveBlochOverlapCalculator, MismatchedShiftGivesZeroOverlap) {
    Fixture fixture;
    std::vector<int> g_vector_indices;
    std::vector<Real> kinetic_energy_squared;
    fixture.basis.compute_gk(fixture.kpoint_grid[0].coordinates,
                             fixture.crystal_structure,
                             g_vector_indices,
                             kinetic_energy_squared);
    int number_of_plane_waves = static_cast<int>(g_vector_indices.size());

    int origin_local_position = -1;
    for (int g_index = 0; g_index < number_of_plane_waves; g_index++) {
        const IntegerVector3D& miller = fixture.basis.g_vectors()[g_vector_indices[g_index]].miller;
        if (miller.x == 0 && miller.y == 0 && miller.z == 0) {
            origin_local_position = g_index;
            break;
        }
    }
    ASSERT_GE(origin_local_position, 0);

    std::vector<Complex> band_at_kpoint(number_of_plane_waves, Complex(0.0, 0.0));
    band_at_kpoint[origin_local_position] = Complex(1.0, 0.0);
    std::vector<Complex> band_at_neighbor = band_at_kpoint;

    /**
     * A shift with no corresponding G-vector in this cutoff sphere at all
     * (far outside any reasonable ecutrho): every one of the k-side plane
     * waves' shifted Miller triples misses (the match is computed once
     * per (k,b) pair over the whole G-set, independent of which band is
     * later queried against it), so the overlap of this delta-function
     * band is exactly zero, and every miss is counted rather than
     * silently ignored.
     */
    PlanewaveBlochOverlapCalculator calculator(fixture.basis, fixture.global_lookup);
    BlochOverlapResult result = calculator.compute(
        g_vector_indices,
        1,
        [&](int) { return band_at_kpoint.data(); },
        g_vector_indices,
        1,
        [&](int) { return band_at_neighbor.data(); },
        IntegerVector3D(500, 500, 500));

    EXPECT_NEAR(result(0, 0).real(), 0.0, 1e-12);
    EXPECT_NEAR(result(0, 0).imag(), 0.0, 1e-12);
    EXPECT_EQ(result.missing_g_vector_count, number_of_plane_waves);
}

// ── Physically meaningful gates on real (synthetic-band) data ─────────

TEST(PlanewaveBlochOverlapCalculator, SingularValuesOfOrthonormalSetOverlapAreBoundedByOne) {
    Fixture fixture;
    KPointNeighborShells shells;
    Matrix3x3 reciprocal_lattice_vectors = fixture.crystal_structure.reciprocal_lattice_vectors();
    shells.compute(reciprocal_lattice_vectors, 4, 4, 4);
    ASSERT_FALSE(shells.neighbor_directions().empty());

    const std::array<int, 3> grid_dimensions = {4, 4, 4};
    int starting_index = 0;
    const KPointNeighborDirection& direction = shells.neighbor_directions()[0];
    ResolvedKPointNeighbor resolved = KPointNeighborShells::resolve_neighbor(
        starting_index, direction.grid_step, grid_dimensions, fixture.kpoint_grid.kpoints());

    std::vector<int> g_vector_indices_at_kpoint;
    auto orbitals_at_kpoint = fixture.orbitals_at(starting_index, g_vector_indices_at_kpoint);
    std::vector<int> g_vector_indices_at_neighbor;
    auto orbitals_at_neighbor =
        fixture.orbitals_at(resolved.neighbor_kpoint_index, g_vector_indices_at_neighbor);

    PlanewaveBlochOverlapCalculator calculator(fixture.basis, fixture.global_lookup);
    BlochOverlapResult result = calculator.compute(
        g_vector_indices_at_kpoint,
        static_cast<int>(orbitals_at_kpoint.size()),
        [&](int band_index) { return orbitals_at_kpoint[band_index].data(); },
        g_vector_indices_at_neighbor,
        static_cast<int>(orbitals_at_neighbor.size()),
        [&](int band_index) { return orbitals_at_neighbor[band_index].data(); },
        resolved.periodic_image_shift);

    EXPECT_LE(largest_singular_value(result), 1.0 + 1e-8);
}

TEST(PlanewaveBlochOverlapCalculator, MIsHermitianConjugateUnderKPlusBAndMinusB) {
    /**
     * M_mn(k,b) must equal conj(M_nm(k+b,-b)): the same physical overlap,
     * computed starting from the other endpoint with the opposite
     * direction. Every shell is closed under negation
     * (KPointNeighborShells), so -direction.grid_step is itself one of
     * the shell's own directions.
     */
    Fixture fixture;
    KPointNeighborShells shells;
    Matrix3x3 reciprocal_lattice_vectors = fixture.crystal_structure.reciprocal_lattice_vectors();
    shells.compute(reciprocal_lattice_vectors, 4, 4, 4);
    const std::array<int, 3> grid_dimensions = {4, 4, 4};
    int starting_index = 5; // an interior point, not Gamma

    const KPointNeighborDirection& direction = shells.neighbor_directions()[0];
    ResolvedKPointNeighbor forward = KPointNeighborShells::resolve_neighbor(
        starting_index, direction.grid_step, grid_dimensions, fixture.kpoint_grid.kpoints());

    IntegerVector3D negated_step(
        -direction.grid_step.x, -direction.grid_step.y, -direction.grid_step.z);
    ResolvedKPointNeighbor backward =
        KPointNeighborShells::resolve_neighbor(forward.neighbor_kpoint_index,
                                               negated_step,
                                               grid_dimensions,
                                               fixture.kpoint_grid.kpoints());
    // Stepping by direction then by its negation must return to the start.
    ASSERT_EQ(backward.neighbor_kpoint_index, starting_index);

    std::vector<int> g_vector_indices_at_start;
    auto orbitals_at_start = fixture.orbitals_at(starting_index, g_vector_indices_at_start);
    std::vector<int> g_vector_indices_at_forward;
    auto orbitals_at_forward =
        fixture.orbitals_at(forward.neighbor_kpoint_index, g_vector_indices_at_forward);

    PlanewaveBlochOverlapCalculator calculator(fixture.basis, fixture.global_lookup);

    BlochOverlapResult forward_result = calculator.compute(
        g_vector_indices_at_start,
        static_cast<int>(orbitals_at_start.size()),
        [&](int band_index) { return orbitals_at_start[band_index].data(); },
        g_vector_indices_at_forward,
        static_cast<int>(orbitals_at_forward.size()),
        [&](int band_index) { return orbitals_at_forward[band_index].data(); },
        forward.periodic_image_shift);

    BlochOverlapResult backward_result = calculator.compute(
        g_vector_indices_at_forward,
        static_cast<int>(orbitals_at_forward.size()),
        [&](int band_index) { return orbitals_at_forward[band_index].data(); },
        g_vector_indices_at_start,
        static_cast<int>(orbitals_at_start.size()),
        [&](int band_index) { return orbitals_at_start[band_index].data(); },
        backward.periodic_image_shift);

    Real tolerance = 1e-8;
    for (int m = 0; m < forward_result.number_of_bands_at_kpoint; m++) {
        for (int n = 0; n < forward_result.number_of_bands_at_neighbor; n++) {
            Complex forward_value = forward_result(m, n);
            Complex backward_value = std::conj(backward_result(n, m));
            EXPECT_NEAR(forward_value.real(), backward_value.real(), tolerance)
                << "m=" << m << " n=" << n;
            EXPECT_NEAR(forward_value.imag(), backward_value.imag(), tolerance)
                << "m=" << m << " n=" << n;
        }
    }
}
