#include <gtest/gtest.h>

#include <cmath>
#include <cstring>

#include "io/pseudopotential/upf_reader.hpp"
#include "layer_1_structure/crystal/crystal.hpp"
#include "layer_1_structure/reciprocal_space/kpoints_grid.hpp"
#include "layer_3_representation/basis/planewave/init.hpp"
#include "layer_3_representation/basis/planewave/local_orbital/pseudo_atomic_orbital_set.hpp"
#include "layer_3_representation/quantum_system/kohn_sham/planewave/init.hpp"
#include "layer_4_computation/interpolator/method/wannier/planewave/internal/projections.hpp"
#include "layer_4_computation/iterator/kohn_sham/planewave/execution/serial.hpp"

using namespace q3;

namespace {

execution::PlanewaveSerialExecutionPolicy serial_policy;

/**
 * A deliberately unphysical but well-defined overlap operator, S = scale * I,
 * used only to prove the calculator actually applies whatever operator it is
 * given rather than silently treating every case as S = I.
 */
class ScalingOverlapOperator : public OverlapOperator {
  public:
    explicit ScalingOverlapOperator(Real scale) : scale_(scale) {}

    void apply(int number_of_plane_waves,
               const Complex* wavefunction,
               Complex* overlap_wavefunction) const override {
        for (int g_index = 0; g_index < number_of_plane_waves; g_index++) {
            overlap_wavefunction[g_index] = scale_ * wavefunction[g_index];
        }
    }

    void build_dense_matrix(int number_of_plane_waves,
                            std::vector<Complex>& overlap_matrix) const override {
        overlap_matrix.assign(static_cast<std::size_t>(number_of_plane_waves) *
                                  number_of_plane_waves,
                              Complex(0.0, 0.0));
        for (int i = 0; i < number_of_plane_waves; i++) {
            overlap_matrix[static_cast<std::size_t>(i) * number_of_plane_waves + i] = scale_;
        }
    }

  private:
    Real scale_;
};

PseudopotentialSet load_nickel_pseudopotential() {
    PseudopotentialSet pseudopotential_set;
    pseudopotential_set.push_back(
        UPFReader::read(std::string(EXAMPLES_DIR) + "/data/pseudopotentials/nc/Ni.upf"));
    return pseudopotential_set;
}

Crystal make_nickel_fcc(const PseudopotentialSet& pseudopotential_set) {
    Crystal crystal_structure;
    crystal_structure.set_from_lattice_type(2, 6.65);
    AtomSpecies nickel{"Ni", 58.69, pseudopotential_set[0].valence_charge, "Ni.upf"};
    crystal_structure.add_species(nickel);
    crystal_structure.add_atom({0, {0.0, 0.0, 0.0}});
    crystal_structure.compute_derived_quantities();
    return crystal_structure;
}

struct Fixture {
    PseudopotentialSet pseudopotential_set;
    Crystal crystal_structure;
    PlanewaveBasis basis;
    KPointsGrid kpoint_grid;
    std::vector<int> g_vector_indices;
    int number_of_plane_waves = 0;

    Fixture()
        : pseudopotential_set(load_nickel_pseudopotential())
        , crystal_structure(make_nickel_fcc(pseudopotential_set)) {
        basis.generate(serial_policy, crystal_structure, 30.0, 120.0);
        kpoint_grid.generate(2, 2, 2, 1, 1, 1);
        std::vector<Real> kinetic_energy_squared;
        basis.compute_gk(kpoint_grid[0].coordinates,
                         crystal_structure,
                         g_vector_indices,
                         kinetic_energy_squared);
        number_of_plane_waves = static_cast<int>(g_vector_indices.size());
    }
};

/**
 * Builds a quantum system whose bands are exactly the calculator's own
 * S-metric-orthogonalized trial orbitals -- so the projections of those
 * "converged" bands onto the same trial-orbital set are analytically
 * known (see the two tests below), independent of whatever the S
 * operator happens to be.
 */
PlanewaveKohnShamQuantumSystem build_quantum_system_from_orthogonalized_orbitals(
    const Fixture& fixture, const OverlapOperator* overlap_operator, int& out_number_of_bands) {
    auto atomic_wavefunction_table = tabulate_atomic_wavefunctions_in_q_space(
        fixture.pseudopotential_set, fixture.crystal_structure.cell_volume(), 0.01, 900);
    std::vector<std::vector<Complex>> reference_orbitals;
    build_atomic_orbitals_g_space(0,
                                  fixture.number_of_plane_waves,
                                  fixture.g_vector_indices,
                                  fixture.crystal_structure,
                                  fixture.kpoint_grid,
                                  fixture.basis,
                                  fixture.pseudopotential_set,
                                  atomic_wavefunction_table,
                                  0.01,
                                  900,
                                  reference_orbitals);
    std::vector<Complex> overlap_in_s_metric =
        compute_orbital_overlap_matrix_in_s_metric(reference_orbitals, overlap_operator);
    lowdin_orthogonalize_orbitals(reference_orbitals, overlap_in_s_metric);

    out_number_of_bands = static_cast<int>(reference_orbitals.size());
    PlanewaveKohnShamQuantumSystem quantum_system(fixture.kpoint_grid,
                                                  fixture.basis,
                                                  out_number_of_bands,
                                                  /*number_of_spin_components_value=*/1,
                                                  /*number_of_electrons=*/1.0);
    quantum_system.allocate_electronic_states(
        std::vector<int>(fixture.kpoint_grid.number_of_kpoints(), fixture.number_of_plane_waves));
    for (int band_index = 0; band_index < out_number_of_bands; band_index++) {
        std::memcpy(quantum_system.wavefunction_band_data(0, band_index),
                    reference_orbitals[band_index].data(),
                    fixture.number_of_plane_waves * sizeof(Complex));
    }
    return quantum_system;
}

} // namespace

// ── S = I (norm-conserving): projections form the identity matrix ─────

TEST(PlanewaveTrialOrbitalProjectionCalculator, IdentityOverlapProjectsAsIdentityMatrix) {
    Fixture fixture;
    int number_of_bands = 0;
    PlanewaveKohnShamQuantumSystem quantum_system =
        build_quantum_system_from_orthogonalized_orbitals(fixture, nullptr, number_of_bands);

    PlanewavePseudoAtomicOrbitalSet orbital_set(fixture.crystal_structure,
                                                fixture.pseudopotential_set);
    PlanewaveTrialOrbitalProjectionCalculator calculator(orbital_set);
    ASSERT_EQ(calculator.number_of_orbitals(), number_of_bands);

    auto projections = calculator.compute_projections_at_kpoint(0,
                                                                0,
                                                                fixture.number_of_plane_waves,
                                                                number_of_bands,
                                                                fixture.g_vector_indices,
                                                                fixture.crystal_structure,
                                                                fixture.kpoint_grid,
                                                                fixture.basis,
                                                                quantum_system,
                                                                /*overlap_operator=*/nullptr);

    Real tolerance = 1e-8;
    for (int band_index = 0; band_index < number_of_bands; band_index++) {
        for (int orbital_index = 0; orbital_index < number_of_bands; orbital_index++) {
            Real expected_real = (band_index == orbital_index) ? 1.0 : 0.0;
            EXPECT_NEAR(projections[band_index][orbital_index].real(), expected_real, tolerance);
            EXPECT_NEAR(projections[band_index][orbital_index].imag(), 0.0, tolerance);
        }
    }
}

// ── S = scale * I: the S operator is actually applied, not ignored ────

TEST(PlanewaveTrialOrbitalProjectionCalculator,
     SelfProjectionOfSMetricOrthonormalOrbitalIsExactlyOneRegardlessOfScale) {
    /**
     * The defining property of Löwdin orthogonalization in a metric S is
     * S-orthonormality: <phi'_m|S|phi'_n> = delta_mn, by construction, for
     * ANY valid S (this is exactly the invariant PDWF's projectability
     * relies on -- a state built entirely from the trial manifold must
     * self-project to 1 no matter which physically valid S is used). So
     * feeding the S-metric-orthogonalized orbitals back in as the
     * "converged" bands and projecting them against the same S must give
     * the identity matrix exactly, for S = I as well as S = 4*I.
     */
    Fixture fixture;
    ScalingOverlapOperator overlap_operator(4.0);

    int number_of_bands = 0;
    PlanewaveKohnShamQuantumSystem quantum_system =
        build_quantum_system_from_orthogonalized_orbitals(
            fixture, &overlap_operator, number_of_bands);

    PlanewavePseudoAtomicOrbitalSet orbital_set(fixture.crystal_structure,
                                                fixture.pseudopotential_set);
    PlanewaveTrialOrbitalProjectionCalculator calculator(orbital_set);

    auto projections = calculator.compute_projections_at_kpoint(0,
                                                                0,
                                                                fixture.number_of_plane_waves,
                                                                number_of_bands,
                                                                fixture.g_vector_indices,
                                                                fixture.crystal_structure,
                                                                fixture.kpoint_grid,
                                                                fixture.basis,
                                                                quantum_system,
                                                                &overlap_operator);

    Real tolerance = 1e-8;
    for (int band_index = 0; band_index < number_of_bands; band_index++) {
        for (int orbital_index = 0; orbital_index < number_of_bands; orbital_index++) {
            Real expected_real = (band_index == orbital_index) ? 1.0 : 0.0;
            EXPECT_NEAR(projections[band_index][orbital_index].real(), expected_real, tolerance);
            EXPECT_NEAR(projections[band_index][orbital_index].imag(), 0.0, tolerance);
        }
    }
}

TEST(PlanewaveTrialOrbitalProjectionCalculator,
     PlainMetricWavefunctionProjectsBySqrtScaleOntoSMetricOrbitals) {
    /**
     * A test that genuinely distinguishes "S applied" from "S silently
     * ignored" (the self-projection test above cannot: it gives 1.0
     * either way). Here psi is orthonormal in the PLAIN metric (S = I
     * when the quantum system was built) but the projection itself is
     * computed with S = c*I:
     *
     *   phi'^S_n = (1/sqrt(c)) * phi'^plain_n   (S-metric Lowdin transform
     *              is the plain-metric one scaled by 1/sqrt(c), see the
     *              file-level derivation in
     *              compute_orbital_overlap_matrix_in_s_metric's caller)
     *   A_mn = <phi'^plain_m|S|phi'^S_n>
     *        = c * <phi'^plain_m|phi'^S_n>
     *        = c * (1/sqrt(c)) * <phi'^plain_m|phi'^plain_n>
     *        = sqrt(c) * delta_mn
     *
     * If the implementation silently used S = I regardless of the operator
     * passed to compute_projections_at_kpoint, this would come out as 1.0
     * instead of sqrt(c).
     */
    Fixture fixture;
    Real scale = 4.0;
    ScalingOverlapOperator overlap_operator(scale);

    int number_of_bands = 0;
    PlanewaveKohnShamQuantumSystem quantum_system =
        build_quantum_system_from_orthogonalized_orbitals(
            fixture, /*overlap_operator=*/nullptr, number_of_bands);

    PlanewavePseudoAtomicOrbitalSet orbital_set(fixture.crystal_structure,
                                                fixture.pseudopotential_set);
    PlanewaveTrialOrbitalProjectionCalculator calculator(orbital_set);

    auto projections = calculator.compute_projections_at_kpoint(0,
                                                                0,
                                                                fixture.number_of_plane_waves,
                                                                number_of_bands,
                                                                fixture.g_vector_indices,
                                                                fixture.crystal_structure,
                                                                fixture.kpoint_grid,
                                                                fixture.basis,
                                                                quantum_system,
                                                                &overlap_operator);

    Real expected_diagonal = std::sqrt(scale);
    Real tolerance = 1e-8;
    for (int band_index = 0; band_index < number_of_bands; band_index++) {
        for (int orbital_index = 0; orbital_index < number_of_bands; orbital_index++) {
            Real expected_real = (band_index == orbital_index) ? expected_diagonal : 0.0;
            EXPECT_NEAR(projections[band_index][orbital_index].real(), expected_real, tolerance);
            EXPECT_NEAR(projections[band_index][orbital_index].imag(), 0.0, tolerance);
        }
    }
}

TEST(PlanewaveTrialOrbitalProjectionCalculator, OrbitalsOrthogonalizedInDifferentMetricsDiffer) {
    /**
     * A structural guard against silently ignoring the overlap operator:
     * the S-metric-orthogonalized orbital set (S != I) must differ from
     * the plain-metric one built with S = null.
     */
    Fixture fixture;
    ScalingOverlapOperator overlap_operator(4.0);

    auto atomic_wavefunction_table = tabulate_atomic_wavefunctions_in_q_space(
        fixture.pseudopotential_set, fixture.crystal_structure.cell_volume(), 0.01, 900);

    std::vector<std::vector<Complex>> plain_orbitals;
    build_atomic_orbitals_g_space(0,
                                  fixture.number_of_plane_waves,
                                  fixture.g_vector_indices,
                                  fixture.crystal_structure,
                                  fixture.kpoint_grid,
                                  fixture.basis,
                                  fixture.pseudopotential_set,
                                  atomic_wavefunction_table,
                                  0.01,
                                  900,
                                  plain_orbitals);
    std::vector<std::vector<Complex>> s_metric_orbitals = plain_orbitals;

    lowdin_orthogonalize_orbitals(
        plain_orbitals, compute_orbital_overlap_matrix_in_s_metric(plain_orbitals, nullptr));
    lowdin_orthogonalize_orbitals(
        s_metric_orbitals,
        compute_orbital_overlap_matrix_in_s_metric(s_metric_orbitals, &overlap_operator));

    ASSERT_FALSE(plain_orbitals.empty());
    Real max_difference = 0.0;
    for (std::size_t orbital_index = 0; orbital_index < plain_orbitals.size(); orbital_index++) {
        for (std::size_t g_index = 0; g_index < plain_orbitals[orbital_index].size(); g_index++) {
            max_difference = std::max(max_difference,
                                      std::abs(plain_orbitals[orbital_index][g_index] -
                                               s_metric_orbitals[orbital_index][g_index]));
        }
    }
    EXPECT_GT(max_difference, 1e-3);
}
