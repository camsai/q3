#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "layer_0_core/types/init.hpp"
#include "layer_1_structure/reciprocal_space/kpoints_grid.hpp"
#include "layer_3_representation/quantum_system/kohn_sham/planewave/init.hpp"
#include "layer_4_computation/fermi_energy/planewave/init.hpp"
#include "layer_4_computation/iterator/internal/smearing/fixed.hpp"
#include "layer_4_computation/iterator/internal/smearing/gaussian.hpp"
#include "layer_4_computation/iterator/internal/smearing/marzari_vanderbilt.hpp"
#include "layer_4_computation/iterator/internal/smearing/methfessel_paxton.hpp"
#include "layer_4_computation/iterator/kohn_sham/planewave/execution/serial.hpp"

using namespace q3;

// ── Helper to build a minimal quantum system for testing ────────────

namespace {

/**
 * Build a PlanewaveKohnShamQuantumSystem with trivial wavefunctions
 * (zero-sized plane-wave coefficients) for Fermi energy / occupation testing.
 *
 * The system has the specified number of k-points (generated via a 1D MP grid),
 * bands, spin components, and electrons. Eigenvalues default to 0.0.
 */
class TestableQuantumSystem : public PlanewaveKohnShamQuantumSystem {
  public:
    TestableQuantumSystem(const KPointsGrid& kpoint_grid,
                          int number_of_bands,
                          int number_of_spin_components_value,
                          Real number_of_electrons) {
        kpoint_grid_ = &kpoint_grid;
        basis_ = nullptr;
        number_of_bands_ = number_of_bands;
        number_of_electrons_ = number_of_electrons;
        this->number_of_spin_components = number_of_spin_components_value;
    }
};

static TestableQuantumSystem build_test_quantum_system(KPointsGrid& kpoint_grid,
                                                       int number_of_kpoints_x,
                                                       int number_of_bands,
                                                       int number_of_spin_components,
                                                       Real number_of_electrons) {
    // Generate a simple Monkhorst-Pack grid to get the right number of k-points
    kpoint_grid.generate(number_of_kpoints_x, 1, 1, 0, 0, 0);

    TestableQuantumSystem quantum_system(
        kpoint_grid, number_of_bands, number_of_spin_components, number_of_electrons);

    // Allocate electronic states with 0 plane waves per k-point (no wavefunction data needed)
    std::vector<int> number_of_plane_waves_per_kpoint(kpoint_grid.number_of_kpoints(), 0);
    quantum_system.allocate_electronic_states(number_of_plane_waves_per_kpoint);

    return quantum_system;
}

// ── Test: Fixed occupation for 2-band insulator (Si-like) ───────────

} // namespace

TEST(PlanewaveFermiEnergyCalculator, FixedOccupationInsulatorTwoBands) {
    KPointsGrid kpoint_grid;
    auto quantum_system = build_test_quantum_system(kpoint_grid,
                                                    /*number_of_kpoints_x=*/1,
                                                    /*number_of_bands=*/2,
                                                    /*number_of_spin_components=*/1,
                                                    /*number_of_electrons=*/2.0);

    // Set known eigenvalues: HOMO = -5.0 Ry, LUMO = 5.0 Ry
    quantum_system.set_eigenvalue(0, 0, -5.0); // band 0 (occupied)
    quantum_system.set_eigenvalue(0, 1, 5.0);  // band 1 (unoccupied)

    FixedOccupation fixed_occupation;
    PlanewaveFermiEnergyCalculator calculator(quantum_system, &fixed_occupation);

    EXPECT_FALSE(calculator.uses_smearing());

    Real fermi_energy = calculator.calculate_occupations();

    // Fermi energy should be midpoint of HOMO-LUMO gap
    EXPECT_NEAR(fermi_energy, 0.0, 1e-10);
    EXPECT_NEAR(quantum_system.fermi_energy(), 0.0, 1e-10);

    // Occupations: band 0 occupied, band 1 empty
    EXPECT_NEAR(quantum_system.occupation_by_kpoint_state_index_and_band_index(0, 0), 1.0, 1e-10);
    EXPECT_NEAR(quantum_system.occupation_by_kpoint_state_index_and_band_index(0, 1), 0.0, 1e-10);
}

// ── Test: Fixed occupation with multiple k-points ───────────────────

TEST(PlanewaveFermiEnergyCalculator, FixedOccupationMultipleKpoints) {
    KPointsGrid kpoint_grid;
    auto quantum_system = build_test_quantum_system(kpoint_grid,
                                                    /*number_of_kpoints_x=*/2,
                                                    /*number_of_bands=*/2,
                                                    /*number_of_spin_components=*/1,
                                                    /*number_of_electrons=*/2.0);

    // Different eigenvalues at each k-point to test HOMO/LUMO finding across k-points
    quantum_system.set_eigenvalue(0, 0, -6.0); // k0, band 0
    quantum_system.set_eigenvalue(0, 1, 4.0);  // k0, band 1
    quantum_system.set_eigenvalue(1, 0, -4.0); // k1, band 0
    quantum_system.set_eigenvalue(1, 1, 6.0);  // k1, band 1

    FixedOccupation fixed_occupation;
    PlanewaveFermiEnergyCalculator calculator(quantum_system, &fixed_occupation);

    Real fermi_energy = calculator.calculate_occupations();

    /**
     * HOMO is max of occupied bands across all k-points = max(-6.0, -4.0) = -4.0
     * LUMO is min of unoccupied bands across all k-points = min(4.0, 6.0) = 4.0
     * Fermi energy = midpoint = 0.0
     */
    EXPECT_NEAR(fermi_energy, 0.0, 1e-10);
}

// ── Test: Gaussian smearing occupation ──────────────────────────────

/**
 * A band path is not a Brillouin-zone sampling, and a smeared Fermi energy
 * bisected over one is not a Fermi energy. Every point carries weight 1, so the
 * electron count comes out by a factor of the path length at every trial
 * energy, the bisection never reaches its target, and it converges to its own
 * lower bracket -- the smallest eigenvalue in the run. Measured on copper
 * before this guard: -7.046 Ry, below the lowest band the run had just
 * computed, where the SCF on the same density gives 1.243 Ry.
 */
TEST(PlanewaveFermiEnergyCalculator, ABandPathCannotDetermineAFermiEnergyAndDoesNotInventOne) {
    KPointsGrid kpoint_grid;
    auto quantum_system = build_test_quantum_system(kpoint_grid,
                                                    /*number_of_kpoints_x=*/1,
                                                    /*number_of_bands=*/2,
                                                    /*number_of_spin_components=*/1,
                                                    /*number_of_electrons=*/2.0);
    // Weight 1.0 per point, exactly as q3.workflows.band_structure writes a path.
    kpoint_grid.scale_weights(0.5);

    quantum_system.set_eigenvalue(0, 0, -0.1);
    quantum_system.set_eigenvalue(0, 1, 0.1);
    quantum_system.set_fermi_energy(0.0);

    GaussianSmearing gaussian_smearing(0.05);
    PlanewaveFermiEnergyCalculator calculator(quantum_system, &gaussian_smearing);

    const Real fermi_energy = calculator.calculate_occupations();

    EXPECT_EQ(fermi_energy, 0.0) << "the untouched value, not one bisected from a path";
    EXPECT_GE(fermi_energy, -0.1) << "never below the lowest band, which is the failure mode";
}

/// The ordinary grid still works: the guard must not fire on a real sampling.
TEST(PlanewaveFermiEnergyCalculator, AGeneratedGridIsAcceptedAsASampling) {
    KPointsGrid kpoint_grid;
    auto quantum_system = build_test_quantum_system(kpoint_grid,
                                                    /*number_of_kpoints_x=*/2,
                                                    /*number_of_bands=*/2,
                                                    /*number_of_spin_components=*/1,
                                                    /*number_of_electrons=*/2.0);
    for (int kpoint = 0; kpoint < 2; kpoint++) {
        quantum_system.set_eigenvalue(kpoint, 0, -0.1);
        quantum_system.set_eigenvalue(kpoint, 1, 0.1);
    }

    GaussianSmearing gaussian_smearing(0.05);
    PlanewaveFermiEnergyCalculator calculator(quantum_system, &gaussian_smearing);

    const Real fermi_energy = calculator.calculate_occupations();

    EXPECT_GT(fermi_energy, -0.1);
    EXPECT_LT(fermi_energy, 0.1);
}

TEST(PlanewaveFermiEnergyCalculator, GaussianSmearingOccupation) {
    KPointsGrid kpoint_grid;
    auto quantum_system = build_test_quantum_system(kpoint_grid,
                                                    /*number_of_kpoints_x=*/1,
                                                    /*number_of_bands=*/2,
                                                    /*number_of_spin_components=*/1,
                                                    /*number_of_electrons=*/2.0);

    // Small gap: eigenvalues at -0.1 and 0.1
    quantum_system.set_eigenvalue(0, 0, -0.1);
    quantum_system.set_eigenvalue(0, 1, 0.1);

    GaussianSmearing gaussian_smearing(0.05); // smearing width = 0.05 Ry
    PlanewaveFermiEnergyCalculator calculator(quantum_system, &gaussian_smearing);

    EXPECT_TRUE(calculator.uses_smearing());

    Real fermi_energy = calculator.calculate_occupations();

    // Fermi energy should be between the two eigenvalues (near 0.0 by symmetry)
    EXPECT_GT(fermi_energy, -0.1);
    EXPECT_LT(fermi_energy, 0.1);
    EXPECT_NEAR(fermi_energy, 0.0, 0.02);

    /**
     * Total occupation should equal number of electrons (2.0)
     * With spin degeneracy = 2.0, each state's occupation is f(x) (0 to 1 range)
     * Total electrons = sum over k-points of weight * spin_degeneracy/2 * occupation
     * For 1 k-point with weight=2.0 and spin_degeneracy=2.0:
     *   total = 2.0 * (2.0/2.0) * (occ_0 + occ_1)
     *         = 2.0 * (occ_0 + occ_1)
     * So occ_0 + occ_1 should equal 1.0
     */
    Real occupation_band_0 = quantum_system.occupation_by_kpoint_state_index_and_band_index(0, 0);
    Real occupation_band_1 = quantum_system.occupation_by_kpoint_state_index_and_band_index(0, 1);
    EXPECT_NEAR(occupation_band_0 + occupation_band_1, 1.0, 1e-6);

    // Band 0 should be more occupied than band 1
    EXPECT_GT(occupation_band_0, occupation_band_1);
}

// ── Test: Accessor methods ──────────────────────────────────────────

TEST(PlanewaveFermiEnergyCalculator, UsesSmearingAccessor) {
    // Without smearing (fixed occupation)
    FixedOccupation fixed_occupation;
    PlanewaveFermiEnergyCalculator fixed_calculator;
    KPointsGrid kpoint_grid;
    auto quantum_system = build_test_quantum_system(kpoint_grid, 1, 2, 1, 2.0);
    quantum_system.set_eigenvalue(0, 0, -1.0);
    quantum_system.set_eigenvalue(0, 1, 1.0);

    PlanewaveFermiEnergyCalculator calculator_fixed(quantum_system, &fixed_occupation);
    EXPECT_FALSE(calculator_fixed.uses_smearing());
    EXPECT_EQ(calculator_fixed.smearing(), &fixed_occupation);

    // With Gaussian smearing
    GaussianSmearing gaussian_smearing(0.1);
    PlanewaveFermiEnergyCalculator calculator_gaussian(quantum_system, &gaussian_smearing);
    EXPECT_TRUE(calculator_gaussian.uses_smearing());
    EXPECT_EQ(calculator_gaussian.smearing(), &gaussian_smearing);
}

// ── Test: Null smearing pointer ─────────────────────────────────────

TEST(PlanewaveFermiEnergyCalculator, NullSmearingDoesNotUseSmearing) {
    PlanewaveFermiEnergyCalculator calculator;
    EXPECT_FALSE(calculator.uses_smearing());
    EXPECT_EQ(calculator.smearing(), nullptr);
}

// ── Test: Default constructor ───────────────────────────────────────

TEST(PlanewaveFermiEnergyCalculator, DefaultConstructor) {
    PlanewaveFermiEnergyCalculator calculator;
    EXPECT_FALSE(calculator.uses_smearing());
    EXPECT_EQ(calculator.smearing(), nullptr);
}

// ── Gapped systems: E_F must land inside the gap, not on its lower edge ──

TEST(PlanewaveFermiEnergyCalculator, WideGapPlacesFermiEnergyInsideTheGapNotAtTheValenceEdge) {
    /**
     * The electron count is flat across a gap, so a bisection that breaks the
     * tie one way converges to the valence band edge -- putting occupied
     * states directly at E_F, where the smearing then spills them. Measured on
     * FeO with U on Fe 3d at degauss 0.02 Ry, that misplacement carried
     * 4.2 mRy of spurious entropy and 2.4 mRy of total-energy error.
     *
     * Two bands, 1 Ry apart, two electrons: the gap runs from -0.5 to +0.5 and
     * E_F belongs near its middle, far from both edges.
     */
    KPointsGrid kpoint_grid;
    auto quantum_system = build_test_quantum_system(kpoint_grid,
                                                    /*number_of_kpoints_x=*/1,
                                                    /*number_of_bands=*/2,
                                                    /*number_of_spin_components=*/1,
                                                    /*number_of_electrons=*/2.0);
    quantum_system.set_eigenvalue(0, 0, -0.5);
    quantum_system.set_eigenvalue(0, 1, 0.5);

    GaussianSmearing gaussian_smearing(0.02);
    PlanewaveFermiEnergyCalculator calculator(quantum_system, &gaussian_smearing);
    Real fermi_energy = calculator.calculate_occupations();

    // Comfortably inside the gap rather than pinned to the -0.5 edge.
    EXPECT_GT(fermi_energy, -0.4);
    EXPECT_LT(fermi_energy, 0.4);

    /**
     * The property that matters downstream: with E_F away from both edges the
     * occupations are integral, so the smearing entropy vanishes and the
     * result stops depending on the smearing width.
     */
    Real occupation_valence = quantum_system.occupation_by_kpoint_state_index_and_band_index(0, 0);
    Real occupation_conduction =
        quantum_system.occupation_by_kpoint_state_index_and_band_index(0, 1);
    EXPECT_NEAR(occupation_valence, 1.0, 1e-9);
    EXPECT_NEAR(occupation_conduction, 0.0, 1e-9);
}

TEST(PlanewaveFermiEnergyCalculator, GappedResultIsIndependentOfTheSmearingWidth) {
    /**
     * A converged result must not depend on a numerical broadening chosen for
     * convergence. Before the plateau is bracketed from both sides, widening
     * the smearing moved FeO's total energy by 2.4 mRy; here it must leave the
     * occupations untouched.
     */
    for (Real smearing_width : {0.005, 0.02, 0.05}) {
        KPointsGrid kpoint_grid;
        auto quantum_system = build_test_quantum_system(kpoint_grid,
                                                        /*number_of_kpoints_x=*/1,
                                                        /*number_of_bands=*/2,
                                                        /*number_of_spin_components=*/1,
                                                        /*number_of_electrons=*/2.0);
        quantum_system.set_eigenvalue(0, 0, -0.5);
        quantum_system.set_eigenvalue(0, 1, 0.5);

        GaussianSmearing gaussian_smearing(smearing_width);
        PlanewaveFermiEnergyCalculator calculator(quantum_system, &gaussian_smearing);
        calculator.calculate_occupations();

        EXPECT_NEAR(quantum_system.occupation_by_kpoint_state_index_and_band_index(0, 0), 1.0, 1e-9)
            << "smearing width " << smearing_width;
        EXPECT_NEAR(quantum_system.occupation_by_kpoint_state_index_and_band_index(0, 1), 0.0, 1e-9)
            << "smearing width " << smearing_width;
    }
}

TEST(PlanewaveFermiEnergyCalculator, ColdSmearingAcrossAWideGapStaysInsideIt) {
    /**
     * The FeO case itself used Marzari-Vanderbilt smearing at degauss 0.02 Ry.
     * Cold smearing overshoots: just below E_F an occupation exceeds 1, so the
     * count crosses the target close above the valence edge -- here 0.92 sigma
     * above it -- which is where a single bisection stops. Across this
     * 12-sigma gap the Gaussian start already sits where the cold count is N,
     * well away from both edges.
     */
    KPointsGrid kpoint_grid;
    auto quantum_system = build_test_quantum_system(kpoint_grid,
                                                    /*number_of_kpoints_x=*/1,
                                                    /*number_of_bands=*/2,
                                                    /*number_of_spin_components=*/1,
                                                    /*number_of_electrons=*/2.0);
    quantum_system.set_eigenvalue(0, 0, -0.12);
    quantum_system.set_eigenvalue(0, 1, 0.12);

    MarzariVanderbiltSmearing cold_smearing(0.02);
    PlanewaveFermiEnergyCalculator calculator(quantum_system, &cold_smearing);
    Real fermi_energy = calculator.calculate_occupations();

    // The single-bisection answer is -0.1016; the midpoint of the bracket, -0.0208.
    EXPECT_GT(fermi_energy, -0.06);
    EXPECT_LT(fermi_energy, 0.06);
    EXPECT_NEAR(quantum_system.occupation_by_kpoint_state_index_and_band_index(0, 0), 1.0, 1e-6);
    EXPECT_NEAR(quantum_system.occupation_by_kpoint_state_index_and_band_index(0, 1), 0.0, 1e-9);
}

TEST(PlanewaveFermiEnergyCalculator, MetalKeepsItsPartiallyOccupiedBandAtTheFermiLevel) {
    /**
     * The gap handling must not disturb a metal, which has no plateau to
     * bracket: both bisections land on the same point and the midpoint is a
     * no-op. Three bands with one electron's worth of weight in the middle one
     * leaves it genuinely fractional.
     */
    KPointsGrid kpoint_grid;
    auto quantum_system = build_test_quantum_system(kpoint_grid,
                                                    /*number_of_kpoints_x=*/1,
                                                    /*number_of_bands=*/3,
                                                    /*number_of_spin_components=*/1,
                                                    /*number_of_electrons=*/3.0);
    quantum_system.set_eigenvalue(0, 0, -0.20);
    quantum_system.set_eigenvalue(0, 1, 0.00);
    quantum_system.set_eigenvalue(0, 2, 0.20);

    GaussianSmearing gaussian_smearing(0.05);
    PlanewaveFermiEnergyCalculator calculator(quantum_system, &gaussian_smearing);
    calculator.calculate_occupations();

    Real middle_occupation = quantum_system.occupation_by_kpoint_state_index_and_band_index(0, 1);
    EXPECT_GT(middle_occupation, 0.1);
    EXPECT_LT(middle_occupation, 0.9);
}

TEST(PlanewaveFermiEnergyCalculator, MetalFermiEnergyLeavesTheCellNeutral) {
    /**
     * A metal has a bracket as well: the plateau-top bisection stops where the
     * count reaches the target plus its 1e-6 slack, a few 1e-8 Ry above the
     * exact answer, and the midpoint of the two leaves half the slack in the
     * cell as extra charge. That is what happened while the midpoint was taken
     * whenever the bracket exceeded the bisection tolerance: copper carried
     * 19 + 5e-7 electrons, and its total energy rose by E_F x 5e-7 = 0.62 μRy.
     * The count a metal ends with must be the one it started with.
     */
    KPointsGrid kpoint_grid;
    auto quantum_system = build_test_quantum_system(kpoint_grid,
                                                    /*number_of_kpoints_x=*/1,
                                                    /*number_of_bands=*/3,
                                                    /*number_of_spin_components=*/1,
                                                    /*number_of_electrons=*/3.0);
    quantum_system.set_eigenvalue(0, 0, -0.20);
    quantum_system.set_eigenvalue(0, 1, 0.00);
    quantum_system.set_eigenvalue(0, 2, 0.20);

    GaussianSmearing gaussian_smearing(0.05);
    PlanewaveFermiEnergyCalculator calculator(quantum_system, &gaussian_smearing);
    calculator.calculate_occupations();

    Real number_of_electrons = 0.0;
    for (int band_index = 0; band_index < 3; band_index++) {
        number_of_electrons +=
            kpoint_grid[0].weight *
            quantum_system.occupation_by_kpoint_state_index_and_band_index(0, band_index);
    }
    EXPECT_NEAR(number_of_electrons, 3.0, 1e-9);
}

/**
 * An NSCF asked for exactly the occupied manifold -- band count equal to
 * occupied-band count, which is what a Wannier workflow requests -- has no
 * unoccupied level to bound the gap from above. The reported Fermi energy is
 * then the highest occupied level, not half of the +1e30 sentinel the LUMO
 * search starts from: that is what put `fermi_energy_ry: 5e+29` in a results
 * file.
 */
TEST(PlanewaveFermiEnergyCalculator, FullyOccupiedBandSetReportsTheHighestOccupiedLevel) {
    KPointsGrid kpoint_grid;
    auto quantum_system = build_test_quantum_system(kpoint_grid,
                                                    /*number_of_kpoints_x=*/2,
                                                    /*number_of_bands=*/2,
                                                    /*number_of_spin_components=*/1,
                                                    /*number_of_electrons=*/4.0);

    // Two bands, two of them occupied: nothing above the manifold.
    quantum_system.set_eigenvalue(0, 0, -5.0);
    quantum_system.set_eigenvalue(0, 1, -1.0);
    quantum_system.set_eigenvalue(1, 0, -4.5);
    quantum_system.set_eigenvalue(1, 1, -1.5); // highest occupied level is -1.0, at k(0)

    FixedOccupation fixed_occupation;
    PlanewaveFermiEnergyCalculator calculator(quantum_system, &fixed_occupation);

    Real fermi_energy = calculator.calculate_occupations();

    EXPECT_NEAR(fermi_energy, -1.0, 1e-10);
    EXPECT_NEAR(quantum_system.fermi_energy(), -1.0, 1e-10);
    EXPECT_LT(std::abs(fermi_energy), 1e10) << "a sentinel leaked into the Fermi energy";

    // Every band is occupied, and stays so.
    EXPECT_NEAR(quantum_system.occupation_by_kpoint_state_index_and_band_index(0, 0), 1.0, 1e-10);
    EXPECT_NEAR(quantum_system.occupation_by_kpoint_state_index_and_band_index(0, 1), 1.0, 1e-10);
    EXPECT_NEAR(quantum_system.occupation_by_kpoint_state_index_and_band_index(1, 1), 1.0, 1e-10);
}

/**
 * The mirror case, guarded for the same reason: with no electrons there is no
 * occupied level to bound the gap from below, and the HOMO search would other-
 * wise hand back its -1e30 sentinel.
 */
TEST(PlanewaveFermiEnergyCalculator, EmptyBandSetReportsTheLowestUnoccupiedLevel) {
    KPointsGrid kpoint_grid;
    auto quantum_system = build_test_quantum_system(kpoint_grid,
                                                    /*number_of_kpoints_x=*/1,
                                                    /*number_of_bands=*/2,
                                                    /*number_of_spin_components=*/1,
                                                    /*number_of_electrons=*/0.0);

    quantum_system.set_eigenvalue(0, 0, 2.0);
    quantum_system.set_eigenvalue(0, 1, 7.0);

    FixedOccupation fixed_occupation;
    PlanewaveFermiEnergyCalculator calculator(quantum_system, &fixed_occupation);

    Real fermi_energy = calculator.calculate_occupations();

    EXPECT_NEAR(fermi_energy, 2.0, 1e-10);
    EXPECT_LT(std::abs(fermi_energy), 1e10) << "a sentinel leaked into the Fermi energy";
}

namespace {

/**
 * A one-process policy that claims to run several pools, so the calculator
 * takes its pool-parallel path, and counts the collectives it is asked for.
 * The gather is the identity, as it is for a single pool holding every state.
 */
class CollectiveCountingExecutionPolicy : public execution::PlanewaveSerialExecutionPolicy {
  public:
    int number_of_pools() const override {
        return 2;
    }
    void reduce_scalar(Real& /*value*/) const override {
        ++scalar_reductions;
    }
    std::vector<Real>
    assemble_global_kpoint_state_data(const std::vector<Real>& local_data,
                                      int /*local_number_of_physical_kpoints*/,
                                      int /*number_of_spin_components*/,
                                      int /*number_of_values_per_state*/) const override {
        ++gathers;
        return local_data;
    }
    mutable int scalar_reductions = 0;
    mutable int gathers = 0;
};

} // namespace

TEST(PlanewaveFermiEnergyCalculator, PoolParallelBisectionGathersOnceAndMatchesSerialBitForBit) {
    KPointsGrid serial_grid;
    auto serial_system = build_test_quantum_system(serial_grid, 4, 3, 1, 3.0);
    KPointsGrid pooled_grid;
    auto pooled_system = build_test_quantum_system(pooled_grid, 4, 3, 1, 3.0);
    for (int state = 0; state < serial_system.number_of_kpoint_states(); state++) {
        for (int band = 0; band < 3; band++) {
            Real energy = -0.3 + 0.17 * band + 0.013 * state;
            serial_system.set_eigenvalue(state, band, energy);
            pooled_system.set_eigenvalue(state, band, energy);
        }
    }
    GaussianSmearing smearing(0.02);
    PlanewaveFermiEnergyCalculator serial_calculator(serial_system, &smearing);
    CollectiveCountingExecutionPolicy policy;
    PlanewaveFermiEnergyCalculator pooled_calculator(pooled_system, &smearing, &policy);

    const Real serial_fermi_energy = serial_calculator.calculate_occupations();
    const Real pooled_fermi_energy = pooled_calculator.calculate_occupations();

    // One gather of the eigenvalue table; only the k-weight sanity sum reduces.
    EXPECT_EQ(policy.gathers, 1);
    EXPECT_EQ(policy.scalar_reductions, 1);
    EXPECT_EQ(serial_fermi_energy, pooled_fermi_energy);
}

/**
 * Cold smearing across a gap only a few smearing widths wide (camsai/q3#5).
 *
 * The eigenvalues are silicon's on Q3's NC reference deck (2x2x2 shifted
 * k-points, two irreducible points weighted 1/4 and 3/4), as a cold-smearing
 * run at degauss 0.02 Ry printed them. Its valence maximum is 5.297 eV and its
 * conduction minimum 7.574 eV. The count never returns exactly to N inside that
 * gap at 0.03 Ry and above, and the first crossing sits about one width above
 * the valence band. Quantum ESPRESSO 7.5, given the same deck, places E_F at
 * 6.6154 eV (0.02 Ry), 6.6876 eV (0.03 Ry) and 5.7980 eV (0.04 Ry, where it
 * reports falling back to bisection); the old bisection gave 6.1579 and
 * 5.6729 eV for the first two, each with a spurious smearing term.
 */
namespace {

constexpr Real electronvolts_per_rydberg = 13.605693122994;

Real silicon_cold_smearing_fermi_energy_in_electronvolts(const SmearingScheme& smearing,
                                                         std::vector<Real>* occupations = nullptr) {
    const std::vector<std::vector<Real>> eigenvalues_in_electronvolts = {
        {-4.9945, 2.1498, 5.2973, 5.2973, 7.9882, 9.5174, 9.5174, 12.7831},
        {-3.1149, -0.2879, 2.5716, 3.8767, 7.5743, 10.4706, 11.6790, 11.9088}};
    KPointsGrid kpoint_grid;
    for (Real weight : {0.5, 1.5}) {
        KPoint kpoint;
        kpoint.weight = weight;
        kpoint_grid.add_kpoint(kpoint);
    }
    TestableQuantumSystem quantum_system(kpoint_grid, 8, 1, 8.0);
    quantum_system.allocate_electronic_states(std::vector<int>(2, 0));
    for (int kpoint_index = 0; kpoint_index < 2; kpoint_index++) {
        for (int band_index = 0; band_index < 8; band_index++) {
            quantum_system.set_eigenvalue(kpoint_index,
                                          band_index,
                                          eigenvalues_in_electronvolts[kpoint_index][band_index] /
                                              electronvolts_per_rydberg);
        }
    }
    PlanewaveFermiEnergyCalculator calculator(quantum_system, &smearing);
    Real fermi_energy = calculator.calculate_occupations();
    if (occupations != nullptr) {
        occupations->clear();
        for (int kpoint_index = 0; kpoint_index < 2; kpoint_index++) {
            for (int band_index = 0; band_index < 8; band_index++) {
                occupations->push_back(
                    quantum_system.occupation_by_kpoint_state_index_and_band_index(kpoint_index,
                                                                                   band_index));
            }
        }
    }
    return fermi_energy * electronvolts_per_rydberg;
}

} // namespace

TEST(PlanewaveFermiEnergyCalculator, ColdSmearingAcrossANarrowGapFindsTheMinimumNotTheEdge) {
    // The input eigenvalues are rounded to 0.1 meV, so E_F is compared to 2 meV.
    MarzariVanderbiltSmearing at_two_hundredths(0.02);
    EXPECT_NEAR(
        silicon_cold_smearing_fermi_energy_in_electronvolts(at_two_hundredths), 6.6154, 2e-3);
    MarzariVanderbiltSmearing at_three_hundredths(0.03);
    EXPECT_NEAR(
        silicon_cold_smearing_fermi_energy_in_electronvolts(at_three_hundredths), 6.6876, 2e-3);
}

TEST(PlanewaveFermiEnergyCalculator, ColdSmearingKeepsTheTopValenceBandFull) {
    // At 0.02 Ry the in-gap minimum holds N to 2.4e-8: no band is partly emptied.
    MarzariVanderbiltSmearing cold_smearing(0.02);
    std::vector<Real> occupations;
    silicon_cold_smearing_fermi_energy_in_electronvolts(cold_smearing, &occupations);
    for (int kpoint_index = 0; kpoint_index < 2; kpoint_index++) {
        for (int band_index = 0; band_index < 4; band_index++) {
            EXPECT_NEAR(occupations[kpoint_index * 8 + band_index], 1.0, 1e-6)
                << "k " << kpoint_index << " band " << band_index;
        }
    }
}

TEST(PlanewaveFermiEnergyCalculator, ColdSmearingTooWideForTheGapFallsBackToBisection) {
    // At 0.04 Ry the in-gap minimum is 1.1e-2 electrons off, too far to accept.
    MarzariVanderbiltSmearing cold_smearing(0.04);
    EXPECT_NEAR(silicon_cold_smearing_fermi_energy_in_electronvolts(cold_smearing), 5.7980, 2e-3);
}

TEST(PlanewaveFermiEnergyCalculator, ColdSmearingOnAMetalLeavesTheCellNeutral) {
    // The Newton refinement converges on the root itself, so the count is exact.
    for (Real smearing_width : {0.01, 0.05}) {
        KPointsGrid kpoint_grid;
        auto quantum_system = build_test_quantum_system(kpoint_grid,
                                                        /*number_of_kpoints_x=*/1,
                                                        /*number_of_bands=*/3,
                                                        /*number_of_spin_components=*/1,
                                                        /*number_of_electrons=*/3.0);
        quantum_system.set_eigenvalue(0, 0, -0.20);
        quantum_system.set_eigenvalue(0, 1, 0.00);
        quantum_system.set_eigenvalue(0, 2, 0.20);

        MarzariVanderbiltSmearing cold_smearing(smearing_width);
        PlanewaveFermiEnergyCalculator calculator(quantum_system, &cold_smearing);
        calculator.calculate_occupations();

        Real number_of_electrons = 0.0;
        for (int band_index = 0; band_index < 3; band_index++) {
            number_of_electrons +=
                kpoint_grid[0].weight *
                quantum_system.occupation_by_kpoint_state_index_and_band_index(0, band_index);
        }
        EXPECT_NEAR(number_of_electrons, 3.0, 1e-9) << "smearing width " << smearing_width;
    }
}

TEST(PlanewaveFermiEnergyCalculator, MethfesselPaxtonAcrossANarrowGapStaysInsideIt) {
    // Methfessel-Paxton occupations overshoot too, so they take the same route.
    MethfesselPaxtonSmearing methfessel_paxton(0.02);
    EXPECT_FALSE(methfessel_paxton.is_occupation_monotonic());
    std::vector<Real> occupations;
    Real fermi_energy =
        silicon_cold_smearing_fermi_energy_in_electronvolts(methfessel_paxton, &occupations);
    EXPECT_GT(fermi_energy, 5.297 + 1.0);
    EXPECT_LT(fermi_energy, 7.574 - 0.5);
}
