#include <gtest/gtest.h>

#include <complex>
#include <vector>

#include "io/pseudopotential/upf_reader.hpp"
#include "layer_0_core/math/phase_factor.hpp"
#include "layer_1_structure/crystal/crystal.hpp"
#include "layer_1_structure/reciprocal_space/kpoints_grid.hpp"
#include "layer_3_representation/basis/planewave/init.hpp"
#include "layer_3_representation/basis/planewave/pseudopotential/ultrasoft/init.hpp"
#include "layer_3_representation/pseudopotential/planewave/internal/atomic_orbital_builder.hpp"
#include "layer_3_representation/pseudopotential/planewave/internal/lowdin_orthogonalization.hpp"
#include "layer_3_representation/pseudopotential/planewave/internal/projector.hpp"
#include "layer_3_representation/pseudopotential/planewave/internal/wavefunction_atomic.hpp"
#include "layer_3_representation/pseudopotential/set.hpp"
#include "layer_4_computation/interpolator/method/wannier/planewave/internal/overlaps.hpp"
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

struct UltrasoftFixture {
    PseudopotentialSet pseudopotential_set;
    Crystal crystal_structure;
    PlanewaveBasis basis;
    UltrasoftPlanewaveBasis ultrasoft_basis;
    KPointsGrid kpoint_grid;

    UltrasoftFixture()
        : pseudopotential_set(load_silicon_uspp())
        , crystal_structure(make_silicon_fcc(pseudopotential_set)) {
        basis.generate(serial_policy, crystal_structure, 24.0, 96.0);
        ultrasoft_basis.init(crystal_structure, basis, pseudopotential_set);
        kpoint_grid.generate(2, 2, 2, 1, 1, 1);
    }
};

} // namespace

/**
 * ── The cell-volume-factor arbiter: compute_q_g_single's general branch ──
 * ── differs from its own b=0 special branch by exactly the cell volume. ──
 */

TEST(BlochOverlapCalculatorAugmentation,
     ComputeQGSingleGeneralBranchIsCellVolumeTimesSmallerThanZeroLimit) {
    /**
     * The plan's own gate, refined by what the numbers actually showed:
     * compute_q_g_single's b=0 special branch (used directly, no scaling)
     * and its general (Gaunt-coefficient) branch at small b do NOT agree
     * without a correction -- they differ by exactly the cell volume (see
     * bloch_overlap_calculator.hpp's normalization note for why: the
     * general branch is normalized for its existing charge-density
     * consumer, not the S-operator convention this class needs). This
     * test pins that factor with a real Si ultrasoft pseudopotential:
     * cell_volume() * general(b->0) must converge to special(b=0).
     */
    UltrasoftFixture fixture;
    Real tpiba = fixture.crystal_structure.two_pi_over_lattice_parameter();
    Real cell_volume = fixture.crystal_structure.cell_volume();
    int number_of_channels = fixture.ultrasoft_basis.number_of_projector_channels(0);
    ASSERT_GT(number_of_channels, 0);

    for (int channel = 0; channel < number_of_channels; channel++) {
        Complex special = fixture.ultrasoft_basis.compute_q_g_single(
            0, channel, channel, Vector3D(0.0, 0.0, 0.0), tpiba);

        Complex general_at_tiny_b = fixture.ultrasoft_basis.compute_q_g_single(
            0, channel, channel, Vector3D(1e-4, 0.0, 0.0), tpiba);
        Complex scaled = cell_volume * general_at_tiny_b;

        EXPECT_NEAR(scaled.real(), special.real(), 1e-4)
            << "channel " << channel << ": cell_volume() * general branch at |b|=1e-4 does "
            << "not converge to the b=0 special branch";
        EXPECT_NEAR(scaled.imag(), special.imag(), 1e-4) << "channel " << channel;
    }
}

TEST(BlochOverlapCalculatorAugmentation, ScaledGeneralBranchConvergesToTheZeroLimitAsBShrinks) {
    /**
     * A stronger version of the above: shrinking |b| should shrink the
     * remaining discrepancy after the cell-volume correction, not leave
     * it at a fixed offset -- a genuine convergence trend.
     */
    UltrasoftFixture fixture;
    Real tpiba = fixture.crystal_structure.two_pi_over_lattice_parameter();
    Real cell_volume = fixture.crystal_structure.cell_volume();
    int channel = 0;

    Complex special = fixture.ultrasoft_basis.compute_q_g_single(
        0, channel, channel, Vector3D(0.0, 0.0, 0.0), tpiba);

    Real previous_difference = 1e300;
    for (Real magnitude : {1e-2, 1e-3, 1e-4}) {
        Complex general = fixture.ultrasoft_basis.compute_q_g_single(
            0, channel, channel, Vector3D(magnitude, 0.0, 0.0), tpiba);
        Real difference = std::abs(cell_volume * general - special);
        EXPECT_LT(difference, previous_difference)
            << "|b|=" << magnitude << " did not converge toward the b=0 value";
        previous_difference = difference;
    }
    EXPECT_LT(previous_difference, 1e-4);
}

TEST(BlochOverlapCalculatorAugmentation, AddAugmentationContributionRejectsExactlyZeroB) {
    UltrasoftFixture fixture;
    std::vector<int> g_vector_indices;
    std::vector<Real> kinetic_energy_squared;
    fixture.basis.compute_gk(fixture.kpoint_grid[0].coordinates,
                             fixture.crystal_structure,
                             g_vector_indices,
                             kinetic_energy_squared);
    int number_of_plane_waves = static_cast<int>(g_vector_indices.size());

    Matrix<Complex> projectors;
    compute_kleinman_bylander_projectors(fixture.kpoint_grid[0].coordinates,
                                         fixture.crystal_structure,
                                         fixture.basis,
                                         fixture.pseudopotential_set,
                                         fixture.ultrasoft_basis.beta_function_table(),
                                         g_vector_indices,
                                         number_of_plane_waves,
                                         fixture.ultrasoft_basis.total_number_of_projectors(),
                                         projectors);

    MillerIndexLookup global_lookup;
    global_lookup.build(fixture.basis.g_vectors());
    PlanewaveBlochOverlapCalculator calculator(fixture.basis, global_lookup);

    std::vector<Complex> dummy_band(number_of_plane_waves, Complex(1.0, 0.0));
    BlochOverlapResult result = calculator.compute(
        g_vector_indices,
        1,
        [&](int) { return dummy_band.data(); },
        g_vector_indices,
        1,
        [&](int) { return dummy_band.data(); },
        IntegerVector3D(0, 0, 0));

    EXPECT_THROW(calculator.add_augmentation_contribution(
                     result,
                     fixture.crystal_structure,
                     fixture.ultrasoft_basis,
                     projectors,
                     number_of_plane_waves,
                     1,
                     [&](int) { return dummy_band.data(); },
                     projectors,
                     number_of_plane_waves,
                     1,
                     [&](int) { return dummy_band.data(); },
                     Vector3D(0.0, 0.0, 0.0)),
                 std::invalid_argument);
}

/**
 * ── Kernel bookkeeping: add_augmentation_contribution vs a from-scratch ──
 * ── reference computed independently in this test, at a small, realistic ─
 * ── nonzero b -- the scenario every real Wannier calculation uses. ───────
 */

TEST(BlochOverlapCalculatorAugmentation,
     AugmentationContributionMatchesIndependentReferenceAtSmallB) {
    UltrasoftFixture fixture;
    std::vector<int> g_vector_indices;
    std::vector<Real> kinetic_energy_squared;
    fixture.basis.compute_gk(fixture.kpoint_grid[0].coordinates,
                             fixture.crystal_structure,
                             g_vector_indices,
                             kinetic_energy_squared);
    int number_of_plane_waves = static_cast<int>(g_vector_indices.size());
    Real tpiba = fixture.crystal_structure.two_pi_over_lattice_parameter();
    Real cell_volume = fixture.crystal_structure.cell_volume();

    /**
     * A small, physically realistic neighbor-shell-like translation (not
     * an exact grid vector -- this test only exercises the augmentation
     * kernel's bookkeeping, not the plane-wave G-matching, so any nonzero
     * b works).
     */
    Vector3D b_cartesian(0.05, 0.02, -0.03);

    auto atomic_wavefunction_table = tabulate_atomic_wavefunctions_in_q_space(
        fixture.pseudopotential_set, fixture.crystal_structure.cell_volume(), 0.01, 900);
    std::vector<std::vector<Complex>> orbitals;
    build_atomic_orbitals_g_space(0,
                                  number_of_plane_waves,
                                  g_vector_indices,
                                  fixture.crystal_structure,
                                  fixture.kpoint_grid,
                                  fixture.basis,
                                  fixture.pseudopotential_set,
                                  atomic_wavefunction_table,
                                  0.01,
                                  900,
                                  orbitals);
    lowdin_orthogonalize_orbitals(orbitals);
    const std::vector<Complex>& psi = orbitals[0];

    Matrix<Complex> projectors;
    compute_kleinman_bylander_projectors(fixture.kpoint_grid[0].coordinates,
                                         fixture.crystal_structure,
                                         fixture.basis,
                                         fixture.pseudopotential_set,
                                         fixture.ultrasoft_basis.beta_function_table(),
                                         g_vector_indices,
                                         number_of_plane_waves,
                                         fixture.ultrasoft_basis.total_number_of_projectors(),
                                         projectors);

    MillerIndexLookup global_lookup;
    global_lookup.build(fixture.basis.g_vectors());
    PlanewaveBlochOverlapCalculator calculator(fixture.basis, global_lookup);

    BlochOverlapResult result = calculator.compute(
        g_vector_indices,
        1,
        [&](int) { return psi.data(); },
        g_vector_indices,
        1,
        [&](int) { return psi.data(); },
        IntegerVector3D(0, 0, 0));

    calculator.add_augmentation_contribution(
        result,
        fixture.crystal_structure,
        fixture.ultrasoft_basis,
        projectors,
        number_of_plane_waves,
        1,
        [&](int) { return psi.data(); },
        projectors,
        number_of_plane_waves,
        1,
        [&](int) { return psi.data(); },
        b_cartesian);
    Complex from_kernel = result(0, 0);

    /**
     * Independent reference, built directly from
     * compute_kleinman_bylander_projectors' own output and
     * compute_q_g_single (cell-volume-scaled, per the arbiter tests
     * above) -- not calling add_augmentation_contribution, so this
     * checks its own bookkeeping (projector offsets, iteration order,
     * translation phase) independently. compute_q_g_single takes its
     * wavevector in 2*pi/alat units (it multiplies by tpiba itself), so
     * the bohr^-1 b is divided by tpiba before the call; the phase, a
     * plain dot product with a bohr position, uses b as is.
     */
    const Vector3D b_in_lattice_units = b_cartesian * (1.0 / tpiba);
    int total_number_of_projectors = fixture.ultrasoft_basis.total_number_of_projectors();
    std::vector<Complex> amplitude(total_number_of_projectors, Complex(0.0, 0.0));
    for (int projector_index = 0; projector_index < total_number_of_projectors; projector_index++) {
        Complex sum(0.0, 0.0);
        for (int g_index = 0; g_index < number_of_plane_waves; g_index++) {
            sum += std::conj(projectors[projector_index][g_index]) * psi[g_index];
        }
        amplitude[projector_index] = sum;
    }

    Complex reference_augmentation(0.0, 0.0);
    int projector_offset = 0;
    for (int atom_index = 0; atom_index < fixture.crystal_structure.number_of_atoms();
         atom_index++) {
        int number_of_channels = fixture.ultrasoft_basis.number_of_projector_channels(0);
        Vector3D atomic_position =
            fixture.crystal_structure.convert_from_crystal_to_cartesian(atom_index);
        Complex translation_phase = compute_phase_factor(
            b_cartesian.dot(atomic_position) * fixture.crystal_structure.lattice_parameter());

        for (int channel_i = 0; channel_i < number_of_channels; channel_i++) {
            for (int channel_j = 0; channel_j < number_of_channels; channel_j++) {
                Complex q_ij = cell_volume *
                               fixture.ultrasoft_basis.compute_q_g_single(
                                   0, channel_i, channel_j, b_in_lattice_units, tpiba) *
                               translation_phase;
                reference_augmentation += std::conj(amplitude[projector_offset + channel_i]) *
                                          q_ij * amplitude[projector_offset + channel_j];
            }
        }
        projector_offset += number_of_channels;
    }
    Complex reference_total = Complex(1.0, 0.0) + reference_augmentation; // <psi|psi> = 1 (Lowdin)

    EXPECT_NEAR(from_kernel.real(), reference_total.real(), 1e-10);
    EXPECT_NEAR(from_kernel.imag(), reference_total.imag(), 1e-10);
}

TEST(BlochOverlapCalculatorAugmentation, TranslationPhaseHasUnitMagnitude) {
    /**
     * The atomic structure-factor phase applied to Q_ij(b) must be a pure
     * phase (|e^{i*angle}| = 1) for any real crystal position and b --
     * guards against an accidental real-valued damping factor sneaking in.
     */
    UltrasoftFixture fixture;
    Vector3D b_cartesian(0.37, -0.12, 0.05);
    for (int atom_index = 0; atom_index < fixture.crystal_structure.number_of_atoms();
         atom_index++) {
        Vector3D atomic_position =
            fixture.crystal_structure.convert_from_crystal_to_cartesian(atom_index);
        Real angle =
            b_cartesian.dot(atomic_position) * fixture.crystal_structure.lattice_parameter();
        Complex phase = compute_phase_factor(angle);
        EXPECT_NEAR(std::abs(phase), 1.0, 1e-12);
    }
}
