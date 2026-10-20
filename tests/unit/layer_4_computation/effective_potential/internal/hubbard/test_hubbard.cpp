#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>

#include "io/pseudopotential/upf_reader.hpp"
#include "layer_1_structure/crystal/crystal.hpp"
#include "layer_1_structure/reciprocal_space/kpoints_grid.hpp"
#include "layer_1_structure/symmetry/symmetry.hpp"
#include "layer_3_representation/basis/planewave/init.hpp"
#include "layer_3_representation/basis/planewave/pseudopotential/factory.hpp"
#include "layer_3_representation/pseudopotential/planewave/internal/atomic_orbital_builder.hpp"
#include "layer_3_representation/pseudopotential/planewave/internal/wavefunction_atomic.hpp"
#include "layer_4_computation/effective_potential/internal/hubbard/hubbard.hpp"
#include "layer_4_computation/effective_potential/internal/hubbard/symmetrization.hpp"
#include "layer_4_computation/iterator/kohn_sham/planewave/execution/serial.hpp"

using namespace q3;

namespace {

Crystal make_nio_rocksalt() {
    Crystal crystal_structure;
    crystal_structure.set_from_lattice_type(2, 7.89); // FCC rocksalt-like cell
    AtomSpecies nickel{"Ni", 58.69, 18.0, "Ni.upf"};
    AtomSpecies oxygen{"O", 16.00, 6.0, "O.upf"};
    crystal_structure.add_species(nickel);
    crystal_structure.add_species(oxygen);
    crystal_structure.add_atom({0, {0.0, 0.0, 0.0}});
    crystal_structure.add_atom({1, {0.5, 0.5, 0.5}});
    crystal_structure.compute_derived_quantities();
    return crystal_structure;
}

/**
 * Cubic perovskite ABO₃: the three oxygens sit at the face centres and are
 * *cycled*, not swapped, by the three-fold rotation about [111]. Each keeps a
 * four-fold axis of its own, so its p manifold stays anisotropic and the group
 * average has to carry each oxygen's axis to the right neighbour rather than to
 * the one two steps along. Only a cell of this shape separates the two ways of
 * pairing `atom_map` with D(R); see the section note further down.
 */
Crystal make_cubic_perovskite() {
    Crystal crystal_structure;
    crystal_structure.set_from_lattice_type(1, 7.4); // simple cubic, alat in Bohr
    crystal_structure.add_species(AtomSpecies{"Sr", 87.62, 10.0, "Sr.upf"});
    crystal_structure.add_species(AtomSpecies{"Ti", 47.867, 12.0, "Ti.upf"});
    crystal_structure.add_species(AtomSpecies{"O", 15.999, 6.0, "O.upf"});
    crystal_structure.add_atom({0, {0.0, 0.0, 0.0}});
    crystal_structure.add_atom({1, {0.5, 0.5, 0.5}});
    crystal_structure.add_atom({2, {0.5, 0.5, 0.0}});
    crystal_structure.add_atom({2, {0.5, 0.0, 0.5}});
    crystal_structure.add_atom({2, {0.0, 0.5, 0.5}});
    crystal_structure.compute_derived_quantities();
    return crystal_structure;
}

/// Two equivalent atoms: the shape every DFT+U case Q3 is gated on already has.
Crystal make_diamond_with_two_hubbard_atoms() {
    Crystal crystal_structure;
    crystal_structure.set_from_lattice_type(2, 6.74); // FCC, alat in Bohr
    crystal_structure.add_species(AtomSpecies{"C", 12.011, 4.0, "C.upf"});
    crystal_structure.add_atom({0, {0.0, 0.0, 0.0}});
    crystal_structure.add_atom({0, {0.25, 0.25, 0.25}});
    crystal_structure.compute_derived_quantities();
    return crystal_structure;
}

constexpr int OXYGEN_MANIFOLD_SIZE = 3;
constexpr Real GROUP_AVERAGE_TOLERANCE = 1e-10;

/// U on the three oxygens' p manifolds, which are atoms 2, 3 and 4 above.
std::vector<HubbardManifoldLayout> oxygen_p_manifolds() {
    return {
        {2, 1, OXYGEN_MANIFOLD_SIZE}, {3, 1, OXYGEN_MANIFOLD_SIZE}, {4, 1, OXYGEN_MANIFOLD_SIZE}};
}

/**
 * A reproducible, deliberately unsymmetric occupation matrix per manifold:
 * every entry distinct, real and Hermitian, the shape a collinear wedge sum
 * produces once time reversal has been projected out.
 */
std::vector<std::vector<std::vector<Complex>>>
seeded_occupation_matrices(size_t number_of_manifolds, int manifold_size) {
    std::vector<std::vector<std::vector<Complex>>> occupation_matrices(number_of_manifolds);
    for (size_t manifold = 0; manifold < number_of_manifolds; manifold++) {
        occupation_matrices[manifold].assign(
            1, std::vector<Complex>(static_cast<size_t>(manifold_size) * manifold_size));
        for (int row = 0; row < manifold_size; row++) {
            for (int column = row; column < manifold_size; column++) {
                Real value = 0.1 + 0.37 * std::sin(1.0 + static_cast<Real>(manifold) + 3.0 * row +
                                                   7.0 * column);
                occupation_matrices[manifold][0][static_cast<size_t>(row) * manifold_size +
                                                 column] = Complex(value, 0.0);
                occupation_matrices[manifold][0][static_cast<size_t>(column) * manifold_size +
                                                 row] = Complex(value, 0.0);
            }
        }
    }
    return occupation_matrices;
}

/// max |n_{R(a)} − D(R) n_a D(R)ᵀ| over every operation, manifold and matrix entry.
Real hubbard_covariance_violation(
    const Crystal& crystal_structure,
    const std::vector<SymmetryOperation>& symmetry_operations,
    const std::vector<HubbardManifoldLayout>& layouts,
    const std::vector<std::vector<std::vector<Complex>>>& occupation_matrices) {
    Real worst = 0.0;

    auto manifold_on_atom = [&](int atom_index, int angular_momentum) {
        for (size_t index = 0; index < layouts.size(); index++)
            if (layouts[index].atom_index == atom_index &&
                layouts[index].angular_momentum == angular_momentum)
                return static_cast<int>(index);
        return -1;
    };

    for (const auto& operation : symmetry_operations) {
        Matrix3x3 cartesian = cartesian_rotation_of(crystal_structure, operation);
        for (size_t manifold = 0; manifold < layouts.size(); manifold++) {
            const auto& layout = layouts[manifold];
            int image_manifold =
                manifold_on_atom(operation.atom_map[layout.atom_index], layout.angular_momentum);
            if (image_manifold < 0)
                continue;
            auto rotation = build_real_harmonic_rotation_matrix(layout.angular_momentum, cartesian);
            int manifold_size = layout.manifold_size;

            for (int row = 0; row < manifold_size; row++) {
                for (int column = 0; column < manifold_size; column++) {
                    Real rotated = 0.0;
                    for (int p = 0; p < manifold_size; p++)
                        for (int q = 0; q < manifold_size; q++)
                            rotated +=
                                rotation[static_cast<size_t>(row) * manifold_size + p] *
                                rotation[static_cast<size_t>(column) * manifold_size + q] *
                                occupation_matrices[manifold][0]
                                                   [static_cast<size_t>(p) * manifold_size + q]
                                                       .real();
                    Real difference = std::abs(
                        occupation_matrices[static_cast<size_t>(image_manifold)][0]
                                           [static_cast<size_t>(row) * manifold_size + column]
                                               .real() -
                        rotated);
                    worst = std::max(worst, difference);
                }
            }
        }
    }
    return worst;
}

Real maximum_occupation_difference(const std::vector<std::vector<std::vector<Complex>>>& left,
                                   const std::vector<std::vector<std::vector<Complex>>>& right) {
    Real worst = 0.0;
    for (size_t manifold = 0; manifold < left.size(); manifold++)
        for (size_t spin = 0; spin < left[manifold].size(); spin++)
            for (size_t index = 0; index < left[manifold][spin].size(); index++)
                worst = std::max(
                    worst, std::abs(left[manifold][spin][index] - right[manifold][spin][index]));
    return worst;
}

Real total_trace(const std::vector<std::vector<std::vector<Complex>>>& occupation_matrices,
                 int manifold_size) {
    Real trace = 0.0;
    for (const auto& per_manifold : occupation_matrices)
        for (const auto& matrix : per_manifold)
            for (int index = 0; index < manifold_size; index++)
                trace += matrix[static_cast<size_t>(index) * manifold_size + index].real();
    return trace;
}

execution::PlanewaveSerialExecutionPolicy serial_policy;

// Ni.upf (PseudoDojo NC-SR-04): PP_PSWFC has 3S(l=0) + 3P(l=1) + 3D(l=2) + 4S(l=0), all occupied.
PseudopotentialSet load_nickel_pseudopotential() {
    PseudopotentialSet pseudopotential_set;
    pseudopotential_set.push_back(
        UPFReader::read(std::string(EXAMPLES_DIR) + "/data/pseudopotentials/nc/Ni.upf"));
    return pseudopotential_set;
}

Crystal make_nickel_fcc(const PseudopotentialSet& pseudopotential_set) {
    Crystal crystal_structure;
    crystal_structure.set_from_lattice_type(2, 6.65); // FCC Ni, approximate alat (Bohr)
    AtomSpecies nickel{"Ni", 58.69, pseudopotential_set[0].valence_charge, "Ni.upf"};
    crystal_structure.add_species(nickel);
    crystal_structure.add_atom({0, {0.0, 0.0, 0.0}});
    crystal_structure.compute_derived_quantities();
    return crystal_structure;
}

/**
 * Displaces one atom by delta_bohr along a Cartesian direction, holding the lattice fixed.
 * Identical construction to tests/unit/layer_4_computation/effective_potential/internal/
 * dispersion/test_d3bj.cpp's own perturb_atom_cartesian (duplicated rather than shared, per
 * this codebase's convention of self-contained test files): cartesian = fractional . L (rows
 * L_i) is equivalent to cartesian = L^T . fractional as a column-vector product, so the
 * fractional shift for a given Cartesian shift is (L^T)^{-1} . delta_cartesian.
 */
Crystal
perturb_atom_cartesian(const Crystal& original, int atom_index, int direction, Real delta_bohr) {
    Matrix3x3 lattice_vectors_bohr = original.lattice_vectors_bohr();
    Vector3D delta_cartesian{0.0, 0.0, 0.0};
    delta_cartesian[direction] = delta_bohr;
    Matrix3x3 lattice_transpose_inverse = lattice_vectors_bohr.transpose().inverse();
    Vector3D delta_fractional = lattice_transpose_inverse * delta_cartesian;

    std::vector<Atom> atoms;
    for (int index = 0; index < original.number_of_atoms(); index++) {
        Vector3D fractional = original.fractional_position(index);
        if (index == atom_index) {
            fractional = fractional + delta_fractional;
        }
        atoms.push_back({original.species_type_index(index), fractional});
    }
    return Crystal::from_lattice_vectors(1.0, lattice_vectors_bohr, original.species(), atoms);
}

std::vector<AtomicOrbitalLabel> make_synthetic_orbital_labels() {
    /**
     * Mimics enumerate_atomic_orbitals() output for a Ni (3S, 3P, 3D) + O (2S, 2P) system:
     * atom 0 = Ni with a 3D shell (5 orbitals), atom 1 = O with no D shell.
     */
    std::vector<AtomicOrbitalLabel> labels;
    labels.push_back({0, 0, "3S", 0, 0});
    for (int m = 0; m < 3; m++)
        labels.push_back({0, 0, "3P", 1, m});
    for (int m = 0; m < 5; m++)
        labels.push_back({0, 0, "3D", 2, m});
    labels.push_back({1, 1, "2S", 0, 0});
    for (int m = 0; m < 3; m++)
        labels.push_back({1, 1, "2P", 1, m});
    return labels;
}

} // namespace

TEST(HubbardManifoldResolution, FindsCorrectOrbitalIndicesForSpeciesAndShell) {
    Crystal crystal_structure = make_nio_rocksalt();
    auto orbital_labels = make_synthetic_orbital_labels();
    std::vector<HubbardSpecification> specifications = {{"Ni", "3D", 0.22}};

    auto manifolds = resolve_hubbard_manifolds(crystal_structure, orbital_labels, specifications);

    ASSERT_EQ(manifolds.size(), 1u);
    EXPECT_EQ(manifolds[0].atom_index, 0);
    EXPECT_EQ(manifolds[0].angular_momentum, 2);
    EXPECT_DOUBLE_EQ(manifolds[0].u_rydberg, 0.22);
    ASSERT_EQ(manifolds[0].orbital_indices.size(), 5u);
    // The 3D shell is the last 5 entries among Ni's 1(3S)+3(3P)+5(3D)=9 orbitals.
    EXPECT_EQ(manifolds[0].orbital_indices, (std::vector<int>{4, 5, 6, 7, 8}));
}

TEST(HubbardManifoldResolution, UnknownSpeciesThrows) {
    Crystal crystal_structure = make_nio_rocksalt();
    auto orbital_labels = make_synthetic_orbital_labels();
    std::vector<HubbardSpecification> specifications = {{"Fe", "3D", 0.22}};
    EXPECT_THROW(resolve_hubbard_manifolds(crystal_structure, orbital_labels, specifications),
                 std::runtime_error);
}

TEST(HubbardManifoldResolution, UnknownShellThrows) {
    Crystal crystal_structure = make_nio_rocksalt();
    auto orbital_labels = make_synthetic_orbital_labels();
    std::vector<HubbardSpecification> specifications = {{"Ni", "4F", 0.22}};
    EXPECT_THROW(resolve_hubbard_manifolds(crystal_structure, orbital_labels, specifications),
                 std::runtime_error);
}

TEST(HubbardManifoldResolution, MultipleAtomsOfSameSpeciesEachGetAManifold) {
    // Two Ni atoms (rocksalt-like but both Ni for this synthetic test).
    Crystal crystal_structure;
    crystal_structure.set_from_lattice_type(2, 7.89);
    AtomSpecies nickel{"Ni", 58.69, 18.0, "Ni.upf"};
    crystal_structure.add_species(nickel);
    crystal_structure.add_atom({0, {0.0, 0.0, 0.0}});
    crystal_structure.add_atom({0, {0.5, 0.5, 0.5}});
    crystal_structure.compute_derived_quantities();

    std::vector<AtomicOrbitalLabel> labels;
    for (int atom_index = 0; atom_index < 2; atom_index++) {
        labels.push_back({atom_index, 0, "3S", 0, 0});
        for (int m = 0; m < 5; m++)
            labels.push_back({atom_index, 0, "3D", 2, m});
    }
    std::vector<HubbardSpecification> specifications = {{"Ni", "3D", 0.22}};

    auto manifolds = resolve_hubbard_manifolds(crystal_structure, labels, specifications);
    ASSERT_EQ(manifolds.size(), 2u);
    EXPECT_EQ(manifolds[0].atom_index, 0);
    EXPECT_EQ(manifolds[1].atom_index, 1);
}

// ─── Occupation matrix accumulation ─────────────────────────────────────────

TEST(HubbardOccupationMatrix, SingleFullyOccupiedBandGivesProjectorOuterProduct) {
    /**
     * One band, fully occupied (weight 1), overlaps <phi_m|psi> = (1, 0) for a
     * 2-orbital manifold (psi is entirely orbital 0's character). Expected
     * n = [[1,0],[0,0]] (outer product of the overlap vector with its conjugate).
     */
    int manifold_size = 2;
    Matrix<Complex> overlaps(1, 2);
    overlaps(0, 0) = Complex(1.0, 0.0);
    overlaps(0, 1) = Complex(0.0, 0.0);
    std::vector<Real> occupations = {1.0};
    std::vector<Complex> occupation_matrix(4, Complex(0.0, 0.0));

    accumulate_hubbard_occupation_matrix(
        1, manifold_size, overlaps, occupations, /*kpoint_weight=*/1.0, occupation_matrix);

    EXPECT_NEAR(occupation_matrix[0].real(), 1.0, 1e-12); // n_00
    EXPECT_NEAR(occupation_matrix[1].real(), 0.0, 1e-12); // n_01
    EXPECT_NEAR(occupation_matrix[2].real(), 0.0, 1e-12); // n_10
    EXPECT_NEAR(occupation_matrix[3].real(), 0.0, 1e-12); // n_11
}

TEST(HubbardOccupationMatrix, EmptyBandContributesNothing) {
    int manifold_size = 1;
    Matrix<Complex> overlaps(1, 1);
    overlaps(0, 0) = Complex(0.7, 0.3);
    std::vector<Real> occupations = {0.0};
    std::vector<Complex> occupation_matrix(1, Complex(0.0, 0.0));

    accumulate_hubbard_occupation_matrix(
        1, manifold_size, overlaps, occupations, 1.0, occupation_matrix);

    EXPECT_NEAR(std::abs(occupation_matrix[0]), 0.0, 1e-12);
}

// ─── Time-reversal symmetrization of the occupation matrix ──────────────────

TEST(HubbardOccupationMatrix, TimeReversalAveragingReproducesTheKAndMinusKSum) {
    /**
     * The wedge holds only one member of each ±k pair, and the other one
     * contributes n(-k) = n(k)* for a collinear Hamiltonian. Averaging over
     * time reversal has to give what summing both members would have given.
     *
     * Built as an explicit two-term sum rather than by asserting "the imaginary
     * part is zero", so the test would still fail if the projection kept the
     * wrong half.
     */
    std::vector<Complex> one_k = {
        Complex(1.2, 0.0), Complex(0.3, -0.4), Complex(0.3, 0.4), Complex(0.8, 0.0)};
    std::vector<Complex> both_k(4);
    for (size_t index = 0; index < one_k.size(); index++)
        both_k[index] = 0.5 * (one_k[index] + std::conj(one_k[index]));

    std::vector<std::vector<std::vector<Complex>>> occupation_matrices = {{one_k}};
    symmetrize_hubbard_occupation_matrices_under_time_reversal(occupation_matrices);

    for (size_t index = 0; index < both_k.size(); index++) {
        EXPECT_NEAR(occupation_matrices[0][0][index].real(), both_k[index].real(), 1e-14);
        EXPECT_NEAR(occupation_matrices[0][0][index].imag(), 0.0, 1e-14);
    }
}

TEST(HubbardOccupationMatrix, TimeReversalAveragingLeavesAFullZoneMatrixAlone) {
    /**
     * A full-zone sum already contains both members of every pair, so the
     * average must be a no-op there -- which is what lets it run
     * unconditionally, with no "was symmetry used?" flag threaded through.
     */
    std::vector<Complex> full_zone = {
        Complex(1.2, 0.0), Complex(0.3, 0.0), Complex(0.3, 0.0), Complex(0.8, 0.0)};
    std::vector<std::vector<std::vector<Complex>>> occupation_matrices = {{full_zone}};
    symmetrize_hubbard_occupation_matrices_under_time_reversal(occupation_matrices);

    for (size_t index = 0; index < full_zone.size(); index++)
        EXPECT_EQ(occupation_matrices[0][0][index], full_zone[index]);
}

/**
 * ─── Spatial symmetrization of the occupation matrix ────────────────────────
 *
 * What makes the group average right is that it lands on the crystal's own
 * covariance and stays there:
 *
 *     n_{R(a)} = D(R) n_a D(R)ᵀ   for every operation R and every Hubbard atom a.
 *
 * That relation is also what fixes the one free choice in the formula.
 * `atom_map[a]` is the atom `a` is carried *to*, so the average that reproduces
 * an already-symmetric n pairs that image with D(R)ᵀ · n · D(R) — the pairing
 * `SymmetryAnalyzer::symmetrize_vector` spells out for forces, and says is the
 * only one an already-symmetric field survives.
 *
 * Seeing the other pairing takes a particular cell. Reindexing the group sum by
 * R → R⁻¹ turns one into the other with `atom_map` replaced by its inverse, so
 * the two agree exactly whenever every Hubbard atom's permutation is its own
 * inverse — always true of one or two Hubbard atoms, and of one per species,
 * which is every DFT+U case Q3 is otherwise gated on. It takes an orbit of
 * three or more equivalent atoms carried round a *cycle*, on sites whose own
 * symmetry still leaves the manifold free. The cubic perovskite below is that
 * cell: the [111] three-fold cycles its three oxygens while each keeps a
 * four-fold axis of its own, so its p manifold stays anisotropic.
 */

TEST(HubbardOccupationMatrix, SpatialAveragingLandsOnTheCrystalsCovariance) {
    Crystal crystal_structure = make_cubic_perovskite();
    auto operations = SymmetryAnalyzer::get_symmetry_operations(crystal_structure);
    ASSERT_GT(operations.size(), 1u);

    auto layouts = oxygen_p_manifolds();
    auto occupation_matrices = seeded_occupation_matrices(layouts.size(), OXYGEN_MANIFOLD_SIZE);
    ASSERT_GT(
        hubbard_covariance_violation(crystal_structure, operations, layouts, occupation_matrices),
        1e-3)
        << "the seed has to start off the invariant subspace for this to mean anything";

    symmetrize_hubbard_occupation_matrices(
        crystal_structure, operations, layouts, occupation_matrices);

    EXPECT_LT(
        hubbard_covariance_violation(crystal_structure, operations, layouts, occupation_matrices),
        GROUP_AVERAGE_TOLERANCE);
}

TEST(HubbardOccupationMatrix, SpatialAveragingTwiceChangesNothingTheSecondTime) {
    /**
     * A group average is a projector: P² = P. A contraction that pairs the
     * rotation with the wrong end of the atom map is not one, and shows it by
     * continuing to move on a second pass.
     */
    Crystal crystal_structure = make_cubic_perovskite();
    auto operations = SymmetryAnalyzer::get_symmetry_operations(crystal_structure);
    auto layouts = oxygen_p_manifolds();

    auto once = seeded_occupation_matrices(layouts.size(), OXYGEN_MANIFOLD_SIZE);
    symmetrize_hubbard_occupation_matrices(crystal_structure, operations, layouts, once);
    auto twice = once;
    symmetrize_hubbard_occupation_matrices(crystal_structure, operations, layouts, twice);

    EXPECT_LT(maximum_occupation_difference(once, twice), GROUP_AVERAGE_TOLERANCE);
}

TEST(HubbardOccupationMatrix, SpatialAveragingPreservesTheTrace) {
    /**
     * Tr n is the rotational invariant of a manifold — the charge it carries,
     * and the first term of the Dudarev energy. A rotation cannot move it, so
     * an average over rotations must not either.
     */
    Crystal crystal_structure = make_cubic_perovskite();
    auto operations = SymmetryAnalyzer::get_symmetry_operations(crystal_structure);
    auto layouts = oxygen_p_manifolds();
    auto occupation_matrices = seeded_occupation_matrices(layouts.size(), OXYGEN_MANIFOLD_SIZE);

    Real trace_before = total_trace(occupation_matrices, OXYGEN_MANIFOLD_SIZE);
    symmetrize_hubbard_occupation_matrices(
        crystal_structure, operations, layouts, occupation_matrices);
    Real trace_after = total_trace(occupation_matrices, OXYGEN_MANIFOLD_SIZE);

    EXPECT_NEAR(trace_after, trace_before, 1e-12);
}

TEST(HubbardOccupationMatrix, ATwoAtomCellsCovarianceHoldsToo) {
    /**
     * The exempt class, stated rather than assumed: with two Hubbard atoms the
     * permutations are the identity and a single transposition, both their own
     * inverses, so either pairing lands on the covariance. This is why every
     * DFT+U case Q3 is gated on was — and stays — unaffected.
     */
    Crystal crystal_structure = make_diamond_with_two_hubbard_atoms();
    auto operations = SymmetryAnalyzer::get_symmetry_operations(crystal_structure);
    ASSERT_GT(operations.size(), 1u);

    std::vector<HubbardManifoldLayout> layouts = {{0, 1, 3}, {1, 1, 3}};
    auto occupation_matrices = seeded_occupation_matrices(layouts.size(), 3);
    symmetrize_hubbard_occupation_matrices(
        crystal_structure, operations, layouts, occupation_matrices);

    EXPECT_LT(
        hubbard_covariance_violation(crystal_structure, operations, layouts, occupation_matrices),
        GROUP_AVERAGE_TOLERANCE);
}

// ─── Energy and potential (Dudarev formula) ─────────────────────────────────

TEST(HubbardEnergyAndPotential, ZeroOccupationGivesZeroEnergy) {
    std::vector<Complex> occupation_matrix(4, Complex(0.0, 0.0)); // 2x2 zero matrix
    auto result = compute_hubbard_energy_and_potential(2, 0.3, occupation_matrix);
    EXPECT_NEAR(result.energy, 0.0, 1e-12);
    // D = U*(0.5*I - 0) = 0.5*U*I
    EXPECT_NEAR(result.d_matrix[0].real(), 0.15, 1e-12);
    EXPECT_NEAR(result.d_matrix[3].real(), 0.15, 1e-12);
    EXPECT_NEAR(std::abs(result.d_matrix[1]), 0.0, 1e-12);
}

TEST(HubbardEnergyAndPotential, FullyOccupiedSingleOrbitalGivesZeroEnergy) {
    /**
     * A completely full (n=1) or completely empty (n=0) orbital is not penalized by
     * DFT+U -- only fractional occupation is.
     */
    std::vector<Complex> occupation_matrix = {Complex(1.0, 0.0)};
    auto result = compute_hubbard_energy_and_potential(1, 0.3, occupation_matrix);
    EXPECT_NEAR(result.energy, 0.0, 1e-12);
}

TEST(HubbardEnergyAndPotential, HalfFilledDegenerateManifoldMatchesAnalyticFormula) {
    /**
     * n = 0.5*I for a manifold of size N: E_U = 0.5*U*N*x*(1-x) at x=0.5 => 0.125*U*N.
     * Verified for N=2 (E=0.25*U) and N=1 (E=0.125*U) by hand.
     */
    constexpr Real u_rydberg = 0.4;
    for (int manifold_size : {1, 2, 5}) {
        std::vector<Complex> occupation_matrix(static_cast<size_t>(manifold_size * manifold_size),
                                               Complex(0.0, 0.0));
        for (int m = 0; m < manifold_size; m++) {
            occupation_matrix[static_cast<size_t>(m * manifold_size + m)] = Complex(0.5, 0.0);
        }
        auto result =
            compute_hubbard_energy_and_potential(manifold_size, u_rydberg, occupation_matrix);
        Real expected_energy = 0.125 * u_rydberg * manifold_size;
        EXPECT_NEAR(result.energy, expected_energy, 1e-12) << "manifold_size=" << manifold_size;
        /**
         * At exactly half filling the linear potential term vanishes identically
         * (the symmetric point of the U*x*(1-x) parabola).
         */
        for (const auto& d_element : result.d_matrix) {
            EXPECT_NEAR(std::abs(d_element), 0.0, 1e-12);
        }
    }
}

TEST(HubbardEnergyAndPotential, OffDiagonalOccupationContributesToTraceSquared) {
    /**
     * n = [[0.5, 0.5], [0.5, 0.5]] (Hermitian, trace=1, but off-diagonal-heavy).
     * Tr(n) = 1. Tr(n^2) = sum |n_mm'|^2 = 4*0.25 = 1. E = 0.5*U*(1-1) = 0.
     */
    std::vector<Complex> occupation_matrix = {
        Complex(0.5, 0.0), Complex(0.5, 0.0), Complex(0.5, 0.0), Complex(0.5, 0.0)};
    auto result = compute_hubbard_energy_and_potential(2, 0.4, occupation_matrix);
    EXPECT_NEAR(result.energy, 0.0, 1e-10);
}

TEST(HubbardEnergyAndPotential, DMatrixUsesTransposedOccupationIndex) {
    /**
     * Asymmetric (non-Hermitian, synthetic-only) n to catch an accidental
     * n_mm' <-> n_m'm swap: n_01=0.1, n_10=0.9 (physically n should be Hermitian,
     * but the *formula* itself must index n_m'm, not n_mm', for D_01).
     */
    std::vector<Complex> occupation_matrix = {
        Complex(0.0, 0.0), Complex(0.1, 0.0), Complex(0.9, 0.0), Complex(0.0, 0.0)};
    auto result = compute_hubbard_energy_and_potential(2, 1.0, occupation_matrix);
    // D_01 = U*(0.5*0 - n_10) = -0.9
    EXPECT_NEAR(result.d_matrix[1].real(), -0.9, 1e-12);
    // D_10 = U*(0.5*0 - n_01) = -0.1
    EXPECT_NEAR(result.d_matrix[2].real(), -0.1, 1e-12);
}

/**
 * accumulate_hubbard_force_gradient() validated against a finite difference of
 * compute_hubbard_energy_and_potential()'s own, already-validated energy formula.
 *
 * This is a "frozen wavefunction" test: it fixes an arbitrary (not self-consistent) set of
 * plane-wave coefficients psi(G) and checks that the analytic gradient formula correctly
 * differentiates E_U(tau) at FIXED psi -- exactly the Hellmann-Feynman quantity the force
 * calculator needs (self-consistency, i.e. that E_U's psi-dependence can be neglected at SCF
 * convergence, is a separate, already-established property this codebase relies on for every
 * other Hellmann-Feynman force term, not something specific to Hubbard to re-validate here).
 * No SCF, diagonalization, or convergence is involved -- only the real G-space atomic-orbital
 * construction (atomic_orbital_builder.hpp) at three close-by atomic positions.
 */
TEST(HubbardForceGradient, AnalyticGradientMatchesFiniteDifferenceOfEnergy) {
    auto pseudopotential_set = load_nickel_pseudopotential();
    Crystal origin_crystal = make_nickel_fcc(pseudopotential_set);
    /**
     * Move off the origin onto a generic (non-symmetric) point first: the D shell's (l=2)
     * angular phase i^l = -1 is purely real, so at tau=0 exactly, phi_m(tau=+delta) and
     * phi_m(tau=-delta) are exact complex conjugates of each other for every G -- a special,
     * symmetric point where this particular (fixed-psi) E_U(tau) construction is degenerate
     * (confirmed empirically: FD gave exactly 0 there, regardless of the analytic gradient).
     * Testing around a displaced, generic reference point avoids that degeneracy.
     */
    Crystal base_crystal = perturb_atom_cartesian(origin_crystal, 0, 0, 0.37);
    base_crystal = perturb_atom_cartesian(base_crystal, 0, 1, -0.21);
    base_crystal = perturb_atom_cartesian(base_crystal, 0, 2, 0.13);

    PlanewaveBasis basis;
    basis.generate(serial_policy, base_crystal, 30.0, 120.0);

    KPointsGrid kpoint_grid;
    kpoint_grid.generate(1, 1, 1, 1, 1, 1); // single off-Gamma point (0.5,0.5,0.5) fractional --
                                            /**
                                             * DFT+U rejects Gamma-only in production, so a
                                             * representative test avoids it too.
                                             */

    std::vector<int> g_vector_indices;
    std::vector<Real> kinetic_energy_squared;
    basis.compute_gk(
        kpoint_grid[0].coordinates, base_crystal, g_vector_indices, kinetic_energy_squared);
    int number_of_plane_waves = static_cast<int>(g_vector_indices.size());
    ASSERT_GT(number_of_plane_waves, 0);

    auto atomic_wavefunction_table = tabulate_atomic_wavefunctions_in_q_space(
        pseudopotential_set, base_crystal.cell_volume(), 0.01, 900);

    auto orbital_labels = enumerate_atomic_orbitals(base_crystal, pseudopotential_set);
    std::vector<HubbardSpecification> specifications = {{"Ni", "3D", 0.3}};
    auto manifolds = resolve_hubbard_manifolds(base_crystal, orbital_labels, specifications);
    ASSERT_EQ(manifolds.size(), 1u);
    int manifold_size = static_cast<int>(manifolds[0].orbital_indices.size());
    ASSERT_EQ(manifold_size, 5); // 3D shell, l=2 -> 2l+1=5

    constexpr int number_of_bands = 2;
    std::vector<Real> band_occupations = {1.0, 0.6};
    constexpr Real kpoint_weight = 1.0;

    /**
     * Deterministic, arbitrary "frozen" wavefunction coefficients (not a converged
     * eigenstate -- see the test docstring above for why that's fine here).
     */
    std::vector<std::vector<Complex>> psi(
        static_cast<size_t>(number_of_bands),
        std::vector<Complex>(static_cast<size_t>(number_of_plane_waves)));
    for (int band_index = 0; band_index < number_of_bands; band_index++) {
        for (int g_index = 0; g_index < number_of_plane_waves; g_index++) {
            Real angle = 0.37 * (g_index + 1) + 0.53 * band_index;
            psi[static_cast<size_t>(band_index)][static_cast<size_t>(g_index)] =
                Complex(std::sin(angle), std::cos(1.7 * angle));
        }
    }

    /**
     * -i(k+G) per direction, identical construction to HubbardForceCalculator's own
     * minus_i_k_plus_g.
     */
    const Matrix3x3& reciprocal_lattice_vectors = base_crystal.reciprocal_lattice_vectors();
    Vector3D kpoint_cartesian;
    for (int direction = 0; direction < 3; direction++) {
        kpoint_cartesian[direction] =
            kpoint_grid[0].coordinates.x * reciprocal_lattice_vectors[0][direction] +
            kpoint_grid[0].coordinates.y * reciprocal_lattice_vectors[1][direction] +
            kpoint_grid[0].coordinates.z * reciprocal_lattice_vectors[2][direction];
    }
    std::vector<std::vector<Complex>> minus_i_k_plus_g(
        3, std::vector<Complex>(static_cast<size_t>(number_of_plane_waves)));
    for (int g_index = 0; g_index < number_of_plane_waves; g_index++) {
        const auto& g_vector = basis.g_vectors()[static_cast<size_t>(g_vector_indices[g_index])];
        for (int direction = 0; direction < 3; direction++) {
            Real k_plus_g_direction = (g_vector.g_cart[direction] + kpoint_cartesian[direction]) *
                                      base_crystal.two_pi_over_lattice_parameter();
            minus_i_k_plus_g[static_cast<size_t>(direction)][static_cast<size_t>(g_index)] =
                Complex(0.0, -k_plus_g_direction);
        }
    }

    auto energy_at = [&](const Crystal& crystal_structure) -> Real {
        std::vector<std::vector<Complex>> orbitals;
        build_atomic_orbitals_g_space(0,
                                      number_of_plane_waves,
                                      g_vector_indices,
                                      crystal_structure,
                                      kpoint_grid,
                                      basis,
                                      pseudopotential_set,
                                      atomic_wavefunction_table,
                                      0.01,
                                      900,
                                      orbitals);
        Matrix<Complex> overlaps(static_cast<size_t>(number_of_bands),
                                 static_cast<size_t>(manifold_size));
        for (int band_index = 0; band_index < number_of_bands; band_index++) {
            for (int m = 0; m < manifold_size; m++) {
                int orbital_index = manifolds[0].orbital_indices[static_cast<size_t>(m)];
                Complex overlap_sum(0.0, 0.0);
                for (int g_index = 0; g_index < number_of_plane_waves; g_index++) {
                    overlap_sum +=
                        std::conj(orbitals[static_cast<size_t>(orbital_index)]
                                          [static_cast<size_t>(g_index)]) *
                        psi[static_cast<size_t>(band_index)][static_cast<size_t>(g_index)];
                }
                overlaps(static_cast<size_t>(band_index), static_cast<size_t>(m)) = overlap_sum;
            }
        }
        std::vector<Complex> occupation_matrix(static_cast<size_t>(manifold_size) * manifold_size,
                                               Complex(0.0, 0.0));
        accumulate_hubbard_occupation_matrix(number_of_bands,
                                             manifold_size,
                                             overlaps,
                                             band_occupations,
                                             kpoint_weight,
                                             occupation_matrix);
        return compute_hubbard_energy_and_potential(
                   manifold_size, manifolds[0].u_rydberg, occupation_matrix)
            .energy;
    };

    // Base position: overlaps needed by every direction's analytic gradient.
    std::vector<std::vector<Complex>> orbitals_base;
    build_atomic_orbitals_g_space(0,
                                  number_of_plane_waves,
                                  g_vector_indices,
                                  base_crystal,
                                  kpoint_grid,
                                  basis,
                                  pseudopotential_set,
                                  atomic_wavefunction_table,
                                  0.01,
                                  900,
                                  orbitals_base);
    Matrix<Complex> overlaps_base(static_cast<size_t>(number_of_bands),
                                  static_cast<size_t>(manifold_size));
    for (int band_index = 0; band_index < number_of_bands; band_index++) {
        for (int m = 0; m < manifold_size; m++) {
            int orbital_index = manifolds[0].orbital_indices[static_cast<size_t>(m)];
            Complex overlap_sum(0.0, 0.0);
            for (int g_index = 0; g_index < number_of_plane_waves; g_index++) {
                overlap_sum += std::conj(orbitals_base[static_cast<size_t>(orbital_index)]
                                                      [static_cast<size_t>(g_index)]) *
                               psi[static_cast<size_t>(band_index)][static_cast<size_t>(g_index)];
            }
            overlaps_base(static_cast<size_t>(band_index), static_cast<size_t>(m)) = overlap_sum;
        }
    }
    std::vector<Complex> occupation_matrix_base(static_cast<size_t>(manifold_size) * manifold_size,
                                                Complex(0.0, 0.0));
    accumulate_hubbard_occupation_matrix(number_of_bands,
                                         manifold_size,
                                         overlaps_base,
                                         band_occupations,
                                         kpoint_weight,
                                         occupation_matrix_base);
    auto result_base = compute_hubbard_energy_and_potential(
        manifold_size, manifolds[0].u_rydberg, occupation_matrix_base);

    constexpr Real delta_bohr = 1e-4;
    for (int direction = 0; direction < 3; direction++) {
        SCOPED_TRACE(testing::Message() << "direction=" << direction);

        Real energy_plus =
            energy_at(perturb_atom_cartesian(base_crystal, 0, direction, delta_bohr));
        Real energy_minus =
            energy_at(perturb_atom_cartesian(base_crystal, 0, direction, -delta_bohr));
        Real numeric_energy_gradient = (energy_plus - energy_minus) / (2.0 * delta_bohr);

        Matrix<Complex> direction_derivative_overlaps(static_cast<size_t>(number_of_bands),
                                                      static_cast<size_t>(manifold_size));
        for (int band_index = 0; band_index < number_of_bands; band_index++) {
            for (int m = 0; m < manifold_size; m++) {
                int orbital_index = manifolds[0].orbital_indices[static_cast<size_t>(m)];
                Complex derivative_sum(0.0, 0.0);
                for (int g_index = 0; g_index < number_of_plane_waves; g_index++) {
                    Complex orbital_conjugate =
                        std::conj(orbitals_base[static_cast<size_t>(orbital_index)]
                                               [static_cast<size_t>(g_index)]);
                    Complex psi_value =
                        psi[static_cast<size_t>(band_index)][static_cast<size_t>(g_index)];
                    derivative_sum +=
                        orbital_conjugate * (minus_i_k_plus_g[static_cast<size_t>(direction)]
                                                             [static_cast<size_t>(g_index)] *
                                             psi_value);
                }
                direction_derivative_overlaps(static_cast<size_t>(band_index),
                                              static_cast<size_t>(m)) = derivative_sum;
            }
        }

        Real analytic_force = accumulate_hubbard_force_gradient(number_of_bands,
                                                                manifold_size,
                                                                overlaps_base,
                                                                direction_derivative_overlaps,
                                                                band_occupations,
                                                                kpoint_weight,
                                                                result_base.d_matrix);

        /**
         * Force = -dE/dtau. Relative tolerance: the analytic force's absolute scale here is an
         * artifact of the unnormalized synthetic psi (~10-16, not a physical Ry/bohr
         * magnitude), so an absolute tolerance tuned for a real force calculator's typical
         * scale would be either meaninglessly loose or fail on this test's own arbitrary
         * units. Observed agreement is ~7 significant figures (relative error ~1e-7),
         * consistent with O(delta^2) central-difference truncation error at this delta_bohr.
         */
        Real relative_tolerance = 1e-5 * std::abs(analytic_force);
        EXPECT_NEAR(analytic_force, -numeric_energy_gradient, relative_tolerance)
            << "analytic force=" << analytic_force
            << ", -dE/dtau (FD)=" << -numeric_energy_gradient;
    }
}

/**
 * ── compute_hubbard_projector_overlaps ───────────────────────────────────────
 *
 * The ZGEMM batching that replaced the scalar (band x orbital x plane-wave) nests in
 * both the SCF iterator and HubbardForceCalculator. These tests pin the layout
 * contract and the equivalence to the scalar formulation, since a transposition or
 * leading-dimension slip in a GEMM is silent -- it produces plausible numbers that
 * simply belong to a different matrix.
 */

namespace {

/// Reference: the scalar triple loop the ZGEMM replaced, in the same output layout.
std::vector<Complex> reference_projector_overlaps(int number_of_plane_waves,
                                                  int total_hubbard_orbitals,
                                                  int number_of_bands,
                                                  const std::vector<Complex>& packed_orbitals,
                                                  const std::vector<Complex>& band_functions) {
    std::vector<Complex> overlaps(static_cast<size_t>(total_hubbard_orbitals) * number_of_bands,
                                  Complex(0.0, 0.0));
    for (int band_index = 0; band_index < number_of_bands; band_index++) {
        const Complex* band =
            band_functions.data() + static_cast<size_t>(band_index) * number_of_plane_waves;
        for (int orbital_index = 0; orbital_index < total_hubbard_orbitals; orbital_index++) {
            const Complex* orbital =
                packed_orbitals.data() + static_cast<size_t>(orbital_index) * number_of_plane_waves;
            Complex sum(0.0, 0.0);
            for (int g_index = 0; g_index < number_of_plane_waves; g_index++) {
                sum += std::conj(orbital[g_index]) * band[g_index];
            }
            overlaps[static_cast<size_t>(band_index) * total_hubbard_orbitals + orbital_index] =
                sum;
        }
    }
    return overlaps;
}

} // namespace

TEST(HubbardProjectorOverlaps, MatchesScalarReferenceFormulation) {
    const int number_of_plane_waves = 97;
    const int total_hubbard_orbitals = 10; // two d-shell manifolds
    const int number_of_bands = 13;

    /**
     * Deterministic pseudo-random operands; the exact values do not matter, only that
     * they are dense and complex enough that a transposition would show up.
     */
    std::vector<Complex> packed_orbitals(static_cast<size_t>(number_of_plane_waves) *
                                         total_hubbard_orbitals);
    std::vector<Complex> band_functions(static_cast<size_t>(number_of_plane_waves) *
                                        number_of_bands);
    for (size_t i = 0; i < packed_orbitals.size(); i++) {
        packed_orbitals[i] = Complex(std::sin(0.7 * static_cast<Real>(i) + 0.3),
                                     std::cos(1.1 * static_cast<Real>(i) - 0.2));
    }
    for (size_t i = 0; i < band_functions.size(); i++) {
        band_functions[i] = Complex(std::cos(0.5 * static_cast<Real>(i) + 1.7),
                                    std::sin(0.9 * static_cast<Real>(i) + 0.4));
    }

    std::vector<Complex> overlaps;
    compute_hubbard_projector_overlaps(number_of_plane_waves,
                                       total_hubbard_orbitals,
                                       number_of_bands,
                                       packed_orbitals.data(),
                                       band_functions.data(),
                                       overlaps);

    auto reference = reference_projector_overlaps(number_of_plane_waves,
                                                  total_hubbard_orbitals,
                                                  number_of_bands,
                                                  packed_orbitals,
                                                  band_functions);

    ASSERT_EQ(overlaps.size(), reference.size());
    /**
     * BLAS reassociates the sum over plane waves, so this is rounding-level agreement,
     * not bit-identity. The scale of the reference elements here is O(10), so ~1e-12 is
     * several orders below any physically meaningful difference.
     */
    for (size_t i = 0; i < reference.size(); i++) {
        EXPECT_NEAR(overlaps[i].real(), reference[i].real(), 1e-12) << "element " << i;
        EXPECT_NEAR(overlaps[i].imag(), reference[i].imag(), 1e-12) << "element " << i;
    }
}

TEST(HubbardProjectorOverlaps, ConjugatesTheOrbitalNotTheBand) {
    /**
     * <phi|chi> must conjugate phi. With phi = i and chi = 1 over a single plane wave,
     * the answer is -i; conjugating the wrong operand would give +i, and conjugating
     * neither would give +i as well -- so this single element pins the convention.
     */
    std::vector<Complex> orbital{Complex(0.0, 1.0)};
    std::vector<Complex> band{Complex(1.0, 0.0)};

    std::vector<Complex> overlaps;
    compute_hubbard_projector_overlaps(1, 1, 1, orbital.data(), band.data(), overlaps);

    ASSERT_EQ(overlaps.size(), 1u);
    EXPECT_NEAR(overlaps[0].real(), 0.0, 1e-15);
    EXPECT_NEAR(overlaps[0].imag(), -1.0, 1e-15);
}

TEST(HubbardProjectorOverlaps, IsColumnMajorInBandIndex) {
    /**
     * Element (orbital j, band b) must live at j + b * total_hubbard_orbitals -- the
     * layout both call sites index with when scattering per-manifold row blocks.
     * Orbital 0 is a delta on plane wave 0, orbital 1 a delta on plane wave 1; band b
     * is the constant (b+1), so the expected matrix has distinct values everywhere.
     */
    const int number_of_plane_waves = 2;
    const int total_hubbard_orbitals = 2;
    const int number_of_bands = 3;

    std::vector<Complex> packed_orbitals{{1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {1.0, 0.0}};
    std::vector<Complex> band_functions{
        {1.0, 0.0}, {1.0, 0.0}, {2.0, 0.0}, {2.0, 0.0}, {3.0, 0.0}, {3.0, 0.0}};

    std::vector<Complex> overlaps;
    compute_hubbard_projector_overlaps(number_of_plane_waves,
                                       total_hubbard_orbitals,
                                       number_of_bands,
                                       packed_orbitals.data(),
                                       band_functions.data(),
                                       overlaps);

    ASSERT_EQ(overlaps.size(), 6u);
    for (int band_index = 0; band_index < number_of_bands; band_index++) {
        for (int orbital_index = 0; orbital_index < total_hubbard_orbitals; orbital_index++) {
            SCOPED_TRACE("orbital " + std::to_string(orbital_index) + ", band " +
                         std::to_string(band_index));
            EXPECT_NEAR(overlaps[static_cast<size_t>(orbital_index) +
                                 static_cast<size_t>(band_index) * total_hubbard_orbitals]
                            .real(),
                        static_cast<Real>(band_index + 1),
                        1e-15);
        }
    }
}

TEST(HubbardProjectorOverlaps, ZeroOccupiedBandsGivesEmptyResult) {
    /**
     * A k-point where every band is below the occupation threshold must not call into
     * BLAS with a zero dimension; both call sites rely on getting an empty result back.
     */
    std::vector<Complex> orbital{Complex(1.0, 0.0)};
    std::vector<Complex> overlaps{Complex(7.0, 7.0)}; // pre-filled: must be cleared

    compute_hubbard_projector_overlaps(1, 1, 0, orbital.data(), nullptr, overlaps);

    EXPECT_TRUE(overlaps.empty());
}

/**
 * A mixed cell -- one ultrasoft species, one norm-conserving -- with the
 * augmentation lookup that the DFT+U force and stress kernels perform per atom.
 *
 * `augmentation_overlap_integral` reaches its dataset through
 * `PseudopotentialSet::as_ultrasoft`, a `dynamic_cast` whose null check is an
 * `assert` and therefore absent from Release builds. Asking a norm-conserving
 * species for an augmentation overlap returns null and dereferences it, so the
 * failure appears only in the builds users run and only for cells with more
 * than one kind of pseudopotential. Single-species test cells cannot see it,
 * which is how four of the five hand-rolled copies of this assembly kept the
 * bug.
 *
 * The gate lives in `hubbard_augmentation_overlap` so all of them share it.
 */
TEST(HubbardAugmentationOverlap, NormConservingSpeciesInAMixedCellReturnsZero) {
    Crystal crystal_structure;
    crystal_structure.set_from_lattice_type(2, 10.263048);
    crystal_structure.add_species(AtomSpecies{"Si", 28.08, 4.0, "Si_PBE_USPP.UPF"});
    crystal_structure.add_species(AtomSpecies{"C", 12.01, 4.0, "C.upf"});
    crystal_structure.add_atom({0, {0.0, 0.0, 0.0}});
    crystal_structure.add_atom({1, {0.25, 0.25, 0.25}});
    crystal_structure.compute_derived_quantities();

    PseudopotentialSet pseudopotential_set;
    pseudopotential_set.push_back(
        UPFReader::read(std::string(EXAMPLES_DIR) + "/data/pseudopotentials/us/Si_PBE_USPP.UPF"));
    pseudopotential_set.push_back(
        UPFReader::read(std::string(EXAMPLES_DIR) + "/data/pseudopotentials/nc/C.upf"));

    // The premise of the test: exactly one of the two species carries them.
    ASSERT_TRUE(pseudopotential_set[0].has_augmentation_charges());
    ASSERT_FALSE(pseudopotential_set[1].has_augmentation_charges());

    PlanewaveBasis planewave_basis;
    planewave_basis.generate(serial_policy, crystal_structure, 16.0, 64.0);
    auto pseudopotential_basis = NormConservingPlanewaveBasisFactory::create(
        crystal_structure, planewave_basis, pseudopotential_set);
    ASSERT_NE(pseudopotential_basis, nullptr);
    ASSERT_TRUE(pseudopotential_basis->is_ultrasoft());
    const auto& ultrasoft_basis =
        static_cast<const UltrasoftPlanewaveBasis&>(*pseudopotential_basis);

    // Without the species gate this call dereferences null in a Release build.
    constexpr int norm_conserving_species = 1;
    ASSERT_GT(pseudopotential_basis->number_of_projector_channels(norm_conserving_species), 0);
    EXPECT_EQ(hubbard_augmentation_overlap(
                  ultrasoft_basis, *pseudopotential_basis, norm_conserving_species, 0, 0),
              0.0);

    /**
     * The gate must not be a blanket zero: the ultrasoft species still has to
     * report its q_ij, or the derivative kernels would silently drop the
     * augmentation term they exist to carry.
     */
    constexpr int ultrasoft_species = 0;
    EXPECT_NE(hubbard_augmentation_overlap(
                  ultrasoft_basis, *pseudopotential_basis, ultrasoft_species, 0, 0),
              0.0);
}
