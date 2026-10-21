/**
 * The point-group average of the PAW on-site occupations.
 *
 * What makes a symmetrization right is not that it changes something — almost
 * any averaging does — but that it lands on the crystal's own covariance
 * relation and stays there:
 *
 *     ρ^{R(a)} = D(R) ρ^a D(R)ᵀ   for every operation R and every atom a.
 *
 * That relation is also what pins the one free choice in the formula. The
 * mistake available here is to pair `atom_map`, which carries a *to* R(a),
 * with D(R) rather than D(R)ᵀ — the pairing
 * `SymmetryAnalyzer::symmetrize_vector` spells out for forces, and says is the
 * only one an already-symmetric field survives.
 *
 * **What it takes to see that swap.** Reindexing the group sum by R → R⁻¹
 * turns the wrong average into the right one with `atom_map` replaced by its
 * inverse, so the two agree exactly whenever every atom permutation is its own
 * inverse — which is *always* true of a two-atom cell, whose permutations can
 * only be the identity or the swap. Diamond and every other two-atom case
 * therefore cannot tell them apart, and neither can a cell of one atom per
 * species, where `atom_map` is the identity throughout. It takes an orbit of
 * three or more equivalent atoms carrying a cycle, on sites whose own symmetry
 * still leaves the manifold some freedom: the cubic perovskite below, whose
 * three oxygens are cycled by the [111] three-fold while each sits on a
 * four-fold axis that keeps its p manifold anisotropic. That is the test that
 * fails on the swap; the diamond ones are here for the physics, not the
 * convention.
 */

#include "layer_1_structure/crystal/crystal.hpp"
#include "layer_1_structure/symmetry/symmetry.hpp"
#include "layer_4_computation/paw/one_center/harmonic_rotation.hpp"
#include "layer_4_computation/paw/one_center/occupation_symmetrization.hpp"
#include <gtest/gtest.h>

#include <cmath>
#include <vector>

using namespace q3;
using namespace q3::paw_one_center;

namespace {

/// Diamond: FCC, two carbon atoms, 48 operations of which 24 swap them.
Crystal make_diamond() {
    Crystal crystal_structure;
    crystal_structure.set_from_lattice_type(2, 6.74); // FCC, alat in Bohr
    crystal_structure.add_species(AtomSpecies{"C", 12.011, 4.0, "C.upf"});
    crystal_structure.add_atom({0, {0.0, 0.0, 0.0}});
    crystal_structure.add_atom({0, {0.25, 0.25, 0.25}});
    crystal_structure.compute_derived_quantities();
    return crystal_structure;
}

/**
 * Cubic perovskite ABO₃: the three oxygens sit at the face centres and are
 * cycled, not swapped, by the three-fold rotation about [111]. Each keeps a
 * four-fold axis of its own, so its p manifold stays anisotropic — a and b in
 * diag(a, b, b) — and the group average has to carry each oxygen's axis to the
 * right neighbour rather than to the one two steps along.
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

/// One s projector and one p projector: channels are (s), (p_z, p_x, p_y).
ChannelAngularLayout make_s_plus_p_layout() {
    return build_channel_angular_layout({0, 1, 1, 1}, {0, 1, 1, 1});
}

/// Two s and one d projector, the copper-shaped case: 1 + 1 + 5 channels.
ChannelAngularLayout make_two_s_plus_d_layout() {
    return build_channel_angular_layout({0, 1, 2, 2, 2, 2, 2}, {0, 0, 2, 2, 2, 2, 2});
}

/**
 * A reproducible, deliberately unsymmetric ρ_ij: every entry distinct, and
 * symmetric in (i,j) the way an accumulated ρ_ij always is.
 */
Matrix<Real> make_seed_occupations(int number_of_atoms, int channels) {
    Matrix<Real> occupations(
        static_cast<size_t>(number_of_atoms), static_cast<size_t>(channels) * channels, 0.0);
    for (int atom = 0; atom < number_of_atoms; atom++) {
        for (int first = 0; first < channels; first++) {
            for (int second = first; second < channels; second++) {
                Real value = 0.1 + 0.37 * std::sin(1.0 + atom + 3.0 * first + 7.0 * second);
                occupations[atom][first * channels + second] = value;
                occupations[atom][second * channels + first] = value;
            }
        }
    }
    return occupations;
}

OnSiteOccupationSymmetrizer make_symmetrizer(const Crystal& crystal_structure,
                                             const std::vector<SymmetryOperation>& operations,
                                             std::vector<ChannelAngularLayout> layouts) {
    OnSiteOccupationSymmetrizer symmetrizer;
    symmetrizer.build(crystal_structure, operations, std::move(layouts));
    return symmetrizer;
}

/// max |ρ^{R(a)} − D(R) ρ^a D(R)ᵀ| over every operation, atom and channel pair.
Real covariance_violation(const Crystal& crystal_structure,
                          const std::vector<SymmetryOperation>& operations,
                          const ChannelAngularLayout& layout,
                          const Matrix<Real>& occupations) {
    int channels = layout.number_of_channels;
    Real worst = 0.0;

    for (const auto& operation : operations) {
        Matrix3x3 cartesian = cartesian_rotation_of(crystal_structure, operation);
        std::vector<std::vector<Real>> rotation_by_channel(static_cast<size_t>(channels));
        for (int channel = 0; channel < channels; channel++)
            rotation_by_channel[static_cast<size_t>(channel)] =
                build_real_harmonic_rotation_matrix(layout.angular_momentum[channel], cartesian);

        for (int atom = 0; atom < crystal_structure.number_of_atoms(); atom++) {
            int image = operation.atom_map[atom];
            for (int first = 0; first < channels; first++) {
                int size_first = 2 * layout.angular_momentum[first] + 1;
                for (int second = 0; second < channels; second++) {
                    int size_second = 2 * layout.angular_momentum[second] + 1;
                    // (D ρ^a Dᵀ)_{first second}
                    Real rotated = 0.0;
                    for (int p = 0; p < size_first; p++) {
                        Real left = rotation_by_channel
                            [static_cast<size_t>(first)]
                            [static_cast<size_t>(layout.manifold_position[first]) * size_first + p];
                        for (int q = 0; q < size_second; q++) {
                            Real right = rotation_by_channel[static_cast<size_t>(
                                second)][static_cast<size_t>(layout.manifold_position[second]) *
                                             size_second +
                                         q];
                            rotated +=
                                left * right *
                                occupations[atom][(layout.manifold_base[first] + p) * channels +
                                                  layout.manifold_base[second] + q];
                        }
                    }
                    Real difference =
                        std::abs(occupations[image][first * channels + second] - rotated);
                    worst = std::max(worst, difference);
                }
            }
        }
    }
    return worst;
}

Real maximum_difference(const Matrix<Real>& left, const Matrix<Real>& right) {
    Real worst = 0.0;
    for (size_t index = 0; index < left.raw().size(); index++)
        worst = std::max(worst, std::abs(left.raw()[index] - right.raw()[index]));
    return worst;
}

constexpr Real GROUP_AVERAGE_TOLERANCE = 1e-10;

} // namespace

// ─── The covariance the average has to land on ──────────────────────────────

TEST(PawOccupationSymmetrizationTest, TheAveragedOccupationsObeyTheCrystalsCovariance) {
    Crystal crystal_structure = make_diamond();
    auto operations = SymmetryAnalyzer::get_symmetry_operations(crystal_structure);
    ASSERT_GT(operations.size(), 1u);
    auto layout = make_s_plus_p_layout();

    auto occupations = make_seed_occupations(2, layout.number_of_channels);
    ASSERT_GT(covariance_violation(crystal_structure, operations, layout, occupations), 1e-3)
        << "the seed has to start off the invariant subspace for this to mean anything";

    make_symmetrizer(crystal_structure, operations, {layout})
        .symmetrize(crystal_structure, occupations);

    EXPECT_LT(covariance_violation(crystal_structure, operations, layout, occupations),
              GROUP_AVERAGE_TOLERANCE);
}

TEST(PawOccupationSymmetrizationTest, CovarianceHoldsWhereAThreeFoldCyclesEquivalentAtoms) {
    /**
     * The gate that separates the two pairings — see the note at the top of
     * this file for why no two-atom cell can. The oxygens' p manifolds are the
     * part that moves: pairing `atom_map` with D instead of Dᵀ sends each
     * oxygen's four-fold axis one step too far around the cycle, and the
     * result stops satisfying the crystal's own covariance.
     */
    Crystal crystal_structure = make_cubic_perovskite();
    auto operations = SymmetryAnalyzer::get_symmetry_operations(crystal_structure);
    ASSERT_GT(operations.size(), 1u);
    auto layout = make_s_plus_p_layout();

    auto occupations = make_seed_occupations(5, layout.number_of_channels);
    make_symmetrizer(crystal_structure, operations, {layout, layout, layout})
        .symmetrize(crystal_structure, occupations);

    EXPECT_LT(covariance_violation(crystal_structure, operations, layout, occupations),
              GROUP_AVERAGE_TOLERANCE);
}

TEST(PawOccupationSymmetrizationTest, AveragingTwiceChangesNothingOnACycledOrbitEither) {
    /**
     * P² = P on the same cell. A contraction that is not a group average is
     * not a projector, and shows it by continuing to move on a second pass.
     */
    Crystal crystal_structure = make_cubic_perovskite();
    auto operations = SymmetryAnalyzer::get_symmetry_operations(crystal_structure);
    auto layout = make_s_plus_p_layout();
    auto symmetrizer = make_symmetrizer(crystal_structure, operations, {layout, layout, layout});

    auto once = make_seed_occupations(5, layout.number_of_channels);
    symmetrizer.symmetrize(crystal_structure, once);
    auto twice = once;
    symmetrizer.symmetrize(crystal_structure, twice);

    EXPECT_LT(maximum_difference(once, twice), GROUP_AVERAGE_TOLERANCE);
}

TEST(PawOccupationSymmetrizationTest, AveragingTwiceChangesNothingTheSecondTime) {
    /**
     * A group average is a projector: P² = P. A contraction that pairs the
     * rotation with the wrong end of the atom map is not one, and shows it
     * here by continuing to move.
     */
    Crystal crystal_structure = make_diamond();
    auto operations = SymmetryAnalyzer::get_symmetry_operations(crystal_structure);
    auto layout = make_two_s_plus_d_layout();
    auto symmetrizer = make_symmetrizer(crystal_structure, operations, {layout});

    auto once = make_seed_occupations(2, layout.number_of_channels);
    symmetrizer.symmetrize(crystal_structure, once);
    auto twice = once;
    symmetrizer.symmetrize(crystal_structure, twice);

    EXPECT_LT(maximum_difference(once, twice), GROUP_AVERAGE_TOLERANCE);
}

// ─── What the average must not disturb ──────────────────────────────────────

TEST(PawOccupationSymmetrizationTest, EachRadialBlocksTraceIsPreserved) {
    /**
     * Σ_m ρ_{(β m)(β' m)} over a manifold is the rotational invariant of that
     * radial block — the on-site charge the block carries, which the grid
     * density and the compensation charge already agree on. A rotation cannot
     * move it, so an average over rotations must not either.
     */
    Crystal crystal_structure = make_diamond();
    auto operations = SymmetryAnalyzer::get_symmetry_operations(crystal_structure);
    auto layout = make_two_s_plus_d_layout();
    int channels = layout.number_of_channels;

    auto occupations = make_seed_occupations(2, channels);
    Real trace_before = 0.0;
    for (int atom = 0; atom < 2; atom++)
        for (int channel = 0; channel < channels; channel++)
            trace_before += occupations[atom][channel * channels + channel];

    make_symmetrizer(crystal_structure, operations, {layout})
        .symmetrize(crystal_structure, occupations);

    Real trace_after = 0.0;
    for (int atom = 0; atom < 2; atom++)
        for (int channel = 0; channel < channels; channel++)
            trace_after += occupations[atom][channel * channels + channel];

    EXPECT_NEAR(trace_after, trace_before, 1e-12);
}

TEST(PawOccupationSymmetrizationTest, ATetrahedralSitesPManifoldComesOutIsotropic) {
    /**
     * The defect, in its smallest form. Carbon in diamond sits on a site whose
     * own group forbids an anisotropic p block, so an irreducible-wedge sum
     * that produced one was wrong on its face — and that anisotropy is what
     * split the Γ valence maximum by 19 meV.
     */
    Crystal crystal_structure = make_diamond();
    auto operations = SymmetryAnalyzer::get_symmetry_operations(crystal_structure);
    auto layout = make_s_plus_p_layout();
    int channels = layout.number_of_channels;

    Matrix<Real> occupations(2, static_cast<size_t>(channels) * channels, 0.0);
    for (int atom = 0; atom < 2; atom++) {
        occupations[atom][0] = 1.5;                // s
        occupations[atom][1 * channels + 1] = 0.9; // p_z, the odd one out
        occupations[atom][2 * channels + 2] = 0.3; // p_x
        occupations[atom][3 * channels + 3] = 0.3; // p_y
    }

    make_symmetrizer(crystal_structure, operations, {layout})
        .symmetrize(crystal_structure, occupations);

    for (int atom = 0; atom < 2; atom++) {
        EXPECT_NEAR(occupations[atom][0], 1.5, 1e-12) << "atom " << atom;
        for (int channel = 1; channel <= 3; channel++)
            EXPECT_NEAR(occupations[atom][channel * channels + channel], 0.5, 1e-10)
                << "atom " << atom << ", p channel " << channel;
        // The s-p cross terms an isotropic site cannot carry.
        EXPECT_NEAR(occupations[atom][1], 0.0, 1e-10) << "atom " << atom;
    }
}

TEST(PawOccupationSymmetrizationTest, EquivalentAtomsEndUpCarryingTheSameCharge) {
    Crystal crystal_structure = make_diamond();
    auto operations = SymmetryAnalyzer::get_symmetry_operations(crystal_structure);
    auto layout = make_s_plus_p_layout();
    int channels = layout.number_of_channels;

    auto occupations = make_seed_occupations(2, channels);
    make_symmetrizer(crystal_structure, operations, {layout})
        .symmetrize(crystal_structure, occupations);

    for (int channel = 0; channel < channels; channel++)
        EXPECT_NEAR(occupations[0][channel * channels + channel],
                    occupations[1][channel * channels + channel],
                    1e-12)
            << "channel " << channel;
}

// ─── Standing down ──────────────────────────────────────────────────────────

TEST(PawOccupationSymmetrizationTest, TheIdentityAloneLeavesTheSymmetrizerInactive) {
    Crystal crystal_structure = make_diamond();
    SymmetryOperation identity;
    identity.rotation = {{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}};
    identity.reciprocal_rotation = identity.rotation;
    identity.fractional_translation = {0.0, 0.0, 0.0};
    identity.atom_map = {0, 1};

    auto symmetrizer = make_symmetrizer(crystal_structure, {identity}, {make_s_plus_p_layout()});
    EXPECT_FALSE(symmetrizer.is_active());

    auto occupations = make_seed_occupations(2, 4);
    auto untouched = occupations;
    symmetrizer.symmetrize(crystal_structure, occupations);
    EXPECT_EQ(maximum_difference(occupations, untouched), 0.0);
}

// ─── The channel layout ─────────────────────────────────────────────────────

TEST(PawOccupationSymmetrizationTest, ChannelLayoutGroupsEachRadialProjectorsMValues) {
    // Two p projectors and one s, the order a UPF can perfectly well list them in.
    auto layout = build_channel_angular_layout({0, 0, 0, 1, 2, 2, 2}, {1, 1, 1, 0, 1, 1, 1});

    EXPECT_EQ(layout.number_of_channels, 7);
    EXPECT_EQ(layout.manifold_base, (std::vector<int>{0, 0, 0, 3, 4, 4, 4}));
    EXPECT_EQ(layout.manifold_position, (std::vector<int>{0, 1, 2, 0, 0, 1, 2}));
}
