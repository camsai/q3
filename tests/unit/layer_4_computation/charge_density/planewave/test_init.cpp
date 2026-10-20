/**
 * PlanewaveChargeDensitySymmetrizer: the symmetrized density must be invariant under every
 * operation of the crystal's space group -- including the pure translations that a supercell
 * carries -- and symmetrizing twice must change nothing.
 *
 * Until 2026-09-10 the symmetrizer dropped every operation with a fractional translation once
 * it detected a supercell, keeping 12 of the 384 operations of a 2x2x2 diamond cell; the
 * supercell then converged to a different density than its primitive cell (camsai/q3#6). The
 * supercell tests below fail on that code at the operation count and at invariance.
 *
 * Until 2026-10-09 it also skipped every G-vector that all rotations leave in place, which in
 * a polar group is where a screw axis or glide forbids components outright: wurtzite kept its
 * (0, 0, odd) density, lost the 6_3 screw, and split the levels it makes degenerate at K, A, L
 * and H (camsai/q3#24). The wurtzite tests below fail on that code at invariance.
 */

#include "layer_0_core/types/constants.hpp"
#include "layer_0_core/types/init.hpp"
#include "layer_1_structure/crystal/crystal.hpp"
#include "layer_1_structure/symmetry/symmetry.hpp"
#include "layer_3_representation/basis/planewave/init.hpp"
#include "layer_4_computation/charge_density/planewave/init.hpp"
#include "layer_4_computation/iterator/kohn_sham/planewave/execution/serial.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <map>
#include <random>
#include <vector>

#include <gtest/gtest.h>

using namespace q3;

namespace {

execution::PlanewaveSerialExecutionPolicy serial_policy;

constexpr Real primitive_lattice_constant_bohr = 10.263;
constexpr Real wavefunction_cutoff_rydberg = 12.0;
constexpr Real charge_density_cutoff_rydberg = 48.0;
/// Averages of O(1) components over at most 384 terms.
constexpr Real invariance_tolerance = 1e-12;

using MillerIndices = std::array<int, 3>;

/**
 * The committed silicon.json: diamond with the origin at the bond centre, so 12 of the 48
 * operations are symmorphic and the other 36 carry a fractional translation.
 */
Crystal make_bond_centred_silicon() {
    Crystal crystal;
    crystal.set_from_lattice_type(2, primitive_lattice_constant_bohr);
    crystal.add_species(AtomSpecies{"Si", 28.08, 4.0, "Si.pz-vbc.UPF"});
    crystal.add_atom({0, {0.125, 0.125, 0.125}});
    crystal.add_atom({0, {0.875, 0.875, 0.875}});
    crystal.compute_derived_quantities();
    return crystal;
}

/**
 * Its exact 2x2x2 replica (the committed silicon_2x2x2.json): 16 atoms at (tau + n)/2. The
 * space group has 48 x 8 = 384 operations, eight of them pure translations of the primitive
 * lattice.
 */
Crystal make_bond_centred_silicon_2x2x2() {
    Crystal crystal;
    crystal.set_from_lattice_type(2, 2.0 * primitive_lattice_constant_bohr);
    crystal.add_species(AtomSpecies{"Si", 28.08, 4.0, "Si.pz-vbc.UPF"});
    for (Real tau : {0.125, 0.875}) {
        for (int n1 : {0, 1}) {
            for (int n2 : {0, 1}) {
                for (int n3 : {0, 1}) {
                    crystal.add_atom({0, {(tau + n1) / 2.0, (tau + n2) / 2.0, (tau + n3) / 2.0}});
                }
            }
        }
    }
    crystal.compute_derived_quantities();
    return crystal;
}

/**
 * Wurtzite 2H-SiC, P6_3mc: polar, so no operation takes G to -G, and its 6_3 screw and
 * c-glides translate by c/2 along the axis every rotation fixes. Six of the twelve operations
 * carry that translation, which forbids every (0, 0, l) with l odd.
 */
Crystal make_wurtzite_silicon_carbide() {
    constexpr Real lattice_constant_bohr = 5.8215;
    constexpr Real c_over_a = 1.6386;
    constexpr Real internal_parameter = 0.3748;
    Crystal crystal;
    crystal.set_from_lattice_type(4, lattice_constant_bohr, 0.0, c_over_a, 0.0, 0.0, 0.0);
    crystal.add_species(AtomSpecies{"Si", 28.0855, 4.0, "Si.upf"});
    crystal.add_species(AtomSpecies{"C", 12.011, 4.0, "C.upf"});
    crystal.add_atom({0, {1.0 / 3.0, 2.0 / 3.0, 0.0}});
    crystal.add_atom({0, {2.0 / 3.0, 1.0 / 3.0, 0.5}});
    crystal.add_atom({1, {1.0 / 3.0, 2.0 / 3.0, internal_parameter}});
    crystal.add_atom({1, {2.0 / 3.0, 1.0 / 3.0, internal_parameter + 0.5}});
    crystal.compute_derived_quantities();
    return crystal;
}

void generate_basis(const Crystal& crystal, PlanewaveBasis& basis) {
    // Not gamma-only (that path skips the symmetrizer); estimated FFT plans, for speed.
    basis.generate(serial_policy,
                   crystal,
                   wavefunction_cutoff_rydberg,
                   charge_density_cutoff_rydberg,
                   /*is_gamma_only=*/false,
                   /*use_measured_fft_planning=*/false);
}

std::map<MillerIndices, int> miller_index_map(const PlanewaveBasis& basis) {
    std::map<MillerIndices, int> index_of;
    const auto& g_vectors = basis.g_vectors();
    for (int g_vector_index = 0; g_vector_index < basis.number_of_g_vectors(); g_vector_index++) {
        const auto& miller = g_vectors[g_vector_index].miller;
        index_of[{miller.x, miller.y, miller.z}] = g_vector_index;
    }
    return index_of;
}

/**
 * Seeded pseudo-random components with rho(-G) = conj(rho(G)), so the density is real in
 * real space and nothing about the input is symmetric beyond that.
 */
std::vector<Complex> hermitian_pseudo_random_density(const PlanewaveBasis& basis,
                                                     const std::map<MillerIndices, int>& index_of,
                                                     unsigned seed) {
    std::mt19937 generator(seed);
    std::uniform_real_distribution<Real> uniform(-1.0, 1.0);
    const auto& g_vectors = basis.g_vectors();
    int number_of_g_vectors = basis.number_of_g_vectors();
    std::vector<Complex> density(number_of_g_vectors, Complex(0.0, 0.0));
    std::vector<bool> assigned(number_of_g_vectors, false);
    for (int g_vector_index = 0; g_vector_index < number_of_g_vectors; g_vector_index++) {
        if (assigned[g_vector_index]) {
            continue;
        }
        const auto& miller = g_vectors[g_vector_index].miller;
        auto partner = index_of.find({-miller.x, -miller.y, -miller.z});
        int partner_index = (partner == index_of.end()) ? g_vector_index : partner->second;
        Real real_part = uniform(generator);
        Real imaginary_part = (partner_index == g_vector_index) ? 0.0 : uniform(generator);
        density[g_vector_index] = Complex(real_part, imaginary_part);
        density[partner_index] = std::conj(density[g_vector_index]);
        assigned[g_vector_index] = true;
        assigned[partner_index] = true;
    }
    return density;
}

MillerIndices rotate(const SymmetryOperation& operation, const MillerIndices& miller) {
    const auto& rotation = operation.reciprocal_rotation;
    MillerIndices rotated{};
    for (int row = 0; row < 3; row++) {
        rotated[row] = rotation[row][0] * miller[0] + rotation[row][1] * miller[1] +
                       rotation[row][2] * miller[2];
    }
    return rotated;
}

/**
 * Largest violation of rho(G) = rho(S G) exp(+2 pi i (S G) . t) over every operation {S|t}
 * and every G whose image S G is also inside the sphere -- the invariance the symmetrizer is
 * meant to enforce, in the source-phase convention it uses.
 */
Real max_invariance_violation(const PlanewaveBasis& basis,
                              const std::map<MillerIndices, int>& index_of,
                              const std::vector<SymmetryOperation>& operations,
                              const std::vector<Complex>& density) {
    Real worst = 0.0;
    const auto& g_vectors = basis.g_vectors();
    for (const auto& operation : operations) {
        const auto& translation = operation.fractional_translation;
        for (int g_vector_index = 0; g_vector_index < basis.number_of_g_vectors();
             g_vector_index++) {
            const auto& miller = g_vectors[g_vector_index].miller;
            MillerIndices rotated = rotate(operation, {miller.x, miller.y, miller.z});
            auto image = index_of.find(rotated);
            if (image == index_of.end()) {
                continue;
            }
            Real argument =
                constants::two_pi * (rotated[0] * translation.x + rotated[1] * translation.y +
                                     rotated[2] * translation.z);
            Complex phase(std::cos(argument), std::sin(argument));
            worst =
                std::max(worst, std::abs(density[g_vector_index] - density[image->second] * phase));
        }
    }
    return worst;
}

Real max_difference(const std::vector<Complex>& first, const std::vector<Complex>& second) {
    Real worst = 0.0;
    for (std::size_t index = 0; index < first.size(); index++) {
        worst = std::max(worst, std::abs(first[index] - second[index]));
    }
    return worst;
}

std::vector<Complex> symmetrized(const PlanewaveChargeDensitySymmetrizer& symmetrizer,
                                 const PlanewaveBasis& basis,
                                 const std::vector<Complex>& density) {
    std::vector<std::vector<Complex>> g_components{density};
    std::vector<std::vector<Real>> real_space(1);
    symmetrizer.symmetrize_from_g_space(basis, g_components, real_space, 1);
    return g_components[0];
}

} // namespace

TEST(PlanewaveChargeDensitySymmetrizer, PrimitiveCellAveragesOverAll48Operations) {
    Crystal crystal = make_bond_centred_silicon();
    PlanewaveBasis basis;
    generate_basis(crystal, basis);
    auto operations = SymmetryAnalyzer::get_symmetry_operations(crystal);
    ASSERT_EQ(operations.size(), 48u);

    PlanewaveChargeDensitySymmetrizer symmetrizer;
    symmetrizer.build_shells(crystal, basis);
    ASSERT_TRUE(symmetrizer.is_initialized());
    EXPECT_EQ(symmetrizer.number_of_symmetry_operations(), 48);

    auto index_of = miller_index_map(basis);
    auto density = hermitian_pseudo_random_density(basis, index_of, 7);
    ASSERT_GT(max_invariance_violation(basis, index_of, operations, density), 0.1)
        << "the input must not be symmetric already, or the test proves nothing";

    auto once = symmetrized(symmetrizer, basis, density);
    EXPECT_LT(max_invariance_violation(basis, index_of, operations, once), invariance_tolerance);
    EXPECT_GT(max_difference(once, density), 0.1) << "symmetrization did nothing";
    EXPECT_EQ(once[index_of.at({0, 0, 0})], density[index_of.at({0, 0, 0})])
        << "the G = 0 component (the total charge) must pass through untouched";
    EXPECT_EQ(symmetrizer.number_of_systematically_absent_g_vectors(), 0)
        << "inversion pairs every G with -G, so no component is forbidden outright";

    auto twice = symmetrized(symmetrizer, basis, once);
    EXPECT_LT(max_difference(twice, once), invariance_tolerance) << "not idempotent";
}

TEST(PlanewaveChargeDensitySymmetrizer, SupercellKeepsThePureTranslations) {
    Crystal crystal = make_bond_centred_silicon_2x2x2();
    PlanewaveBasis basis;
    generate_basis(crystal, basis);
    auto operations = SymmetryAnalyzer::get_symmetry_operations(crystal);
    ASSERT_EQ(operations.size(), 384u);

    PlanewaveChargeDensitySymmetrizer symmetrizer;
    symmetrizer.build_shells(crystal, basis);
    ASSERT_TRUE(symmetrizer.is_initialized());
    EXPECT_EQ(symmetrizer.number_of_symmetry_operations(), 384)
        << "every operation spglib reports, pure translations included";

    auto index_of = miller_index_map(basis);
    auto density = hermitian_pseudo_random_density(basis, index_of, 11);
    ASSERT_GT(max_invariance_violation(basis, index_of, operations, density), 0.1);

    auto once = symmetrized(symmetrizer, basis, density);
    EXPECT_LT(max_invariance_violation(basis, index_of, operations, once), invariance_tolerance);

    /**
     * Averaging over the eight sublattice translations n/2 multiplies rho(G) by
     * (1/8) sum_n (-1)^(G.n), which vanishes unless every Miller index is even: the symmetrized
     * density has the periodicity of the primitive cell the supercell repeats.
     */
    const auto& g_vectors = basis.g_vectors();
    Real largest_odd_component = 0.0;
    int number_of_odd_g_vectors = 0;
    for (int g_vector_index = 0; g_vector_index < basis.number_of_g_vectors(); g_vector_index++) {
        const auto& miller = g_vectors[g_vector_index].miller;
        if (miller.x % 2 != 0 || miller.y % 2 != 0 || miller.z % 2 != 0) {
            number_of_odd_g_vectors++;
            largest_odd_component = std::max(largest_odd_component, std::abs(once[g_vector_index]));
        }
    }
    ASSERT_GT(number_of_odd_g_vectors, 0);
    EXPECT_LT(largest_odd_component, invariance_tolerance);

    auto twice = symmetrized(symmetrizer, basis, once);
    EXPECT_LT(max_difference(twice, once), invariance_tolerance) << "not idempotent";
}

TEST(PlanewaveChargeDensitySymmetrizer, SupercellFoldsOntoThePrimitiveCell) {
    Crystal primitive = make_bond_centred_silicon();
    Crystal supercell = make_bond_centred_silicon_2x2x2();
    PlanewaveBasis primitive_basis;
    PlanewaveBasis supercell_basis;
    generate_basis(primitive, primitive_basis);
    generate_basis(supercell, supercell_basis);

    PlanewaveChargeDensitySymmetrizer primitive_symmetrizer;
    PlanewaveChargeDensitySymmetrizer supercell_symmetrizer;
    primitive_symmetrizer.build_shells(primitive, primitive_basis);
    supercell_symmetrizer.build_shells(supercell, supercell_basis);

    auto primitive_index_of = miller_index_map(primitive_basis);
    auto supercell_index_of = miller_index_map(supercell_basis);
    auto primitive_density =
        hermitian_pseudo_random_density(primitive_basis, primitive_index_of, 3);
    auto supercell_density =
        hermitian_pseudo_random_density(supercell_basis, supercell_index_of, 5);

    /**
     * The primitive G-vector g is the supercell G-vector 2g: plant the primitive density there,
     * leaving the odd (non-folding) components pseudo-random.
     */
    int number_planted = 0;
    for (const auto& [miller, primitive_index] : primitive_index_of) {
        auto folded = supercell_index_of.find({2 * miller[0], 2 * miller[1], 2 * miller[2]});
        if (folded == supercell_index_of.end()) {
            continue; // sphere-boundary rounding differs between the two cells
        }
        supercell_density[folded->second] = primitive_density[primitive_index];
        number_planted++;
    }
    ASSERT_GT(number_planted, static_cast<int>(0.99 * primitive_index_of.size()));

    auto primitive_symmetrized =
        symmetrized(primitive_symmetrizer, primitive_basis, primitive_density);
    auto supercell_symmetrized =
        symmetrized(supercell_symmetrizer, supercell_basis, supercell_density);

    Real worst = 0.0;
    for (const auto& [miller, primitive_index] : primitive_index_of) {
        auto folded = supercell_index_of.find({2 * miller[0], 2 * miller[1], 2 * miller[2]});
        if (folded == supercell_index_of.end()) {
            continue;
        }
        worst = std::max(worst,
                         std::abs(supercell_symmetrized[folded->second] -
                                  primitive_symmetrized[primitive_index]));
    }
    EXPECT_LT(worst, invariance_tolerance)
        << "the supercell's even components must be the primitive cell's symmetrized density";
}

TEST(PlanewaveChargeDensitySymmetrizer, WurtziteScrewAxisRemovesTheComponentsItForbids) {
    Crystal crystal = make_wurtzite_silicon_carbide();
    PlanewaveBasis basis;
    generate_basis(crystal, basis);
    auto operations = SymmetryAnalyzer::get_symmetry_operations(crystal);
    ASSERT_EQ(operations.size(), 12u);

    PlanewaveChargeDensitySymmetrizer symmetrizer;
    symmetrizer.build_shells(crystal, basis);
    ASSERT_TRUE(symmetrizer.is_initialized());

    auto index_of = miller_index_map(basis);
    auto density = hermitian_pseudo_random_density(basis, index_of, 13);
    ASSERT_GT(max_invariance_violation(basis, index_of, operations, density), 0.1);

    auto once = symmetrized(symmetrizer, basis, density);
    EXPECT_LT(max_invariance_violation(basis, index_of, operations, once), invariance_tolerance);

    /**
     * Every rotation fixes (0, 0, l), so the average multiplies it by the mean of its six
     * phases 1 and six phases (-1)^l: odd l vanishes, even l passes through exactly.
     */
    int number_of_odd_components = 0;
    int number_of_even_components = 0;
    for (int l = 1; index_of.count({0, 0, l}) > 0; l++) {
        for (int sign : {1, -1}) {
            int g_vector_index = index_of.at({0, 0, sign * l});
            if (l % 2 != 0) {
                EXPECT_EQ(once[g_vector_index], Complex(0.0, 0.0)) << "l = " << sign * l;
                number_of_odd_components++;
            } else {
                EXPECT_EQ(once[g_vector_index], density[g_vector_index]) << "l = " << sign * l;
                number_of_even_components++;
            }
        }
    }
    ASSERT_GT(number_of_odd_components, 0);
    ASSERT_GT(number_of_even_components, 0);
    EXPECT_EQ(symmetrizer.number_of_systematically_absent_g_vectors(), number_of_odd_components)
        << "in P6_3mc the axis is the only direction every rotation fixes";

    auto twice = symmetrized(symmetrizer, basis, once);
    EXPECT_LT(max_difference(twice, once), invariance_tolerance) << "not idempotent";
}

TEST(PlanewaveChargeDensitySymmetrizer, RealSpaceEntryPointRemovesThemToo) {
    Crystal crystal = make_wurtzite_silicon_carbide();
    PlanewaveBasis basis;
    generate_basis(crystal, basis);
    auto operations = SymmetryAnalyzer::get_symmetry_operations(crystal);

    PlanewaveChargeDensitySymmetrizer symmetrizer;
    symmetrizer.build_shells(crystal, basis);

    auto index_of = miller_index_map(basis);
    auto density = hermitian_pseudo_random_density(basis, index_of, 17);
    std::vector<std::vector<Real>> real_space(1);
    basis.rho_g_to_r(density, real_space[0]);
    symmetrizer.symmetrize(basis, real_space, 1);
    std::vector<Complex> round_trip;
    basis.rho_r_to_g(real_space[0], round_trip);

    // Two transforms more than the G-space path, so the comparison allows for their rounding.
    constexpr Real transform_tolerance = 1e-10;
    EXPECT_LT(max_invariance_violation(basis, index_of, operations, round_trip),
              transform_tolerance);
    EXPECT_LT(max_difference(round_trip, symmetrized(symmetrizer, basis, density)),
              transform_tolerance);
    EXPECT_LT(std::abs(round_trip[index_of.at({0, 0, 1})]), transform_tolerance);
}
