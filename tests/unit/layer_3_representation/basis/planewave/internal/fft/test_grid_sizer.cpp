#include <gtest/gtest.h>

#include "layer_1_structure/crystal/crystal.hpp"
#include "layer_3_representation/basis/planewave/internal/fft/grid_sizer.hpp"

using namespace q3;

/**
 * FCC silicon's cubic lattice parameter (Bohr), and the same crystal described as
 * a 2x2x2 supercell of it. The cutoffs are the ones the committed silicon decks
 * use (examples/total_energy--semiconductor--silicon/nc.yml and nc_2x2x2.yml):
 * 192 Ry for the charge density, and min(2 * 48, 192) = 96 Ry for the smooth
 * wavefunction grid, as PlanewaveFFTPair pairs them.
 */
static constexpr Real PRIMITIVE_LATTICE_PARAMETER = 10.263;
static constexpr Real SUPERCELL_LATTICE_PARAMETER = 2.0 * PRIMITIVE_LATTICE_PARAMETER;
static constexpr Real CHARGE_DENSITY_CUTOFF = 192.0;
static constexpr Real SMOOTH_GRID_CUTOFF = 96.0;

static Crystal make_silicon_crystal(Real lattice_parameter,
                                    const std::vector<Vector3D>& fractional_positions) {
    Crystal crystal_structure;
    crystal_structure.set_from_lattice_type(2, lattice_parameter);
    crystal_structure.add_species(AtomSpecies{"Si", 28.08, 4.0, "Si.upf"});
    for (const Vector3D& fractional_position : fractional_positions) {
        crystal_structure.add_atom({0, fractional_position});
    }
    crystal_structure.compute_derived_quantities();
    return crystal_structure;
}

/// Two-atom primitive cell, at the coordinates of examples/data/materials/silicon.json.
static Crystal make_primitive_silicon() {
    return make_silicon_crystal(PRIMITIVE_LATTICE_PARAMETER,
                                {{0.875, 0.875, 0.875}, {0.125, 0.125, 0.125}});
}

/**
 * The same crystal as a 16-atom 2x2x2 supercell, at the coordinates of
 * examples/data/materials/silicon_2x2x2.json. Every same-species pair here
 * differs by 0.125, 0.375 or 0.5 along each axis; only the halves are
 * translations of the space group.
 */
static Crystal make_silicon_2x2x2_supercell() {
    return make_silicon_crystal(SUPERCELL_LATTICE_PARAMETER,
                                {{0.4375, 0.4375, 0.4375},
                                 {0.0625, 0.0625, 0.0625},
                                 {0.4375, 0.4375, 0.9375},
                                 {0.0625, 0.0625, 0.5625},
                                 {0.4375, 0.9375, 0.4375},
                                 {0.0625, 0.5625, 0.0625},
                                 {0.4375, 0.9375, 0.9375},
                                 {0.0625, 0.5625, 0.5625},
                                 {0.9375, 0.4375, 0.4375},
                                 {0.5625, 0.0625, 0.0625},
                                 {0.9375, 0.4375, 0.9375},
                                 {0.5625, 0.0625, 0.5625},
                                 {0.9375, 0.9375, 0.4375},
                                 {0.5625, 0.5625, 0.0625},
                                 {0.9375, 0.9375, 0.9375},
                                 {0.5625, 0.5625, 0.5625}});
}

/// Dense and smooth grid dimensions for one crystal, the way PlanewaveFFTPair sizes them.
static void compute_grid_dimensions(const Crystal& crystal_structure,
                                    Real energy_cutoff,
                                    int grid_dimensions[3]) {
    int fft_factors[3];
    FFTGridSizer::compute_fft_factors(crystal_structure, fft_factors);
    FFTGridSizer::calculate_grid_dimensions(
        energy_cutoff,
        crystal_structure.two_pi_over_lattice_parameter_squared(),
        crystal_structure.lattice_vectors(),
        crystal_structure.reciprocal_lattice_vectors(),
        grid_dimensions[0],
        grid_dimensions[1],
        grid_dimensions[2],
        fft_factors);
}

// ─── get_good_fft_order (no division factor) ────────────────────────────────

TEST(FFTGridSizer, GoodFFTOrderPreservesAlreadyGoodSizes) {
    // Sizes that are already products of {2,3,5} should be returned as-is
    EXPECT_EQ(FFTGridSizer::get_good_fft_order(1), 1);
    EXPECT_EQ(FFTGridSizer::get_good_fft_order(2), 2);
    EXPECT_EQ(FFTGridSizer::get_good_fft_order(4), 4);
    EXPECT_EQ(FFTGridSizer::get_good_fft_order(6), 6);
    EXPECT_EQ(FFTGridSizer::get_good_fft_order(16), 16);
    EXPECT_EQ(FFTGridSizer::get_good_fft_order(30), 30);
}

TEST(FFTGridSizer, GoodFFTOrderRoundsPrimesUp) {
    // Primes that are not 2, 3, or 5 must be rounded up
    EXPECT_EQ(FFTGridSizer::get_good_fft_order(7), 8);   // 7 → 8 = 2^3
    EXPECT_EQ(FFTGridSizer::get_good_fft_order(11), 12); // 11 → 12 = 2^2 * 3
    EXPECT_EQ(FFTGridSizer::get_good_fft_order(13), 15); // 13 → 15 = 3 * 5
    EXPECT_EQ(FFTGridSizer::get_good_fft_order(17), 18); // 17 → 18 = 2 * 3^2
    EXPECT_EQ(FFTGridSizer::get_good_fft_order(31), 32); // 31 → 32 = 2^5
}

TEST(FFTGridSizer, GoodFFTOrderRoundsCompositeWithBadFactorUp) {
    // 14 = 2 * 7 contains factor 7, must round up to 15 = 3 * 5
    EXPECT_EQ(FFTGridSizer::get_good_fft_order(14), 15);
}

// ─── get_good_fft_order (with division factor) ─────────────────────────────

TEST(FFTGridSizer, GoodFFTOrderWithDivisionFactor) {
    // Result must be >= input, divisible by factor, and a product of {2,3,5}
    EXPECT_EQ(FFTGridSizer::get_good_fft_order(7, 2), 8);   // 8 = 2^3, divisible by 2
    EXPECT_EQ(FFTGridSizer::get_good_fft_order(7, 3), 9);   // 9 = 3^2, divisible by 3
    EXPECT_EQ(FFTGridSizer::get_good_fft_order(10, 4), 12); // 12 = 2^2 * 3, divisible by 4
}

// ─── get_good_fft_dimension ─────────────────────────────────────────────────

TEST(FFTGridSizer, GoodFFTDimensionMatchesOrderWithFactorOne) {
    EXPECT_EQ(FFTGridSizer::get_good_fft_dimension(1), FFTGridSizer::get_good_fft_order(1, 1));
    EXPECT_EQ(FFTGridSizer::get_good_fft_dimension(7), FFTGridSizer::get_good_fft_order(7, 1));
    EXPECT_EQ(FFTGridSizer::get_good_fft_dimension(13), FFTGridSizer::get_good_fft_order(13, 1));
    EXPECT_EQ(FFTGridSizer::get_good_fft_dimension(31), FFTGridSizer::get_good_fft_order(31, 1));
}

// ─── fft_factor_for_fractional_translation ──────────────────────────────────

TEST(FFTGridSizer, FractionalTranslationFactorRejectsWhatIsNoFraction) {
    /**
     * 0.375 is the offset between the two sublattices of a 2x2x2 silicon
     * supercell. round(1/0.375) is 3, and taking that unchecked forced a grid
     * divisible by 3 onto a cell whose translations are halves.
     */
    EXPECT_EQ(FFTGridSizer::fft_factor_for_fractional_translation(0.375), 1);
    EXPECT_EQ(FFTGridSizer::fft_factor_for_fractional_translation(-0.375), 1);
    EXPECT_EQ(FFTGridSizer::fft_factor_for_fractional_translation(0.625), 1); // -0.375 folded
    EXPECT_EQ(FFTGridSizer::fft_factor_for_fractional_translation(0.4), 1);
    EXPECT_EQ(FFTGridSizer::fft_factor_for_fractional_translation(0.3), 1);
}

TEST(FFTGridSizer, FractionalTranslationFactorAcceptsGenuineFractions) {
    EXPECT_EQ(FFTGridSizer::fft_factor_for_fractional_translation(0.5), 2);
    EXPECT_EQ(FFTGridSizer::fft_factor_for_fractional_translation(1.0 / 3.0), 3);
    EXPECT_EQ(FFTGridSizer::fft_factor_for_fractional_translation(0.25), 4);
    EXPECT_EQ(FFTGridSizer::fft_factor_for_fractional_translation(1.0 / 6.0), 6);
}

TEST(FFTGridSizer, FractionalTranslationFactorFoldsIntoTheHalfOpenHalfInterval) {
    // spglib reports translations in [0, 1); a translation and its negative ask the same.
    EXPECT_EQ(FFTGridSizer::fft_factor_for_fractional_translation(2.0 / 3.0), 3);
    EXPECT_EQ(FFTGridSizer::fft_factor_for_fractional_translation(0.75), 4);
    EXPECT_EQ(FFTGridSizer::fft_factor_for_fractional_translation(5.0 / 6.0), 6);
    EXPECT_EQ(FFTGridSizer::fft_factor_for_fractional_translation(1.0), 1);
}

TEST(FFTGridSizer, FractionalTranslationFactorToleratesSymmetryPrecisionNoise) {
    // The translations carry the precision spglib was given, not exact thirds.
    EXPECT_EQ(FFTGridSizer::fft_factor_for_fractional_translation(1.0 / 3.0 + 1e-6), 3);
    EXPECT_EQ(FFTGridSizer::fft_factor_for_fractional_translation(0.5 - 1e-6), 2);
}

TEST(FFTGridSizer, FractionalTranslationFactorIgnoresDenominatorsTheGridCannotHonour) {
    /**
     * Only 2, 3, 4 and 6 are accepted: those are the denominators a space
     * group's own fractional translations take, and get_good_fft_order() can
     * only ever satisfy a factor whose primes are 2, 3 and 5.
     */
    EXPECT_EQ(FFTGridSizer::fft_factor_for_fractional_translation(0.0), 1);
    EXPECT_EQ(FFTGridSizer::fft_factor_for_fractional_translation(0.2), 1);   // 1/5
    EXPECT_EQ(FFTGridSizer::fft_factor_for_fractional_translation(0.125), 1); // 1/8
    EXPECT_EQ(FFTGridSizer::fft_factor_for_fractional_translation(1.0 / 7.0), 1);
}

// ─── compute_fft_factors ────────────────────────────────────────────────────

TEST(FFTGridSizer, PrimitiveSiliconAsksForTheHalvesItsSpaceGroupCarries) {
    /**
     * Diamond's 48 operations carry four distinct translations in the primitive
     * basis: zero, and a half along each axis.
     */
    int fft_factors[3];
    FFTGridSizer::compute_fft_factors(make_primitive_silicon(), fft_factors);

    EXPECT_EQ(fft_factors[0], 2);
    EXPECT_EQ(fft_factors[1], 2);
    EXPECT_EQ(fft_factors[2], 2);
}

TEST(FFTGridSizer, SiliconSupercellScalesThatConstraintAndInventsNoThird) {
    /**
     * The 2x2x2 cell's 384 operations carry the primitive lattice's own
     * translations, halves here, and the primitive cell's halves, which are
     * quarters here -- so it asks for 4, the same spacing constraint one cell
     * finer. What it must not ask for is 3: that came from reading the 0.375
     * between its two sublattices as 1/3, and no translation of this group is a
     * third of anything.
     */
    int fft_factors[3];
    FFTGridSizer::compute_fft_factors(make_silicon_2x2x2_supercell(), fft_factors);

    for (int dimension = 0; dimension < 3; dimension++) {
        EXPECT_EQ(fft_factors[dimension], 4) << "axis " << dimension;
        EXPECT_NE(fft_factors[dimension] % 3, 0) << "axis " << dimension;
    }
}

TEST(FFTGridSizer, SupercellFactorsAreThePrimitiveOnesScaledByTheCellMultiplicity) {
    /**
     * Two descriptions of one crystal impose one constraint on the grid spacing,
     * so a cell twice as long must be divisible by twice as much. That is what
     * makes the two cells' real-space grids commensurate, and it is the relation
     * a factor read off atom pairs had no reason to respect.
     */
    int primitive_factors[3];
    int supercell_factors[3];
    FFTGridSizer::compute_fft_factors(make_primitive_silicon(), primitive_factors);
    FFTGridSizer::compute_fft_factors(make_silicon_2x2x2_supercell(), supercell_factors);

    for (int dimension = 0; dimension < 3; dimension++) {
        EXPECT_EQ(supercell_factors[dimension], 2 * primitive_factors[dimension])
            << "axis " << dimension;
    }
}

TEST(FFTGridSizer, SupercellGridsAreTwiceThePrimitiveOnesAtTheCommittedCutoffs) {
    /**
     * A supercell and its primitive cell describe the same crystal, so the energy
     * per formula unit agrees only while their real-space grids do -- the
     * supercell's spacing must be the primitive cell's. The spurious factor of 3
     * left that intact at these cutoffs by accident; nothing was checking it.
     */
    const Crystal primitive_silicon = make_primitive_silicon();
    const Crystal supercell_silicon = make_silicon_2x2x2_supercell();

    for (Real energy_cutoff : {CHARGE_DENSITY_CUTOFF, SMOOTH_GRID_CUTOFF}) {
        int primitive_dimensions[3];
        int supercell_dimensions[3];
        compute_grid_dimensions(primitive_silicon, energy_cutoff, primitive_dimensions);
        compute_grid_dimensions(supercell_silicon, energy_cutoff, supercell_dimensions);

        for (int dimension = 0; dimension < 3; dimension++) {
            EXPECT_EQ(supercell_dimensions[dimension], 2 * primitive_dimensions[dimension])
                << "cutoff " << energy_cutoff << " Ry, axis " << dimension;
        }
    }
}
