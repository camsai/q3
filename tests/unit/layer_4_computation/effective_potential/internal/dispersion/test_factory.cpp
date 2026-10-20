#include <gtest/gtest.h>

#include <stdexcept>

#include "layer_1_structure/crystal/crystal.hpp"
#include "layer_4_computation/effective_potential/internal/dispersion/factory.hpp"

using namespace q3;

namespace {

Crystal make_silicon_fcc() {
    Crystal crystal_structure;
    crystal_structure.set_from_lattice_type(2, 10.263048); // FCC silicon, same as test_d3bj.cpp
    AtomSpecies silicon{"Si", 28.08, 4.0, "Si.pz-vbc.UPF"};
    crystal_structure.add_species(silicon);
    crystal_structure.add_atom({0, {0.0, 0.0, 0.0}});
    crystal_structure.add_atom({0, {0.25, 0.25, 0.25}});
    crystal_structure.compute_derived_quantities();
    return crystal_structure;
}

} // namespace

// ── Scheme resolution ───────────────────────────────────────────────

TEST(DispersionCalculatorFactory, NoneResolvesToNullCalculator) {
    Crystal crystal_structure = make_silicon_fcc();
    auto calculator = create_dispersion_calculator("none", crystal_structure);
    EXPECT_EQ(calculator->name(), "none");
}

TEST(DispersionCalculatorFactory, EmptyStringResolvesToNullCalculator) {
    Crystal crystal_structure = make_silicon_fcc();
    auto calculator = create_dispersion_calculator("", crystal_structure);
    EXPECT_EQ(calculator->name(), "none");
}

TEST(DispersionCalculatorFactory, D3bjResolvesToD3bjCalculator) {
    Crystal crystal_structure = make_silicon_fcc();
    auto calculator = create_dispersion_calculator("d3bj", crystal_structure);
    EXPECT_EQ(calculator->name(), "d3bj");
    /**
     * Same physical value the direct-construction tests in test_d3bj.cpp pin
     * against QE -- here only asserting the factory wired up the real scheme,
     * not the Null object (which would return exactly zero).
     */
    EXPECT_LT(calculator->compute_energy(), 0.0);
}

TEST(DispersionCalculatorFactory, UnknownSchemeThrows) {
    /**
     * An unrecognized scheme must fail loudly, not silently run without the
     * correction the user asked for (the pre-factory behavior).
     */
    Crystal crystal_structure = make_silicon_fcc();
    EXPECT_THROW(create_dispersion_calculator("d3", crystal_structure), std::runtime_error);
    EXPECT_THROW(create_dispersion_calculator("ts", crystal_structure), std::runtime_error);
}

/**
 * ── Null object contract ────────────────────────────────────────────
 *
 * The orchestrators (forces/stress assembly, SCF energy hookup) call the
 * dispersion interface unconditionally, so the "none" scheme must contribute
 * exact zeros of the right shape.
 */

TEST(NullDispersionCalculator, EnergyIsExactlyZero) {
    Crystal crystal_structure = make_silicon_fcc();
    auto calculator = create_dispersion_calculator("none", crystal_structure);
    EXPECT_EQ(calculator->compute_energy(), 0.0);
}

TEST(NullDispersionCalculator, GradientsAreZeroWithOneForcePerAtom) {
    Crystal crystal_structure = make_silicon_fcc();
    auto calculator = create_dispersion_calculator("none", crystal_structure);
    DispersionGradients gradients = calculator->compute_gradients();

    ASSERT_EQ(gradients.forces.size(),
              static_cast<std::size_t>(crystal_structure.number_of_atoms()));
    for (const auto& force : gradients.forces) {
        EXPECT_EQ(force.x, 0.0);
        EXPECT_EQ(force.y, 0.0);
        EXPECT_EQ(force.z, 0.0);
    }
    for (int direction_a = 0; direction_a < 3; direction_a++) {
        for (int direction_b = 0; direction_b < 3; direction_b++) {
            EXPECT_EQ(gradients.stress[direction_a][direction_b], 0.0);
        }
    }
}
