#include <gtest/gtest.h>

#include <stdexcept>
#include <vector>

#include "layer_1_structure/crystal/crystal.hpp"
#include "layer_4_computation/state/kohn_sham/planewave/norm_conserving/init.hpp"
#include "layer_5_properties/isolated_system_correction/planewave/makov_payne.hpp"

using namespace q3;

namespace {

std::vector<AtomSpecies> single_silicon_species() {
    return {AtomSpecies{"Si", 28.0855, 4.0, "Si.upf"}};
}

std::vector<Atom> single_atom_at_origin() {
    return {Atom{0, {0.0, 0.0, 0.0}}};
}

} // namespace

/**
 * ── Cubic-lattice gate ──────────────────────────────────────────────
 *
 * Makov-Payne's Madelung constant is only tabulated for cubic lattices (QE
 * ibrav 1/2/3 = SC/FCC/BCC). This must throw immediately (before touching any
 * charge-density data, which is why these tests can leave the rest of the SCF
 * state at its default/null values) rather than silently applying a wrong
 * constant or defaulting to 0. ibrav=-3 (BCC's alternate-axes convention) is
 * deliberately included in the rejected set: it is not simply "BCC with other
 * axes" here, because the Madelung-constant lookup is keyed by the ibrav index
 * itself and -3 lies outside the cubic 1..3 range. ibrav=0 is rejected for the
 * same reason -- note this is stricter than Quantum ESPRESSO, which resolves
 * ibrav=0 to a Bravais index from the lattice vectors before applying its bound.
 */

TEST(MakovPayneCorrectionCalculator, RejectsExplicitFreeLatticeVectors) {
    Matrix3x3 identity_lattice{{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};
    Crystal crystal_structure = Crystal::from_lattice_vectors(
        10.0, identity_lattice, single_silicon_species(), single_atom_at_origin());
    ASSERT_EQ(crystal_structure.bravais_lattice_index(), 0);

    NormConservingPlanewaveKohnShamSCFState scf_state;
    scf_state.crystal_structure = &crystal_structure;

    EXPECT_THROW(MakovPayneCorrectionCalculator{}.compute(scf_state), std::runtime_error);
}

TEST(MakovPayneCorrectionCalculator, RejectsBccAlternateAxesConvention) {
    Crystal crystal_structure =
        Crystal::from_input_data(-3, 10.0, single_silicon_species(), single_atom_at_origin());
    ASSERT_EQ(crystal_structure.bravais_lattice_index(), -3);

    NormConservingPlanewaveKohnShamSCFState scf_state;
    scf_state.crystal_structure = &crystal_structure;

    EXPECT_THROW(MakovPayneCorrectionCalculator{}.compute(scf_state), std::runtime_error);
}

TEST(MakovPayneCorrectionCalculator, RejectsHexagonalLattice) {
    Crystal crystal_structure =
        Crystal::from_input_data(4, 10.0, single_silicon_species(), single_atom_at_origin());
    ASSERT_EQ(crystal_structure.bravais_lattice_index(), 4);

    NormConservingPlanewaveKohnShamSCFState scf_state;
    scf_state.crystal_structure = &crystal_structure;

    EXPECT_THROW(MakovPayneCorrectionCalculator{}.compute(scf_state), std::runtime_error);
}
