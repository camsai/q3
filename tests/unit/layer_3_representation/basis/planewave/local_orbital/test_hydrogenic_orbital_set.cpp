#include <gtest/gtest.h>

#include <cmath>
#include <complex>
#include <memory>
#include <vector>

#include "io/pseudopotential/upf_reader.hpp"
#include "layer_1_structure/crystal/crystal.hpp"
#include "layer_1_structure/reciprocal_space/kpoints_grid.hpp"
#include "layer_3_representation/basis/planewave/init.hpp"
#include "layer_3_representation/basis/planewave/local_orbital/factory.hpp"
#include "layer_3_representation/basis/planewave/local_orbital/hydrogenic_orbital_set.hpp"
#include "layer_3_representation/basis/planewave/local_orbital/pseudo_atomic_orbital_set.hpp"
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

struct Fixture {
    PseudopotentialSet pseudopotential_set;
    Crystal crystal_structure;
    PlanewaveBasis basis;
    KPointsGrid kpoint_grid;
    std::vector<int> g_vector_indices;
    int number_of_plane_waves = 0;

    Fixture()
        : pseudopotential_set(load_silicon_pseudopotential())
        , crystal_structure(make_silicon_fcc(pseudopotential_set)) {
        basis.generate(serial_policy, crystal_structure, 24.0, 96.0);
        // A shifted grid, so the k-point exercised below is not Gamma.
        kpoint_grid.generate(2, 2, 2, 1, 1, 1);
        std::vector<Real> kinetic_energy_squared;
        basis.compute_gk(kpoint_grid[0].coordinates,
                         crystal_structure,
                         g_vector_indices,
                         kinetic_energy_squared);
        number_of_plane_waves = static_cast<int>(g_vector_indices.size());
    }
};

Complex inner_product(const std::vector<Complex>& left, const std::vector<Complex>& right) {
    Complex sum(0.0, 0.0);
    for (std::size_t index = 0; index < left.size(); index++) {
        sum += std::conj(left[index]) * right[index];
    }
    return sum;
}

} // namespace

TEST(PlanewaveHydrogenicOrbitalSet, RendersOneCoefficientVectorPerOrbital) {
    Fixture fixture;
    PlanewaveHydrogenicOrbitalSet set({{Vector3D(0.0, 0.0, 0.0), 1.0},
                                       {Vector3D(0.5, 0.0, 0.0), 1.0},
                                       {Vector3D(0.0, 0.5, 0.0), 1.2}});
    EXPECT_EQ(set.number_of_orbitals(), 3);
    std::vector<std::vector<Complex>> orbitals;
    set.build(0,
              fixture.number_of_plane_waves,
              fixture.g_vector_indices,
              fixture.crystal_structure,
              fixture.kpoint_grid,
              fixture.basis,
              orbitals);
    ASSERT_EQ(orbitals.size(), 3u);
    for (const std::vector<Complex>& orbital : orbitals) {
        ASSERT_EQ(static_cast<int>(orbital.size()), fixture.number_of_plane_waves);
        EXPECT_GT(std::abs(inner_product(orbital, orbital)), 0.0);
    }
    EXPECT_FALSE(set.description().empty());
}

TEST(PlanewaveHydrogenicOrbitalSet, ALatticeTranslationOfTheCentreIsAPurePhase) {
    /**
     * The same orbital placed at tau and at tau + a_1 is the same function up
     * to the Bloch phase e^{-i k.a_1}: every coefficient's magnitude must
     * match, and the ratio must be one constant phase across all G. This
     * pins the structure-factor convention without depending on any
     * normalisation choice.
     */
    Fixture fixture;
    PlanewaveHydrogenicOrbitalSet set(
        {{Vector3D(0.1, 0.2, 0.3), 1.0}, {Vector3D(1.1, 0.2, 0.3), 1.0}});
    std::vector<std::vector<Complex>> orbitals;
    set.build(0,
              fixture.number_of_plane_waves,
              fixture.g_vector_indices,
              fixture.crystal_structure,
              fixture.kpoint_grid,
              fixture.basis,
              orbitals);
    const std::vector<Complex>& original = orbitals[0];
    const std::vector<Complex>& translated = orbitals[1];
    Complex ratio = translated[0] / original[0];
    EXPECT_NEAR(std::abs(ratio), 1.0, 1e-12);
    for (int g_index = 0; g_index < fixture.number_of_plane_waves; g_index++) {
        EXPECT_NEAR(std::abs(translated[g_index]), std::abs(original[g_index]), 1e-12);
        EXPECT_NEAR(std::abs(translated[g_index] / original[g_index] - ratio), 0.0, 1e-10)
            << "G index " << g_index;
    }
}

TEST(PlanewaveHydrogenicOrbitalSet, OrbitalsAtDifferentCentresAreNotIdentical) {
    Fixture fixture;
    PlanewaveHydrogenicOrbitalSet set(
        {{Vector3D(0.0, 0.0, 0.0), 1.0}, {Vector3D(0.5, 0.0, 0.0), 1.0}});
    std::vector<std::vector<Complex>> orbitals;
    set.build(0,
              fixture.number_of_plane_waves,
              fixture.g_vector_indices,
              fixture.crystal_structure,
              fixture.kpoint_grid,
              fixture.basis,
              orbitals);
    Real norm_product = std::sqrt(std::abs(inner_product(orbitals[0], orbitals[0])) *
                                  std::abs(inner_product(orbitals[1], orbitals[1])));
    Real overlap = std::abs(inner_product(orbitals[0], orbitals[1]));
    EXPECT_LT(overlap, 0.99 * norm_product);
}

TEST(PlanewaveHydrogenicOrbitalSet, RejectsANonPositiveDecayLength) {
    EXPECT_THROW(PlanewaveHydrogenicOrbitalSet({{Vector3D(0.0, 0.0, 0.0), 0.0}}),
                 std::invalid_argument);
    EXPECT_THROW(PlanewaveHydrogenicOrbitalSet({{Vector3D(0.0, 0.0, 0.0), -1.0}}),
                 std::invalid_argument);
}

TEST(PlanewaveLocalOrbitalSetFactory, EmptyCentresSelectThePseudoAtomicSet) {
    Fixture fixture;
    std::unique_ptr<PlanewaveLocalOrbitalSet> set = create_planewave_local_orbital_set(
        fixture.crystal_structure, fixture.pseudopotential_set, {}, 1.0);
    ASSERT_NE(dynamic_cast<PlanewavePseudoAtomicOrbitalSet*>(set.get()), nullptr);
    // Silicon's pseudopotential carries 3s + 3p on each of two atoms.
    EXPECT_EQ(set->number_of_orbitals(), 8);
}

TEST(PlanewaveLocalOrbitalSetFactory, ExplicitCentresSelectTheHydrogenicSet) {
    Fixture fixture;
    std::unique_ptr<PlanewaveLocalOrbitalSet> set =
        create_planewave_local_orbital_set(fixture.crystal_structure,
                                           fixture.pseudopotential_set,
                                           {Vector3D(0.0, 0.0, 0.0),
                                            Vector3D(0.5, 0.0, 0.0),
                                            Vector3D(0.0, 0.5, 0.0),
                                            Vector3D(0.0, 0.0, 0.5)},
                                           1.0);
    auto* hydrogenic = dynamic_cast<PlanewaveHydrogenicOrbitalSet*>(set.get());
    ASSERT_NE(hydrogenic, nullptr);
    EXPECT_EQ(set->number_of_orbitals(), 4);
    EXPECT_NEAR(hydrogenic->orbitals()[0].decay_length_bohr, 1.0, 0.0);
}
