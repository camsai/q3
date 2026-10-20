#include "io/output/material/init.hpp"

#include <gtest/gtest.h>

#include "layer_0_core/types/constants.hpp"

using namespace q3;

namespace {

/// Two-atom FCC silicon from a Bravais index, the second atom holding x and y fixed.
InputData make_fcc_silicon() {
    InputData input;
    input.prefix = "silicon";
    input.bravais_lattice_index = 2;
    input.cell_dimension_1 = 10.2;
    AtomSpecies silicon;
    silicon.symbol = "Si";
    silicon.mass = 28.086;
    silicon.valence_charge = 4.0;
    input.species = {silicon};
    input.number_of_species = 1;
    Atom first;
    first.species_index = 0;
    first.position = {0.0, 0.0, 0.0};
    Atom second;
    second.species_index = 0;
    second.position = {0.25, 0.25, 0.25};
    second.movable_components = {false, false, true};
    input.atoms = {first, second};
    input.number_of_atoms = 2;
    return input;
}

} // namespace

/// The cell is the engine's own, here built from ibrav = 2, written as vectors in angstrom.
TEST(MaterialJson, CarriesTheCellTheEngineBuilt) {
    const nlohmann::json material = build_material_json(make_fcc_silicon());

    const Real half_alat_angstrom = 0.5 * 10.2 * constants::BOHR_RADIUS_ANGSTROM;
    EXPECT_EQ(material["lattice"]["vectors"]["units"], "angstrom");
    EXPECT_NEAR(material["lattice"]["vectors"]["a"][0].get<Real>(), -half_alat_angstrom, 1e-12);
    EXPECT_NEAR(material["lattice"]["vectors"]["a"][2].get<Real>(), half_alat_angstrom, 1e-12);
    EXPECT_NEAR(material["lattice"]["alpha"].get<Real>(), 60.0, 1e-9);
    EXPECT_EQ(material["basis"]["units"], "crystal");
    EXPECT_EQ(material["basis"]["elements"][1]["value"], "Si");
    EXPECT_DOUBLE_EQ(material["basis"]["coordinates"][1]["value"][0].get<Real>(), 0.25);
}

/// pw.x if_pos 0 0 1 is ESSE [false, false, true]: true lets a component move.
TEST(MaterialJson, ConstraintsFollowTheMat3raConvention) {
    const nlohmann::json material = build_material_json(make_fcc_silicon());

    EXPECT_EQ(material["basis"]["constraints"][0]["value"], nlohmann::json({true, true, true}));
    EXPECT_EQ(material["basis"]["constraints"][1]["value"], nlohmann::json({false, false, true}));
}
