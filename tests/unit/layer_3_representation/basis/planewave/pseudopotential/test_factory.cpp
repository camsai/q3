#include <gtest/gtest.h>

#include <memory>

#include "layer_0_core/types/init.hpp"
#include "layer_1_structure/crystal/atom.hpp"
#include "layer_1_structure/crystal/atom_species.hpp"
#include "layer_1_structure/crystal/crystal.hpp"
#include "layer_3_representation/basis/planewave/init.hpp"
#include "layer_3_representation/basis/planewave/pseudopotential/factory.hpp"
#include "layer_3_representation/pseudopotential/init.hpp"
#include "layer_3_representation/pseudopotential/set.hpp"
#include "layer_3_representation/pseudopotential/ultrasoft/init.hpp"

#include "io/pseudopotential/upf_reader.hpp"
#include "layer_4_computation/iterator/kohn_sham/planewave/execution/serial.hpp"

using namespace q3;

static execution::PlanewaveSerialExecutionPolicy serial_policy;

/**
 * Helper to build a minimal FCC silicon crystal for factory tests.
 */
static Crystal build_silicon_crystal() {
    Crystal crystal;
    crystal.set_from_lattice_type(2, 10.263048);
    AtomSpecies silicon{"Si", 28.08, 4.0, ""};
    crystal.add_species(silicon);
    crystal.add_atom({0, {0.0, 0.0, 0.0}});
    crystal.add_atom({0, {0.25, 0.25, 0.25}});
    crystal.compute_derived_quantities();
    return crystal;
}

/**
 * Factory should create a NormConservingPlanewaveBasis (not ultrasoft)
 * when given norm-conserving pseudopotentials.
 */
TEST(NormConservingPlanewaveBasisFactory, CreatesNormConservingForNCPseudopotential) {
    Crystal crystal = build_silicon_crystal();

    auto pseudopotential =
        UPFReader::read(std::string(EXAMPLES_DIR) + "/data/pseudopotentials/nc/Si.pz-vbc.UPF");

    PseudopotentialSet pseudopotential_set;
    pseudopotential_set.push_back(std::move(pseudopotential));

    PlanewaveBasis planewave_basis;
    planewave_basis.generate(serial_policy, crystal, 16.0, 64.0);

    auto basis =
        NormConservingPlanewaveBasisFactory::create(crystal, planewave_basis, pseudopotential_set);

    ASSERT_NE(basis, nullptr);
    EXPECT_FALSE(basis->is_ultrasoft());
}

/**
 * Factory should create an UltrasoftPlanewaveBasis when given
 * ultrasoft pseudopotentials.
 */
TEST(NormConservingPlanewaveBasisFactory, CreatesUltrasoftForUSPseudopotential) {
    Crystal crystal = build_silicon_crystal();

    auto pseudopotential =
        UPFReader::read(std::string(EXAMPLES_DIR) + "/data/pseudopotentials/us/Si_PBE_USPP.UPF");

    PseudopotentialSet pseudopotential_set;
    pseudopotential_set.push_back(std::move(pseudopotential));

    PlanewaveBasis planewave_basis;
    planewave_basis.generate(serial_policy, crystal, 16.0, 64.0);

    auto basis =
        NormConservingPlanewaveBasisFactory::create(crystal, planewave_basis, pseudopotential_set);

    ASSERT_NE(basis, nullptr);
    EXPECT_TRUE(basis->is_ultrasoft());
}
