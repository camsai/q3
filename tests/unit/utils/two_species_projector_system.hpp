#pragma once

#include <string>
#include <vector>

#include "io/pseudopotential/upf_reader.hpp"
#include "layer_1_structure/crystal/crystal.hpp"
#include "layer_3_representation/basis/planewave/init.hpp"
#include "layer_3_representation/pseudopotential/set.hpp"
#include "layer_3_representation/quantum_system/kohn_sham/planewave/norm_conserving/init.hpp"
#include "layer_4_computation/iterator/kohn_sham/planewave/execution/serial.hpp"

namespace q3::test_support {

/**
 * Four atoms of two species, in the order A, B, A, B, so that neither species' atoms are
 * consecutive: species A has an s and a p beta function (4 projectors per atom), species B
 * two each of s, p and d (18 per atom). Both are silicon datasets; only that their projectors
 * differ matters. One k-point off every symmetry axis, with its plane-wave list.
 */
struct TwoSpeciesProjectorSystem {
    execution::PlanewaveSerialExecutionPolicy serial_policy;
    PseudopotentialSet pseudopotential_set;
    Crystal crystal_structure;
    PlanewaveBasis planewave_basis;
    NormConservingPlanewaveKohnShamQuantumSystem pseudopotential_quantum_system;
    Vector3D kpoint{0.25, 0.125, -0.375};
    std::vector<int> g_vector_indices;

    TwoSpeciesProjectorSystem() {
        const std::string directory = std::string(EXAMPLES_DIR) + "/data/pseudopotentials/nc/";
        pseudopotential_set.push_back(UPFReader::read(directory + "Si.pz-vbc.UPF"));
        pseudopotential_set.push_back(UPFReader::read(directory + "Si.upf"));
        crystal_structure.set_from_lattice_type(1, 10.2);
        crystal_structure.add_species(
            {"SiA", 28.08, pseudopotential_set[0].valence_charge, "Si.pz-vbc.UPF"});
        crystal_structure.add_species(
            {"SiB", 28.08, pseudopotential_set[1].valence_charge, "Si.upf"});
        crystal_structure.add_atom({0, {0.0, 0.0, 0.0}});
        crystal_structure.add_atom({1, {0.25, 0.25, 0.25}});
        crystal_structure.add_atom({0, {0.5, 0.5, 0.1}});
        crystal_structure.add_atom({1, {0.7, 0.2, 0.6}});
        crystal_structure.compute_derived_quantities();
        planewave_basis.generate(serial_policy, crystal_structure, 6.0, 24.0);
        pseudopotential_quantum_system.init(
            crystal_structure, planewave_basis, pseudopotential_set);
        std::vector<Real> kinetic_energy_squared;
        planewave_basis.compute_gk(
            kpoint, crystal_structure, g_vector_indices, kinetic_energy_squared);
    }

    [[nodiscard]] const NormConservingPlanewaveBasis& pseudopotential_basis() const {
        return pseudopotential_quantum_system.pseudopotential_basis();
    }
    [[nodiscard]] int number_of_plane_waves() const {
        return static_cast<int>(g_vector_indices.size());
    }
};

} // namespace q3::test_support
