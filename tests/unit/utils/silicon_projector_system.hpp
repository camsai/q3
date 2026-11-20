#pragma once

#include <string>
#include <vector>

#include "io/pseudopotential/upf_reader.hpp"
#include "layer_1_structure/crystal/crystal.hpp"
#include "layer_1_structure/reciprocal_space/kpoints_grid.hpp"
#include "layer_3_representation/basis/planewave/init.hpp"
#include "layer_3_representation/pseudopotential/set.hpp"
#include "layer_3_representation/quantum_system/kohn_sham/planewave/norm_conserving/init.hpp"
#include "layer_4_computation/iterator/kohn_sham/planewave/execution/serial.hpp"

namespace q3::test_support {

/**
 * Two-atom silicon with a norm-conserving pseudopotential, a small basis and
 * two k-points with their plane-wave lists: everything a Kleinman-Bylander
 * projector set is built from. Each atom carries an s and a p beta function,
 * so there are 2 × (1 + 3) = 8 projectors.
 */
struct SiliconProjectorSystem {
    execution::PlanewaveSerialExecutionPolicy serial_policy;
    PseudopotentialSet pseudopotential_set;
    Crystal crystal_structure;
    PlanewaveBasis planewave_basis;
    KPointsGrid kpoint_grid;
    NormConservingPlanewaveKohnShamQuantumSystem pseudopotential_quantum_system;
    std::vector<std::vector<int>> g_vector_indices_by_kpoint;

    SiliconProjectorSystem() {
        pseudopotential_set.push_back(
            UPFReader::read(std::string(EXAMPLES_DIR) + "/data/pseudopotentials/nc/Si.pz-vbc.UPF"));
        crystal_structure.set_from_lattice_type(2, 10.263048);
        AtomSpecies si_species{"Si", 28.08, pseudopotential_set[0].valence_charge, "Si.pz-vbc.UPF"};
        crystal_structure.add_species(si_species);
        crystal_structure.add_atom({0, {0.0, 0.0, 0.0}});
        crystal_structure.add_atom({0, {0.25, 0.25, 0.25}});
        crystal_structure.compute_derived_quantities();

        planewave_basis.generate(serial_policy, crystal_structure, 8.0, 32.0);
        pseudopotential_quantum_system.init(
            crystal_structure, planewave_basis, pseudopotential_set);

        for (const Vector3D& coordinates : {Vector3D(0.0, 0.0, 0.0), Vector3D(0.25, 0.25, 0.25)}) {
            KPoint kpoint;
            kpoint.coordinates = coordinates;
            kpoint.weight = 0.5;
            kpoint_grid.add_kpoint(kpoint);
            std::vector<int> g_vector_indices;
            std::vector<Real> kinetic_energy_squared;
            planewave_basis.compute_gk(
                coordinates, crystal_structure, g_vector_indices, kinetic_energy_squared);
            g_vector_indices_by_kpoint.push_back(g_vector_indices);
        }
    }

    [[nodiscard]] const NormConservingPlanewaveBasis& pseudopotential_basis() const {
        return pseudopotential_quantum_system.pseudopotential_basis();
    }

    [[nodiscard]] int number_of_plane_waves(int kpoint_index) const {
        return static_cast<int>(g_vector_indices_by_kpoint[kpoint_index].size());
    }

    /// Move the second atom, as a relaxation step would.
    void displace_second_atom() {
        crystal_structure.set_atomic_positions(
            {Vector3D(0.0, 0.0, 0.0), Vector3D(0.26, 0.25, 0.25)});
    }
};

} // namespace q3::test_support
