#pragma once

/**
 * SerialStartingSubspaceSolverProvider — the starting rotation stays with the caller's host
 * solver.
 *
 * For a run whose device applicator is not known to reproduce the host Hamiltonian of the
 * rotation (see create_gpu_starting_subspace_solver_provider()).
 */

#include <vector>

#include "layer_0_core/types/init.hpp"
#include "layer_1_structure/reciprocal_space/kpoints_grid.hpp"
#include "layer_3_representation/wavefunction/planewave/starting_subspace_solver.hpp"
#include "layer_4_computation/diagonalizer/planewave/internal/hamiltonian_applicator/starting_subspace_solver_provider.hpp"

namespace q3 {

/// Returns the caller's host solver at every k-point.
class SerialStartingSubspaceSolverProvider : public StartingSubspaceSolverProvider {
  public:
    /// `serial_solver`, unchanged.
    StartingSubspaceSolver& prepare(int /*kpoint_index_state*/,
                                    int /*kpoint_index*/,
                                    int /*number_of_plane_waves*/,
                                    const std::vector<int>& /*g_vector_indices*/,
                                    const std::vector<Real>& /*kinetic_energy_squared*/,
                                    const std::vector<Real>& /*real_space_potential*/,
                                    const KPointsGrid& /*kpoint_grid*/,
                                    StartingSubspaceSolver& serial_solver) override {
        return serial_solver;
    }
};

} // namespace q3
