#pragma once

/**
 * StartingSubspaceSolverProvider — who does a diagonalizer's starting rotation, chosen once.
 *
 * The starting rotation's large operations (H on the starting vectors, their subspace
 * matrices and the rotation) are a StartingSubspaceSolver. The caller always has a host one
 * for the k-point; a diagonalizer that can do them faster somewhere else hands out its own.
 * Which of the two a run uses is fixed by the run: the representation, the band group and
 * the overlap. A provider is that decision made once, in a factory, so that the diagonalizer
 * asks none of those questions at each k-point.
 */

#include <vector>

#include "layer_0_core/types/init.hpp"
#include "layer_1_structure/reciprocal_space/kpoints_grid.hpp"
#include "layer_3_representation/wavefunction/planewave/starting_subspace_solver.hpp"

namespace q3 {

/// Hands out the solver for one k-point's starting rotation.
class StartingSubspaceSolverProvider {
  public:
    virtual ~StartingSubspaceSolverProvider() = default;

    /**
     * The solver for this k-point and potential, prepared and valid until the next call.
     *
     * `serial_solver` is the caller's host solver, already prepared for the same k-point;
     * a provider with nothing faster returns it.
     */
    virtual StartingSubspaceSolver& prepare(int kpoint_index_state,
                                            int kpoint_index,
                                            int number_of_plane_waves,
                                            const std::vector<int>& g_vector_indices,
                                            const std::vector<Real>& kinetic_energy_squared,
                                            const std::vector<Real>& real_space_potential,
                                            const KPointsGrid& kpoint_grid,
                                            StartingSubspaceSolver& serial_solver) = 0;
};

} // namespace q3
