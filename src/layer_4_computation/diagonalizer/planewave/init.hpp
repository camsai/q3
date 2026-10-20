#pragma once

/**
 * PlanewaveDiagonalizer — planewave-specific eigenvalue solver interface.
 *
 * Extends the generic Diagonalizer with the planewave-specific solve()
 * contract, cached KB projector management, and smooth potential cache
 * invalidation. Concrete implementations (PlanewaveDirectDiagonalizer,
 * PlanewaveDavidsonDiagonalizer) inherit from this class.
 */

#include <memory>
#include <string>
#include <vector>

#include "layer_0_core/types/init.hpp"
#include "layer_0_core/types/matrix.hpp"
#include "layer_3_representation/basis/planewave/init.hpp"
#include "layer_3_representation/pseudopotential/init.hpp"
#include "layer_3_representation/quantum_system/kohn_sham/planewave/norm_conserving/init.hpp"
#include "layer_4_computation/diagonalizer/init.hpp"

namespace q3 {

// Planewave-specific abstract interface for eigenvalue solvers.
class PlanewaveDiagonalizer : public Diagonalizer {
  public:
    ~PlanewaveDiagonalizer() override = default;

    /**
     * Solve the KS eigenvalue problem at one k-point.
     * For NC-PP: H·ψ = ε·ψ  (standard eigenvalue problem)
     * For US-PP: H·ψ = ε·S·ψ (generalized eigenvalue problem)
     */
    virtual void
    solve(int kpoint_index_state,
          int kpoint_index,
          int number_of_plane_waves,
          int number_of_bands,
          const std::vector<int>& g_vector_indices,
          const std::vector<Real>& kinetic_energy_squared,
          const std::vector<Real>& real_space_potential,
          const Crystal& crystal_structure,
          const KPointsGrid& kpoint_grid,
          const PlanewaveBasis& planewave_basis,
          const NormConservingPlanewaveKohnShamQuantumSystem& pseudopotential_quantum_system,
          PlanewaveKohnShamQuantumSystem& quantum_system) = 0;

    /**
     * Set pre-computed KB projectors for a specific k-point.
     * If set, the solver may use these instead of recomputing them.
     * No-op by default; overridden by solvers that support caching.
     */
    virtual void set_cached_projectors(int /*kpoint_index*/,
                                       const q3::Matrix<Complex>* /*projectors*/) {}

    /**
     * Size the iterative working basis, as a multiple of the band count.
     *
     * No-op for solvers that do not build one -- the direct diagonalizer forms
     * the whole Hamiltonian and has no working basis to size.
     */
    virtual void set_basis_expansion_factor(int /*factor*/) {}

    /**
     * Cap the iterations one solve() may take before it returns what it has.
     *
     * No-op for solvers that are not iterative.
     */
    virtual void set_maximum_iterations(int /*maximum_iterations*/) {}

    /**
     * How many bands the last solve() returned without meeting its tolerance:
     * zero when it converged, and always zero for a solver that is not
     * iterative.
     */
    [[nodiscard]] virtual int number_of_unconverged_bands() const {
        return 0;
    }

    /**
     * Free the working arrays once nothing will call solve() again in this
     * run: the eigenvectors live in the quantum system and survive, the
     * working basis and its Hamiltonian image do not need to. A later solve()
     * allocates afresh. No-op for solvers without a persistent workspace.
     */
    virtual void release_workspace() {}

    /**
     * Set pre-computed DFT+U (Hubbard) atomic-orbital projectors for a specific k-point.
     * No-op by default; overridden by solvers that support DFT+U (v1: Davidson only).
     */
    virtual void set_cached_hubbard_projectors(int /*kpoint_index*/,
                                               const q3::Matrix<Complex>* /*projectors*/) {}

    /**
     * Set the dense block-diagonal Hubbard D-matrix (see PlanewaveHamiltonianApplicator).
     * No-op by default; overridden by solvers that support DFT+U (v1: Davidson only).
     */
    virtual void set_hubbard_d_matrix(const std::vector<Complex>* /*dense_d_matrix_column_major*/) {
    }

    /**
     * Invalidate any cached smooth grid potential.
     * Called by the SCF iterator when the effective potential changes.
     * No-op by default; overridden by solvers that cache the smooth potential.
     */
    virtual void invalidate_smooth_potential_cache() {}
};

} // namespace q3
