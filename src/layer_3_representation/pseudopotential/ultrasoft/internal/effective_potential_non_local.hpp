#pragma once

/**
 * EffectiveNonlocalPotential — abstract interface for the effective
 * D-matrix computation (QE's "newd").
 *
 * Combines the bare D_ij^ion matrix with the G-space integral of
 * V_eff(G) × Q_ij(G) × e^{-iG·τ} to produce the effective D-matrix
 * used in the non-local Hamiltonian.
 */

#include <complex>
#include <vector>

#include "layer_0_core/types/init.hpp"

namespace q3 {

// Forward declarations
class Crystal;
class PlanewaveBasis;

class EffectiveNonlocalPotential {
  public:
    virtual ~EffectiveNonlocalPotential() = default;

    /**
     * Compute the effective D-matrix for all atoms.
     *
     * @param crystal_structure    Crystal structure with atom positions.
     * @param planewave_basis     Plane-wave basis set.
     * @param total_potential_g    Total effective potential V_eff(G).
     * @return effective_d_matrix[atom_index][projector_i * nproj + projector_j]
     */
    [[nodiscard]] virtual std::vector<std::vector<Real>>
    compute(const Crystal& crystal_structure,
            const PlanewaveBasis& planewave_basis,
            const std::vector<Complex>& total_potential_g) const = 0;
};

} // namespace q3
