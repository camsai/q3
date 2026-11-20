#pragma once

/**
 * StructureFactorCalculator — sums over atoms and G-vectors of the phases e^{∓iG·τ}.
 *
 * The local, core-correction and Ewald terms of the forces and the stress all reduce to
 * two sums over a list of density G-vectors and the atoms of the crystal. Everything else
 * they do is per species and per G-vector, which is cheap; the phases, one per atom and
 * G-vector, are what costs. An implementation forms the phases where it is fastest, on
 * the host or on a device, and the calculators of those terms are the same code either way.
 *
 * Positions τ_a are Cartesian in units of alat and G-vectors are GVector::g_cart in units
 * of 2π/alat, so the phase of atom a at G-vector j is 2π G_j·τ_a. The G-vectors are named
 * by their indices into PlanewaveBasis::g_vectors().
 *
 *   structure_factors:      S[set][j] = Σ_a atom_weights[set][a] · e^{-iG_j·τ_a}
 *   gradient_phase_sums:    F[a][α]   = Σ_j G_{j,α} · Im[ phase_weights[set_of_atom[a]][j]
 *                                                         · e^{+iG_j·τ_a} ]
 *
 * In the second, G_{j,α} is in units of 2π/alat, and an atom whose set_of_atom is negative
 * has a zero row. Neither sum is reduced across ranks: a caller that splits the G-vectors
 * completes the sums itself.
 */

#include <vector>

#include "layer_0_core/types/init.hpp"

namespace q3 {

/// Phase sums over atoms and G-vectors; see the file comment.
class StructureFactorCalculator {
  public:
    virtual ~StructureFactorCalculator() = default;

    /**
     * One weighted structure factor per set of atom weights, at the G-vectors
     * `g_vector_indices`: result[set][j] = Σ_a atom_weights[set][a] e^{-iG_j·τ_a}.
     * Each set of weights has one entry per atom of the crystal.
     */
    [[nodiscard]] virtual std::vector<std::vector<Complex>>
    structure_factors(const std::vector<int>& g_vector_indices,
                      const std::vector<std::vector<Real>>& atom_weights) const = 0;

    /**
     * Per atom, the gradient sum of the file comment, at the G-vectors `g_vector_indices`.
     * phase_weights[set] has one entry per G-vector of the list; set_of_atom has one entry
     * per atom of the crystal, naming its set of phase weights, or negative for none.
     */
    [[nodiscard]] virtual std::vector<Vector3D>
    gradient_phase_sums(const std::vector<int>& g_vector_indices,
                        const std::vector<std::vector<Complex>>& phase_weights,
                        const std::vector<int>& set_of_atom) const = 0;
};

} // namespace q3
