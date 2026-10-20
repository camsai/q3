#pragma once

/**
 * KohnShamQuantumSystemSnapshot — data transfer object for Kohn-Sham SCF state.
 *
 * Contains the minimal set of KS-level arrays needed to resume an SCF
 * calculation: iteration metadata, spin-resolved real-space charge density,
 * and band eigenvalues.
 */

#include <vector>

#include "layer_0_core/types/matrix.hpp"
#include "layer_0_core/types/type_aliases.hpp"

namespace q3 {

struct KohnShamQuantumSystemSnapshot {
    int iteration{};
    int number_of_spin_components{};
    int number_of_bands{};
    int number_of_kpoint_states{};
    Real total_energy{};
    bool converged{false};

    std::vector<std::vector<Real>> charge_density_real;
    Matrix<Real> eigenvalues;

    /**
     * ─── Version 2 fields ────────────────────────────────────────────────────
     * Effective potential in real space V_eff(r) = V_H(r) + V_xc(r) + V_loc(r).
     * Populated only in v2 checkpoints; left empty for v1 (NSCF falls back to
     * reconstruction from charge density + input parameters in that case).
     */
    std::vector<std::vector<Real>> effective_potential_real;

    /**
     * ─── Version 3 fields ────────────────────────────────────────────────────
     * The PAW on-site occupations ρ_ij per spin channel: one row per atom, each
     * the channel-by-channel matrix flattened at the widest species' row
     * length, exactly as the iterator accumulates them. D's one-centre term is
     * built from these, and a run that diagonalizes before it constructs any
     * density of its own -- a non-self-consistent pass -- has nothing else to
     * build it from. Empty for datasets without that term, and in v1 and v2
     * checkpoints.
     */
    std::vector<Matrix<Real>> one_center_occupations;

    virtual ~KohnShamQuantumSystemSnapshot() = default;
};

} // namespace q3
