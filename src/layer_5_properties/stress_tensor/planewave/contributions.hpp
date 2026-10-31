#pragma once

/**
 * StressTensorContributions — per-contribution stress breakdown.
 *
 * Plain data holder written into by each sub-calculator during compute_stress().
 * Returned alongside the 3×3 stress tensor from compute_stress().
 *
 * All values are in kbar (converted from Ry/bohr³ inside
 * assemble_total_stress_and_log() before being returned).
 *
 * This type has no JSON serialization dependency. The SCF flavor copies these
 * primitive tensors into ESSE stress_tensor records for results.json.
 */

#include "layer_0_core/types/init.hpp"

namespace q3 {

/**
 * StressTensorTotal — assembled stress tensor (kbar).
 *
 * Plain data holder returned by compute_stress() alongside
 * StressTensorContributions.  Values are in kbar (converted from
 * Ry/bohr³ inside assemble_total_stress_and_log()).
 */
struct StressTensorTotal {
    Real values[3][3]{}; // 3×3 Voigt stress tensor (kbar)
    Real pressure = 0.0; // hydrostatic pressure = Tr(σ)/3 (kbar)
};

/// Per-contribution stress tensor breakdown (kbar after conversion).
struct StressTensorContributions {
    Real kinetic[3][3]{};              // kinetic
    Real hartree[3][3]{};              // Hartree
    Real exchange_correlation[3][3]{}; // exchange-correlation
    Real ewald[3][3]{};                // Ewald ionic
    Real local[3][3]{};                // local pseudopotential
    Real nonlocal[3][3]{};             // nonlocal (KB); includes augmentation after merge
    Real core_correction[3][3]{};      // NLCC core correction
    Real augmentation[3][3]{};         // USPP augmentation (before merge into nonlocal)
    Real hubbard[3][3]{};              // DFT+U (Hubbard) projector strain derivative
    Real dispersion[3][3]{};           // DFT-D3(BJ) semi-empirical dispersion
};

} // namespace q3
