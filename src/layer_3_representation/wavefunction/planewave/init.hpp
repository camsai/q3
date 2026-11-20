#pragma once

/**
 * Plane-wave Kohn-Sham wavefunction of one band at one k-point state.
 *
 * ψ(r) = Σ_G c_G · e^{i(k+G)·r}
 *
 * The coefficients c_G are not held here. All the bands of a k-point state are
 * stored together, as one contiguous block (plane waves × bands, column-major)
 * that PlanewaveKohnShamQuantumSystem owns, because everything that works on
 * them — the eigensolver, the density, the projections — works on the whole
 * block, on the host or on a device. This type is what a QuantumState carries
 * in their place; the coefficients are read through the quantum system's
 * wavefunction_band_data() and band_block().
 */

#include <stdexcept>

#include "layer_0_core/types/init.hpp"
#include "layer_2_quantum_system/core/wavefunction/reciprocal_space.hpp"

namespace q3 {

/// One band at one k-point state in the plane-wave basis; its coefficients are in the band block.
class PlanewaveWavefunction : public ReciprocalSpaceWavefunction {
  public:
    PlanewaveWavefunction() = default;

    [[nodiscard]] int number_of_components() const override {
        return 1;
    }

    Complex evaluate(const ReciprocalSpaceVector& g_vector) const override {
        (void)g_vector;
        throw std::runtime_error("PlanewaveWavefunction::evaluate() is not yet implemented. "
                                 "Use index-based access instead.");
    }
};

/// Pseudopotential plane-wave Kohn-Sham wavefunction (same as base PW for now).
using NormConservingPlanewaveWavefunction = PlanewaveWavefunction;

/// Ultrasoft pseudopotential plane-wave Kohn-Sham wavefunction (same as base PW for now).
using UltrasoftPlanewaveWavefunction = NormConservingPlanewaveWavefunction;

} // namespace q3
