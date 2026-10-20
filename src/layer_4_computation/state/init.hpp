#pragma once

#include "layer_4_computation/state/internal/energies/init.hpp"

namespace q3 {

/**
 * Abstract base for a converged self-consistent field state.
 *
 * Produced by an SCFIterator at convergence. Carries the iteration count,
 * the convergence flag, and the energy breakdown (via the energies()
 * accessor). Concrete subtypes add the quantum system snapshot, charge
 * density, and effective potential at the appropriate representation level.
 *
 * Mirrors the SCFIterator / Diagonalizer / FermiEnergyCalculator pattern:
 * each top-level L4 concept has an abstract base in its folder's init.hpp.
 */
struct SCFState {
    virtual ~SCFState() = default;
    SCFState() = default;
    SCFState(SCFState&&) = default;
    SCFState& operator=(SCFState&&) = default;
    SCFState(const SCFState&) = default;
    SCFState& operator=(const SCFState&) = default;

    int scf_iterations{};
    bool converged{false};

    /// Return the converged energy breakdown.
    virtual const SCFEnergies& energies() const = 0;
};

} // namespace q3
