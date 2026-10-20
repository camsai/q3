#pragma once

#include "layer_4_computation/state/internal/energies/init.hpp"

namespace q3 {

/**
 * Kohn-Sham energy components in Rydberg.
 *
 * Adds the standard DFT energy terms that arise from solving the
 * Kohn-Sham equations: band energy (sum of eigenvalues), double-counting
 * correction, Hartree energy, and exchange-correlation energy.
 */
struct KohnShamEnergies : public SCFEnergies {
    Real band_energy = 0.0;
    Real double_counting_correction = 0.0;
    Real hartree_energy = 0.0;
    Real exchange_correlation_energy = 0.0;
    Real exchange_correlation_potential_energy = 0.0;

    Real total() const override {
        return band_energy + double_counting_correction + hartree_energy +
               exchange_correlation_energy;
    }
};

} // namespace q3
