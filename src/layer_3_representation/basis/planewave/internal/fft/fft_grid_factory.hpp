#pragma once

/**
 * FFTGridFactory — abstract factory for FFT grid instances.
 *
 * Layer-3 consumers (PlanewaveFFTPair, PlanewaveBasis) need to create FFT
 * grids without knowing which concrete implementation (serial FFTW, FFTW-MPI
 * slab-decomposed, GPU) the run is using. That choice belongs to the
 * execution policy at layer 4 — so layer 3 defines this interface and the
 * PlanewaveExecutionPolicy implements it (dependency inversion: lower layers
 * must not include higher-layer headers).
 */

#include "layer_3_representation/basis/planewave/internal/fft/fft_grid.hpp"

#include <memory>

namespace q3 {

class FFTGridFactory {
  public:
    virtual ~FFTGridFactory() = default;

    /**
     * Create a new FFT grid instance appropriate for the active execution
     * configuration (serial, MPI slab-decomposed, ...).
     */
    virtual std::unique_ptr<FFTGrid> create_fft_grid() const = 0;
};

} // namespace q3
