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

    /**
     * Create the grid the density and the potentials are transformed on. The same kind of
     * grid as create_fft_grid() unless the configuration knows better: that grid's transforms
     * are made one at a time, never from inside a loop over bands, so a configuration whose
     * band loops do not run on the host can give each of them every thread.
     */
    virtual std::unique_ptr<FFTGrid> create_density_fft_grid() const {
        return create_fft_grid();
    }
};

} // namespace q3
