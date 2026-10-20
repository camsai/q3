#pragma once

/**
 * PlanewaveHamiltonianApplicator — abstract interface for H|ψ⟩ application.
 *
 * Represents the full Kohn-Sham Hamiltonian application in the planewave
 * representation: H|ψ⟩ = (T + V_local + V_NL)|ψ⟩.
 *
 * Lifecycle:
 *   1. Construct with run-constant context (crystal, basis, pseudopotentials, FFT grid)
 *   2. Call update_potential() at each SCF iteration (computes smooth-grid potential)
 *   3. Call prepare_kpoint() once per k-point (sets up FFT mappings, KB projectors)
 *   4. Call apply_batch() per Davidson iteration (lean H|ψ⟩ application)
 *
 * GPU implementations override to use GPU BLAS, GPU FFT, and device memory.
 */

#include <complex>
#include <stdexcept>
#include <vector>

#include "layer_0_core/types/init.hpp"
#include "layer_0_core/types/matrix.hpp"

namespace q3 {

class PlanewaveHamiltonianApplicator {
  public:
    virtual ~PlanewaveHamiltonianApplicator() = default;

    /**
     * Update the effective potential for a new SCF iteration.
     * Must be called before prepare_kpoint() / apply_batch().
     *
     * Internally computes the smooth-grid potential (FFT → filter → inverse FFT)
     * and caches the result. For GPU implementations, copies the smooth potential
     * to a GPUDeviceBuffer.
     */
    virtual void update_potential(const std::vector<Real>& real_space_potential) = 0;

    /**
     * Configure for a specific k-point. Must be called before apply_batch().
     * Sets up k+G vectors, kinetic factors, KB projectors, FFT mappings.
     * For GPU implementations, copies per-kpoint data to GPUDeviceBuffer.
     */
    virtual void prepare_kpoint(int kpoint_index,
                                int number_of_plane_waves,
                                const std::vector<int>& g_vector_indices,
                                const std::vector<Real>& kinetic_energy_squared) = 0;

    /**
     * Apply H|ψ⟩ in double precision (FP64) to a batch of vectors.
     *
     * Only takes data that changes per Davidson iteration: the wavefunctions
     * and their output. All other context (potential, k-point data, projectors)
     * was configured via update_potential() and prepare_kpoint().
     */
    virtual void apply_batch(const Complex* input_wavefunctions,
                             Complex* output_hamiltonian_wavefunctions,
                             int number_of_vectors) = 0;

    /**
     * Which spin channel the following applications belong to.
     *
     * The nonlocal D matrix is spin-dependent for ultrasoft and PAW —
     * D_ij = D_ion + ∫v_eff·Q_ij, and v_eff differs between channels — so
     * applying H needs to know which channel it is in. The caller sets this
     * beside `update_potential`, which already receives that channel's
     * potential; the cached dense D is dropped whenever it changes, since it
     * was built for the other channel.
     *
     * Unpolarized runs never call it and stay on channel 0.
     */
    void set_spin_channel(int spin_channel) {
        if (spin_channel == spin_channel_)
            return;
        spin_channel_ = spin_channel;
        invalidate_smooth_potential_cache();
    }

    [[nodiscard]] int spin_channel() const {
        return spin_channel_;
    }

    /**
     * Invalidate any cached smooth potential.
     * Called when the effective potential changes (e.g., at start of each SCF iteration).
     */
    virtual void invalidate_smooth_potential_cache() = 0;

    /**
     * Whether the smooth potential cache is currently valid.
     * Default: false (conservative — forces recomputation).
     */
    virtual bool is_smooth_potential_cache_valid() const {
        return false;
    }

    /**
     * Whether the current k-point uses gamma-only half-sphere G-vectors.
     * Default: false.
     */
    virtual bool is_gamma_only_current() const {
        return false;
    }

    /**
     * Whether this rank's plane-wave list for the current k-point begins with
     * G = 0, so that the gamma trick's origin correction belongs to it. True
     * whenever the coefficients are replicated; under a reciprocal-space
     * distribution exactly one rank answers true. Meaningful only when
     * is_gamma_only_current().
     */
    virtual bool gamma_origin_is_local_current() const {
        return true;
    }

    /**
     * Get pre-computed KB projectors for a specific k-point, or nullptr if none.
     * Default: nullptr (no cached projectors).
     */
    virtual const q3::Matrix<Complex>* get_cached_projectors(int kpoint_index) const {
        (void)kpoint_index;
        return nullptr;
    }

    /**
     * Set pre-computed KB projectors for a specific k-point.
     * Default: no-op (implementations that cache projectors override this).
     */
    virtual void set_cached_projectors(int kpoint_index, const q3::Matrix<Complex>* projectors) {
        (void)kpoint_index;
        (void)projectors;
    }

    /**
     * Set pre-computed DFT+U (Hubbard) atomic-orbital projectors for a specific k-point.
     * Geometry-only (like KB projectors): set once per k-point at SCF setup.
     * Default: no-op -- only the CPU serial applicator implements DFT+U (v1: NC + CPU only).
     */
    virtual void set_cached_hubbard_projectors(int kpoint_index,
                                               const q3::Matrix<Complex>* projectors) {
        (void)kpoint_index;
        (void)projectors;
    }

    /**
     * Set the dense block-diagonal Hubbard D-matrix (Dudarev V_U coefficients), spanning all
     * Hubbard-active orbitals across all manifolds. Column-major (Fortran/BLAS convention),
     * dimension total_hubbard_orbitals x total_hubbard_orbitals. Unlike the KB D-matrix (fixed,
     * lazily built from geometry), this depends on the occupation matrix and is refreshed once
     * per SCF iteration (not per k-point). Default: no-op.
     */
    virtual void set_hubbard_d_matrix(const std::vector<Complex>* dense_d_matrix_column_major) {
        (void)dense_d_matrix_column_major;
    }

  protected:
    /// Spin channel the current potential and D belong to; 0 when unpolarized.
    int spin_channel_{0};
};

} // namespace q3
