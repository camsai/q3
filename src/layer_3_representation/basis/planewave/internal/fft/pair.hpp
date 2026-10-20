#pragma once

/**
 * PlanewaveFFTPair — charge-density and wavefunction FFT grids sized together.
 */

#include "layer_1_structure/crystal/crystal.hpp"
#include "layer_3_representation/basis/planewave/internal/fft/fft_grid.hpp"
#include "layer_3_representation/basis/planewave/internal/fft/fft_grid_factory.hpp"
#include "layer_3_representation/basis/planewave/internal/fft/fft_grid_serial.hpp"
#include "layer_3_representation/basis/planewave/internal/fft/grid_sizer.hpp"
#include <memory>

namespace q3 {

class PlanewaveFFTPair {
  public:
    /**
     * @param use_exact_wavefunction_grid  false (default) sizes the wavefunction grid for
     *        2 x ecutwfc; true sizes it exactly, for 4 x ecutwfc, as pw.x does. See
     *        InputData::use_exact_wavefunction_grid.
     */
    void initialize(const FFTGridFactory& fft_grid_factory,
                    const Crystal& crystal_structure,
                    Real energy_cutoff_charge_density,
                    Real energy_cutoff_wavefunction,
                    bool use_measured_fft_planning = true,
                    bool use_exact_wavefunction_grid = false) {
        Real two_pi_over_lattice_parameter_squared =
            crystal_structure.two_pi_over_lattice_parameter_squared();
        const Matrix3x3& reciprocal_lattice = crystal_structure.reciprocal_lattice_vectors();
        const Matrix3x3& direct_lattice = crystal_structure.lattice_vectors();

        int fft_factors[3];
        FFTGridSizer::compute_fft_factors(crystal_structure, fft_factors);

        int grid_dimension_rho_1, grid_dimension_rho_2, grid_dimension_rho_3;
        FFTGridSizer::calculate_grid_dimensions(energy_cutoff_charge_density,
                                                two_pi_over_lattice_parameter_squared,
                                                direct_lattice,
                                                reciprocal_lattice,
                                                grid_dimension_rho_1,
                                                grid_dimension_rho_2,
                                                grid_dimension_rho_3,
                                                fft_factors);
        /**
         * Rho grid uses FFTW_MEASURE by default (use_measured_fft_planning=true);
         * pass false for bit-reproducible FFTW_ESTIMATE planning (see
         * InputData::use_deterministic_fft_planning).
         */
        fft_rho_ = fft_grid_factory.create_fft_grid();
        fft_rho_->init(grid_dimension_rho_1,
                       grid_dimension_rho_2,
                       grid_dimension_rho_3,
                       use_measured_fft_planning);

        /**
         * The smooth (wavefunction) FFT grid carries two products, V(r)ψ(r) and |ψ(r)|². A
         * wavefunction holds plane waves out to G_max = sqrt(ecutwfc), so |ψ|² reaches 2 G_max,
         * and a grid that holds the sphere of radius 2 G_max -- a 4 x ecutwfc cutoff -- takes
         * both products without aliasing: |ψ|² fits whole, and any component of Vψ that wraps
         * around lands outside the wavefunction sphere, where it is discarded. That is the
         * exact grid, and the one pw.x builds; enlarging it changes nothing.
         *
         * The default grid holds only √2 G_max (2 x ecutwfc). Parts of |ψ|² and of Vψ wrap onto
         * components that are kept, so it is faster but approximate: harmless where the
         * wavefunctions have little weight near the cutoff, and up to mRy off pw.x on a hard
         * deck at a low cutoff -- 9.5 mRy on a Ni/CO slab at 20 Ry. The exact grid costs 1.4x to
         * 2.3x the solver time on the silicon and copper examples, depending on the thread count.
         *
         * Either is capped at the dense grid, which already holds everything the density can
         * carry.
         */
        Real wavefunction_grid_cutoff_factor = use_exact_wavefunction_grid ? 4.0 : 2.0;
        Real smooth_cutoff = std::min(wavefunction_grid_cutoff_factor * energy_cutoff_wavefunction,
                                      energy_cutoff_charge_density);
        int grid_dimension_wave_1, grid_dimension_wave_2, grid_dimension_wave_3;
        FFTGridSizer::calculate_grid_dimensions(smooth_cutoff,
                                                two_pi_over_lattice_parameter_squared,
                                                direct_lattice,
                                                reciprocal_lattice,
                                                grid_dimension_wave_1,
                                                grid_dimension_wave_2,
                                                grid_dimension_wave_3,
                                                fft_factors);
        // Wave grid uses FFTW_MEASURE by default; see the rho grid comment above.
        fft_wave_ = fft_grid_factory.create_fft_grid();
        fft_wave_->init(grid_dimension_wave_1,
                        grid_dimension_wave_2,
                        grid_dimension_wave_3,
                        use_measured_fft_planning);

        /**
         * Serial wave grid for band-distributed parallelism (task groups).
         * Every rank gets a full-sized serial grid so each can do independent
         * FFTs for its assigned bands without MPI collective calls.
         */
        fft_wave_serial_ = std::make_unique<SerialFFTGrid>();
        fft_wave_serial_->init(grid_dimension_wave_1,
                               grid_dimension_wave_2,
                               grid_dimension_wave_3,
                               use_measured_fft_planning);
    }

    [[nodiscard]] FFTGrid& fft_rho() {
        return *fft_rho_;
    }
    [[nodiscard]] const FFTGrid& fft_rho() const {
        return *fft_rho_;
    }
    [[nodiscard]] FFTGrid& fft_wave() {
        return *fft_wave_;
    }
    [[nodiscard]] const FFTGrid& fft_wave() const {
        return *fft_wave_;
    }
    /**
     * Serial (non-MPI) wave FFT grid for band-distributed parallelism.
     * Identical dimensions to fft_wave() but runs locally on each rank.
     */
    [[nodiscard]] const SerialFFTGrid& fft_wave_serial() const {
        return *fft_wave_serial_;
    }

  private:
    std::unique_ptr<FFTGrid> fft_rho_;
    std::unique_ptr<FFTGrid> fft_wave_;
    std::unique_ptr<SerialFFTGrid> fft_wave_serial_;
};

} // namespace q3
