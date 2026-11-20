#pragma once

/**
 * SerialBandProjectionCalculator — band projections with the host BLAS.
 *
 * Reads the host copy of the columns and of the band block. Bands that are one
 * run of the block, which the occupied bands of a k-point almost always are,
 * are read in place: the product is handed a pointer into the block, and no
 * copy of the bands is made. Bands with a gap between them are gathered into
 * one matrix first.
 */

#include <algorithm>
#include <complex>
#include <cstddef>
#include <vector>

#include "layer_0_core/math/linear_algebra/blas.hpp"
#include "layer_0_core/types/init.hpp"
#include "layer_3_representation/pseudopotential/planewave/band_projection_calculator.hpp"

namespace q3 {

/// Contracts plane-wave columns with bands on the host; see the file comment.
class SerialBandProjectionCalculator : public BandProjectionCalculator {
  public:
    /// See BandProjectionCalculator::project(); taken with the host BLAS.
    void project(const execution::gpu::device::DeviceMirroredArray<Complex>& columns,
                 int number_of_columns,
                 int number_of_plane_waves,
                 const PlanewaveKohnShamQuantumSystem& quantum_system,
                 int kpoint_state_index,
                 const std::vector<int>& band_indices,
                 Complex* overlaps) const override {
        if (number_of_columns == 0 || band_indices.empty()) {
            return;
        }
        std::vector<Complex> gathered_bands;
        const Complex* bands = bands_as_one_matrix(quantum_system,
                                                   kpoint_state_index,
                                                   number_of_plane_waves,
                                                   band_indices,
                                                   gathered_bands);
        multiply(columns.host(),
                 number_of_columns,
                 number_of_plane_waves,
                 bands,
                 static_cast<int>(band_indices.size()),
                 overlaps);
    }

    /// See BandProjectionCalculator::project_scaled_bands(); taken with the host BLAS.
    void project_scaled_bands(const execution::gpu::device::DeviceMirroredArray<Complex>& columns,
                              int number_of_columns,
                              int number_of_plane_waves,
                              const PlanewaveKohnShamQuantumSystem& quantum_system,
                              int kpoint_state_index,
                              const std::vector<int>& band_indices,
                              const Real* plane_wave_weights,
                              Complex prefactor,
                              Complex* overlaps) const override {
        if (number_of_columns == 0 || band_indices.empty()) {
            return;
        }
        const int number_of_bands = static_cast<int>(band_indices.size());
        const std::size_t plane_waves = static_cast<std::size_t>(number_of_plane_waves);
        std::vector<Complex> factors(plane_waves);
        for (std::size_t plane_wave = 0; plane_wave < plane_waves; ++plane_wave) {
            factors[plane_wave] = Complex(prefactor.real() * plane_wave_weights[plane_wave],
                                          prefactor.imag() * plane_wave_weights[plane_wave]);
        }
        std::vector<Complex> scaled_bands(plane_waves * static_cast<std::size_t>(number_of_bands));
#pragma omp parallel for schedule(static)
        for (int position = 0; position < number_of_bands; ++position) {
            const Complex* band = quantum_system.wavefunction_band_data(
                kpoint_state_index, band_indices[static_cast<std::size_t>(position)]);
            Complex* scaled =
                scaled_bands.data() + static_cast<std::size_t>(position) * plane_waves;
            for (std::size_t plane_wave = 0; plane_wave < plane_waves; ++plane_wave) {
                scaled[plane_wave] = band[plane_wave] * factors[plane_wave];
            }
        }
        multiply(columns.host(),
                 number_of_columns,
                 number_of_plane_waves,
                 scaled_bands.data(),
                 number_of_bands,
                 overlaps);
    }

    /// See BandProjectionCalculator::project_half_sphere(); taken with the host BLAS.
    void project_half_sphere(const execution::gpu::device::DeviceMirroredArray<Complex>& columns,
                             int number_of_columns,
                             int number_of_plane_waves,
                             const PlanewaveKohnShamQuantumSystem& quantum_system,
                             int kpoint_state_index,
                             const std::vector<int>& band_indices,
                             Real* overlaps) const override {
        if (number_of_columns == 0 || band_indices.empty()) {
            return;
        }
        std::vector<Complex> gathered_bands;
        const Complex* bands = bands_as_one_matrix(quantum_system,
                                                   kpoint_state_index,
                                                   number_of_plane_waves,
                                                   band_indices,
                                                   gathered_bands);
        int number_of_bands = static_cast<int>(band_indices.size());
        int number_of_real_rows = 2 * number_of_plane_waves;
        double two = 2.0;
        double zero = 0.0;
        blas::dgemm_("T",
                     "N",
                     &number_of_columns,
                     &number_of_bands,
                     &number_of_real_rows,
                     &two,
                     reinterpret_cast<const double*>(columns.host()),
                     &number_of_real_rows,
                     reinterpret_cast<const double*>(bands),
                     &number_of_real_rows,
                     &zero,
                     overlaps,
                     &number_of_columns);
    }

  private:
    /**
     * The chosen bands as one column-major matrix: a pointer into the host block when they are
     * one run of it, and otherwise a copy gathered into `gathered_bands`.
     */
    [[nodiscard]] static const Complex*
    bands_as_one_matrix(const PlanewaveKohnShamQuantumSystem& quantum_system,
                        int kpoint_state_index,
                        int number_of_plane_waves,
                        const std::vector<int>& band_indices,
                        std::vector<Complex>& gathered_bands) {
        if (is_one_run(band_indices)) {
            return quantum_system.wavefunction_band_data(kpoint_state_index, band_indices.front());
        }
        const std::size_t plane_waves = static_cast<std::size_t>(number_of_plane_waves);
        gathered_bands.resize(plane_waves * band_indices.size());
        for (std::size_t position = 0; position < band_indices.size(); ++position) {
            const Complex* band =
                quantum_system.wavefunction_band_data(kpoint_state_index, band_indices[position]);
            std::copy(band, band + plane_waves, gathered_bands.data() + position * plane_waves);
        }
        return gathered_bands.data();
    }

    /// overlaps(number_of_columns, number_of_bands) = columns† · bands.
    static void multiply(const Complex* columns,
                         int number_of_columns,
                         int number_of_plane_waves,
                         const Complex* bands,
                         int number_of_bands,
                         Complex* overlaps) {
        Complex one(1.0, 0.0);
        Complex zero(0.0, 0.0);
        blas::zgemm_("C",
                     "N",
                     &number_of_columns,
                     &number_of_bands,
                     &number_of_plane_waves,
                     reinterpret_cast<const std::complex<double>*>(&one),
                     reinterpret_cast<const std::complex<double>*>(columns),
                     &number_of_plane_waves,
                     reinterpret_cast<const std::complex<double>*>(bands),
                     &number_of_plane_waves,
                     reinterpret_cast<const std::complex<double>*>(&zero),
                     reinterpret_cast<std::complex<double>*>(overlaps),
                     &number_of_columns);
    }
};

} // namespace q3
