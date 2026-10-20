#pragma once

/**
 * Put a field stored on a G-vector set onto the real-space grid.
 *
 * A saved density and a saved wavefunction are both lists of coefficients
 * against Miller indices; every volumetric format wants samples on a grid. The
 * transform between them is `f(r) = sum_G f(G) exp(i G·r)`, which is exactly
 * what an unnormalized inverse FFT computes, so this is a scatter onto the
 * grid followed by `FFTGrid::inverse_raw()` — and the grid's own
 * `g_vector_to_fft_grid_index()` decides where each G lands, so the ordering
 * convention is the engine's rather than a second copy of it here.
 *
 * G-vectors that do not fit the grid are dropped rather than wrapped. A
 * sphere cut at the density cutoff always fits the grid that cutoff sized, so
 * this only fires for a mismatched pair of inputs, where wrapping would fold a
 * high-frequency component onto a low one and produce a plausible, wrong
 * field.
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <vector>

#include "layer_0_core/types/init.hpp"
#include "layer_3_representation/basis/planewave/internal/fft/fft_grid_serial.hpp"

namespace q3 {

/**
 * `f(r) = sum_G f(G) exp(i G·r)` on the grid, complex.
 *
 * @param holds_half_sphere  true for a gamma-only set, which stores only one
 *        of each ±G pair; the missing half is filled by conjugation, which is
 *        what makes the result real for a real field.
 */
inline std::vector<Complex>
complex_field_from_reciprocal_space(const std::array<int, 3>& grid_dimensions,
                                    const std::vector<IntegerVector3D>& miller_indices,
                                    const std::vector<Complex>& coefficients,
                                    bool holds_half_sphere) {
    SerialFFTGrid grid;
    grid.init(grid_dimensions[0],
              grid_dimensions[1],
              grid_dimensions[2],
              /*use_measured_planning=*/false);

    std::vector<Complex> work(static_cast<std::size_t>(grid.total_grid_points()),
                              Complex(0.0, 0.0));
    const std::size_t count = std::min(miller_indices.size(), coefficients.size());
    for (std::size_t index = 0; index < count; index++) {
        const IntegerVector3D& miller = miller_indices[index];
        if (std::abs(miller.x) > grid_dimensions[0] / 2 ||
            std::abs(miller.y) > grid_dimensions[1] / 2 ||
            std::abs(miller.z) > grid_dimensions[2] / 2) {
            continue;
        }
        work[grid.g_vector_to_fft_grid_index(miller.x, miller.y, miller.z)] = coefficients[index];
        const bool is_origin = miller.x == 0 && miller.y == 0 && miller.z == 0;
        if (holds_half_sphere && !is_origin) {
            work[grid.g_vector_to_fft_grid_index(-miller.x, -miller.y, -miller.z)] =
                std::conj(coefficients[index]);
        }
    }

    grid.inverse_raw(work.data());
    return work;
}

/// The same transform for a field that is real by construction, e.g. a density.
inline std::vector<Real>
real_field_from_reciprocal_space(const std::array<int, 3>& grid_dimensions,
                                 const std::vector<IntegerVector3D>& miller_indices,
                                 const std::vector<Complex>& coefficients,
                                 bool holds_half_sphere) {
    const std::vector<Complex> complex_field = complex_field_from_reciprocal_space(
        grid_dimensions, miller_indices, coefficients, holds_half_sphere);
    std::vector<Real> field(complex_field.size());
    for (std::size_t index = 0; index < complex_field.size(); index++)
        field[index] = complex_field[index].real();
    return field;
}

} // namespace q3
