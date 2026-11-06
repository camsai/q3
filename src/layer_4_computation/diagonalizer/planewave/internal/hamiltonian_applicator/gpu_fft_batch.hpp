#pragma once

/**
 * How many bands one batched FFT pass of the GPU Hamiltonian applicator holds.
 *
 * The GPU applicator applies the local potential to a block of bands at once:
 * every band is scattered onto its own copy of the smooth FFT grid in one
 * workspace, the whole workspace is transformed in one batched FFT, and the
 * bands are gathered back. That workspace has grid points × bands elements, and
 * a large cell overflows it: a 144³ grid with 800 bands is 2.39e9 elements,
 * past the 2³¹ − 1 that the workspace size, the kernels' element counts and
 * their thread indices are held in. The block is therefore split into passes
 * whose workspace stays at or below a fixed element budget.
 *
 * The budget, 2²⁸ elements, sits far enough below 2³¹ that a kernel's last
 * thread block cannot carry its index past the limit, and it bounds memory: the
 * FP64 workspace is then at most 4 GiB, and the batched FFT keeps a second
 * buffer of the same size. A 144³ grid still takes 90 bands per pass.
 */

#include <algorithm>
#include <cstdint>

namespace q3 {
namespace diagonalizer {
namespace planewave {

/// Element budget of one batched FFT workspace (grid points × bands).
inline constexpr std::int64_t maximum_fft_batch_workspace_elements = std::int64_t{1} << 28;

/**
 * Bands per batched FFT pass, for `number_of_vectors` bands on a grid of
 * `local_grid_points` points.
 *
 * All the bands, when their workspace fits the budget. Otherwise the bands are
 * shared out evenly over the fewest passes that fit, so that every pass but the
 * last holds the same count and the batched FFT plan, built for one batch size,
 * serves every pass. At least one band per pass, whatever the grid.
 */
inline int fft_batch_band_count(
    int local_grid_points,
    int number_of_vectors,
    std::int64_t maximum_workspace_elements = maximum_fft_batch_workspace_elements) {
    if (number_of_vectors <= 0 || local_grid_points <= 0) {
        return std::max(number_of_vectors, 0);
    }
    const std::int64_t grid_points = local_grid_points;
    const std::int64_t vectors = number_of_vectors;
    if (grid_points * vectors <= maximum_workspace_elements) {
        return number_of_vectors;
    }
    const std::int64_t bands_that_fit =
        std::max<std::int64_t>(1, maximum_workspace_elements / grid_points);
    const std::int64_t passes = (vectors + bands_that_fit - 1) / bands_that_fit;
    return static_cast<int>((vectors + passes - 1) / passes);
}

} // namespace planewave
} // namespace diagonalizer
} // namespace q3
