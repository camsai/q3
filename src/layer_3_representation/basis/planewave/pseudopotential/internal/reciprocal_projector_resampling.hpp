#pragma once

/**
 * Resample a dataset-supplied β(q) onto Q3's q-grid.
 *
 * Most pseudopotential formats give projectors radially, as r·β(r), and Q3
 * obtains β(q) by spherical Bessel transform. Some datasets give β(q)
 * directly, in the same convention Q3's own transform produces
 * (β(q) = 4π ∫ β(r) j_l(qr) r² dr), so the only work is moving it from the
 * dataset's uniform q-grid onto Q3's.
 *
 * Doing that instead of inverting to r and transforming back is not merely
 * cheaper. The dataset's table is finite, so an inversion truncates, and the
 * truncation shows up as a projector that no
 * longer integrates to unity against its own partial wave. Resampling touches
 * only q values below the plane-wave cutoff, well inside the tabulated range,
 * and inherits no such error.
 *
 * Interpolation is the Catmull-Rom cubic through the four surrounding samples,
 * which is local and continuous in value and slope. Beyond the
 * tabulated range the projector is zero: the dataset's q_max is far above any
 * cutoff a calculation would use, so reaching past it means the basis exceeded
 * what the dataset describes, and extrapolating a projector there would invent
 * structure the dataset never claimed.
 */

#include <cstddef>
#include <vector>

#include "layer_0_core/types/init.hpp"

namespace q3 {

/**
 * Interpolate a uniformly tabulated function onto arbitrary q values.
 *
 * The table starts at q = 0, so the cubic's first interval needs a sample at
 * q = −spacing that no dataset stores. It follows from parity rather than
 * being invented: β_l(q) is even in q for even l and odd for odd l, so the
 * point below zero is ±β(+spacing), signed by the parity of l. The choice
 * matters: clamping instead
 * costs an order of magnitude in accuracy over the first interval, which is
 * the long-wavelength end where projectors carry the most weight.
 *
 * @param samples           f(q_i) at q_i = i · spacing, i = 0 … samples.size()-1
 * @param spacing           grid spacing of the supplied table
 * @param angular_momentum  l, which fixes the parity of the below-zero sample
 * @param q_values          q values to evaluate at
 * @param result            [out] f(q), resized to q_values.size()
 */
inline void resample_uniform_reciprocal_table(const std::vector<Real>& samples,
                                              Real spacing,
                                              int angular_momentum,
                                              const std::vector<Real>& q_values,
                                              std::vector<Real>& result) {
    result.assign(q_values.size(), 0.0);
    if (samples.size() < 2 || spacing <= 0.0)
        return;

    const Real parity_sign = (angular_momentum % 2 == 0) ? 1.0 : -1.0;
    auto sample_at = [&samples, parity_sign](long index) -> Real {
        if (index < 0)
            return parity_sign * samples[static_cast<size_t>(-index)];
        if (index >= static_cast<long>(samples.size()))
            return 0.0;
        return samples[static_cast<size_t>(index)];
    };

    const Real last_q = static_cast<Real>(samples.size() - 1) * spacing;
    for (size_t point = 0; point < q_values.size(); point++) {
        Real q_value = q_values[point];
        if (q_value < 0.0 || q_value > last_q)
            continue; // outside what the dataset describes

        Real scaled = q_value / spacing;
        auto lower = static_cast<long>(scaled);
        if (lower >= static_cast<long>(samples.size()) - 1)
            lower = static_cast<long>(samples.size()) - 2;
        Real fraction = scaled - static_cast<Real>(lower);

        Real previous = sample_at(lower - 1);
        Real current = sample_at(lower);
        Real next = sample_at(lower + 1);
        Real after = sample_at(lower + 2);

        // Catmull-Rom.
        Real a = -0.5 * previous + 1.5 * current - 1.5 * next + 0.5 * after;
        Real b = previous - 2.5 * current + 2.0 * next - 0.5 * after;
        Real c = -0.5 * previous + 0.5 * next;
        result[point] = ((a * fraction + b) * fraction + c) * fraction + current;
    }
}

} // namespace q3
