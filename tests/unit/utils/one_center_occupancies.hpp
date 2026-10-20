#pragma once

/**
 * Random on-site occupancies for one-center tests.
 *
 * Symmetric, as a real SCF produces them, and seeded so a failing case can be
 * reproduced. Randomized occupancies matter: an identity can hold
 * accidentally for the diagonal, physically plausible case and fail for the
 * cross terms a real SCF actually produces.
 */

#include <random>

#include "layer_0_core/types/init.hpp"
#include "layer_4_computation/paw/one_center/densities.hpp"

namespace q3::test_support {

/// Symmetric random occupancies, as a real SCF would produce.
inline paw_one_center::OnSiteOccupancies random_occupancies(int number_of_channels, unsigned seed) {
    std::mt19937 generator(seed);
    std::uniform_real_distribution<Real> distribution(-0.4, 1.0);
    auto occupancies = paw_one_center::OnSiteOccupancies::zeros(number_of_channels);
    for (int first = 0; first < number_of_channels; first++) {
        for (int second = first; second < number_of_channels; second++) {
            Real value = distribution(generator);
            occupancies.at(0, first, second) = value;
            occupancies.at(0, second, first) = value;
        }
    }
    return occupancies;
}

} // namespace q3::test_support
