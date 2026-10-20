#pragma once

/**
 * FFTGridSizer — determines FFT grid dimensions from energy cutoffs and crystal symmetry.
 */

#include <algorithm>
#include <cmath>
#include <vector>

#include "layer_0_core/math/tolerances.hpp"
#include "layer_0_core/types/init.hpp"
#include "layer_1_structure/crystal/crystal.hpp"
#include "layer_1_structure/symmetry/operation.hpp"
#include "layer_1_structure/symmetry/symmetry.hpp"

namespace q3 {

class FFTGridSizer {
  public:
    /**
     * Divisibility that one component of a fractional translation imposes on the
     * FFT grid dimension along its axis.
     *
     * A translation by tau carries a grid point onto a grid point only when
     * tau * N is a whole number of steps, so a component equal to 1/n asks for N
     * divisible by n. The component is first folded into [-1/2, 1/2], since a
     * translation and its negative are the same constraint and spglib reports
     * translations in [0, 1) -- 3/4 there is -1/4 here, denominator 4 rather
     * than 1.
     *
     * round(1/|tau|) is an integer for any tau whatsoever, so the candidate
     * denominator is kept only when it reproduces the component it came from.
     * Accepting it unchecked is what read the 0.375 between the two sublattices
     * of the committed 2x2x2 silicon supercell -- no fraction of this form at
     * all -- as 1/3, and asked for a grid divisible by 3 in a cell whose
     * translations are halves.
     *
     * Returns 1, no constraint, for a component that is zero, that is not within
     * tolerance of 1/n, or whose denominator is outside {2, 3, 4, 6}. Those four
     * are the denominators a space group's own fractional translations can take
     * (the screw axes and glide planes), and they are also the only ones
     * get_good_fft_order() can be sure to satisfy, as it looks for multiples of
     * the factor among products of 2, 3 and 5 -- a denominator of 7 would send it
     * searching forever.
     */
    static int fft_factor_for_fractional_translation(Real translation_component) {
        Real folded_component = translation_component - std::round(translation_component);
        Real translation_magnitude = std::abs(folded_component);
        if (translation_magnitude <= math::tolerances::fractional_translation_rounding_tolerance) {
            return 1;
        }
        int fractional_denominator = static_cast<int>(std::round(1.0 / translation_magnitude));
        if (std::abs(translation_magnitude - 1.0 / fractional_denominator) >
            math::tolerances::fractional_translation_rounding_tolerance) {
            return 1;
        }
        if (fractional_denominator == 2 || fractional_denominator == 3 ||
            fractional_denominator == 4 || fractional_denominator == 6) {
            return fractional_denominator;
        }
        return 1;
    }

    /**
     * Compute FFT grid divisibility factors from the space group's fractional
     * translations, so that every operation the code applies moves the real-space
     * grid onto itself (Quantum ESPRESSO's fft_fact).
     *
     * The translations are those of the symmetry operations spglib accepts, not
     * differences between same-species atom pairs. Two like atoms separated by a
     * fraction of a lattice vector need not be related by any operation of the
     * group -- in a supercell most such pairs are not -- and the scan also had no
     * way to tell a real fraction from a difference that is none, which is how
     * the committed 2x2x2 silicon supercell ended up with factors (6, 6, 6)
     * where its translations ask for (4, 4, 4).
     *
     * Reading the group also makes the factor scale with the cell: the primitive
     * cell's own halves are quarters in a cell twice as long, so the supercell
     * asks for twice the divisor and the two grids keep the same spacing. A
     * factor read off atom pairs had no reason to respect that.
     */
    static void compute_fft_factors(const Crystal& crystal_structure, int fft_factors[3]) {
        fft_factors[0] = fft_factors[1] = fft_factors[2] = 1;

        const std::vector<SymmetryOperation> symmetry_operations =
            SymmetryAnalyzer::get_symmetry_operations(crystal_structure);
        for (const SymmetryOperation& operation : symmetry_operations) {
            for (int dimension = 0; dimension < 3; dimension++) {
                fft_factors[dimension] =
                    least_common_multiple(fft_factors[dimension],
                                          fft_factor_for_fractional_translation(
                                              operation.fractional_translation[dimension]));
            }
        }
    }

    /**
     * Determine minimum FFT grid dimensions for a given energy cutoff.
     */
    static void calculate_grid_dimensions(Real energy_cutoff,
                                          Real two_pi_over_lattice_parameter_squared,
                                          const Matrix3x3& direct_lattice,
                                          const Matrix3x3& reciprocal_lattice,
                                          int& grid_dimension_1,
                                          int& grid_dimension_2,
                                          int& grid_dimension_3,
                                          const int fft_factors[3] = nullptr) {
        Real g_vector_cutoff_squared = energy_cutoff / two_pi_over_lattice_parameter_squared;
        Real square_root_g_vector_cutoff_squared = std::sqrt(g_vector_cutoff_squared);

        int grid_estimate_1 = static_cast<int>(std::floor(square_root_g_vector_cutoff_squared *
                                                          direct_lattice[0].norm())) +
                              1;
        int grid_estimate_2 = static_cast<int>(std::floor(square_root_g_vector_cutoff_squared *
                                                          direct_lattice[1].norm())) +
                              1;
        int grid_estimate_3 = static_cast<int>(std::floor(square_root_g_vector_cutoff_squared *
                                                          direct_lattice[2].norm())) +
                              1;

        int maximum_miller_index_1 = 0;
        int maximum_miller_index_2 = 0;
        int maximum_miller_index_3 = 0;

        for (int index_3 = -grid_estimate_3; index_3 <= grid_estimate_3; index_3++) {
            for (int index_2 = -grid_estimate_2; index_2 <= grid_estimate_2; index_2++) {
                for (int index_1 = -grid_estimate_1; index_1 <= grid_estimate_1; index_1++) {
                    Vector3D g_cartesian;
                    for (int dimension = 0; dimension < 3; dimension++) {
                        g_cartesian[dimension] = index_1 * reciprocal_lattice[0][dimension] +
                                                 index_2 * reciprocal_lattice[1][dimension] +
                                                 index_3 * reciprocal_lattice[2][dimension];
                    }
                    Real g_squared = g_cartesian.norm2();
                    if (g_squared < g_vector_cutoff_squared +
                                        math::tolerances::energy_cutoff_rounding_tolerance) {
                        maximum_miller_index_1 =
                            std::max(maximum_miller_index_1, std::abs(index_1));
                        maximum_miller_index_2 =
                            std::max(maximum_miller_index_2, std::abs(index_2));
                        maximum_miller_index_3 =
                            std::max(maximum_miller_index_3, std::abs(index_3));
                    }
                }
            }
        }

        grid_dimension_1 = 2 * maximum_miller_index_1 + 1;
        grid_dimension_2 = 2 * maximum_miller_index_2 + 1;
        grid_dimension_3 = 2 * maximum_miller_index_3 + 1;

        int factor_1 = (fft_factors != nullptr) ? fft_factors[0] : 1;
        int factor_2 = (fft_factors != nullptr) ? fft_factors[1] : 1;
        int factor_3 = (fft_factors != nullptr) ? fft_factors[2] : 1;

        grid_dimension_1 = get_good_fft_order(grid_dimension_1, factor_1);
        grid_dimension_2 = get_good_fft_order(grid_dimension_2, factor_2);
        grid_dimension_3 = get_good_fft_order(grid_dimension_3, factor_3);
    }

    /**
     * Find good FFT dimension (multiple of small primes).
     */
    static int get_good_fft_dimension(int number_of_points) {
        return get_good_fft_order(number_of_points);
    }

    /**
     * Find smallest n >= number_of_points that is a product of 2, 3, 5 and divisible by
     * division_factor.
     */
    static int get_good_fft_order(int number_of_points, int division_factor = 1) {
        int candidate_points = number_of_points;
        while (true) {
            if (division_factor > 1 && candidate_points % division_factor != 0) {
                candidate_points++;
                continue;
            }
            int remaining_factors = candidate_points;
            while (remaining_factors % 2 == 0)
                remaining_factors /= 2;
            while (remaining_factors % 3 == 0)
                remaining_factors /= 3;
            while (remaining_factors % 5 == 0)
                remaining_factors /= 5;
            if (remaining_factors == 1)
                return candidate_points;
            candidate_points++;
        }
    }

  private:
    static int least_common_multiple(int a, int b) {
        int greater_value = a;
        int smaller_value = b;
        while (smaller_value) {
            int temporary = smaller_value;
            smaller_value = greater_value % smaller_value;
            greater_value = temporary;
        }
        return a / greater_value * b;
    }
};

} // namespace q3
