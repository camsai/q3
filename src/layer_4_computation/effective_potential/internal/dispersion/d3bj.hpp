#pragma once

/**
 * D3BJDispersionCalculator — Grimme DFT-D3 dispersion energy, Becke-Johnson
 * damping (two-body term only; three-body Axilrod-Teller-Muto is out of
 * scope, matching this feature's own matched-inputs QE protocol which pins
 * dftd3_threebody=.false.).
 *
 * S. Grimme, J. Antony, S. Ehrlich, H. Krieg, J. Chem. Phys. 132, 154104 (2010)
 * S. Grimme, S. Ehrlich, L. Goerigk, J. Comput. Chem. 32, 1456 (2011) (BJ damping)
 *
 * Algorithm (energy only):
 *   1. Coordination number CN_i = sum over periodic images of a smooth
 *      bond-counting function of r_ij vs. the covalent-radius sum.
 *   2. C6_ij from a Gaussian-weighted average over the reference (CN, CN, C6)
 *      grid for that element pair — NOT a lookup at fixed CN, per Grimme
 *      eq. 8. C8_ij = 3 * C6_ij * r2r4_i * r2r4_j.
 *   3. E_disp = -(s6 * E6 + s8 * E8), with
 *        E6 = sum_{pairs, images} C6_ij / (r_ij^6 + R0_ij^6)
 *        E8 = sum_{pairs, images} C8_ij / (r_ij^8 + R0_ij^8)
 *      and BJ damping radius R0_ij = a1*sqrt(3*r2r4_i*r2r4_j) + a2.
 *
 * Functional-specific (a1, a2, s8) parameters below are PBE's, matching the
 * only functional Q3 currently ships; s6 = 1 always for BJ damping.
 *
 * Only elements in d3bj_reference_data.hpp are supported; species outside
 * that set fail loudly rather than silently contributing zero dispersion.
 */

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "layer_0_core/types/init.hpp"
#include "layer_1_structure/crystal/crystal.hpp"
#include "layer_4_computation/effective_potential/internal/dispersion/d3bj_reference_data.hpp"
#include "layer_4_computation/effective_potential/internal/dispersion/init.hpp"

namespace q3 {

class D3BJDispersionCalculator : public DispersionCalculator {
  public:
    explicit D3BJDispersionCalculator(const Crystal& crystal_structure)
        : crystal_structure_(crystal_structure) {}

    [[nodiscard]] std::string name() const override {
        return "d3bj";
    }

    /// Total D3(BJ) dispersion energy correction, in Rydberg.
    [[nodiscard]] Real compute_energy() const override {
        int number_of_atoms = crystal_structure_.number_of_atoms();
        std::vector<int> atomic_numbers(number_of_atoms);
        std::vector<Vector3D> cartesian_positions_bohr(number_of_atoms);
        Matrix3x3 lattice_vectors_bohr = crystal_structure_.lattice_vectors_bohr();
        for (int atom_index = 0; atom_index < number_of_atoms; atom_index++) {
            atomic_numbers[atom_index] = crystal_structure_.atomic_number(atom_index);
            if (dispersion::rcov_by_atomic_number(atomic_numbers[atom_index]) == 0.0) {
                throw std::runtime_error("D3(BJ) dispersion: no reference data for element '" +
                                         crystal_structure_.element(atom_index) +
                                         "'. Supported elements: C, Si, Cr, Fe, Ni, Cu.");
            }
            /**
             * convert_from_crystal_to_cartesian() returns positions in units
             * of alat, not Bohr -- project through the Bohr-scaled lattice
             * matrix directly instead.
             */
            const Vector3D& fractional = crystal_structure_.fractional_position(atom_index);
            cartesian_positions_bohr[atom_index] = fractional.x * lattice_vectors_bohr[0] +
                                                   fractional.y * lattice_vectors_bohr[1] +
                                                   fractional.z * lattice_vectors_bohr[2];
        }

        std::vector<Real> coordination_numbers = compute_coordination_numbers(
            atomic_numbers, cartesian_positions_bohr, lattice_vectors_bohr);
        std::vector<Real> r2r4(number_of_atoms);
        for (int atom_index = 0; atom_index < number_of_atoms; atom_index++) {
            r2r4[atom_index] = dispersion::r2r4_by_atomic_number(atomic_numbers[atom_index]);
        }

        /**
         * Periodic-image shifts covering the van der Waals cutoff sphere
         * (QE default: cutoff_squared = 9000 Bohr^2), precomputed once for
         * the whole pair sum -- see enumerate_image_shifts().
         */
        constexpr Real vdw_cutoff_squared_bohr = 9000.0;
        const std::vector<LatticeImageShift> vdw_image_shifts =
            enumerate_image_shifts(lattice_vectors_bohr, vdw_cutoff_squared_bohr);

        Real energy_6 = 0.0;
        Real energy_8 = 0.0;
        for (int atom_i = 0; atom_i < number_of_atoms; atom_i++) {
            for (int atom_j = atom_i; atom_j < number_of_atoms; atom_j++) {
                Real c6 = interpolate_c6(atomic_numbers[atom_i],
                                         atomic_numbers[atom_j],
                                         coordination_numbers[atom_i],
                                         coordination_numbers[atom_j]);
                Real c8 = 3.0 * c6 * r2r4[atom_i] * r2r4[atom_j];
                Real damping_radius = becke_johnson_damping_radius(r2r4[atom_i], r2r4[atom_j]);
                Real damping_radius_6 = ipow(damping_radius, 6);
                Real damping_radius_8 = ipow(damping_radius, 8);

                Vector3D separation_bohr =
                    cartesian_positions_bohr[atom_i] - cartesian_positions_bohr[atom_j];
                bool is_self_pair = (atom_i == atom_j);
                /**
                 * Unordered-pair (i<j) contributions are counted once; the
                 * self-pair (i==i) periodic-image sum double-counts every
                 * image with its mirror translation, hence 0.5.
                 */
                Real weight = is_self_pair ? 0.5 : 1.0;

                for (const LatticeImageShift& image : vdw_image_shifts) {
                    if (is_self_pair && image.is_origin) {
                        continue;
                    }
                    Vector3D image_separation_bohr = separation_bohr + image.shift_bohr;
                    Real distance_squared_bohr = image_separation_bohr.x * image_separation_bohr.x +
                                                 image_separation_bohr.y * image_separation_bohr.y +
                                                 image_separation_bohr.z * image_separation_bohr.z;
                    if (distance_squared_bohr > vdw_cutoff_squared_bohr) {
                        continue;
                    }
                    Real distance_6_bohr = ipow(distance_squared_bohr, 3);
                    Real distance_8_bohr = distance_6_bohr * distance_squared_bohr;
                    energy_6 += weight * c6 / (distance_6_bohr + damping_radius_6);
                    energy_8 += weight * c8 / (distance_8_bohr + damping_radius_8);
                }
            }
        }

        constexpr Real s6 = 1.0;    // PBE, BJ damping (always 1 for this damping scheme)
        constexpr Real s8 = 0.7875; // PBE, BJ damping (Grimme & al. 2011, version "4")
        constexpr Real hartree_to_rydberg = 2.0;
        return -(s6 * energy_6 + s8 * energy_8) * hartree_to_rydberg;
    }

  private:
    static constexpr Real k1_ = 16.0;      // CN counting-function steepness
    static constexpr Real k3_ = -4.0;      // C6 interpolation Gaussian weight
    static constexpr Real bj_a1_ = 0.4289; // PBE, BJ damping radius scale
    static constexpr Real bj_a2_ = 4.4407; // PBE, BJ damping radius offset

    [[nodiscard]] static Real ipow(Real base, int exponent) {
        Real result = 1.0;
        for (int i = 0; i < exponent; i++) {
            result *= base;
        }
        return result;
    }

    [[nodiscard]] static Real becke_johnson_damping_radius(Real r2r4_i, Real r2r4_j) {
        return bj_a1_ * std::sqrt(3.0 * r2r4_i * r2r4_j) + bj_a2_;
    }

    /**
     * Real-space translation range (in lattice-vector units) needed to cover
     * a sphere of the given squared radius, matching QE's SET_CRITERIA: the
     * number of cells along each reciprocal direction is set by how many
     * interplanar spacings fit inside the cutoff.
     */
    [[nodiscard]] static IntegerVector3D
    lattice_translation_range(const Matrix3x3& lattice_vectors_bohr, Real cutoff_squared_bohr) {
        Real cutoff_bohr = std::sqrt(cutoff_squared_bohr);
        Real cell_volume = std::abs(lattice_vectors_bohr.determinant());
        IntegerVector3D range;
        for (int direction = 0; direction < 3; direction++) {
            const Vector3D& first = lattice_vectors_bohr[(direction + 1) % 3];
            const Vector3D& second = lattice_vectors_bohr[(direction + 2) % 3];
            Real interplanar_spacing = cell_volume / first.cross(second).norm();
            range[direction] = static_cast<int>(cutoff_bohr / interplanar_spacing) + 1;
        }
        return range;
    }

    /**
     * One precomputed periodic-image shift; is_origin marks the (0,0,0) cell,
     * which self-pair (i == i) sums must skip.
     */
    struct LatticeImageShift {
        Vector3D shift_bohr;
        bool is_origin;
    };

    /**
     * Every periodic-image shift inside the translation range covering the
     * given cutoff, precomputed once per lattice sum instead of re-deriving
     * the shift vector inside triple-nested translation loops at every atom
     * pair. Enumeration order is exactly the previous x-outer / y-middle /
     * z-inner loop order, so pair sums iterating this list accumulate in the
     * identical order and stay bit-identical to the pre-flattened code.
     */
    [[nodiscard]] static std::vector<LatticeImageShift>
    enumerate_image_shifts(const Matrix3x3& lattice_vectors_bohr, Real cutoff_squared_bohr) {
        IntegerVector3D range =
            lattice_translation_range(lattice_vectors_bohr, cutoff_squared_bohr);
        std::vector<LatticeImageShift> image_shifts;
        image_shifts.reserve(static_cast<std::size_t>(2 * range.x + 1) *
                             static_cast<std::size_t>(2 * range.y + 1) *
                             static_cast<std::size_t>(2 * range.z + 1));
        for (int translation_x = -range.x; translation_x <= range.x; translation_x++) {
            for (int translation_y = -range.y; translation_y <= range.y; translation_y++) {
                for (int translation_z = -range.z; translation_z <= range.z; translation_z++) {
                    image_shifts.push_back(
                        {translation_x * lattice_vectors_bohr[0] +
                             translation_y * lattice_vectors_bohr[1] +
                             translation_z * lattice_vectors_bohr[2],
                         translation_x == 0 && translation_y == 0 && translation_z == 0});
                }
            }
        }
        return image_shifts;
    }

    [[nodiscard]] std::vector<Real>
    compute_coordination_numbers(const std::vector<int>& atomic_numbers,
                                 const std::vector<Vector3D>& cartesian_positions_bohr,
                                 const Matrix3x3& lattice_vectors_bohr) const {
        constexpr Real cn_cutoff_squared_bohr = 1600.0; // QE default: cutoff_cn = sqrt(1600) Bohr
        const std::vector<LatticeImageShift> cn_image_shifts =
            enumerate_image_shifts(lattice_vectors_bohr, cn_cutoff_squared_bohr);
        int number_of_atoms = static_cast<int>(atomic_numbers.size());
        std::vector<Real> covalent_radii(number_of_atoms);
        for (int atom_index = 0; atom_index < number_of_atoms; atom_index++) {
            covalent_radii[atom_index] =
                dispersion::rcov_by_atomic_number(atomic_numbers[atom_index]);
        }
        std::vector<Real> coordination_numbers(number_of_atoms, 0.0);

        for (int atom_i = 0; atom_i < number_of_atoms; atom_i++) {
            for (int atom_j = 0; atom_j < number_of_atoms; atom_j++) {
                Real covalent_radius_sum = covalent_radii[atom_i] + covalent_radii[atom_j];
                Vector3D separation_bohr =
                    cartesian_positions_bohr[atom_j] - cartesian_positions_bohr[atom_i];
                for (const LatticeImageShift& image : cn_image_shifts) {
                    if (atom_i == atom_j && image.is_origin) {
                        continue;
                    }
                    Vector3D image_separation_bohr = separation_bohr + image.shift_bohr;
                    Real distance_squared_bohr = image_separation_bohr.x * image_separation_bohr.x +
                                                 image_separation_bohr.y * image_separation_bohr.y +
                                                 image_separation_bohr.z * image_separation_bohr.z;
                    if (distance_squared_bohr > cn_cutoff_squared_bohr) {
                        continue;
                    }
                    Real distance_bohr = std::sqrt(distance_squared_bohr);
                    Real ratio = covalent_radius_sum / distance_bohr;
                    coordination_numbers[atom_i] += 1.0 / (1.0 + std::exp(-k1_ * (ratio - 1.0)));
                }
            }
        }
        return coordination_numbers;
    }

    /**
     * Value and coordination-number partial derivatives of the
     * Gaussian-weighted C6 average (Grimme 2010, eq. 8), computed in a
     * single pass over the reference table for this element pair so that
     * `interpolate_c6()` and the gradient path can never disagree on the
     * value itself.
     */
    struct C6InterpolationResult {
        Real value = 0.0;
        Real derivative_with_respect_to_coordination_number_i = 0.0;
        Real derivative_with_respect_to_coordination_number_j = 0.0;
    };

    [[nodiscard]] static C6InterpolationResult
    interpolate_c6_and_derivatives(int atomic_number_i,
                                   int atomic_number_j,
                                   Real coordination_number_i,
                                   Real coordination_number_j) {
        Real weighted_sum = 0.0;         // sum_ref w_ref * C6_ref
        Real weight_sum = 0.0;           // sum_ref w_ref
        Real weighted_offset_i = 0.0;    // sum_ref w_ref * (CN_i - CNref_i)
        Real weighted_offset_j = 0.0;    // sum_ref w_ref * (CN_j - CNref_j)
        Real weighted_c6_offset_i = 0.0; // sum_ref w_ref * C6_ref * (CN_i - CNref_i)
        Real weighted_c6_offset_j = 0.0; // sum_ref w_ref * C6_ref * (CN_j - CNref_j)
        /**
         * Fallback for when the actual coordination number falls far outside
         * the fitted reference grid (common for close-packed bulk metals,
         * whose CN often exceeds every tabulated reference point): matches
         * QE's getc6() c6mem/r_save behavior of using the single nearest
         * reference point's C6 rather than an interpolated (numerically
         * underflowing) average.
         */
        Real nearest_c6 = 0.0;
        Real nearest_distance_squared = std::numeric_limits<Real>::max();
        bool found_reference_point = false;
        for (const auto& point : dispersion::c6_reference_table()) {
            /**
             * The table is pre-symmetrized (both (Z_i,Z_j) and (Z_j,Z_i)
             * orderings are present as separate entries), so a direct match
             * is sufficient -- no separate swapped-order check needed.
             */
            if (point.atomic_number_i != atomic_number_i ||
                point.atomic_number_j != atomic_number_j) {
                continue;
            }
            found_reference_point = true;
            Real offset_i = coordination_number_i - point.coordination_number_i;
            Real offset_j = coordination_number_j - point.coordination_number_j;
            Real cn_distance_squared = offset_i * offset_i + offset_j * offset_j;
            if (cn_distance_squared < nearest_distance_squared) {
                nearest_distance_squared = cn_distance_squared;
                nearest_c6 = point.c6;
            }
            Real weight = std::exp(k3_ * cn_distance_squared);
            weighted_sum += point.c6 * weight;
            weight_sum += weight;
            weighted_offset_i += weight * offset_i;
            weighted_offset_j += weight * offset_j;
            weighted_c6_offset_i += weight * point.c6 * offset_i;
            weighted_c6_offset_j += weight * point.c6 * offset_j;
        }
        if (!found_reference_point) {
            throw std::runtime_error("D3(BJ) dispersion: no C6 reference data for element pair (" +
                                     std::to_string(atomic_number_i) + ", " +
                                     std::to_string(atomic_number_j) + ")");
        }
        /**
         * Underflow guard, mirroring dftd3's reference implementation: when all
         * Gaussian CN weights underflow, fall back to the nearest reference point.
         * In that regime C6 is locally constant (deep in the Gaussian tail), so
         * its coordination-number derivative is consistently taken to be zero.
         */
        constexpr Real coordination_weight_underflow_floor = 1e-99;
        if (weight_sum <= coordination_weight_underflow_floor) {
            return {nearest_c6, 0.0, 0.0};
        }
        Real value = weighted_sum / weight_sum;
        Real derivative_i =
            2.0 * k3_ / weight_sum * (weighted_c6_offset_i - value * weighted_offset_i);
        Real derivative_j =
            2.0 * k3_ / weight_sum * (weighted_c6_offset_j - value * weighted_offset_j);
        return {value, derivative_i, derivative_j};
    }

    /**
     * Gaussian-weighted average of C6 over all (CN_ref_i, CN_ref_j) grid
     * points for this element pair (Grimme 2010, eq. 8) — not a lookup at
     * the nearest reference point.
     */
    [[nodiscard]] static Real interpolate_c6(int atomic_number_i,
                                             int atomic_number_j,
                                             Real coordination_number_i,
                                             Real coordination_number_j) {
        return interpolate_c6_and_derivatives(
                   atomic_number_i, atomic_number_j, coordination_number_i, coordination_number_j)
            .value;
    }

  public:
    /**
     * Analytical gradients of the D3(BJ) dispersion energy (forces and
     * stress; units and sign conventions documented on DispersionGradients).
     *
     * Two coupled mechanisms contribute, mirroring compute_energy() exactly:
     *   1. Direct: r_ij appears explicitly in 1/(r^6+R0^6) and 1/(r^8+R0^8).
     *   2. Indirect: C6_ij depends on CN_i and CN_j (interpolate_c6), and
     *      each coordination number is itself a smooth sum over ALL of that
     *      atom's neighbors -- so a displaced neighbor changes C6_ij even
     *      for pairs it is not directly part of. Handled via a standard
     *      two-pass algorithm: pass 1 walks the same (C6, C8) pair loop as
     *      compute_energy() and accumulates dE/dCN_i per atom; pass 2 walks
     *      the same neighbor loop as compute_coordination_numbers() and
     *      applies the chain rule through dCN_i/dR.
     */
    [[nodiscard]] DispersionGradients compute_gradients() const override {
        int number_of_atoms = crystal_structure_.number_of_atoms();
        std::vector<int> atomic_numbers(number_of_atoms);
        std::vector<Vector3D> cartesian_positions_bohr(number_of_atoms);
        Matrix3x3 lattice_vectors_bohr = crystal_structure_.lattice_vectors_bohr();
        for (int atom_index = 0; atom_index < number_of_atoms; atom_index++) {
            atomic_numbers[atom_index] = crystal_structure_.atomic_number(atom_index);
            if (dispersion::rcov_by_atomic_number(atomic_numbers[atom_index]) == 0.0) {
                throw std::runtime_error("D3(BJ) dispersion: no reference data for element '" +
                                         crystal_structure_.element(atom_index) +
                                         "'. Supported elements: C, Si, Cr, Fe, Ni, Cu.");
            }
            const Vector3D& fractional = crystal_structure_.fractional_position(atom_index);
            cartesian_positions_bohr[atom_index] = fractional.x * lattice_vectors_bohr[0] +
                                                   fractional.y * lattice_vectors_bohr[1] +
                                                   fractional.z * lattice_vectors_bohr[2];
        }

        std::vector<Real> coordination_numbers = compute_coordination_numbers(
            atomic_numbers, cartesian_positions_bohr, lattice_vectors_bohr);
        std::vector<Real> r2r4(number_of_atoms);
        for (int atom_index = 0; atom_index < number_of_atoms; atom_index++) {
            r2r4[atom_index] = dispersion::r2r4_by_atomic_number(atomic_numbers[atom_index]);
        }

        constexpr Real vdw_cutoff_squared_bohr = 9000.0;
        const std::vector<LatticeImageShift> vdw_image_shifts =
            enumerate_image_shifts(lattice_vectors_bohr, vdw_cutoff_squared_bohr);

        constexpr Real s6 = 1.0;
        constexpr Real s8 = 0.7875;
        constexpr Real hartree_to_rydberg = 2.0;

        std::vector<Vector3D> forces(number_of_atoms, {0.0, 0.0, 0.0});
        Real stress_raw[3][3] = {};
        /**
         * g_i = dE/dCN_i, accumulated across every (C6, C8) pair atom i takes
         * part in (either role, including its own self-pair).
         */
        std::vector<Real> coordination_number_energy_derivative(number_of_atoms, 0.0);

        // PASS 1: direct pairwise force + stress, plus accumulation of g_i.
        for (int atom_i = 0; atom_i < number_of_atoms; atom_i++) {
            for (int atom_j = atom_i; atom_j < number_of_atoms; atom_j++) {
                C6InterpolationResult c6_result =
                    interpolate_c6_and_derivatives(atomic_numbers[atom_i],
                                                   atomic_numbers[atom_j],
                                                   coordination_numbers[atom_i],
                                                   coordination_numbers[atom_j]);
                Real c6 = c6_result.value;
                Real c8 = 3.0 * c6 * r2r4[atom_i] * r2r4[atom_j];
                Real r2r4_product_times_three = 3.0 * r2r4[atom_i] * r2r4[atom_j];
                Real damping_radius = becke_johnson_damping_radius(r2r4[atom_i], r2r4[atom_j]);
                Real damping_radius_6 = ipow(damping_radius, 6);
                Real damping_radius_8 = ipow(damping_radius, 8);

                Vector3D separation_bohr =
                    cartesian_positions_bohr[atom_i] - cartesian_positions_bohr[atom_j];
                bool is_self_pair = (atom_i == atom_j);
                Real weight = is_self_pair ? 0.5 : 1.0;

                // dE_pair/dCN_i and dE_pair/dCN_j, summed over this pair's images.
                Real pair_derivative_with_respect_to_cn_i = 0.0;
                Real pair_derivative_with_respect_to_cn_j = 0.0;

                for (const LatticeImageShift& image : vdw_image_shifts) {
                    if (is_self_pair && image.is_origin) {
                        continue;
                    }
                    Vector3D image_separation_bohr = separation_bohr + image.shift_bohr;
                    Real distance_squared_bohr = image_separation_bohr.x * image_separation_bohr.x +
                                                 image_separation_bohr.y * image_separation_bohr.y +
                                                 image_separation_bohr.z * image_separation_bohr.z;
                    if (distance_squared_bohr > vdw_cutoff_squared_bohr) {
                        continue;
                    }
                    Real distance_6_bohr = ipow(distance_squared_bohr, 3);
                    Real distance_8_bohr = distance_6_bohr * distance_squared_bohr;
                    Real denominator_6 = distance_6_bohr + damping_radius_6;
                    Real denominator_8 = distance_8_bohr + damping_radius_8;

                    pair_derivative_with_respect_to_cn_i +=
                        weight * (s6 * c6_result.derivative_with_respect_to_coordination_number_i /
                                      denominator_6 +
                                  s8 * r2r4_product_times_three *
                                      c6_result.derivative_with_respect_to_coordination_number_i /
                                      denominator_8);
                    pair_derivative_with_respect_to_cn_j +=
                        weight * (s6 * c6_result.derivative_with_respect_to_coordination_number_j /
                                      denominator_6 +
                                  s8 * r2r4_product_times_three *
                                      c6_result.derivative_with_respect_to_coordination_number_j /
                                      denominator_8);

                    /**
                     * Direct term: d(1/(r^6+R0^6))/dR and d(1/(r^8+R0^8))/dR,
                     * expressed sqrt-free via r^2 (matching compute_energy()).
                     */
                    Real distance_squared_power_2 = distance_squared_bohr * distance_squared_bohr;
                    Real direct_coefficient =
                        hartree_to_rydberg * weight *
                        (6.0 * s6 * c6 * distance_squared_power_2 /
                             (denominator_6 * denominator_6) +
                         8.0 * s8 * c8 * distance_6_bohr / (denominator_8 * denominator_8));

                    /**
                     * Force: F_i = -dE/dR_i. Self-pair separations are pure
                     * lattice vectors (position-independent), so the two
                     * increments below cancel automatically for atom_i==atom_j
                     * without any special-casing -- they still contribute to
                     * stress below, since the cell shape does move them.
                     */
                    Vector3D pair_force_on_i = (-direct_coefficient) * image_separation_bohr;
                    forces[atom_i] += pair_force_on_i;
                    forces[atom_j] += (-pair_force_on_i);

                    for (int direction_a = 0; direction_a < 3; direction_a++) {
                        for (int direction_b = 0; direction_b < 3; direction_b++) {
                            stress_raw[direction_a][direction_b] +=
                                direct_coefficient * image_separation_bohr[direction_a] *
                                image_separation_bohr[direction_b];
                        }
                    }
                }
                coordination_number_energy_derivative[atom_i] +=
                    -hartree_to_rydberg * pair_derivative_with_respect_to_cn_i;
                coordination_number_energy_derivative[atom_j] +=
                    -hartree_to_rydberg * pair_derivative_with_respect_to_cn_j;
            }
        }

        /**
         * PASS 2: chain rule through dCN_i/dR, walking the identical neighbor
         * loop as compute_coordination_numbers().
         */
        constexpr Real cn_cutoff_squared_bohr = 1600.0;
        const std::vector<LatticeImageShift> cn_image_shifts =
            enumerate_image_shifts(lattice_vectors_bohr, cn_cutoff_squared_bohr);
        std::vector<Real> covalent_radii(number_of_atoms);
        for (int atom_index = 0; atom_index < number_of_atoms; atom_index++) {
            covalent_radii[atom_index] =
                dispersion::rcov_by_atomic_number(atomic_numbers[atom_index]);
        }
        for (int atom_i = 0; atom_i < number_of_atoms; atom_i++) {
            Real energy_derivative_i = coordination_number_energy_derivative[atom_i];
            for (int atom_j = 0; atom_j < number_of_atoms; atom_j++) {
                Real covalent_radius_sum = covalent_radii[atom_i] + covalent_radii[atom_j];
                Vector3D separation_bohr =
                    cartesian_positions_bohr[atom_j] - cartesian_positions_bohr[atom_i];
                for (const LatticeImageShift& image : cn_image_shifts) {
                    if (atom_i == atom_j && image.is_origin) {
                        continue;
                    }
                    Vector3D image_separation_bohr = separation_bohr + image.shift_bohr;
                    Real distance_squared_bohr = image_separation_bohr.x * image_separation_bohr.x +
                                                 image_separation_bohr.y * image_separation_bohr.y +
                                                 image_separation_bohr.z * image_separation_bohr.z;
                    if (distance_squared_bohr > cn_cutoff_squared_bohr) {
                        continue;
                    }
                    Real distance_bohr = std::sqrt(distance_squared_bohr);
                    Real ratio = covalent_radius_sum / distance_bohr;
                    Real logistic_argument = -k1_ * (ratio - 1.0);
                    Real logistic_value = 1.0 / (1.0 + std::exp(logistic_argument));
                    /**
                     * d(logistic)/dr = -k1*(ratio/r)*L*(1-L); CN decreases as
                     * atoms move apart, so this is negative.
                     */
                    Real counting_function_derivative =
                        -k1_ * (ratio / distance_bohr) * logistic_value * (1.0 - logistic_value);

                    Real coefficient =
                        energy_derivative_i * counting_function_derivative / distance_bohr;
                    Vector3D gradient_contribution = coefficient * image_separation_bohr;
                    forces[atom_i] += gradient_contribution;
                    forces[atom_j] += (-gradient_contribution);

                    for (int direction_a = 0; direction_a < 3; direction_a++) {
                        for (int direction_b = 0; direction_b < 3; direction_b++) {
                            stress_raw[direction_a][direction_b] +=
                                coefficient * image_separation_bohr[direction_a] *
                                image_separation_bohr[direction_b];
                        }
                    }
                }
            }
        }

        Real cell_volume_bohr_cubed = crystal_structure_.cell_volume();
        Matrix3x3 stress;
        for (int direction_a = 0; direction_a < 3; direction_a++) {
            for (int direction_b = 0; direction_b < 3; direction_b++) {
                stress[direction_a][direction_b] =
                    -stress_raw[direction_a][direction_b] / cell_volume_bohr_cubed;
            }
        }

        return {forces, stress};
    }

  private:
    const Crystal& crystal_structure_;
};

} // namespace q3
