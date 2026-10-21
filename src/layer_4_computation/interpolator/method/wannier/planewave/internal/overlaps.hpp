#pragma once

/**
 * PlanewaveBlochOverlapCalculator — M_mn(k,b) = <u_mk|u_n,k+b>, the
 * neighbor overlap between Bloch periodic parts that Wannier localization
 * and disentanglement (Marzari & Vanderbilt, PRB 56, 12847 (1997)) are
 * built from.
 *
 * The plane-wave part exploits that b is always an exact grid-step
 * multiple (KPointNeighborShells::resolve_neighbor()): k+b equals some
 * OTHER physical k-point k' on the same grid plus an integer
 * reciprocal-lattice (Miller-index) shift G0, so
 *
 *   c_{n,k+b}(G) = c_{n,k'}(G + G0)
 *
 * exactly -- no interpolation, no recomputation of anything at an
 * off-grid k. M_mn(k,b) = sum_G conj(c_mk(G)) * c_{n,k'}(G + G0) is then a
 * plain dot product over k's own G-set, matched Miller-index-by-Miller-index
 * against k' 's G-set via the shared MillerIndexLookup, once per starting
 * G-vector -- both codes' G-sets live in the same global charge-density
 * sphere (PlanewaveBasis::g_vectors()), which is what makes this matching
 * well-defined and MillerIndexLookup reusable across every (k,b) pair.
 *
 * For norm-conserving pseudopotentials S = I, so this plane-wave dot
 * product is the whole answer: compute() alone. add_augmentation_contribution()
 * adds the ultrasoft/PAW term -- Q_ij(b) between the beta projectors at k
 * and k' -- on top of an already-computed compute() result.
 *
 * Normalization (resolved empirically, see this file's unit tests):
 * UltrasoftPlanewaveBasis::compute_q_g_single()'s general (nonzero-argument)
 * branch is normalized for its existing charge-density consumer
 * (add_augmentation_charges(), via the identically-normalized
 * precompute_q_functions_g() table -- neither includes a cell-volume
 * factor), not for the S-operator/overlap convention this augmentation
 * term needs (Q_ij = the cell-size-independent real-space integral
 * integral Q_ij(r) d^3r, exactly what the function's own b=0 special
 * branch returns). Measured on a real Si ultrasoft pseudopotential: the
 * general branch is smaller than the special branch by exactly the cell
 * volume, to 8 significant figures -- so add_augmentation_contribution()
 * multiplies compute_q_g_single()'s result by cell_volume(). This is
 * unconditionally correct for real Wannier neighbor directions (b is
 * never exactly zero there), but would be wrong applied to the b=0
 * special branch, so this method refuses b_cartesian = (0,0,0).
 */

#include <complex>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include "layer_0_core/math/phase_factor.hpp"
#include "layer_0_core/math/tolerances.hpp"
#include "layer_0_core/types/init.hpp"
#include "layer_0_core/types/matrix.hpp"
#include "layer_1_structure/crystal/crystal.hpp"
#include "layer_1_structure/reciprocal_space/kpoint_neighbor_shells.hpp"
#include "layer_3_representation/basis/planewave/init.hpp"
#include "layer_3_representation/basis/planewave/internal/miller_index_lookup.hpp"
#include "layer_3_representation/basis/planewave/pseudopotential/ultrasoft/init.hpp"

namespace q3 {

/**
 * Per-band-pair Bloch overlaps for one (k, k') pair, column-major (LAPACK
 * convention -- band_index_at_kpoint varies fastest, matching how the
 * localizer's dense linear algebra and the WannierInputs it feeds
 * both expect it):
 * values[band_index_at_kpoint + band_index_at_neighbor * number_of_bands_at_kpoint].
 * `missing_g_vector_count` is the number of k-side plane waves whose
 * shifted Miller triple fell outside the shared charge-density sphere, or
 * inside it but outside k' 's own selected G-set -- a real, expected
 * outcome at the cutoff-sphere edge (see the file header), never an
 * error; the caller decides whether to log or gate on it.
 */
struct [[nodiscard]] BlochOverlapResult {
    std::vector<Complex> values;
    int number_of_bands_at_kpoint = 0;
    int number_of_bands_at_neighbor = 0;
    int missing_g_vector_count = 0;

    [[nodiscard]] const Complex& operator()(int band_index_at_kpoint,
                                            int band_index_at_neighbor) const {
        return values[static_cast<std::size_t>(band_index_at_neighbor) * number_of_bands_at_kpoint +
                      band_index_at_kpoint];
    }
};

class PlanewaveBlochOverlapCalculator {
  public:
    /**
     * `global_miller_lookup` must already be built over
     * `planewave_basis.g_vectors()` -- the same lookup is reused across
     * every (k,b) pair in a calculation, so building it is the caller's
     * responsibility, not this calculator's.
     */
    explicit PlanewaveBlochOverlapCalculator(const PlanewaveBasis& planewave_basis,
                                             const MillerIndexLookup& global_miller_lookup)
        : planewave_basis_(planewave_basis), global_miller_lookup_(global_miller_lookup) {}

    /**
     * Compute M_mn(k,b) for one (k, neighbor) pair.
     *
     * `wavefunction_at_kpoint(band_index)` and
     * `wavefunction_at_neighbor(band_index)` return a pointer to that
     * band's plane-wave coefficients, contiguous over its own
     * `g_vector_indices_*` list (the PlanewaveKohnShamQuantumSystem
     * convention) -- passed as callables rather than a quantum system
     * reference so this kernel stays independent of the SCF/NSCF state
     * shape and is directly unit-testable against synthetic data.
     */
    template <typename WavefunctionAtKpoint, typename WavefunctionAtNeighbor>
    [[nodiscard]] BlochOverlapResult compute(const std::vector<int>& g_vector_indices_at_kpoint,
                                             int number_of_bands_at_kpoint,
                                             WavefunctionAtKpoint wavefunction_at_kpoint,
                                             const std::vector<int>& g_vector_indices_at_neighbor,
                                             int number_of_bands_at_neighbor,
                                             WavefunctionAtNeighbor wavefunction_at_neighbor,
                                             const IntegerVector3D& periodic_image_shift) const {
        const std::vector<GVector>& global_g_vectors = planewave_basis_.g_vectors();
        int number_of_plane_waves_at_kpoint = static_cast<int>(g_vector_indices_at_kpoint.size());
        int number_of_plane_waves_at_neighbor =
            static_cast<int>(g_vector_indices_at_neighbor.size());

        /**
         * Local position within the neighbor's own G-set, keyed by the
         * GLOBAL charge-density-sphere index -- built once per (k,b) pair
         * here; a caller driving many pairs against the same neighbor may
         * cache this externally, but correctness does not depend on it.
         */
        std::unordered_map<int, int> neighbor_local_position_by_global_index;
        neighbor_local_position_by_global_index.reserve(number_of_plane_waves_at_neighbor);
        for (int local_index = 0; local_index < number_of_plane_waves_at_neighbor; local_index++) {
            neighbor_local_position_by_global_index[g_vector_indices_at_neighbor[local_index]] =
                local_index;
        }

        /**
         * For every plane wave in k's own set, resolve which local
         * position (if any) in the neighbor's set holds the same physical
         * plane wave once the periodic-image shift is applied.
         */
        std::vector<int> neighbor_local_position_for_kpoint_plane_wave(
            number_of_plane_waves_at_kpoint, -1);
        int missing_count = 0;
        for (int g_index = 0; g_index < number_of_plane_waves_at_kpoint; g_index++) {
            const IntegerVector3D& miller =
                global_g_vectors[g_vector_indices_at_kpoint[g_index]].miller;
            IntegerVector3D shifted_miller(miller.x + periodic_image_shift.x,
                                           miller.y + periodic_image_shift.y,
                                           miller.z + periodic_image_shift.z);
            int global_index_of_shifted = global_miller_lookup_.find(shifted_miller);
            if (global_index_of_shifted < 0) {
                missing_count++;
                continue;
            }
            auto iterator = neighbor_local_position_by_global_index.find(global_index_of_shifted);
            if (iterator == neighbor_local_position_by_global_index.end()) {
                missing_count++;
                continue;
            }
            neighbor_local_position_for_kpoint_plane_wave[g_index] = iterator->second;
        }

        BlochOverlapResult result;
        result.number_of_bands_at_kpoint = number_of_bands_at_kpoint;
        result.number_of_bands_at_neighbor = number_of_bands_at_neighbor;
        result.missing_g_vector_count = missing_count;
        result.values.assign(static_cast<std::size_t>(number_of_bands_at_kpoint) *
                                 number_of_bands_at_neighbor,
                             Complex(0.0, 0.0));

        std::vector<const Complex*> band_data_at_kpoint(number_of_bands_at_kpoint);
        for (int band_index = 0; band_index < number_of_bands_at_kpoint; band_index++) {
            band_data_at_kpoint[band_index] = wavefunction_at_kpoint(band_index);
        }
        std::vector<const Complex*> band_data_at_neighbor(number_of_bands_at_neighbor);
        for (int band_index = 0; band_index < number_of_bands_at_neighbor; band_index++) {
            band_data_at_neighbor[band_index] = wavefunction_at_neighbor(band_index);
        }

        for (int band_index_at_neighbor = 0; band_index_at_neighbor < number_of_bands_at_neighbor;
             band_index_at_neighbor++) {
            for (int band_index_at_kpoint = 0; band_index_at_kpoint < number_of_bands_at_kpoint;
                 band_index_at_kpoint++) {
                Complex overlap(0.0, 0.0);
                for (int g_index = 0; g_index < number_of_plane_waves_at_kpoint; g_index++) {
                    int neighbor_local_position =
                        neighbor_local_position_for_kpoint_plane_wave[g_index];
                    if (neighbor_local_position < 0) {
                        continue;
                    }
                    overlap +=
                        std::conj(band_data_at_kpoint[band_index_at_kpoint][g_index]) *
                        band_data_at_neighbor[band_index_at_neighbor][neighbor_local_position];
                }
                result.values[static_cast<std::size_t>(band_index_at_neighbor) *
                                  number_of_bands_at_kpoint +
                              band_index_at_kpoint] = overlap;
            }
        }
        return result;
    }

    /**
     * Add the ultrasoft/PAW augmentation contribution to an already-computed
     * NC-only `result`, in place:
     *
     *   M_mn(k,b) += sum_I sum_ij <psi_mk|beta_i^I(k)> * Q_ij^I(b) *
     *                e^{-i*b.tau_I} * <beta_j^I(k+b)|psi_n,k+b>
     *
     * `projectors_at_kpoint`/`projectors_at_neighbor` are
     * compute_kleinman_bylander_projectors()'s output at k and at the
     * neighbor k' -- already carry each their own atomic structure factor
     * e^{-i(k+G).tau_I}, so only the *translation* phase for Q_ij (which is
     * defined atom-origin-centred, like compute_q_g_single itself) needs to
     * be applied here. `b_cartesian` is the *unwrapped* translation (i.e.
     * KPointNeighborDirection::reciprocal_lattice_translation, not folded
     * by the periodic-image shift used for the plane-wave G-matching in
     * compute() above), a physical wavevector in Bohr^-1 (tpiba-scaled --
     * built from Crystal::reciprocal_vectors_bohr(), NOT the same
     * convention as the raw GVector::g_cart field: g_cart is left
     * unscaled by tpiba precisely so that structure-factor code elsewhere
     * in this codebase can multiply it by an explicit 2*pi against a
     * dimensionless alat-fraction position; b_cartesian already carries
     * that 2*pi, so its own translation phase below multiplies the
     * position by the plain lattice parameter instead, not by 2*pi again
     * -- doing both would double the factor). See
     * atomic_position_alat's own definition below for the matching
     * position-side convention.
     *
     * `b_cartesian` must not be exactly (0,0,0) -- see the cell-volume
     * scaling note below; every real Wannier neighbor direction
     * (KPointNeighborShells::neighbor_directions()) already satisfies
     * this, since the origin is never itself a shell member.
     */
    template <typename WavefunctionAtKpoint, typename WavefunctionAtNeighbor>
    void add_augmentation_contribution(BlochOverlapResult& result,
                                       const Crystal& crystal_structure,
                                       const UltrasoftPlanewaveBasis& ultrasoft_basis,
                                       const Matrix<Complex>& projectors_at_kpoint,
                                       int number_of_plane_waves_at_kpoint,
                                       int number_of_bands_at_kpoint,
                                       WavefunctionAtKpoint wavefunction_at_kpoint,
                                       const Matrix<Complex>& projectors_at_neighbor,
                                       int number_of_plane_waves_at_neighbor,
                                       int number_of_bands_at_neighbor,
                                       WavefunctionAtNeighbor wavefunction_at_neighbor,
                                       const Vector3D& b_cartesian) const {
        if (b_cartesian.norm() < math::tolerances::zero_vector_magnitude_threshold) {
            throw std::invalid_argument(
                "PlanewaveBlochOverlapCalculator::add_augmentation_contribution: |b_cartesian| "
                "is below compute_q_g_single's zero-vector threshold, where it takes its b=0 "
                "special branch. That branch is already correctly scaled and must not be "
                "multiplied by cell_volume(), which this method does unconditionally for its "
                "(always nonzero, in real use) general-branch case.");
        }
        int total_number_of_projectors = projectors_at_kpoint.rows();

        // amplitude_at_kpoint[projector][band] = <beta_i^I(k)|psi_mk>
        Matrix<Complex> amplitude_at_kpoint(static_cast<std::size_t>(total_number_of_projectors),
                                            number_of_bands_at_kpoint);
        for (int band_index = 0; band_index < number_of_bands_at_kpoint; band_index++) {
            const Complex* band_data = wavefunction_at_kpoint(band_index);
            for (int projector_index = 0; projector_index < total_number_of_projectors;
                 projector_index++) {
                Complex sum(0.0, 0.0);
                const Complex* projector_row = projectors_at_kpoint[projector_index];
                for (int g_index = 0; g_index < number_of_plane_waves_at_kpoint; g_index++) {
                    sum += std::conj(projector_row[g_index]) * band_data[g_index];
                }
                amplitude_at_kpoint[projector_index][band_index] = sum;
            }
        }

        // amplitude_at_neighbor[projector][band] = <beta_j^I(k')|psi_n,k'>
        Matrix<Complex> amplitude_at_neighbor(static_cast<std::size_t>(total_number_of_projectors),
                                              number_of_bands_at_neighbor);
        for (int band_index = 0; band_index < number_of_bands_at_neighbor; band_index++) {
            const Complex* band_data = wavefunction_at_neighbor(band_index);
            for (int projector_index = 0; projector_index < total_number_of_projectors;
                 projector_index++) {
                Complex sum(0.0, 0.0);
                const Complex* projector_row = projectors_at_neighbor[projector_index];
                for (int g_index = 0; g_index < number_of_plane_waves_at_neighbor; g_index++) {
                    sum += std::conj(projector_row[g_index]) * band_data[g_index];
                }
                amplitude_at_neighbor[projector_index][band_index] = sum;
            }
        }

        Real two_pi_over_lattice_parameter = crystal_structure.two_pi_over_lattice_parameter();

        /**
         * compute_q_g_single()'s general (nonzero-argument) branch is
         * normalized for its existing charge-density consumer
         * (add_augmentation_charges() / precompute_q_functions_g(), which
         * use the identical interpolate_augmentation_function()-based formula with no
         * volume factor of their own) -- not for the S-operator/overlap
         * convention this augmentation term needs, where Q_ij must be the
         * cell-size-independent real-space integral
         * augmentation_overlap_integrals(i,j) = integral Q_ij(r) d^3r
         * (exactly what its own b=0 special branch returns, unscaled).
         * Measured empirically (see this file's unit tests): the general
         * branch is smaller than the special branch by exactly a factor
         * of the cell volume, to 8 significant figures on a real Si
         * ultrasoft pseudopotential -- so the general branch must be
         * multiplied by cell_volume() to convert it to the overlap
         * convention. b is never exactly zero for a real Wannier neighbor
         * direction (KPointNeighborShells never enumerates the origin),
         * so this multiplication is applied unconditionally; it would be
         * wrong at b = 0 exactly, where compute_q_g_single() takes its
         * already-correctly-scaled special branch instead -- callers must
         * not invoke this method with b_cartesian = (0,0,0).
         */
        Real cell_volume = crystal_structure.cell_volume();

        /**
         * compute_q_g_single() takes its wavevector in the plane-wave
         * basis's own 2*pi/alat units (it multiplies by
         * two_pi_over_lattice_parameter internally before interpolating
         * the radial table), while b_cartesian arrives in physical
         * bohr^-1 from KPointNeighborShells. Convert once here; passing
         * b_cartesian directly would sample Q_ij at |b| * tpiba instead
         * of |b| -- too small by tpiba (0.6 for silicon), so an
         * over-estimated augmentation term that can push the overlap's
         * singular values above 1.
         */
        const Vector3D b_in_lattice_units = b_cartesian * (1.0 / two_pi_over_lattice_parameter);

        int projector_offset = 0;
        for (int atom_index = 0; atom_index < crystal_structure.number_of_atoms(); atom_index++) {
            int species_type_index = crystal_structure.species_type_index(atom_index);
            int number_of_channels =
                ultrasoft_basis.number_of_projector_channels(species_type_index);
            if (number_of_channels == 0) {
                continue;
            }

            /**
             * convert_from_crystal_to_cartesian() returns a position in
             * alat units (dimensionless, 1.0 = one lattice parameter), not
             * Bohr -- multiply by lattice_parameter() to get a physical
             * Bohr position matching b_cartesian's own physical Bohr^-1
             * units, so the dot product is already a phase angle in
             * radians with no separate 2*pi factor (unlike the g_cart-based
             * structure factors elsewhere in this codebase, which multiply
             * an explicit 2*pi against a *dimensionless* reduced G-vector
             * instead -- a different, not-tpiba-scaled convention).
             */
            Vector3D atomic_position_alat =
                crystal_structure.convert_from_crystal_to_cartesian(atom_index);
            Real translation_phase_angle =
                b_cartesian.dot(atomic_position_alat) * crystal_structure.lattice_parameter();
            Complex translation_phase = compute_phase_factor(translation_phase_angle);

            for (int channel_i = 0; channel_i < number_of_channels; channel_i++) {
                int projector_i = projector_offset + channel_i;
                for (int channel_j = 0; channel_j < number_of_channels; channel_j++) {
                    int projector_j = projector_offset + channel_j;

                    Complex q_ij =
                        cell_volume *
                        ultrasoft_basis.compute_q_g_single(species_type_index,
                                                           channel_i,
                                                           channel_j,
                                                           b_in_lattice_units,
                                                           two_pi_over_lattice_parameter) *
                        translation_phase;
                    if (q_ij == Complex(0.0, 0.0)) {
                        continue;
                    }

                    for (int band_index_at_neighbor = 0;
                         band_index_at_neighbor < number_of_bands_at_neighbor;
                         band_index_at_neighbor++) {
                        Complex right =
                            q_ij * amplitude_at_neighbor[projector_j][band_index_at_neighbor];
                        for (int band_index_at_kpoint = 0;
                             band_index_at_kpoint < number_of_bands_at_kpoint;
                             band_index_at_kpoint++) {
                            /**
                             * amplitude_at_kpoint stores <beta_i|psi_mk>; the formula above
                             * needs <psi_mk|beta_i>, its conjugate.
                             */
                            result.values[static_cast<std::size_t>(band_index_at_neighbor) *
                                              number_of_bands_at_kpoint +
                                          band_index_at_kpoint] +=
                                std::conj(amplitude_at_kpoint[projector_i][band_index_at_kpoint]) *
                                right;
                        }
                    }
                }
            }
            projector_offset += number_of_channels;
        }
    }

  private:
    const PlanewaveBasis& planewave_basis_;
    const MillerIndexLookup& global_miller_lookup_;
};

} // namespace q3
