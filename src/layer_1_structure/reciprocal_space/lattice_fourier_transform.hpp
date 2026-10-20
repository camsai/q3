#pragma once

/**
 * lattice_fourier_transform — the discrete Fourier pair between a k-point set
 * and a set of lattice vectors.
 *
 *   f(R) = (1/N_k) sum_k  e^{-2 pi i k.R} f(k)          to_real_space()
 *   f(k) = sum_R    w_R   e^{+2 pi i k.R} f(R)          to_kpoint()
 *
 * k-points and translations are both in crystal coordinates, so the phase is
 * the plain dot product with no lattice matrix in sight -- which is why this
 * belongs beside the k-point grid and the Wigner-Seitz supercell rather than
 * inside any one consumer.
 *
 * Values are flat blocks of arbitrary, equal size, transformed element by
 * element: a matrix (Wannier interpolation transforms U^dagger E U) and a
 * single band energy (a Fourier star-function fit; a block of one) go through
 * the same code.
 *
 * This is NOT an FFT and deliberately does not use the FFT machinery in
 * layer 3. The forward transform lands on a Wigner-Seitz set with fractional
 * boundary weights rather than on a rectangular box, and the inverse is
 * evaluated at arbitrary k off any grid, which an FFT cannot do at all. At the
 * sizes involved -- hundreds of k-points, blocks of a few orbitals -- a direct
 * sum costs microseconds and an FFT plan would cost more than the work.
 *
 * The weights belong to the inverse alone. The forward transform divides by
 * the k-point count; the degeneracy weights of a Wigner-Seitz boundary vector
 * enter when the sum over R is taken, and their total must come back to the
 * k-point count -- the identity every consumer of a transformed model checks.
 */

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include "layer_0_core/types/constants.hpp"
#include "layer_0_core/types/init.hpp"

namespace q3::lattice_fourier_transform {

namespace internal {

/// e^{sign * 2 pi i k.R} for crystal-coordinate k and integer R.
[[nodiscard]] inline Complex
phase(const Vector3D& kpoint_crystal, const IntegerVector3D& translation, Real sign) {
    const Real angle = sign * constants::two_pi *
                       (kpoint_crystal.x * static_cast<Real>(translation.x) +
                        kpoint_crystal.y * static_cast<Real>(translation.y) +
                        kpoint_crystal.z * static_cast<Real>(translation.z));
    return {std::cos(angle), std::sin(angle)};
}

} // namespace internal

/**
 * f(R) = (1/N_k) sum_k e^{-2 pi i k.R} f(k), one block per lattice vector.
 *
 * `values_at_kpoints` holds one equally sized block per k-point, in the order
 * of `kpoints_crystal`. The k-set must be the full, unreduced grid the model
 * was built on: a symmetry-reduced set would weight the sum wrongly, and this
 * function has no way to detect that, so its callers own that guarantee.
 */
[[nodiscard]] inline std::vector<std::vector<Complex>>
to_real_space(const std::vector<Vector3D>& kpoints_crystal,
              const std::vector<std::vector<Complex>>& values_at_kpoints,
              const std::vector<IntegerVector3D>& lattice_translations) {
    const std::size_t number_of_kpoints = kpoints_crystal.size();
    if (number_of_kpoints == 0) {
        throw std::invalid_argument("lattice_fourier_transform::to_real_space: no k-points");
    }
    if (values_at_kpoints.size() != number_of_kpoints) {
        throw std::invalid_argument(
            "lattice_fourier_transform::to_real_space: " + std::to_string(number_of_kpoints) +
            " k-points but " + std::to_string(values_at_kpoints.size()) + " value blocks");
    }
    if (lattice_translations.empty()) {
        throw std::invalid_argument("lattice_fourier_transform::to_real_space: no lattice vectors");
    }
    const std::size_t block_size = values_at_kpoints[0].size();
    if (block_size == 0) {
        throw std::invalid_argument("lattice_fourier_transform::to_real_space: empty value block");
    }
    for (std::size_t kpoint_index = 1; kpoint_index < number_of_kpoints; kpoint_index++) {
        if (values_at_kpoints[kpoint_index].size() != block_size) {
            throw std::invalid_argument(
                "lattice_fourier_transform::to_real_space: the block at k-point " +
                std::to_string(kpoint_index) + " holds " +
                std::to_string(values_at_kpoints[kpoint_index].size()) + " values, not " +
                std::to_string(block_size) + " as the first one does");
        }
    }

    const Real inverse_kpoint_count = 1.0 / static_cast<Real>(number_of_kpoints);
    std::vector<std::vector<Complex>> values_at_lattice_vectors(
        lattice_translations.size(), std::vector<Complex>(block_size, Complex(0.0, 0.0)));

    /**
     * k outermost so each k-point's phase is computed once per lattice vector
     * rather than once per element, and the block is streamed contiguously.
     */
    for (std::size_t kpoint_index = 0; kpoint_index < number_of_kpoints; kpoint_index++) {
        const std::vector<Complex>& block_at_kpoint = values_at_kpoints[kpoint_index];
        for (std::size_t vector_index = 0; vector_index < lattice_translations.size();
             vector_index++) {
            const Complex weighted_phase = internal::phase(kpoints_crystal[kpoint_index],
                                                           lattice_translations[vector_index],
                                                           -1.0) *
                                           inverse_kpoint_count;
            std::vector<Complex>& block_at_lattice_vector = values_at_lattice_vectors[vector_index];
            for (std::size_t element = 0; element < block_size; element++) {
                block_at_lattice_vector[element] += weighted_phase * block_at_kpoint[element];
            }
        }
    }
    return values_at_lattice_vectors;
}

/**
 * f(k) = sum_R w_R e^{+2 pi i k.R} f(R), the inverse, at one arbitrary k.
 *
 * `kpoint_crystal` need not lie on any mesh -- evaluating between the mesh
 * points is what an interpolated model is for. `weights` are the Bloch-sum
 * weights, 1/degeneracy for a Wigner-Seitz boundary vector and 1 otherwise;
 * they are passed alongside the translations rather than as a struct so that a
 * caller storing them in its own type needs no per-call conversion on what is,
 * for a dense mesh sampling, the hot path.
 */
[[nodiscard]] inline std::vector<Complex>
to_kpoint(const Vector3D& kpoint_crystal,
          const std::vector<std::vector<Complex>>& values_at_lattice_vectors,
          const std::vector<IntegerVector3D>& lattice_translations,
          const std::vector<Real>& weights) {
    const std::size_t number_of_vectors = lattice_translations.size();
    if (number_of_vectors == 0) {
        throw std::invalid_argument("lattice_fourier_transform::to_kpoint: no lattice vectors");
    }
    if (values_at_lattice_vectors.size() != number_of_vectors ||
        weights.size() != number_of_vectors) {
        throw std::invalid_argument(
            "lattice_fourier_transform::to_kpoint: " + std::to_string(number_of_vectors) +
            " lattice vectors but " + std::to_string(values_at_lattice_vectors.size()) +
            " value blocks and " + std::to_string(weights.size()) + " weights");
    }
    const std::size_t block_size = values_at_lattice_vectors[0].size();
    if (block_size == 0) {
        throw std::invalid_argument("lattice_fourier_transform::to_kpoint: empty value block");
    }

    std::vector<Complex> value_at_kpoint(block_size, Complex(0.0, 0.0));
    for (std::size_t vector_index = 0; vector_index < number_of_vectors; vector_index++) {
        const std::vector<Complex>& block = values_at_lattice_vectors[vector_index];
        if (block.size() != block_size) {
            throw std::invalid_argument(
                "lattice_fourier_transform::to_kpoint: the block at lattice vector " +
                std::to_string(vector_index) + " holds " + std::to_string(block.size()) +
                " values, not " + std::to_string(block_size) + " as the first one does");
        }
        const Complex weighted_phase =
            internal::phase(kpoint_crystal, lattice_translations[vector_index], 1.0) *
            weights[vector_index];
        for (std::size_t element = 0; element < block_size; element++) {
            value_at_kpoint[element] += weighted_phase * block[element];
        }
    }
    return value_at_kpoint;
}

} // namespace q3::lattice_fourier_transform
