#pragma once

/**
 * A one-dimensional chain whose Wannier overlaps are known in closed form, and
 * the machinery to scramble its gauge.
 *
 * The point of the chain is that its minimum is known exactly rather than
 * measured. Put each Wannier function at a fixed centre and give every
 * neighbour overlap the diagonal form
 *
 *     M_nn(k, b) = rho exp(-i b r_n),   M_mn = 0 for m != n,
 *
 * and both minimisable parts of the spread vanish identically: there is no
 * off-diagonal weight, and each diagonal phase is exactly the one its centre
 * implies. Omega_I is rho-dependent and gauge-invariant. So a minimiser handed
 * a scrambled version of this model has a target to reach -- zero -- and a
 * quantity that must not move on the way there.
 *
 * The scrambling is a k-dependent unitary applied on both sides, which is the
 * same transformation a different choice of trial orbitals would produce. It
 * cannot change Omega_I, and it raises Omega_D + Omega_OD off zero.
 *
 * The model keeps its blocks as the nested vectors a record stores, because a
 * test that wants a malformed input mutates them in place; `stencil_of` and
 * `overlaps_of` are the two conversions the engine's objects take, and both
 * refuse what the mutation broke.
 */

#include <cmath>
#include <complex>
#include <cstddef>
#include <random>
#include <vector>

#include "layer_0_core/math/linear_algebra/dense_matrix.hpp"
#include "layer_0_core/math/linear_algebra/unitary_exponential.hpp"
#include "layer_0_core/types/init.hpp"
#include "layer_1_structure/reciprocal_space/kpoint_neighbor_shells.hpp"
#include "layer_1_structure/reciprocal_space/kpoint_stencil.hpp"
#include "layer_4_computation/interpolator/method/wannier/inputs.hpp"
#include "layer_4_computation/interpolator/method/wannier/internal/bloch_overlaps.hpp"

namespace q3::test_utils {

struct [[nodiscard]] WannierChainModel {
    /// [kpoint][direction], each a dimension x dimension column-major block.
    std::vector<std::vector<std::vector<Complex>>> overlaps;
    std::vector<std::vector<int>> neighbor_kpoint_index;
    std::vector<Vector3D> b_vectors;
    std::vector<Real> weights;
    std::vector<Real> centers;
    int number_of_wannier_functions = 0;
};

/**
 * `centers` are in bohr along x, one per Wannier function; `diagonal_modulus`
 * is rho, which sets Omega_I without touching the minimisable parts.
 */
[[nodiscard]] inline WannierChainModel make_wannier_chain(int number_of_kpoints,
                                                          const std::vector<Real>& centers,
                                                          Real lattice_constant_bohr = 5.0,
                                                          Real diagonal_modulus = 0.9) {
    WannierChainModel model;
    model.number_of_wannier_functions = static_cast<int>(centers.size());
    model.centers = centers;

    const int dimension = model.number_of_wannier_functions;
    const Real spacing =
        2.0 * M_PI / (lattice_constant_bohr * static_cast<Real>(number_of_kpoints));

    /**
     * Forward and backward along x. The completeness condition in one
     * dimension is sum_b w_b b^2 = 1 over both of them, so each weight is
     * 1/(2 b^2); the transverse directions carry no b at all, which is what a
     * chain means.
     */
    model.b_vectors = {Vector3D{spacing, 0.0, 0.0}, Vector3D{-spacing, 0.0, 0.0}};
    model.weights = {1.0 / (2.0 * spacing * spacing), 1.0 / (2.0 * spacing * spacing)};

    for (int kpoint = 0; kpoint < number_of_kpoints; kpoint++) {
        model.neighbor_kpoint_index.push_back(
            {(kpoint + 1) % number_of_kpoints,
             (kpoint - 1 + number_of_kpoints) % number_of_kpoints});
        std::vector<std::vector<Complex>> blocks;
        for (const Vector3D& b_vector : model.b_vectors) {
            std::vector<Complex> block(static_cast<std::size_t>(dimension) * dimension,
                                       Complex(0.0, 0.0));
            for (int wannier = 0; wannier < dimension; wannier++) {
                block[static_cast<std::size_t>(wannier) * dimension + wannier] =
                    diagonal_modulus *
                    std::exp(
                        Complex(0.0, -b_vector[0] * centers[static_cast<std::size_t>(wannier)]));
            }
            blocks.push_back(std::move(block));
        }
        model.overlaps.push_back(std::move(blocks));
    }
    return model;
}

/// The model's stencil, as the engine's objects take it; a broken neighbour table is refused here.
[[nodiscard]] inline KPointStencil stencil_of(const WannierChainModel& model) {
    return KPointStencil::from_table(model.neighbor_kpoint_index, model.weights, model.b_vectors);
}

/// The model's overlaps as matrices; a misshapen block is refused here.
[[nodiscard]] inline BlochOverlaps overlaps_of(const WannierChainModel& model) {
    return BlochOverlaps::square(model.overlaps, model.number_of_wannier_functions);
}

/// A deterministic Haar-ish unitary: Gram-Schmidt on a seeded Gaussian matrix.
[[nodiscard]] inline std::vector<Complex> random_unitary(int dimension, std::mt19937& generator) {
    std::normal_distribution<Real> gaussian(0.0, 1.0);
    std::vector<Complex> matrix(static_cast<std::size_t>(dimension) * dimension);
    for (Complex& element : matrix) {
        element = Complex(gaussian(generator), gaussian(generator));
    }

    for (int column = 0; column < dimension; column++) {
        for (int previous = 0; previous < column; previous++) {
            Complex projection(0.0, 0.0);
            for (int row = 0; row < dimension; row++) {
                projection +=
                    std::conj(matrix[static_cast<std::size_t>(previous) * dimension + row]) *
                    matrix[static_cast<std::size_t>(column) * dimension + row];
            }
            for (int row = 0; row < dimension; row++) {
                matrix[static_cast<std::size_t>(column) * dimension + row] -=
                    projection * matrix[static_cast<std::size_t>(previous) * dimension + row];
            }
        }
        Real norm = 0.0;
        for (int row = 0; row < dimension; row++) {
            norm += std::norm(matrix[static_cast<std::size_t>(column) * dimension + row]);
        }
        norm = std::sqrt(norm);
        for (int row = 0; row < dimension; row++) {
            matrix[static_cast<std::size_t>(column) * dimension + row] /= norm;
        }
    }
    return matrix;
}

/**
 * left^dagger M right, the same rotation a different trial-orbital set
 * applies. Written out by hand on purpose: it is the oracle the engine's
 * products are checked against, so it must not be one of them.
 */
[[nodiscard]] inline std::vector<Complex> rotate_block(const std::vector<Complex>& overlap,
                                                       const std::vector<Complex>& left,
                                                       const std::vector<Complex>& right,
                                                       int dimension) {
    std::vector<Complex> intermediate(overlap.size(), Complex(0.0, 0.0));
    for (int column = 0; column < dimension; column++) {
        for (int row = 0; row < dimension; row++) {
            Complex sum(0.0, 0.0);
            for (int inner = 0; inner < dimension; inner++) {
                sum += std::conj(left[static_cast<std::size_t>(row) * dimension + inner]) *
                       overlap[static_cast<std::size_t>(column) * dimension + inner];
            }
            intermediate[static_cast<std::size_t>(column) * dimension + row] = sum;
        }
    }
    std::vector<Complex> result(overlap.size(), Complex(0.0, 0.0));
    for (int column = 0; column < dimension; column++) {
        for (int row = 0; row < dimension; row++) {
            Complex sum(0.0, 0.0);
            for (int inner = 0; inner < dimension; inner++) {
                sum += intermediate[static_cast<std::size_t>(inner) * dimension + row] *
                       right[static_cast<std::size_t>(column) * dimension + inner];
            }
            result[static_cast<std::size_t>(column) * dimension + row] = sum;
        }
    }
    return result;
}

/// Apply one rotation per k-point on both sides of every overlap block.
inline void apply_rotations(WannierChainModel& model,
                            const std::vector<std::vector<Complex>>& rotations) {
    const int dimension = model.number_of_wannier_functions;
    std::vector<std::vector<std::vector<Complex>>> rotated(model.overlaps.size());
    for (std::size_t kpoint = 0; kpoint < model.overlaps.size(); kpoint++) {
        for (std::size_t direction = 0; direction < model.overlaps[kpoint].size(); direction++) {
            const std::size_t neighbor =
                static_cast<std::size_t>(model.neighbor_kpoint_index[kpoint][direction]);
            rotated[kpoint].push_back(rotate_block(model.overlaps[kpoint][direction],
                                                   rotations[kpoint],
                                                   rotations[neighbor],
                                                   dimension));
        }
    }
    model.overlaps = std::move(rotated);
}

/// A seeded anti-Hermitian matrix, the generator of a rotation of given size.
[[nodiscard]] inline std::vector<Complex>
random_anti_hermitian(int dimension, Real magnitude, std::mt19937& generator) {
    std::normal_distribution<Real> gaussian(0.0, magnitude);
    std::vector<Complex> matrix(static_cast<std::size_t>(dimension) * dimension);
    for (Complex& element : matrix) {
        element = Complex(gaussian(generator), gaussian(generator));
    }
    std::vector<Complex> anti(matrix.size());
    for (int column = 0; column < dimension; column++) {
        for (int row = 0; row < dimension; row++) {
            const std::size_t index = static_cast<std::size_t>(column) * dimension + row;
            const std::size_t transposed = static_cast<std::size_t>(row) * dimension + column;
            anti[index] = 0.5 * (matrix[index] - std::conj(matrix[transposed]));
        }
    }
    return anti;
}

/// Apply an independent random unitary at every k-point, in place.
inline void scramble_gauge(WannierChainModel& model, unsigned int seed) {
    std::mt19937 generator(seed);
    const int dimension = model.number_of_wannier_functions;
    std::vector<std::vector<Complex>> rotations;
    rotations.reserve(model.overlaps.size());
    for (std::size_t kpoint = 0; kpoint < model.overlaps.size(); kpoint++) {
        rotations.push_back(random_unitary(dimension, generator));
    }

    apply_rotations(model, rotations);
}

/**
 * A rotation of controlled size, exp(A) with A anti-Hermitian of the given
 * magnitude. This is the realistic disturbance: a projection gauge is a
 * sensible guess that sits near the localised one, not a random point on the
 * unitary group, and the two cases behave differently enough that a test which
 * only exercises the random one is testing the wrong problem.
 */
inline void perturb_gauge(WannierChainModel& model, unsigned int seed, Real magnitude) {
    std::mt19937 generator(seed);
    const int dimension = model.number_of_wannier_functions;
    std::vector<std::vector<Complex>> rotations;
    rotations.reserve(model.overlaps.size());
    for (std::size_t kpoint = 0; kpoint < model.overlaps.size(); kpoint++) {
        rotations.push_back(
            exponential_of_anti_hermitian(
                DenseMatrix<Complex>(
                    dimension, dimension, random_anti_hermitian(dimension, magnitude, generator)))
                .raw());
    }
    apply_rotations(model, rotations);
}

/**
 * The chain as the record a construction reads: one row of k-points along
 * x, its two directions as grid steps, and identity projections, so the
 * projection gauge is the identity and the overlaps reach the spread as they
 * are. The eigenvalues are placeholders that no test of a gauge reads.
 */
[[nodiscard]] inline WannierInputs inputs_of(const WannierChainModel& model) {
    const int number_of_kpoints = static_cast<int>(model.overlaps.size());
    const int dimension = model.number_of_wannier_functions;
    WannierInputs inputs;
    inputs.kpoint_grid_dimensions = {number_of_kpoints, 1, 1};
    inputs.number_of_bands = dimension;
    inputs.number_of_trial_orbitals = dimension;
    for (int kpoint = 0; kpoint < number_of_kpoints; kpoint++) {
        inputs.kpoint_coordinates.emplace_back(
            static_cast<Real>(kpoint) / static_cast<Real>(number_of_kpoints), 0.0, 0.0);
        std::vector<Real> eigenvalues;
        for (int band = 0; band < dimension; band++) {
            eigenvalues.push_back(static_cast<Real>(band));
        }
        inputs.eigenvalues_rydberg.push_back(eigenvalues);
        inputs.projection_matrices.push_back(DenseMatrix<Complex>::identity(dimension).raw());
    }
    inputs.neighbor_directions = {
        KPointNeighborDirection{IntegerVector3D(1, 0, 0), model.b_vectors[0], model.weights[0]},
        KPointNeighborDirection{IntegerVector3D(-1, 0, 0), model.b_vectors[1], model.weights[1]}};
    inputs.overlap_matrices = model.overlaps;
    return inputs;
}

} // namespace q3::test_utils
