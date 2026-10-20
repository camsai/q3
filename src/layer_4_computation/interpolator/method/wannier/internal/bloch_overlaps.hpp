#pragma once

/**
 * BlochOverlaps — M(k, b) for every k-point and neighbour direction, as the
 * dense matrices the construction multiplies, rather than the nested vectors
 * the record carries.
 *
 * A record stores each block flat and column-major so it can be written and
 * read as bytes; the engine reads it as a matrix, restricted to whichever
 * bands survive at k and at k+b, and rotated by whichever gauge is current.
 * The restriction is done once for a construction, and the rotation is two
 * products.
 */

#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "layer_0_core/math/linear_algebra/dense_matrix.hpp"
#include "layer_0_core/math/linear_algebra/dense_matrix_product.hpp"
#include "layer_0_core/math/linear_algebra/matrix_operation.hpp"
#include "layer_0_core/types/type_aliases.hpp"
#include "layer_1_structure/reciprocal_space/kpoint_stencil.hpp"
#include "layer_4_computation/interpolator/method/wannier/internal/band_classification.hpp"

namespace q3 {

class BlochOverlaps {
  public:
    /**
     * From a record's blocks, `blocks[kpoint][direction]` each a column-major
     * `rows` x `columns` matrix with the k-side band as the row. A block of
     * another size is refused naming its k-point and direction.
     */
    BlochOverlaps(const std::vector<std::vector<std::vector<Complex>>>& blocks,
                  int rows,
                  int columns) {
        if (blocks.empty()) {
            throw std::invalid_argument("BlochOverlaps: no k-points");
        }
        const std::size_t directions = blocks.front().size();
        blocks_.reserve(blocks.size());
        for (std::size_t kpoint = 0; kpoint < blocks.size(); kpoint++) {
            if (blocks[kpoint].size() != directions) {
                throw std::invalid_argument("BlochOverlaps: k-point " + std::to_string(kpoint) +
                                            " carries " + std::to_string(blocks[kpoint].size()) +
                                            " overlap blocks where k-point 0 carries " +
                                            std::to_string(directions));
            }
            std::vector<DenseMatrix<Complex>> row;
            row.reserve(directions);
            for (std::size_t direction = 0; direction < directions; direction++) {
                const std::vector<Complex>& block = blocks[kpoint][direction];
                if (block.size() !=
                    static_cast<std::size_t>(rows) * static_cast<std::size_t>(columns)) {
                    throw std::invalid_argument(
                        "BlochOverlaps: the overlap block at k-point " + std::to_string(kpoint) +
                        ", direction " + std::to_string(direction) + " holds " +
                        std::to_string(block.size()) + " entries, not the " +
                        std::to_string(static_cast<std::size_t>(rows) * columns) + " a " +
                        std::to_string(rows) + " x " + std::to_string(columns) + " block requires");
                }
                row.emplace_back(rows, columns, block);
            }
            blocks_.push_back(std::move(row));
        }
    }

    /// The square case: N x N at every (k, b).
    [[nodiscard]] static BlochOverlaps
    square(const std::vector<std::vector<std::vector<Complex>>>& blocks, int dimension) {
        return BlochOverlaps(blocks, dimension, dimension);
    }

    /// From matrices already built, one row of directions per k-point.
    explicit BlochOverlaps(std::vector<std::vector<DenseMatrix<Complex>>> blocks)
        : blocks_(std::move(blocks)) {
        if (blocks_.empty()) {
            throw std::invalid_argument("BlochOverlaps: no k-points");
        }
    }

    [[nodiscard]] int number_of_kpoints() const {
        return static_cast<int>(blocks_.size());
    }
    [[nodiscard]] int number_of_directions() const {
        return static_cast<int>(blocks_.front().size());
    }

    [[nodiscard]] const DenseMatrix<Complex>& at(int kpoint, int direction) const {
        return blocks_[static_cast<std::size_t>(kpoint)][static_cast<std::size_t>(direction)];
    }

    /**
     * M restricted to the kept bands: rows from k's keep-list, columns from
     * k+b's. The two lists differ, because which bands survive is decided
     * k-point by k-point; slicing both sides with one of them is the mistake
     * this signature exists to make impossible.
     */
    [[nodiscard]] BlochOverlaps
    restricted_to(const std::vector<KPointBandClassification>& classification,
                  const KPointStencil& stencil) const {
        require_stencil(stencil);
        if (static_cast<int>(classification.size()) != number_of_kpoints()) {
            throw std::invalid_argument("BlochOverlaps: the classification covers " +
                                        std::to_string(classification.size()) +
                                        " k-points, the overlaps " +
                                        std::to_string(number_of_kpoints()));
        }
        std::vector<std::vector<DenseMatrix<Complex>>> restricted(blocks_.size());
        for (int kpoint = 0; kpoint < number_of_kpoints(); kpoint++) {
            const std::vector<int>& rows =
                classification[static_cast<std::size_t>(kpoint)].kept_bands;
            require_bands_within(rows, kpoint);
            for (int direction = 0; direction < number_of_directions(); direction++) {
                const int neighbor = stencil.neighbor(kpoint, direction);
                const std::vector<int>& columns =
                    classification[static_cast<std::size_t>(neighbor)].kept_bands;
                require_bands_within(columns, neighbor);
                restricted[static_cast<std::size_t>(kpoint)].push_back(
                    at(kpoint, direction).submatrix(rows, columns));
            }
        }
        return BlochOverlaps(std::move(restricted));
    }

    /// left(k)^dagger M(k, b) right(k + b) at every (k, b).
    [[nodiscard]] BlochOverlaps transformed_by(const std::vector<DenseMatrix<Complex>>& left,
                                               const std::vector<DenseMatrix<Complex>>& right,
                                               const KPointStencil& stencil) const {
        require_stencil(stencil);
        if (static_cast<int>(left.size()) != number_of_kpoints() ||
            static_cast<int>(right.size()) != number_of_kpoints()) {
            throw std::invalid_argument("BlochOverlaps: one left and one right matrix per k-point "
                                        "are required, " +
                                        std::to_string(number_of_kpoints()) + " of each, not " +
                                        std::to_string(left.size()) + " and " +
                                        std::to_string(right.size()));
        }
        std::vector<std::vector<DenseMatrix<Complex>>> transformed(blocks_.size());
        for (int kpoint = 0; kpoint < number_of_kpoints(); kpoint++) {
            for (int direction = 0; direction < number_of_directions(); direction++) {
                const int neighbor = stencil.neighbor(kpoint, direction);
                transformed[static_cast<std::size_t>(kpoint)].push_back(
                    multiply(MatrixOperation::ConjugateTranspose,
                             left[static_cast<std::size_t>(kpoint)],
                             MatrixOperation::None,
                             multiply(MatrixOperation::None,
                                      at(kpoint, direction),
                                      MatrixOperation::None,
                                      right[static_cast<std::size_t>(neighbor)])));
            }
        }
        return BlochOverlaps(std::move(transformed));
    }

  private:
    void require_stencil(const KPointStencil& stencil) const {
        if (stencil.number_of_kpoints() != number_of_kpoints() ||
            stencil.number_of_directions() != number_of_directions()) {
            throw std::invalid_argument(
                "BlochOverlaps: the stencil covers " + std::to_string(stencil.number_of_kpoints()) +
                " k-points and " + std::to_string(stencil.number_of_directions()) +
                " directions, the overlaps " + std::to_string(number_of_kpoints()) + " and " +
                std::to_string(number_of_directions()));
        }
    }

    void require_bands_within(const std::vector<int>& bands, int kpoint) const {
        const int captured = blocks_.front().front().rows();
        for (const int band : bands) {
            if (band < 0 || band >= captured) {
                throw std::invalid_argument("BlochOverlaps: k-point " + std::to_string(kpoint) +
                                            " keeps band " + std::to_string(band) +
                                            ", outside the " + std::to_string(captured) +
                                            " captured bands the overlaps are indexed by");
            }
        }
    }

    /// [kpoint][direction]
    std::vector<std::vector<DenseMatrix<Complex>>> blocks_;
};

} // namespace q3
